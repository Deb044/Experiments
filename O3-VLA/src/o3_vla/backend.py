"""Native OpenVLA action loss/generation and an explicitly synthetic CPU fixture."""

import hashlib
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .actions import ActionCodec
from .io import write_json
from .lora import Adapters


def center_crop(image):
    # PIL fallback is used ONLY by synthetic tests. Native path uses official TF crop/resize.
    width, height = image.size
    scale = np.sqrt(0.9)
    left, top = width * (1 - scale) / 2, height * (1 - scale) / 2
    return image.crop((left, top, width - left, height - top)).resize((224, 224))


class OpenVLA:
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA research backend requires CUDA; use smoke.json only for CPU plumbing")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this pilot requires BF16 compute support")
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        self.cfg, self.device = cfg, torch.device("cuda:0")
        kwargs = {
            "revision": cfg.model_revision,
            "code_revision": cfg.code_revision,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": cfg.attention,
            "low_cpu_mem_usage": True,
            "device_map": {"": 0},
        }
        if cfg.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForVision2Seq.from_pretrained(cfg.model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, code_revision=cfg.code_revision, trust_remote_code=True
        )
        self.model.eval()
        key = cfg.suite if cfg.suite in self.model.norm_stats else cfg.suite + "_no_noops"
        if key not in self.model.norm_stats:
            raise ValueError("checkpoint has no matching LIBERO action statistics")
        self.unnorm_key = key
        self.codec = ActionCodec(
            self.model.vocab_size, self.model.get_action_stats(key), self.model.config.n_action_bins
        )
        if self.processor.tokenizer.vocab_size != self.model.vocab_size:
            raise ValueError("action vocabulary mismatch")
        self.adapters = Adapters(self.model, cfg.rank, cfg.alpha)
        # Checkpointing is confined to the language stack. Frozen vision never requires gradients.
        self.model.language_model.gradient_checkpointing_enable()
        self.model.language_model.enable_input_require_grads()
        self.initial = self.adapters.state()
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if any(
            not name.startswith("language_model.") or not name.endswith((".A", ".B")) for name in trainable
        ):
            raise RuntimeError("unexpected trainable base, vision or projector parameters")
        write_json(
            cfg.root / "model-audit.json",
            {
                "model_id": cfg.model_id,
                "revision": cfg.model_revision,
                "code_revision": cfg.code_revision,
                "action_statistics_key": self.unnorm_key,
                "action_statistics": self.model.get_action_stats(key),
                "quantization": cfg.quantization,
                "attention_requested": cfg.attention,
                "attention_language": getattr(self.model.language_model.config, "_attn_implementation", None),
                "adapter_layers": list(self.adapters.layers),
                "trainable_names": trainable,
                "trainable_parameters": sum(p.numel() for p in self.adapters.parameters()),
                "proprioception_policy_input": False,
            },
        )

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def image(self, image):
        import tensorflow as tf

        # Exact official OpenVLA center crop, after JPEG -> Lanczos resize used by LIBERO evaluation.
        tensor = tf.io.decode_image(tf.image.encode_jpeg(np.asarray(image)), expand_animations=False)
        tensor = tf.image.resize(tensor, (224, 224), method="lanczos3", antialias=True)
        tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
        tensor = tf.image.convert_image_dtype(tensor, tf.float32)
        offset = (1 - np.sqrt(0.9)) / 2
        cropped = tf.image.crop_and_resize(
            tensor[None], [[offset, offset, 1 - offset, 1 - offset]], [0], (224, 224)
        )[0]
        pixels = tf.image.convert_image_dtype(tf.clip_by_value(cropped, 0, 1), tf.uint8, saturate=True)
        return Image.fromarray(pixels.numpy())

    def inputs(self, window):
        prompt = f"In: What action should the robot take to {window['instruction'].lower()}?\nOut:"
        batch = self.processor(prompt, self.image(window["image"]), return_tensors="pt")
        return {
            key: value.to(self.device, dtype=torch.bfloat16 if key == "pixel_values" else value.dtype)
            for key, value in batch.items()
        }

    def prompt_ids(self, inputs):
        ids = inputs["input_ids"]
        if int(ids[0, -1]) != 29871:
            ids = torch.cat([ids, torch.tensor([[29871]], device=self.device)], dim=1)
        return ids

    def supervised(self, window, target):
        inputs = self.inputs(window)
        prefix = self.prompt_ids(inputs)
        tokens = torch.tensor(self.codec.encode(target), device=self.device, dtype=torch.long)[None]
        ids = torch.cat([prefix, tokens], dim=1)
        labels = torch.full_like(ids, -100)
        labels[:, -7:] = tokens
        with self.autocast():
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                pixel_values=inputs["pixel_values"],
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        # Image tokens are inserted after BOS. The last 8:-1 logits predict the 7 action tokens.
        action_logits = output.logits[:, -8:-1, :].float()
        ce = F.cross_entropy(action_logits.reshape(-1, action_logits.shape[-1]), tokens.reshape(-1))
        if not torch.isfinite(ce):
            raise FloatingPointError("nonfinite action-token loss")
        predicted = self.codec.decode(action_logits.argmax(-1).detach().cpu().numpy()[0])
        return ce, predicted

    def features(self, window):
        inputs = self.inputs(window)
        self.model.eval()
        with torch.no_grad(), self.autocast():
            visual = self.model.vision_backbone(inputs["pixel_values"]).mean(dim=1)
            embedding = self.model.get_input_embeddings()(inputs["input_ids"])
            mask = inputs["attention_mask"].unsqueeze(-1)
            text = (embedding * mask).sum(dim=1) / mask.sum(dim=1)
        return visual[0].float().cpu().numpy(), text[0].float().cpu().numpy()

    def action(self, window, gate):
        self.model.eval()
        self.adapters.gate(gate)
        inputs = self.inputs(window)
        with torch.no_grad(), self.autocast():
            native = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        from .actions import native_to_simulator

        return native_to_simulator(native)

    def training(self, enabled):
        # Eval mode prevents dropout differences between arms; gradients still work.
        self.model.eval()
        self.model.language_model.train(enabled)
        self.adapters.gate(1)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.head = nn.Linear(16, 7 * 256)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.weight.normal_(0, 0.02)
            bias = self.head.bias.view(7, 256)
            bias.zero_()
            bias[:, 153] = 2
            bias[6, 254] = 3

    def forward(self, x):
        return self.head(torch.tanh(self.q_proj(x) + self.v_proj(x))).view(7, 256)


class Synthetic:
    def __init__(self, cfg):
        self.cfg, self.device = cfg, torch.device("cpu")
        self.model = ToyModel()
        self.adapters = Adapters(self.model, cfg.rank, cfg.alpha, {"q_proj", "v_proj"})
        self.initial = self.adapters.state()
        self.codec = ActionCodec(320, {"q01": [-1] * 7, "q99": [1] * 7, "mask": [True] * 6 + [False]})

    def autocast(self):
        return nullcontext()

    def features(self, window):
        visual = np.asarray(window["image"]).mean((0, 1)) / 255
        text = np.zeros(8, dtype=np.float32)
        text[int(hashlib.sha256(window["instruction"].encode()).hexdigest()[:4], 16) % 8] = 1
        return visual.astype(np.float32), text

    def vector(self, window):
        visual, text = self.features(window)
        return torch.tensor(np.pad(np.concatenate([visual, text]), (0, 5)), dtype=torch.float32)

    def supervised(self, window, target):
        logits = self.model(self.vector(window))
        tokens = self.codec.encode(target)
        bins = np.clip(320 - tokens - 1, 0, 255)
        ce = F.cross_entropy(logits, torch.tensor(bins, dtype=torch.long))
        predicted_tokens = 320 - logits.argmax(-1).detach().numpy() - 1
        return ce, self.codec.decode(predicted_tokens)

    def action(self, window, gate):
        self.adapters.gate(gate)
        with torch.no_grad():
            _, action = self.supervised(window, np.zeros(7))
        return action

    def training(self, enabled):
        self.model.train(enabled)
        self.adapters.gate(1)


def make_backend(cfg):
    return Synthetic(cfg) if cfg.backend == "synthetic" else OpenVLA(cfg)
