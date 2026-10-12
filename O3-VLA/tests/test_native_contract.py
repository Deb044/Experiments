"""CPU interface tests; these do not claim to run the native CUDA checkpoint."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from o3_vla.actions import ActionCodec
from o3_vla.backend import OpenVLA


def test_supervised_masks_prompt_and_accounts_for_image_tokens():
    backend = object.__new__(OpenVLA)
    backend.device = torch.device("cpu")
    backend.codec = ActionCodec(320, {"q01": [-1] * 7, "q99": [1] * 7, "mask": [True] * 6 + [False]})
    target = np.array([0, 0, 0, 0, 0, 0, -1.0])
    tokens = torch.as_tensor(backend.codec.encode(target))[None]
    prefix = torch.tensor([[1, 25, 27]])
    backend.inputs = lambda window: {"input_ids": prefix, "pixel_values": torch.zeros(1, 6, 224, 224)}
    backend.autocast = nullcontext

    def forward(**kwargs):
        ids, labels = kwargs["input_ids"], kwargs["labels"]
        assert ids[0, -8] == 29871
        assert torch.equal(labels[0, :-7], torch.full_like(labels[0, :-7], -100))
        assert torch.equal(labels[:, -7:], tokens)
        assert not kwargs["use_cache"]
        # Native forward inserts image patches; predictions must remain tail-aligned.
        logits = torch.zeros(1, ids.shape[1] + 3, 320)
        logits[:, -8:-1, :].scatter_(2, tokens[..., None], 20)
        return SimpleNamespace(logits=logits)

    backend.model = forward
    loss, decoded = backend.supervised({}, target)
    assert loss < 1e-5
    assert decoded[6] == -1
    assert np.all(np.abs(decoded[:6]) < 0.01)
