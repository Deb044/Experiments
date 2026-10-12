"""Small explicit LoRA wrapper supporting fixed per-control-step scaling."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.multiplier = alpha / rank
        self.gate = 1.0

    def forward(self, x):
        base = self.base(x)
        if self.gate == 0:
            return base
        update = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B)
        return base + update.to(base.dtype) * self.multiplier * self.gate


class Adapters:
    def __init__(self, model, rank, alpha, target_names=None):
        model.requires_grad_(False)
        self.layers = {}
        for name, layer in list(model.named_modules()):
            match = (
                (name in target_names)
                if target_names is not None
                else (name.startswith("language_model.") and name.endswith((".q_proj", ".v_proj")))
            )
            if not match:
                continue
            if not all(hasattr(layer, key) for key in ["in_features", "out_features", "weight"]):
                raise TypeError(f"not a linear projection: {name}")
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapped = LoRALinear(layer, rank, alpha)
            setattr(parent, leaf, wrapped)
            self.layers[name] = wrapped
        if not self.layers:
            raise ValueError("no language q/v projections found; refuse silently ineffective training")

    def parameters(self):
        return [p for layer in self.layers.values() for p in [layer.A, layer.B]]

    def gate(self, value):
        if not 0 <= value <= 1:
            raise ValueError("adapter gate must be in [0,1]")
        for layer in self.layers.values():
            layer.gate = float(value)

    def state(self):
        return {
            name: {"A": layer.A.detach().cpu().clone(), "B": layer.B.detach().cpu().clone()}
            for name, layer in self.layers.items()
        }

    def load(self, state):
        if set(state) != set(self.layers):
            raise ValueError("adapter layer specification changed")
        with torch.no_grad():
            for name, layer in self.layers.items():
                for key in ["A", "B"]:
                    value = state[name][key]
                    target = getattr(layer, key)
                    if target.shape != value.shape:
                        raise ValueError(f"adapter shape mismatch: {name}.{key}")
                    target.copy_(value)

    def orthogonal_loss(self, previous):
        # O3 paper uses AB: its output-side A corresponds to our B, not our A.
        # Previous output factors are constants. Penalize column-space overlap.
        losses = [
            (previous[name]["B"].detach().to(layer.B.device).T @ layer.B).square().sum()
            for name, layer in self.layers.items()
        ]
        return torch.stack(losses).sum()

    def effective_norms(self):
        return {
            name: float(torch.linalg.vector_norm(layer.B.detach() @ layer.A.detach()).cpu())
            for name, layer in self.layers.items()
        }
