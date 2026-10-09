"""Small explicit LoRA wrappers; effective updates and gradients are identifiable."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.override = None
        self.capture_gradient = False
        self.gradient_leaves = []

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def delta(self):
        return self.B @ self.A

    def forward(self, x):
        if self.capture_gradient:
            effective = self.delta().detach().requires_grad_(True)
            self.gradient_leaves.append(effective)
            return F.linear(x, self.base.weight + effective.to(self.base.weight.dtype), self.base.bias)
        if self.override is not None:
            return F.linear(x, self.base.weight + self.override.to(self.base.weight.dtype), self.base.bias)
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


def attach_lora(model, rank):
    model.requires_grad_(False)
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
                and not any(token in name for token in ("vision_model", "vision_tower", "vision_encoder"))]
    if not selected:
        raise ValueError("No language/action q_proj or v_proj linear layers found")
    for name, base in selected:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, LoRALinear(base, rank))
    return {name: model.get_submodule(name) for name, _ in selected}


def factors(modules):
    return {f"{name}.{factor}": getattr(module, factor).detach().cpu().float().numpy().copy()
            for name, module in modules.items() for factor in ("A", "B")}


def load_factors(modules, values):
    expected = {f"{name}.{factor}" for name in modules for factor in ("A", "B")}
    if set(values) != expected:
        raise ValueError("Checkpoint adapter layer layout does not match the policy")
    with torch.no_grad():
        for name, module in modules.items():
            for factor in ("A", "B"):
                parameter = getattr(module, factor)
                value = values[f"{name}.{factor}"]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"Shape mismatch for {name}.{factor}")
                parameter.copy_(torch.as_tensor(value, device=parameter.device))
            module.override = None


def effective_updates(values):
    names = sorted(key[:-2] for key in values if key.endswith(".A"))
    return {name: values[name + ".B"] @ values[name + ".A"] for name in names}


def apply_updates(modules, updates):
    if set(modules) != set(updates):
        raise ValueError("Predicted effective updates have a different layer layout")
    for name, module in modules.items():
        if tuple(updates[name].shape) != tuple(module.delta().shape):
            raise ValueError(f"Invalid effective-update shape for {name}")
        module.override = torch.as_tensor(updates[name], dtype=torch.float32, device=module.A.device)


def effective_gradients(backend, batch, noise, time):
    modules = backend.modules
    for module in modules.values():
        module.capture_gradient = True
        module.gradient_leaves = []
    try:
        loss = backend.loss(batch, noise, time)
        leaves = [(name, leaf) for name, module in modules.items() for leaf in module.gradient_leaves]
        if not leaves:
            raise RuntimeError("No selected attention layers participated in the native objective")
        gradients = torch.autograd.grad(loss, [leaf for _, leaf in leaves], allow_unused=True)
        result = {name: np.zeros(tuple(module.delta().shape), dtype=np.float32) for name, module in modules.items()}
        for (name, _), gradient in zip(leaves, gradients):
            if gradient is not None:
                result[name] += gradient.detach().cpu().float().numpy()
        return result
    finally:
        for module in modules.values():
            module.capture_gradient = False
            module.gradient_leaves = []

