"""Exercise the native-flow hook contract without downloading a foundation model."""
import unittest

import torch
from torch import nn

from vla_weightspace.backends.smolvla import SmolVLABackend


class NativePolicyFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.action_out_proj = nn.Linear(3, 4)

    def forward(self, batch, noise=None, time=None):
        flow = self.model.action_out_proj(batch["hidden"])
        return (flow - noise).square().mean(), {}


class SmolVLAContractTests(unittest.TestCase):
    def test_hook_returns_native_flow_not_squared_training_loss(self):
        backend = SmolVLABackend.__new__(SmolVLABackend)
        backend.policy = NativePolicyFixture()
        backend.action_dim = 2
        batch = {"hidden": torch.randn(2, 3, 3)}
        noise = torch.randn(2, 3, 4)
        time = torch.rand(2)
        actual = backend.flow(batch, noise, time)
        expected = backend.policy.model.action_out_proj(batch["hidden"])[:, :, :2]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(backend.policy.model.action_out_proj._forward_hooks), 0)

    def test_native_validity_mask_excludes_padded_actions(self):
        backend = SmolVLABackend.__new__(SmolVLABackend)
        left = torch.tensor([[[1., 1.], [100., 100.]]])
        right = torch.zeros_like(left)
        batch = {"action_is_pad": torch.tensor([[False, True]])}
        self.assertEqual(float(backend.valid_mse(left, right, batch)), 1)


if __name__ == "__main__":
    unittest.main()
