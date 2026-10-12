"""Explicit simulator, native OpenVLA and action-token coordinate transforms."""

import numpy as np


def simulator_to_native(action):
    result = np.asarray(action, dtype=np.float64).copy()
    if result.shape[-1] != 7:
        raise ValueError("expected seven-dimensional LIBERO action")
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite action")
    # LIBERO: -1=open, +1=close. OpenVLA dataset: 1=open, 0=close.
    result[..., 6] = (1 - result[..., 6]) / 2
    return result


def native_to_simulator(action):
    result = np.asarray(action, dtype=np.float64).copy()
    result[..., 6] = np.where(result[..., 6] > 0.5, -1.0, 1.0)
    return result


def hold_action(gripper_qpos, threshold):
    # Binary open/closed approximation: intermediate apertures may not be preserved.
    # The rollout controller sanity check measures this limitation instead of assuming it away.
    aperture = float(np.abs(np.asarray(gripper_qpos)).sum())
    result = np.zeros(7, dtype=np.float64)
    result[6] = -1.0 if aperture >= threshold else 1.0
    return result


class ActionCodec:
    def __init__(self, vocab_size, stats, bins=256):
        self.vocab_size = vocab_size
        self.bins = np.linspace(-1, 1, bins)
        self.centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.low = np.asarray(stats["q01"], dtype=np.float64)
        self.high = np.asarray(stats["q99"], dtype=np.float64)
        self.mask = np.asarray(stats.get("mask", [True] * 7), dtype=bool)
        if self.low.shape != (7,) or self.high.shape != (7,) or self.mask.shape != (7,):
            raise ValueError("expected seven-dimensional action statistics")
        if (
            not np.all(np.isfinite(self.low))
            or not np.all(np.isfinite(self.high))
            or np.any((self.high <= self.low) & self.mask)
        ):
            raise ValueError("invalid checkpoint action statistics")

    def normalize(self, action):
        native = simulator_to_native(action)
        denominator = np.where(self.mask, self.high - self.low, 1)
        scaled = 2 * (native - self.low) / denominator - 1
        return np.clip(np.where(self.mask, scaled, native), -1, 1)

    def encode(self, action):
        return (self.vocab_size - np.digitize(self.normalize(action), self.bins)).astype(np.int64)

    def decode(self, tokens):
        indices = np.clip(self.vocab_size - np.asarray(tokens) - 1, 0, len(self.centers) - 1)
        normalized = self.centers[indices]
        native = np.where(self.mask, (normalized + 1) / 2 * (self.high - self.low) + self.low, normalized)
        return native_to_simulator(native)

    @property
    def half_bin_tolerance(self):
        return (self.high - self.low) / (len(self.centers) * 2)
