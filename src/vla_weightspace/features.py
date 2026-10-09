"""Request-conditioned features without target-weight information."""
import hashlib

import numpy as np


def summaries(value):
    flat = value.ravel().astype(np.float64)
    return np.array([flat.mean(), flat.std(), np.linalg.norm(flat), np.abs(flat).mean(), flat.min(), flat.max()])


def request_features(current, forget_gradient, retain_gradient):
    values = [current, forget_gradient, retain_gradient]
    result = [summaries(value) for value in values]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        dot = np.sum(values[left].astype(np.float64) * values[right])
        norm = np.linalg.norm(values[left].astype(np.float64)) * np.linalg.norm(values[right].astype(np.float64))
        result.append(np.array([dot, dot / norm if norm > 0 else 0.0]))
    # Fixed, non-learned projections retain information beyond aggregate moments.
    rng = np.random.default_rng(83)
    projected = np.zeros((3, 16), dtype=np.float64)
    size = current.size
    flats = np.stack([value.reshape(-1) for value in values])
    for start in range(0, size, 4096):
        signs = rng.integers(0, 2, size=(min(4096, size - start), 16), dtype=np.int8) * 2 - 1
        projected += flats[:, start:start + len(signs)] @ signs
    result.append(projected.ravel() / np.sqrt(size))
    return np.concatenate(result).astype(np.float32)


def deterministic_seed(*parts):
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "little")

