"""
Minimal error-correction helpers for validation experiments.
"""

import numpy as np


def repetition_encode_bits(bits: np.ndarray, repeats: int = 3) -> np.ndarray:
    if repeats <= 0 or repeats % 2 == 0:
        raise ValueError(f"repeats must be a positive odd integer: {repeats}")

    flat = bits.astype(np.uint8).flatten()
    if not set(np.unique(flat).tolist()).issubset({0, 1}):
        raise ValueError("bits must contain only 0 or 1")

    return np.repeat(flat, repeats).astype(np.uint8)


def repetition_decode_bits(encoded: np.ndarray, repeats: int = 3) -> np.ndarray:
    if repeats <= 0 or repeats % 2 == 0:
        raise ValueError(f"repeats must be a positive odd integer: {repeats}")

    flat = encoded.astype(np.uint8).flatten()
    usable = (flat.size // repeats) * repeats
    if usable == 0:
        return np.array([], dtype=np.uint8)

    grouped = flat[:usable].reshape(-1, repeats)
    threshold = repeats // 2 + 1
    return (grouped.sum(axis=1) >= threshold).astype(np.uint8)


def interleave_bits(bits: np.ndarray, seed: int) -> np.ndarray:
    flat = bits.astype(np.uint8).flatten()
    order = np.random.default_rng(seed).permutation(flat.size)
    return flat[order]


def deinterleave_bits(bits: np.ndarray, seed: int) -> np.ndarray:
    flat = bits.astype(np.uint8).flatten()
    order = np.random.default_rng(seed).permutation(flat.size)
    restored = np.empty_like(flat)
    restored[order] = flat
    return restored

