"""
Alignment template helpers for screen-capture validation.
"""

import numpy as np


def apply_sync_template(pattern: np.ndarray, marker_size: int = 8) -> np.ndarray:
    if pattern.ndim != 2:
        raise ValueError(f"pattern must be 2D: shape={pattern.shape}")
    if marker_size <= 0:
        raise ValueError(f"marker_size must be positive: {marker_size}")
    if pattern.shape[0] < marker_size * 2 or pattern.shape[1] < marker_size * 2:
        raise ValueError("pattern is too small for sync markers")

    marked = pattern.copy().astype(np.uint8)
    marker = checkerboard(marker_size)
    inverse = 1 - marker

    marked[:marker_size, :marker_size] = marker
    marked[:marker_size, -marker_size:] = inverse
    marked[-marker_size:, :marker_size] = inverse
    marked[-marker_size:, -marker_size:] = marker
    return marked


def sync_score(pattern: np.ndarray, marker_size: int = 8) -> float:
    expected = apply_sync_template(np.zeros_like(pattern, dtype=np.uint8), marker_size)
    mask = np.zeros_like(pattern, dtype=bool)
    mask[:marker_size, :marker_size] = True
    mask[:marker_size, -marker_size:] = True
    mask[-marker_size:, :marker_size] = True
    mask[-marker_size:, -marker_size:] = True

    observed = pattern.astype(np.uint8)
    matches = int(np.sum(observed[mask] == expected[mask]))
    return round(matches / int(np.sum(mask)), 4)


def checkerboard(size: int) -> np.ndarray:
    rows, cols = np.indices((size, size))
    return ((rows + cols) % 2).astype(np.uint8)

