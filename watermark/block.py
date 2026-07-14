"""
Block-grid watermarking for validation.

The original pixel-tile mode is fragile under JPEG and resizing because the
signal changes at pixel frequency. Block mode embeds each pattern bit into a
contiguous image cell, which is easier to recover after mild transformations.
"""

import cv2
import numpy as np

from .alignment import apply_sync_template
from .generator import PATTERN_SIZE


def embed_block_image(
    image: np.ndarray,
    pattern: np.ndarray,
    strength: int = 4,
    use_sync: bool = False,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"BGR 3-channel image required: shape={image.shape}")
    if pattern.shape != (PATTERN_SIZE, PATTERN_SIZE):
        raise ValueError(f"Pattern shape must be ({PATTERN_SIZE}, {PATTERN_SIZE}): {pattern.shape}")

    h, w = image.shape[:2]
    block_h = h // PATTERN_SIZE
    block_w = w // PATTERN_SIZE
    if block_h == 0 or block_w == 0:
        raise ValueError(f"Image too small for block mode: {h}x{w}")

    encoded_pattern = apply_sync_template(pattern) if use_sync else pattern
    watermarked = image.astype(np.int16).copy()

    for row in range(PATTERN_SIZE):
        y0 = row * block_h
        y1 = (row + 1) * block_h if row < PATTERN_SIZE - 1 else h
        for col in range(PATTERN_SIZE):
            x0 = col * block_w
            x1 = (col + 1) * block_w if col < PATTERN_SIZE - 1 else w
            delta = strength if int(encoded_pattern[row, col]) == 1 else -strength
            watermarked[y0:y1, x0:x1, 2] += delta

    return np.clip(watermarked, 0, 255).astype(np.uint8)


def extract_block_from_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"BGR 3-channel image required: shape={image.shape}")

    h, w = image.shape[:2]
    block_h = h // PATTERN_SIZE
    block_w = w // PATTERN_SIZE
    if block_h == 0 or block_w == 0:
        raise ValueError(f"Image too small for block mode: {h}x{w}")

    r = image[:, :, 2].astype(np.float32)
    means = np.zeros((PATTERN_SIZE, PATTERN_SIZE), dtype=np.float32)
    for row in range(PATTERN_SIZE):
        y0 = row * block_h
        y1 = (row + 1) * block_h if row < PATTERN_SIZE - 1 else h
        for col in range(PATTERN_SIZE):
            x0 = col * block_w
            x1 = (col + 1) * block_w if col < PATTERN_SIZE - 1 else w
            means[row, col] = float(np.mean(r[y0:y1, x0:x1]))

    return otsu_binarize(detrend_grid(means))


def extract_block_from_frames(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("frames must not be empty")
    extracted = [extract_block_from_image(frame) for frame in frames]
    avg = np.mean(extracted, axis=0)
    return (avg >= 0.5).astype(np.uint8)


def otsu_binarize(values: np.ndarray) -> np.ndarray:
    mn = float(values.min())
    mx = float(values.max())
    if mx == mn:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = ((values - mn) / (mx - mn) * 255.0).astype(np.uint8)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary // 255).astype(np.uint8)


def detrend_grid(values: np.ndarray) -> np.ndarray:
    row_mean = values.mean(axis=1, keepdims=True)
    col_mean = values.mean(axis=0, keepdims=True)
    global_mean = float(values.mean())
    return values - row_mean - col_mean + global_mean
