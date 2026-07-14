"""
Screen-capture preprocessing helpers.
"""

import cv2
import numpy as np
from typing import Optional


def glare_mask(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"BGR 3-channel image required: shape={image.shape}")
    return np.all(image >= threshold, axis=2)


def apply_glare_mask(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    masked = image.copy()
    mask = glare_mask(masked, threshold=threshold)
    if np.any(mask):
        median = np.median(masked[~mask], axis=0) if np.any(~mask) else np.array([0, 0, 0])
        masked[mask] = median.astype(np.uint8)
    return masked


def order_quad(points: np.ndarray) -> np.ndarray:
    if points.shape != (4, 2):
        raise ValueError(f"Expected four points: shape={points.shape}")
    points = points.astype(np.float32)
    summed = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(summed)],
            points[np.argmin(diff)],
            points[np.argmax(summed)],
            points[np.argmax(diff)],
        ],
        dtype=np.float32,
    )


def warp_perspective(
    image: np.ndarray,
    quad: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    ordered = order_quad(quad)
    width, height = output_size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def detect_screen_quad(image: np.ndarray) -> Optional[np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"BGR 3-channel image required: shape={image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > image.shape[0] * image.shape[1] * 0.1:
            return order_quad(approx.reshape(4, 2))
    return None
