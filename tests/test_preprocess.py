import cv2
import numpy as np

from watermark.preprocess import apply_glare_mask, detect_screen_quad, glare_mask, warp_perspective


def test_glare_mask_detects_saturated_pixels() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[2, 3] = [255, 255, 255]

    mask = glare_mask(image)

    assert mask[2, 3]
    assert not mask[0, 0]


def test_apply_glare_mask_replaces_saturated_pixels() -> None:
    image = np.full((10, 10, 3), 50, dtype=np.uint8)
    image[2, 3] = [255, 255, 255]

    masked = apply_glare_mask(image)

    assert masked[2, 3, 0] == 50


def test_warp_perspective_shape() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    quad = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)

    warped = warp_perspective(image, quad, (64, 64))

    assert warped.shape == (64, 64, 3)


def test_detect_screen_quad() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (170, 160), (255, 255, 255), 3)

    quad = detect_screen_quad(image)

    assert quad is not None
    assert quad.shape == (4, 2)

