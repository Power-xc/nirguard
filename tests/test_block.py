import cv2
import numpy as np

from watermark import calculate_ber, calculate_psnr, embed_block_image, extract_block_from_image
from watermark.generator import generate_pattern


def make_image(size: int = 960) -> np.ndarray:
    y = np.linspace(40, 180, size).reshape(size, 1)
    x = np.linspace(30, 100, size).reshape(1, size)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[:, :, 0] = 50 + x * 0.4
    image[:, :, 1] = 70 + y * 0.4
    image[:, :, 2] = 90 + x * 0.6 + y * 0.25
    return np.clip(image, 0, 255).astype(np.uint8)


def test_block_roundtrip_clean() -> None:
    pattern = generate_pattern("user_003")
    original = make_image()
    watermarked = embed_block_image(original, pattern, strength=4)
    extracted = extract_block_from_image(watermarked)

    assert calculate_psnr(original, watermarked) >= 40.0
    assert calculate_ber(pattern, extracted) <= 0.05


def test_block_roundtrip_jpeg_q80() -> None:
    pattern = generate_pattern("user_003")
    original = make_image()
    watermarked = embed_block_image(original, pattern, strength=6)
    ok, encoded = cv2.imencode(".jpg", watermarked, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    extracted = extract_block_from_image(degraded)

    assert calculate_ber(pattern, extracted) <= 0.15

