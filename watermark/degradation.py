"""
watermark/degradation.py — 재촬영 열화 시뮬레이션

JPEG 압축 → 가우시안 노이즈 → 다운스케일 순서로 디지털 열화를 적용한다.
소프트웨어 파이프라인 검증용이며, 실물 화면 재촬영(광학 경로)을 대체하지 않는다.
"""

from typing import Optional

import cv2
import numpy as np


def apply_degradation(
    image: np.ndarray,
    jpeg_quality: int = 100,
    noise_sigma: float = 0.0,
    downscale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    이미지에 재촬영 열화(JPEG → 노이즈 → 다운스케일)를 순서대로 적용한다.

    Args:
        image:        (H, W, 3) uint8 BGR 이미지
        jpeg_quality: JPEG 품질 (100이면 압축 생략)
        noise_sigma:  가우시안 노이즈 표준편차 (0이면 생략)
        downscale:    축소 비율 (1.0이면 생략, 축소 후 원 해상도로 복원)
        rng:          노이즈용 난수 생성기 (미지정 시 비결정론적)

    Returns:
        열화된 (H, W, 3) uint8 BGR 이미지
    """
    degraded = image.copy()

    if jpeg_quality < 100:
        ok, encoded = cv2.imencode(".jpg", degraded, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise ValueError("JPEG 인코딩 실패")
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if noise_sigma > 0:
        generator = rng if rng is not None else np.random.default_rng()
        noise = generator.normal(0, noise_sigma, degraded.shape).astype(np.float32)
        degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if downscale < 1.0:
        h, w = degraded.shape[:2]
        small = cv2.resize(degraded, (max(1, int(w * downscale)), max(1, int(h * downscale))))
        degraded = cv2.resize(small, (w, h))

    return degraded
