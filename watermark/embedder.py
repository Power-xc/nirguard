"""
watermark/embedder.py — 워터마크 삽입기

영상(MP4) 또는 단일 이미지의 R(Red) 픽셀을 ±strength 조정해
64×64 이진 패턴을 삽입한다. 가시 품질 영향은 PSNR로 조건별 측정한다
(목표 PSNR ≥ 40dB).
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .generator import PATTERN_SIZE


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    64×64 패턴을 프레임 크기(H×W)에 맞게 반복 타일링한다.
    타일 경계에서 잘라내어 정확히 (height, width) 크기로 반환.
    """
    # ceil 나눗셈으로 필요한 타일 수 계산
    tiles_y = -(-height // PATTERN_SIZE)
    tiles_x = -(-width // PATTERN_SIZE)
    tiled = np.tile(pattern, (tiles_y, tiles_x))
    return tiled[:height, :width]


def _embed_frame(frame: np.ndarray, pattern: np.ndarray, strength: int) -> np.ndarray:
    """
    단일 프레임(BGR)에 워터마크를 삽입한다.

    OpenCV BGR 포맷 기준: R 채널 = 인덱스 2
    비트=1 → R += strength / 비트=0 → R -= strength
    클리핑: [0, 255] 범위 보장

    Args:
        frame:    (H, W, 3) uint8 BGR 프레임
        pattern:  (64, 64) uint8 {0, 1} 이진 패턴
        strength: R픽셀 조정 강도 (기본값 4, 옵션 6)
    """
    h, w = frame.shape[:2]
    tiled = _tile_pattern(pattern, h, w)

    # int16으로 캐스팅해 오버플로우 방지
    watermarked = frame.astype(np.int16)

    # {0 → -1, 1 → +1} 변환 후 strength 곱셈
    delta = (tiled.astype(np.int16) * 2 - 1) * strength  # shape: (H, W)

    # R 채널(BGR[2])에만 delta 적용
    watermarked[:, :, 2] += delta

    # [0, 255] 범위 클리핑 후 uint8 복원
    return np.clip(watermarked, 0, 255).astype(np.uint8)


def embed_image(
    image: np.ndarray,
    pattern: np.ndarray,
    strength: int = 4,
) -> np.ndarray:
    """
    단일 이미지(BGR ndarray)에 워터마크를 삽입해 반환한다.

    Args:
        image:    (H, W, 3) uint8 BGR 이미지
        pattern:  generate_pattern()으로 생성한 64×64 패턴
        strength: R픽셀 조정 강도 (기본 4, 가시 품질 영향은 PSNR로 측정)

    Returns:
        워터마크가 삽입된 (H, W, 3) uint8 BGR 이미지
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"BGR 3채널 이미지 필요: shape={image.shape}")
    if pattern.shape != (PATTERN_SIZE, PATTERN_SIZE):
        raise ValueError(f"패턴 shape 오류: 기대={PATTERN_SIZE}×{PATTERN_SIZE}, 실제={pattern.shape}")

    return _embed_frame(image, pattern, strength)


def embed_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    pattern: np.ndarray,
    strength: int = 4,
) -> dict:
    """
    MP4 영상 전체 프레임에 워터마크를 삽입한다.

    모든 프레임에 동일 패턴을 삽입 → 멀티프레임 평균화로 노이즈 제거 가능.

    Args:
        input_path:  원본 영상 경로
        output_path: 출력 경로 (.mp4)
        pattern:     64×64 이진 패턴
        strength:    R픽셀 조정 강도

    Returns:
        {"output_path": str, "frame_count": int, "fps": float}
    """
    input_path = str(input_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없음: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        watermarked_frame = _embed_frame(frame, pattern, strength)
        out.write(watermarked_frame)
        frame_count += 1

    cap.release()
    out.release()

    return {
        "output_path": output_path,
        "frame_count": frame_count,
        "fps": fps,
    }
