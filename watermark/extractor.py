"""
watermark/extractor.py — 워터마크 추출기

멀티프레임 R채널 평균화 + Otsu 이진화로 64×64 패턴을 복원한다.
재촬영 열화 시뮬레이션(JPEG 압축/노이즈/다운스케일) 조건에서 검증한다.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .generator import PATTERN_SIZE


def _sample_r_frames(cap: cv2.VideoCapture, n_frames: int) -> list[np.ndarray]:
    """
    VideoCapture에서 n_frames개를 균등 샘플링해 R 채널(float32) 리스트 반환.

    총 프레임 수를 알 수 없는 경우(일부 코덱) 순차 읽기로 폴백.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total > 0:
        # 균등 샘플링: 0 ~ total-1 사이에서 n_frames개 인덱스 선택
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        r_frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                # R 채널: OpenCV BGR에서 인덱스 2
                r_frames.append(frame[:, :, 2].astype(np.float32))
        return r_frames
    else:
        # 총 프레임 수 미지원 코덱 — 순차 읽기
        r_frames = []
        while len(r_frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            r_frames.append(frame[:, :, 2].astype(np.float32))
        return r_frames


def _block_mean(avg_r: np.ndarray, pattern_size: int = PATTERN_SIZE) -> np.ndarray:
    """
    평균 R 채널(H×W)에서 pattern_size×pattern_size 블록 단위 평균을 계산한다.

    타일링의 역연산:
      - 프레임을 pattern_size×pattern_size 셀로 분할
      - 각 셀의 R평균값 → (pattern_size, pattern_size) 배열 반환

    블록 경계 나머지 픽셀은 무시 (오차 최소화를 위해 완전한 블록만 사용).
    """
    h, w = avg_r.shape
    block_h = h // pattern_size
    block_w = w // pattern_size

    if block_h == 0 or block_w == 0:
        raise ValueError(
            f"프레임이 너무 작음: {h}×{w} < 패턴 크기 {pattern_size}×{pattern_size}"
        )

    # 완전한 블록만 사용 (나머지 잘라냄)
    cropped = avg_r[: block_h * pattern_size, : block_w * pattern_size]

    # reshape: (block_h, 64, block_w, 64) → mean(axis=(0,2)) → (64, 64)
    # 타일 역연산: 동일 패턴 위치(p,q)의 모든 픽셀(k*64+p, m*64+q)을 평균
    # 잘못된 방식: reshape(64, block_h, 64, block_w) — 연속 블록 평균 (BER ~0.5)
    reshaped = cropped.reshape(block_h, pattern_size, block_w, pattern_size)
    block_means: np.ndarray = reshaped.mean(axis=(0, 2))

    return block_means


def _otsu_binarize(block_means: np.ndarray) -> np.ndarray:
    """
    Otsu 알고리즘으로 block_means를 0/1 이진 패턴으로 변환한다.

    block_means를 [0, 255] uint8로 정규화 후 cv2.threshold(THRESH_OTSU) 적용.
    +strength 삽입 셀(높은 평균) → 1 / -strength 삽입 셀(낮은 평균) → 0
    """
    mn: float = float(block_means.min())
    mx: float = float(block_means.max())

    if mx == mn:
        # 모든 블록 평균이 동일 → 패턴 복원 불가 (영상이 단색이거나 삽입 안 됨)
        return np.zeros(block_means.shape, dtype=np.uint8)

    # 정규화: [min, max] → [0, 255]
    normalized = ((block_means - mn) / (mx - mn) * 255.0).astype(np.uint8)

    # Otsu 이진화: 반환값 binary는 0 또는 255
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # [0, 255] → [0, 1] 변환
    return (binary // 255).astype(np.uint8)


def extract_from_image(image: np.ndarray) -> np.ndarray:
    """
    단일 이미지에서 워터마크 패턴을 복원한다.
    멀티프레임 평균화 없이 단일 프레임 R채널만 사용.

    Args:
        image: (H, W, 3) uint8 BGR 이미지

    Returns:
        복원된 (64, 64) uint8 이진 패턴
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"BGR 3채널 이미지 필요: shape={image.shape}")

    r_channel = image[:, :, 2].astype(np.float32)
    block_means = _block_mean(r_channel)
    return _otsu_binarize(block_means)


def extract_from_frames(frames: list[np.ndarray]) -> np.ndarray:
    """
    프레임 배열에서 워터마크 패턴을 복원한다.
    비디오 코덱 저장 없이 멀티프레임 복원 알고리즘 자체를 검증할 때 사용한다.
    """
    if not frames:
        raise ValueError("프레임 목록이 비어 있음")

    r_frames = []
    expected_shape = frames[0].shape
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"BGR 3채널 프레임 필요: shape={frame.shape}")
        if frame.shape != expected_shape:
            raise ValueError("모든 프레임의 shape이 같아야 함")
        r_frames.append(frame[:, :, 2].astype(np.float32))

    avg_r = np.mean(r_frames, axis=0)
    block_means = _block_mean(avg_r)
    return _otsu_binarize(block_means)


def extract_from_video(
    video_path: Union[str, Path],
    n_frames: int = 60,
) -> np.ndarray:
    """
    영상에서 워터마크 패턴을 복원한다.

    멀티프레임 R채널 평균화로 프레임 단위 노이즈를 상쇄한다.
    조건별 BER은 검증 하네스에서 측정한다 (목표 BER ≤ 5%).

    Args:
        video_path: 워터마크 삽입된 (또는 아날로그 시뮬레이션된) 영상 경로
        n_frames:   샘플링할 프레임 수 (기본 60)

    Returns:
        복원된 (64, 64) uint8 이진 패턴
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없음: {video_path}")

    r_frames = _sample_r_frames(cap, n_frames)
    cap.release()

    if not r_frames:
        raise ValueError("유효한 프레임을 읽지 못함")

    # 멀티프레임 평균화: 랜덤 노이즈는 평균으로 상쇄, 워터마크 신호는 강화
    avg_r: np.ndarray = np.mean(r_frames, axis=0)

    block_means = _block_mean(avg_r)
    return _otsu_binarize(block_means)
