"""
tests/test_embedder.py — embedder.py 단위 테스트

검증 항목:
  1. PSNR ≥ 40dB (strength=4 기본값)
  2. strength=6 시 PSNR 여전히 허용 범위 내
  3. R채널만 변경, G/B 채널 불변
  4. 변경량이 정확히 ±strength
  5. [0, 255] 클리핑 보장
  6. 잘못된 입력 → ValueError
"""

import numpy as np
import pytest

from watermark.generator import generate_pattern, PATTERN_SIZE
from watermark.embedder import embed_image, _embed_frame, _tile_pattern
from watermark.metrics import calculate_psnr


# 테스트용 공통 픽스처
PATTERN = generate_pattern("test_embed_user")


class TestTilePattern:
    """_tile_pattern: 패턴 타일링 정확성"""

    def test_tile_exact_multiple(self) -> None:
        """패턴의 정확한 배수 크기 — 잘라냄 없이 완전 타일"""
        result = _tile_pattern(PATTERN, PATTERN_SIZE * 3, PATTERN_SIZE * 2)
        assert result.shape == (PATTERN_SIZE * 3, PATTERN_SIZE * 2)

    def test_tile_non_multiple(self) -> None:
        """정확한 배수가 아닌 크기 — 크롭해서 정확한 크기 반환"""
        h, w = 100, 150  # 64의 배수 아님
        result = _tile_pattern(PATTERN, h, w)
        assert result.shape == (h, w)

    def test_tile_preserves_pattern_values(self) -> None:
        """타일링 후 값이 원본 패턴과 동일"""
        result = _tile_pattern(PATTERN, PATTERN_SIZE, PATTERN_SIZE)
        np.testing.assert_array_equal(result, PATTERN)


class TestEmbedImage:
    """embed_image: PSNR, 채널 격리, 클리핑"""

    def _make_frame(self, h: int = 480, w: int = 640, val: int = 128) -> np.ndarray:
        """균일한 BGR 이미지 생성"""
        return np.full((h, w, 3), val, dtype=np.uint8)

    def test_psnr_meets_target_strength4(self) -> None:
        """strength=4 시 PSNR ≥ 40dB (PoC 핵심 지표)"""
        original = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        watermarked = embed_image(original, PATTERN, strength=4)
        psnr = calculate_psnr(original, watermarked)
        assert psnr >= 40.0, f"PSNR {psnr:.2f}dB — 목표 40dB 미달"

    def test_psnr_meets_target_strength6(self) -> None:
        """strength=6 (옵션) 시 PSNR ≥ 37dB (허용 하한)"""
        original = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        watermarked = embed_image(original, PATTERN, strength=6)
        psnr = calculate_psnr(original, watermarked)
        assert psnr >= 37.0, f"PSNR {psnr:.2f}dB — strength=6 허용 하한 미달"

    def test_only_r_channel_modified(self) -> None:
        """R채널(BGR[2])만 변경, G/B채널 불변"""
        original = self._make_frame()
        watermarked = embed_image(original, PATTERN, strength=4)

        # B 채널 (index 0) 불변
        np.testing.assert_array_equal(
            original[:, :, 0], watermarked[:, :, 0],
            err_msg="B 채널이 변경됨"
        )
        # G 채널 (index 1) 불변
        np.testing.assert_array_equal(
            original[:, :, 1], watermarked[:, :, 1],
            err_msg="G 채널이 변경됨"
        )
        # R 채널 (index 2) — 변경 있어야 함
        assert not np.array_equal(original[:, :, 2], watermarked[:, :, 2])

    def test_r_delta_is_exactly_strength(self) -> None:
        """
        클리핑이 발생하지 않는 중간값(128) 픽셀에서
        R 채널 변화량이 정확히 ±strength인지 검증.
        """
        original = self._make_frame(val=128)
        strength = 4
        watermarked = embed_image(original, PATTERN, strength=strength)

        # R 채널 차이 계산
        diff = watermarked[:, :, 2].astype(np.int16) - original[:, :, 2].astype(np.int16)
        tiled = _tile_pattern(PATTERN, 480, 640)

        # 비트=1인 위치는 +strength, 비트=0인 위치는 -strength
        expected_diff = (tiled.astype(np.int16) * 2 - 1) * strength
        np.testing.assert_array_equal(diff, expected_diff)

    def test_clipping_no_overflow(self) -> None:
        """극단값(0, 255) 픽셀에서 클리핑 후 [0, 255] 범위 보장"""
        # 최솟값 이미지 (0): R-=4 → -4 → 클리핑 → 0
        dark_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result_dark = embed_image(dark_frame, PATTERN, strength=4)
        assert result_dark.min() >= 0

        # 최댓값 이미지 (255): R+=4 → 259 → 클리핑 → 255
        bright_frame = np.full((64, 64, 3), 255, dtype=np.uint8)
        result_bright = embed_image(bright_frame, PATTERN, strength=4)
        assert result_bright.max() <= 255

    def test_output_dtype_uint8(self) -> None:
        """출력 dtype = uint8 유지"""
        original = self._make_frame()
        result = embed_image(original, PATTERN)
        assert result.dtype == np.uint8

    def test_output_shape_unchanged(self) -> None:
        """출력 shape = 입력 shape"""
        original = self._make_frame(h=360, w=640)
        result = embed_image(original, PATTERN)
        assert result.shape == original.shape


class TestEmbedImageValidation:
    """잘못된 입력 → ValueError"""

    def test_wrong_image_ndim(self) -> None:
        """2D 이미지 입력 시 ValueError"""
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="BGR 3채널"):
            embed_image(gray, PATTERN)

    def test_wrong_image_channels(self) -> None:
        """4채널 BGRA 이미지 입력 시 ValueError"""
        bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="BGR 3채널"):
            embed_image(bgra, PATTERN)

    def test_wrong_pattern_shape(self) -> None:
        """잘못된 패턴 shape 시 ValueError"""
        wrong_pattern = np.zeros((32, 32), dtype=np.uint8)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="패턴 shape"):
            embed_image(image, wrong_pattern)
