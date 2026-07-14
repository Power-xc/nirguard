"""
tests/test_generator.py — generator.py 단위 테스트

검증 항목:
  1. 동일 user_id → 동일 패턴 (결정론적)
  2. 다른 user_id → 다른 패턴 (고유성)
  3. 출력 shape/dtype/값 범위 검증
  4. 두 패턴 간 해밍 거리 ≈ 50% (균등 분포)
  5. 빈 문자열/유니코드 edge case
"""

import numpy as np
import pytest

from watermark.generator import generate_pattern, PATTERN_SIZE


class TestGeneratePatternDeterminism:
    """동일 입력 → 동일 출력 (결정론적 보장)"""

    def test_same_id_returns_same_pattern(self) -> None:
        """같은 user_id 두 번 호출 → 완전히 동일한 패턴"""
        p1 = generate_pattern("user_001")
        p2 = generate_pattern("user_001")
        np.testing.assert_array_equal(p1, p2)

    def test_determinism_across_calls(self) -> None:
        """10회 반복 호출 — 항상 동일 결과"""
        expected = generate_pattern("user_abc")
        for _ in range(10):
            np.testing.assert_array_equal(generate_pattern("user_abc"), expected)


class TestGeneratePatternUniqueness:
    """다른 user_id → 확연히 다른 패턴"""

    def test_different_ids_produce_different_patterns(self) -> None:
        """user_001 ≠ user_002"""
        p1 = generate_pattern("user_001")
        p2 = generate_pattern("user_002")
        # 4,096비트 중 최소 1비트 이상 달라야 함
        assert not np.array_equal(p1, p2), "서로 다른 ID가 동일 패턴을 생성함"

    def test_hamming_distance_near_50_percent(self) -> None:
        """
        서로 다른 두 패턴의 해밍 거리 ≈ 40~60%.
        균등 분포 PRNG 특성상 기대값은 50%.
        이 범위를 벗어나면 패턴 구별성이 낮다는 신호.
        """
        p1 = generate_pattern("alice")
        p2 = generate_pattern("bob")
        hamming = float(np.sum(p1 ^ p2)) / p1.size
        assert 0.40 <= hamming <= 0.60, (
            f"해밍 거리 {hamming:.3f} — 예상 범위 [0.40, 0.60] 벗어남"
        )

    def test_multiple_pair_uniqueness(self) -> None:
        """100명의 패턴 — 모두 달라야 함 (충돌 없음)"""
        user_ids = [f"user_{i:04d}" for i in range(100)]
        patterns = [generate_pattern(uid) for uid in user_ids]
        # 각 패턴을 bytes로 변환해 집합에 넣어 중복 확인
        pattern_set = {p.tobytes() for p in patterns}
        assert len(pattern_set) == 100, "100개 패턴 중 충돌 발생"


class TestGeneratePatternShape:
    """출력 배열 형태/타입/값 검증"""

    def test_output_shape(self) -> None:
        """출력 shape = (PATTERN_SIZE, PATTERN_SIZE)"""
        p = generate_pattern("test_user")
        assert p.shape == (PATTERN_SIZE, PATTERN_SIZE), (
            f"shape 불일치: {p.shape} ≠ ({PATTERN_SIZE}, {PATTERN_SIZE})"
        )

    def test_output_dtype_uint8(self) -> None:
        """출력 dtype = uint8"""
        p = generate_pattern("test_user")
        assert p.dtype == np.uint8, f"dtype 불일치: {p.dtype}"

    def test_output_values_binary(self) -> None:
        """출력 값은 0 또는 1만 존재"""
        p = generate_pattern("test_user")
        unique_vals = np.unique(p)
        # 0만, 1만, 또는 0과 1 모두 — 다른 값 없어야 함
        assert set(unique_vals.tolist()).issubset({0, 1}), (
            f"이진값 외 발견: {unique_vals}"
        )

    def test_output_has_both_zero_and_one(self) -> None:
        """정상 패턴은 0과 1 모두 포함 (단색 패턴 방지)"""
        p = generate_pattern("test_user")
        assert 0 in p and 1 in p, "패턴이 단일 값만 포함"


class TestGeneratePatternEdgeCases:
    """경계값/특수 입력 처리"""

    def test_empty_string_user_id(self) -> None:
        """빈 문자열도 유효한 패턴 생성 (에러 없음)"""
        p = generate_pattern("")
        assert p.shape == (PATTERN_SIZE, PATTERN_SIZE)

    def test_unicode_user_id(self) -> None:
        """한국어/이모지 포함 ID — UTF-8 인코딩 후 정상 처리"""
        p = generate_pattern("사용자_001")
        assert p.shape == (PATTERN_SIZE, PATTERN_SIZE)
        assert p.dtype == np.uint8

    def test_very_long_user_id(self) -> None:
        """매우 긴 ID — SHA-256이 항상 32바이트를 반환하므로 정상"""
        long_id = "x" * 10000
        p = generate_pattern(long_id)
        assert p.shape == (PATTERN_SIZE, PATTERN_SIZE)

    def test_similar_ids_different_patterns(self) -> None:
        """
        거의 동일한 ID("user_001" vs "user_001 ")도 달라야 함.
        SHA-256 눈사태 효과(avalanche effect) 검증.
        """
        p1 = generate_pattern("user_001")
        p2 = generate_pattern("user_001 ")  # 공백 1개 차이
        assert not np.array_equal(p1, p2), "눈사태 효과 실패"
