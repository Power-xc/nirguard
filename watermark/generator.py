"""
watermark/generator.py — 워터마크 패턴 생성기

User ID → SHA-256 해시 → PRNG 시드 → 64×64 이진 패턴 생성.
동일 user_id는 항상 동일 패턴을 반환한다 (결정론적).
"""

import hashlib

import numpy as np

# 패턴 크기: 64×64 = 4,096비트 — 충분한 엔트로피, 빠른 처리
PATTERN_SIZE = 64


def generate_pattern(user_id: str) -> np.ndarray:
    """
    user_id로부터 64×64 이진 워터마크 패턴을 생성한다.

    알고리즘:
      1. user_id → UTF-8 인코딩 → SHA-256 다이제스트 (32바이트)
      2. 다이제스트 전체를 정수로 변환 → PRNG(PCG64) 시드
      3. 64×64 = 4,096개의 0/1 난수 생성

    Args:
        user_id: 유출자 추적 대상 식별자 (예: "user_001")

    Returns:
        shape=(64, 64), dtype=uint8, values={0, 1} 이진 패턴
    """
    # SHA-256: 동일 입력 → 동일 32바이트 다이제스트 (결정론적 보장)
    digest: bytes = hashlib.sha256(user_id.encode("utf-8")).digest()

    # 32바이트(256비트)를 정수로 변환 — numpy PCG64 시드로 사용
    # numpy default_rng는 임의 정밀도 정수를 시드로 수용
    seed: int = int.from_bytes(digest, byteorder="big")

    rng = np.random.default_rng(seed)

    # 0 또는 1의 균등 분포 64×64 배열 생성
    # 기대 해밍 거리: 서로 다른 두 패턴 간 ≈ 50% (최적 구별성)
    pattern: np.ndarray = rng.integers(0, 2, size=(PATTERN_SIZE, PATTERN_SIZE), dtype=np.uint8)

    return pattern
