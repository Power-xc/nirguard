"""
watermark/metrics.py — 성능 측정 모듈

PSNR(Peak Signal-to-Noise Ratio): 워터마크 삽입 후 화질 저하 수치화.
BER(Bit Error Rate): 패턴 복원 정확도 측정.
목표: PSNR ≥ 40dB, BER ≤ 5%
"""

import numpy as np


def calculate_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    두 이미지(또는 단일 프레임) 간 PSNR을 계산한다.

    PSNR = 10 × log10(MAX² / MSE)
    MAX = 255 (uint8 최대값), MSE = 평균 제곱 오차

    Args:
        original:    원본 이미지/프레임 (H, W, C) uint8
        watermarked: 워터마크 삽입된 이미지/프레임 (H, W, C) uint8

    Returns:
        PSNR (dB). MSE=0이면 inf 반환 (완전 동일).
    """
    if original.shape != watermarked.shape:
        raise ValueError(
            f"이미지 크기 불일치: original={original.shape}, watermarked={watermarked.shape}"
        )

    # float64로 캐스팅해 정밀도 확보
    diff = original.astype(np.float64) - watermarked.astype(np.float64)
    mse: float = float(np.mean(diff ** 2))

    if mse == 0.0:
        return float("inf")

    # MAX=255 기준 PSNR (dB)
    psnr: float = 10.0 * np.log10((255.0 ** 2) / mse)
    return round(psnr, 2)


def calculate_ber(
    original_pattern: np.ndarray,
    extracted_pattern: np.ndarray,
) -> float:
    """
    두 패턴 간 Bit Error Rate를 계산한다.

    BER = (불일치 비트 수) / (전체 비트 수)
    목표: BER ≤ 0.05 (5%)

    Args:
        original_pattern:  원본 64×64 이진 패턴 (dtype=uint8, values={0,1})
        extracted_pattern: 복원된 64×64 이진 패턴 (dtype=uint8, values={0,1})

    Returns:
        BER [0.0, 1.0]. 0.0 = 완전 일치, 1.0 = 완전 불일치.
    """
    if original_pattern.shape != extracted_pattern.shape:
        raise ValueError(
            f"패턴 크기 불일치: original={original_pattern.shape}, "
            f"extracted={extracted_pattern.shape}"
        )

    # XOR로 불일치 비트 카운트: uint8 {0,1}이므로 XOR=1이 오류
    errors: int = int(np.sum(original_pattern ^ extracted_pattern))
    total: int = original_pattern.size

    return round(errors / total, 4)


def print_metrics_report(
    psnr: float,
    ber: float,
    label: str = "",
) -> None:
    """
    PSNR/BER 결과를 목표값 대비 Pass/Fail로 출력한다.

    목표 기준:
      PSNR ≥ 40dB → PASS
      BER  ≤ 0.05 → PASS

    Args:
        psnr:  calculate_psnr() 반환값
        ber:   calculate_ber() 반환값
        label: 테스트 시나리오 이름 (예: "JPEG Q=80")
    """
    header = f"[ NIRGuard 성능 리포트{' — ' + label if label else ''} ]"
    print(header)
    print("-" * len(header))

    psnr_pass = psnr >= 40.0
    ber_pass = ber <= 0.05

    psnr_str = f"{psnr:.2f} dB" if psnr != float("inf") else "∞ dB"
    print(f"  PSNR : {psnr_str:>10}  | 목표 ≥ 40dB  | {'PASS ✓' if psnr_pass else 'FAIL ✗'}")
    print(f"  BER  : {ber * 100:>9.2f}%  | 목표 ≤ 5%    | {'PASS ✓' if ber_pass else 'FAIL ✗'}")
    print("-" * len(header))
    print(f"  종합 : {'PASS ✓ — 시뮬레이션 목표 기준 충족' if psnr_pass and ber_pass else 'FAIL ✗ — 개선 필요'}")
    print()
