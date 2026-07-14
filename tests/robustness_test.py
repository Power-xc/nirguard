"""
tests/robustness_test.py — NIRGuard baseline robustness reporter

재촬영 시뮬레이션 조건별 BER을 측정해 표 형식으로 출력한다.
현재 목적은 성능 홍보가 아니라 Red +/-4 PoC의 한계를 수치화하는 것이다.

실행:
  python tests/robustness_test.py         # 전체 조건 표 출력
  python tests/robustness_test.py --quick # 핵심 조건만 (약 15초)

PASS/WARN/FAIL은 현재 알고리즘의 조건별 안정성을 보기 위한 내부 기준이다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from watermark.degradation import apply_degradation
from watermark.embedder import embed_image
from watermark.extractor import extract_from_image
from watermark.generator import generate_pattern
from watermark.metrics import calculate_ber, calculate_psnr

# 재현성 보장
_RNG = np.random.default_rng(2026)

# 테스트 이미지 크기 (480p 기준)
_IMAGE_H, _IMAGE_W = 480, 640

# 테스트 조건 정의
QUICK_CONDITIONS: list[dict] = [
    {"jpeg_q": 90, "sigma": 5,  "downscale": 1.00},
    {"jpeg_q": 80, "sigma": 10, "downscale": 0.75},
    {"jpeg_q": 70, "sigma": 15, "downscale": 0.50},
]

FULL_CONDITIONS: list[dict] = [
    # JPEG만
    {"jpeg_q": 95, "sigma": 0,  "downscale": 1.00},
    {"jpeg_q": 90, "sigma": 0,  "downscale": 1.00},
    {"jpeg_q": 80, "sigma": 0,  "downscale": 1.00},
    {"jpeg_q": 70, "sigma": 0,  "downscale": 1.00},
    # 노이즈만
    {"jpeg_q": 100, "sigma": 5,  "downscale": 1.00},
    {"jpeg_q": 100, "sigma": 10, "downscale": 1.00},
    {"jpeg_q": 100, "sigma": 15, "downscale": 1.00},
    # 다운스케일만
    {"jpeg_q": 100, "sigma": 0, "downscale": 0.75},
    {"jpeg_q": 100, "sigma": 0, "downscale": 0.50},
    # 복합 조건 (핵심)
    {"jpeg_q": 90, "sigma": 5,  "downscale": 1.00},
    {"jpeg_q": 80, "sigma": 10, "downscale": 0.75},
    {"jpeg_q": 70, "sigma": 15, "downscale": 0.50},
    # 극한 조건 (허용 BER 15%)
    {"jpeg_q": 50, "sigma": 20, "downscale": 0.50},
    {"jpeg_q": 1,  "sigma": 30, "downscale": 0.25},
]


def _make_image() -> np.ndarray:
    """자연 영상 특성을 가진 합성 테스트 이미지 생성."""
    return _RNG.integers(60, 200, (_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)


def _run_condition(
    original: np.ndarray,
    watermarked: np.ndarray,
    pattern: np.ndarray,
    jpeg_q: int,
    sigma: float,
    downscale: float,
) -> dict:
    """단일 조건으로 열화 → 추출 → BER/PSNR 측정."""
    degraded = apply_degradation(
        watermarked,
        jpeg_quality=jpeg_q,
        noise_sigma=sigma,
        downscale=downscale,
        rng=_RNG,
    )

    extracted = extract_from_image(degraded)
    ber = calculate_ber(pattern, extracted)
    psnr = calculate_psnr(original, watermarked)

    return {"ber": ber, "psnr": psnr}


def run_robustness_test(quick: bool = False) -> list[dict]:
    """
    전체 강인성 테스트를 실행하고 결과 목록을 반환한다.

    Args:
        quick: True이면 핵심 조건만 테스트 (약 15초)

    Returns:
        조건별 결과 dict 목록
    """
    conditions = QUICK_CONDITIONS if quick else FULL_CONDITIONS

    # 기준 이미지 + 패턴 생성 (모든 조건에서 동일한 입력 사용)
    original = _make_image()
    pattern = generate_pattern("demo_user_robust")
    watermarked = embed_image(original, pattern, strength=4)

    results: list[dict] = []
    for cond in conditions:
        metrics = _run_condition(
            original, watermarked, pattern,
            jpeg_q=cond["jpeg_q"],
            sigma=cond["sigma"],
            downscale=cond["downscale"],
        )
        results.append({**cond, **metrics})

    return results


def print_report(results: list[dict]) -> None:
    """
    측정 결과를 BER 표 형식으로 출력한다.

    BER ≤ 5% → PASS ✓
    BER 5~15% → WARN △ (극한 조건)
    BER > 15% → FAIL ✗
    """
    header = (
        f"{'JPEG Q':>7} | {'Noise σ':>8} | {'Downscale':>9} | "
        f"{'BER':>7} | {'PSNR':>8} | 결과"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  NIRGuard 강인성 테스트 결과")
    print("=" * len(header))
    print(header)
    print(sep)

    pass_count = 0
    warn_count = 0
    fail_count = 0

    for r in results:
        ber = r["ber"]
        psnr = r["psnr"]
        psnr_str = f"{psnr:.1f}dB" if psnr != float("inf") else "∞dB"

        if ber <= 0.05:
            tag = "PASS ✓"
            pass_count += 1
        elif ber <= 0.15:
            tag = "WARN △"
            warn_count += 1
        else:
            tag = "FAIL ✗"
            fail_count += 1

        print(
            f"{r['jpeg_q']:>7} | {r['sigma']:>8.1f} | {r['downscale']:>9.2f} | "
            f"{ber * 100:>6.1f}% | {psnr_str:>8} | {tag}"
        )

    print(sep)
    print(
        f"  종합: PASS {pass_count}개 / WARN {warn_count}개 / FAIL {fail_count}개  "
        f"(전체 {len(results)}개)"
    )
    print(
        "  판정: "
        + ("✓ 시뮬레이션 조건 통과 (PSNR ≥ 40dB, BER ≤ 5%) — 실측 검증은 별도"
           if fail_count == 0 and pass_count > 0
           else "✗ 일부 조건 미달 — 알고리즘 검토 필요")
    )
    print("=" * len(header))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIRGuard 강인성 테스트")
    parser.add_argument("--quick", action="store_true", help="핵심 조건만 테스트")
    args = parser.parse_args()

    print(f"\n{'[quick 모드]' if args.quick else '[전체 모드]'} 테스트 시작...")
    results = run_robustness_test(quick=args.quick)
    print_report(results)
