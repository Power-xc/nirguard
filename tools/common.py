"""
tools/common.py — 실험 러너 공용 헬퍼

이미지 I/O, 후보 패턴 구성, confidence 정책 CLI 인자, 결과 저장을 공유한다.
정책 기본값은 watermark.ConfidencePolicy 한 곳에서만 정의한다.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from watermark import ConfidencePolicy, apply_sync_template, generate_pattern


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write image: {path}")


def build_candidate_patterns(user_id: str, count: int) -> list[tuple[str, np.ndarray]]:
    user_ids = [user_id]
    for index in range(count):
        candidate = f"user_{index:03d}"
        if candidate not in user_ids:
            user_ids.append(candidate)
    return [(candidate, generate_pattern(candidate)) for candidate in user_ids]


def candidate_patterns(user_id: str, count: int, use_sync: bool) -> list[tuple[str, np.ndarray]]:
    patterns = build_candidate_patterns(user_id, count)
    if not use_sync:
        return patterns
    return [(candidate, apply_sync_template(pattern)) for candidate, pattern in patterns]


def to_dict(candidate: Any) -> dict[str, Any]:
    return {
        "user_id": candidate.user_id,
        "hamming": candidate.hamming,
        "confidence": candidate.confidence,
    }


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    defaults = ConfidencePolicy()
    parser.add_argument("--max-hamming", type=float, default=defaults.max_hamming)
    parser.add_argument("--min-confidence", type=float, default=defaults.min_confidence)
    parser.add_argument("--min-confidence-gap", type=float, default=defaults.min_confidence_gap)


def policy_from_args(args: argparse.Namespace) -> ConfidencePolicy:
    return ConfidencePolicy(
        max_hamming=args.max_hamming,
        min_confidence=args.min_confidence,
        min_confidence_gap=args.min_confidence_gap,
    )


def write_result(out_dir: Path, run_id: str, result: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{run_id}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved result: {result_path}")
    return result_path
