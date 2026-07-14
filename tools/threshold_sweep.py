#!/usr/bin/env python3
"""
Evaluate confidence and hamming thresholds against stored result JSON files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_results import load_results


def evaluate(results: list[dict[str, Any]], confidence_threshold: float, max_hamming: float) -> dict[str, Any]:
    accepted = 0
    true_positive = 0
    false_positive = 0
    abstained = 0

    for result in results:
        user_id = result.get("user_id")
        best = result.get("best", {})
        best_user = best.get("user_id")
        confidence = float(best.get("confidence", 0.0))
        hamming = float(best.get("hamming", 1.0))
        is_accepted = confidence >= confidence_threshold and hamming <= max_hamming

        if is_accepted:
            accepted += 1
            if best_user == user_id:
                true_positive += 1
            else:
                false_positive += 1
        else:
            abstained += 1

    total = len(results)
    return {
        "confidence_threshold": confidence_threshold,
        "max_hamming": max_hamming,
        "total": total,
        "accepted": accepted,
        "abstained": abstained,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "accept_rate": round(accepted / total * 100.0, 1) if total else 0.0,
        "abstain_rate": round(abstained / total * 100.0, 1) if total else 0.0,
        "false_positive_rate": round(false_positive / accepted * 100.0, 1) if accepted else 0.0,
    }


def sweep(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for confidence in (50.0, 60.0, 70.0, 80.0, 90.0):
        for hamming in (0.05, 0.10, 0.12, 0.15, 0.20):
            rows.append(evaluate(results, confidence, hamming))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep confidence/hamming thresholds")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(Path(args.results_dir))
    rows = sweep(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, ensure_ascii=False, indent=2))
