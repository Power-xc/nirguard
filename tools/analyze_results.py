#!/usr/bin/env python3
"""
Summarize NIRGuard validation JSON results.
"""

import argparse
import json
from pathlib import Path
from math import sqrt
from statistics import mean, stdev
from typing import Any


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "metrics" in data:
            results.append(data)
    return results


def rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values) * 100.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"count": 0}

    metrics = [result["metrics"] for result in results]
    bers = [metric["ber"] for metric in metrics]
    return {
        "count": len(results),
        "avg_psnr": round(mean(metric["psnr"] for metric in metrics), 2),
        "avg_ber": round(mean(bers), 4),
        "ber_95ci": confidence_interval(bers),
        "top1_rate": round(rate([metric["top1_success"] for metric in metrics]), 1),
        "top3_rate": round(rate([metric["top3_success"] for metric in metrics]), 1),
        "identified_rate": round(rate([metric["identified"] for metric in metrics]), 1),
        "abstain_rate": round(rate([metric["abstained"] for metric in metrics]), 1),
    }


def confidence_interval(values: list[float]) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        value = round(values[0], 4)
        return {"low": value, "high": value}
    margin = 1.96 * stdev(values) / sqrt(len(values))
    center = mean(values)
    return {
        "low": round(max(0.0, center - margin), 4),
        "high": round(min(1.0, center + margin), 4),
    }


def summarize_by_condition(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        name = result["condition"]["name"]
        grouped.setdefault(name, []).append(result)
    return {name: summarize(items) for name, items in grouped.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize NIRGuard validation results")
    parser.add_argument("--results-dir", default="experiments/results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(Path(args.results_dir))
    output = {
        "overall": summarize(results),
        "by_condition": summarize_by_condition(results),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
