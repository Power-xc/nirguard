#!/usr/bin/env python3
"""
Build a clean external validation package from result files.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_results import load_results, summarize, summarize_by_condition


DOCS = [
    "README.md",
    "docs/CLAIMS_POLICY.md",
    "docs/VALIDATION_PLAN.md",
    "docs/HARDWARE_TEST_BENCH.md",
    "docs/CAPTURE_DATASET_SCHEMA.md",
    "docs/REAL_CAPTURE_PROTOCOL.md",
    "docs/LEGAL_EVIDENCE_PLAYBOOK.md",
    "docs/TAKEDOWN_WORKFLOW.md",
    "docs/EXTERNAL_VALIDATION_PACKAGE.md",
    "business/PILOT_BRIEF.md",
    "business/EXIT_STRATEGY.md",
]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def build(args: argparse.Namespace) -> dict:
    root = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc in DOCS:
        copy_if_exists(root / doc, out_dir / doc)

    results_dir = Path(args.results_dir)
    package_results = out_dir / "results"
    package_results.mkdir(parents=True, exist_ok=True)
    for path in sorted(results_dir.glob("*.json")):
        copy_if_exists(path, package_results / path.name)
    for path in sorted(results_dir.glob("*.pdf")):
        copy_if_exists(path, package_results / path.name)
    for path in sorted(results_dir.glob("*threshold*.json")):
        copy_if_exists(path, package_results / path.name)

    results = load_results(results_dir)
    analysis = {
        "overall": summarize(results),
        "by_condition": summarize_by_condition(results),
    }
    analysis_path = out_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = out_dir / "PACKAGE_README.md"
    readme.write_text(
        "# NIRGuard External Validation Package\n\n"
        "This package contains scoped technical validation materials.\n"
        "It does not contain broad market or universal performance claims.\n\n"
        "Start with `analysis.json`, then review representative evidence PDFs.\n",
        encoding="utf-8",
    )

    result = {"package": str(out_dir), "analysis": str(analysis_path), "result_count": len(results)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an external validation package")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
