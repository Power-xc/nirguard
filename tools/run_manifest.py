#!/usr/bin/env python3
"""
Run a batch of NIRGuard experiments from a manifest.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_experiment import run as run_experiment  # noqa: E402
from watermark import ConfidencePolicy  # noqa: E402

_POLICY_DEFAULTS = ConfidencePolicy()


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_temp_metadata(out_dir: Path, index: int, metadata: dict[str, Any]) -> Optional[str]:
    if not metadata:
        return None
    metadata_dir = out_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / f"metadata_{index:04d}.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def base_args(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    policy = manifest.get("policy", {})
    return {
        "out_dir": str(out_dir),
        "strength": int(manifest.get("strength", 4)),
        "candidate_count": int(manifest.get("candidate_count", 100)),
        "top_k": int(manifest.get("top_k", 3)),
        "max_hamming": float(policy.get("max_hamming", _POLICY_DEFAULTS.max_hamming)),
        "min_confidence": float(policy.get("min_confidence", _POLICY_DEFAULTS.min_confidence)),
        "min_confidence_gap": float(policy.get("min_confidence_gap", _POLICY_DEFAULTS.min_confidence_gap)),
        "save_images": bool(manifest.get("save_images", False)),
        "mode": manifest.get("mode", "pixel"),
        "use_sync": bool(manifest.get("use_sync", False)),
    }


def synthetic_jobs(manifest: dict[str, Any], out_dir: Path) -> list[SimpleNamespace]:
    jobs: list[SimpleNamespace] = []
    defaults = base_args(manifest, out_dir)
    conditions = manifest.get("conditions", [])
    for source in manifest.get("inputs", []):
        for condition in conditions:
            jobs.append(SimpleNamespace(
                **defaults,
                input=source["path"],
                user_id=source["user_id"],
                suspect_image=None,
                condition_name=condition["name"],
                jpeg_quality=int(condition.get("jpeg_quality", 100)),
                noise_sigma=float(condition.get("noise_sigma", 0.0)),
                downscale=float(condition.get("downscale", 1.0)),
                metadata_json=None,
            ))
    return jobs


def capture_jobs(manifest: dict[str, Any], out_dir: Path) -> list[SimpleNamespace]:
    jobs: list[SimpleNamespace] = []
    defaults = base_args(manifest, out_dir)
    index = 0
    for source in manifest.get("inputs", []):
        for capture in source.get("captures", []):
            index += 1
            metadata_json = write_temp_metadata(out_dir, index, capture.get("metadata", {}))
            jobs.append(SimpleNamespace(
                **defaults,
                input=source["path"],
                user_id=source["user_id"],
                suspect_image=capture["path"],
                condition_name=capture["name"],
                jpeg_quality=100,
                noise_sigma=0.0,
                downscale=1.0,
                metadata_json=metadata_json,
            ))
    return jobs


def run(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_manifest(manifest_path)
    out_dir = Path(manifest.get("out_dir", "experiments/results")) / manifest.get("name", "manifest")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = synthetic_jobs(manifest, out_dir) + capture_jobs(manifest, out_dir)
    results = [run_experiment(job) for job in jobs]

    summary_path = out_dir / "manifest_results.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBatch complete: {len(results)} experiments")
    print(f"Saved manifest results: {summary_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a NIRGuard experiment manifest")
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(Path(parse_args().manifest))
