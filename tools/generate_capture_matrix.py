#!/usr/bin/env python3
"""
Generate a capture matrix manifest template.
"""

import argparse
import json
from itertools import product
from pathlib import Path


def generate(args: argparse.Namespace) -> dict:
    distances = [30, 50, 100]
    angles = [0, 15, 30]
    lighting = ["bright", "dim"]
    compression = ["source", "jpeg80", "jpeg70"]
    scale = [1.0, 0.75, 0.5]

    conditions = []
    for distance, angle, light, comp, sc in product(distances, angles, lighting, compression, scale):
        conditions.append(
            {
                "name": f"d{distance}_a{angle}_{light}_{comp}_s{str(sc).replace('.', '')}",
                "capture": {
                    "distance_cm": distance,
                    "angle_degrees": angle,
                    "lighting": light,
                },
                "postprocess": {
                    "compression": comp,
                    "scale": sc,
                },
            }
        )

    manifest = {
        "name": args.name,
        "asset_path": args.asset,
        "user_id": args.user_id,
        "recommended_mode": "block",
        "conditions": conditions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "condition_count": len(conditions)}, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a capture matrix manifest template")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--name", default="capture_matrix")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

