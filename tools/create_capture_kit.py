#!/usr/bin/env python3
"""
Create a real-capture kit: original asset, watermarked asset, metadata template, and manifest.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.common import read_image, write_image  # noqa: E402
from watermark import embed_block_image, embed_image, generate_pattern  # noqa: E402


def create(args: argparse.Namespace) -> dict:
    source = Path(args.input)
    out_dir = Path(args.out_dir)
    originals = out_dir / "originals"
    watermarked_dir = out_dir / "watermarked"
    recaptures = out_dir / "recaptures"
    metadata_dir = out_dir / "metadata"
    for directory in (originals, watermarked_dir, recaptures, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    original_path = originals / source.name
    shutil.copyfile(source, original_path)

    image = read_image(source)
    pattern = generate_pattern(args.user_id)
    if args.mode == "block":
        watermarked = embed_block_image(image, pattern, strength=args.strength, use_sync=args.use_sync)
    else:
        watermarked = embed_image(image, pattern, strength=args.strength)
    watermarked_path = watermarked_dir / f"{source.stem}_{args.user_id}_{args.mode}.png"
    write_image(watermarked_path, watermarked)

    placeholder_capture = recaptures / f"{source.stem}_{args.user_id}_recapture.png"
    cv2.imwrite(str(placeholder_capture), watermarked)

    metadata = {
        "sample_id": f"{source.stem}_{args.user_id}_capture_001",
        "original_path": str(original_path),
        "suspect_path": str(placeholder_capture),
        "user_id": args.user_id,
        "content_type": "image",
        "display": {
            "model": "TBD",
            "panel_type": "TBD",
            "resolution": "TBD",
            "brightness_percent": None,
            "color_mode": "default",
        },
        "camera": {
            "model": "TBD",
            "mode": "photo",
            "resolution": None,
            "codec": None,
        },
        "capture": {
            "distance_cm": None,
            "angle_degrees": None,
            "ambient_lux": None,
            "tripod": False,
            "focus_locked": False,
            "exposure_locked": False,
        },
        "postprocess": {
            "crop": False,
            "resize": None,
            "compression": None,
        },
        "notes": "Replace placeholder recapture with a real camera capture before external validation.",
    }
    metadata_path = metadata_dir / f"{source.stem}_{args.user_id}_capture_001.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "name": f"{source.stem}_{args.user_id}_real_capture",
        "out_dir": "experiments/results",
        "strength": args.strength,
        "mode": args.mode,
        "use_sync": args.use_sync,
        "candidate_count": 100,
        "top_k": 3,
        "policy": {
            "max_hamming": 0.12 if args.mode == "block" else 0.10,
            "min_confidence": 55,
            "min_confidence_gap": 8,
        },
        "inputs": [
            {
                "path": str(original_path),
                "user_id": args.user_id,
                "captures": [
                    {
                        "name": "replace_with_real_capture",
                        "path": str(placeholder_capture),
                        "metadata": metadata,
                    }
                ],
            }
        ],
        "conditions": [],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "original": str(original_path),
        "watermarked": str(watermarked_path),
        "placeholder_capture": str(placeholder_capture),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a NIRGuard real-capture kit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--strength", type=int, default=4)
    parser.add_argument("--mode", choices=["pixel", "block"], default="block")
    parser.add_argument("--use-sync", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    create(parse_args())

