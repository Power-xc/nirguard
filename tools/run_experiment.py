#!/usr/bin/env python3
"""
Run one NIRGuard validation experiment and write a JSON result.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.common import (  # noqa: E402
    add_policy_args,
    candidate_patterns,
    new_run_id,
    policy_from_args,
    read_image,
    to_dict,
    write_image,
    write_result,
)
from watermark import (  # noqa: E402
    apply_degradation,
    apply_glare_mask,
    apply_sync_template,
    calculate_ber,
    calculate_psnr,
    decide_attribution,
    detect_screen_quad,
    embed_block_image,
    embed_image,
    extract_block_from_image,
    extract_from_image,
    generate_pattern,
    rank_patterns,
    sync_score,
    warp_perspective,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    original = read_image(input_path)

    pattern = generate_pattern(args.user_id)
    match_pattern = apply_sync_template(pattern) if args.use_sync else pattern
    if args.mode == "block":
        watermarked = embed_block_image(
            original,
            pattern,
            strength=args.strength,
            use_sync=args.use_sync,
        )
    else:
        watermarked = embed_image(original, pattern, strength=args.strength)
    psnr = calculate_psnr(original, watermarked)

    if args.suspect_image:
        suspect_path = Path(args.suspect_image)
        suspect = read_image(suspect_path)
        capture_type = "real_or_manual"
    else:
        suspect_path = None
        suspect = apply_degradation(
            watermarked,
            jpeg_quality=args.jpeg_quality,
            noise_sigma=args.noise_sigma,
            downscale=args.downscale,
        )
        capture_type = "synthetic"

    if args.glare_mask:
        suspect = apply_glare_mask(suspect)

    if args.auto_warp:
        quad = detect_screen_quad(suspect)
        if quad is not None:
            h, w = original.shape[:2]
            suspect = warp_perspective(suspect, quad, (w, h))

    extracted = extract_block_from_image(suspect) if args.mode == "block" else extract_from_image(suspect)
    ber = calculate_ber(match_pattern, extracted)

    candidates = rank_patterns(
        extracted,
        candidate_patterns(args.user_id, args.candidate_count, args.use_sync),
        top_k=args.top_k,
    )
    decision = decide_attribution(candidates, policy_from_args(args))

    run_id = new_run_id()
    out_dir = Path(args.out_dir)
    metadata = {}
    metadata_path = getattr(args, "metadata_json", None)
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    image_outputs = {}
    if args.save_images:
        watermarked_path = out_dir / f"{run_id}_watermarked.png"
        suspect_out_path = out_dir / f"{run_id}_suspect.png"
        write_image(watermarked_path, watermarked)
        write_image(suspect_out_path, suspect)
        image_outputs = {
            "watermarked": str(watermarked_path),
            "suspect": str(suspect_out_path),
        }

    result = {
        "run_id": run_id,
        "input_path": str(input_path),
        "suspect_path": str(suspect_path) if suspect_path else None,
        "user_id": args.user_id,
        "capture_type": capture_type,
        "condition": {
            "name": args.condition_name,
            "strength": args.strength,
            "jpeg_quality": args.jpeg_quality,
            "noise_sigma": args.noise_sigma,
            "downscale": args.downscale,
            "mode": args.mode,
            "use_sync": args.use_sync,
            "auto_warp": args.auto_warp,
            "glare_mask": args.glare_mask,
        },
        "capture_metadata": metadata,
        "metrics": {
            "psnr": psnr,
            "ber": ber,
            "top1_success": bool(candidates and candidates[0].user_id == args.user_id),
            "top3_success": any(candidate.user_id == args.user_id for candidate in candidates),
            "identified": decision.identified,
            "abstained": decision.abstained,
            "reason": decision.reason,
            "sync_score": sync_score(extracted) if args.use_sync else None,
        },
        "best": to_dict(decision.best),
        "candidates": [to_dict(candidate) for candidate in candidates],
        "outputs": image_outputs,
    }

    write_result(out_dir, run_id, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one NIRGuard validation experiment")
    parser.add_argument("--input", required=True, help="Original image path")
    parser.add_argument("--user-id", required=True, help="Watermarked user/session ID")
    parser.add_argument("--suspect-image", help="Real recapture or manually degraded image path")
    parser.add_argument("--out-dir", default="experiments/results", help="Result directory")
    parser.add_argument("--condition-name", default="default", help="Human-readable condition name")
    parser.add_argument("--strength", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--noise-sigma", type=float, default=10.0)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--mode", choices=["pixel", "block"], default="pixel")
    parser.add_argument("--use-sync", action="store_true")
    parser.add_argument("--auto-warp", action="store_true")
    parser.add_argument("--glare-mask", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    add_policy_args(parser)
    parser.add_argument("--metadata-json", help="Optional capture metadata JSON path")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
