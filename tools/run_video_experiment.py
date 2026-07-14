#!/usr/bin/env python3
"""
Run one multi-frame NIRGuard validation experiment.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.common import (  # noqa: E402
    add_policy_args,
    candidate_patterns,
    new_run_id,
    policy_from_args,
    to_dict,
    write_result,
)
from watermark import (  # noqa: E402
    apply_degradation,
    apply_sync_template,
    calculate_ber,
    calculate_psnr,
    decide_attribution,
    embed_block_image,
    embed_image,
    embed_video,
    extract_block_from_frames,
    extract_from_frames,
    extract_from_video,
    generate_pattern,
    rank_patterns,
)


def first_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read first frame: {path}")
    return frame


def degrade_video(
    input_path: Path,
    output_path: Path,
    jpeg_quality: int,
    noise_sigma: float,
    downscale: float,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(apply_degradation(frame, jpeg_quality, noise_sigma, downscale))
        count += 1

    cap.release()
    writer.release()
    return {"frame_count": count, "fps": fps, "width": width, "height": height}


def read_frames(path: Path, limit: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Could not read frames: {path}")
    return frames


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    pattern = generate_pattern(args.user_id)
    match_pattern = apply_sync_template(pattern) if args.use_sync else pattern

    if args.suspect_video:
        suspect_path = Path(args.suspect_video)
        capture_type = "real_or_manual_video"
        psnr = float("inf")
        extracted = extract_from_video(suspect_path, n_frames=args.n_frames)
        embed_info = {}
        degrade_info = {}
        watermarked_path = None
    else:
        original_frames = read_frames(input_path, args.n_frames)
        if args.mode == "block":
            watermarked_frames = [
                embed_block_image(frame, pattern, strength=args.strength, use_sync=args.use_sync)
                for frame in original_frames
            ]
        else:
            watermarked_frames = [
                embed_image(frame, pattern, strength=args.strength)
                for frame in original_frames
            ]
        degraded_frames = [
            apply_degradation(frame, args.jpeg_quality, args.noise_sigma, args.downscale)
            for frame in watermarked_frames
        ]
        psnr = calculate_psnr(original_frames[0], watermarked_frames[0])
        extracted = (
            extract_block_from_frames(degraded_frames)
            if args.mode == "block"
            else extract_from_frames(degraded_frames)
        )
        suspect_path = None
        capture_type = "synthetic_video"
        embed_info = {"frame_count": len(watermarked_frames), "fps": None}
        degrade_info = {"frame_count": len(degraded_frames)}
        watermarked_path = None

        if args.save_video:
            watermarked_path = out_dir / f"{run_id}_watermarked.mp4"
            suspect_video_path = out_dir / f"{run_id}_suspect.mp4"
            embed_info = embed_video(input_path, watermarked_path, pattern, strength=args.strength)
            degrade_info = degrade_video(
                watermarked_path,
                suspect_video_path,
                jpeg_quality=args.jpeg_quality,
                noise_sigma=args.noise_sigma,
                downscale=args.downscale,
            )

    ber = calculate_ber(match_pattern, extracted)
    candidates = rank_patterns(
        extracted,
        candidate_patterns(args.user_id, args.candidate_count, args.use_sync),
        top_k=args.top_k,
    )
    decision = decide_attribution(candidates, policy_from_args(args))

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
            "n_frames": args.n_frames,
            "mode": args.mode,
            "use_sync": args.use_sync,
        },
        "metrics": {
            "psnr": psnr,
            "ber": ber,
            "top1_success": bool(candidates and candidates[0].user_id == args.user_id),
            "top3_success": any(candidate.user_id == args.user_id for candidate in candidates),
            "identified": decision.identified,
            "abstained": decision.abstained,
            "reason": decision.reason,
        },
        "best": to_dict(decision.best),
        "candidates": [to_dict(candidate) for candidate in candidates],
        "video": {
            "embed": embed_info,
            "degrade": degrade_info,
            "watermarked_path": str(watermarked_path) if watermarked_path else None,
        },
    }

    write_result(out_dir, run_id, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one NIRGuard multi-frame validation experiment")
    parser.add_argument("--input", required=True, help="Original video path")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--suspect-video", help="Real recapture or manually degraded video path")
    parser.add_argument("--out-dir", default="experiments/results/video")
    parser.add_argument("--condition-name", default="video_default")
    parser.add_argument("--strength", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--noise-sigma", type=float, default=5.0)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--mode", choices=["pixel", "block"], default="pixel")
    parser.add_argument("--use-sync", action="store_true")
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    add_policy_args(parser)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
