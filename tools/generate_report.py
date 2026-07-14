#!/usr/bin/env python3
"""
Generate a minimal PDF evidence report from a validation result JSON.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def flatten_result(result: dict[str, Any]) -> list[str]:
    metrics = result.get("metrics", {})
    best = result.get("best", {})
    condition = result.get("condition", {})
    lines = [
        "NIRGuard Evidence Report",
        "",
        f"Run ID: {result.get('run_id', '')}",
        f"User ID: {result.get('user_id', '')}",
        f"Capture Type: {result.get('capture_type', '')}",
        f"Input: {result.get('input_path', '')}",
        f"Suspect: {result.get('suspect_path', '')}",
        "",
        "Condition",
        f"Name: {condition.get('name', '')}",
        f"Strength: {condition.get('strength', '')}",
        f"JPEG Quality: {condition.get('jpeg_quality', '')}",
        f"Noise Sigma: {condition.get('noise_sigma', '')}",
        f"Downscale: {condition.get('downscale', '')}",
        f"Frames: {condition.get('n_frames', 'n/a')}",
        "",
        "Decision",
        f"Identified: {metrics.get('identified', '')}",
        f"Abstained: {metrics.get('abstained', '')}",
        f"Reason: {metrics.get('reason', '')}",
        f"BER: {metrics.get('ber', '')}",
        f"PSNR: {metrics.get('psnr', '')}",
        f"Top-1 Success: {metrics.get('top1_success', '')}",
        f"Top-3 Success: {metrics.get('top3_success', '')}",
        "",
        "Best Candidate",
        f"User ID: {best.get('user_id', '')}",
        f"Hamming: {best.get('hamming', '')}",
        f"Confidence: {best.get('confidence', '')}",
        "",
        "Top Candidates",
    ]
    for index, candidate in enumerate(result.get("candidates", []), start=1):
        lines.append(
            f"{index}. {candidate.get('user_id', '')} | "
            f"hamming={candidate.get('hamming', '')} | "
            f"confidence={candidate.get('confidence', '')}"
        )
    lines.extend([
        "",
        "Validation Notice",
        "This report is a technical validation artifact.",
        "It is not a standalone legal accusation.",
    ])
    return lines


def build_pdf(lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 12 Tf", "50 770 Td", "14 TL"]
    for line in lines[:48]:
        content_lines.append(f"({escape_pdf_text(line)}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    content = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def generate(result_path: Path, output_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_pdf(flatten_result(result)))
    print(f"Saved PDF report: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDF evidence report")
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(Path(args.result), Path(args.output))

