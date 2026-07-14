import json
from pathlib import Path

from tools.generate_report import generate


def test_generate_pdf_report(tmp_path: Path) -> None:
    result = {
        "run_id": "test",
        "user_id": "user_003",
        "capture_type": "synthetic",
        "input_path": "input.png",
        "suspect_path": "suspect.png",
        "condition": {"name": "clean", "strength": 4},
        "metrics": {
            "identified": True,
            "abstained": False,
            "reason": "accepted",
            "ber": 0.01,
            "psnr": 40.0,
            "top1_success": True,
            "top3_success": True,
        },
        "best": {"user_id": "user_003", "hamming": 0.01, "confidence": 98.0},
        "candidates": [{"user_id": "user_003", "hamming": 0.01, "confidence": 98.0}],
    }
    result_path = tmp_path / "result.json"
    output_path = tmp_path / "report.pdf"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    generate(result_path, output_path)

    assert output_path.read_bytes().startswith(b"%PDF-1.4")

