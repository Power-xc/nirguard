from tools.threshold_sweep import evaluate


def test_threshold_sweep_counts_false_positive() -> None:
    results = [
        {
            "user_id": "user_001",
            "best": {"user_id": "user_001", "confidence": 95.0, "hamming": 0.02},
        },
        {
            "user_id": "user_002",
            "best": {"user_id": "user_003", "confidence": 92.0, "hamming": 0.03},
        },
        {
            "user_id": "user_004",
            "best": {"user_id": "user_004", "confidence": 40.0, "hamming": 0.30},
        },
    ]

    summary = evaluate(results, confidence_threshold=80.0, max_hamming=0.05)

    assert summary["accepted"] == 2
    assert summary["true_positive"] == 1
    assert summary["false_positive"] == 1
    assert summary["abstained"] == 1

