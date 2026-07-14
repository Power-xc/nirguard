import numpy as np
import pytest

from watermark.confidence import (
    ConfidencePolicy,
    MatchCandidate,
    confidence_from_hamming,
    decide_attribution,
    hamming_distance,
    rank_patterns,
)


def test_confidence_from_hamming() -> None:
    assert confidence_from_hamming(0.0) == 100.0
    assert confidence_from_hamming(0.25) == 50.0
    assert confidence_from_hamming(0.5) == 0.0


def test_hamming_distance() -> None:
    left = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    right = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    assert hamming_distance(left, right) == 0.25


def test_hamming_distance_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Pattern shape mismatch"):
        hamming_distance(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((3, 3), dtype=np.uint8),
        )


def test_rank_patterns_returns_best_first() -> None:
    recovered = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    stored = [
        ("wrong", np.array([[1, 1], [0, 0]], dtype=np.uint8)),
        ("right", recovered.copy()),
    ]

    ranked = rank_patterns(recovered, stored, top_k=2)

    assert ranked[0].user_id == "right"
    assert ranked[0].confidence == 100.0


def test_decide_attribution_accepts_clear_best() -> None:
    decision = decide_attribution(
        [
            MatchCandidate("user_001", hamming=0.02, confidence=96.0),
            MatchCandidate("user_002", hamming=0.30, confidence=40.0),
        ]
    )

    assert decision.identified is True
    assert decision.abstained is False
    assert decision.reason == "accepted"


def test_decide_attribution_abstains_on_high_hamming() -> None:
    decision = decide_attribution(
        [MatchCandidate("user_001", hamming=0.25, confidence=50.0)]
    )

    assert decision.identified is False
    assert decision.abstained is True
    assert decision.reason == "best_hamming_above_threshold"


def test_decide_attribution_abstains_on_small_gap() -> None:
    decision = decide_attribution(
        [
            MatchCandidate("user_001", hamming=0.04, confidence=92.0),
            MatchCandidate("user_002", hamming=0.06, confidence=88.0),
        ],
        ConfidencePolicy(min_confidence_gap=8.0),
    )

    assert decision.identified is False
    assert decision.abstained is True
    assert decision.reason == "candidate_gap_too_small"

