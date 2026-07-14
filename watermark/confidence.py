"""
Confidence and abstain policy for NIRGuard candidate attribution.

The matcher can always return the closest pattern. This module decides whether
the closest pattern is strong enough to name a candidate.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MatchCandidate:
    user_id: str
    hamming: float
    confidence: float


@dataclass(frozen=True)
class AttributionDecision:
    identified: bool
    abstained: bool
    reason: str
    best: MatchCandidate
    candidates: list[MatchCandidate]


@dataclass(frozen=True)
class ConfidencePolicy:
    max_hamming: float = 0.10
    min_confidence: float = 60.0
    min_confidence_gap: float = 8.0


def confidence_from_hamming(hamming: float) -> float:
    score = max(0.0, 1.0 - 2.0 * hamming) * 100.0
    return round(score, 1)


def hamming_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Pattern shape mismatch: left={left.shape}, right={right.shape}")

    errors = int(np.sum(left.astype(np.uint8) ^ right.astype(np.uint8)))
    return round(errors / left.size, 4)


def rank_patterns(
    recovered: np.ndarray,
    stored_patterns: Sequence[tuple[str, np.ndarray]],
    top_k: int = 3,
) -> list[MatchCandidate]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive: {top_k}")

    candidates: list[MatchCandidate] = []
    for user_id, pattern in stored_patterns:
        hamming = hamming_distance(recovered, pattern)
        candidates.append(
            MatchCandidate(
                user_id=user_id,
                hamming=hamming,
                confidence=confidence_from_hamming(hamming),
            )
        )

    candidates.sort(key=lambda c: c.hamming)
    return candidates[:top_k]


def decide_attribution(
    candidates: Sequence[MatchCandidate],
    policy: Optional[ConfidencePolicy] = None,
) -> AttributionDecision:
    if not candidates:
        raise ValueError("At least one candidate is required")

    policy = policy or ConfidencePolicy()
    ordered = sorted(candidates, key=lambda c: c.hamming)
    best = ordered[0]

    if best.hamming > policy.max_hamming:
        return AttributionDecision(
            identified=False,
            abstained=True,
            reason="best_hamming_above_threshold",
            best=best,
            candidates=list(ordered),
        )

    if best.confidence < policy.min_confidence:
        return AttributionDecision(
            identified=False,
            abstained=True,
            reason="best_confidence_below_threshold",
            best=best,
            candidates=list(ordered),
        )

    if len(ordered) > 1:
        gap = best.confidence - ordered[1].confidence
        if gap < policy.min_confidence_gap:
            return AttributionDecision(
                identified=False,
                abstained=True,
                reason="candidate_gap_too_small",
                best=best,
                candidates=list(ordered),
            )

    return AttributionDecision(
        identified=True,
        abstained=False,
        reason="accepted",
        best=best,
        candidates=list(ordered),
    )
