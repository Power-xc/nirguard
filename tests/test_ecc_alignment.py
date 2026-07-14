import numpy as np
import pytest

from watermark.alignment import apply_sync_template, sync_score
from watermark.ecc import (
    deinterleave_bits,
    interleave_bits,
    repetition_decode_bits,
    repetition_encode_bits,
)


def test_repetition_ecc_recovers_single_error_per_group() -> None:
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    encoded = repetition_encode_bits(bits, repeats=3)
    encoded[1] = 1
    encoded[4] = 0

    decoded = repetition_decode_bits(encoded, repeats=3)

    np.testing.assert_array_equal(decoded, bits)


def test_repetition_ecc_rejects_even_repeats() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        repetition_encode_bits(np.array([0, 1], dtype=np.uint8), repeats=2)


def test_interleave_roundtrip() -> None:
    bits = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)

    interleaved = interleave_bits(bits, seed=42)
    restored = deinterleave_bits(interleaved, seed=42)

    np.testing.assert_array_equal(restored, bits)


def test_sync_template_scores_clean_pattern() -> None:
    pattern = np.zeros((64, 64), dtype=np.uint8)
    marked = apply_sync_template(pattern, marker_size=8)

    assert sync_score(marked, marker_size=8) == 1.0


def test_sync_template_rejects_tiny_pattern() -> None:
    with pytest.raises(ValueError, match="too small"):
        apply_sync_template(np.zeros((8, 8), dtype=np.uint8), marker_size=8)

