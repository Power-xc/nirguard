# watermark 패키지 — 패턴 생성·삽입·복원·판정 코어 파이프라인
from .generator import generate_pattern
from .embedder import embed_image, embed_video
from .extractor import extract_from_frames, extract_from_image, extract_from_video
from .metrics import calculate_psnr, calculate_ber, print_metrics_report
from .confidence import (
    AttributionDecision,
    ConfidencePolicy,
    MatchCandidate,
    confidence_from_hamming,
    decide_attribution,
    rank_patterns,
)
from .alignment import apply_sync_template, sync_score
from .block import embed_block_image, extract_block_from_frames, extract_block_from_image
from .degradation import apply_degradation
from .ecc import (
    deinterleave_bits,
    interleave_bits,
    repetition_decode_bits,
    repetition_encode_bits,
)
from .preprocess import apply_glare_mask, detect_screen_quad, glare_mask, warp_perspective

__all__ = [
    "generate_pattern",
    "embed_image",
    "embed_video",
    "extract_from_image",
    "extract_from_frames",
    "extract_from_video",
    "calculate_psnr",
    "calculate_ber",
    "print_metrics_report",
    "AttributionDecision",
    "ConfidencePolicy",
    "MatchCandidate",
    "confidence_from_hamming",
    "decide_attribution",
    "rank_patterns",
    "apply_sync_template",
    "sync_score",
    "apply_degradation",
    "embed_block_image",
    "extract_block_from_frames",
    "extract_block_from_image",
    "deinterleave_bits",
    "interleave_bits",
    "repetition_decode_bits",
    "repetition_encode_bits",
    "apply_glare_mask",
    "detect_screen_quad",
    "glare_mask",
    "warp_perspective",
]
