# NIRGuard Validation Plan

## Goal

Build evidence before product claims.
The first system is NIRGuard Validation Lab: a toolchain that records where watermark recovery works, fails, or must abstain.

## Initial Scope

Target use case:

- secure review screens
- confidential document/image review
- short pre-release media review

Not initial scope:

- full OTT anti-piracy
- live sports watermarking
- broad consumer-device guarantees

## Capture Matrix

Each validation sample should record:

- display model and panel type
- camera model
- distance
- angle
- ambient brightness
- screen brightness
- content type
- file format
- compression level
- crop/resize/downscale
- capture length for video

## Required Metrics

- PSNR after embed
- BER after extraction
- Top-1 success
- Top-3 success
- confidence score
- abstain decision
- false positive rate
- false negative rate

## First 90 Days

1. Correct public claims in README and SPEC.
2. Add experiment CLI and JSON result logging.
3. Build synthetic degradation baseline.
4. Collect real display-camera samples.
5. Quantify current Red +/-4 limitation.
6. Define thresholds for confidence and abstain.
7. Run manifest batches for each device/capture matrix.
8. Generate PDF evidence reports from representative cases.
9. Compare pixel mode against block mode before making any external claim.

## Six-Month Milestone

NIRGuard can be called a pilot candidate only after it has:

- reproducible results on named hardware conditions
- clear failure and abstain behavior
- Top-3 candidate reports
- false accusation controls
- evidence report generation path
