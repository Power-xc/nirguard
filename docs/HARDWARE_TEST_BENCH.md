# Hardware Test Bench

## Purpose

NIRGuard claims must be tied to named hardware conditions. All robustness claims so far come from digital simulation; this bench is what turns them into (or refutes them as) physical claims.

Ordering constraint: the spectral premise check comes first. Whether a commercial display emits any usable near-infrared signal at all — and whether an R-channel adjustment modulates it above the noise floor — is an open empirical question, and published display-spectrum literature suggests the answer is likely no. Until the sweep below says otherwise, NIRGuard makes no NIR emission claims; the working model is visible-light red-channel watermarking.

## Spectral Premise Check (run before anything else)

- spectroradiometer measurement of each target panel, 400–1000 nm
- sweep R channel 0 → 255; record emission delta in the 700–850 nm band
- pass rule: R ±4 must produce a change above the measurement noise floor (SNR > 3) to justify any NIR-based capture path
- fail rule: below noise floor → drop the NIR capture path from the matrix and validate the visible-light path only

## Recommended Equipment

Core capture:

- monochrome CMOS camera with 850 nm or 940 nm sensitivity
- global shutter when available
- 30 fps or higher
- IR-pass filter such as 850 nm long-pass
- fixed tripod or rail mount
- calibrated monitor stand

Comparison capture:

- iPhone camera
- Android camera
- DSLR or mirrorless camera if available

Displays:

- laptop LCD
- external IPS monitor
- OLED monitor or TV
- QLED/mini-LED monitor or TV if available

Environment:

- controllable room lighting
- optional IR illuminator
- glare control using hooding or polarizing setup
- raw or minimally compressed file transfer path

## Minimum Matrix

Initial matrix:

- distance: 30 cm, 50 cm, 100 cm
- angle: 0, 15, 30 degrees
- lighting: bright, dim
- compression: source, JPEG Q80, JPEG Q70
- scale: 1.0, 0.75, 0.5
- frames: 1, 15, 30 for video tests

The full matrix can reach 270 samples. Start with 50 samples for smoke validation before expanding.

## Success Criteria

The first hardware milestone is not universal attribution. It is a reproducible table that shows:

- where attribution works
- where it abstains
- where it fails
- whether block mode improves over pixel mode
- whether the spectral premise check passes, and if it does, whether NIR-band capture adds any usable signal over smartphone capture

