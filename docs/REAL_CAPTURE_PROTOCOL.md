# Real Capture Validation Protocol

## Goal

Measure whether NIRGuard can attribute a leak from real off-screen captures, and when it must abstain.

## Setup

1. Pick one original asset.
2. Embed one known `user_id`.
3. Display the watermarked asset full-screen.
4. Capture the screen with a phone or camera.
5. Save the capture without manual enhancement.
6. Record metadata using `capture_metadata.schema.json`.
7. Run `tools/run_manifest.py`.
8. Generate an evidence PDF from the result JSON.

## Capture Rules

- Keep the original and recapture files.
- Do not crop unless the crop is recorded.
- Do not manually sharpen or color-correct the capture.
- Record failures as first-class results.
- Use abstain as a success path when confidence is weak.

## First Validation Batch

Run at least 20 captures:

- 5 center captures at 100 cm
- 5 angled captures at 20 degrees
- 5 angled captures at 40 degrees
- 5 low-brightness captures

## Report

Each batch should include:

- manifest file
- all result JSON files
- aggregate analysis JSON
- one PDF evidence report for at least one successful or abstained case

