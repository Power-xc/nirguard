# External Validation Package

## Purpose

This package is the material used for pilots, partner diligence, or investor technical review.

## Required Contents

- `README.md`
- `docs/CLAIMS_POLICY.md`
- `docs/VALIDATION_PLAN.md`
- `docs/HARDWARE_TEST_BENCH.md`
- `docs/CAPTURE_DATASET_SCHEMA.md`
- `docs/REAL_CAPTURE_PROTOCOL.md`
- `docs/LEGAL_EVIDENCE_PLAYBOOK.md`
- `docs/TAKEDOWN_WORKFLOW.md`
- manifest file
- raw result JSON files
- aggregate analysis JSON
- representative evidence PDFs
- threshold sweep JSON
- short demo script

## Required Tables

Each validation package should include:

| Condition | Samples | Top-1 | Top-3 | Abstain | Avg BER | False Positive |
|-----------|---------|-------|-------|---------|---------|----------------|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Demo Script

1. Open Secure Viewer.
2. Select a reviewer/session ID.
3. Embed a watermark into a review asset.
4. Upload a suspected leak capture.
5. Show Top-3 candidates, confidence, and abstain state.
6. Generate an evidence PDF.
7. Explain which capture conditions are validated and which are not.
