# Capture Dataset Schema

## Purpose

Real capture validation must be reproducible. Every suspect image or video needs metadata that explains how it was captured.

## Directory Layout

```text
experiments/datasets/
  project_name/
    originals/
    watermarked/
    recaptures/
    metadata/
```

## Capture Metadata

Required fields:

```json
{
  "sample_id": "review_screen_001_iphone_1m_center",
  "original_path": "experiments/datasets/project/originals/review_screen_001.png",
  "suspect_path": "experiments/datasets/project/recaptures/review_screen_001_iphone_1m_center.png",
  "user_id": "user_003",
  "content_type": "image",
  "display": {
    "model": "TBD",
    "panel_type": "LCD",
    "resolution": "2560x1440",
    "brightness_percent": 80,
    "color_mode": "default"
  },
  "camera": {
    "model": "iPhone 15 Pro",
    "mode": "photo",
    "resolution": "4032x3024",
    "codec": "HEIC"
  },
  "capture": {
    "distance_cm": 100,
    "angle_degrees": 0,
    "ambient_lux": null,
    "tripod": false,
    "focus_locked": false,
    "exposure_locked": false
  },
  "postprocess": {
    "crop": false,
    "resize": null,
    "compression": null
  },
  "notes": ""
}
```

## Minimum Real-Capture Matrix

Start with:

- 2 display types: laptop LCD, external monitor
- 2 camera types: iPhone, Android
- 3 distances: 50 cm, 100 cm, 150 cm
- 3 angles: 0, 20, 40 degrees
- 3 brightness levels: 40%, 70%, 100%
- 3 content types: document, UI screenshot, image/video frame

## Acceptance Rule

Do not turn a scenario into an external claim until it has:

- at least 20 real captures
- Top-1, Top-3, abstain, and false-positive metrics
- saved result JSON files
- one generated evidence PDF

