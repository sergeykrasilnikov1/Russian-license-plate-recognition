# Dataset: Non-standard Russian GRZ (types 1 / 1A / 1B)

## Purpose

Training and evaluation data for the AIS Gorod semifinal task:
recognition of Russian license plates of types **type1**, **type1a** (square two-line),
**type1b** (yellow taxi/bus), and negative **other** examples.

## License

This dataset is published under **CC BY 4.0** (see `LICENSE`).

## Contents

| Path | Description |
|------|-------------|
| `images/real/` | Real photos from open licensed sources (auto-scraped) |
| `images/synthetic/` | Procedural plates (GOST geometry) |
| `labels/` | One YOLO-style `.txt` per image |
| `meta.csv` | One row per plate (`;`-separated) |
| `generator/` | Synthetic generator package + requirements |

## Collection method (honest disclosure)

- **Real images**: downloaded programmatically from open datasets / public APIs
  (Hugging Face, Roboflow Universe, PlatesMania XML export, etc.). No CAPTCHA bypass.
- **Primary labels**: produced automatically (detector + OCR) and **programmatically
  validated** against the GOST plate mask and allowed alphabet. This is **not**
  human-verified ground truth.
- **Faces**: blurred automatically (MediaPipe) before inclusion.
- **Debug set**: organizer debug images are excluded via perceptual hash.

## Class targets (recommended minima)

| Group | Images | Unique plates |
|-------|--------|---------------|
| type1a | ≥ 150 | ≥ 300 |
| type1b | ≥ 50 | ≥ 50 |
| other | ≥ 100 | — |
| synthetic (all) | ≥ 5000 | — |

If real type1a/type1b volume is insufficient, synthetic share for those classes is increased (Stage 3).

## Statistics

_Filled after Stages 2–3._

## Sources

_Filled after Stage 2 scrape; each row in `meta.csv` has `source` and `license`._
