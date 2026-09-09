# Dataset: Non-standard Russian GRZ (types 1 / 1A / 1B)

## Purpose

Training and evaluation data for the AIS Gorod semifinal task: recognition of
Russian license plates of types **type1** (white, single-line), **type1a**
(white, square two-line), **type1b** (yellow, passenger transport), plus
negative **other** examples that must be filtered out.

## License

This dataset is published under **CC BY 4.0** (see `LICENSE`). Every row in
`meta.csv` additionally records the license of its own source, so mixed-license
provenance stays auditable.

Admission policy (`src/data_collection/licensing.py`):

- **Rejected**: unknown / all-rights-reserved licenses, and any **NoDerivatives**
  license — crops and augmentations are derivative works.
- **Accepted** (default `noncommercial` policy): CC0/Public Domain, CC BY,
  CC BY-SA, CC BY-NC, CC BY-NC-SA, Apache-2.0, MIT.
- A stricter `--license-policy strict` mode limits admission to CC0/PD/CC BY/
  Apache/MIT, i.e. only licenses that redistribute cleanly under CC BY 4.0.

## Contents

| Path | Description |
|------|-------------|
| `images/real/` | Real photos, fully labeled (one `meta.csv` row per plate) |
| `images/synthetic/` | Procedurally generated plates (Stage 3) |
| `labels/` | One YOLO-format `.txt` per labeled image |
| `meta.csv` | One row per plate, `;`-separated, ten fixed columns |
| `raw_downloads/` | Staging pool: filtered candidates awaiting Stage 4 labeling |
| `raw_downloads/manifest.csv` | Provenance/license of every staged candidate |
| `generator/` | Synthetic generator package + requirements |
| `validation_report.txt` | Output of `scripts/validate_dataset.py` |

`meta.csv` contains **only** fully labeled plates, so the dataset passes the
organizers' validator at every point in time. Candidates without a bbox live in
`raw_downloads/` and are promoted after the Stage 4 detector labels them.

## Collection method (honest disclosure)

All collection is automatic (`scripts/scrape_real_data.py`), CPU-only, and uses
official APIs/SDKs. No CAPTCHA, paywall or rate-limit circumvention: the
collector sends an honest `User-Agent` (`grz-dataset-collector/...`), honours
`robots.txt` and `Retry-After`, and stops a source when it is throttled.

### Sources actually used

| Source | Access | License | Result |
|--------|--------|---------|--------|
| `AY000554/Car_plate_detecting_dataset` (Hugging Face) | official `huggingface_hub` SDK, `val.zip` | CC BY 4.0 (from AUTO.RIA Numberplate) | 887 images / 931 plates, **bboxes from the source dataset** |
| Wikimedia Commons | public MediaWiki API (`list=search`, `list=categorymembers`, `prop=imageinfo`) | per-file, from `extmetadata` | 194 staged candidates, unlabeled |

### Sources implemented but unavailable in this environment

| Source | Reason |
|--------|--------|
| Roboflow Universe | no `ROBOFLOW_API_KEY` |
| Kaggle (MADE CV 2021) | no `~/.kaggle/kaggle.json` |
| Openverse | anonymous daily quota exhausted (HTTP 401/429); supports `OPENVERSE_API_TOKEN` |
| PlatesMania | documented XML endpoints return HTTP 403/404; bypassing is forbidden, so the source is skipped |

### Labeling provenance

- **bbox / quad**: taken from the source dataset's own YOLO annotations. Nothing
  in `meta.csv` was localized by our own model.
- **plate_type**: when a source documents a single plate type (as the Hugging
  Face dataset does), that declared type is authoritative. The geometric
  classifier (background color + width/height ratio) is only used for sources
  without a declared type, because distant or strongly angled type1 plates lose
  width and would otherwise be misfiled as type1a.
- **plate_num**: **not yet recognized** — every row carries `########`. Plate
  text will be filled by the Stage 5 OCR pass and accepted only after passing
  the GOST mask validator; unreadable positions stay `#`.
- **is_vehicle**: `1` for source datasets that consist of car photographs.
- **conditions**: inferred from pixel statistics (brightness → `day`/`night`,
  clipped highlights → `glare`, low Laplacian variance → `motion_blur`).

This is automatic labeling with programmatic validation, **not** human-verified
ground truth, and it is not presented as such.

### Privacy and dedup filters

| Filter | Implementation |
|--------|----------------|
| Face blur | MediaPipe BlazeFace (`weights/face/blaze_face_short_range.tflite`), OpenCV Haar cascade as offline fallback |
| Deduplication | perceptual hash (`imagehash.phash`, Hamming ≤ 4), within the set and across runs |
| Debug-set exclusion | `--debug-set` hashes the organizers' 30 images into a blocklist |
| Quality gate | minimum side 240 px, Laplacian variance ≥ 20 |

## Statistics

### Labeled data in `meta.csv`

| Group | Plates | Images | Unique plate numbers |
|-------|--------|--------|----------------------|
| type1 | 931 | 887 | 0 (text pending Stage 5) |
| type1a | 0 | 0 | 0 |
| type1b | 0 | 0 | 0 |
| other | 0 | 0 | 0 |
| synthetic | 0 | 0 | — (Stage 3) |

### Staging pool `raw_downloads/manifest.csv`

| Query hint | Candidates |
|------------|------------|
| type1b | 102 |
| type1a | 55 |
| type1 | 37 |
| **total** | **194** |

The hint is the search/category that produced the image, not a verified label.
A manual inspection of 27 sampled candidates found roughly 15–20 % containing a
plate of the intended group at readable size; the rest are street scenes where
the plate is absent or too small. Stage 4 filters this pool with the trained
detector.

## Real vs synthetic balance per class

| Group | Recommended real minimum | Real (labeled) | Deficit | Coverage plan |
|-------|--------------------------|----------------|---------|---------------|
| type1a | 150 images / 300 plates | 0 | full | synthetic (Stage 3, increased share) + Stage 4 filtering of the staging pool |
| type1b | 50 images / 50 plates | 0 | full | synthetic (Stage 3, increased share) + Stage 4 filtering of the staging pool |
| other | 100 images | 0 | full | synthetic negatives + Stage 4 filtering |
| type1 | — | 887 images / 931 plates | none | — |

The long-tail groups could not be filled from the open sources reachable here:
the only richly annotated public dataset covers type1 only, and the sources that
do contain squares and yellow plates are either credential-gated or throttled.
Per the task fallback, the deficit is covered by generated data in Stage 3
rather than by manual photography, and the final real/synthetic ratio per class
is reported here once Stage 3 completes.

## Reproduction

```bash
pip install -r ../requirements-dev.txt

python ../scripts/scrape_real_data.py --probe-only
python ../scripts/scrape_real_data.py --sources huggingface --hf-archives val.zip --limit 900
python ../scripts/scrape_real_data.py --sources commons --plate-types type1a type1b other --limit 200
python ../scripts/validate_dataset.py --dataset .
```
