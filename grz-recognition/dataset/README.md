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
| `AY000554/Car_plate_detecting_dataset` (Hugging Face) | official `huggingface_hub` SDK, `val.zip` | CC BY 4.0 (from AUTO.RIA Numberplate) | 1181 images / 1239 plates, **bboxes from the source dataset** |
| `tatmantech/russian-license-plates-ec7zg` v2 (Roboflow Universe) | official `roboflow` SDK, YOLOv8 export | CC BY 4.0 (reported by the Universe API) | 787 images / 982 plates, **331 plates with source-supplied text** |
| `carplates/russian-plate-kuabh` v1 (Roboflow Universe) | official `roboflow` SDK, YOLOv8 export | CC BY 4.0 | 108 images / 108 plates |
| `andrewmvd/car-plate-detection` (Kaggle) | documented REST API v1, Bearer token | CC0-1.0 | 241 images / 265 plates, all **`other`** (foreign plates) |
| Wikimedia Commons | public MediaWiki API (`list=search`, `list=categorymembers`, `prop=imageinfo`) | per-file, from `extmetadata` | 194 staged candidates, unlabeled |

Roboflow exports arrive pre-augmented by the source (flips, crops, rotations),
so some images are mirrored copies of the same original. They are kept for
detection only; mirrored plates never receive text (`plate_num = ########`).

### Sources deliberately excluded

| Source | Reason |
|--------|--------|
| `evgrafovmaxim/nomeroff-russian-license-plates` (Kaggle) | LGPL-3.0 copyleft cannot be relicensed into this CC BY 4.0 dataset |
| `adilshamim8/license-plate-recognition` (Kaggle) | 10 125 **Vietnamese** plates, zero annotation files |
| `egorandreasyan/car-number-segment` (Kaggle) | Russian, but already cropped to the plate (38–145 px per side): trivial boxes, fails the resolution gate |
| `plate-tsusp/russian-plate` v3 (Roboflow) | byte-identical re-upload of `carplates/russian-plate-kuabh`; all 262 images rejected by the deduper |
| `symbolplate/russian-license-plate-characters` (Roboflow) | per-character boxes, not plate boxes — reserved for Stage 5 OCR |
| PlatesMania | documented XML endpoints return HTTP 403/404; bypassing is forbidden, so the source is skipped |
| Openverse | anonymous daily quota exhausted (HTTP 401/429); supports `OPENVERSE_API_TOKEN` |

### Labeling provenance

- **bbox / quad**: taken from the source dataset's own YOLO annotations. Nothing
  in `meta.csv` was localized by our own model.
- **plate_type**: when a source documents a single plate type, that declared
  type is authoritative. The geometric classifier (background color + width/
  height ratio) is only used for sources without a declared type, because
  distant or strongly angled type1 plates lose width and would otherwise be
  misfiled as type1a. One exception: a decisively yellow crop (≥ 50 % yellow
  pixels) inside a **Russian** type1/type1a source is relabeled `type1b`, since
  background color survives perspective while aspect ratio does not. The
  override is not applied to `other` sources, where a yellow plate is a foreign
  plate rather than a Russian taxi one.
- **plate_num**: `########` for all but 331 plates. Those 331 come from export
  filenames of `tatmantech/russian-license-plates-ec7zg`, which encode the
  plate number (116 distinct numbers); each was normalized and re-validated
  against the GOST mask, and is used only when the image holds exactly one
  annotated plate. This is source metadata, not recognition output. Remaining
  text is filled by the Stage 5 OCR pass, again gated by the mask validator.
  Filenames from that source matching the diplomatic series pattern
  (`002CD178`) mark the plate as `other` instead of supplying text.
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
| type1 | 3419 | 3138 | 1243 |
| type1a | 1922 | 1922 | 1922 |
| type1b | 1588 | 1587 | 1553 |
| other | 665 | 639 | 398 |
| **total** | **7594** | **7281** | — |
| of which real | 2594 | 2281 | 116 |
| of which synthetic | 5000 | 5000 | 5000 |
| style_b type1a (val holdout) | 150 | 150 | 150 |

After Stage 4 split prep (`prepare_detector_split.py --build-style-b 150`):
**7744** meta rows / **7431** images (includes style_b). style_b never enters train.

Licenses of labeled rows: CC-BY-4.0 — 7479, CC0-1.0 — 265.
Synthetic rows are all `CC-BY-4.0` / `is_synthetic=1`.
style_b rows: `source=synthetic_generator_style_b`.

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

| Group | Recommended real minimum | Real (labeled) | Synthetic | Coverage |
|-------|--------------------------|----------------|-----------|----------|
| type1a | 150 images / 300 plates | 0 | 1922 | synthetic closed the Stage 2 deficit |
| type1b | 50 images / 50 plates | 34 images / 35 plates | 1553 | real shortfall covered by synthetic |
| other | 100 images | 241 images / 267 plates | 398 | above recommendation |
| type1 | — | 2011 images / 2292 plates | 1127 | abundant |

Synthetic mix (`--seed 42`): `type1=0.22, type1a=0.38, type1b=0.32, other=0.08`.
Reproduction: `python scripts/build_synthetic.py --count 5000 --seed 42`.


## Reproduction

```bash
pip install -r ../requirements-dev.txt

# Credentials are read from the environment only, never stored in the repo
export ROBOFLOW_API_KEY=...            # Roboflow Universe
export KAGGLE_API_TOKEN=...            # or ~/.kaggle/access_token

python ../scripts/scrape_real_data.py --probe-only
python ../scripts/scrape_real_data.py --sources huggingface --limit 1200
python ../scripts/scrape_real_data.py --sources roboflow --limit 1700
python ../scripts/scrape_real_data.py --sources kaggle --limit 440
python ../scripts/scrape_real_data.py --sources commons --plate-types type1a type1b other --limit 200
python ../scripts/validate_dataset.py --dataset .
```
