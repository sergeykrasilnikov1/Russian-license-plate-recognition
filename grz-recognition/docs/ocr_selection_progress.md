# OCR selection — measured decision, 2026-09-27

## Objective and constraints

Select or train the best plate OCR on held-out project data and verify the full
image → detection → vehicle filtering → warp → OCR → CSV pipeline. Target
type1/type1a/type1b separately, including real square and yellow plates. Do not
modify the main project environment; experiments use isolated tooling, Kaggle,
or already installed dependencies. Preserve existing uncommitted project changes.

## Evidence obtained

- Selected production engine: the plate-specific CRNN/CTC plus YOLO-pose corner
  detector from `Lin-Lini/volga-it-2026-lpr`, pinned at commit
  `cc1dee8bc59cdb71db3ccd587841c8ea487be37a`. English PP-OCRv5 mobile is retained
  only as the lightweight `--prefer-onnx` fallback.
- FastPlateOCR offers pretrained plate-specific CCT S/XS v2 models. Upstream
  recommends S v2, but this does not establish superiority on Russian plates.
  https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/
- Nomeroff Net includes Russian numberplate OCR data and specialized OCR, but no
  independently verified, drop-in checkpoint was established for this exact
  alphabet/preprocessing during this comparison; it was not scored.
  https://github.com/ria-com/nomeroff-net
- The reference repository has a plate-specific CRNN/CTC, bundled OCR weights
  and YOLO pose corners, with independent reading of two lines on square plates.
  Its source, weights, preprocessing and license were inspected, pinned, scored
  on the project corpus and integrated through an adapter.
  https://github.com/Lin-Lini/volga-it-2026-lpr
- The legacy fallback derives a quad from an axis-aligned box; the selected
  engine predicts four corners directly and avoids that measured bottleneck.
- Kaggle authentication was repaired without issuing a new token: Kaggle CLI
  2.2.4 uses the existing token via `~/.kaggle/access_token`. A private OCR
  comparison and reference-evaluation kernels completed successfully.

## Candidate comparison (GT quadrilaterals)

Leakage-aware validation used 1,219 readable crops. No exact SHA-256 overlap with
the OCR train split remained in this run. The set contains 69 real type1 plates,
but no readable real type1a/type1b labels, so rare real formats remain unproven.

- PP-OCRv4: 40.11% exact, 55.48% character accuracy.
- PP-OCRv5: 40.94% exact, 65.13% character accuracy.
- FastPlate CCT-S v2 global: 2.87% exact.
- FastPlate CCT-XS v2 global: 21.33% exact; best on the small real-type1 slice
  (43.48%), but unsuitable as one universal model.
- A validation-selected route (type1/type1b/other → v4, type1a → v5) reached
  45.28% on GT quads. It did not remain best in the deployed bbox pipeline.

Artifacts: `kaggle_results/ocr_compare_v4/ocr_comparison_summary.json` and
`ocr_candidate_predictions.csv`.

## Full-pipeline evaluation

`kaggle_pipeline_e2e/pipeline_e2e_kernel.py` was executed locally with the same
datasets and assets after Kaggle required separate consent to upload that exact
evaluation file. It excluded five OCR train/validation SHA-256 overlaps, found
no exact overlap with the YOLO26n Roboflow-v3 source, and evaluated 1,293 images,
1,344 objects and 1,219 readable ground truths.

- GT quad, routed OCR: **45.28%** exact.
- The same GT objects cropped by their axis-aligned bbox: routed **21.49%**,
  universal v5 **23.71%**. The current bbox rectification is the largest measured
  quality loss (367 quad-only vs 77 bbox-only routed successes).
- YOLO26n detection at IoU ≥ 0.5: recall **86.46%**, precision **93.63%**;
  matched type accuracy **79.43%** on this project validation domain.
- Full pipeline without vehicle filtering: routed **23.95%**, universal v5
  **24.77%** exact over every readable GT. PP-OCRv5 therefore remains the active
  legacy ONNX fallback; type routing is only a research candidate.
- Vehicle filtering removed one false positive and no matched object (precision
  93.63% → 93.71%). It adds negligible value on this set and is not evidence of
  robust off-vehicle rejection.
- CPU detector mean/median/p95: 108.53/104.58/150.85 ms. Estimated mean for one
  v5 policy is ~146.83 ms/image; the 100 ms CPU budget is not met. Prior T4
  detector measurements are much faster, so deployment hardware must be tested.
- Runtime smoke and semicolon CSV contract passed.

Artifacts: `kaggle_results/pipeline_e2e_local/full_pipeline_summary.json`,
`full_pipeline_predictions.csv`, `full_pipeline_matches.csv`,
`oracle_geometry_predictions.csv`, and `pipeline_smoke.csv`.

## Selected pose+CRNN evaluation

The pinned reference runtime was evaluated in a private, offline Kaggle kernel
on the archived project dataset with the same 1,219-readable-object denominator
and label rules. Seven project OCR train/validation content overlaps were
excluded by SHA-256 in the server corpus (two additional paths were unreadable,
so the readable denominator stayed equal to the legacy evaluation). Upstream
checkpoint-training overlap cannot be reconstructed from the published
repository and is therefore an explicit limitation.

- CRNN on GT quadrilaterals: **91.14% exact**, **94.09% character accuracy**
  over 1,219 readable objects. Synthetic per-type exact: type1 **92.44%**,
  type1a **98.88%**, type1b **91.64%**; real type1 **36.23%** (69 objects).
- Pose detector at IoU ≥ 0.5: recall **88.44%**, precision **93.42%**;
  matched type accuracy **86.25%**.
- Full image→CSV pipeline: OCR exact given a readable match **59.65%** and
  end-to-end exact **56.52%** (95% Wilson CI **53.72–59.28%**), more than twice
  the legacy bbox+PP-OCRv5 result of **24.77%** on the comparable corpus.
- Full-pipeline readable per-group exact: other/synthetic **46.25%**,
  type1/real **43.48%**, type1/synthetic **40.00%**, type1a/synthetic **80.15%**,
  type1b/synthetic **33.44%**. There are no readable real type1a/type1b objects,
  so those two real-target claims remain unproven.
- Kaggle CPU mean/median/p95 at detector `imgsz=960`:
  **200.01/196.05/264.94 ms/image**. The quality winner does not meet the
  100 ms CPU budget; deployment needs GPU measurement and possibly ONNX/TensorRT
  export or detector-size optimization.
- Applying the existing vehicle heuristic removed one prediction and did not
  change readable end-to-end exact accuracy. It is retained for contract
  compatibility, not counted as a demonstrated quality gain.
- The packaged production CLI also passed a separate Kaggle smoke with internet
  disabled: 20 images, 27 CSV rows, correct header, no skipped images, and
  254.9 ms/image mean over 19 timed images. Pinned Ultralytics wheels were
  installed from the private bundle with `pip --no-index --no-deps`.

Artifacts: `kaggle_results/reference_pose_crnn_v1/reference_evaluation_summary.json`,
`reference_ocr_predictions.csv`, `reference_pipeline_predictions.csv`, and
`reference_pipeline_matches.csv`; integrated runtime evidence is under
`kaggle_results/integration_smoke_v6/`.

## Changes and validation

- Added optional offline FastPlateOCR adapter and candidate-only YAML. Requires
  the upstream API returning PlatePrediction objects; pin a verified upstream
  revision/version in the server experiment. ONNX and matching YAML are local;
  no hub model argument or inference-time download is used.
- Adapter checks BGR→RGB/grayscale conversion, confidence alignment and missing
  artifacts. Unit tests use a stub recognizer; model accuracy is NOT tested.
- Full local regression suite: **182 passed** on 2026-09-28 after adding the
  reference runtime adapter and selection tests.
- Default `configs/pipeline.yaml` engine changed to `reference_pose_crnn`;
  PP-OCRv5 remains the explicit `--prefer-onnx` fallback.
- Added optional `active_by_plate_type` routing and regression tests. It remains
  disabled in the production config because it lost to universal v5 end to end.

## Integration and remaining evidence gaps

The archived Kaggle corpus contains 331 readable real type1 rows with images,
including 69 in validation. `logs/ocr_upload_corpus_audit.json` records 7,371
metadata rows, 1,181 missing unique images and five exact-content train/val
leakage pairs; the full-pipeline evaluation excluded those five pairs. No
readable real type1a/type1b validation exists. The PlatesMania ZIP contains only
two JPGs and no annotations, so it cannot close that evidence gap.

`scripts/audit_ocr_corpus.py` generated
`logs/ocr_corpus_audit.json`: 6,132 metadata rows; 754 missing unique images
(982 metadata rows reference missing images); train/val sizes 4,606/1,298;
zero existing readable real validation rows for all three target types; no
exact-content train/val overlaps among existing files. This prevents a
defensible real-target claim for type1a/type1b. Recover the actual server corpus
or assemble a reviewed, independent labeled real set.
The reference CRNN source was inspected: 32x160 grayscale inputs, right padding,
two-layer BiLSTM, last alphabet index as CTC blank; square plates are read as
separate rows. It is not compatible with blindly loading its weights into PARSeq.

The selected source is vendored under `src/vendor/volga_it_2026_lpr/` with its
MIT license and a provenance notice. Model checksums:

- detector: `9959f887421f80bbe247f6babccb8366bc1aef742eda1c5b339e60df77eaaa80`
- OCR: `96e4f374c3aae85dcd398fc899ac11ae8094204c8136fd23589bbde7cbce1dae`

Ultralytics is AGPL-3.0; proprietary redistribution requires a separate license
review. Remaining work needed to strengthen (not reproduce) the measured choice:

1. Obtain a reviewed, independent readable real test set for type1/type1a/type1b.
2. Establish disjoint train/validation/test sets by original image (including
   duplicates), inspect label validity and real per-type counts. Never select a
   model on the final test set; disclose possible upstream pretraining overlap.
3. Evaluate two-line preprocessing choices on the real validation set; fine-tune
   if required for rare types, keeping source images out of held-out data.
4. Measure warmed-up mean/median/p95 latency and memory on the actual deployment
   GPU; export/optimize only after checking that exact accuracy is preserved.
