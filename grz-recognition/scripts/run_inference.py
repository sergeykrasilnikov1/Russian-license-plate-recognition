#!/usr/bin/env python3
"""Offline GRZ inference: detect → vehicle filter → warp → OCR → CSV.

Usage:
  python scripts/run_inference.py --input <dir> --output result.csv
  GRZ_INPUT_DIR=/path GRZ_OUTPUT_CSV=out.csv python scripts/run_inference.py

Default path is the local YOLO-pose + plate-specific CRNN bundle (no network).
Use --prefer-onnx for the lighter legacy fallback; --from-meta is CPU debug only.
"""

from __future__ import annotations

import argparse
import os
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd

from src.detection.infer import DetectorUnavailable
from src.ocr.backends import BackendUnavailable
from src.ocr.warp import parse_quad, DegenerateQuadError
from src.pipeline.infer import GrzPipeline, PlateResult, load_yaml, write_results_csv
from src.pipeline.reference import ReferencePoseCrnnPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run offline GRZ recognition")
    p.add_argument(
        "--input",
        type=Path,
        default=os.environ.get("GRZ_INPUT_DIR"),
        help="Directory with .jpg/.png images (or set GRZ_INPUT_DIR)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=os.environ.get("GRZ_OUTPUT_CSV", "result.csv"),
        help="Output CSV path (or set GRZ_OUTPUT_CSV)",
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "pipeline.yaml")
    p.add_argument("--ocr-config", type=Path, default=ROOT / "configs" / "ocr.yaml")
    p.add_argument(
        "--from-meta",
        type=Path,
        default=None,
        help="Optional meta.csv: use GT quads instead of the detector (debug / CPU-only)",
    )
    p.add_argument(
        "--prefer-onnx",
        action="store_true",
        help="Use the lightweight YOLO26n+bbox+PP-OCR ONNX fallback instead of the selected pose+CRNN engine",
    )
    p.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    p.add_argument(
        "--no-vehicle-filter",
        action="store_true",
        help="Skip is_vehicle heuristic (useful with --from-meta on synthetic crops)",
    )
    p.add_argument("--warmup", type=int, default=1, help="Images to skip in the latency mean")
    return p.parse_args()


def _meta_index(meta_csv: Path) -> dict[str, list[dict]]:
    df = pd.read_csv(meta_csv, sep=";")
    by_name: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        rel = str(row["image"])
        name = str((meta_csv.parent / rel).resolve())
        rec = {
            "quad": str(row.get("quad", "")),
            "plate_type": str(row.get("plate_type", "type1")),
            "bbox": str(row.get("bbox", "")),
        }
        by_name.setdefault(name, []).append(rec)

    return by_name


def _bbox_xyxy(bbox: str) -> tuple[float, float, float, float] | None:
    parts = [float(x) for x in str(bbox).replace(" ", "").split(",") if x]
    if len(parts) != 4:
        return None
    x, y, w, h = parts
    return x, y, x + w, y + h


def _process_one(
    pipeline: GrzPipeline,
    image,
    name: str,
    *,
    use_meta: bool,
    recs: list[dict] | None,
) -> list[PlateResult]:
    if use_meta:
        rows: list[PlateResult] = []
        if not recs:
            return rows
        for rec in recs:
            try:
                quad = parse_quad(rec["quad"])
            except ValueError:
                continue
            try:
                item = pipeline.process_with_gt_quad(
                    image,
                    name,
                    quad,
                    rec["plate_type"],
                    apply_vehicle_filter=pipeline.vehicle_enabled,
                    bbox_xyxy=_bbox_xyxy(rec["bbox"]),
                )
            except (ValueError, DegenerateQuadError) as exc:
                logging.warning("Skipping invalid metadata for %s: %s", name, exc)
                continue
            if item is not None:
                rows.append(item)
        return rows
    return pipeline.process_image(image, name)


def main() -> int:
    args = parse_args()
    if not args.input:
        raise SystemExit("Provide --input or set GRZ_INPUT_DIR")
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"input is not a directory: {input_dir}")

    pipe_cfg = load_yaml(args.config)
    ocr_cfg = load_yaml(args.ocr_config)
    exts = {e.lower() for e in (pipe_cfg.get("image_extensions") or [".jpg", ".jpeg", ".png"])}
    images = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if not images:
        write_results_csv([], Path(args.output))
        print(f"wrote 0 rows → {args.output} (no images)")
        return 0
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    names = [path.name for path in images]
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate image filenames in input subdirectories; use unique filenames for the CSV contract")

    use_meta = args.from_meta is not None
    try:
        engine_name = str(pipe_cfg.get("engine", "legacy_onnx"))
        if not use_meta and not args.prefer_onnx and engine_name == "reference_pose_crnn":
            pipeline = ReferencePoseCrnnPipeline(pipe_cfg)
        else:
            pipeline = GrzPipeline(
                pipe_cfg,
                ocr_cfg,
                prefer_onnx=args.prefer_onnx,
                require_detector=not use_meta,
            )
    except DetectorUnavailable as exc:
        raise SystemExit(
            f"{exc}\nHint: install requirements-train.txt and keep weights/reference/*.pt "
            "for the default engine; use --prefer-onnx for the fallback or --from-meta for debug."
        ) from exc

    if args.no_vehicle_filter:
        pipeline.vehicle_enabled = False

    meta_idx = _meta_index(args.from_meta) if use_meta else {}
    rows: list[PlateResult] = []
    times_ms: list[float] = []
    processed = 0
    skipped = 0

    for i, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            logging.warning("Cannot read image: %s", path)
            skipped += 1
            continue
        name = path.name
        recs = None
        if use_meta:
            recs = meta_idx.get(str(path.resolve()))
        t0 = time.perf_counter()
        try:
            got = _process_one(pipeline, image, name, use_meta=use_meta, recs=recs)
        except (BackendUnavailable, DetectorUnavailable) as exc:
            raise SystemExit(str(exc)) from exc
        dt = (time.perf_counter() - t0) * 1000.0
        if processed >= max(0, args.warmup):
            times_ms.append(dt)
        processed += 1
        rows.extend(got)

    out = Path(args.output)
    write_results_csv(rows, out)
    header = out.read_text(encoding="utf-8").splitlines()[0] if out.stat().st_size else ""
    ok_fmt = header == "image;plate_num;plate_type;confidence"
    mean_ms = sum(times_ms) / len(times_ms) if times_ms else float("nan")
    budget = float(pipe_cfg.get("latency_budget_ms", 100))
    fits = mean_ms <= budget if times_ms else False
    print(
        f"wrote {len(rows)} rows → {out}  header_ok={ok_fmt}  "
        f"images={processed} skipped={skipped} timed={len(times_ms)}  mean_ms={mean_ms:.1f}  budget={budget:.0f}  fits={fits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
