#!/usr/bin/env python3
"""Build stratified train/val lists + Ultralytics data.yaml for the detector.

CPU-only. Also optionally builds a type1a style_b holdout for domain-gap eval
(never enters train).

  python scripts/prepare_detector_split.py --seed 42 --val-ratio 0.2
  python scripts/prepare_detector_split.py --build-style-b 150 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.generator.scene import apply_conditions, build_background, paste_plate, quad_bbox
from dataset.generator.numbers import random_plate_text
from dataset.generator.plate_render import render_plate
from src.data_collection.meta_store import MetaRow, MetaStore, write_yolo_label
from src.detection.split import (
    load_image_records,
    stratified_split,
    summarize_split,
    write_data_yaml,
    write_split_lists,
)
from src.utils.geometry import CLASS_NAME_TO_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare YOLO detector train/val split")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--data-yaml", type=Path, default=ROOT / "configs" / "data.yaml")
    p.add_argument("--summary", type=Path, default=ROOT / "logs" / "detector_split.json")
    p.add_argument(
        "--build-style-b",
        type=int,
        default=0,
        metavar="N",
        help="Generate N type1a images with alternate scene style (val-only domain gap)",
    )
    p.add_argument("--style-b-seed", type=int, default=777)
    return p.parse_args(argv)


def _generate_style_b(dataset: Path, count: int, seed: int) -> int:
    """Alternate type1a style: night-biased, smaller plates, stronger blur/dirt."""
    images_dir = dataset / "images" / "synthetic_style_b"
    labels_dir = dataset / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta = MetaStore(dataset / "meta.csv")
    # Drop previous style_b rows/files for idempotent rebuild.
    keep = [r for r in meta.rows if "style_b" not in str(r.get("source", ""))]
    meta._rows = keep  # noqa: SLF001
    for old in images_dir.glob("*.jpg"):
        old.unlink()
        (labels_dir / f"{old.stem}.txt").unlink(missing_ok=True)

    for i in range(count):
        rng = np.random.default_rng(seed + i * 1_000_003)
        plate = random_plate_text(rng)
        plate_bgra = render_plate(plate, "type1a", px_per_mm=float(rng.choice([2.2, 2.5, 2.8])))
        night = True  # style_b: always night-ish
        iw = int(rng.choice([800, 960, 1024]))
        ih = int(round(iw * float(rng.choice([0.75, 0.85, 1.0]))))
        scene = build_background(rng, iw, ih, night=True)
        # Smaller plate footprint than default generator.
        aspect = plate_bgra.shape[1] / plate_bgra.shape[0]
        target_w = int(iw * float(rng.uniform(0.12, 0.22)))
        target_h = max(16, int(target_w / aspect))
        margin = 8
        x0 = int(rng.integers(margin, max(margin + 1, iw - target_w - margin)))
        y0 = int(rng.integers(int(ih * 0.45), max(int(ih * 0.45) + 1, ih - target_h - margin)))
        jx, jy = target_w * 0.25, target_h * 0.25

        def j(span: float) -> float:
            return float(rng.uniform(-span, span))

        quad = np.array(
            [
                (x0 + j(jx), y0 + j(jy)),
                (x0 + target_w + j(jx), y0 + j(jy)),
                (x0 + target_w + j(jx), y0 + target_h + j(jy)),
                (x0 + j(jx), y0 + target_h + j(jy)),
            ],
            dtype=np.float32,
        )
        quad[:, 0] = np.clip(quad[:, 0], 0, iw - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, ih - 1)
        composed = paste_plate(scene, plate_bgra, quad)
        # Force dirt + blur more often for style_b.
        composed, conditions = apply_conditions(rng, composed, night=True)
        if "dirt" not in conditions and rng.random() < 0.7:
            from dataset.generator.scene import _dirt

            composed = _dirt(rng, composed)
            conditions.append("dirt")
        if "motion_blur" not in conditions:
            k = int(rng.choice([7, 9, 11]))
            composed = cv2.GaussianBlur(composed, (k, 1), 0)
            conditions.append("motion_blur")
        conditions = list(dict.fromkeys(["night", *conditions]))
        x, y, bw, bh = quad_bbox(quad)
        filename = f"syn_styleb_{seed}_{i:04d}.jpg"
        image_path = images_dir / filename
        cv2.imwrite(str(image_path), composed, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        rel = f"images/synthetic_style_b/{filename}"
        meta.append(
            MetaRow(
                image=rel,
                plate_num=plate,
                plate_type="type1a",
                bbox=MetaRow.format_bbox(x, y, bw, bh),
                quad=MetaRow.format_quad(quad),
                is_vehicle=1,
                is_synthetic=1,
                source="synthetic_generator_style_b",
                license="CC-BY-4.0",
                conditions=",".join(conditions),
            )
        )
        write_yolo_label(
            labels_dir / f"{image_path.stem}.txt",
            CLASS_NAME_TO_ID["type1a"],
            (x, y, bw, bh),
            (iw, ih),
        )
    meta.flush()
    return count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.resolve()
    if args.build_style_b > 0:
        n = _generate_style_b(dataset, args.build_style_b, args.style_b_seed)
        print(f"built style_b type1a images: {n} (seed={args.style_b_seed})")

    records = load_image_records(dataset / "meta.csv")
    train, val = stratified_split(records, val_ratio=args.val_ratio, seed=args.seed)
    paths = write_split_lists(dataset, train, val)
    data_yaml = write_data_yaml(
        args.data_yaml,
        dataset_root=dataset,
        train_list=paths["train"],
        val_list=paths["val"],
    )
    summary = summarize_split(train, val)
    summary["data_yaml"] = str(data_yaml)
    summary["lists"] = {k: str(v) for k, v in paths.items()}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"train={summary['train']}  val={summary['val']}")
    print(f"data.yaml → {data_yaml}")
    print(f"summary → {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
