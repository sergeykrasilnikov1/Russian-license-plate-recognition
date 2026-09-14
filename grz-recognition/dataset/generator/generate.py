"""Dataset-level synthetic generation: assign types, write images + meta.csv."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_collection.meta_store import MetaRow, MetaStore, write_yolo_label
from src.utils.geometry import CLASS_NAME_TO_ID, PLATE_TYPES

from .numbers import random_plate_text
from .plate_render import render_plate
from .scene import apply_conditions, build_background, paste_plate, quad_bbox, random_destination_quad

DEFAULT_WEIGHTS = {
    # Stage 2 left type1a/type1b empty in meta.csv — overweight them here.
    "type1": 0.22,
    "type1a": 0.38,
    "type1b": 0.32,
    "other": 0.08,
}


def parse_type_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for chunk in text.split(","):
        name, _, value = chunk.partition("=")
        name = name.strip()
        if not name:
            continue
        if name not in PLATE_TYPES:
            raise ValueError(f"unknown plate type in --type-weights: {name}")
        weights[name] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("type weights must sum to a positive number")
    return {k: v / total for k, v in weights.items()}


def _choose_type(rng: np.random.Generator, names: list[str], probs: list[float]) -> str:
    return str(rng.choice(names, p=probs))


def _scene_size(rng: np.random.Generator) -> tuple[int, int]:
    width = int(rng.choice([960, 1024, 1280, 1600]))
    height = int(round(width * float(rng.choice([0.56, 0.625, 0.75]))))
    return width, height


def generate_one(index: int, seed: int, plate_type: str) -> dict:
    rng = np.random.default_rng(int(seed) + index * 1_000_003)
    plate = random_plate_text(rng)
    plate_bgra = render_plate(plate, plate_type)
    night = bool(rng.random() < 0.22)
    iw, ih = _scene_size(rng)
    scene = build_background(rng, iw, ih, night)
    quad = random_destination_quad(rng, (iw, ih), (plate_bgra.shape[1], plate_bgra.shape[0]), plate_type)
    composed = paste_plate(scene, plate_bgra, quad)
    composed, conditions = apply_conditions(rng, composed, night)
    x, y, bw, bh = quad_bbox(quad)
    on_vehicle = 0 if (plate_type == "other" and rng.random() < 0.35) else 1
    return {
        "index": index,
        "image": composed,
        "plate": plate,
        "plate_type": plate_type,
        "quad": quad,
        "bbox": (x, y, bw, bh),
        "conditions": conditions,
        "is_vehicle": on_vehicle,
        "size": (iw, ih),
    }


def generate_dataset(
    out_dir: Path,
    count: int,
    seed: int,
    type_weights: dict[str, float] | None = None,
    jpeg_quality: int = 90,
    overwrite: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    images_dir = out_dir / "images" / "synthetic"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    weights = type_weights or DEFAULT_WEIGHTS
    names = list(weights)
    probs = [weights[n] for n in names]

    meta = MetaStore(out_dir / "meta.csv")
    if overwrite:
        keep = [r for r in meta.rows if r.get("is_synthetic") != "1"]
        meta._rows = keep  # noqa: SLF001 — ledger rewrite of the synthetic slice
        for old in images_dir.glob("*.jpg"):
            old.unlink()
            (labels_dir / f"{old.stem}.txt").unlink(missing_ok=True)

    assign_rng = np.random.default_rng(seed)
    type_of = [_choose_type(assign_rng, names, probs) for _ in range(count)]

    type_counts: Counter[str] = Counter()
    unique: dict[str, set[str]] = {t: set() for t in PLATE_TYPES}

    for i in range(count):
        sample = generate_one(i, seed, type_of[i])
        filename = f"syn_{seed}_{i:05d}.jpg"
        image_path = images_dir / filename
        cv2.imwrite(str(image_path), sample["image"], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        rel = f"images/synthetic/{filename}"
        x, y, bw, bh = sample["bbox"]
        meta.append(
            MetaRow(
                image=rel,
                plate_num=sample["plate"],
                plate_type=sample["plate_type"],
                bbox=MetaRow.format_bbox(x, y, bw, bh),
                quad=MetaRow.format_quad(sample["quad"]),
                is_vehicle=sample["is_vehicle"],
                is_synthetic=1,
                source="synthetic_generator",
                license="CC-BY-4.0",
                conditions=",".join(sample["conditions"]),
            )
        )
        write_yolo_label(
            labels_dir / f"{image_path.stem}.txt",
            CLASS_NAME_TO_ID[sample["plate_type"]],
            sample["bbox"],
            sample["size"],
        )
        type_counts[sample["plate_type"]] += 1
        unique[sample["plate_type"]].add(sample["plate"])
        if (i + 1) % 250 == 0 or i + 1 == count:
            meta.flush()

    meta.flush()
    return {
        "count": count,
        "seed": seed,
        "type_counts": dict(type_counts),
        "unique": {k: len(v) for k, v in unique.items()},
        "weights": weights,
    }
