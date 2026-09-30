"""Stratified train/val split for the plate detector."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.geometry import PLATE_TYPES, class_names


@dataclass(frozen=True)
class ImageRecord:
    image: str  # relative path under dataset/
    plate_type: str
    is_synthetic: int
    source: str
    style: str  # "real" | "synth" | "style_b"


def _primary_type(types: list[str]) -> str:
    """Majority plate_type on a multi-plate image; ties → first in PLATE_TYPES order."""
    counts: dict[str, int] = defaultdict(int)
    for t in types:
        counts[t] += 1
    best = max(counts.values())
    candidates = [t for t, c in counts.items() if c == best]
    for name in PLATE_TYPES:
        if name in candidates:
            return name
    return candidates[0]


def _style_of(row: pd.Series) -> str:
    src = str(row.get("source", ""))
    if int(row["is_synthetic"]) != 1:
        return "real"
    if "style_b" in src:
        return "style_b"
    return "synth"


def load_image_records(meta_csv: Path) -> list[ImageRecord]:
    meta = pd.read_csv(meta_csv, sep=";")
    records: list[ImageRecord] = []
    for image, group in meta.groupby("image", sort=False):
        types = group["plate_type"].astype(str).tolist()
        is_syn = int(group["is_synthetic"].iloc[0])
        source = str(group["source"].iloc[0])
        style = _style_of(group.iloc[0])
        records.append(
            ImageRecord(
                image=str(image),
                plate_type=_primary_type(types),
                is_synthetic=is_syn,
                source=source,
                style=style,
            )
        )
    return records


def stratified_split(
    records: list[ImageRecord],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    holdout_styles: tuple[str, ...] = ("style_b",),
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split images by (plate_type, is_synthetic). Holdout styles never enter train."""
    rng = np.random.default_rng(seed)
    train: list[ImageRecord] = []
    val: list[ImageRecord] = []

    holdout = [r for r in records if r.style in holdout_styles]
    pool = [r for r in records if r.style not in holdout_styles]
    val.extend(holdout)

    buckets: dict[tuple[str, int], list[ImageRecord]] = defaultdict(list)
    for r in pool:
        buckets[(r.plate_type, r.is_synthetic)].append(r)

    for key in sorted(buckets.keys()):
        items = buckets[key]
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
        # Keep at least one train example when the bucket is tiny but >1.
        if len(items) > 1 and n_val >= len(items):
            n_val = len(items) - 1
        val_idx = set(int(i) for i in idx[:n_val])
        for i, item in enumerate(items):
            (val if i in val_idx else train).append(item)

    return train, val


def write_split_lists(
    dataset_root: Path,
    train: list[ImageRecord],
    val: list[ImageRecord],
    splits_dir: Path | None = None,
) -> dict[str, Path]:
    """Write Ultralytics path lists (absolute or dataset-relative)."""
    splits_dir = splits_dir or (dataset_root / "splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[ImageRecord]) -> Path:
        path = splits_dir / name
        lines = []
        for r in rows:
            img = dataset_root / r.image
            lines.append(str(img.resolve()))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    paths = {
        "train": dump("train.txt", train),
        "val": dump("val.txt", val),
        "val_real_type1": dump(
            "val_real_type1.txt",
            [r for r in val if r.style == "real" and r.plate_type == "type1"],
        ),
        "val_syn_type1a": dump(
            "val_syn_type1a.txt",
            [r for r in val if r.style == "synth" and r.plate_type == "type1a"],
        ),
        "val_syn_type1a_style_b": dump(
            "val_syn_type1a_style_b.txt",
            [r for r in val if r.style == "style_b" and r.plate_type == "type1a"],
        ),
        "val_real_type1b": dump(
            "val_real_type1b.txt",
            [r for r in val if r.style == "real" and r.plate_type == "type1b"],
        ),
    }
    return paths


def write_data_yaml(
    out_path: Path,
    *,
    dataset_root: Path,
    train_list: Path,
    val_list: Path,
) -> Path:
    """Ultralytics data.yaml — class names from geometry.PLATE_TYPES only."""
    names = class_names()
    lines = [
        f"# Auto-generated. Class names from src.utils.geometry.PLATE_TYPES.",
        f"path: {dataset_root.resolve()}",
        f"train: {train_list.resolve()}",
        f"val: {val_list.resolve()}",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def summarize_split(train: list[ImageRecord], val: list[ImageRecord]) -> dict:
    def counts(rows: list[ImageRecord]) -> dict:
        out: dict[str, int] = defaultdict(int)
        for r in rows:
            out[f"{r.plate_type}|syn={r.is_synthetic}|{r.style}"] += 1
        return dict(sorted(out.items()))

    return {"train": len(train), "val": len(val), "train_buckets": counts(train), "val_buckets": counts(val)}
