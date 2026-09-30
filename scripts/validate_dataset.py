#!/usr/bin/env python3
"""Validate dataset/ against the competition meta.csv schema and constraints.

Checks the fixed layout, the ten meta.csv columns, plate mask conformance,
bbox/quad geometry and image/label consistency, then writes a report that is
submitted together with the dataset.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plate_mask import validate_plate

REQUIRED_META_COLUMNS = [
    "image",
    "plate_num",
    "plate_type",
    "bbox",
    "quad",
    "is_vehicle",
    "is_synthetic",
    "source",
    "license",
    "conditions",
]

VALID_TYPES = {"type1", "type1a", "type1b", "other"}
VALID_CONDITIONS = {"day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Recommended minimums from the brief (images per group)
RECOMMENDED_MIN_IMAGES = {"type1a": 150, "type1b": 300, "other": 50}
RECOMMENDED_MIN_UNIQUE_PLATES = {"type1a": 50, "type1b": 100}
RECOMMENDED_MIN_SYNTHETIC = 5000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate GRZ dataset structure")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    p.add_argument("--report", type=Path, default=ROOT / "dataset" / "validation_report.txt")
    p.add_argument(
        "--strict-minimums",
        action="store_true",
        help="Treat unmet recommended volumes as errors instead of warnings",
    )
    return p.parse_args(argv)


def _check_int_list(value: str, count: int) -> bool:
    parts = value.split(",")
    if len(parts) != count:
        return False
    try:
        [int(p) for p in parts]
    except ValueError:
        return False
    return True


def validate(dataset: Path, strict_minimums: bool = False) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []

    for name in ("README.md", "LICENSE", "meta.csv"):
        if not (dataset / name).exists():
            errors.append(f"missing {name}")
    for sub in ("images/real", "images/synthetic", "labels", "generator"):
        if not (dataset / sub).is_dir():
            errors.append(f"missing directory {sub}")

    stats = {
        "rows": 0,
        "images": 0,
        "by_type": Counter(),
        "images_by_type": Counter(),
        "unique_plates_by_type": {},
        "real_images_by_type": {},
        "real_unique_plates_by_type": {},
        "real_rows": 0,
        "synthetic_rows": 0,
        "synthetic_images": 0,
        "licenses": Counter(),
    }

    meta = dataset / "meta.csv"
    if not meta.exists():
        return errors, warnings, stats

    with meta.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            errors.append("meta.csv has no header")
            return errors, warnings, stats
        missing = [c for c in REQUIRED_META_COLUMNS if c not in reader.fieldnames]
        if missing:
            errors.append(f"meta.csv missing columns: {missing}")
            return errors, warnings, stats

        seen_images: set[str] = set()
        plates_by_type: dict[str, set[str]] = {}
        images_by_type: dict[str, set[str]] = {}
        real_images: dict[str, set[str]] = {}
        real_plates: dict[str, set[str]] = {}
        synthetic_images: set[str] = set()
        image_sizes: dict[str, tuple[int, int]] = {}

        for i, row in enumerate(reader, start=2):
            stats["rows"] += 1
            if any(row.get(c) is None for c in REQUIRED_META_COLUMNS):
                errors.append(f"line {i}: incomplete row")
                continue
            image_rel = row["image"]
            plate_type = row["plate_type"]

            if plate_type not in VALID_TYPES:
                errors.append(f"line {i}: invalid plate_type={plate_type!r}")
                continue
            stats["by_type"][plate_type] += 1
            images_by_type.setdefault(plate_type, set()).add(image_rel)

            image_path = dataset / image_rel
            try:
                image_path.resolve().relative_to(dataset.resolve())
                if Path(image_rel).is_absolute():
                    raise ValueError
            except ValueError:
                errors.append(f"line {i}: image path must be relative and inside dataset")
                continue
            if image_rel not in seen_images:
                seen_images.add(image_rel)
                if not image_path.is_file():
                    errors.append(f"line {i}: image not found: {image_rel}")
                elif image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    errors.append(f"line {i}: unsupported image type: {image_rel}")
                else:
                    try:
                        with Image.open(image_path) as image:
                            image_sizes[image_rel] = image.size
                            image.verify()
                    except (OSError, ValueError):
                        errors.append(f"line {i}: unreadable image: {image_rel}")
                if not (dataset / "labels" / f"{Path(image_rel).stem}.txt").is_file():
                    errors.append(f"line {i}: label not found: {Path(image_rel).stem}.txt")

            ok, reason = validate_plate(row["plate_num"])
            if not ok:
                errors.append(f"line {i}: plate_num {row['plate_num']!r} fails mask ({reason})")

            if not _check_int_list(row["bbox"], 4):
                errors.append(f"line {i}: bbox must be 4 integers, got {row['bbox']!r}")
            else:
                x, y, w, h = map(int, row["bbox"].split(","))
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    errors.append(f"line {i}: bbox must have nonnegative origin and positive size")
                elif image_rel in image_sizes:
                    width, height = image_sizes[image_rel]
                    if x + w > width or y + h > height:
                        errors.append(f"line {i}: bbox outside image")
            if not _check_int_list(row["quad"], 8):
                errors.append(f"line {i}: quad must be 8 integers, got {row['quad']!r}")
            else:
                coords = list(map(int, row["quad"].split(",")))
                points = list(zip(coords[::2], coords[1::2]))
                turns = []
                for j in range(4):
                    a, b, c = points[j], points[(j + 1) % 4], points[(j + 2) % 4]
                    turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
                if not all(turn > 0 for turn in turns):
                    errors.append(f"line {i}: quad must be convex and clockwise, with distinct corners")
                if image_rel in image_sizes:
                    width, height = image_sizes[image_rel]
                    if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
                        errors.append(f"line {i}: quad outside image")

            for field in ("is_vehicle", "is_synthetic"):
                if row[field] not in {"0", "1"}:
                    errors.append(f"line {i}: {field} must be 0 or 1, got {row[field]!r}")

            if row["is_synthetic"] == "1":
                stats["synthetic_rows"] += 1
                synthetic_images.add(image_rel)
            else:
                stats["real_rows"] += 1

            if not row["source"].strip():
                errors.append(f"line {i}: empty source")
            if not row["license"].strip():
                errors.append(f"line {i}: empty license")
            stats["licenses"][row["license"].split(" ")[0]] += 1

            bad_conditions = set(filter(None, row["conditions"].split(","))) - VALID_CONDITIONS
            if bad_conditions:
                warnings.append(f"line {i}: unknown conditions {sorted(bad_conditions)}")

            if "#" not in row["plate_num"] and ok:
                plates_by_type.setdefault(plate_type, set()).add(row["plate_num"])
            if row["is_synthetic"] == "0":
                real_images.setdefault(plate_type, set()).add(image_rel)
                if "#" not in row["plate_num"] and ok:
                    real_plates.setdefault(plate_type, set()).add(row["plate_num"])

        stats["real_images_by_type"] = {k: len(v) for k, v in real_images.items()}
        stats["synthetic_images"] = len(synthetic_images)
        stats["real_unique_plates_by_type"] = {k: len(v) for k, v in real_plates.items()}
        stats["images"] = len(seen_images)
        stats["images_by_type"] = Counter({k: len(v) for k, v in images_by_type.items()})
        stats["unique_plates_by_type"] = {k: len(v) for k, v in plates_by_type.items()}

    orphan_labels = 0
    labels_dir = dataset / "labels"
    if labels_dir.is_dir():
        image_stems = {Path(p).stem for p in seen_images}
        for label in labels_dir.glob("*.txt"):
            if label.stem not in image_stems:
                orphan_labels += 1
    if orphan_labels:
        warnings.append(f"{orphan_labels} label files have no matching meta.csv image")

    bucket = errors if strict_minimums else warnings
    for plate_type, minimum in RECOMMENDED_MIN_IMAGES.items():
        actual = stats["real_images_by_type"].get(plate_type, 0)
        if actual < minimum:
            bucket.append(
                f"real group {plate_type}: {actual} images < recommended {minimum}"
            )
    for plate_type, minimum in RECOMMENDED_MIN_UNIQUE_PLATES.items():
        actual = stats["real_unique_plates_by_type"].get(plate_type, 0)
        if actual < minimum:
            bucket.append(
                f"real group {plate_type}: {actual} unique plates < recommended {minimum}"
            )
    if stats["synthetic_images"] < RECOMMENDED_MIN_SYNTHETIC:
        bucket.append(
            f"synthetic: {stats['synthetic_images']} images < recommended {RECOMMENDED_MIN_SYNTHETIC}"
        )

    return errors, warnings, stats


def format_report(errors: list[str], warnings: list[str], stats: dict) -> str:
    lines = ["GRZ dataset validation report", "=" * 40, ""]
    lines.append(f"status : {'FAIL' if errors else 'PASS'}")
    lines.append(f"rows   : {stats['rows']} (real={stats['real_rows']}, synthetic={stats['synthetic_rows']})")
    lines.append(f"images : {stats['images']}")
    lines.append("")
    lines.append("plates per type:")
    for plate_type in sorted(VALID_TYPES):
        lines.append(
            f"  {plate_type:<8} plates={stats['by_type'].get(plate_type, 0):<6}"
            f" images={stats['images_by_type'].get(plate_type, 0):<6}"
            f" unique_plate_nums={stats['unique_plates_by_type'].get(plate_type, 0)}"
            f" real_images={stats['real_images_by_type'].get(plate_type, 0)}"
            f" real_unique={stats['real_unique_plates_by_type'].get(plate_type, 0)}"
        )
    lines.append("")
    lines.append("licenses:")
    for name, count in sorted(stats["licenses"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:<16} {count}")
    lines.append("")
    lines.append(f"errors   ({len(errors)}):")
    lines.extend(f"  - {e}" for e in errors[:80])
    if len(errors) > 80:
        lines.append(f"  ... and {len(errors) - 80} more")
    lines.append("")
    lines.append(f"warnings ({len(warnings)}):")
    lines.extend(f"  - {w}" for w in warnings[:80])
    if len(warnings) > 80:
        lines.append(f"  ... and {len(warnings) - 80} more")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors, warnings, stats = validate(args.dataset, args.strict_minimums)
    report = format_report(errors, warnings, stats)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
