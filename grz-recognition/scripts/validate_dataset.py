#!/usr/bin/env python3
"""Validate dataset/ against competition meta.csv schema and constraints.

Adapted from organizers' validate_dataset.py requirements (Stage 7 polish).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate GRZ dataset structure")
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--report", type=Path, default=Path("dataset/validation_report.txt"))
    return p.parse_args()


def validate(dataset: Path) -> list[str]:
    errors: list[str] = []
    for name in ("README.md", "LICENSE", "meta.csv"):
        if not (dataset / name).exists():
            errors.append(f"missing {name}")
    for sub in ("images/real", "images/synthetic", "labels", "generator"):
        if not (dataset / sub).is_dir():
            errors.append(f"missing directory {sub}")

    meta = dataset / "meta.csv"
    if meta.exists():
        with meta.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames is None:
                errors.append("meta.csv has no header")
            else:
                missing = [c for c in REQUIRED_META_COLUMNS if c not in reader.fieldnames]
                if missing:
                    errors.append(f"meta.csv missing columns: {missing}")
                for i, row in enumerate(reader, start=2):
                    pt = row.get("plate_type", "")
                    if pt and pt not in VALID_TYPES:
                        errors.append(f"line {i}: invalid plate_type={pt!r}")
    return errors


def main() -> int:
    args = parse_args()
    errors = validate(args.dataset)
    report = "OK\n" if not errors else "FAIL\n" + "\n".join(errors) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
