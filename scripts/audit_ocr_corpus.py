#!/usr/bin/env python3
"""Audit OCR labels and split coverage without optional dependencies.

Exit 2 means the corpus cannot support a complete real-target OCR evaluation.
Foreign absolute split paths are relocated only through an explicit images/
component; ambiguous/unknown paths remain visible in the report.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def split_key(raw, root):
    path = Path(raw.strip())
    if not path.is_absolute():
        return path.as_posix().removeprefix("./")
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        if path.parts.count("images") == 1:
            return Path(*path.parts[path.parts.index("images"):]).as_posix()
        return path.as_posix()


def audit(root):
    root = root.resolve()
    meta = root / "meta.csv"
    with meta.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    splits = {}
    for name in ("train", "val"):
        path = root / "splits" / f"{name}.txt"
        splits[name] = {split_key(s, root) for s in path.read_text().splitlines() if s.strip()} if path.exists() else set()
    strata = Counter()
    missing = set()
    known = {r["image"] for r in rows}
    real_readable_val = Counter()
    for row in rows:
        present = (root / row["image"]).is_file()
        readable = bool(row["plate_num"].strip()) and "#" not in row["plate_num"]
        real = row["is_synthetic"] == "0"
        in_val = row["image"] in splits["val"]
        key = f"{row['plate_type']}/{'real' if real else 'synthetic'}/{'readable' if readable else 'unknown'}/{'present' if present else 'missing'}"
        strata[key] += 1
        if not present:
            missing.add(row["image"])
        if present and readable and real and in_val:
            real_readable_val[row["plate_type"]] += 1
    # Exact content duplicates can leak even when filenames differ.
    hashes = {}
    content_overlap = []
    for name in ("train", "val"):
        for key in sorted(splits[name]):
            path = root / key
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if name == "train":
                hashes.setdefault(digest, []).append(key)
            elif digest in hashes:
                content_overlap.append({"val": key, "train": hashes[digest], "sha256": digest})
    absent_types = [t for t in ("type1", "type1a", "type1b") if not real_readable_val[t]]
    return {
        "meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
        "rows": len(rows), "unique_images": len(known),
        "strata": dict(sorted(strata.items())),
        "missing_images": sorted(missing),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_images_without_meta": {k: sorted(v - known) for k, v in splits.items()},
        "train_val_path_overlap": sorted(splits["train"] & splits["val"]),
        "train_val_exact_content_overlap": content_overlap,
        "real_readable_val_rows": dict(real_readable_val),
        "target_types_without_real_readable_val": absent_types,
        "ready_for_full_target_comparison": not (
            missing or absent_types or content_overlap or splits["train"] & splits["val"]
            or not splits["train"] or not splits["val"]
            or any(v - known for v in splits.values())
        ),
        "limitations": "Readability means nonempty text without #, not verified label correctness. Exact hashes do not detect near duplicates. No upstream-pretraining overlap audit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key in ("rows", "split_sizes", "real_readable_val_rows", "target_types_without_real_readable_val", "ready_for_full_target_comparison"):
        print(f"{key}: {report[key]}")
    print(f"missing_images: {len(report['missing_images'])}")
    print(f"train_val_exact_content_overlap: {len(report['train_val_exact_content_overlap'])}")
    return 0 if report["ready_for_full_target_comparison"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
