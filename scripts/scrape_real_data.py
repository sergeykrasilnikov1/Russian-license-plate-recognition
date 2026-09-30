#!/usr/bin/env python3
"""Automatic collection of real GRZ images from open, licensed sources.

CPU-only, no GPU required. Sources that lack credentials or refuse access are
skipped with a recorded reason instead of aborting the run.

Examples:
  python scripts/scrape_real_data.py --sources commons openverse --limit 200
  python scripts/scrape_real_data.py --sources huggingface --limit 500 --hf-archives val.zip
  python scripts/scrape_real_data.py --probe-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_collection.coarse_label import CoarseTypeClassifier
from src.data_collection.collectors import COLLECTOR_REGISTRY
from src.data_collection.collectors.huggingface import HuggingFaceCollector
from src.data_collection.filters import FaceBlurrer, PerceptualDeduper, QualityFilter
from src.data_collection.licensing import LicensePolicy
from src.data_collection.pipeline import (
    DatasetPaths,
    Ingestor,
    format_summary_table,
)

DEFAULT_SOURCES = ("huggingface", "commons", "openverse", "roboflow", "kaggle", "platesmania")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape/download real GRZ images")
    p.add_argument("--out", type=Path, default=ROOT / "dataset", help="Dataset root")
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        choices=sorted(COLLECTOR_REGISTRY),
        help="Collectors to run",
    )
    p.add_argument("--limit", type=int, default=200, help="Max images per source")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--license-policy",
        choices=[p.value for p in LicensePolicy],
        default=LicensePolicy.NONCOMMERCIAL.value,
        help="strict = only CC0/PD/CC-BY/Apache/MIT; noncommercial = any CC without ND",
    )
    p.add_argument("--min-side", type=int, default=240, help="Reject images smaller than this")
    p.add_argument("--min-sharpness", type=float, default=20.0, help="Laplacian variance floor")
    p.add_argument("--dedup-distance", type=int, default=4, help="Max Hamming distance for phash dedup")
    p.add_argument(
        "--debug-set",
        type=Path,
        default=None,
        help="Organizers' 30-image debug set; hashed into a blocklist so it never enters the dataset",
    )
    p.add_argument(
        "--face-model",
        type=Path,
        default=ROOT / "weights" / "face" / "blaze_face_short_range.tflite",
        help="Optional MediaPipe face detector bundle; OpenCV Haar is used when absent",
    )
    p.add_argument(
        "--hf-archives",
        nargs="+",
        default=None,
        help="Restrict Hugging Face download to these archives (e.g. val.zip)",
    )
    p.add_argument(
        "--plate-types",
        nargs="+",
        default=None,
        choices=["type1", "type1a", "type1b", "other"],
        help="Restrict search-based sources (commons/openverse) to these target groups",
    )
    p.add_argument(
        "--no-rescan",
        action="store_true",
        help="Skip hashing already-collected images (faster, but weaker cross-run dedup)",
    )
    p.add_argument("--summary", type=Path, default=ROOT / "logs" / "scrape_summary.json")
    p.add_argument("--probe-only", action="store_true", help="Only report source availability")
    p.add_argument("--dry-run", action="store_true", help="Run all gates but write no files")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def build_collector(name: str, cache_dir: Path, args: argparse.Namespace):
    if name == "huggingface" and args.hf_archives:
        return HuggingFaceCollector(cache_dir, archives=tuple(args.hf_archives))
    if name in {"commons", "openverse"} and args.plate_types:
        return COLLECTOR_REGISTRY[name](cache_dir, plate_types=tuple(args.plate_types))
    return COLLECTOR_REGISTRY[name](cache_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    paths = DatasetPaths(args.out)
    collectors = [build_collector(n, paths.cache, args) for n in args.sources]

    if args.probe_only:
        print(f"{'source':<26} {'available':<10} reason")
        print("-" * 78)
        for c in collectors:
            ok, reason = c.check_available()
            print(f"{c.name:<26} {str(ok):<10} {reason}")
        return 0

    ingestor = Ingestor(
        paths=paths,
        policy=LicensePolicy(args.license_policy),
        quality=QualityFilter(min_side=args.min_side, min_sharpness=args.min_sharpness),
        deduper=PerceptualDeduper(max_distance=args.dedup_distance),
        face_blurrer=FaceBlurrer(args.face_model),
        classifier=CoarseTypeClassifier(),
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        orphans = ingestor.reconcile()
        if any(orphans.values()):
            logging.info(
                "removed orphans: %d staged, %d in images/real",
                orphans["raw_downloads"],
                orphans["images_real"],
            )

    blocked = ingestor.load_debug_blocklist(args.debug_set)
    logging.info("face blur backend: %s", ingestor.face_blurrer.backend)
    if args.debug_set:
        logging.info("debug-set blocklist hashes: %d", blocked)
    if not args.no_rescan:
        logging.info("hashed %d already-collected images for dedup", ingestor.load_collected_hashes())

    try:
        for collector in collectors:
            logging.info("=== %s (limit=%d) ===", collector.name, args.limit)
            stats = ingestor.run_source(collector, args.limit)
            logging.info(
                "%s: fetched=%d meta=%d pending=%d reason=%s",
                collector.name,
                stats.fetched,
                stats.labeled_images,
                stats.pending_images,
                stats.reason,
            )
    finally:
        ingestor.face_blurrer.close()

    summary = ingestor.summary()
    if not args.dry_run:
        ingestor.write_summary(args.summary)
    print()
    print(format_summary_table(summary))
    if not args.dry_run:
        print(f"\nsummary written to {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
