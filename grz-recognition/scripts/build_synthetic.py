#!/usr/bin/env python3
"""Build synthetic GRZ images (GOST R 50577-2018 types 1 / 1A / 1B / other).

CPU-only. Reproducible for a given --seed. Default class mix over-weights
type1a and type1b to close the Stage 2 real-data deficit.

  python scripts/build_synthetic.py --count 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.generator.generate import DEFAULT_WEIGHTS, generate_dataset, parse_type_weights


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic GRZ dataset")
    p.add_argument("--out", type=Path, default=ROOT / "dataset")
    p.add_argument("--count", type=int, default=5000, help="Total synthetic images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--type-weights",
        type=str,
        default="type1=0.22,type1a=0.38,type1b=0.32,other=0.08",
        help="Class mix; type1a/type1b are boosted because real data is scarce",
    )
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--keep-existing", action="store_true", help="Append instead of replacing previous synthetics")
    p.add_argument("--summary", type=Path, default=ROOT / "logs" / "synthetic_summary.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weights = parse_type_weights(args.type_weights) if args.type_weights else DEFAULT_WEIGHTS
    started = time.perf_counter()
    stats = generate_dataset(
        out_dir=args.out,
        count=args.count,
        seed=args.seed,
        type_weights=weights,
        jpeg_quality=args.jpeg_quality,
        overwrite=not args.keep_existing,
    )
    stats["elapsed_s"] = round(time.perf_counter() - started, 1)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"generated {stats['count']} images  seed={stats['seed']}  {stats['elapsed_s']}s")
    for name, n in sorted(stats["type_counts"].items()):
        print(f"  {name:<8} plates={n:<5} unique={stats['unique'].get(name, 0)}")
    print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
