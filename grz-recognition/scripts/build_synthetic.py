#!/usr/bin/env python3
"""Build synthetic GRZ images (GOST R 50577-2018 types 1 / 1A / 1B).

Implemented in Stage 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic GRZ dataset")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--count", type=int, default=5000, help="Total synthetic images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--type-weights",
        type=str,
        default="type1=0.35,type1a=0.35,type1b=0.25,other=0.05",
        help="Class mix; bump type1a/type1b if real data is scarce",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Stage 3: generate {args.count} images with seed={args.seed} → {args.out}"
    )


if __name__ == "__main__":
    main()
