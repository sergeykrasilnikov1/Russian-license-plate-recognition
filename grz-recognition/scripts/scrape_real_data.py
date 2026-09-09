#!/usr/bin/env python3
"""Automatic collection of real GRZ images from open licensed sources.

Implemented in Stage 2. Entry point stub for Stage 1 scaffolding.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape/download real GRZ images")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("dataset"),
        help="Dataset root (images/real, labels, meta.csv)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sources",
        nargs="+",
        default=["huggingface", "roboflow", "platesmania"],
        help="Which collectors to run",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Stage 2: implement collectors for {args.sources} → {args.out}"
    )


if __name__ == "__main__":
    main()
