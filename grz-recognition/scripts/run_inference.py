#!/usr/bin/env python3
"""Offline GRZ inference entry point.

Usage:
  python scripts/run_inference.py --input <dir> --output result.csv
  GRZ_INPUT_DIR=/path GRZ_OUTPUT_CSV=out.csv python scripts/run_inference.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run offline GRZ recognition")
    p.add_argument(
        "--input",
        type=Path,
        default=os.environ.get("GRZ_INPUT_DIR"),
        help="Directory with .jpg/.png images (or set GRZ_INPUT_DIR)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=os.environ.get("GRZ_OUTPUT_CSV", "result.csv"),
        help="Output CSV path (or set GRZ_OUTPUT_CSV)",
    )
    p.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input:
        raise SystemExit("Provide --input or set GRZ_INPUT_DIR")
    raise NotImplementedError(
        f"Stage 6: run pipeline on {args.input} → {args.output}"
    )


if __name__ == "__main__":
    main()
