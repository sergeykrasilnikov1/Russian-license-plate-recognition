#!/usr/bin/env python3
"""Benchmark end-to-end latency (detect + warp + OCR) vs 100 ms budget."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark GRZ pipeline latency")
    p.add_argument("--input", type=Path, required=True, help="Directory of images")
    p.add_argument("--n", type=int, default=100, help="Max images to time")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Stage 5/6: benchmark latency on {args.input} (n={args.n})"
    )


if __name__ == "__main__":
    main()
