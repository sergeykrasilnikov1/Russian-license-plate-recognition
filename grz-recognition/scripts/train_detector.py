#!/usr/bin/env python3
"""Train YOLOv11n plate detector (classes = plate types). Stage 4."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GRZ detector")
    p.add_argument("--config", type=Path, default=Path("configs/detector.yaml"))
    p.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Stage 4: train detector with {args.config}")


if __name__ == "__main__":
    main()
