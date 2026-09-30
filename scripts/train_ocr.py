#!/usr/bin/env python3
"""Fine-tune OCR on plate crops. Stage 5."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train / fine-tune GRZ OCR")
    p.add_argument("--config", type=Path, default=Path("configs/ocr.yaml"))
    p.add_argument("--crops-dir", type=Path, default=Path("dataset/crops"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit("OCR fine-tuning is not implemented. Inference uses pretrained local pose/CRNN or ONNX weights; see README.md.")


if __name__ == "__main__":
    main()
