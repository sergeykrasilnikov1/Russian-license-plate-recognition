#!/usr/bin/env python3
"""Export a trained YOLO detector to ONNX.

Real export requires ultralytics on the GPU server. Locally use --dry-run.

  python scripts/export_onnx.py --weights runs/detect/grz_yolo11n/weights/best.pt \\
      --imgsz 640 --half --out weights/detector.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.train import export_onnx, load_detector_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export GRZ detector to ONNX")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "detector.yaml")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--opset", type=int, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "weights" / "detector.onnx")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_detector_config(args.config)
    imgsz = args.imgsz if args.imgsz is not None else int(cfg.get("imgsz", 640))
    opset = args.opset if args.opset is not None else int(cfg.get("opset", 12))

    if args.dry_run:
        if not args.weights.name.endswith(".pt"):
            print(f"warning: expected .pt weights, got {args.weights}", file=sys.stderr)
        print(f"dry-run OK  weights={args.weights}  imgsz={imgsz}  half={args.half}  out={args.out}")
        return 0

    if not args.weights.is_file():
        print(f"error: weights not found: {args.weights}", file=sys.stderr)
        return 2

    exported = export_onnx(
        args.weights,
        imgsz=imgsz,
        half=args.half,
        opset=opset,
        simplify=bool(cfg.get("simplify", True)),
        dynamic=bool(cfg.get("dynamic", False)),
    )
    if exported is None:
        print("error: export returned nothing", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported).resolve() != args.out.resolve():
        shutil.copy2(exported, args.out)
    print(f"onnx → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
