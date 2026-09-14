#!/usr/bin/env python3
"""Train YOLOv11n plate detector (classes = plate types from geometry.PLATE_TYPES).

Local CPU machine: use --dry-run to validate config + data.yaml without ultralytics.
Real training runs on the GPU server (see docs/server_training_guide.md).

  python scripts/prepare_detector_split.py --seed 42 --build-style-b 150
  python scripts/train_detector.py --config configs/detector.yaml --data configs/data.yaml \\
      --profile accurate --device 0 --seed 42
  python scripts/train_detector.py --dry-run --profile fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.train import assert_class_list_matches_geometry, load_detector_config, resolve_profile, train_yolo
from src.utils.geometry import class_names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GRZ YOLOv11n detector")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "detector.yaml")
    p.add_argument("--data", type=Path, default=ROOT / "configs" / "data.yaml")
    p.add_argument("--profile", type=str, default="accurate", choices=("accurate", "fast", "balanced"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None, help='GPU id (0) or "cpu"')
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-onnx", action="store_true", help="Export best.pt to ONNX after train")
    p.add_argument("--half", action="store_true", help="fp16 ONNX export")
    p.add_argument("--dry-run", action="store_true", help="Validate config only (no ultralytics)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_detector_config(args.config)
    try:
        resolved = resolve_profile(cfg, args.profile)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.epochs is not None:
        resolved["epochs"] = args.epochs
    if args.imgsz is not None:
        resolved["imgsz"] = args.imgsz
    if args.batch is not None:
        resolved["batch"] = args.batch
    if args.device is not None:
        # keep int if numeric
        resolved["device"] = int(args.device) if str(args.device).isdigit() else args.device
    if args.seed is not None:
        resolved["seed"] = args.seed
    if args.resume:
        resolved["resume"] = True

    names = assert_class_list_matches_geometry()
    if not args.data.is_file():
        print(
            f"error: {args.data} missing. Run: python scripts/prepare_detector_split.py",
            file=sys.stderr,
        )
        return 2

    data_text = args.data.read_text(encoding="utf-8")
    for name in names:
        if name not in data_text:
            print(f"error: class {name!r} missing from {args.data}", file=sys.stderr)
            return 2

    print(f"profile={args.profile}  imgsz={resolved['imgsz']}  batch={resolved['batch']}  "
          f"epochs={resolved['epochs']}  classes={class_names()}")
    if args.dry_run:
        print("dry-run OK (ultralytics not invoked)")
        return 0

    best = train_yolo(args.data, resolved, dry_run=False)
    print(f"weights → {best}")
    if args.export_onnx and best is not None:
        from src.detection.train import export_onnx

        onnx_path = export_onnx(
            Path(best) if Path(str(best)).suffix == ".pt" else Path(best) / "weights" / "best.pt",
            imgsz=int(resolved["imgsz"]),
            half=args.half,
            opset=int(resolved.get("opset", 12)),
            simplify=bool(resolved.get("simplify", True)),
            dynamic=bool(resolved.get("dynamic", False)),
        )
        print(f"onnx → {onnx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
