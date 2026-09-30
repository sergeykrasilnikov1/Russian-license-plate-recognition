"""Ultralytics training / export helpers. Heavy deps imported lazily."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.geometry import class_names


def load_detector_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid detector config: {path}")
    return raw


def assert_class_list_matches_geometry(cfg: dict[str, Any] | None = None) -> list[str]:
    """Detector classes must come from PLATE_TYPES — reject hardcoded drift."""
    names = class_names()
    if cfg and "names" in cfg:
        # Optional sanity check if someone left a names block in yaml.
        raw = cfg["names"]
        if isinstance(raw, dict):
            listed = [raw[i] for i in sorted(raw)]
        else:
            listed = list(raw)
        if listed != names:
            raise ValueError(
                f"detector.yaml names {listed} != geometry.PLATE_TYPES {names}. "
                "Remove names from yaml; they are generated into data.yaml from geometry."
            )
    return names


def train_yolo(data_yaml: Path, cfg: dict[str, Any], *, dry_run: bool = False) -> Path | None:
    """Run ultralytics YOLO train. Returns best.pt path, or None on dry_run."""
    assert_class_list_matches_geometry()
    if dry_run:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - local CPU machine
        raise RuntimeError(
            "ultralytics is not installed. Use the GPU server: pip install -r requirements-train.txt"
        ) from exc

    model_name = cfg.get("model", "yolo11n.pt")
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=int(cfg.get("epochs", 50)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        device=cfg.get("device", 0),
        seed=int(cfg.get("seed", 42)),
        workers=int(cfg.get("workers", 4)),
        project=str(cfg.get("project", "runs/detect")),
        name=str(cfg.get("name", "grz_yolo11n")),
        exist_ok=bool(cfg.get("exist_ok", True)),
        pretrained=bool(cfg.get("pretrained", True)),
        patience=int(cfg.get("patience", 15)),
        lr0=float(cfg.get("lr0", 0.01)),
        lrf=float(cfg.get("lrf", 0.01)),
        momentum=float(cfg.get("momentum", 0.937)),
        weight_decay=float(cfg.get("weight_decay", 0.0005)),
        warmup_epochs=float(cfg.get("warmup_epochs", 3.0)),
        optimizer=str(cfg.get("optimizer", "SGD")),
        amp=bool(cfg.get("amp", True)),
        resume=bool(cfg.get("resume", False)),
    )
    # ultralytics returns a list-like; best weights under save_dir
    save_dir = Path(str(results.save_dir)) if hasattr(results, "save_dir") else Path(cfg.get("project", "runs/detect"))
    best = save_dir / "weights" / "best.pt"
    return best if best.is_file() else save_dir


def export_onnx(
    weights: Path,
    *,
    imgsz: int = 640,
    half: bool = False,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
    dry_run: bool = False,
) -> Path | None:
    if dry_run:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is not installed. Use the GPU server: pip install -r requirements-train.txt"
        ) from exc
    model = YOLO(str(weights))
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )
    return Path(str(out))
