"""Adapter for the measured Lin-Lini YOLO-pose + CRNN reference runtime."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from src.detection.infer import DetectorUnavailable
from src.detection.vehicle import VehicleFilterConfig, is_on_vehicle
from src.utils.plate_mask import normalize_plate, validate_plate

ROOT = Path(__file__).resolve().parents[2]


class ReferencePoseCrnnPipeline:
    """Offline pose-corner detector and plate-specific CRNN.

    The third-party implementation is imported lazily so the local CPU-only
    development environment can still run tests without torch/ultralytics.
    """

    def __init__(self, pipeline_cfg: dict, *, engine=None) -> None:
        self.pipeline_cfg = pipeline_cfg
        cfg = pipeline_cfg.get("reference_pose_crnn") or {}
        self.ocr_name = "reference_crnn"

        vehicle_cfg = pipeline_cfg.get("vehicle_filter") or {}
        self.vehicle_enabled = bool(vehicle_cfg.get("enabled", True))
        self.vehicle_cfg = VehicleFilterConfig(
            context_pad=float(vehicle_cfg.get("context_pad", 0.5)),
            min_context_score=float(vehicle_cfg.get("min_context_score", 0.3)),
            min_context_evidence=float(vehicle_cfg.get("min_context_evidence", 0.1)),
        )

        if engine is not None:
            self._engine = engine
            return

        detector_weights = ROOT / str(cfg.get("detector_weights", "weights/reference/det.pt"))
        ocr_weights = ROOT / str(cfg.get("ocr_weights", "weights/reference/ocr.pt"))
        missing = [str(path) for path in (detector_weights, ocr_weights) if not path.is_file()]
        if missing:
            raise DetectorUnavailable(f"missing reference pose/CRNN weights: {missing}")

        try:
            from src.vendor.volga_it_2026_lpr.lpr.pipeline import LPRPipeline, pick_device

            device = pick_device(str(cfg.get("device", "auto")))
            self._engine = LPRPipeline(
                str(detector_weights),
                str(ocr_weights),
                vehicle_weights=None,
                device=device,
                det_imgsz=int(cfg.get("imgsz", 960)),
                det_conf=float(cfg.get("det_conf", 0.2)),
                unk_thr=float(cfg.get("unk_thr", 0.45)),
                min_conf=float(cfg.get("min_conf", 0.2)),
                vehicle_gate=False,
                emit_other=bool(cfg.get("emit_other", True)),
                half=bool(cfg.get("half", False)),
                unk_drop_frac=float(cfg.get("unk_drop_frac", 0.34)),
                unk_drop_conf=float(cfg.get("unk_drop_conf", 0.5)),
            )
        except ImportError as exc:
            raise DetectorUnavailable(
                "reference_pose_crnn requires torch and ultralytics from requirements-train.txt"
            ) from exc

    def process_image(self, image_bgr: np.ndarray, image_name: str):
        # Imported here to avoid a module cycle: infer owns the public row type.
        from src.pipeline.infer import PlateResult

        rows = []
        for result in self._engine.process(image_bgr):
            if self.vehicle_enabled:
                box = tuple(float(value) for value in result.box)
                on_vehicle, _ = is_on_vehicle(image_bgr, box, self.vehicle_cfg)
                if not on_vehicle:
                    continue
            text = normalize_plate(str(result.plate_num))
            plate_type = str(result.plate_type)
            # The upstream runtime also accepts taxi-specific seven-character
            # masks and unrestricted three-digit regions. The competition uses
            # one shared mask for all three target classes.
            if plate_type not in {"type1", "type1a", "type1b", "other"}:
                plate_type = "other"
            if plate_type != "other" and not validate_plate(text)[0]:
                plate_type = "other"
            confidence = float(result.confidence)
            confidence = min(1.0, max(0.0, confidence)) if math.isfinite(confidence) else 0.0
            rows.append(
                PlateResult(
                    image=image_name,
                    plate_num=text,
                    plate_type=plate_type,
                    confidence=round(confidence, 4),
                )
            )
        return rows

    def warmup(self) -> None:
        warmup = getattr(self._engine, "warmup", None)
        if warmup is not None:
            warmup()
