"""Offline detect → vehicle filter → warp → OCR → CSV rows."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import logging
from pathlib import Path

import numpy as np
import yaml

from src.detection.infer import DetectorUnavailable, load_detector
from src.detection.vehicle import VehicleFilterConfig, is_on_vehicle
from src.ocr.backends import create_backend
from src.ocr.charset import finalize_plate
from src.ocr.warp import DegenerateQuadError, parse_quad, warp_plate, xyxy_to_quad
from src.utils.plate_mask import looks_like_special_plate

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PlateResult:
    image: str
    plate_num: str
    plate_type: str
    confidence: float


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def aggregate_confidence(det_conf: float, ocr_conf: float, mode: str) -> float:
    """Combine detector bbox score with OCR confidence (default: conservative min)."""
    det_conf = float(np.clip(det_conf, 0.0, 1.0)) if np.isfinite(det_conf) else 0.0
    ocr_conf = float(np.clip(ocr_conf, 0.0, 1.0)) if np.isfinite(ocr_conf) else 0.0
    if mode in ("min", "det_ocr_min"):
        return min(det_conf, ocr_conf)
    if mode in ("mean", "det_ocr_mean"):
        return 0.5 * (det_conf + ocr_conf)
    return min(det_conf, ocr_conf)


def write_results_csv(rows: list[PlateResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";", lineterminator="\n")
        writer.writerow(["image", "plate_num", "plate_type", "confidence"])
        for r in rows:
            writer.writerow([r.image, r.plate_num, r.plate_type, f"{r.confidence:.4f}"])


class GrzPipeline:
    def __init__(
        self,
        pipeline_cfg: dict,
        ocr_cfg: dict,
        *,
        prefer_onnx: bool = False,
        require_detector: bool = True,
        detector=None,
        ocr=None,
    ) -> None:
        self.pipeline_cfg = pipeline_cfg
        self.ocr_cfg = ocr_cfg
        det_yaml = ROOT / pipeline_cfg.get("detector", {}).get("config", "configs/detector.yaml")
        det_cfg = load_yaml(det_yaml) if det_yaml.is_file() else {}
        self.imgsz = int(det_cfg.get("imgsz", 640))
        self.conf = float(det_cfg.get("conf", 0.25))
        self.iou = float(det_cfg.get("iou", 0.45))
        self.max_det = int(det_cfg.get("max_det", 10))
        vf = pipeline_cfg.get("vehicle_filter") or {}
        self.vehicle_enabled = bool(vf.get("enabled", True))
        self.vehicle_cfg = VehicleFilterConfig(
            context_pad=float(vf.get("context_pad", 0.5)),
            min_context_score=float(vf.get("min_context_score", 0.3)),
            min_context_evidence=float(vf.get("min_context_evidence", 0.1)),
        )
        self.confidence_mode = str(ocr_cfg.get("confidence_mode", "det_ocr_min"))
        self.min_char_conf = float(ocr_cfg.get("min_char_conf", 0.35))
        self.warp_scale = float(ocr_cfg.get("warp_scale", 0.8))
        if not np.isfinite(self.warp_scale) or self.warp_scale <= 0:
            raise ValueError("warp_scale must be positive and finite")
        self.type1a_mode = str(ocr_cfg.get("type1a_mode", "flatten"))
        self.ocr_name = str(ocr_cfg.get("active", "ppocrv4_mobile"))
        raw_routes = ocr_cfg.get("active_by_plate_type", {})
        if raw_routes is None:
            raw_routes = {}
        if not isinstance(raw_routes, dict):
            raise ValueError("active_by_plate_type must be a mapping of plate type to OCR backend")
        self.ocr_names_by_plate_type = {
            str(plate_type).strip().lower(): str(backend).strip().lower()
            for plate_type, backend in raw_routes.items()
        }
        if any(not plate_type or not backend for plate_type, backend in self.ocr_names_by_plate_type.items()):
            raise ValueError("active_by_plate_type keys and backend names must be nonempty")

        # A supplied test/application backend remains authoritative for every
        # plate type. Configured routing is enabled only when this class owns
        # backend construction.
        if ocr is not None:
            self.ocr_names_by_plate_type = {}
            self._ocr_backends = {self.ocr_name: ocr}
        else:
            names = {self.ocr_name, *self.ocr_names_by_plate_type.values()}
            self._ocr_backends = {name: self._create_ocr_backend(name) for name in sorted(names)}
        # Backwards-compatible handle used by callers that inspect the active
        # default backend directly.
        self.ocr = self._ocr_backends[self.ocr_name]
        if detector is not None:
            self.detector = detector
        elif require_detector:
            onnx = ROOT / str(pipeline_cfg.get("detector", {}).get("weights", "weights/detector.onnx"))
            pt_name = pipeline_cfg.get("detector", {}).get("pt_weights", "weights/detector_best.pt")
            pt = ROOT / pt_name
            if prefer_onnx:
                self.detector = load_detector(
                    pt_path=None, onnx_path=onnx, imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det, providers=pipeline_cfg.get("detector", {}).get("providers")
                )
            else:
                self.detector = load_detector(
                    pt_path=pt, onnx_path=onnx, imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det, providers=pipeline_cfg.get("detector", {}).get("providers")
                )
        else:
            self.detector = None

    def _create_ocr_backend(self, name: str):
        parseq_w = self.ocr_cfg.get("parseq_weights")
        rec_w = self.ocr_cfg.get("rec_weights_by_backend", {}).get(name)
        if rec_w is None and name == "ppocrv4_mobile":
            rec_w = self.ocr_cfg.get("rec_weights")
        return create_backend(
            name,
            ROOT / parseq_w if parseq_w else None,
            ROOT / rec_w if rec_w else None,
        )

    def _ocr_for_plate_type(self, plate_type: str | None):
        normalized_type = str(plate_type or "").strip().lower()
        name = self.ocr_names_by_plate_type.get(normalized_type, self.ocr_name)
        return self._ocr_backends[name]

    def _ocr_crop(self, crop: np.ndarray, plate_type: str | None = None) -> tuple[str, list[float]]:
        backend = self._ocr_for_plate_type(plate_type)
        detailed = getattr(backend, "recognize_detailed", None)
        if detailed:
            return detailed(crop)
        raw, conf_one = backend.recognize(crop)
        return raw, [conf_one] * max(1, len(raw))

    def process_image(self, image_bgr: np.ndarray, image_name: str) -> list[PlateResult]:
        if self.detector is None:
            raise DetectorUnavailable("pipeline has no detector; use process_with_gt_quad or --from-meta")
        dets = self.detector.predict(image_bgr)
        rows: list[PlateResult] = []
        for det in dets:
            if self.vehicle_enabled:
                ok, _ = is_on_vehicle(image_bgr, det.xyxy, self.vehicle_cfg)
                if not ok:
                    continue
            quad = xyxy_to_quad(det.xyxy)
            try:
                crop = warp_plate(image_bgr, quad, det.plate_type, type1a_mode=self.type1a_mode, scale=self.warp_scale)
            except DegenerateQuadError:
                logging.getLogger(__name__).warning("Skipping invalid detection in %s: %s", image_name, det.xyxy)
                continue
            raw, char_scores = self._ocr_crop(crop, det.plate_type)
            plate_type = "other" if looks_like_special_plate(raw) else det.plate_type
            plate, ocr_conf = finalize_plate(raw, char_scores, self.min_char_conf)
            conf = aggregate_confidence(det.conf, ocr_conf, self.confidence_mode)
            rows.append(
                PlateResult(image=image_name, plate_num=plate, plate_type=plate_type, confidence=round(conf, 4))
            )
        return rows

    def process_with_gt_quad(
        self,
        image_bgr: np.ndarray,
        image_name: str,
        quad,
        plate_type: str,
        det_conf: float = 1.0,
        *,
        apply_vehicle_filter: bool = False,
        bbox_xyxy: tuple[float, float, float, float] | None = None,
    ) -> PlateResult | None:
        """OCR path with a known quad (meta GT, or detector missing)."""
        if apply_vehicle_filter and bbox_xyxy is not None:
            ok, _ = is_on_vehicle(image_bgr, bbox_xyxy, self.vehicle_cfg)
            if not ok:
                return None
        crop = warp_plate(image_bgr, quad, plate_type, type1a_mode=self.type1a_mode, scale=self.warp_scale)  # type: ignore[arg-type]
        raw, char_scores = self._ocr_crop(crop, plate_type)
        if looks_like_special_plate(raw):
            plate_type = "other"
        plate, ocr_conf = finalize_plate(raw, char_scores, self.min_char_conf)
        conf = aggregate_confidence(det_conf, ocr_conf, self.confidence_mode)
        return PlateResult(image=image_name, plate_num=plate, plate_type=plate_type, confidence=round(conf, 4))


def try_load_pipeline(pipeline_yaml: Path, ocr_yaml: Path, *, require_detector: bool = True) -> GrzPipeline | None:
    try:
        return GrzPipeline(load_yaml(pipeline_yaml), load_yaml(ocr_yaml), require_detector=require_detector)
    except Exception:
        logging.getLogger(__name__).exception("Could not load inference pipeline")
        return None


def iter_images(folder: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return files
