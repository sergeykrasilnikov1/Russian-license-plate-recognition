"""Plate detector inference: Ultralytics .pt if present, else ONNX Runtime."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.geometry import CLASS_NAME_TO_ID, class_names


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    plate_type: str

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.xyxy
        return x0, y0, x1 - x0, y1 - y0


class DetectorUnavailable(RuntimeError):
    pass


def _parse_model_class_names(raw: Any) -> list[str] | None:
    """Parse Ultralytics ONNX ``names`` metadata into model class-id order."""
    if raw is None or raw == "":
        return None
    value = raw
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if isinstance(value, dict):
        try:
            indexed = {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError):
            return None
        if sorted(indexed) != list(range(len(indexed))):
            return None
        return [indexed[index] for index in range(len(indexed))]
    if isinstance(value, (list, tuple)):
        return [str(name) for name in value]
    return None


def _project_class_remap(model_names: list[str]) -> list[int]:
    """Return model class-id -> project class-id and reject incompatible models."""
    project_names = class_names()
    if len(model_names) != len(project_names) or set(model_names) != set(project_names):
        raise DetectorUnavailable(
            f"model classes {model_names} do not match project classes {project_names}"
        )
    return [CLASS_NAME_TO_ID[name] for name in model_names]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0).clip(min=0) * (y1 - y0).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        inter = (xx1 - xx0).clip(min=0) * (yy1 - yy0).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def letterbox(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_w, pad_h = (imgsz - nw) / 2, (imgsz - nh) / 2
    top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


class OnnxYoloDetector:
    def __init__(
        self,
        weights: Path,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 10,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise DetectorUnavailable("onnxruntime is not installed") from exc
        if not Path(weights).is_file():
            raise DetectorUnavailable(f"missing ONNX weights {weights}")
        available = ort.get_available_providers()
        providers = [p for p in (providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]) if p in available]
        if not providers:
            raise DetectorUnavailable("None of the requested ONNX providers are available")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(weights), sess_options=so, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        if any(isinstance(v, int) and v != imgsz for v in shape[-2:]):
            raise DetectorUnavailable(f"ONNX input shape {shape} does not match imgsz={imgsz}; re-export or fix config")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        metadata = self.sess.get_modelmeta().custom_metadata_map or {}
        self.model_names = _parse_model_class_names(metadata.get("names")) or class_names()
        self.model_to_project_id = _project_class_remap(self.model_names)
        self.names = class_names()

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        canvas, scale, (pad_x, pad_y) = letterbox(image_bgr, self.imgsz)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        out = self.sess.run(None, {self.input_name: blob})[0]
        # Ultralytics: [1, 4+nc, n] or [1, n, 4+nc]
        pred = np.squeeze(out, axis=0)
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 4 + len(self.model_names) + 4:
            pred = pred.T
        # pred: [n, 4+nc] xywh + classes
        xywh = pred[:, :4]
        cls_scores = pred[:, 4 : 4 + len(self.model_names)]
        model_cls_id = cls_scores.argmax(axis=1)
        scores = cls_scores.max(axis=1)
        mask = scores >= self.conf
        xywh, scores, model_cls_id = xywh[mask], scores[mask], model_cls_id[mask]
        # xywh in letterbox pixels → xyxy original
        x, y, w, h = xywh.T
        x0 = (x - w / 2 - pad_x) / scale
        y0 = (y - h / 2 - pad_y) / scale
        x1 = (x + w / 2 - pad_x) / scale
        y1 = (y + h / 2 - pad_y) / scale
        boxes = np.stack([x0, y0, x1, y1], axis=1)
        keep = _nms(boxes, scores, self.iou)[: self.max_det]
        ih, iw = image_bgr.shape[:2]
        dets: list[Detection] = []
        for i in keep:
            a, b, c, d = [float(v) for v in boxes[i]]
            a, b = max(0.0, a), max(0.0, b)
            c, d = min(float(iw), c), min(float(ih), d)
            if not np.isfinite([a, b, c, d]).all() or c <= a or d <= b:
                continue
            model_cid = int(model_cls_id[i])
            project_cid = self.model_to_project_id[model_cid]
            dets.append(
                Detection(
                    xyxy=(a, b, c, d),
                    conf=float(scores[i]),
                    cls_id=project_cid,
                    plate_type=self.model_names[model_cid],
                )
            )
        return dets


class UltralyticsDetector:
    def __init__(self, weights: Path, *, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45, max_det: int = 10) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise DetectorUnavailable("ultralytics is not installed") from exc
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )[0]
        dets: list[Detection] = []
        if res.boxes is None:
            return dets
        names = res.names or {i: n for i, n in enumerate(class_names())}
        for b in res.boxes:
            xyxy = tuple(float(x) for x in b.xyxy[0].tolist())
            model_cid = int(b.cls[0])
            plate_type = str(names.get(model_cid, "other"))
            project_cid = CLASS_NAME_TO_ID.get(plate_type, CLASS_NAME_TO_ID["other"])
            dets.append(
                Detection(
                    xyxy=xyxy,
                    conf=float(b.conf[0]),
                    cls_id=project_cid,
                    plate_type=plate_type,
                )
            )
        return dets


def load_detector(
    *,
    pt_path: Path | None,
    onnx_path: Path | None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 10,
    providers: list[str] | None = None,
):
    if pt_path and Path(pt_path).is_file():
        try:
            return UltralyticsDetector(Path(pt_path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det)
        except DetectorUnavailable:
            pass
    if onnx_path and Path(onnx_path).is_file():
        return OnnxYoloDetector(Path(onnx_path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, providers=providers)
    raise DetectorUnavailable(
        "No runnable detector: need ultralytics+weights/detector_best.pt or weights/detector.onnx"
    )
