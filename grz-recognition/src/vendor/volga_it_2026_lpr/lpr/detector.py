"""Plate detector (YOLO11-pose, 4 corner keypoints) and vehicle gate (COCO YOLO11n)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

CLASSES = ["type1", "type1a", "type1b", "other"]
VEHICLE_COCO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class PlateDet:
    box: np.ndarray          # x0,y0,x1,y1
    quad: np.ndarray         # 4x2, TL,TR,BR,BL
    cls: int
    score: float
    cls_probs: Optional[np.ndarray] = None


def _order(q: np.ndarray) -> np.ndarray:
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


class PlateDetector:
    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, half: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half and device.startswith("cuda")

    def __call__(self, img_bgr: np.ndarray) -> List[PlateDet]:
        kw = {"half": True} if self.half else {}
        res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
                                 verbose=False, max_det=50, agnostic_nms=True, **kw)[0]
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        sc = res.boxes.conf.cpu().numpy()
        kps = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        h, w = img_bgr.shape[:2]
        for i in range(len(boxes)):
            b = boxes[i]
            if kps is not None and kps.shape[1] == 4 and np.all(np.isfinite(kps[i])) and (kps[i] > 0).any():
                q = _order(kps[i].astype(np.float32))
                # sanity: keypoints must lie near the box, otherwise fall back to the box corners
                bw, bh = b[2] - b[0], b[3] - b[1]
                if (q[:, 0].min() < b[0] - 0.3 * bw or q[:, 0].max() > b[2] + 0.3 * bw or
                        q[:, 1].min() < b[1] - 0.5 * bh or q[:, 1].max() > b[3] + 0.5 * bh):
                    q = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
            else:
                q = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            out.append(PlateDet(box=b.astype(np.float32), quad=q, cls=int(cls[i]), score=float(sc[i])))
        return out


class VehicleDetector:
    """COCO-pretrained YOLO11n used only to check that a plate sits on a vehicle."""

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640, conf: float = 0.25, half: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.half = half and device.startswith("cuda")

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        kw = {"half": True} if self.half else {}
        res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                 verbose=False, classes=list(VEHICLE_COCO), max_det=100, **kw)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), np.float32)
        return res.boxes.xyxy.cpu().numpy().astype(np.float32)


def plate_on_vehicle(box: np.ndarray, vehicles: np.ndarray, img_hw, min_inside: float = 0.7) -> bool:
    """True if >= min_inside of the plate box lies inside some vehicle box (slightly dilated)."""
    if len(vehicles) == 0:
        return False
    x0, y0, x1, y1 = box
    area = max(1.0, (x1 - x0) * (y1 - y0))
    for v in vehicles:
        vw, vh = v[2] - v[0], v[3] - v[1]
        vx0, vy0, vx1, vy1 = v[0] - 0.05 * vw, v[1] - 0.05 * vh, v[2] + 0.05 * vw, v[3] + 0.08 * vh
        ix = max(0.0, min(x1, vx1) - max(x0, vx0))
        iy = max(0.0, min(y1, vy1) - max(y0, vy0))
        if ix * iy / area >= min_inside:
            return True
    return False
