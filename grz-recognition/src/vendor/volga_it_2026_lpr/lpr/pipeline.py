"""End-to-end pipeline: image -> list of (plate_num, plate_type, confidence, ...)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detector import CLASSES, PlateDetector, VehicleDetector, plate_on_vehicle
from .ocr import Recognizer, apply_unknown
from .plates import UNK, fix_lookalikes
from .typing import decide_type

# canonical rectified sizes (aspect of the physical plates)
RECT_1ROW = (208, 48)      # 520x112 -> 4.64:1
RECT_2ROW = (192, 112)     # 290x170 -> 1.71:1


@dataclass
class PlateResult:
    plate_num: str
    plate_type: str
    confidence: float
    box: List[float]
    quad: List[float]
    det_score: float
    ocr_score: float
    is_vehicle: int
    raw_text: str = ""
    timings: Dict[str, float] = field(default_factory=dict)


def rectify(img: np.ndarray, quad: np.ndarray, size, margin: float = 0.03) -> np.ndarray:
    """Perspective-warp the quad to a canonical rectangle, with a small margin so clipped corners survive."""
    w, h = size
    q = quad.astype(np.float32).copy()
    c = q.mean(axis=0)
    q = c + (q - c) * (1.0 + margin)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def split_rows(rect: np.ndarray):
    h = rect.shape[0]
    return rect[int(h * 0.06): int(h * 0.54)], rect[int(h * 0.48): int(h * 0.97)]


class LPRPipeline:
    def __init__(self, det_weights: str, ocr_weights: str, vehicle_weights: Optional[str] = None, device: str = "cpu",
                 det_imgsz: int = 640, det_conf: float = 0.25, unk_thr: float = 0.45, min_conf: float = 0.15,
                 vehicle_gate: bool = True, emit_other: bool = True, half: bool = False, veh_imgsz: int = 416,
                 unk_drop_frac: float = 0.34, unk_drop_conf: float = 0.5):
        self.det = PlateDetector(det_weights, device=device, imgsz=det_imgsz, conf=det_conf, half=half)
        self.ocr = Recognizer(ocr_weights, device=device, unk_thr=unk_thr, fp16=half)
        # vehicles are large objects: a smaller input is enough for the gate and ~2x faster
        self.veh = VehicleDetector(vehicle_weights, device=device, imgsz=veh_imgsz, half=half) if (vehicle_gate and vehicle_weights) else None
        self.unk_thr = unk_thr
        self.min_conf = min_conf
        self.emit_other = emit_other
        self.device = device
        self.unk_drop_frac = unk_drop_frac      # a reading with >= this fraction of '#' ...
        self.unk_drop_conf = unk_drop_conf      # ... is dropped unless the confidence is at least this

    def warmup(self):
        dummy = np.full((480, 640, 3), 128, np.uint8)
        for _ in range(2):
            self.process(dummy)

    def process(self, img: np.ndarray) -> List[PlateResult]:
        t0 = time.perf_counter()
        dets = self.det(img)
        t1 = time.perf_counter()
        vehicles = self.veh(img) if (self.veh is not None and dets) else None
        t2 = time.perf_counter()
        results: List[PlateResult] = []
        # every detection is read as one row; geometrically square ones are additionally read as two rows
        rows, owners, rects = [], [], {}
        for i, d in enumerate(dets):
            q = d.quad
            w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            rect1 = rectify(img, q, RECT_1ROW)
            rects[i] = rect1
            rows.append(rect1)
            owners.append((i, "one"))
            if w / max(h, 1e-3) < 2.4 or CLASSES[d.cls] == "type1a":
                rect2 = rectify(img, q, RECT_2ROW)
                top, bot = split_rows(rect2)
                rows += [top, bot]
                owners += [(i, "top"), (i, "bot")]
        decoded = self.ocr.read_rows(rows)
        t3 = time.perf_counter()
        per: Dict[int, Dict[str, tuple]] = {}
        for (i, part), dec in zip(owners, decoded):
            per.setdefault(i, {})[part] = dec
        for i, d in enumerate(dets):
            if i not in per:
                continue
            one = per[i]["one"]
            text1, score1 = apply_unknown(one[0], one[1], self.unk_thr)
            rows_text = None
            text2, score2 = "", 0.0
            if "top" in per[i]:
                top, bot = per[i]["top"], per[i]["bot"]
                t_top, s_top = apply_unknown(top[0], top[1], self.unk_thr)
                t_bot, s_bot = apply_unknown(bot[0], bot[1], self.unk_thr)
                rows_text = (fix_lookalikes_rows(t_top, t_bot))
                text2, score2 = rows_text[0] + rows_text[1], min(s_top, s_bot) if (s_top and s_bot) else 0.0
            det_cls = CLASSES[d.cls]
            # the single-row reading (with type1 look-alike correction) is the evidence for the type decision;
            # the two-row reading is used when the plate turns out to be a square type 1A
            text1_std = fix_lookalikes(text1, "type1")
            ptype, tconf = decide_type(det_cls, d.score, d.quad, rects[i], text1_std, score1, rows_text)
            if ptype == "type1a":
                text, ocr_score = text2, score2
            elif ptype == "type1b":
                text, ocr_score = fix_lookalikes(text1, "type1b"), score1
            elif ptype == "other":
                text, ocr_score = text1, score1          # no Russian-mask correction for foreign/other plates
            else:
                text, ocr_score = text1_std, score1
            if not text:
                text = UNK * (8 if ptype != "other" else 1)
            if ptype == "other" and not self.emit_other:
                continue
            # an 'other' candidate that reads as almost nothing and is not a confident detection is noise
            if ptype == "other" and d.score < 0.5 and (not text.strip("#") or text.count(UNK) >= 0.6 * len(text)):
                continue
            conf = float(d.score) * (0.5 + 0.5 * ocr_score) * (0.7 + 0.3 * tconf)
            on_vehicle = 1 if (vehicles is None or plate_on_vehicle(d.box, vehicles, img.shape[:2])) else 0
            if conf < self.min_conf:
                continue
            if text.count(UNK) >= self.unk_drop_frac * max(1, len(text)) and conf < self.unk_drop_conf:
                continue
            results.append(PlateResult(plate_num=text, plate_type=ptype, confidence=round(min(1.0, conf), 4),
                                       box=[float(v) for v in d.box], quad=[float(v) for v in d.quad.reshape(-1)],
                                       det_score=float(d.score), ocr_score=float(ocr_score), is_vehicle=on_vehicle, raw_text=one[0],
                                       timings=dict(det=t1 - t0, veh=t2 - t1, ocr=t3 - t2)))
        return results


def fix_lookalikes_rows(top: str, bot: str):
    """Position-wise look-alike correction for the two rows of a square plate (L DDD / LL RR)."""
    d2l = {"0": "O", "8": "B", "4": "A", "3": "E", "6": "B", "1": "T", "7": "T", "5": "C", "2": "E", "9": "P"}
    l2d = {"O": "0", "B": "8", "A": "4", "E": "3", "T": "7", "C": "0", "P": "9", "H": "4", "K": "4", "M": "4", "X": "4", "Y": "4"}
    def fix(txt, mask):
        out = []
        for ch, m in zip(txt, mask):
            if ch == UNK:
                out.append(ch)
            elif m == "L" and ch.isdigit():
                out.append(d2l.get(ch, UNK))
            elif m == "D" and ch.isalpha():
                out.append(l2d.get(ch, UNK))
            else:
                out.append(ch)
        return "".join(out) + txt[len(mask):]
    top = fix(top, "LDDD") if len(top) == 4 else top
    bot = fix(bot, "LL" + "D" * (len(bot) - 2)) if 4 <= len(bot) <= 5 else bot
    return top, bot


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
