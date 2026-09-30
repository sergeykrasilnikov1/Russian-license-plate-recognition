"""PP-OCR recognition from a vendored ONNX file (no RapidOCR, no downloads)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .base import BackendUnavailable

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V4 = ROOT / "weights" / "ocr" / "en_PP-OCRv4_rec_mobile.onnx"
DEFAULT_V5 = ROOT / "weights" / "ocr" / "en_PP-OCRv5_rec_mobile.onnx"
REC_SHAPE = (3, 48, 320)  # C, H, W — PP-OCR mobile rec


def _ctc_greedy(logits: np.ndarray, charset: list[str]) -> tuple[str, list[float]]:
    """logits [T, C] or [1, T, C]; charset[0] is CTC blank."""
    if logits.ndim == 3:
        logits = logits[0]
    ids = logits.argmax(axis=1)
    probs = logits.max(axis=1)
    chars: list[str] = []
    scores: list[float] = []
    prev = -1
    for idx, p in zip(ids, probs):
        idx = int(idx)
        if idx == 0 or idx == prev:
            prev = idx
            continue
        prev = idx
        if 0 < idx < len(charset):
            ch = charset[idx]
            if ch == "blank" or ch.isspace() or ch == "-":
                continue
            chars.append(ch)
            scores.append(float(p))
    text = "".join(chars).replace(" ", "")
    return text, scores or ([0.0] if text else [])


def _resize_norm(img_bgr: np.ndarray, rec_shape: tuple[int, int, int] = REC_SHAPE) -> np.ndarray:
    img_c, img_h, img_w = rec_shape
    if img_bgr.ndim != 3:
        raise ValueError("expected HxWxC crop")
    h, w = img_bgr.shape[:2]
    max_wh = img_w / float(img_h)
    max_wh = max(max_wh, w / float(h))
    img_width = int(img_h * max_wh)
    ratio = w / float(h)
    resized_w = img_width if math.ceil(img_h * ratio) > img_width else int(math.ceil(img_h * ratio))
    resized = cv2.resize(img_bgr, (resized_w, img_h)).astype(np.float32)
    resized = resized.transpose((2, 0, 1)) / 255.0
    resized = (resized - 0.5) / 0.5
    canvas = np.zeros((img_c, img_h, img_width), dtype=np.float32)
    canvas[:, :, :resized_w] = resized
    return canvas[None, ...]


class LocalPPOCRBackend:
    """ONNX Runtime rec-only. Weights must already sit under weights/ocr/."""

    def __init__(self, name: str, weights: Path) -> None:
        self.name = name
        self.weights = Path(weights)
        self._session = None
        self._input_name = None
        self._charset: list[str] = []

    def _session_or_raise(self):
        if self._session is not None:
            return self._session
        if not self.weights.is_file():
            raise BackendUnavailable(
                f"OCR ONNX missing: {self.weights}. Vendor the RapidOCR rec model into weights/ocr/ "
                "(setup only — inference must stay offline)."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("onnxruntime is not installed") from exc
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self._session = ort.InferenceSession(str(self.weights), sess_options=so, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        raw = self._session.get_modelmeta().custom_metadata_map.get("character", "")
        chars = [ln for ln in raw.splitlines() if ln != ""]
        # CTC: index 0 = blank, then the ONNX character list (already ends with space).
        self._charset = ["blank"] + chars
        return self._session

    def recognize_detailed(self, crop: np.ndarray) -> tuple[str, list[float]]:
        sess = self._session_or_raise()
        x = _resize_norm(crop)
        logits = sess.run(None, {self._input_name: x})[0]
        return _ctc_greedy(logits, self._charset)

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        text, scores = self.recognize_detailed(crop)
        return text, float(min(scores)) if scores else 0.0

    def rec_logits_fingerprint(self, crop: np.ndarray) -> dict:
        import hashlib

        sess = self._session_or_raise()
        x = _resize_norm(crop)
        preds = sess.run(None, {self._input_name: x})[0]
        return {
            "backend": self.name,
            "graph": sess.get_modelmeta().graph_name,
            "shape": tuple(int(v) for v in preds.shape),
            "sha256_16": hashlib.sha256(preds.tobytes()).hexdigest()[:16],
        }


class PPOCRV4Backend(LocalPPOCRBackend):
    def __init__(self, weights: Path | None = None) -> None:
        super().__init__("ppocrv4_mobile", Path(weights) if weights else DEFAULT_V4)


class PPOCRV5Backend(LocalPPOCRBackend):
    def __init__(self, weights: Path | None = None) -> None:
        super().__init__("ppocrv5_mobile", Path(weights) if weights else DEFAULT_V5)
