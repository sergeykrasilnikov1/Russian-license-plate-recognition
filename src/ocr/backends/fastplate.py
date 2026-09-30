"""Offline adapter for FastPlateOCR's pretrained plate-specific CCT models.

Provision the ONNX and its matching plate config on the evaluation server.
Never pass a hub model name at inference: that would permit downloads.
"""

from pathlib import Path

import cv2
import numpy as np

from .base import BackendUnavailable

ROOT = Path(__file__).resolve().parents[3]


class FastPlateBackend:
    name = "fastplate"

    def __init__(self, weights: Path | None = None) -> None:
        self.weights = Path(weights) if weights else ROOT / "weights/ocr/fastplate.onnx"
        self.config_path = self.weights.with_suffix(".yaml")
        self._recognizer = None

    def _load(self):
        if self._recognizer is None:
            for path in (self.weights, self.config_path):
                if not path.is_file():
                    raise BackendUnavailable(f"FastPlateOCR local artifact missing: {path}")
            try:
                from fast_plate_ocr import LicensePlateRecognizer
            except ImportError as exc:
                raise BackendUnavailable("fast-plate-ocr is required on the inference server") from exc
            self._recognizer = LicensePlateRecognizer(
                onnx_model_path=self.weights,
                plate_config_path=self.config_path,
                device="cpu",
            )
        return self._recognizer

    def recognize_detailed(self, crop: np.ndarray) -> tuple[str, list[float]]:
        if crop.ndim != 3 or crop.shape[2] != 3 or not crop.size:
            raise ValueError("expected a nonempty HxWx3 BGR crop")
        recognizer = self._load()
        # The upstream ndarray interface expects the model's color mode.
        mode = recognizer.config.image_color_mode
        if mode == "grayscale":
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif mode == "rgb":
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            raise BackendUnavailable(f"unsupported FastPlateOCR color mode: {mode}")
        prediction = recognizer.run(image, return_confidence=True)[0]
        text = prediction.plate
        if prediction.char_probs is None:
            raise BackendUnavailable("FastPlateOCR did not return character confidences")
        scores = np.asarray(prediction.char_probs, dtype=float).reshape(-1).tolist()
        # Some releases retain probabilities for trailing padding slots.
        scores = scores[:len(text)]
        if len(scores) != len(text) or not all(np.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise BackendUnavailable("FastPlateOCR returned invalid character confidences")
        return text, scores

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        text, scores = self.recognize_detailed(crop)
        return text, min(scores) if scores else 0.0
