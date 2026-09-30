"""PARSeq STR backend (ONNX). Weights optional: weights/ocr/parseq.onnx.

Export note (torch): before `model.export()`, cast the causal mask from bool
to float — a known ONNX issue in baudm/parseq. Runtime here is onnxruntime only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ocr.charset import CHARSET
from .base import BackendUnavailable

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "weights" / "ocr" / "parseq.onnx"

# PARSeq paper default charset (94 printable ASCII, no space)
_PARSEQ_CHARSET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class ParseqOnnxBackend:
    name = "parseq"

    def __init__(self, weights: Path | None = None, charset: str = CHARSET) -> None:
        self.weights = Path(weights) if weights else DEFAULT_WEIGHTS
        self.charset = charset
        self._session = None
        self._input_name = None
        self._allowed = set(charset)

    def _session_or_raise(self):
        if self._session is not None:
            return self._session
        if not self.weights.is_file():
            raise BackendUnavailable(
                f"PARSeq ONNX not found at {self.weights}. "
                "Export baudm/parseq with bool→float mask, then place the file there."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("onnxruntime is not installed") from exc
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.weights), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        return self._session

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else crop
        img = cv2.resize(img, (128, 32), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        # Match the official PARSeq training transform.
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    def recognize_detailed(self, crop: np.ndarray) -> tuple[str, list[float]]:
        sess = self._session_or_raise()
        x = self._preprocess(crop)
        outputs = sess.run(None, {self._input_name: x})
        logits = outputs[0]
        if logits.ndim == 3:
            # [B, T, C]
            probs = _softmax(logits[0], axis=-1)
        else:
            raise BackendUnavailable(f"unexpected PARSeq output shape {logits.shape}")
        # index 0 is usually EOS / blank in parseq
        ids = probs.argmax(axis=-1)
        chars = []
        scores = []
        for t, idx in enumerate(ids):
            idx = int(idx)
            if idx <= 0:
                break
            # map onto parseq charset then keep GRZ subset
            pidx = idx - 1
            if pidx >= len(_PARSEQ_CHARSET):
                continue
            ch = _PARSEQ_CHARSET[pidx]
            if ch not in self._allowed:
                if ch.upper() in self._allowed:
                    ch = ch.upper()
                else:
                    ch = "#"
            chars.append(ch)
            scores.append(float(probs[t, idx]))
        text = "".join(chars)
        return text, scores or ([0.0] if text else [])

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        text, scores = self.recognize_detailed(crop)
        return text, float(min(scores)) if scores else 0.0
