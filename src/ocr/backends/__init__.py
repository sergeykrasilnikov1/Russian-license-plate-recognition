"""OCR backend factory."""

from __future__ import annotations

from pathlib import Path

from .base import OcrBackend, BackendUnavailable
from .parseq import ParseqOnnxBackend
from .ppocr import PPOCRV4Backend, PPOCRV5Backend
from .fastplate import FastPlateBackend

BACKENDS = {
    "fastplate": FastPlateBackend,
    "ppocrv4_mobile": PPOCRV4Backend,
    "ppocrv5_mobile": PPOCRV5Backend,
    "parseq": ParseqOnnxBackend,
}


def create_backend(
    name: str,
    parseq_weights: Path | None = None,
    rec_weights: Path | None = None,
) -> OcrBackend:
    key = name.strip().lower()
    if key not in BACKENDS:
        raise KeyError(f"unknown OCR backend {name!r}; available: {sorted(BACKENDS)}")
    cls = BACKENDS[key]
    if cls is ParseqOnnxBackend:
        return cls(weights=parseq_weights)
    if rec_weights is not None:
        return cls(weights=rec_weights)
    return cls()


def backend_status(name: str, parseq_weights: Path | None = None) -> str:
    try:
        be = create_backend(name, parseq_weights)
        w = getattr(be, "weights", None)
        if w is not None and not Path(w).is_file():
            return f"missing {w}"
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def available_backends(parseq_weights: Path | None = None) -> dict[str, str]:
    return {name: backend_status(name, parseq_weights) for name in BACKENDS}
