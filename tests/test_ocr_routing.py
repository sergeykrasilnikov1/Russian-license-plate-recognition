"""Type-aware OCR routing remains optional and preserves the default path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.pipeline.infer as pipeline_infer
from src.pipeline.infer import GrzPipeline


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def recognize_detailed(self, crop):
        self.calls += 1
        text = "A123BC77" if self.name == "ppocrv4_mobile" else "B456EK777"
        return text, [0.9] * len(text)

    def recognize(self, crop):
        text, scores = self.recognize_detailed(crop)
        return text, min(scores)


def _cfg() -> dict:
    return {
        "active": "ppocrv5_mobile",
        "active_by_plate_type": {
            "type1": "ppocrv4_mobile",
            "type1a": "ppocrv5_mobile",
            "type1b": "ppocrv4_mobile",
        },
        "min_char_conf": 0.35,
        "warp_scale": 0.8,
    }


def test_routes_by_detector_plate_type_and_falls_back(monkeypatch):
    created: dict[str, _FakeBackend] = {}

    def fake_create(name, parseq_weights=None, rec_weights=None):
        created[name] = _FakeBackend(name)
        return created[name]

    monkeypatch.setattr(pipeline_infer, "create_backend", fake_create)
    pipe = GrzPipeline({}, _cfg(), require_detector=False)
    crop = np.zeros((32, 160, 3), dtype=np.uint8)

    assert pipe._ocr_crop(crop, "type1")[0] == "A123BC77"
    assert pipe._ocr_crop(crop, "TYPE1A")[0] == "B456EK777"
    assert pipe._ocr_crop(crop, "unknown")[0] == "B456EK777"
    assert created["ppocrv4_mobile"].calls == 1
    assert created["ppocrv5_mobile"].calls == 2


def test_injected_backend_disables_configured_routing():
    injected = _FakeBackend("injected")
    pipe = GrzPipeline({}, _cfg(), require_detector=False, ocr=injected)
    crop = np.zeros((32, 160, 3), dtype=np.uint8)

    pipe._ocr_crop(crop, "type1")
    pipe._ocr_crop(crop, "type1a")
    assert injected.calls == 2
    assert pipe.ocr_names_by_plate_type == {}


@pytest.mark.parametrize("routes", [[], {"": "ppocrv4_mobile"}, {"type1": ""}])
def test_rejects_invalid_routing_config(routes):
    cfg = _cfg()
    cfg["active_by_plate_type"] = routes
    with pytest.raises(ValueError):
        GrzPipeline({}, cfg, require_detector=False)
