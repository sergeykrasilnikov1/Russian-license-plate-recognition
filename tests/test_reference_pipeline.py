"""Selected pose+CRNN runtime adapter without importing torch locally."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.reference import ReferencePoseCrnnPipeline
from src.detection.infer import DetectorUnavailable


class _FakeEngine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.warmed = False

    def process(self, image):
        return self.outputs

    def warmup(self):
        self.warmed = True


def _result(text="A123BC77", plate_type="type1", confidence=0.87):
    return SimpleNamespace(
        plate_num=text,
        plate_type=plate_type,
        confidence=confidence,
        box=[20.0, 20.0, 180.0, 60.0],
    )


def test_adapter_maps_reference_rows_and_warmup():
    engine = _FakeEngine([_result()])
    pipeline = ReferencePoseCrnnPipeline({"vehicle_filter": {"enabled": False}}, engine=engine)
    rows = pipeline.process_image(np.full((80, 200, 3), 120, np.uint8), "x.jpg")

    assert [(row.image, row.plate_num, row.plate_type, row.confidence) for row in rows] == [
        ("x.jpg", "A123BC77", "type1", 0.87)
    ]
    pipeline.warmup()
    assert engine.warmed


def test_adapter_applies_existing_vehicle_filter(monkeypatch):
    engine = _FakeEngine([_result()])
    pipeline = ReferencePoseCrnnPipeline({"vehicle_filter": {"enabled": True}}, engine=engine)
    monkeypatch.setattr("src.pipeline.reference.is_on_vehicle", lambda image, box, cfg: (False, 0.0))

    assert pipeline.process_image(np.zeros((80, 200, 3), np.uint8), "off.jpg") == []


@pytest.mark.parametrize("text", ["A123BC456", "AB12377", "002CD178"])
def test_adapter_rejects_non_target_masks(text):
    pipeline = ReferencePoseCrnnPipeline(
        {"vehicle_filter": {"enabled": False}}, engine=_FakeEngine([_result(text)])
    )
    assert pipeline.process_image(np.zeros((80, 200, 3), np.uint8), "x.jpg")[0].plate_type == "other"


@pytest.mark.parametrize("confidence, expected", [(float("nan"), 0), (float("inf"), 0), (-1, 0), (2, 1)])
def test_adapter_confidence_is_finite_and_bounded(confidence, expected):
    pipeline = ReferencePoseCrnnPipeline(
        {"vehicle_filter": {"enabled": False}}, engine=_FakeEngine([_result(confidence=confidence)])
    )
    assert pipeline.process_image(np.zeros((80, 200, 3), np.uint8), "x.jpg")[0].confidence == expected


def test_adapter_reports_lazy_runtime_dependency_failure(monkeypatch):
    module_name = "src.vendor.volga_it_2026_lpr.lpr.pipeline"
    fake_module = ModuleType(module_name)

    class _MissingRuntime:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("No module named 'ultralytics'")

    fake_module.LPRPipeline = _MissingRuntime
    fake_module.pick_device = lambda preference: "cpu"
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    cfg = {
        "reference_pose_crnn": {
            "detector_weights": "weights/reference/det.pt",
            "ocr_weights": "weights/reference/ocr.pt",
        }
    }
    with pytest.raises(DetectorUnavailable, match="requires torch and ultralytics"):
        ReferencePoseCrnnPipeline(cfg)
