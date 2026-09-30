"""Detector class ids are normalized to the project's geometry registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.infer import DetectorUnavailable, _parse_model_class_names, _project_class_remap


def test_parse_ultralytics_onnx_names_metadata():
    raw = "{0: 'other', 1: 'type1', 2: 'type1a', 3: 'type1b'}"
    assert _parse_model_class_names(raw) == ["other", "type1", "type1a", "type1b"]


def test_roboflow_ids_are_remapped_to_project_ids():
    model_names = ["other", "type1", "type1a", "type1b"]
    # Project order is type1=0, type1a=1, type1b=2, other=3.
    assert _project_class_remap(model_names) == [3, 0, 1, 2]


def test_existing_project_order_remains_identity_mapping():
    assert _project_class_remap(["type1", "type1a", "type1b", "other"]) == [0, 1, 2, 3]


def test_incompatible_model_classes_are_rejected():
    with pytest.raises(DetectorUnavailable, match="do not match"):
        _project_class_remap(["plate"])
