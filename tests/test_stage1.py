"""Minimal smoke tests for Stage 1 scaffolding."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.geometry import CLASS_NAME_TO_ID, PLATE_TYPES, aspect_ratio
from src.utils.plate_mask import normalize_plate, validate_plate


def test_validate_ok():
    ok, p = validate_plate("a123bc77")
    assert ok and p == "A123BC77"
    ok, p = validate_plate("A123BC777")
    assert ok and p == "A123BC777"


def test_validate_cyrillic():
    ok, p = validate_plate("А123ВС777")
    assert ok and p == "A123BC777"


def test_validate_hash():
    ok, p = validate_plate("#246#C73")
    assert ok and p == "#246#C73"


def test_validate_bad_region():
    ok, _ = validate_plate("A123BC999")  # 3-digit must start 1/2/7
    assert not ok


def test_normalize():
    assert normalize_plate("a 123-bc 77") == "A123BC77"


def test_geometry_registry():
    assert set(PLATE_TYPES) >= {"type1", "type1a", "type1b", "other"}
    assert abs(aspect_ratio("type1") - 520 / 112) < 1e-6
    assert CLASS_NAME_TO_ID["type1a"] == 1


def test_dataset_layout():
    ds = ROOT / "dataset"
    assert (ds / "meta.csv").exists()
    assert (ds / "LICENSE").exists()
    assert (ds / "README.md").exists()
    for sub in ("images/real", "images/synthetic", "labels", "generator"):
        assert (ds / sub).is_dir()


def test_configs_exist():
    for name in ("detector.yaml", "ocr.yaml", "pipeline.yaml"):
        assert (ROOT / "configs" / name).is_file()
