"""Tests for the dataset validator: it must fail on real defects, not just pass."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import validate_dataset

HEADER = "image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions\n"
GOOD_ROW = "images/real/a.jpg;A123BC77;type1;10,20,100,30;10,20,110,20,110,50,10,50;1;0;src;CC-BY-4.0;day\n"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for sub in ("images/real", "images/synthetic", "labels", "generator"):
        (root / sub).mkdir(parents=True)
    (root / "README.md").write_text("datasheet", encoding="utf-8")
    (root / "LICENSE").write_text("CC BY 4.0", encoding="utf-8")
    cv2.imwrite(str(root / "images/real/a.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
    (root / "meta.csv").write_text(HEADER + GOOD_ROW, encoding="utf-8")
    (root / "labels/a.txt").write_text("0 0.3 0.35 0.5 0.3\n")
    return root


def errors_for(dataset: Path) -> list[str]:
    errors, _, _ = validate_dataset.validate(dataset)
    return errors


def test_valid_dataset_passes(dataset):
    assert errors_for(dataset) == []


def test_missing_license_is_error(dataset):
    (dataset / "LICENSE").unlink()
    assert any("LICENSE" in e for e in errors_for(dataset))


def test_missing_directory_is_error(dataset):
    (dataset / "generator").rmdir()
    assert any("generator" in e for e in errors_for(dataset))


def test_missing_column_is_error(dataset):
    (dataset / "meta.csv").write_text(
        "image;plate_num;plate_type\nimages/real/a.jpg;A123BC77;type1\n", encoding="utf-8"
    )
    assert any("missing columns" in e for e in errors_for(dataset))


def test_bad_plate_type_is_error(dataset):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace("type1;", "type9;", 1), encoding="utf-8")
    assert any("invalid plate_type" in e for e in errors_for(dataset))


def test_plate_mask_violation_is_error(dataset):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace("A123BC77", "AB12CD34"), encoding="utf-8")
    assert any("fails mask" in e for e in errors_for(dataset))


def test_missing_image_file_is_error(dataset):
    (dataset / "images/real/a.jpg").unlink()
    assert any("image not found" in e for e in errors_for(dataset))


def test_malformed_bbox_is_error(dataset):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace("10,20,100,30", "10,20,100"), encoding="utf-8")
    assert any("bbox must be 4 integers" in e for e in errors_for(dataset))


def test_malformed_quad_is_error(dataset):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace("10,20,110,20,110,50,10,50", "1,2,3"), encoding="utf-8")
    assert any("quad must be 8 integers" in e for e in errors_for(dataset))


def test_missing_label_is_error(dataset):
    (dataset / "labels/a.txt").unlink()
    assert any("label not found" in e for e in errors_for(dataset))


@pytest.mark.parametrize("bbox", ["10,20,0,30", "10,20,300,30", "-1,20,100,30"])
def test_invalid_bbox_geometry_is_error(dataset, bbox):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace("10,20,100,30", bbox))
    assert any("bbox" in e for e in errors_for(dataset))


def test_degenerate_quad_is_error(dataset):
    row = GOOD_ROW.replace("10,20,110,20,110,50,10,50", "10,20,10,20,10,20,10,20")
    (dataset / "meta.csv").write_text(HEADER + row)
    assert any("quad must be convex" in e for e in errors_for(dataset))


def test_non_binary_flag_is_error(dataset):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace(";1;0;src", ";2;0;src"), encoding="utf-8")
    assert any("is_vehicle must be 0 or 1" in e for e in errors_for(dataset))


def test_empty_license_is_error(dataset):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace(";CC-BY-4.0;", ";;"), encoding="utf-8")
    assert any("empty license" in e for e in errors_for(dataset))


def test_recommended_minimums_are_warnings_by_default(dataset):
    errors, warnings, _ = validate_dataset.validate(dataset)
    assert errors == []
    assert any("recommended" in w for w in warnings)


def test_strict_minimums_turns_them_into_errors(dataset):
    errors, _, _ = validate_dataset.validate(dataset, strict_minimums=True)
    assert any("recommended" in e for e in errors)


def test_statistics_are_reported(dataset):
    _, _, stats = validate_dataset.validate(dataset)
    assert stats["rows"] == 1
    assert stats["images"] == 1
    assert stats["by_type"]["type1"] == 1
    assert stats["unique_plates_by_type"]["type1"] == 1


def test_unknown_condition_is_warning_not_error(dataset):
    (dataset / "meta.csv").write_text(HEADER + GOOD_ROW.replace(";day", ";sunny"), encoding="utf-8")
    errors, warnings, _ = validate_dataset.validate(dataset)
    assert errors == []
    assert any("unknown conditions" in w for w in warnings)


def test_report_marks_fail(dataset):
    (dataset / "LICENSE").unlink()
    errors, warnings, stats = validate_dataset.validate(dataset)
    assert "status : FAIL" in validate_dataset.format_report(errors, warnings, stats)


def test_synthetic_does_not_fill_real_minimums(dataset):
    row = GOOD_ROW.replace('type1;', 'type1a;').replace(';1;0;src', ';1;1;src')
    (dataset / 'meta.csv').write_text(HEADER + row)
    errors, warnings, stats = validate_dataset.validate(dataset)
    assert not errors
    assert stats['images_by_type']['type1a'] == 1
    assert stats['real_images_by_type'].get('type1a', 0) == 0
    assert any('real group type1a: 0 images' in w for w in warnings)
    assert validate_dataset.RECOMMENDED_MIN_IMAGES['type1b'] == 300
    assert validate_dataset.RECOMMENDED_MIN_UNIQUE_PLATES['type1a'] == 50
