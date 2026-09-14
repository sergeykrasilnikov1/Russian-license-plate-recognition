"""Stage 4 tests: detector split, config profiles, vehicle heuristic, CLI dry-run.

CPU-only — no ultralytics / torch required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import export_onnx, prepare_detector_split, train_detector
from src.detection.split import load_image_records, stratified_split, write_data_yaml, write_split_lists
from src.detection.train import assert_class_list_matches_geometry, load_detector_config, resolve_profile
from src.detection.vehicle import VehicleFilterConfig, is_on_vehicle, vehicle_context_score
from src.utils.geometry import CLASS_NAME_TO_ID, PLATE_TYPES, class_names


class TestGeometryClasses:
    def test_class_names_follow_plate_types(self):
        assert class_names() == list(PLATE_TYPES.keys())
        assert CLASS_NAME_TO_ID["type1a"] == 1

    def test_assert_rejects_drift(self):
        with pytest.raises(ValueError):
            assert_class_list_matches_geometry({"names": {0: "foo", 1: "bar", 2: "baz", 3: "qux"}})


class TestConfig:
    def test_profiles_exist(self):
        cfg = load_detector_config(ROOT / "configs" / "detector.yaml")
        for name in ("accurate", "fast", "balanced"):
            resolved = resolve_profile(cfg, name)
            assert "imgsz" in resolved
        assert resolve_profile(cfg, "accurate")["imgsz"] == 640
        assert resolve_profile(cfg, "fast")["imgsz"] == 416

    def test_unknown_profile(self):
        cfg = load_detector_config(ROOT / "configs" / "detector.yaml")
        with pytest.raises(KeyError):
            resolve_profile(cfg, "turbo")


class TestSplit:
    def test_stratified_holdout_style_b(self, tmp_path):
        # Minimal fake meta
        meta = tmp_path / "meta.csv"
        lines = [
            "image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions",
        ]
        for i in range(10):
            lines.append(
                f"images/synthetic/a{i}.jpg;A123BC7{i};type1a;1,1,10,10;1,1,11,1,11,11,1,11;1;1;"
                f"synthetic_generator;CC-BY-4.0;day"
            )
        for i in range(4):
            lines.append(
                f"images/synthetic_style_b/b{i}.jpg;B123BC7{i};type1a;1,1,10,10;1,1,11,1,11,11,1,11;1;1;"
                f"synthetic_generator_style_b;CC-BY-4.0;night,style_b"
            )
        for i in range(8):
            lines.append(
                f"images/real/r{i}.jpg;########;type1;1,1,10,10;1,1,11,1,11,11,1,11;1;0;"
                f"hf;CC-BY-4.0;day"
            )
        meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
        records = load_image_records(meta)
        train, val = stratified_split(records, val_ratio=0.25, seed=0)
        assert all(r.style != "style_b" for r in train)
        assert any(r.style == "style_b" for r in val)
        assert len(train) + len(val) == len(records)

    def test_write_data_yaml_uses_geometry(self, tmp_path):
        out = tmp_path / "data.yaml"
        write_data_yaml(out, dataset_root=tmp_path, train_list=tmp_path / "t.txt", val_list=tmp_path / "v.txt")
        text = out.read_text(encoding="utf-8")
        for name in class_names():
            assert name in text
        assert "type1a" in text


class TestVehicleHeuristic:
    def test_bumper_like_scores_higher_than_bright_poster(self):
        h, w = 480, 640
        # Dark asphalt + dark bumper band under the plate
        car = np.full((h, w, 3), 40, dtype=np.uint8)
        car[300:, :] = (30, 30, 30)
        plate = (200, 280, 440, 340)
        car[280:340, 200:440] = 220
        # Bright flat "billboard"
        poster = np.full((h, w, 3), 230, dtype=np.uint8)
        poster[100:160, 50:590] = 250
        poster_box = (50, 100, 590, 160)
        s_car = vehicle_context_score(car, plate)
        s_poster = vehicle_context_score(poster, poster_box)
        assert s_car > s_poster
        ok, _ = is_on_vehicle(car, plate, VehicleFilterConfig(min_context_score=0.25))
        assert ok


class TestCli:
    def test_train_dry_run(self, tmp_path):
        # Need a data.yaml
        data = tmp_path / "data.yaml"
        write_data_yaml(data, dataset_root=tmp_path, train_list=tmp_path / "t.txt", val_list=tmp_path / "v.txt")
        code = train_detector.main(
            ["--dry-run", "--profile", "fast", "--data", str(data), "--config", str(ROOT / "configs" / "detector.yaml")]
        )
        assert code == 0

    def test_export_dry_run(self):
        code = export_onnx.main(
            ["--dry-run", "--weights", "runs/detect/fake/best.pt", "--imgsz", "416", "--half"]
        )
        assert code == 0

    def test_prepare_split_cli_parse(self):
        args = prepare_detector_split.parse_args(["--val-ratio", "0.15", "--seed", "1"])
        assert args.val_ratio == 0.15
        assert args.seed == 1
