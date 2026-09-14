"""Stage 3 tests: GOST plate rendering, seed reproducibility, meta rows.

CPU-only. A tiny --count run hits the filesystem under tmp_path.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.generator.generate import generate_dataset, generate_one, parse_type_weights
from dataset.generator.numbers import random_plate_text
from dataset.generator.plate_render import render_plate
from scripts import build_synthetic
from src.utils.geometry import PLATE_TYPES
from src.utils.plate_mask import validate_plate


class TestNumbers:
    def test_always_valid(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            ok, plate = validate_plate(random_plate_text(rng), allow_hash=False)
            assert ok, plate

    def test_reproducible(self):
        a = [random_plate_text(np.random.default_rng(7)) for _ in range(5)]
        b = [random_plate_text(np.random.default_rng(7)) for _ in range(5)]
        assert a == b


class TestRender:
    @pytest.mark.parametrize("plate_type", list(PLATE_TYPES))
    def test_aspect_matches_gost(self, plate_type):
        img = render_plate("A123BC77", plate_type, px_per_mm=2.0)
        geom = PLATE_TYPES[plate_type]
        ratio = img.shape[1] / img.shape[0]
        assert abs(ratio - geom.width_mm / geom.height_mm) < 0.02

    def test_type1b_is_yellow(self):
        img = render_plate("A123BC77", "type1b", px_per_mm=2.0)
        # Gap under the top inner border, left of the first glyph.
        h, w = img.shape[:2]
        patch = img[int(0.18 * h) : int(0.28 * h), int(0.02 * w) : int(0.06 * w), :3]
        b, g, r = patch.mean(axis=(0, 1))
        assert r > 140 and g > 120 and b < 120

    def test_type1_is_white(self):
        img = render_plate("A123BC77", "type1", px_per_mm=2.0)
        h, w = img.shape[:2]
        patch = img[int(0.18 * h) : int(0.28 * h), int(0.02 * w) : int(0.06 * w), :3]
        assert float(patch.mean()) > 140

    def test_type1_series_stays_left_of_divider(self):
        img = render_plate("A123BC77", "type1", px_per_mm=4.0)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        y0, y1 = int(0.20 * h), int(0.80 * h)
        x0, x1 = int(0.18 * w), int(0.72 * w)
        ys, xs = np.where(gray[y0:y1, x0:x1] < 50)
        assert xs.size > 0
        right_mm = (x0 + int(xs.max())) / (w / 520.0)
        assert right_mm < 378.0

    def test_type1a_top_row_is_four_glyphs(self):
        img = render_plate("A123BC777", "type1a", px_per_mm=4.0)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        band = gray[int(0.12 * h) : int(0.48 * h), int(0.08 * w) : int(0.92 * w)]
        ink = band < 60
        col = ink.mean(axis=0)
        runs, inside = [], False
        for x, v in enumerate(col > 0.04):
            if v and not inside:
                start = x
                inside = True
            elif not v and inside:
                runs.append((start, x))
                inside = False
        assert 3 <= len(runs) <= 10, runs

    def test_type1a_region_is_framed(self):
        img = render_plate("A123BC777", "type1a", px_per_mm=4.0)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Vertical wall of the region box on the blank (~617/1024).
        x = int(617 / 1024 * w)
        col = gray[int(0.50 * h) : int(0.95 * h), max(0, x - 2) : x + 3]
        assert float((col < 40).mean()) > 0.15
        # Top edge of the region box (~294/603).
        y = int(294 / 603 * h)
        row = gray[max(0, y - 2) : y + 3, int(0.60 * w) : int(0.95 * w)]
        assert float((row < 40).mean()) > 0.10

    def test_unknown_type_rejected(self):
        with pytest.raises(KeyError):
            render_plate("A123BC77", "type99")


class TestWeights:
    def test_parse_and_normalize(self):
        w = parse_type_weights("type1=1,type1a=3")
        assert abs(w["type1"] - 0.25) < 1e-9
        assert abs(w["type1a"] - 0.75) < 1e-9

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            parse_type_weights("type9=1")


class TestSceneRepro:
    def test_same_seed_same_pixels(self):
        a = generate_one(0, seed=42, plate_type="type1")
        b = generate_one(0, seed=42, plate_type="type1")
        assert a["plate"] == b["plate"]
        assert hashlib.md5(a["image"].tobytes()).hexdigest() == hashlib.md5(b["image"].tobytes()).hexdigest()

    def test_different_index_differs(self):
        a = generate_one(0, seed=42, plate_type="type1")
        b = generate_one(1, seed=42, plate_type="type1")
        assert hashlib.md5(a["image"].tobytes()).hexdigest() != hashlib.md5(b["image"].tobytes()).hexdigest()

    def test_quad_has_four_points(self):
        sample = generate_one(3, seed=1, plate_type="type1a")
        assert sample["quad"].shape == (4, 2)


class TestDatasetWrite:
    def test_writes_meta_and_images(self, tmp_path):
        ds = tmp_path / "dataset"
        (ds / "images" / "real").mkdir(parents=True)
        stats = generate_dataset(ds, count=8, seed=0, jpeg_quality=80)
        assert stats["count"] == 8
        assert sum(stats["type_counts"].values()) == 8
        meta_lines = (ds / "meta.csv").read_text(encoding="utf-8").strip().splitlines()
        assert meta_lines[0].startswith("image;plate_num")
        rows = meta_lines[1:]
        assert len(rows) == 8
        for row in rows:
            parts = row.split(";")
            assert parts[6] == "1"  # is_synthetic
            assert parts[7] == "synthetic_generator"
            ok, _ = validate_plate(parts[1], allow_hash=False)
            assert ok
        assert len(list((ds / "images" / "synthetic").glob("*.jpg"))) == 8
        assert len(list((ds / "labels").glob("syn_*.txt"))) == 8

    def test_overwrite_replaces_synthetics(self, tmp_path):
        ds = tmp_path / "dataset"
        generate_dataset(ds, count=4, seed=1)
        generate_dataset(ds, count=2, seed=2, overwrite=True)
        assert len(list((ds / "images" / "synthetic").glob("*.jpg"))) == 2


class TestCli:
    def test_defaults(self):
        args = build_synthetic.parse_args([])
        assert args.count == 5000
        assert args.seed == 42
        assert "type1a=0.38" in args.type_weights
        assert "type1b=0.32" in args.type_weights
