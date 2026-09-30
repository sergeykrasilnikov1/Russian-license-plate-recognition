"""Stage 6: offline inference contract — CSV, empty detections, no HTTP in runtime."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.infer import Detection
from src.ocr.backends.ppocr import PPOCRV4Backend
from src.pipeline.infer import GrzPipeline, write_results_csv

INFER_PATHS = [
    ROOT / "src" / "pipeline",
    ROOT / "src" / "detection" / "infer.py",
    ROOT / "src" / "detection" / "vehicle.py",
    ROOT / "src" / "ocr",
    ROOT / "scripts" / "run_inference.py",
]

BANNED_HTTP_NAMES = {"requests", "urllib", "httpx", "aiohttp"}


def _py_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.py"))


class TestOfflineInferenceSources:
    def test_no_http_imports_in_runtime(self):
        offenders = []
        for root in INFER_PATHS:
            for py in _py_files(root):
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            if top in BANNED_HTTP_NAMES:
                                offenders.append(f"{py.relative_to(ROOT)}:import {alias.name}")
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        top = node.module.split(".")[0]
                        if top in BANNED_HTTP_NAMES:
                            offenders.append(f"{py.relative_to(ROOT)}:from {node.module}")
        assert offenders == []

    def test_no_hardcoded_filename_lookup(self):
        text = (ROOT / "scripts" / "run_inference.py").read_text(encoding="utf-8")
        # Answers must come from the model, not a dict of jury filenames.
        assert "HARDCODE" not in text.upper()
        assert "ANSWER_KEY" not in text
        assert ".jpg\":" not in text


class TestCsvContract:
    def test_empty_detections_omit_image(self, tmp_path):
        class EmptyDet:
            def predict(self, image):
                return []

        class FakeOcr:
            name = "fake"

            def recognize_detailed(self, crop):
                return "A123BC77", [0.9] * 8

        img = np.full((40, 80, 3), 120, dtype=np.uint8)
        pipe = GrzPipeline(
            {"vehicle_filter": {"enabled": False}},
            {"min_char_conf": 0.35, "confidence_mode": "det_ocr_min"},
            require_detector=False,
            detector=EmptyDet(),
            ocr=FakeOcr(),
        )
        rows = pipe.process_image(img, "none.jpg")
        assert rows == []
        out = tmp_path / "out.csv"
        write_results_csv(rows, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines == ["image;plate_num;plate_type;confidence"]

    def test_multiple_plates_multiple_rows(self, tmp_path):
        class TwoDet:
            def predict(self, image):
                h, w = image.shape[:2]
                return [
                    Detection(xyxy=(2, 2, w // 2, h - 2), conf=0.9, cls_id=0, plate_type="type1"),
                    Detection(xyxy=(w // 2, 2, w - 2, h - 2), conf=0.8, cls_id=2, plate_type="type1b"),
                ]

        class FakeOcr:
            name = "fake"
            n = 0

            def recognize_detailed(self, crop):
                self.n += 1
                text = "A111AA11" if self.n == 1 else "B222BB22"
                return text, [0.9] * 8

        img = np.full((60, 160, 3), 140, dtype=np.uint8)
        pipe = GrzPipeline(
            {"vehicle_filter": {"enabled": False}},
            {"min_char_conf": 0.35, "confidence_mode": "det_ocr_min"},
            require_detector=False,
            detector=TwoDet(),
            ocr=FakeOcr(),
        )
        rows = pipe.process_image(img, "two.jpg")
        assert len(rows) == 2
        assert {r.plate_type for r in rows} == {"type1", "type1b"}
        out = tmp_path / "m.csv"
        write_results_csv(rows, out)
        body = [ln for ln in out.read_text(encoding="utf-8").splitlines()[1:] if ln]
        assert all(ln.count(";") == 3 for ln in body)


class TestVendoredOcr:
    def test_local_v4_weights_exist(self):
        w = ROOT / "weights" / "ocr" / "en_PP-OCRv4_rec_mobile.onnx"
        assert w.is_file() and w.stat().st_size > 1_000_000

    def test_local_v4_reads_toy_plate(self):
        import cv2

        img = np.full((48, 200, 3), 230, dtype=np.uint8)
        cv2.putText(img, "A123BC77", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        text, conf = PPOCRV4Backend().recognize(img)
        assert isinstance(text, str)
        assert 0.0 <= conf <= 1.0
        # Font is not GOST; require that we at least emit charset glyphs, not crash.
        assert text == "" or all(ch.isalnum() or ch in "# " for ch in text)
