"""Stage 5: perspective warp, charset, pipeline CSV — CPU geometry, no GPU/OCR weights."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.infer import Detection
from src.ocr.charset import CHARSET, apply_char_threshold, finalize_plate, write_dict_file
from src.ocr.metrics import char_accuracy, full_plate_match
from src.ocr.warp import (
    DegenerateQuadError,
    choose_type1a_mode,
    order_quad,
    warp_plate,
    warp_quad,
)
from src.pipeline.infer import GrzPipeline, PlateResult, aggregate_confidence, write_results_csv
from src.utils.geometry import warp_size


def _corner_canvas() -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[20:90, 20:90] = (0, 0, 255)  # BGR red @ TL
    img[20:90, 210:280] = (0, 255, 0)  # green @ TR
    img[110:180, 210:280] = (255, 0, 0)  # blue @ BR
    img[110:180, 20:90] = (0, 255, 255)  # yellow @ BL
    quad = np.array([[20, 20], [280, 20], [280, 180], [20, 180]], dtype=np.float32)
    return img, quad


def _mean_bgr(patch: np.ndarray) -> tuple[int, int, int]:
    m = patch.reshape(-1, 3).mean(axis=0)
    return tuple(int(round(c)) for c in m)


class TestOrderQuad:
    def test_shuffled_and_clockwise(self):
        img, quad = _corner_canvas()
        ordered = order_quad(quad)
        cw = quad[[0, 3, 2, 1]]
        shuffled = np.roll(quad, 1, axis=0)
        np.testing.assert_allclose(order_quad(cw), ordered, atol=1e-3)
        np.testing.assert_allclose(order_quad(shuffled), ordered, atol=1e-3)
        w = warp_quad(img, shuffled, (100, 50))
        tl, tr = _mean_bgr(w[2:10, 2:10]), _mean_bgr(w[2:10, -10:-2])
        br, bl = _mean_bgr(w[-10:-2, -10:-2]), _mean_bgr(w[-10:-2, 2:10])
        assert tl[2] > 150  # red
        assert tr[1] > 150  # green
        assert br[0] > 150  # blue
        assert bl[1] > 150 and bl[2] > 150  # yellow

    def test_degenerate_raises(self):
        line = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
        with pytest.raises(DegenerateQuadError):
            order_quad(line)
        with pytest.raises(DegenerateQuadError):
            order_quad(np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.float32))

    def test_rotated_180_content(self):
        img, quad = _corner_canvas()
        rot = cv2.rotate(img, cv2.ROTATE_180)
        h, w = img.shape[:2]
        q180 = np.array([[w - 1 - x, h - 1 - y] for x, y in quad], dtype=np.float32)
        warped = warp_quad(rot, q180, (100, 50))
        tl = _mean_bgr(warped[2:10, 2:10])
        # geometric TL of the rotated image is the original BR (blue)
        assert tl[0] > 150

    def test_mirrored_quad_still_orders_tl(self):
        img, quad = _corner_canvas()
        mirrored = img[:, ::-1].copy()
        w = img.shape[1]
        q = np.array([[w - 1 - x, y] for x, y in quad], dtype=np.float32)
        warped = warp_quad(mirrored, q, (100, 50))
        tl = _mean_bgr(warped[2:10, 2:10])
        # geometric TL after horizontal flip is original TR (green)
        assert tl[1] > 150


class TestType1aWarp:
    def _synthetic_type1a(self) -> tuple[np.ndarray, np.ndarray]:
        pw, ph = warp_size("type1a", scale=0.8)
        plate = np.full((ph, pw, 3), 240, dtype=np.uint8)
        plate[: ph // 2] = (20, 20, 200)  # red-ish top line band
        plate[ph // 2 :] = (200, 20, 20)  # blue-ish bottom band
        canvas = np.full((400, 500, 3), 40, dtype=np.uint8)
        dst = np.array([[80, 60], [360, 50], [380, 310], [70, 300]], dtype=np.float32)
        src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(plate, m, (500, 400))
        mask = (warped.sum(axis=2) > 0)[:, :, None]
        canvas = np.where(mask, warped, canvas).astype(np.uint8)
        return canvas, dst

    def test_flatten_wider_than_stacked(self):
        img, quad = self._synthetic_type1a()
        flat = warp_plate(img, quad, "type1a", type1a_mode="flatten")
        stacked = warp_plate(img, quad, "type1a", type1a_mode="stacked")
        assert flat.shape[1] > stacked.shape[1]
        assert stacked.shape[0] > flat.shape[0]
        mid = flat.shape[1] // 2
        left = flat[:, :mid].reshape(-1, 3).mean(axis=0)
        right = flat[:, mid:].reshape(-1, 3).mean(axis=0)
        # left half from top band (red / high B), right from bottom (blue / high B wait)
        assert left[2] > left[0]  # top was reddish (high R in BGR → index 2)
        assert right[0] > right[2]

    def test_choose_flatten(self):
        samples = [self._synthetic_type1a() for _ in range(3)]
        assert choose_type1a_mode(samples) == "flatten"


class TestCharset:
    def test_spec_charset(self):
        assert CHARSET == "0123456789ABEKMHOPCTYX#"
        assert set("ABEKMHOPCTYX") <= set(CHARSET)

    def test_min_char_conf_hash(self):
        text = apply_char_threshold("A123BC77", [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9], 0.35)
        assert text[2] == "#"

    def test_finalize_empty_scores(self):
        plate, conf = finalize_plate("", [], 0.35)
        assert plate == ""
        assert conf == 0.0
        plate, conf = finalize_plate("a123bc77", [0.9] * 8, 0.35)
        assert plate == "A123BC77"
        assert conf == pytest.approx(0.9)

    def test_write_dict(self, tmp_path):
        path = write_dict_file(tmp_path / "d.txt")
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]
        assert lines == list(CHARSET)


class TestConfidenceCsv:
    def test_aggregate_min(self):
        assert aggregate_confidence(0.8, 0.3, "det_ocr_min") == pytest.approx(0.3)
        assert aggregate_confidence(0.8, 0.3, "mean") == pytest.approx(0.55)

    def test_pipeline_csv(self, tmp_path):
        class FakeDet:
            def predict(self, image):
                h, w = image.shape[:2]
                return [
                    Detection(xyxy=(5, 5, w - 5, h - 5), conf=0.91, cls_id=0, plate_type="type1"),
                ]

        class FakeOcr:
            name = "fake"

            def recognize_detailed(self, crop):
                return "A123BC77", [0.8] * 8

            def recognize(self, crop):
                return "A123BC77", 0.8

        img = np.full((80, 200, 3), 180, dtype=np.uint8)
        pipe = GrzPipeline(
            {"vehicle_filter": {"enabled": False}},
            {"min_char_conf": 0.35, "confidence_mode": "det_ocr_min", "active": "ppocrv4_mobile"},
            require_detector=False,
            detector=FakeDet(),
            ocr=FakeOcr(),
        )
        rows = pipe.process_image(img, "x.jpg")
        assert len(rows) == 1
        assert rows[0].plate_num == "A123BC77"
        assert rows[0].confidence == pytest.approx(min(0.91, 0.8), rel=0, abs=1e-4)
        out = tmp_path / "r.csv"
        write_results_csv(rows, out)
        text = out.read_text(encoding="utf-8")
        assert text.startswith("image;plate_num;plate_type;confidence")
        assert "x.jpg;A123BC77;type1;" in text

    def test_ocr_yaml_has_mask_and_threshold(self):
        cfg = yaml.safe_load((ROOT / "configs" / "ocr.yaml").read_text(encoding="utf-8"))
        assert "min_char_conf" in cfg
        assert cfg["charset"]
        assert (ROOT / "configs" / "ocr_dict.txt").is_file()
        assert char_accuracy("A123BC77", "A123BC77") == 1.0
        assert full_plate_match("a123bc77", "A123BC77")


def test_wilson_interval_n60_wide():
    from src.ocr.metrics import wilson_interval

    p, lo, hi = wilson_interval(23, 60)  # 38.3%
    assert abs(p - 23 / 60) < 1e-9
    assert hi - lo > 0.2  # ~±12 pp band at n=60
    _, lo400, hi400 = wilson_interval(int(0.383 * 400), 400)
    assert (hi400 - lo400) < (hi - lo)
