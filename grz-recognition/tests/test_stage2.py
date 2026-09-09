"""Stage 2 tests: license policy, filters, coarse typing, ledgers, CLI.

Everything here runs offline on synthetic images — no network, no GPU.
Collector network paths are exercised through mocks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import scrape_real_data
from src.data_collection.coarse_label import CoarseTypeClassifier, yellow_ratio
from src.data_collection.collectors import COLLECTOR_REGISTRY
from src.data_collection.collectors.base import RateLimited, RawItem, YoloBox, request_with_retry
from src.data_collection.collectors.kaggle_ds import DEFAULT_DATASETS as KAGGLE_DEFAULTS
from src.data_collection.collectors.kaggle_ds import KaggleCollector
from src.data_collection.collectors.roboflow_universe import RoboflowCollector
from src.data_collection.filters import ConditionEstimator, PerceptualDeduper, QualityFilter
from src.data_collection.licensing import LicensePolicy, is_acceptable, normalize_license
from src.data_collection.meta_store import (
    META_COLUMNS,
    ManifestRow,
    ManifestStore,
    MetaRow,
    MetaStore,
    write_yolo_label,
)
from src.data_collection.pipeline import DatasetPaths, Ingestor
from src.utils.plate_mask import looks_like_special_plate


def make_plate(width: int, height: int, background=(255, 255, 255)) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = background
    img[height // 3 : 2 * height // 3, width // 6 : 5 * width // 6] = (20, 20, 20)
    return img


def noisy(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


class TestLicensing:
    @pytest.mark.parametrize(
        "raw,spdx",
        [
            ("CC BY 4.0", "CC-BY-4.0"),
            ("cc-by-sa-3.0", "CC-BY-SA-3.0"),
            ("by-nc", "CC-BY-NC-4.0"),
            ("CC0", "CC0-1.0"),
            ("apache-2.0", "Apache-2.0"),
            ("Public domain", "CC0-1.0"),
        ],
    )
    def test_normalization(self, raw, spdx):
        assert normalize_license(raw).spdx == spdx

    def test_unknown_is_rejected(self):
        info = normalize_license("All rights reserved")
        assert info.spdx == "UNKNOWN"
        accepted, reason = is_acceptable(info, LicensePolicy.NONCOMMERCIAL)
        assert not accepted and reason == "unknown_license"

    def test_no_derivatives_rejected_because_we_crop(self):
        accepted, reason = is_acceptable(normalize_license("CC BY-ND 4.0"), LicensePolicy.NONCOMMERCIAL)
        assert not accepted and reason == "no_derivatives"

    def test_strict_policy_excludes_sa_and_nc(self):
        for raw in ("cc-by-sa-4.0", "cc-by-nc-4.0"):
            accepted, _ = is_acceptable(normalize_license(raw), LicensePolicy.STRICT)
            assert not accepted
        accepted, _ = is_acceptable(normalize_license("CC BY 4.0"), LicensePolicy.STRICT)
        assert accepted

    def test_empty_license_rejected(self):
        accepted, _ = is_acceptable(normalize_license(""), LicensePolicy.NONCOMMERCIAL)
        assert not accepted


class TestQualityFilter:
    def test_rejects_small(self):
        ok, reason = QualityFilter(min_side=240).check(noisy(320, 100))
        assert not ok and reason.startswith("too_small")

    def test_rejects_flat(self):
        flat = np.full((400, 400, 3), 128, dtype=np.uint8)
        ok, reason = QualityFilter().check(flat)
        assert not ok and reason.startswith("too_blurry")

    def test_accepts_textured(self):
        ok, _ = QualityFilter().check(noisy(400, 400))
        assert ok


class TestDeduper:
    def test_detects_near_duplicate(self):
        deduper = PerceptualDeduper()
        image = noisy(320, 320)
        assert deduper.check(image)[0]
        slightly_scaled = np.array(image[:, :, :])
        assert deduper.check(slightly_scaled)[1] == "duplicate_internal"

    def test_blocklist_blocks(self, tmp_path):
        import cv2

        debug_image = noisy(320, 320)
        path = tmp_path / "debug.jpg"
        cv2.imwrite(str(path), debug_image)

        deduper = PerceptualDeduper()
        assert deduper.load_blocklist([path]) == 1
        loaded = cv2.imread(str(path))
        ok, reason = deduper.check(loaded)
        assert not ok and reason == "duplicate_of_debug_set"


class TestCoarseType:
    def test_yellow_background_detected(self):
        yellow = make_plate(520, 112, background=(0, 200, 255))
        assert yellow_ratio(yellow) > 0.3
        assert CoarseTypeClassifier().classify(yellow).plate_type == "type1b"

    def test_square_is_type1a(self):
        assert CoarseTypeClassifier().classify(make_plate(290, 170)).plate_type == "type1a"

    def test_wide_white_is_type1(self):
        assert CoarseTypeClassifier().classify(make_plate(520, 112)).plate_type == "type1"

    def test_confidence_in_range(self):
        for img in (make_plate(520, 112), make_plate(290, 170)):
            guess = CoarseTypeClassifier().classify(img)
            assert 0.0 <= guess.confidence <= 1.0


class TestConditions:
    def test_night_when_dark(self):
        dark = (noisy(320, 320) * 0.15).astype(np.uint8)
        assert "night" in ConditionEstimator().estimate(dark)

    def test_day_when_bright(self):
        bright = np.clip(noisy(320, 320).astype(int) // 2 + 150, 0, 255).astype(np.uint8)
        assert "day" in ConditionEstimator().estimate(bright)


class TestLedgers:
    def test_meta_columns_match_brief(self):
        assert list(META_COLUMNS) == [
            "image",
            "plate_num",
            "plate_type",
            "bbox",
            "quad",
            "is_vehicle",
            "is_synthetic",
            "source",
            "license",
            "conditions",
        ]

    def test_meta_roundtrip(self, tmp_path):
        store = MetaStore(tmp_path / "meta.csv")
        store.append(
            MetaRow(
                image="images/real/a.jpg",
                plate_num="A123BC77",
                plate_type="type1",
                bbox="10,20,100,30",
                quad="10,20,110,20,110,50,10,50",
                is_vehicle=1,
                is_synthetic=0,
                source="unit-test",
                license="CC-BY-4.0",
                conditions="day",
            )
        )
        store.flush()

        reloaded = MetaStore(tmp_path / "meta.csv")
        assert len(reloaded) == 1
        assert reloaded.type_counts() == {"type1": 1}
        assert reloaded.unique_plates() == {"A123BC77"}

    def test_unknown_plates_excluded_from_unique(self, tmp_path):
        store = MetaStore(tmp_path / "meta.csv")
        store.append(
            MetaRow("images/real/b.jpg", "########", "type1", "0,0,10,10", "", 1, 0, "s", "l", "day")
        )
        assert store.unique_plates() == set()

    def test_quad_from_bbox_is_clockwise(self):
        assert MetaRow.quad_from_bbox(10, 20, 100, 30) == "10,20,110,20,110,50,10,50"

    def test_manifest_roundtrip(self, tmp_path):
        store = ManifestStore(tmp_path / "manifest.csv")
        store.append(ManifestRow("raw/a.jpg", "src", "http://u", "CC BY 4.0", "CC-BY-4.0", coarse_type="type1b"))
        store.flush()
        assert ManifestStore(tmp_path / "manifest.csv").coarse_counts() == {"type1b": 1}

    def test_yolo_label_normalization(self, tmp_path):
        path = tmp_path / "img.txt"
        write_yolo_label(path, 2, (100, 50, 200, 100), (1000, 500))
        cls, xc, yc, w, h = path.read_text().split()
        assert cls == "2"
        assert float(xc) == pytest.approx(0.2)
        assert float(yc) == pytest.approx(0.2)
        assert float(w) == pytest.approx(0.2)
        assert float(h) == pytest.approx(0.2)


class TestIngestor:
    def _ingestor(self, tmp_path, **kwargs):
        paths = DatasetPaths(tmp_path / "dataset")
        return Ingestor(
            paths=paths,
            quality=QualityFilter(min_side=64, min_sharpness=1.0),
            face_blurrer=mock.Mock(blur=lambda img: (img, 0), backend="stub", close=lambda: None),
            **kwargs,
        )

    def _encode(self, image) -> bytes:
        import cv2

        return cv2.imencode(".jpg", image)[1].tobytes()

    def test_annotated_item_lands_in_meta(self, tmp_path):
        ingestor = self._ingestor(tmp_path)
        item = RawItem(
            source="unit:src",
            source_url="http://example/1",
            license_raw="CC BY 4.0",
            filename="car.jpg",
            data=self._encode(noisy(640, 480)),
            suggested_type="type1",
            type_is_authoritative=True,
            boxes=[YoloBox(0, 0.5, 0.5, 0.2, 0.06)],
        )
        collector = mock.Mock(name="c", access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])

        stats = ingestor.run_source(collector, limit=5)

        assert stats.labeled_plates == 1
        assert stats.pending_images == 0
        assert len(ingestor.meta) == 1
        row = ingestor.meta.rows[0]
        assert row["plate_type"] == "type1"
        assert row["is_synthetic"] == "0"
        assert row["plate_num"] == "########"
        assert (ingestor.paths.labels / "car.txt").is_file()

    def test_authoritative_type_beats_geometry(self, tmp_path):
        """A square-looking crop from a documented type1 dataset stays type1."""
        ingestor = self._ingestor(tmp_path)
        item = RawItem(
            source="unit:src",
            source_url="http://example/2",
            license_raw="CC BY 4.0",
            filename="angled.jpg",
            data=self._encode(noisy(640, 480)),
            suggested_type="type1",
            type_is_authoritative=True,
            boxes=[YoloBox(0, 0.5, 0.5, 0.1, 0.1)],
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])

        stats = ingestor.run_source(collector, limit=5)

        assert ingestor.meta.rows[0]["plate_type"] == "type1"
        assert stats.type_disagreements

    def test_unannotated_item_goes_pending(self, tmp_path):
        ingestor = self._ingestor(tmp_path)
        item = RawItem(
            source="commons",
            source_url="http://example/3",
            license_raw="CC BY-SA 4.0",
            filename="taxi.jpg",
            data=self._encode(noisy(640, 480)),
            suggested_type="type1b",
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])

        stats = ingestor.run_source(collector, limit=5)

        assert stats.pending_images == 1
        assert len(ingestor.meta) == 0
        assert ingestor.manifest.rows[0]["label_status"] == "pending_stage4"

    def test_bad_license_never_touches_disk(self, tmp_path):
        ingestor = self._ingestor(tmp_path)
        item = RawItem(
            source="commons",
            source_url="http://example/4",
            license_raw="All rights reserved",
            filename="nope.jpg",
            data=self._encode(noisy(640, 480)),
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])

        stats = ingestor.run_source(collector, limit=5)

        assert stats.rejected_license == {"unknown_license": 1}
        assert not any(ingestor.paths.raw_downloads.rglob("*.jpg"))

    def test_unavailable_source_is_recorded_not_raised(self, tmp_path):
        ingestor = self._ingestor(tmp_path)
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (False, "no_api_key")
        stats = ingestor.run_source(collector, limit=5)
        assert not stats.available and stats.reason == "no_api_key"
        collector.collect.assert_not_called()

    def test_reconcile_removes_orphans(self, tmp_path):
        import cv2

        ingestor = self._ingestor(tmp_path)
        staged = ingestor.paths.raw_downloads / "src"
        staged.mkdir(parents=True)
        cv2.imwrite(str(staged / "orphan.jpg"), noisy(100, 100))
        cv2.imwrite(str(staged / "known.jpg"), noisy(100, 100))
        labeled = ingestor.paths.images_real / "src"
        labeled.mkdir(parents=True)
        cv2.imwrite(str(labeled / "unreferenced.jpg"), noisy(100, 100))
        cv2.imwrite(str(labeled / "referenced.jpg"), noisy(100, 100))

        # Non-empty ledgers: only the rows below are known, the rest are orphans
        ingestor.manifest.append(
            ManifestRow(
                file="raw_downloads/src/known.jpg",
                source="unit",
                source_url="http://example/known",
                license="CC BY 4.0",
                license_spdx="CC-BY-4.0",
            )
        )
        ingestor.meta.append(
            MetaRow(
                image="images/real/src/referenced.jpg",
                plate_num="########",
                plate_type="type1",
                bbox="1 2 3 4",
                quad="",
                is_vehicle=1,
                is_synthetic=0,
                source="unit",
                license="CC-BY-4.0",
                conditions="day",
            )
        )

        assert ingestor.reconcile() == {"raw_downloads": 1, "images_real": 1}
        assert (staged / "known.jpg").is_file()
        assert (labeled / "referenced.jpg").is_file()
        assert not (staged / "orphan.jpg").exists()
        assert not (labeled / "unreferenced.jpg").exists()

    def test_degenerate_boxes_leave_no_orphan_file(self, tmp_path):
        """All bboxes too small to crop: the image must not stay on disk."""
        ingestor = self._ingestor(tmp_path)
        item = RawItem(
            source="unit:src",
            source_url="http://example/5",
            license_raw="CC BY 4.0",
            filename="tiny.jpg",
            data=self._encode(noisy(640, 480)),
            suggested_type="type1",
            type_is_authoritative=True,
            boxes=[YoloBox(0, 0.5, 0.5, 0.001, 0.001)],
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])

        ingestor.run_source(collector, limit=5)

        assert len(ingestor.meta) == 0
        assert not list(ingestor.paths.images_real.rglob("*.jpg"))
        assert not list(ingestor.paths.labels.glob("*.txt"))


class TestSourcePlateText:
    """Plate numbers encoded in export filenames are metadata, not guesses."""

    def test_valid_grz_filename_is_accepted(self):
        text, override = RoboflowCollector._plate_from_filename("A146AB799_png.rf.deadbeef")
        assert (text, override) == ("A146AB799", None)

    def test_diplomatic_filename_becomes_other(self):
        text, override = RoboflowCollector._plate_from_filename("002CD178_jpg.rf.cafe1234")
        assert text is None and override == "other"

    def test_unparseable_filename_yields_nothing(self):
        assert RoboflowCollector._plate_from_filename("IMG_20210101_png.rf.abc123") == (None, None)

    def test_special_series_detected(self):
        assert looks_like_special_plate("002CD178")
        assert not looks_like_special_plate("A146AB799")

    def test_ingestor_uses_source_plate_text(self, tmp_path):
        ingestor = TestIngestor()._ingestor(tmp_path)
        item = RawItem(
            source="unit:src",
            source_url="http://example/9",
            license_raw="CC BY 4.0",
            filename="A000AA78.jpg",
            data=TestIngestor()._encode(noisy(640, 480)),
            suggested_type="type1",
            type_is_authoritative=True,
            boxes=[YoloBox(0, 0.5, 0.5, 0.2, 0.06)],
            plate_text="A000AA78",
            plate_text_source="source_filename",
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])

        stats = ingestor.run_source(collector, limit=5)

        assert ingestor.meta.rows[0]["plate_num"] == "A000AA78"
        assert stats.plate_text_from_source == 1

    def test_invalid_source_text_falls_back_to_unknown(self, tmp_path):
        ingestor = TestIngestor()._ingestor(tmp_path)
        item = RawItem(
            source="unit:src",
            source_url="http://example/10",
            license_raw="CC BY 4.0",
            filename="bad.jpg",
            data=TestIngestor()._encode(noisy(640, 480)),
            suggested_type="type1",
            type_is_authoritative=True,
            boxes=[YoloBox(0, 0.5, 0.5, 0.2, 0.06)],
            plate_text="ZZ!!99",
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])
        ingestor.run_source(collector, limit=5)

        assert ingestor.meta.rows[0]["plate_num"] == "########"

    def test_text_dropped_when_image_has_several_plates(self, tmp_path):
        ingestor = TestIngestor()._ingestor(tmp_path)
        item = RawItem(
            source="unit:src",
            source_url="http://example/11",
            license_raw="CC BY 4.0",
            filename="two.jpg",
            data=TestIngestor()._encode(noisy(640, 480)),
            suggested_type="type1",
            type_is_authoritative=True,
            boxes=[YoloBox(0, 0.3, 0.5, 0.2, 0.06), YoloBox(0, 0.7, 0.5, 0.2, 0.06)],
            plate_text="A000AA78",
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])
        ingestor.run_source(collector, limit=5)

        assert {r["plate_num"] for r in ingestor.meta.rows} == {"########"}


class TestYellowOverride:
    """Yellow survives perspective, so it may override a declared type1."""

    def _run(self, tmp_path, declared: str, background) -> str:
        import cv2

        ingestor = TestIngestor()._ingestor(tmp_path)
        image = noisy(640, 480)
        image[200:280, 220:420] = background
        item = RawItem(
            source="unit:src",
            source_url="http://example/12",
            license_raw="CC BY 4.0",
            filename="plate.jpg",
            data=cv2.imencode(".jpg", image)[1].tobytes(),
            suggested_type=declared,
            type_is_authoritative=True,
            boxes=[YoloBox(0, 0.5, 0.5, 0.3, 0.16)],
        )
        collector = mock.Mock(access_note="")
        collector.name = "unit"
        collector.check_available.return_value = (True, "ok")
        collector.collect.return_value = iter([item])
        ingestor.run_source(collector, limit=5)
        return ingestor.meta.rows[0]["plate_type"]

    def test_yellow_in_russian_source_becomes_type1b(self, tmp_path):
        assert self._run(tmp_path, "type1", (0, 200, 255)) == "type1b"

    def test_yellow_in_foreign_source_stays_other(self, tmp_path):
        assert self._run(tmp_path, "other", (0, 200, 255)) == "other"

    def test_white_plate_keeps_declared_type(self, tmp_path):
        assert self._run(tmp_path, "type1", (255, 255, 255)) == "type1"


class TestAnnotationParsers:
    def test_voc_xml_to_yolo(self, tmp_path):
        (tmp_path / "car.xml").write_text(
            """<annotation><size><width>200</width><height>100</height></size>
            <object><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>60</ymax></bndbox></object>
            </annotation>""",
            encoding="utf-8",
        )
        boxes = KaggleCollector._voc_boxes(tmp_path, tmp_path / "car.jpg")
        assert len(boxes) == 1
        assert boxes[0].xc == pytest.approx(0.5)
        assert boxes[0].yc == pytest.approx(0.4)
        assert boxes[0].w == pytest.approx(0.5)
        assert boxes[0].h == pytest.approx(0.4)

    def test_voc_without_size_is_skipped(self, tmp_path):
        (tmp_path / "car.xml").write_text("<annotation><object/></annotation>", encoding="utf-8")
        assert KaggleCollector._voc_boxes(tmp_path, tmp_path / "car.jpg") == []

    def test_coco_index_to_yolo(self, tmp_path):
        (tmp_path / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg", "width": 400, "height": 200}],
                    "annotations": [{"image_id": 1, "bbox": [100, 40, 200, 80]}],
                }
            ),
            encoding="utf-8",
        )
        collector = KaggleCollector(tmp_path, token="stub")
        index = collector._coco_index(tmp_path)
        assert list(index) == ["a.jpg"]
        box = index["a.jpg"][0]
        assert (box.xc, box.yc, box.w, box.h) == pytest.approx((0.5, 0.4, 0.5, 0.4))

    def test_coco_ignores_degenerate_bbox(self, tmp_path):
        (tmp_path / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg", "width": 400, "height": 200}],
                    "annotations": [{"image_id": 1, "bbox": [10, 10, 0, 40]}],
                }
            ),
            encoding="utf-8",
        )
        assert KaggleCollector(tmp_path, token="stub")._coco_index(tmp_path) == {}


class TestCredentialGating:
    """Both SDK sources must skip themselves instead of scraping without keys."""

    def test_roboflow_without_key_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
        ok, reason = RoboflowCollector(tmp_path).check_available()
        assert not ok and "no_api_key" in reason

    def test_kaggle_without_token_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setattr(KaggleCollector, "_read_token", staticmethod(lambda: ""))
        ok, reason = KaggleCollector(tmp_path).check_available()
        assert not ok and "no_token" in reason

    def test_kaggle_reports_expired_token(self, tmp_path):
        collector = KaggleCollector(tmp_path, token="KGAT_expired")
        with mock.patch.object(
            KaggleCollector, "_session", return_value=mock.Mock(get=lambda *a, **k: mock.Mock(status_code=401))
        ):
            ok, reason = collector.check_available()
        assert not ok and "token_rejected_or_expired" in reason

    def test_excluded_datasets_stay_excluded(self):
        refs = {d.ref for d in KAGGLE_DEFAULTS}
        assert "evgrafovmaxim/nomeroff-russian-license-plates" not in refs
        assert "adilshamim8/license-plate-recognition" not in refs


class TestLedgerSafety:
    """A ledger that cannot be parsed must never be silently overwritten."""

    def test_mismatched_header_raises(self, tmp_path):
        path = tmp_path / "meta.csv"
        path.write_text("image;plate_num\nfoo.jpg;A000AA78\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected columns"):
            MetaStore(path)

    def test_empty_file_is_treated_as_new(self, tmp_path):
        path = tmp_path / "meta.csv"
        path.write_text("", encoding="utf-8")
        assert len(MetaStore(path)) == 0

    def test_existing_rows_are_marked_loaded(self, tmp_path):
        path = tmp_path / "meta.csv"
        store = MetaStore(path)
        store.append(
            MetaRow(
                image="images/real/a.jpg",
                plate_num="A000AA78",
                plate_type="type1",
                bbox="1 2 3 4",
                quad="",
                is_vehicle=1,
                is_synthetic=0,
                source="unit",
                license="CC-BY-4.0",
                conditions="day",
            )
        )
        store.flush()
        assert MetaStore(path).loaded_from_disk is True

    def test_reconcile_keeps_files_when_ledger_is_empty(self, tmp_path):
        """The failure mode that once deleted an entire collected source."""
        ingestor = TestIngestor()._ingestor(tmp_path)
        stray = ingestor.paths.images_real / "src" / "keep.jpg"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"not-an-image")

        removed = ingestor.reconcile()

        assert removed["images_real"] == 0
        assert stray.is_file()


class TestRetry:
    def test_rate_limited_after_attempts(self):
        response = mock.Mock(status_code=429, headers={})
        session = mock.Mock(get=mock.Mock(return_value=response))
        with mock.patch("time.sleep"):
            with pytest.raises(RateLimited):
                request_with_retry(session, "http://x", attempts=2, base_delay=0)
        assert session.get.call_count == 2

    def test_returns_on_success_after_retry(self):
        bad = mock.Mock(status_code=429, headers={})
        good = mock.Mock(status_code=200, headers={})
        session = mock.Mock(get=mock.Mock(side_effect=[bad, good]))
        with mock.patch("time.sleep"):
            assert request_with_retry(session, "http://x", attempts=3, base_delay=0) is good


class TestCli:
    def test_registry_covers_documented_sources(self):
        assert set(COLLECTOR_REGISTRY) == {
            "huggingface",
            "commons",
            "openverse",
            "roboflow",
            "kaggle",
            "platesmania",
        }

    def test_defaults(self):
        args = scrape_real_data.parse_args([])
        assert args.limit == 200
        assert args.license_policy == "noncommercial"
        assert args.seed == 42

    def test_plate_type_restriction_is_passed_through(self, tmp_path):
        args = scrape_real_data.parse_args(["--plate-types", "type1b", "--sources", "commons"])
        collector = scrape_real_data.build_collector("commons", tmp_path, args)
        assert set(collector.queries) == {"type1b"}

    def test_hf_archive_restriction(self, tmp_path):
        args = scrape_real_data.parse_args(["--hf-archives", "val.zip"])
        collector = scrape_real_data.build_collector("huggingface", tmp_path, args)
        assert collector.archive_filter == ("val.zip",)

    def test_rejects_unknown_source(self):
        with pytest.raises(SystemExit):
            scrape_real_data.parse_args(["--sources", "definitely-not-a-source"])


class TestCommonsModes:
    def test_category_mode_is_default(self, tmp_path):
        collector = COLLECTOR_REGISTRY["commons"](tmp_path)
        kinds = {kind for _, kind, _ in collector._work_items()}
        assert kinds == {"category"}

    def test_both_mode_covers_search_and_category(self, tmp_path):
        from src.data_collection.collectors.commons import WikimediaCommonsCollector

        collector = WikimediaCommonsCollector(tmp_path, mode="both", plate_types=("type1b",))
        kinds = {kind for _, kind, _ in collector._work_items()}
        assert kinds == {"category", "search"}
        assert all(pt == "type1b" for pt, _, _ in collector._work_items())

    def test_category_traversal_descends_into_subcategories(self, tmp_path):
        from src.data_collection.collectors.commons import WikimediaCommonsCollector

        collector = WikimediaCommonsCollector(tmp_path, delay=0.0, category_depth=1)

        def fake_members(self, session, category, member_type, limit):
            if category == "Root" and member_type == "file":
                return ["File:a.jpg"]
            if category == "Root" and member_type == "subcat":
                return ["Category:Child"]
            if category == "Child" and member_type == "file":
                return ["File:b.jpg", "File:c.jpg"]
            return []

        with mock.patch.object(WikimediaCommonsCollector, "_category_members", fake_members):
            files = collector._category_files(None, "Root", limit=5, depth=1)

        assert files == ["File:a.jpg", "File:b.jpg", "File:c.jpg"]

    def test_unscaled_originals_are_skipped(self, tmp_path):
        """Wikimedia asks clients to use thumbnails, not full-size originals."""
        from src.data_collection.collectors.commons import WikimediaCommonsCollector

        collector = WikimediaCommonsCollector(tmp_path, delay=0.0, plate_types=("type1b",))
        same = "https://upload.wikimedia.org/x/diagram.png"
        infos = [{"mime": "image/png", "thumburl": same, "url": same, "_title": "File:diagram.png"}]

        with mock.patch.object(WikimediaCommonsCollector, "_category_files", lambda *a, **k: ["File:diagram.png"]), \
             mock.patch.object(WikimediaCommonsCollector, "_image_info", lambda *a, **k: infos), \
             mock.patch.object(WikimediaCommonsCollector, "_session", lambda self: mock.Mock()):
            assert list(collector.collect(limit=3)) == []


class TestNoBypass:
    """The rules forbid circumventing technical restrictions; assert we don't."""

    def test_platesmania_declares_documented_export_only(self, tmp_path):
        collector = COLLECTOR_REGISTRY["platesmania"](tmp_path)
        source = Path(ROOT / "src/data_collection/collectors/platesmania.py").read_text(encoding="utf-8")
        assert "robots" in source.lower()
        for forbidden in ("captcha_solve", "2captcha", "anticaptcha", "cloudscraper", "undetected_chromedriver"):
            assert forbidden not in source
        assert "No CAPTCHA bypass" in collector.access_note

    def test_user_agent_is_honest(self, tmp_path):
        collector = COLLECTOR_REGISTRY["commons"](tmp_path)
        ua = collector._session().headers["User-Agent"]
        assert "grz-dataset-collector" in ua
        assert "Mozilla" not in ua
