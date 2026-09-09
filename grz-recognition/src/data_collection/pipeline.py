"""Ingestion pipeline shared by every collector.

Each candidate image walks the same gate sequence — license, decodability,
resolution/sharpness, perceptual dedup, face blur — before it is either
written into the dataset with a real bbox or parked in `raw_downloads/` for
Stage 4 labeling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..utils.geometry import CLASS_NAME_TO_ID
from .coarse_label import CoarseTypeClassifier, yellow_ratio
from .collectors.base import Collector, RawItem
from .filters import ConditionEstimator, FaceBlurrer, PerceptualDeduper, QualityFilter
from .licensing import LicensePolicy, is_acceptable, normalize_license
from .meta_store import ManifestRow, ManifestStore, MetaRow, MetaStore, append_yolo_label

log = logging.getLogger(__name__)

UNKNOWN_PLATE = "########"


@dataclass
class SourceStats:
    source: str
    available: bool = False
    reason: str = ""
    access_note: str = ""
    fetched: int = 0
    rejected_license: dict[str, int] = field(default_factory=dict)
    rejected_undecodable: int = 0
    rejected_quality: dict[str, int] = field(default_factory=dict)
    rejected_duplicate: dict[str, int] = field(default_factory=dict)
    images_with_faces: int = 0
    faces_blurred: int = 0
    labeled_images: int = 0
    labeled_plates: int = 0
    pending_images: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    type_disagreements: dict[str, int] = field(default_factory=dict)

    @property
    def accepted(self) -> int:
        return self.labeled_images + self.pending_images

    def bump(self, bucket: dict[str, int], key: str) -> None:
        bucket[key] = bucket.get(key, 0) + 1

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
            "access_note": self.access_note,
            "fetched": self.fetched,
            "accepted": self.accepted,
            "labeled_images": self.labeled_images,
            "labeled_plates": self.labeled_plates,
            "pending_images": self.pending_images,
            "images_with_faces": self.images_with_faces,
            "faces_blurred": self.faces_blurred,
            "rejected_license": self.rejected_license,
            "rejected_quality": self.rejected_quality,
            "rejected_duplicate": self.rejected_duplicate,
            "by_type": self.by_type,
            "type_disagreements_source_vs_geometry": self.type_disagreements,
        }


@dataclass
class DatasetPaths:
    root: Path

    @property
    def images_real(self) -> Path:
        return self.root / "images" / "real"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def meta(self) -> Path:
        return self.root / "meta.csv"

    @property
    def raw_downloads(self) -> Path:
        return self.root / "raw_downloads"

    @property
    def manifest(self) -> Path:
        return self.raw_downloads / "manifest.csv"

    @property
    def cache(self) -> Path:
        return self.root.parent / ".cache" / "downloads"


class Ingestor:
    def __init__(
        self,
        paths: DatasetPaths,
        policy: LicensePolicy = LicensePolicy.NONCOMMERCIAL,
        quality: QualityFilter | None = None,
        deduper: PerceptualDeduper | None = None,
        face_blurrer: FaceBlurrer | None = None,
        classifier: CoarseTypeClassifier | None = None,
        conditions: ConditionEstimator | None = None,
        dry_run: bool = False,
        flush_every: int = 25,
    ) -> None:
        self.paths = paths
        self.policy = policy
        self.quality = quality or QualityFilter()
        self.deduper = deduper or PerceptualDeduper()
        self.face_blurrer = face_blurrer or FaceBlurrer()
        self.classifier = classifier or CoarseTypeClassifier()
        self.conditions = conditions or ConditionEstimator()
        self.dry_run = dry_run
        self.flush_every = max(1, flush_every)

        self.meta = MetaStore(paths.meta)
        self.manifest = ManifestStore(paths.manifest)
        self.stats: dict[str, SourceStats] = {}

    def load_debug_blocklist(self, debug_dir: Path | None) -> int:
        if not debug_dir or not Path(debug_dir).is_dir():
            return 0
        return self.deduper.load_blocklist(_iter_images(Path(debug_dir)))

    def load_collected_hashes(self) -> int:
        """Seed the deduper with images collected by previous runs.

        Without this an incremental run re-downloads and re-admits everything
        it already has, since dedup state lives only in memory.
        """
        count = 0
        for directory in (self.paths.images_real, self.paths.raw_downloads):
            for image_path in _iter_images(directory):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                self.deduper.check(image)
                count += 1
        return count

    @staticmethod
    def _decode(data: bytes) -> np.ndarray | None:
        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _unique_name(self, directory: Path, filename: str) -> Path:
        candidate = directory / filename
        if not candidate.exists():
            return candidate
        stem, suffix = candidate.stem, candidate.suffix
        for i in range(1, 10000):
            alt = directory / f"{stem}_{i}{suffix}"
            if not alt.exists():
                return alt
        raise RuntimeError(f"cannot find free filename for {filename}")

    def run_source(self, collector: Collector, limit: int) -> SourceStats:
        stats = SourceStats(source=collector.name, access_note=collector.access_note)
        self.stats[collector.name] = stats

        available, reason = collector.check_available()
        stats.available, stats.reason = available, reason
        if not available:
            log.warning("source %s unavailable: %s", collector.name, reason)
            return stats

        for item in collector.collect(limit):
            stats.fetched += 1
            try:
                self._ingest(item, stats)
            except Exception as exc:
                log.warning("ingest failed for %s: %s", item.filename, exc)
            if not self.dry_run and stats.fetched % self.flush_every == 0:
                self.meta.flush()
                self.manifest.flush()

        if not self.dry_run:
            self.meta.flush()
            self.manifest.flush()
        return stats

    def reconcile(self) -> dict[str, int]:
        """Delete staged files that no ledger references.

        An interrupted run can leave images on disk whose ledger row was never
        flushed; those orphans would otherwise be invisible to dedup and to the
        dataset statistics.
        """
        removed = {"raw_downloads": 0, "images_real": 0}

        staged = {(self.paths.root / row).as_posix() for row in self.manifest.existing_files()}
        for image_path in _iter_images(self.paths.raw_downloads):
            if image_path.as_posix() not in staged:
                image_path.unlink()
                removed["raw_downloads"] += 1

        labeled = {(self.paths.root / row).as_posix() for row in self.meta.existing_images()}
        for image_path in _iter_images(self.paths.images_real):
            if image_path.as_posix() not in labeled:
                image_path.unlink()
                (self.paths.labels / f"{image_path.stem}.txt").unlink(missing_ok=True)
                removed["images_real"] += 1
        return removed

    def _ingest(self, item: RawItem, stats: SourceStats) -> None:
        info = normalize_license(item.license_raw)
        accepted, reason = is_acceptable(info, self.policy)
        if not accepted:
            stats.bump(stats.rejected_license, reason)
            return

        image = self._decode(item.read_bytes())
        if image is None:
            stats.rejected_undecodable += 1
            return

        ok, why = self.quality.check(image)
        if not ok:
            stats.bump(stats.rejected_quality, why.split(":")[0])
            return

        unique, why = self.deduper.check(image)
        if not unique:
            stats.bump(stats.rejected_duplicate, why)
            return

        image, n_faces = self.face_blurrer.blur(image)
        if n_faces:
            stats.images_with_faces += 1
            stats.faces_blurred += n_faces

        conditions = ",".join(self.conditions.estimate(image))

        if item.boxes:
            self._write_labeled(item, image, conditions, info.spdx, stats)
        else:
            self._write_pending(item, image, conditions, info.spdx, stats)

    def _resolve_type(self, item: RawItem, crop: np.ndarray, stats: SourceStats) -> str:
        """Decide plate_type for one crop.

        A source that documents its plate type wins over the geometric guess:
        distant or strongly angled type1 plates lose width and look square, so
        trusting the classifier there would relabel ordinary plates as type1a.
        """
        guess = self.classifier.classify(crop)
        if item.type_is_authoritative and item.suggested_type:
            if guess.plate_type != item.suggested_type:
                stats.bump(stats.type_disagreements, f"{item.suggested_type}->{guess.plate_type}")
            return item.suggested_type
        if guess.confidence >= 0.5:
            return guess.plate_type
        return item.suggested_type or guess.plate_type

    def _write_labeled(self, item: RawItem, image: np.ndarray, conditions: str, license_spdx: str, stats: SourceStats) -> None:
        """Source-annotated image: real bbox available, goes straight into meta.csv."""
        ih, iw = image.shape[:2]
        source_dir = self.paths.images_real / _slug(item.source)
        if self.dry_run:
            target = source_dir / item.filename
        else:
            source_dir.mkdir(parents=True, exist_ok=True)
            target = self._unique_name(source_dir, item.filename)
            if not cv2.imwrite(str(target), image):
                raise RuntimeError(f"failed to write {target}")

        rel_image = target.relative_to(self.paths.root).as_posix()
        label_path = self.paths.labels / f"{target.stem}.txt"
        if not self.dry_run and label_path.exists():
            label_path.unlink()

        plates = 0
        for box in item.boxes:
            x, y, w, h = box.to_pixels(iw, ih)
            x, y = max(0.0, x), max(0.0, y)
            w, h = min(w, iw - x), min(h, ih - y)
            if w < 8 or h < 4:
                continue

            crop = image[int(y) : int(y + h), int(x) : int(x + w)]
            plate_type = self._resolve_type(item, crop, stats)

            self.meta.append(
                MetaRow(
                    image=rel_image,
                    plate_num=UNKNOWN_PLATE,
                    plate_type=plate_type,
                    bbox=MetaRow.format_bbox(x, y, w, h),
                    quad=MetaRow.quad_from_bbox(x, y, w, h),
                    is_vehicle=1,
                    is_synthetic=0,
                    source=f"{item.source} <{item.source_url}>",
                    license=f"{license_spdx} ({item.license_raw})" if item.license_raw else license_spdx,
                    conditions=conditions,
                )
            )
            if not self.dry_run:
                append_yolo_label(
                    label_path, CLASS_NAME_TO_ID[plate_type], (x, y, w, h), (iw, ih)
                )
            stats.bump(stats.by_type, plate_type)
            plates += 1

        if plates:
            stats.labeled_images += 1
            stats.labeled_plates += plates
        elif not self.dry_run:
            # Every bbox was degenerate: drop the image so no file exists
            # without a meta.csv row referencing it.
            target.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            stats.rejected_undecodable += 1

    def _write_pending(self, item: RawItem, image: np.ndarray, conditions: str, license_spdx: str, stats: SourceStats) -> None:
        """No source bbox: park the image for Stage 4 auto-labeling."""
        ih, iw = image.shape[:2]
        source_dir = self.paths.raw_downloads / _slug(item.source)
        if self.dry_run:
            target = source_dir / item.filename
        else:
            source_dir.mkdir(parents=True, exist_ok=True)
            target = self._unique_name(source_dir, item.filename)
            if not cv2.imwrite(str(target), image):
                raise RuntimeError(f"failed to write {target}")

        hint = item.suggested_type or ""
        self.manifest.append(
            ManifestRow(
                file=target.relative_to(self.paths.root).as_posix(),
                source=item.source,
                source_url=item.source_url,
                license=item.license_raw,
                license_spdx=license_spdx,
                query=item.query,
                coarse_type=hint,
                coarse_confidence="query_hint",
                aspect=f"{(iw / ih) if ih else 0:.3f}",
                yellow_ratio=f"{yellow_ratio(image):.3f}",
                width=iw,
                height=ih,
                faces_blurred=0,
                conditions=conditions,
                label_status="pending_stage4",
            )
        )
        stats.pending_images += 1
        if hint:
            stats.bump(stats.by_type, f"{hint}?")

    def summary(self) -> dict:
        return {
            "meta_rows": len(self.meta),
            "meta_by_type": self.meta.type_counts(),
            "pending_rows": len(self.manifest),
            "pending_by_hint": self.manifest.coarse_counts(),
            "face_blur_backend": self.face_blurrer.backend,
            "license_policy": self.policy.value,
            "sources": [s.as_dict() for s in self.stats.values()],
        }

    def write_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False), encoding="utf-8")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:60]


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    return [
        p for p in sorted(directory.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def format_summary_table(summary: dict) -> str:
    lines = [
        f"{'source':<26} {'avail':<6} {'fetch':>6} {'meta':>6} {'pend':>6}  reason",
        "-" * 78,
    ]
    for s in summary["sources"]:
        lines.append(
            f"{s['source']:<26} {str(s['available']):<6} {s['fetched']:>6} "
            f"{s['labeled_images']:>6} {s['pending_images']:>6}  {s['reason'][:28]}"
        )
    lines.append("-" * 78)
    lines.append(f"meta.csv rows: {summary['meta_rows']} {summary['meta_by_type']}")
    lines.append(f"pending rows : {summary['pending_rows']} {summary['pending_by_hint']}")
    lines.append(f"face blur    : {summary['face_blur_backend']}")
    return "\n".join(lines)
