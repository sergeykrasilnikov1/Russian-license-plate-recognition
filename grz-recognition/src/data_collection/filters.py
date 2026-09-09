"""CPU-only image filters: quality gate, perceptual dedup, face blur, conditions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class QualityFilter:
    """Drop images too small or too blurry to carry a readable plate."""

    min_side: int = 240
    min_sharpness: float = 20.0

    def check(self, image: np.ndarray) -> tuple[bool, str]:
        h, w = image.shape[:2]
        if min(h, w) < self.min_side:
            return False, f"too_small:{w}x{h}"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            return False, f"too_blurry:{sharpness:.1f}"
        return True, "ok"


class PerceptualDeduper:
    """Perceptual-hash dedup within the collected set and against a blocklist.

    The organizers' 30-image debug set must never enter the dataset, so its
    hashes are loaded as a blocklist before collection starts.
    """

    def __init__(self, hash_size: int = 8, max_distance: int = 4) -> None:
        self.hash_size = hash_size
        self.max_distance = max_distance
        self._hashes: list[int] = []
        self._blocked: list[int] = []

    @staticmethod
    def _to_int(image_hash) -> int:
        return int(str(image_hash), 16)

    def _phash(self, image: np.ndarray) -> int:
        import imagehash
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._to_int(imagehash.phash(Image.fromarray(rgb), hash_size=self.hash_size))

    @staticmethod
    def _distance(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def load_blocklist(self, paths: Iterable[Path]) -> int:
        count = 0
        for p in paths:
            image = cv2.imread(str(p))
            if image is None:
                continue
            self._blocked.append(self._phash(image))
            count += 1
        return count

    def check(self, image: np.ndarray) -> tuple[bool, str]:
        """Return (unique, reason); registers the hash when unique."""
        h = self._phash(image)
        for b in self._blocked:
            if self._distance(h, b) <= self.max_distance:
                return False, "duplicate_of_debug_set"
        for known in self._hashes:
            if self._distance(h, known) <= self.max_distance:
                return False, "duplicate_internal"
        self._hashes.append(h)
        return True, "ok"


class FaceBlurrer:
    """Blur human faces before an image enters the dataset.

    Two CPU backends, tried in order: MediaPipe face detection (better recall,
    needs a local .tflite bundle) and the Haar cascade bundled with OpenCV,
    which always works fully offline.
    """

    def __init__(self, model_path: Path | None = None, min_confidence: float = 0.4) -> None:
        self.min_confidence = min_confidence
        self.backend = "none"
        self._detector = None
        self._cascade = None

        if model_path and Path(model_path).is_file():
            self._detector = self._init_mediapipe(Path(model_path))
            if self._detector is not None:
                self.backend = "mediapipe"
        if self._detector is None:
            self._cascade = self._init_cascade()
            if self._cascade is not None:
                self.backend = "opencv_haar"

    def _init_mediapipe(self, model_path: Path):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=self.min_confidence,
            )
            return vision.FaceDetector.create_from_options(options)
        except Exception as exc:
            log.warning("mediapipe face detector unavailable: %s", exc)
            return None

    @staticmethod
    def _init_cascade():
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.is_file():
            return None
        cascade = cv2.CascadeClassifier(str(path))
        return None if cascade.empty() else cascade

    def _detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._detector is not None:
            import mediapipe as mp

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)
            boxes = []
            for det in getattr(result, "detections", []):
                bb = det.bounding_box
                boxes.append((bb.origin_x, bb.origin_y, bb.width, bb.height))
            return boxes
        if self._cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24))
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]
        return []

    def close(self) -> None:
        """Release the MediaPipe graph explicitly; its atexit path is noisy."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    def blur(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (image_with_blurred_faces, n_faces)."""
        boxes = self._detect(image)
        if not boxes:
            return image, 0
        out = image.copy()
        ih, iw = out.shape[:2]
        for x, y, w, h in boxes:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(iw, x + w), min(ih, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = out[y0:y1, x0:x1]
            k = max(9, (min(x1 - x0, y1 - y0) // 2) * 2 + 1)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        return out, len(boxes)


@dataclass
class ConditionEstimator:
    """Infer the `conditions` meta field from pixel statistics.

    Heuristic by design: it labels shooting conditions for dataset statistics,
    not plate content, so a wrong tag costs nothing downstream.
    """

    night_mean: float = 70.0
    glare_ratio: float = 0.02
    blur_threshold: float = 60.0
    tags: Sequence[str] = field(
        default_factory=lambda: ("day", "night", "glare", "motion_blur")
    )

    def estimate(self, image: np.ndarray) -> list[str]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        conditions = ["night" if mean < self.night_mean else "day"]
        if float((gray > 245).mean()) > self.glare_ratio:
            conditions.append("glare")
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.blur_threshold:
            conditions.append("motion_blur")
        return conditions
