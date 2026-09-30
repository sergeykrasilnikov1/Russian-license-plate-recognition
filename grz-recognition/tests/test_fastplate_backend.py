"""Adapter contract checks without installing models or third-party runtimes."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.ocr.backends import create_backend
from src.ocr.backends.base import BackendUnavailable


def test_missing_artifacts_fail_before_optional_import(tmp_path):
    backend = create_backend("fastplate", rec_weights=tmp_path / "model.onnx")
    with pytest.raises(BackendUnavailable, match="artifact missing"):
        backend.recognize(np.zeros((8, 24, 3), dtype=np.uint8))


@pytest.mark.parametrize("mode", ["rgb", "grayscale"])
def test_color_conversion_and_padding_confidences(mode):
    backend = create_backend("fastplate")
    captured = []

    def run(image, return_confidence):
        assert return_confidence
        captured.append(image)
        return [SimpleNamespace(plate="A123BC77", char_probs=[0.9] * 8 + [0.1] * 2)]

    backend._recognizer = SimpleNamespace(config=SimpleNamespace(image_color_mode=mode), run=run)
    crop = np.zeros((8, 24, 3), dtype=np.uint8)
    crop[:, :, 2] = 255
    text, scores = backend.recognize_detailed(crop)
    assert text == "A123BC77"
    assert scores == [0.9] * 8
    if mode == "rgb":
        assert captured[0][0, 0].tolist() == [255, 0, 0]
    else:
        assert captured[0].ndim == 2


def test_missing_character_confidence_is_not_fabricated():
    backend = create_backend("fastplate")
    backend._recognizer = SimpleNamespace(
        config=SimpleNamespace(image_color_mode="rgb"),
        run=lambda *a, **kw: [SimpleNamespace(plate="A123BC77", char_probs=None)],
    )
    with pytest.raises(BackendUnavailable, match="confidences"):
        backend.recognize(np.zeros((8, 24, 3), dtype=np.uint8))
