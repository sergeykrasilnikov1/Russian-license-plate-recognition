"""Regressions for the September audit: public entry points and data contracts."""
import csv
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.ocr.backends.ppocr import _ctc_greedy
from src.ocr.charset import finalize_plate
from src.ocr.warp import order_quad, DegenerateQuadError
from src.detection.vehicle import is_on_vehicle
from src.pipeline.infer import GrzPipeline, PlateResult, write_results_csv
from scripts.run_inference import _meta_index

ROOT = Path(__file__).resolve().parents[1]


def test_ctc_space_does_not_shift_scores():
    charset = ['blank', 'A', ' ', '1', '2', '3', 'B', 'C', '7']
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 0, 8, 0, 8]
    logits = np.zeros((len(ids), len(charset)))
    logits[np.arange(len(ids)), ids] = [.99, .1] + [.99] * 10
    text, scores = _ctc_greedy(logits, charset)
    assert len(text) == len(scores) == 9
    assert finalize_plate(text, scores, .35) == ('A123BC777', .99)


def test_normalization_keeps_scores_aligned():
    assert finalize_plate('A 123BC77', [.9, .01] + [.9] * 7, .35) == ('A123BC77', .9)


def test_invalid_region_and_missing_scores_are_not_confident():
    assert finalize_plate('A123BC456', [.99] * 9, .35) == ('A123BC#56', 0)
    assert finalize_plate('A123BC77', [], .35) == ('########', 0)
    assert finalize_plate('A123', [.99] * 4, .35) == ('', 0)


def test_quad_all_permutations():
    corners = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], dtype=np.float32)
    for indices in itertools.permutations(range(4)):
        np.testing.assert_equal(order_quad(corners[list(indices)]), corners)
    with pytest.raises(DegenerateQuadError):
        order_quad(np.array([[0, 0], [100, 0], [0, 100], [1, 1]], dtype=np.float32))


def test_blank_paper_is_not_vehicle_context():
    assert not is_on_vehicle(np.full((200, 400, 3), 255, np.uint8), (100, 70, 250, 120))[0]


def test_csv_quotes_names(tmp_path):
    name = 'nested/a;"b\n.jpg'
    path = tmp_path / 'out.csv'
    write_results_csv([PlateResult(name, 'A123BC77', 'type1', .9)], path)
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    assert rows[0]['image'] == name
    assert len(rows) == 1


def test_meta_index_does_not_merge_equal_basenames(tmp_path):
    path = tmp_path / 'meta.csv'
    path.write_text('image;plate_type;quad;bbox\na/x.jpg;type1;0;0\nb/x.jpg;type1a;0;0\n')
    data = _meta_index(path)
    assert len(data) == 2
    assert data[str((tmp_path / 'a/x.jpg').resolve())][0]['plate_type'] == 'type1'
    assert len(data[str((tmp_path / 'b/x.jpg').resolve())]) == 1


def test_pipeline_honors_scale_and_skips_invalid_box():
    from src.detection.infer import Detection
    class Detector:
        def predict(self, image):
            return [Detection((0, 0, 0, 0), .9, 0, 'type1'),
                    Detection((0, 0, 100, 40), .9, 0, 'type1')]
    class OCR:
        def recognize_detailed(self, crop):
            assert crop.shape[:2] == (112, 520)
            return 'A123BC77', [.9] * 8
    pipe = GrzPipeline({'vehicle_filter': {'enabled': False}}, {'warp_scale': 1},
                       detector=Detector(), ocr=OCR())
    assert len(pipe.process_image(np.zeros((50, 110, 3), np.uint8), 'x.jpg')) == 1


@pytest.mark.parametrize('script', ['benchmark_latency.py', 'run_inference.py', 'train_ocr.py'])
def test_cli_help(script):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--help'],
                            cwd=ROOT.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_training_cli_reports_unsupported_without_traceback():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_ocr.py')],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'not implemented' in result.stderr
    assert 'Traceback' not in result.stderr


def test_parseq_matches_training_normalization():
    from src.ocr.backends.parseq import ParseqOnnxBackend
    backend = ParseqOnnxBackend()
    np.testing.assert_allclose(backend._preprocess(np.zeros((32, 128, 3), np.uint8)), -1)
    np.testing.assert_allclose(backend._preprocess(np.full((32, 128, 3), 255, np.uint8)), 1)


def test_invalid_gt_metadata_is_skipped():
    from scripts.run_inference import _process_one
    class Pipeline:
        vehicle_enabled = False
        def process_with_gt_quad(self, *args, **kwargs):
            raise DegenerateQuadError('bad geometry')
    rows = _process_one(Pipeline(), np.zeros((40, 100, 3), np.uint8), 'x.jpg',
                        use_meta=True, recs=[{'quad': '0,0,0,0,0,0,0,0',
                                             'plate_type': 'type1', 'bbox': '0,0,0,0'}])
    assert rows == []


def test_metrics_follow_actual_class_ids():
    import ast
    from types import SimpleNamespace
    tree = ast.parse((ROOT / 'kaggle_eval/eval_kernel.py').read_text())
    # Exercise the exact serialization expressions without running the Kaggle job.
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.DictComp):
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                assignments[target.slice.value] = node.value
    box = SimpleNamespace(ap_class_index=np.array([2]), ap50=np.array([.92]))
    ns = {'CLASS_NAMES': ['type1','type1a','type1b','other'], 'box': box,
          'maps': [.75,.75,.81,.75], 'n':4}
    assert eval(compile(ast.Expression(assignments['ap50_per_class']), '<test>', 'eval'), ns) == {'type1b': .92}
    assert eval(compile(ast.Expression(assignments['map50_95_per_class']), '<test>', 'eval'), ns) == {'type1b': .81}


def test_cli_uses_basename_for_nested_images(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import cv2
    from scripts import run_inference
    folder = tmp_path / 'nested'
    folder.mkdir()
    cv2.imwrite(str(folder / 'car.jpg'), np.zeros((40, 100, 3), np.uint8))
    output = tmp_path / 'result.csv'
    args = SimpleNamespace(input=tmp_path, output=output, config=ROOT / 'configs/pipeline.yaml',
                           ocr_config=ROOT / 'configs/ocr.yaml', limit=0, from_meta=None,
                           prefer_onnx=False, no_vehicle_filter=False, warmup=0)
    class FakePipeline:
        def __init__(self, cfg):
            pass
        def process_image(self, image, name):
            return [PlateResult(name, 'A123BC77', 'type1', .9)]
    monkeypatch.setattr(run_inference, 'parse_args', lambda: args)
    monkeypatch.setattr(run_inference, 'ReferencePoseCrnnPipeline', FakePipeline)
    assert run_inference.main() == 0
    with output.open() as stream:
        assert list(csv.DictReader(stream, delimiter=';'))[0]['image'] == 'car.jpg'


def test_benchmark_times_selected_engine_without_gt(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import cv2
    from scripts import benchmark_latency
    cv2.imwrite(str(tmp_path / 'car.jpg'), np.zeros((40, 100, 3), np.uint8))
    calls = []
    class FakeReference:
        def __init__(self, cfg):
            calls.append(cfg['engine'])
        def process_image(self, image, name):
            calls.append(name)
    monkeypatch.setattr(benchmark_latency, 'ReferencePoseCrnnPipeline', FakeReference)
    args = SimpleNamespace(input=tmp_path, config=ROOT / 'configs/pipeline.yaml',
                           prefer_onnx=False, n=1, warmup=0)
    assert benchmark_latency.benchmark_pipeline(args, {}) == 0
    assert calls == ['reference_pose_crnn', 'car.jpg']
    assert 'ms/image' in capsys.readouterr().out
