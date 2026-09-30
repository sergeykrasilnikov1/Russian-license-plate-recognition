# Server training guide — Stage 4 (YOLOv11n detector)

Run this on a machine **with an NVIDIA GPU**. The local CPU workstation only
validates CLI/config/splits (`--dry-run`).

## 1. Environment

```bash
cd grz-recognition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r requirements-train.txt
# Confirm CUDA is visible to PyTorch:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Record in the Stage 4 report: CUDA driver, `torch.__version__`,
`torch.version.cuda`, GPU model (ideally close to GTX 1050 Ti for latency notes).

## 2. Dataset on the server

Copy `dataset/meta.csv`, `dataset/images/{real,synthetic,synthetic_style_b}/`,
and `dataset/labels/` (or regenerate synthetics with the same seeds).

```bash
# Rebuild style_b holdout + stratified lists + configs/data.yaml
python scripts/prepare_detector_split.py --seed 42 --val-ratio 0.2 --build-style-b 150
```

`synthetic_generator_style_b` images **never enter train** — they are for
domain-gap eval of type1a only.

Class names in `configs/data.yaml` are generated from
`src.utils.geometry.PLATE_TYPES` (do not hardcode elsewhere).

## 3. Train

```bash
python scripts/train_detector.py \
  --config configs/detector.yaml \
  --data configs/data.yaml \
  --device 0 \
  --seed 42
```

Weights land under `runs/detect/grz_yolo11n*/weights/best.pt`.

## 4. Class-wise and domain-gap metrics

After training, evaluate with Ultralytics (example):

```bash
yolo detect val model=runs/detect/grz_yolo11n/weights/best.pt data=configs/data.yaml split=val
```

Additionally score these lists from `dataset/splits/` (build custom loops or
temporary data.yaml pointing `val:` at each list):

| List | Meaning |
|------|---------|
| `val_real_type1.txt` | real type1 — "normal" quality reference |
| `val_syn_type1a.txt` | synthetic type1a **same style as train** (optimistic) |
| `val_syn_type1a_style_b.txt` | synthetic type1a **other style** (honest domain gap) |
| `val_real_type1b.txt` | all real type1b in val (small N) |

Record mAP50 / mAP50-95 for each. If style_b ≪ same-style type1a, document the
domain gap in the explanatory note.

## 5. Export ONNX

```bash
python scripts/export_onnx.py \
  --weights runs/detect/grz_yolo11n/weights/best.pt \
  --imgsz 640 \
  --out weights/detector.onnx

# fp16 variant — measure accuracy drop on 1050 Ti (Pascal):
python scripts/export_onnx.py \
  --weights runs/detect/grz_yolo11n/weights/best.pt \
  --imgsz 640 --half \
  --out weights/detector_fp16.onnx
```

## 6. Latency (detector only)

```bash
python - <<'PY'
import time
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "weights/detector.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("providers:", sess.get_providers())
inp = sess.get_inputs()[0]
# Ultralytics ONNX usually NCHW float32
h = w = 640
x = np.random.randn(1, 3, h, w).astype(np.float32)
# warmup
for _ in range(10):
    sess.run(None, {inp.name: x})
times = []
for _ in range(100):
    t0 = time.perf_counter()
    sess.run(None, {inp.name: x})
    times.append((time.perf_counter() - t0) * 1000)
print(f"mean={np.mean(times):.2f} ms  p95={np.percentile(times,95):.2f} ms")
PY
```

If `TensorrtExecutionProvider` is available, repeat and compare. Do **not**
copy third-party TensorRT speedups — measure on the target GPU.

## 7. Vehicle filter

Classical CV heuristic: `src/detection/vehicle.py` (`is_on_vehicle`). Wired via
`configs/detector.yaml` → `vehicle_filter` and later `pipeline.yaml`. Replace
with a tiny CNN only if validation false-positive rate on `is_vehicle=0` rows
is unacceptable.
