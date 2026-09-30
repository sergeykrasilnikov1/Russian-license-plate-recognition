"""Kaggle GPU kernel: evaluate stopped Stage-4 YOLOv11n weights."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/grz-plates-detector")
WEIGHTS_ROOT = Path("/kaggle/input/grz-detector-weights")
WORK = Path("/kaggle/working")
CLASS_NAMES = ["type1", "type1a", "type1b", "other"]
IMGSZ = 640
DEVICE: int | str = 0


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return ""


def drop_torch_modules() -> None:
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch.") or mod == "torchvision" or mod.startswith("torchvision."):
            del sys.modules[mod]


def install_pascal_torch() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "--no-deps",
            "torch==2.3.1+cu118",
            "torchvision==0.18.1+cu118",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
    )
    drop_torch_modules()


def install_deps() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    gpu = gpu_name()
    torch_v = pkg_version("torch")
    numpy_v = pkg_version("numpy")
    vision_v = pkg_version("torchvision")
    pascal = any(x in gpu for x in ("P100", "K80", "M60"))
    print("preinstalled", torch_v, numpy_v, vision_v, gpu, "pascal", pascal, flush=True)
    if pascal:
        install_pascal_torch()
    constraint = WORK / "pip-constraints.txt"
    pins = []
    if numpy_v:
        pins.append(f"numpy=={numpy_v}")
    if not pascal:
        if torch_v:
            pins.append(f"torch=={torch_v}")
        if vision_v:
            pins.append(f"torchvision=={vision_v}")
    constraint.write_text("\n".join(pins) + ("\n" if pins else ""), encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics==8.3.40",
        "onnx==1.17.0",
        "onnxruntime-gpu==1.20.2",
    ]
    if pins:
        cmd.extend(["-c", str(constraint)])
    run(cmd)
    if pascal:
        install_pascal_torch()
    run([sys.executable, "-m", "pip", "uninstall", "-y", "ray"])


def resolve_device() -> int | str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    x = torch.zeros(1, device="cuda")
    x = x + 1
    print("cuda_ok", torch.__version__, torch.cuda.get_device_name(0), flush=True)
    return 0


def find_data_root() -> Path:
    """Kaggle may mount datasets under /kaggle/input/datasets/<user>/<slug>."""
    hits = []
    root = Path("/kaggle/input")
    if root.exists():
        for splits in root.rglob("splits"):
            if (splits / "val.txt").is_file() or (splits / "train.txt").is_file():
                hits.append(splits.parent)
    print("data_root_candidates", [str(p) for p in hits], flush=True)
    if hits:
        return hits[0]
    if DATA_ROOT.is_dir():
        return DATA_ROOT
    raise FileNotFoundError("dataset with splits/ not found under /kaggle/input")


def find_weights() -> Path:
    """Prefer bundled kernel file, then any Kaggle input dataset."""
    run(["bash", "-lc", "ls -la /kaggle/input && ls -la /kaggle/input/* 2>/dev/null | head -80"])
    names = {"detector_best.pt", "best.pt"}
    roots = [
        Path(__file__).resolve().parent,
        Path.cwd(),
        Path("/kaggle/src"),
        Path("/kaggle/input"),
        WORK,
    ]
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            if p.name == "yolo11n.pt":
                continue
            if p.name in names or "best" in p.name:
                hits.append(p)
    print("weight_candidates", [str(p) for p in hits], flush=True)
    for p in hits:
        if p.name == "detector_best.pt":
            return p
    if hits:
        return hits[0]
    raise FileNotFoundError("no detector weights found under input/src/cwd")


def prepare_absolute_splits(data_root: Path) -> Path:
    split_dir = WORK / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    src_dir = data_root / "splits"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"no splits at {src_dir}")
    for src in sorted(src_dir.glob("*.txt")):
        abs_lines = []
        missing = 0
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = data_root / line
            if not p.is_file():
                missing += 1
                continue
            abs_lines.append(str(p))
        (split_dir / src.name).write_text("\n".join(abs_lines) + ("\n" if abs_lines else ""), encoding="utf-8")
        stats[src.name] = {"ok": len(abs_lines), "missing": missing}
    print("split_stats", json.dumps(stats), flush=True)
    if stats.get("val.txt", {}).get("ok", 0) < 10:
        raise RuntimeError(f"val split empty after absolutize: {stats}")
    return split_dir


def eval_on_list(weights: Path, list_name: str, split_dir: Path, data_root: Path) -> dict:
    from ultralytics import YOLO

    list_path = split_dir / list_name
    if not list_path.is_file() or list_path.stat().st_size == 0:
        return {"list": list_name, "skipped": True}
    tmp = WORK / f"data_{list_name.replace('.txt', '')}.yaml"
    tmp.write_text(
        f"path: {data_root}\ntrain: {split_dir / 'train.txt'}\nval: {list_path}\nnc: 4\n"
        + "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    model = YOLO(str(weights))
    res = model.val(data=str(tmp), imgsz=IMGSZ, split="val", device=DEVICE, plots=False, cache=False)
    out: dict = {"list": list_name, "imgsz": IMGSZ}
    try:
        box = res.box
        out["map50"] = float(box.map50)
        out["map50_95"] = float(box.map)
        n = len(CLASS_NAMES)
        if getattr(box, "ap50", None) is not None:
            out["ap50_per_class"] = {
                CLASS_NAMES[int(cid)]: float(ap)
                for cid, ap in zip(box.ap_class_index, box.ap50)
            }
        maps = getattr(box, "maps", None)
        if maps is not None:
            out["map50_95_per_class"] = {
                CLASS_NAMES[int(cid)]: float(maps[int(cid)]) for cid in box.ap_class_index
            }
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def smoke_predict(weights: Path, split_dir: Path, n: int = 12) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    lines = (split_dir / "val.txt").read_text(encoding="utf-8").splitlines()
    paths = [ln.strip() for ln in lines if ln.strip()][:n]
    rows = []
    for p in paths:
        t0 = time.perf_counter()
        pred = model.predict(p, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        boxes = []
        if pred.boxes is not None:
            for b in pred.boxes:
                cls_id = int(b.cls[0])
                boxes.append(
                    {
                        "cls": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                        "conf": round(float(b.conf[0]), 4),
                        "xyxy": [round(float(x), 1) for x in b.xyxy[0].tolist()],
                    }
                )
        rows.append({"image": Path(p).name, "n": len(boxes), "ms": round(ms, 2), "boxes": boxes})
    return rows


def export_and_latency(weights: Path) -> dict:
    from ultralytics import YOLO
    import numpy as np
    import onnxruntime as ort

    def bench(onnx_path: Path) -> dict:
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            note = None
        except Exception as exc:  # noqa: BLE001
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            note = f"cuda_failed:{exc}"
        inp = sess.get_inputs()[0]
        x = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
        for _ in range(10):
            sess.run(None, {inp.name: x})
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            sess.run(None, {inp.name: x})
            times.append((time.perf_counter() - t0) * 1000)
        out = {
            "onnx": onnx_path.name,
            "providers": sess.get_providers(),
            "mean_ms": round(float(np.mean(times)), 3),
            "p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
        }
        if note:
            out["note"] = note
        return out

    model = YOLO(str(weights))
    lat: dict = {}
    exported = Path(str(model.export(format="onnx", imgsz=IMGSZ, half=False, simplify=True, dynamic=False, opset=12)))
    fp32 = WORK / "detector_fp32.onnx"
    shutil.copy2(exported, fp32)
    lat["fp32"] = bench(fp32)
    try:
        exported16 = Path(str(model.export(format="onnx", imgsz=IMGSZ, half=True, simplify=True, dynamic=False, opset=12)))
        fp16 = WORK / "detector_fp16.onnx"
        shutil.copy2(exported16, fp16)
        lat["fp16"] = bench(fp16)
        lat["fp16_delta_ms"] = round(lat["fp16"]["mean_ms"] - lat["fp32"]["mean_ms"], 3)
    except Exception as exc:  # noqa: BLE001
        lat["fp16"] = {"error": str(exc)}
    return lat


def main() -> None:
    global DEVICE

    os.environ.setdefault("MPLBACKEND", "Agg")
    install_deps()
    DEVICE = resolve_device()
    weights = find_weights()
    data_root = find_data_root()
    print("weights", weights, "size", weights.stat().st_size, flush=True)
    print("data_root", data_root, flush=True)
    shutil.copy2(weights, WORK / "detector_best.pt")

    split_dir = prepare_absolute_splits(data_root)
    report: dict = {
        "weights": str(weights),
        "data_root": str(data_root),
        "device": DEVICE,
        "gpu": gpu_name(),
        "torch": pkg_version("torch"),
        "numpy": pkg_version("numpy"),
        "domain_gap": {},
        "smoke": [],
        "latency": {},
    }
    out = WORK / "eval_metrics.json"

    def flush() -> None:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for list_name in (
        "val.txt",
        "val_real_type1.txt",
        "val_syn_type1a.txt",
        "val_syn_type1a_style_b.txt",
        "val_real_type1b.txt",
    ):
        print(f"=== eval {list_name} ===", flush=True)
        report["domain_gap"][list_name] = eval_on_list(weights, list_name, split_dir, data_root)
        flush()

    print("=== smoke predict ===", flush=True)
    try:
        report["smoke"] = smoke_predict(weights, split_dir)
    except Exception as exc:  # noqa: BLE001
        report["smoke"] = [{"error": str(exc)}]
    flush()
    print("=== export onnx ===", flush=True)
    try:
        report["latency"] = export_and_latency(weights)
    except Exception as exc:  # noqa: BLE001
        report["latency"] = {"error": str(exc)}
    flush()
    print("wrote", out, flush=True)
    print(json.dumps(report["domain_gap"], indent=2), flush=True)


if __name__ == "__main__":
    main()
