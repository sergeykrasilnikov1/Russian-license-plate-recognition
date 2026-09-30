"""Kaggle GPU kernel: train GRZ YOLOv11n detector (Stage 4).

Expects dataset at /kaggle/input/grz-plates-detector with Ultralytics layout.
Writes metrics + ONNX + best.pt under /kaggle/working/.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/grz-plates-detector")
WORK = Path("/kaggle/working")
IMGSZ = 640
BATCH = 16
EPOCHS = 80
CLASS_NAMES = ["type1", "type1a", "type1b", "other"]
SEED = 42
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
    """cu118 wheels still include sm_60. --no-deps leaves numpy as-is."""
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
    """Install ultralytics/onnx. Keep Kaggle numpy; Pascal GPU gets cu118 torch only."""
    WORK.mkdir(parents=True, exist_ok=True)
    gpu = gpu_name()
    torch_v = pkg_version("torch")
    numpy_v = pkg_version("numpy")
    vision_v = pkg_version("torchvision")
    pascal = any(x in gpu for x in ("P100", "K80", "M60"))
    print(
        "preinstalled",
        f"python={sys.version.split()[0]}",
        f"torch={torch_v}",
        f"numpy={numpy_v}",
        f"torchvision={vision_v}",
        f"gpu={gpu}",
        f"pascal={pascal}",
        flush=True,
    )
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
    # Kaggle ships a Ray that is incompatible with ultralytics 8.3 raytune callback.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "ray"])
    print(
        "after_install",
        f"torch={pkg_version('torch')}",
        f"numpy={pkg_version('numpy')}",
        f"torchvision={pkg_version('torchvision')}",
        flush=True,
    )


def resolve_device() -> int | str:
    """Use GPU 0 if torch can actually run a CUDA op."""
    import torch

    print(
        "torch",
        torch.__version__,
        "cuda",
        torch.cuda.is_available(),
        "arch",
        getattr(torch.cuda, "get_arch_list", lambda: [])(),
        flush=True,
    )
    if not torch.cuda.is_available():
        print("cuda unavailable -> cpu", flush=True)
        return "cpu"
    try:
        print("gpu", torch.cuda.get_device_name(0), flush=True)
        x = torch.zeros(1, device="cuda")
        x = x + 1
        print("cuda_ok", x.device, flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch cannot execute CUDA on this GPU") from exc


def prepare_absolute_splits() -> Path:
    """Rewrite split lists with absolute image paths (Ultralytics does not
    prepend yaml `path` to lines inside a .txt list)."""
    split_dir = WORK / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for src in sorted((DATA_ROOT / "splits").glob("*.txt")):
        abs_lines = []
        missing = 0
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = DATA_ROOT / line
            if not p.is_file():
                missing += 1
                continue
            abs_lines.append(str(p))
        (split_dir / src.name).write_text("\n".join(abs_lines) + ("\n" if abs_lines else ""), encoding="utf-8")
        stats[src.name] = {"ok": len(abs_lines), "missing": missing}
    print("split_stats", json.dumps(stats), flush=True)
    if stats.get("train.txt", {}).get("ok", 0) < 100:
        raise RuntimeError(f"too few train images after absolutize: {stats}")
    return split_dir


def write_data_yaml(path: Path, split_dir: Path) -> Path:
    text = f"""path: {DATA_ROOT}
train: {split_dir / 'train.txt'}
val: {split_dir / 'val.txt'}
nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        text += f"  {i}: {name}\n"
    path.write_text(text, encoding="utf-8")
    return path


def disable_ultralytics_raytune() -> None:
    """Ray on Kaggle 3.12 has no `_get_session`; skip the auto-registered callback."""
    try:
        import ultralytics.utils.callbacks.raytune as raytune

        raytune.on_fit_epoch_end = lambda trainer: None  # noqa: ARG005
        raytune.on_train_end = lambda trainer: None  # noqa: ARG005
    except Exception as exc:  # noqa: BLE001
        print("raytune patch skipped:", exc, flush=True)


def train_once(data_yaml: Path) -> dict:
    from ultralytics import YOLO

    disable_ultralytics_raytune()
    model = YOLO("yolo11n.pt")
    t0 = time.perf_counter()
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        seed=SEED,
        workers=2,
        project=str(WORK / "runs"),
        name="grz_yolo11n",
        exist_ok=True,
        pretrained=True,
        patience=15,
        amp=True,
        cache=False,  # /kaggle/input is read-only
    )
    elapsed = time.perf_counter() - t0
    save_dir = Path(str(results.save_dir))
    best = save_dir / "weights" / "best.pt"
    metrics = {}
    try:
        metrics = {k: float(v) for k, v in dict(results.results_dict).items() if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001
        metrics = {"parse_error": str(exc)}
    return {
        "imgsz": IMGSZ,
        "batch": BATCH,
        "epochs": EPOCHS,
        "elapsed_s": round(elapsed, 1),
        "best_pt": str(best),
        "save_dir": str(save_dir),
        "metrics": metrics,
    }


def export_onnx(weights: Path, imgsz: int, half: bool, out: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=imgsz, half=half, simplify=True, dynamic=False, opset=12)
    exported = Path(str(exported))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, out)
    return out


def eval_on_list(weights: Path, list_name: str, imgsz: int, split_dir: Path) -> dict:
    """Validate on a split list by writing a temporary data.yaml."""
    from ultralytics import YOLO

    list_path = split_dir / list_name
    if not list_path.is_file() or list_path.stat().st_size == 0:
        return {"list": list_name, "skipped": True, "reason": "missing_or_empty"}
    tmp = WORK / f"data_{list_name.replace('.txt', '')}.yaml"
    tmp.write_text(
        f"path: {DATA_ROOT}\ntrain: {split_dir / 'train.txt'}\nval: {list_path}\nnc: 4\n"
        + "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    model = YOLO(str(weights))
    res = model.val(data=str(tmp), imgsz=imgsz, split="val", device=DEVICE, plots=False, cache=False)
    out = {"list": list_name, "imgsz": imgsz}
    try:
        box = res.box
        out["map50"] = float(box.map50)
        out["map50_95"] = float(box.map)
        # per-class AP50 if available
        if getattr(box, "ap50", None) is not None:
            out["ap50_per_class"] = {
                CLASS_NAMES[i]: float(box.ap50[i]) for i in range(min(len(CLASS_NAMES), len(box.ap50)))
            }
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def measure_onnx_latency(onnx_path: Path, imgsz: int, runs: int = 50) -> dict:
    import numpy as np
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as exc:  # noqa: BLE001
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        providers_note = f"cuda_failed:{exc}"
    else:
        providers_note = None
    active = sess.get_providers()
    inp = sess.get_inputs()[0]
    x = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(10):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t0) * 1000)
    out = {
        "onnx": str(onnx_path.name),
        "imgsz": imgsz,
        "providers": active,
        "mean_ms": round(float(np.mean(times)), 3),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
    }
    if providers_note:
        out["note"] = providers_note
    return out


def main() -> None:
    global DEVICE

    os.environ.setdefault("MPLBACKEND", "Agg")
    WORK.mkdir(parents=True, exist_ok=True)
    install_deps()
    disable_ultralytics_raytune()
    DEVICE = resolve_device()
    print("train device:", DEVICE, flush=True)

    assert DATA_ROOT.is_dir(), f"missing dataset {DATA_ROOT}"
    # quick existence probe
    syn0 = DATA_ROOT / "images" / "synthetic" / "syn_42_00000.jpg"
    print("probe", syn0, "exists", syn0.is_file(), flush=True)
    print("input listing sample:", flush=True)
    run(["bash", "-lc", f"ls -la {DATA_ROOT} && ls {DATA_ROOT}/images && ls {DATA_ROOT}/images/synthetic 2>/dev/null | head"])

    split_dir = prepare_absolute_splits()
    data_yaml = write_data_yaml(WORK / "data.yaml", split_dir)

    report: dict = {
        "dataset": str(DATA_ROOT),
        "seed": SEED,
        "device": DEVICE,
        "python": sys.version.split()[0],
        "torch": pkg_version("torch"),
        "numpy": pkg_version("numpy"),
        "gpu": gpu_name(),
        "train": {},
        "domain_gap": {},
        "latency": [],
        "artifacts": {},
    }

    print("=== train ===", flush=True)
    info = train_once(data_yaml)
    report["train"] = info
    primary = None
    best = Path(info["best_pt"])
    if best.is_file():
        dst_pt = WORK / "best.pt"
        shutil.copy2(best, dst_pt)
        report["artifacts"]["best_pt"] = str(dst_pt.name)
        onnx_fp32 = export_onnx(best, info["imgsz"], half=False, out=WORK / "detector_fp32.onnx")
        onnx_fp16 = export_onnx(best, info["imgsz"], half=True, out=WORK / "detector_fp16.onnx")
        report["artifacts"]["onnx_fp32"] = onnx_fp32.name
        report["artifacts"]["onnx_fp16"] = onnx_fp16.name
        report["latency"].append(measure_onnx_latency(onnx_fp32, info["imgsz"]))
        try:
            report["latency"].append(measure_onnx_latency(onnx_fp16, info["imgsz"]))
        except Exception as exc:  # noqa: BLE001
            report["latency"].append({"onnx": onnx_fp16.name, "error": str(exc)})
        primary = best
    else:
        print("WARNING: missing best.pt", flush=True)

    if primary and primary.is_file():
        for list_name in (
            "val.txt",
            "val_real_type1.txt",
            "val_syn_type1a.txt",
            "val_syn_type1a_style_b.txt",
            "val_real_type1b.txt",
        ):
            print(f"=== eval {list_name} ===", flush=True)
            report["domain_gap"][list_name] = eval_on_list(primary, list_name, IMGSZ, split_dir)

    out_json = WORK / "stage4_metrics.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out_json, flush=True)
    print(json.dumps({k: report[k] for k in ("train", "domain_gap") if k in report}, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
