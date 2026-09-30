"""Train and compare YOLO12n and YOLO26n on Roboflow dataset version 3.

The two models use the same split, image size, optimizer, epoch budget, seed,
and Kaggle GPU session. All user-facing artifacts are written to
``/kaggle/working``.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DATA_ROOT = Path("/kaggle/input/grz-plates-v3")
WORK = Path("/kaggle/working")
RUNS = WORK / "runs"
ULTRALYTICS_VERSION = "8.4.156"
MODELS = ("yolo12n.pt", "yolo26n.pt")
CLASS_NAMES = ["other", "type1", "type1a", "type1b"]
IMGSZ = 640
BATCH = 16
EPOCHS = 80
PATIENCE = 15
SEED = 42
DEVICE: int | str = 0


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gpu_name() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        return output.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return ""


def drop_torch_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "torch" or module_name.startswith("torch."):
            del sys.modules[module_name]
        elif module_name == "torchvision" or module_name.startswith("torchvision."):
            del sys.modules[module_name]


def install_pascal_torch() -> None:
    """Install the last CUDA wheel used here that still supports P100 sm_60."""
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


def install_dependencies() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    gpu = gpu_name()
    torch_version = package_version("torch")
    torchvision_version = package_version("torchvision")
    numpy_version = package_version("numpy")
    pascal = any(name in gpu for name in ("P100", "K80", "M60"))
    print(
        "environment_before",
        json.dumps(
            {
                "python": sys.version.split()[0],
                "gpu": gpu,
                "torch": torch_version,
                "torchvision": torchvision_version,
                "numpy": numpy_version,
                "pascal": pascal,
            }
        ),
        flush=True,
    )
    if pascal:
        install_pascal_torch()

    constraints = WORK / "pip-constraints.txt"
    pins: list[str] = []
    if numpy_version:
        pins.append(f"numpy=={numpy_version}")
    if not pascal:
        if torch_version:
            pins.append(f"torch=={torch_version}")
        if torchvision_version:
            pins.append(f"torchvision=={torchvision_version}")
    constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        f"ultralytics=={ULTRALYTICS_VERSION}",
        "onnx>=1.17,<2",
    ]
    if pins:
        command.extend(["-c", str(constraints)])
    run(command)
    if pascal:
        install_pascal_torch()
    # Kaggle's preinstalled Ray has historically conflicted with the optional
    # Ultralytics tuning callback; it is not needed for this fixed experiment.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "ray"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print(
        "environment_after",
        json.dumps(
            {
                "ultralytics": package_version("ultralytics"),
                "torch": package_version("torch"),
                "torchvision": package_version("torchvision"),
                "numpy": package_version("numpy"),
            }
        ),
        flush=True,
    )


def resolve_device() -> int | str:
    import torch

    print(
        "cuda_probe",
        json.dumps(
            {
                "torch": torch.__version__,
                "available": torch.cuda.is_available(),
                "architectures": getattr(torch.cuda, "get_arch_list", lambda: [])(),
            }
        ),
        flush=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is enabled but CUDA is unavailable")
    value = torch.ones(1, device="cuda") + 1
    print("cuda_ok", value.device, torch.cuda.get_device_name(0), flush=True)
    return 0


def locate_data_root() -> Path:
    """Support both legacy flat and newer versioned Kaggle input mounts."""
    input_root = Path("/kaggle/input")
    candidates = [
        DATA_ROOT,
        input_root / "grz-plates-detector-roboflow-v3",
    ]
    if input_root.is_dir():
        candidates.extend(path.parent.parent for path in input_root.rglob("train/images"))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "train" / "images").is_dir():
            print("dataset_root", candidate, flush=True)
            return candidate
    listing = sorted(str(path) for path in input_root.glob("**/*") if path.is_dir())[:120]
    raise FileNotFoundError(
        "could not locate train/images under /kaggle/input; directories=" + json.dumps(listing)
    )


def write_data_yaml() -> Path:
    required = {
        "train": DATA_ROOT / "train" / "images",
        "val": DATA_ROOT / "valid" / "images",
        "test": DATA_ROOT / "test" / "images",
    }
    for split, path in required.items():
        if not path.is_dir():
            raise FileNotFoundError(f"missing {split} images: {path}")

    data_yaml = WORK / "data_v3.yaml"
    lines = [
        f"path: {DATA_ROOT}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def split_counts() -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        output[split] = {
            "images": sum(
                1
                for path in (DATA_ROOT / split / "images").iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ),
            "labels": sum(1 for _ in (DATA_ROOT / split / "labels").glob("*.txt")),
        }
    return output


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    names = metrics.names
    class_indices = [int(index) for index in box.ap_class_index.tolist()]
    ap50 = box.ap50.tolist()
    ap5095 = box.ap.tolist()
    per_class: dict[str, dict[str, float | None]] = {}
    for position, class_index in enumerate(class_indices):
        class_name = str(names[class_index])
        per_class[class_name] = {
            "ap50": finite_float(ap50[position]),
            "ap50_95": finite_float(ap5095[position]),
        }
    return {
        "precision": finite_float(box.mp),
        "recall": finite_float(box.mr),
        "map50": finite_float(box.map50),
        "map50_95": finite_float(box.map),
        "per_class": per_class,
        "speed_ms_per_image": {
            key: finite_float(value) for key, value in metrics.speed.items()
        },
        "fitness": finite_float(getattr(metrics, "fitness", None)),
    }


def best_epoch_from_csv(results_csv: Path) -> dict[str, Any]:
    if not results_csv.is_file():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    metric_key = next(
        (key for key in rows[0] if key.strip() == "metrics/mAP50-95(B)"),
        None,
    )
    if metric_key is None:
        return {"epochs_completed": len(rows)}
    best_row = max(rows, key=lambda row: float(row[metric_key]))
    epoch_key = next((key for key in best_row if key.strip() == "epoch"), "epoch")
    return {
        "epochs_completed": len(rows),
        "best_epoch": int(float(best_row[epoch_key])),
        "best_training_val_map50_95": float(best_row[metric_key]),
    }


def benchmark_predict(model: Any, image_paths: list[Path]) -> dict[str, Any]:
    import numpy as np
    import torch

    if not image_paths:
        return {}
    for _ in range(5):
        model.predict(
            source=str(image_paths[0]),
            imgsz=IMGSZ,
            device=DEVICE,
            conf=0.25,
            iou=0.45,
            max_det=10,
            verbose=False,
        )
    torch.cuda.synchronize()
    elapsed_ms: list[float] = []
    for image_path in image_paths:
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            device=DEVICE,
            conf=0.25,
            iou=0.45,
            max_det=10,
            verbose=False,
        )
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "images": len(elapsed_ms),
        "mean_ms": round(float(np.mean(elapsed_ms)), 3),
        "median_ms": round(float(np.median(elapsed_ms)), 3),
        "p95_ms": round(float(np.percentile(elapsed_ms, 95)), 3),
        "fps_from_mean": round(1000.0 / float(np.mean(elapsed_ms)), 3),
        "scope": "single-image end-to-end Ultralytics predict, warm model and filesystem cache",
    }


def export_onnx(model: Any, stem: str) -> dict[str, Any]:
    try:
        exported = Path(
            model.export(
                format="onnx",
                imgsz=IMGSZ,
                opset=12,
                simplify=False,
                dynamic=False,
                half=False,
                device=DEVICE,
            )
        )
        destination = WORK / f"{stem}_best.onnx"
        if exported.resolve() != destination.resolve():
            shutil.copy2(exported, destination)
        return {
            "file": destination.name,
            "bytes": destination.stat().st_size,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"ONNX export failed for {stem}: {exc}", flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


def train_and_evaluate(model_file: str, data_yaml: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    stem = Path(model_file).stem
    started = time.perf_counter()
    model = YOLO(model_file)
    train_metrics = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=4,
        project=str(RUNS),
        name=stem,
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        seed=SEED,
        deterministic=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        save_period=-1,
        verbose=True,
    )
    training_seconds = time.perf_counter() - started
    run_dir = Path(train_metrics.save_dir)
    best_source = run_dir / "weights" / "best.pt"
    if not best_source.is_file():
        raise FileNotFoundError(f"training did not produce {best_source}")
    best_destination = WORK / f"{stem}_best.pt"
    shutil.copy2(best_source, best_destination)
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        shutil.copy2(results_csv, WORK / f"{stem}_results.csv")

    best_model = YOLO(str(best_destination))
    pytorch_module = best_model.model
    parameters = sum(parameter.numel() for parameter in pytorch_module.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in pytorch_module.parameters() if parameter.requires_grad
    )
    model_yaml = getattr(pytorch_module, "yaml", {})
    validation = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=4,
        plots=True,
        project=str(RUNS),
        name=f"{stem}_val",
        exist_ok=True,
        verbose=True,
    )
    test = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=4,
        plots=True,
        project=str(RUNS),
        name=f"{stem}_test",
        exist_ok=True,
        verbose=True,
    )
    test_images = sorted(
        path
        for path in (DATA_ROOT / "test" / "images").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    output = {
        "model": stem,
        "pretrained_weights": model_file,
        "training_seconds": round(training_seconds, 3),
        "training_minutes": round(training_seconds / 60.0, 3),
        "training": best_epoch_from_csv(results_csv),
        "validation": metrics_to_dict(validation),
        "test": metrics_to_dict(test),
        "benchmark": benchmark_predict(best_model, test_images),
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "end2end_architecture": bool(model_yaml.get("end2end", False)),
        "artifacts": {
            "best_pt": best_destination.name,
            "best_pt_bytes": best_destination.stat().st_size,
            "results_csv": f"{stem}_results.csv",
        },
    }
    output["artifacts"]["onnx"] = export_onnx(best_model, stem)
    return output


def main() -> None:
    global DATA_ROOT, DEVICE

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WANDB_DISABLED", "true")
    WORK.mkdir(parents=True, exist_ok=True)
    install_dependencies()
    DEVICE = resolve_device()
    DATA_ROOT = locate_data_root()
    data_yaml = write_data_yaml()

    report: dict[str, Any] = {
        "experiment": "YOLO12n vs YOLO26n on Roboflow 1-djo2q version 3",
        "fairness": "same Kaggle GPU session, data splits, image size, batch, optimizer, seed, and epoch budget",
        "dataset": {
            "root": str(DATA_ROOT),
            "roboflow_version": 3,
            "class_names": CLASS_NAMES,
            "splits": split_counts(),
            "preprocessing": "Auto-orient; resize 640x640 Fill with center crop",
            "baked_augmentations": "3 train variants; rotation -5..+5 degrees; brightness -15..+15 percent",
        },
        "settings": {
            "imgsz": IMGSZ,
            "batch": BATCH,
            "max_epochs": EPOCHS,
            "patience": PATIENCE,
            "seed": SEED,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "amp": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "gpu": gpu_name(),
            "ultralytics": package_version("ultralytics"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
        },
        "models": {},
    }

    report_path = WORK / "comparison_metrics.json"
    for model_file in MODELS:
        stem = Path(model_file).stem
        print(f"=== {stem}: train, validate, test, benchmark, export ===", flush=True)
        report["models"][stem] = train_and_evaluate(model_file, data_yaml)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"checkpointed {report_path}", flush=True)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
