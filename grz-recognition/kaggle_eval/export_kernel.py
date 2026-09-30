"""Kaggle GPU kernel: export Stage-4 detector stub to ONNX (writable copy)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
IMGSZ = 640
OPSET = 12


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
    pascal = any(x in gpu for x in ("P100", "K80", "M60"))
    print("gpu", gpu, "pascal", pascal, "torch", pkg_version("torch"), flush=True)
    if pascal:
        install_pascal_torch()
    numpy_v = pkg_version("numpy")
    constraint = WORK / "pip-constraints.txt"
    pins = [f"numpy=={numpy_v}"] if numpy_v else []
    constraint.write_text("\n".join(pins) + ("\n" if pins else ""), encoding="utf-8")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.40", "onnx==1.17.0"]
    if pins:
        cmd.extend(["-c", str(constraint)])
    run(cmd)
    if pascal:
        install_pascal_torch()
    run([sys.executable, "-m", "pip", "uninstall", "-y", "ray"])


def find_pt() -> Path:
    names = {"detector_best.pt", "best.pt"}
    hits: list[Path] = []
    for root in (Path("/kaggle/input"), Path("/kaggle/src"), Path.cwd(), Path(__file__).resolve().parent):
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            if p.name in names:
                hits.append(p)
    print("pt_hits", [str(h) for h in hits], flush=True)
    if not hits:
        raise FileNotFoundError("detector_best.pt not found")
    return hits[0]


def main() -> int:
    install_deps()
    src = find_pt()
    local = WORK / "detector_best.pt"
    shutil.copy2(src, local)
    print("copied", src, "->", local, "bytes", local.stat().st_size, flush=True)

    from ultralytics import YOLO

    model = YOLO(str(local))
    exported = model.export(format="onnx", imgsz=IMGSZ, opset=OPSET, simplify=True, dynamic=False)
    exported = Path(str(exported))
    dest = WORK / "detector.onnx"
    if exported.resolve() != dest.resolve():
        shutil.copy2(exported, dest)
    print("onnx", dest, "bytes", dest.stat().st_size, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
