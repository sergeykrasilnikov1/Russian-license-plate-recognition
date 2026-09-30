"""Execute the packaged production CLI offline on a small real input slice."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

runtime_matches = [path.parent.parent for path in INPUT.glob("**/scripts/run_inference.py")]
if not runtime_matches:
    archives = list(INPUT.glob("**/runtime.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"expected unpacked runtime or one runtime.zip, got {archives}")
    unpacked = WORK / "runtime_bundle"
    shutil.unpack_archive(archives[0], unpacked)
    runtime_matches = [path.parent.parent for path in unpacked.glob("**/scripts/run_inference.py")]
data_matches = [path.parent for path in INPUT.glob("**/meta.csv") if (path.parent / "images").is_dir()]
if len(runtime_matches) != 1 or len(data_matches) != 1:
    raise RuntimeError(f"unexpected mounts: runtime={runtime_matches}, data={data_matches}")
runtime, data = runtime_matches[0], data_matches[0]
output = WORK / "integrated_smoke.csv"

# Kaggle's minimal CPU image does not include Ultralytics. Install the exact
# evaluated wheels from the private bundle, without contacting a package index.
wheels = sorted((runtime / "wheels").glob("*.whl"))
if len(wheels) != 2:
    raise RuntimeError(f"expected two bundled runtime wheels, got {wheels}")
setup_command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    "--no-index",
    "--no-deps",
    *(str(path) for path in wheels),
]
setup = subprocess.run(setup_command, text=True, capture_output=True, check=False)
if setup.stdout:
    print(setup.stdout, flush=True)
if setup.stderr:
    print(setup.stderr, file=sys.stderr, flush=True)
if setup.returncode:
    raise RuntimeError(f"offline wheel setup exited with code {setup.returncode}")

command = [
    sys.executable,
    str(runtime / "scripts/run_inference.py"),
    "--input",
    str(data / "images"),
    "--output",
    str(output),
    "--config",
    str(runtime / "configs/pipeline.yaml"),
    "--ocr-config",
    str(runtime / "configs/ocr.yaml"),
    "--limit",
    "20",
    "--warmup",
    "1",
]
completed = subprocess.run(command, cwd=runtime, text=True, capture_output=True, check=False)
print(completed.stdout, flush=True)
if completed.stderr:
    print(completed.stderr, file=sys.stderr, flush=True)
if completed.returncode:
    raise RuntimeError(f"production CLI exited with code {completed.returncode}")

lines = output.read_text(encoding="utf-8").splitlines()
if not lines or lines[0] != "image;plate_num;plate_type;confidence":
    raise AssertionError(f"invalid output header: {lines[:1]}")
summary = {
    "setup_command": setup_command,
    "command": command,
    "stdout": completed.stdout.strip(),
    "header": lines[0],
    "rows": len(lines) - 1,
    "internet_enabled": False,
    "passed": True,
}
(WORK / "integrated_smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
