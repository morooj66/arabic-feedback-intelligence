#!/usr/bin/env python3
"""
Hardware preflight for institutional NVIDIA GPU training.

Read-only: reports GPU model, VRAM, CUDA driver, Python, and PyTorch CUDA
availability. Does not download models or start training.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        msg = getattr(exc, "output", None) or str(exc)
        return 1, str(msg).strip()


def _nvidia_gpus() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    code, out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        return [{"error": out}]
    gpus: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mib": parts[2],
                "memory_free_mib": parts[3],
                "driver_version": parts[4],
            }
        )
    return gpus


def _cuda_toolkit_version() -> str | None:
    if shutil.which("nvcc") is None:
        return None
    code, out = _run(["nvcc", "--version"])
    if code != 0:
        return None
    for line in out.splitlines():
        if "release" in line.lower():
            return line.strip()
    return out.splitlines()[-1] if out else None


def _torch_info() -> dict:
    snippet = """
import json
try:
    import torch
    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["device_total_memory_gib"] = round(props.total_memory / (1024**3), 2)
    print(json.dumps(info))
except Exception as exc:
    print(json.dumps({"installed": False, "error": str(exc)}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "installed": False,
            "error": (proc.stderr or proc.stdout or "torch probe failed").strip(),
        }
    try:
        return json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"installed": False, "error": proc.stdout.strip() or "invalid torch probe output"}


def build_report() -> dict:
    gpus = _nvidia_gpus()
    torch_info = _torch_info()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "gpus": gpus,
        "cuda_toolkit": _cuda_toolkit_version(),
        "torch": torch_info,
        "ready_for_cuda_training": bool(
            gpus and torch_info.get("installed") and torch_info.get("cuda_available")
        ),
    }


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ready_for_cuda_training"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
