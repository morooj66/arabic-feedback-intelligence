#!/usr/bin/env python3
"""
Google Colab GPU environment bootstrap for RCS MARBERTv2 training.

Colab may ship a PyTorch build incompatible with assigned GPU (e.g. cu128 vs P100).
Detect compute capability via nvidia-smi and install a matching torch wheel.

Usage:
  Cell 1: python colab/setup_gpu_env.py --install   # then restart runtime
  Cell 2: python colab/setup_gpu_env.py --verify
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKER_PATH = Path("/content/.rcs_torch_install.json")

TORCH_BUILD_MATRIX: dict[str, tuple[str, str]] = {
    "6.0": ("https://download.pytorch.org/whl/cu118", "2.5.1"),  # P100
    "7.5": ("https://download.pytorch.org/whl/cu121", "2.5.1"),  # T4
}
TORCH_FALLBACK = ("https://download.pytorch.org/whl/cu118", "2.5.1")

ML_PACKAGES = (
    "transformers>=4.40,<5",
    "datasets>=2.18,<4",
    "accelerate>=0.28,<2",
    "scikit-learn>=1.3,<1.7",
    "pandas>=2.0,<3",
    "numpy>=1.24",
    "pydantic>=2.0",
    "safetensors>=0.4",
)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def detect_gpu() -> dict:
    info: dict = {
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "gpus": [],
        "primary_compute_cap": None,
        "primary_name": None,
    }
    if not info["nvidia_smi"]:
        return info

    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        info["error"] = proc.stderr.strip() or proc.stdout.strip()
        return info

    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu = {
            "index": parts[0],
            "name": parts[1],
            "compute_cap": parts[2],
            "memory_total_mib": parts[3],
        }
        info["gpus"].append(gpu)
        if info["primary_compute_cap"] is None:
            info["primary_compute_cap"] = gpu["compute_cap"]
            info["primary_name"] = gpu["name"]
    return info


def select_torch_build(compute_cap: str | None) -> dict:
    cap = (compute_cap or "").strip()
    index_url, torch_version = TORCH_BUILD_MATRIX.get(cap, TORCH_FALLBACK)
    sm = cap.replace(".", "")
    return {
        "compute_cap": cap or "unknown",
        "sm_tag": f"sm_{sm}" if cap else "unknown",
        "torch_version": torch_version,
        "index_url": index_url,
        "reason": (
            f"matched Colab matrix for compute capability {cap}"
            if cap in TORCH_BUILD_MATRIX
            else "fallback cu118 (P100/T4 safe; avoids Colab cu128 sm_60 issues)"
        ),
    }


def install_stack(gpu_info: dict, build: dict) -> None:
    _run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        check=False,
    )
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"torch=={build['torch_version']}",
            "--index-url",
            build["index_url"],
        ]
    )
    _run([sys.executable, "-m", "pip", "install", *ML_PACKAGES])

    marker = MARKER_PATH if MARKER_PATH.parent.exists() else PROJECT_ROOT / ".rcs_torch_install.json"
    payload = {
        "platform": "google_colab",
        "gpu": gpu_info,
        "torch_build": build,
        "restart_required": True,
        "ml_packages": list(ML_PACKAGES),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        "\n>>> RESTART RUNTIME NOW (Runtime → Restart runtime), then run:\n"
        "    python colab/setup_gpu_env.py --verify\n"
        "    python colab/preflight.py\n",
        flush=True,
    )


def verify_cuda() -> dict:
    snippet = """
import json
import torch
x = torch.tensor([1.0, 2.0, 3.0])
if torch.cuda.is_available():
    x = x.cuda()
    y = x * 2
    cap = torch.cuda.get_device_capability(0)
    out = {
        "ok": True,
        "torch": torch.__version__,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "cuda_tensor_sum": float(y.sum().item()),
    }
else:
    out = {"ok": False, "torch": torch.__version__, "cuda_available": False}
print(json.dumps(out))
"""
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    result = json.loads(proc.stdout.strip())
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit("CUDA verification failed — restart runtime after --install")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab GPU-compatible PyTorch setup.")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()

    gpu_info = detect_gpu()
    build = select_torch_build(gpu_info.get("primary_compute_cap"))
    report = {"platform": "google_colab", "gpu": gpu_info, "torch_build": build}

    if args.print_plan:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.install:
        if not gpu_info.get("gpus"):
            raise SystemExit(f"No GPU detected: {gpu_info.get('error', 'enable GPU runtime')}")
        install_stack(gpu_info, build)
        return
    if args.verify:
        verify_cuda()
        return

    print(json.dumps(report, indent=2, ensure_ascii=False))
    parser.print_help()


if __name__ == "__main__":
    main()
