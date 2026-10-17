"""Non-training checks for institutional GPU helper scripts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_preflight_build_report():
    mod = _load_script("gpu_preflight.py")
    report = mod.build_report()
    assert "python" in report
    assert "gpus" in report
    assert "torch" in report
    assert "ready_for_cuda_training" in report


def test_gpu_preflight_cli_runs():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gpu_preflight.py")],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    payload = json.loads(proc.stdout)
    assert payload["platform"]
    # Exit 1 is expected on Mac/no CUDA; stdout must still be valid JSON.


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="torch/transformers not installed locally",
)
def test_gpu_smoke_test_importable():
    mod = _load_script("gpu_smoke_test.py")
    assert mod.MARBERT_MODEL_ID == "UBC-NLP/MARBERTv2"
