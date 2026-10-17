"""Tests for Kaggle GPU setup script (no GPU required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_setup():
    path = PROJECT_ROOT / "kaggle" / "setup_gpu_env.py"
    spec = importlib.util.spec_from_file_location("setup_gpu_env", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_torch_build_matrix_p100():
    mod = _load_setup()
    build = mod.select_torch_build("6.0")
    assert build["index_url"] == "https://download.pytorch.org/whl/cu118"
    assert build["torch_version"] == "2.5.1"
    assert build["sm_tag"] == "sm_60"


def test_torch_build_matrix_t4():
    mod = _load_setup()
    build = mod.select_torch_build("7.5")
    assert build["index_url"] == "https://download.pytorch.org/whl/cu121"
    assert build["torch_version"] == "2.5.1"
    assert build["sm_tag"] == "sm_75"


def test_torch_build_unknown_fallback():
    mod = _load_setup()
    build = mod.select_torch_build(None)
    assert build["index_url"] == "https://download.pytorch.org/whl/cu118"


def test_requirements_excludes_torch():
    text = (PROJECT_ROOT / "kaggle" / "requirements.txt").read_text(encoding="utf-8")
    assert "torch>=" not in text
    assert "torchvision" not in text
    assert "torchaudio" not in text
