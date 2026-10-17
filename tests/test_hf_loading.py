"""Tests for Hugging Face safetensors-only loading policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_LOAD_PATHS = (
    PROJECT_ROOT / "src/rcs/training/hf_loading.py",
    PROJECT_ROOT / "src/rcs/training/transformer.py",
    PROJECT_ROOT / "src/rcs/training/sentiment_pretrained.py",
    PROJECT_ROOT / "scripts/gpu_smoke_test.py",
    PROJECT_ROOT / "colab/preflight.py",
)


def test_no_direct_model_from_pretrained_outside_hf_loading():
    """Training paths must not call AutoModel*.from_pretrained without hf_loading."""
    forbidden_snippets = (
        "AutoModelForSequenceClassification.from_pretrained",
        "AutoModel.from_pretrained",
    )
    hf_loading = PROJECT_ROOT / "src/rcs/training/hf_loading.py"
    for path in TRAINING_LOAD_PATHS:
        if not path.exists() or path == hf_loading:
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in text, f"{path.name} must use rcs.training.hf_loading"


def test_hf_loading_forces_use_safetensors():
    import sys
    import types

    from rcs.training import hf_loading

    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model
    fake_transformers = types.SimpleNamespace(
        AutoModelForSequenceClassification=mock_cls,
    )
    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        result = hf_loading.load_sequence_classifier("UBC-NLP/MARBERTv2", num_labels=6)

    assert result is mock_model
    assert mock_cls.from_pretrained.call_args.kwargs["use_safetensors"] is True


def test_hf_loading_respects_explicit_use_safetensors_false():
    import sys
    import types

    from rcs.training import hf_loading

    mock_cls = MagicMock()
    fake_transformers = types.SimpleNamespace(
        AutoModelForSequenceClassification=mock_cls,
    )
    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        hf_loading.load_sequence_classifier("test-model", use_safetensors=False)
    assert mock_cls.from_pretrained.call_args.kwargs["use_safetensors"] is False


def test_hf_loading_source_policy():
    text = (PROJECT_ROOT / "src/rcs/training/hf_loading.py").read_text(encoding="utf-8")
    assert 'kwargs["use_safetensors"] = True' in text


@pytest.mark.parametrize(
    "model_id,expect_hub_safetensors",
    [
        ("CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment", False),
    ],
)
def test_hub_safetensors_availability(model_id, expect_hub_safetensors):
    import urllib.request

    url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            available = resp.status == 200
    except Exception:
        available = False
    assert available is expect_hub_safetensors
