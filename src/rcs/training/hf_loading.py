"""
Hugging Face model loading helpers.

Prefer safetensors weights (CVE-2025-32434): torch<2.6 cannot safely load
pytorch_model.bin via transformers. Callers must pass use_safetensors=True for
all AutoModel* loads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _PROJECT_ROOT / "config" / "frozen_models_manifest.json"
_REVISION_MARKER = ".revision"


class PinnedModelUnavailable(RuntimeError):
    """A model that the manifest pins is not present locally and must not be fetched from the Hub (production). Never answered with a different model."""


def _pinned_entry(model_id: str) -> dict | None:
    try:
        manifest = json.loads(_MANIFEST_PATH.read_text())
    except (OSError, ValueError):
        return None
    for entry in manifest.get("public_models") or []:
        if entry.get("repo_id") == model_id:
            return entry
    return None


def _is_production() -> bool:
    return os.environ.get("RCS_ENV", "").strip().lower() in {"production", "prod"}


def _configured_local_path(entry: dict) -> Path:
    """Resolve the verified local copy, with an explicit CAMeLBERT override.

    The override changes only the directory, never the pinned revision,
    expected filenames, sizes, or fail-closed behavior from the manifest.
    """
    override = os.environ.get("RCS_CAMELBERT_SENTIMENT_PATH")
    if override and entry.get("repo_id") == "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment":
        path = Path(override)
        if _is_production() and not path.is_absolute():
            raise PinnedModelUnavailable("RCS_CAMELBERT_SENTIMENT_PATH must be an absolute path in production")
        return path
    return _PROJECT_ROOT / entry["local_dir"]


def resolve_model_source(model_id: str) -> tuple[str, dict[str, Any]]:
    """Where to load `model_id` from, and extra from_pretrained kwargs.

    A model the manifest pins (config/frozen_models_manifest.json -> public_models) is loaded from its verified local directory when that is complete at the pinned
    revision. If it is missing: in production this FAILS (no Hub download at request time, no floating "main", no substitute); in development it falls back to the Hub
    at the SAME pinned revision. Models the manifest does not mention are loaded exactly as before."""
    entry = _pinned_entry(model_id)
    if entry is None:
        return model_id, {}
    local = _configured_local_path(entry)
    marker = local / _REVISION_MARKER
    complete = marker.exists() and marker.read_text().strip() == entry["revision"] and all(
        (local / f["name"]).exists() and (local / f["name"]).stat().st_size == f["size_bytes"] for f in entry["files"]
    )
    if complete:
        return str(local), {}
    if _is_production():
        raise PinnedModelUnavailable(
            f"Pinned model '{model_id}' (revision {entry['revision']}) is not present at {entry['local_dir']}. Run scripts/fetch_frozen_models.py during the build; "
            "production never downloads or substitutes models at request time."
        )
    return model_id, {"revision": entry["revision"]}


def load_tokenizer(model_id: str, **kwargs: Any):
    from transformers import AutoTokenizer

    source, extra = resolve_model_source(model_id)
    return AutoTokenizer.from_pretrained(source, **{**extra, **kwargs})


def load_sequence_classifier(model_id: str, **kwargs: Any):
    """Load a sequence-classification head; requires safetensors when available."""
    from transformers import AutoModelForSequenceClassification

    if "use_safetensors" not in kwargs:
        kwargs["use_safetensors"] = True
    source, extra = resolve_model_source(model_id)
    return AutoModelForSequenceClassification.from_pretrained(source, **{**extra, **kwargs})
