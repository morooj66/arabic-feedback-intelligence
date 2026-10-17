"""
Live inference using the FROZEN model selection
(data/models/selection/FROZEN_MODEL_SELECTION.json), for new/unlabeled rows.

This module does NOT retrain, tune, or reselect anything. It only loads the
already-frozen artifacts and reuses the exact inference code that scored
human_test, instead of duplicating that logic:

  - intent  (tfidf_logistic_intent): loads the already-fitted sklearn
    Pipeline saved at data/models/final_test/intent_tfidf_logistic.joblib
    (produced by scripts/run_frozen_human_test.py's
    fit_frozen_intent_pipeline). Loaded once and reused; never refit here.
  - topic   (marbertv2_topic / UBC-NLP/MARBERTv2): reuses
    find_topic_checkpoint() and evaluate_frozen_topic() imported directly
    from scripts/run_frozen_human_test.py — the same checkpoint-loading and
    batch-inference code that produced the sealed human_test topic scores.
  - sentiment (CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment): reuses
    rcs.training.sentiment_pretrained.evaluate_pretrained_sentiment(), the
    same pretrained-inference function used for human_test.

Each underlying function expects labeled examples (it was built for
evaluation). For live/unlabeled rows we pass a placeholder label — it is
never read except to populate an EvaluationReport this module discards.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from pathlib import Path

from config import settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_FREEZE_JSON = _PROJECT_ROOT / "data" / "models" / "selection" / "FROZEN_MODEL_SELECTION.json"
_DEFAULT_INTENT_JOBLIB = _PROJECT_ROOT / "data" / "models" / "final_test" / "intent_tfidf_logistic.joblib"
_DEFAULT_TOPIC_CHECKPOINT = _PROJECT_ROOT / "data" / "models" / "final_test" / "checkpoints" / "marbertv2_topic" / "best"
_HUMAN_TEST_SCRIPT = _PROJECT_ROOT / "scripts" / "run_frozen_human_test.py"

_intent_pipeline = None
_intent_source = None
_human_test_module = None
_freeze = None
_freeze_source = None
# The HTTP server is threaded while both transformer wrappers load large
# checkpoints inside each call. Serializing their load+forward sections keeps
# two concurrent imports from multiplying the ~0.8 GB CPU peak and exhausting
# a 2 GB production worker. Intent is small/cached and does not need this lock.
_transformer_inference_lock = threading.Lock()


def _configured_path(env_name: str, default: Path) -> Path:
    """Return a deployment-configurable artifact path without relying on cwd.

    Production overrides must be absolute so a Render working-directory
    change cannot silently select a different artifact. Defaults remain the
    existing project-relative frozen locations.
    """
    raw = os.environ.get(env_name)
    if not raw:
        return default
    path = Path(raw)
    if os.environ.get("RCS_ENV", "").strip().lower() in {"production", "prod"} and not path.is_absolute():
        raise ValueError(f"{env_name} must be an absolute path in production")
    return path


def frozen_selection_path() -> Path:
    return _configured_path("RCS_FROZEN_SELECTION_PATH", _DEFAULT_FREEZE_JSON)


def frozen_intent_path() -> Path:
    return _configured_path("RCS_FROZEN_INTENT_PATH", _DEFAULT_INTENT_JOBLIB)


def frozen_topic_path() -> Path:
    return _configured_path("RCS_MARBERTV2_TOPIC_CHECKPOINT", _DEFAULT_TOPIC_CHECKPOINT)


def _load_freeze() -> dict:
    global _freeze, _freeze_source
    path = frozen_selection_path()
    if _freeze is None or _freeze_source != path:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen model-selection metadata not found at {path}")
        _freeze = json.loads(path.read_text())
        _freeze_source = path
    return _freeze


def _load_human_test_module():
    """Import scripts/run_frozen_human_test.py by path, so its
    evaluate_frozen_topic() is reused as-is rather than reimplemented.
    Runtime checkpoint selection stays here so production never consults
    the evaluation script's historical Colab fallback paths."""
    global _human_test_module
    if _human_test_module is None:
        spec = importlib.util.spec_from_file_location(
            "rcs_frozen_human_test_script", _HUMAN_TEST_SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _human_test_module = module
    return _human_test_module


def _load_intent_pipeline():
    global _intent_pipeline, _intent_source
    artifact = frozen_intent_path()
    if _intent_pipeline is None or _intent_source != artifact:
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Frozen intent artifact not found at {artifact}. "
                "Do not refit on the fly here — regenerate it via "
                "scripts/run_frozen_human_test.py (fit_frozen_intent_pipeline) first."
            )
        import joblib

        _intent_pipeline = joblib.load(artifact)
        _intent_source = artifact
    return _intent_pipeline


def predict_intent(texts: list[str]) -> list[tuple[str, float]]:
    """Returns [(primary_intent, confidence), ...] aligned to `texts`."""
    if not texts:
        return []
    pipe = _load_intent_pipeline()
    probs = pipe.predict_proba(texts)
    classes = list(pipe.classes_)
    out = []
    for row in probs:
        idx = int(row.argmax())
        out.append((str(classes[idx]), float(row[idx])))
    return out


def predict_topic(comment_ids: list[str], texts: list[str]) -> dict[str, tuple[str, float]]:
    """Returns {comment_id: (topic_subtype, confidence)}."""
    if not comment_ids:
        return {}
    module = _load_human_test_module()
    from rcs.training.gold_splits import LabeledExample

    checkpoint = frozen_topic_path()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"Frozen MARBERTv2 topic checkpoint is incomplete at {checkpoint}. "
            "Expected config.json and model.safetensors; do not retrain or substitute weights."
        )
    examples = [
        LabeledExample(
            comment_id=cid,
            text=text,
            label="GENERAL",  # placeholder; discarded, never read for prediction
            annotation_stage="frozen_inference_live",
            split="live",
        )
        for cid, text in zip(comment_ids, texts)
    ]
    with _transformer_inference_lock:
        result = module.evaluate_frozen_topic(examples, checkpoint)
    out = {}
    for cid, label, probs in zip(result["comment_ids"], result["y_pred"], result["probabilities"]):
        out[cid] = (label, float(max(probs)) if probs else None)
    return out


def predict_sentiment(comment_ids: list[str], texts: list[str]) -> dict[str, tuple[str, float]]:
    """Returns {comment_id: (sentiment, confidence)}."""
    if not comment_ids:
        return {}
    from rcs.training.gold_splits import LabeledExample
    from rcs.training.sentiment_pretrained import evaluate_pretrained_sentiment

    freeze = _load_freeze()
    examples = [
        LabeledExample(
            comment_id=cid,
            text=text,
            label="NEUTRAL",  # placeholder; discarded, never read for prediction
            annotation_stage="frozen_inference_live",
            split="live",
        )
        for cid, text in zip(comment_ids, texts)
    ]
    with _transformer_inference_lock:
        result = evaluate_pretrained_sentiment(
            examples,
            model_id=freeze["sentiment"]["model_id"],
            batch_size=int(freeze["sentiment"]["preprocessing"]["batch_size"]),
        )
    out = {}
    probs = result.probabilities or [None] * len(result.comment_ids)
    for cid, label, row in zip(result.comment_ids, result.y_pred, probs):
        out[cid] = (label, float(max(row)) if row else None)
    return out
