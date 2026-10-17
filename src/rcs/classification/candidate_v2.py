"""OFFLINE loader/featurizer for the classification v2 candidates (intent, topic, sentiment). EXPERIMENTAL - NOT imported by any runtime path and
NOT activated. Features are computed exactly as in scripts/offline_eval/classification_v2: multilingual-e5-small passage embeddings (pinned
revision) and the frozen pretrained CAMeLBERT sentiment log-probabilities (inference only); both are label-free."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

CANDIDATE_FAMILY = "classification_v2"


def verify_artifact(model_dir: Path) -> None:
    """Refuse a modified artifact: every file must match SHA256SUMS."""
    model_dir = Path(model_dir)
    for line in (model_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if hashlib.sha256((model_dir / name.strip()).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name.strip()} does not match SHA256SUMS")


def load_candidate(model_dir: Path):
    model_dir = Path(model_dir)
    verify_artifact(model_dir)
    return joblib.load(model_dir / "model.joblib"), json.loads((model_dir / "config.json").read_text())


def featurize(texts: list[str], names: tuple[str, ...]) -> np.ndarray | None:
    """Dense extra columns in the order the model was trained with. Needs the cached local weights (no download is attempted)."""
    if not names:
        return None
    parts = []
    if "e5" in names:
        from rcs.response_intelligence.retrieval.embeddings import E5EmbeddingAdapter

        ad = E5EmbeddingAdapter(); ad.load()
        parts.append(("e5", np.asarray(ad.embed_passages(list(texts)), dtype=float)))
    if "camel" in names:
        from rcs.training.gold_splits import LabeledExample
        from rcs.training.sentiment_pretrained import evaluate_pretrained_sentiment

        ex = [LabeledExample(comment_id=str(i), text=t, label="NEUTRAL", annotation_stage="live", split="live") for i, t in enumerate(texts)]
        res = evaluate_pretrained_sentiment(ex, model_id="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment", batch_size=16)
        parts.append(("camel", np.log(np.clip(np.asarray(res.probabilities, dtype=float), 1e-6, 1))))
    by = dict(parts)
    return np.hstack([by[n] for n in names])


def predict_proba(model_dir: Path, texts: list[str]) -> np.ndarray:
    model, cfg = load_candidate(model_dir)
    return model.predict_proba(texts, extra=featurize(texts, tuple(cfg["extra_features"])))
