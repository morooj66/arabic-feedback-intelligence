"""OPT-IN local Intent (I10) and Sentiment (S5) predictors from the classification_v2 experimental artifacts.

Selected only by `RCS_CLASSIFICATION_MODE=local_v2_pilot` (see rcs.classification.runtime). The default stack stays the frozen one. This module
changes NO prediction semantics: it computes the same label-free features as the offline study (multilingual-e5-small passage embeddings and the
frozen CAMeLBERT log-probabilities) and applies the saved, hash-verified logistic-regression artifacts.

Failure handling is explicit: a missing / modified artifact, taxonomy mismatch, missing weights or an inference error raises `V2StackError`.
There is NO fallback to the frozen models and no default label - a failed prediction is an error, never a wrong label.

Shared work: Intent (I10) and Sentiment (S5) use the same two feature blocks, so both are computed once per distinct text and kept in a small
bounded cache; the second task (and any repeated text) costs no model inference."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from config import settings
from rcs.classification.candidate_v2 import verify_artifact

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = ROOT / "data" / "models" / "experimental" / "classification_v2"
CAMEL_MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
EXPECTED_FEATURES = ("e5", "camel")           # column order the artifacts were trained with: [e5 (384) | camel log-probs (3)]
CACHE_MAX = 4096
BATCH = 32


class V2StackError(RuntimeError):
    """Any problem with the v2 stack. Never caught and replaced by a frozen / default prediction."""


def _clean(text) -> str:
    """Total function over malformed input: None -> '', non-str -> str(). Never raises, never drops a row (alignment is preserved)."""
    if text is None:
        return ""
    return text if isinstance(text, str) else str(text)


class _Features:
    def __init__(self):
        self._lock = threading.RLock()
        self._e5 = None
        self._cam = None
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "e5_batches": 0, "camel_batches": 0}

    # -- loading -------------------------------------------------------------------------------------------------------------------
    def load(self):
        with self._lock:
            if self._e5 is None:
                try:
                    from rcs.response_intelligence.retrieval.embeddings import E5EmbeddingAdapter

                    ad = E5EmbeddingAdapter()
                    ad.load()
                    self._e5 = ad
                except Exception as exc:  # noqa: BLE001
                    raise V2StackError("embedding model could not be loaded (cached local weights required; nothing is downloaded)") from exc
            if self._cam is None:
                try:
                    import torch

                    from rcs.training.hf_loading import load_sequence_classifier, load_tokenizer
                    from rcs.training.sentiment_pretrained import _normalize_sentiment

                    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                    tok, model = load_tokenizer(CAMEL_MODEL_ID), load_sequence_classifier(CAMEL_MODEL_ID)
                    model.to(device).eval()
                    id2label = {int(k): v for k, v in model.config.id2label.items()}
                    col = []
                    for label in settings.SENTIMENTS:
                        col.append(next(j for j, lbl in id2label.items() if _normalize_sentiment(lbl) == label))
                    self._cam = (tok, model, device, col)
                except Exception as exc:  # noqa: BLE001
                    raise V2StackError("CAMeLBERT sentiment model could not be loaded (cached local weights required)") from exc

    @property
    def loaded(self) -> bool:
        return self._e5 is not None and self._cam is not None

    # -- inference -----------------------------------------------------------------------------------------------------------------
    def _camel(self, texts: list[str]) -> np.ndarray:
        import torch

        tok, model, device, col = self._cam
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        probs = probs[:, col]                                              # project label order, as sentiment_pretrained does
        probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-12, None)
        self.stats["camel_batches"] += 1
        return np.log(np.clip(probs, 1e-6, 1))

    def get(self, texts: list[str]) -> np.ndarray:
        """(n, 387) feature matrix [e5 | camel log-probs] aligned with `texts`; each distinct text is computed once."""
        self.load()
        with self._lock:
            missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
            self.stats["hits"] += len(texts) - len([t for t in texts if t in missing])
            self.stats["misses"] += len(missing)
            try:
                for i in range(0, len(missing), BATCH):
                    chunk = missing[i:i + BATCH]
                    e5 = np.asarray(self._e5.embed_passages(chunk), dtype=float)
                    self.stats["e5_batches"] += 1
                    cam = self._camel(chunk)
                    for t, a, b in zip(chunk, e5, cam):
                        self._cache[t] = (a, b)
            except V2StackError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise V2StackError("feature extraction failed") from exc
            for t in texts:
                self._cache.move_to_end(t)
            # Assemble the answer BEFORE evicting: a single batch may contain more distinct texts than the cache holds.
            result = np.vstack([np.concatenate(self._cache[t]) for t in texts]) if texts else np.zeros((0, 387))
            while len(self._cache) > CACHE_MAX:
                self._cache.popitem(last=False)
            return result


class V2Stack:
    def __init__(self, artifact_dir: Path | None = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else Path(getattr(settings, "V2_ARTIFACT_DIR", None) or DEFAULT_ARTIFACT_DIR)
        self.features = _Features()
        self.models: dict[str, object] = {}
        self._lock = threading.RLock()

    # -- artifacts -----------------------------------------------------------------------------------------------------------------
    def load_artifacts(self):
        with self._lock:
            if self.models:
                return
            import joblib
            import json

            loaded = {}
            for task, taxonomy in (("intent", settings.PRIMARY_INTENTS), ("sentiment", settings.SENTIMENTS)):
                d = self.artifact_dir / task
                try:
                    verify_artifact(d)
                    cfg = json.loads((d / "config.json").read_text())
                    model = joblib.load(d / "model.joblib")
                except Exception as exc:  # noqa: BLE001
                    raise V2StackError(f"{task} artifact is missing, modified or unreadable") from exc
                if cfg.get("status") != "EXPERIMENTAL - NOT ACTIVATED":
                    raise V2StackError(f"{task} artifact has an unexpected status")
                if tuple(cfg.get("extra_features", ())) != EXPECTED_FEATURES:
                    raise V2StackError(f"{task} artifact feature layout differs from the stack's")
                if set(model.labels) != set(taxonomy) or len(model.labels) != len(taxonomy):
                    raise V2StackError(f"{task} artifact labels are not the frozen taxonomy")
                loaded[task] = model
            self.models = loaded

    def load(self):
        self.load_artifacts()
        self.features.load()

    # -- prediction ----------------------------------------------------------------------------------------------------------------
    def _predict(self, task: str, texts: list[str], taxonomy) -> list[tuple[str, float]]:
        if not texts:
            return []
        self.load_artifacts()
        clean = [_clean(t) for t in texts]
        X = self.features.get(clean)
        try:
            model = self.models[task]
            P = model.predict_proba(clean, extra=X)
        except Exception as exc:  # noqa: BLE001
            raise V2StackError(f"{task} prediction failed") from exc
        out = []
        for row in P:
            k = int(np.argmax(row)); conf = float(row[k])
            label = model.labels[k]
            if label not in taxonomy or not math.isfinite(conf) or not (0.0 <= conf <= 1.0 + 1e-9):
                raise V2StackError(f"{task} produced an invalid prediction")
            out.append((str(label), min(conf, 1.0)))
        return out

    def predict_intent(self, texts: list[str]) -> list[tuple[str, float]]:
        return self._predict("intent", list(texts), settings.PRIMARY_INTENTS)

    def predict_sentiment(self, comment_ids: list[str], texts: list[str]) -> dict[str, tuple[str, float]]:
        if len(comment_ids) != len(texts):
            raise V2StackError("comment_ids and texts differ in length")
        preds = self._predict("sentiment", list(texts), settings.SENTIMENTS)
        return {cid: p for cid, p in zip(comment_ids, preds)}


_STACK: V2Stack | None = None
_STACK_LOCK = threading.Lock()


def get_stack() -> V2Stack:
    global _STACK
    with _STACK_LOCK:
        if _STACK is None:
            _STACK = V2Stack()
        return _STACK


def reset_for_testing(stack: V2Stack | None = None) -> None:
    global _STACK
    with _STACK_LOCK:
        _STACK = stack
