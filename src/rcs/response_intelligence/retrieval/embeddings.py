"""
Phase M5B.3 — E5 embedding adapter for Approved Response Memory retrieval.

Model: intfloat/multilingual-e5-small, pinned to the exact commit SHA
M5B.2's benchmark evaluated (614241f622f53c4eeff9890bdc4f31cfecc418b3) —
not a moving branch, never silently substituted. MiniLM is deliberately
not used here, per instruction (M5B.2 found it underperformed the
lexical baseline on every metric).

Reuses the exact mean-pooling + attention-mask-weighting + L2-
normalization approach already validated in M5B.2's benchmark
(scripts/benchmarks/retrieval/candidates.py's
TransformerMeanPoolEmbeddingCandidate) via the already-installed
transformers/torch — no sentence-transformers, no new production
dependency.

This module knows nothing about corpus eligibility, sanitization, or
indexing (see corpus.py/approved_memory.py) — it only turns text into
vectors, and it never falls back to a different model. If the model
cannot be loaded, `load()`/`embed_*` raise `EmbeddingModelUnavailable`;
the caller (approved_memory.py) is responsible for turning that into the
public UNAVAILABLE retrieval outcome.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

E5_MODEL_IDENTIFIER = "intfloat/multilingual-e5-small"
E5_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "
_MAX_LENGTH = 64


class EmbeddingModelUnavailable(Exception):
    """Raised for any embedding load/inference failure. The message is
    intentionally short and generic (never the raw underlying exception
    text or a filesystem path) so it is always safe to surface as a
    public `failure_detail` without a caller needing to sanitize it
    again."""


class EmbeddingAdapter(Protocol):
    """Narrow interface approved_memory.py depends on -- lets unit tests
    substitute a deterministic fake instead of downloading a real model
    (per instruction: unit tests must never download Hugging Face
    models). `E5EmbeddingAdapter` below is the only production
    implementation."""

    def load(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...

    @property
    def embedding_dimension(self) -> int | None: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class E5EmbeddingAdapter:
    """Loads intfloat/multilingual-e5-small once via `load()`, then
    embeds text via mean-pooling + L2 normalization. Stateless with
    respect to any corpus -- knows nothing about eligibility, corpus
    records, or an index."""

    model_identifier = E5_MODEL_IDENTIFIER
    model_revision = E5_MODEL_REVISION

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def embedding_dimension(self) -> int | None:
        if self._model is None:
            return None
        return int(self._model.config.hidden_size)

    def load(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EmbeddingModelUnavailable("embedding dependency not importable") from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_identifier, revision=self.model_revision)
            model = AutoModel.from_pretrained(self.model_identifier, revision=self.model_revision)
            model.eval()
        except Exception as exc:  # noqa: BLE001 -- any load failure => UNAVAILABLE, never crash, never substitute
            raise EmbeddingModelUnavailable("approved response memory embedding model could not be loaded") from exc

        self._tokenizer = tokenizer
        self._model = model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([_PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([_QUERY_PREFIX + text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None or self._tokenizer is None:
            raise EmbeddingModelUnavailable("embed() called before a successful load()")

        try:
            import torch
            import torch.nn.functional as F

            with torch.no_grad():
                encoded = self._tokenizer(
                    texts, padding=True, truncation=True, max_length=_MAX_LENGTH, return_tensors="pt"
                )
                output = self._model(**encoded)
                pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                return normalized.numpy()
        except EmbeddingModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 -- never leak a raw inference exception to the caller
            raise EmbeddingModelUnavailable("approved response memory embedding inference failed") from exc
