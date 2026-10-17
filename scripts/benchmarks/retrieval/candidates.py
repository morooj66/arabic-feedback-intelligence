"""
Phase M5B.2 — candidate retrieval runners for the benchmark ONLY.

Every runner exposes the same tiny interface: `setup()` (corpus
embedding/indexing, timed separately) and `rank(query_text) ->
list[candidate_id]` (a full ranking, best match first). No runner here
is wired into `rcs.response_intelligence.retrieval` or any production
code path.

Execution status vocabulary (never silently substituted):
    EXECUTED
    NOT_EXECUTED_DEPENDENCY_MISSING
    NOT_EXECUTED_DOWNLOAD_BLOCKED
    FAILED_RUNTIME

No new production dependency is used here. Embeddings are extracted via
the ALREADY-installed `transformers`/`torch` (both already in
requirements.txt for the frozen classifiers) using raw
AutoTokenizer/AutoModel + manual mean-pooling -- `sentence-transformers`
is deliberately NOT installed or imported, so this benchmark cannot
create a dependency-ownership question for production requirements.txt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

EXECUTED = "EXECUTED"
NOT_EXECUTED_DEPENDENCY_MISSING = "NOT_EXECUTED_DEPENDENCY_MISSING"
NOT_EXECUTED_DOWNLOAD_BLOCKED = "NOT_EXECUTED_DOWNLOAD_BLOCKED"
FAILED_RUNTIME = "FAILED_RUNTIME"


@dataclass
class CandidateReport:
    """One candidate's full benchmark outcome. `rankings` maps
    query_id -> ranked list of candidate_ids (empty if not executed).
    No raw vectors are stored here -- only IDs and scores, per
    instruction ("do not store embeddings themselves")."""

    name: str
    kind: str  # "lexical_baseline" | "embedding"
    status: str
    rankings: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)  # query_id -> {candidate_id: score}
    setup_seconds: float | None = None
    total_query_seconds: float | None = None
    embedding_dimension: int | None = None
    error_detail: str | None = None
    model_identifier: str | None = None
    model_revision: str | None = None


class TfidfLexicalBaseline:
    """LEXICAL BASELINE -- not an embedding candidate. Deterministic,
    zero download, uses the already-installed scikit-learn. Included
    per instruction to measure whether semantic embeddings actually
    improve enough over simple term-overlap matching to justify their
    resource cost."""

    name = "tfidf_lexical_baseline"
    kind = "lexical_baseline"

    def __init__(self, candidate_ids: list[str], candidate_texts: list[str]):
        self._candidate_ids = candidate_ids
        self._candidate_texts = candidate_texts
        self._vectorizer = None
        self._matrix = None

    def setup(self) -> float:
        from sklearn.feature_extraction.text import TfidfVectorizer

        start = time.perf_counter()
        # analyzer="char_wb" (character n-grams) is deliberately used instead
        # of the default whitespace-token analyzer: Arabic has rich
        # morphology (attached prefixes/suffixes like و/ال/ها), and
        # character n-grams tolerate small spelling variation far better
        # than exact whitespace-token matching would for a fair lexical
        # baseline -- this is standard practice for Arabic TF-IDF, not a
        # thumb on the scale for any candidate.
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = self._vectorizer.fit_transform(self._candidate_texts)
        return time.perf_counter() - start

    def rank(self, query_text: str) -> tuple[list[str], dict[str, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self._vectorizer.transform([query_text])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = np.argsort(-sims)
        ranked_ids = [self._candidate_ids[i] for i in order]
        scores = {self._candidate_ids[i]: float(sims[i]) for i in order}
        return ranked_ids, scores


class TransformerMeanPoolEmbeddingCandidate:
    """Generic mean-pooled sentence embedding via raw
    AutoTokenizer/AutoModel -- covers both benchmark embedding
    candidates (they differ only in prefix convention, per each
    model's documented usage)."""

    kind = "embedding"

    def __init__(
        self,
        name: str,
        model_identifier: str,
        model_revision: str,
        candidate_ids: list[str],
        candidate_texts: list[str],
        *,
        query_prefix: str = "",
        passage_prefix: str = "",
        max_length: int = 64,
    ):
        self.name = name
        self.model_identifier = model_identifier
        self.model_revision = model_revision
        self._candidate_ids = candidate_ids
        self._candidate_texts = candidate_texts
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._candidate_embeddings: np.ndarray | None = None
        self.embedding_dimension: int | None = None

    def setup(self) -> float:
        import torch
        from transformers import AutoModel, AutoTokenizer

        start = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_identifier, revision=self.model_revision)
        self._model = AutoModel.from_pretrained(self.model_identifier, revision=self.model_revision)
        self._model.eval()
        prefixed = [self._passage_prefix + t for t in self._candidate_texts]
        self._candidate_embeddings = self._embed(prefixed)
        self.embedding_dimension = self._candidate_embeddings.shape[1]
        return time.perf_counter() - start

    def _mean_pool(self, last_hidden_state, attention_mask):
        import torch

        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            encoded = self._tokenizer(
                texts, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
            )
            output = self._model(**encoded)
            pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            return normalized.numpy()

    def rank(self, query_text: str) -> tuple[list[str], dict[str, float]]:
        q_emb = self._embed([self._query_prefix + query_text])[0]
        sims = self._candidate_embeddings @ q_emb  # both L2-normalized -> dot product == cosine similarity
        order = np.argsort(-sims)
        ranked_ids = [self._candidate_ids[i] for i in order]
        scores = {self._candidate_ids[i]: float(sims[i]) for i in order}
        return ranked_ids, scores
