"""
Phase M5B.3 — tests for the real Approved Response Memory retrieval
implementation (rcs.response_intelligence.retrieval.{corpus,embeddings,
approved_memory}).

No Hugging Face model is ever downloaded here: all embedding-dependent
tests use `FakeEmbeddingAdapter`, a deterministic in-test double
satisfying `embeddings.EmbeddingAdapter`. Every fixture value is clearly
synthetic Arabic placeholder text, never anything resembling real
organizational/production data, matching the existing convention in
tests/test_retrieval_contracts.py.
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from rcs.response_intelligence.retrieval import approved_memory as approved_memory_module
from rcs.response_intelligence.retrieval import corpus as corpus_module
from rcs.response_intelligence.retrieval import embeddings as embeddings_module
from rcs.response_intelligence.retrieval.approved_memory import (
    ApprovedResponseIndex,
    ApprovedResponseMemoryRetrieverImpl,
    build_approved_response_memory_retriever,
    build_index,
)
from rcs.response_intelligence.retrieval.contracts import (
    ERROR,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    TIMEOUT,
    UNAVAILABLE,
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryResult,
)
from rcs.response_intelligence.retrieval.corpus import (
    ApprovedResponseCorpusRecord,
    RawApprovedResponseCandidate,
    build_governed_corpus,
)
from rcs.response_intelligence.retrieval.embeddings import E5EmbeddingAdapter, EmbeddingModelUnavailable, _mean_pool

_TRUSTED_SOURCE = "synthetic_trusted_source.xlsx"
_UNTRUSTED_SOURCE = "synthetic_untrusted_source.xlsx"
_TRUSTED_ALLOWLIST = frozenset({_TRUSTED_SOURCE})
_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)


def _raw(**overrides) -> RawApprovedResponseCandidate:
    review_item_id = overrides.get("review_item_id", "synthetic-item-1")
    base = dict(
        review_item_id=review_item_id,
        source_file=_TRUSTED_SOURCE,
        review_status="APPROVED",
        reviewer_action="SEND_AS_IS",
        final_reply="شكرًا لتواصلكم معنا حول هذه الكلمة.",
        dedupe_canonical_id=review_item_id,  # defaults to "this row IS canonical"; override to test otherwise
        approved_at=_SYNTHETIC_NOW,
        primary_intent="INQUIRY",
        topic_subtype="GENERAL",
        sentiment="NEUTRAL",
    )
    base.update(overrides)
    return RawApprovedResponseCandidate(**base)


class FakeEmbeddingAdapter:
    """Deterministic double for embeddings.EmbeddingAdapter. Each text
    is looked up by exact-substring key against a fixed table of
    already-L2-normalized vectors supplied at construction; unknown text
    maps to a zero vector. No torch, no transformers, no network."""

    def __init__(self, vectors: dict[str, np.ndarray] | None = None, dim: int = 4):
        self._vectors = vectors or {}
        self._dim = dim
        self._loaded = False
        self.load_calls = 0
        self.embed_passages_calls = 0
        self.embed_query_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def embedding_dimension(self) -> int | None:
        return self._dim if self._loaded else None

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.embed_passages_calls += 1
        return np.stack([self._lookup(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.embed_query_calls += 1
        return self._lookup(text)

    def _lookup(self, text: str) -> np.ndarray:
        for key, vec in self._vectors.items():
            if key in text:
                return vec
        return np.zeros(self._dim)


class UnavailableEmbeddingAdapter:
    """Always raises EmbeddingModelUnavailable on load()."""

    is_loaded = False
    embedding_dimension = None

    def load(self) -> None:
        raise EmbeddingModelUnavailable("synthetic: model unavailable in test")

    def embed_passages(self, texts):
        raise EmbeddingModelUnavailable("synthetic: model unavailable in test")

    def embed_query(self, text):
        raise EmbeddingModelUnavailable("synthetic: model unavailable in test")


class QueryTimeUnavailableEmbeddingAdapter(FakeEmbeddingAdapter):
    """Loads and embeds the corpus fine, but fails only at query time —
    exercises the UNAVAILABLE path inside retrieve(), not build_index."""

    def embed_query(self, text):
        raise EmbeddingModelUnavailable("synthetic: query-time model unavailable")


class ErroringEmbeddingAdapter(FakeEmbeddingAdapter):
    """Raises a generic (non-EmbeddingModelUnavailable) exception at
    query time, to exercise the sanitized ERROR path."""

    def embed_query(self, text):
        raise RuntimeError("synthetic: unexpected internal failure with a fake /file/path/leak")


class SlowEmbeddingAdapter(FakeEmbeddingAdapter):
    """Deliberately slow, but still a normal synchronous completion --
    used to prove this implementation never mislabels a slow-but-
    completed call as TIMEOUT (see M5B.3a's timeout-semantics
    correction: there is no enforceable deadline here, so we do not
    pretend there is one)."""

    def embed_query(self, text):
        import time

        time.sleep(0.05)
        return super().embed_query(text)


# --- 1-7. Structural eligibility ------------------------------------------


def test_empty_allowlist_yields_zero_eligible_corpus_entries():
    records = build_governed_corpus([_raw()], allowlist=frozenset())
    assert records == ()


def test_untrusted_source_excluded():
    records = build_governed_corpus([_raw(source_file=_UNTRUSTED_SOURCE)], allowlist=_TRUSTED_ALLOWLIST)
    assert records == ()


def test_approved_trusted_source_eligible():
    records = build_governed_corpus([_raw()], allowlist=_TRUSTED_ALLOWLIST)
    assert len(records) == 1
    assert records[0].review_item_id == "synthetic-item-1"


def test_blank_final_reply_excluded():
    records = build_governed_corpus([_raw(final_reply="   ")], allowlist=_TRUSTED_ALLOWLIST)
    assert records == ()
    records_none = build_governed_corpus([_raw(final_reply=None)], allowlist=_TRUSTED_ALLOWLIST)
    assert records_none == ()


def test_noncanonical_duplicate_excluded():
    records = build_governed_corpus(
        [_raw(review_item_id="synthetic-item-2", dedupe_canonical_id="synthetic-item-1")],
        allowlist=_TRUSTED_ALLOWLIST,
    )
    assert records == ()


def test_null_dedupe_canonical_id_excluded():
    """NULL means dedup hasn't run for this row -- fails closed, not
    assumed safe (see corpus.py module docstring point 4)."""
    records = build_governed_corpus([_raw(dedupe_canonical_id=None)], allowlist=_TRUSTED_ALLOWLIST)
    assert records == ()


def test_unknown_reviewer_action_excluded():
    for bad_action in ("approve", "edit_approve", "NO_RESPONSE_NEEDED", "SOMETHING_MADE_UP", None):
        records = build_governed_corpus([_raw(reviewer_action=bad_action)], allowlist=_TRUSTED_ALLOWLIST)
        assert records == (), f"reviewer_action={bad_action!r} should be excluded"


def test_unapproved_review_status_excluded():
    for bad_status in ("UNPROCESSED", "IN_REVIEW", "DRAFT_READY", "SENT", "NO_REPLY", "DUPLICATE_CONTENT"):
        records = build_governed_corpus([_raw(review_status=bad_status)], allowlist=_TRUSTED_ALLOWLIST)
        assert records == (), f"review_status={bad_status!r} should be excluded"


# --- 8-11. PII / redaction / identity ------------------------------------


def test_pii_redaction_discipline_applied_before_corpus_embedding():
    raw = _raw(final_reply="تواصل معنا على الرقم 0512345678 لمزيد من التفاصيل.")
    records = build_governed_corpus([raw], allowlist=_TRUSTED_ALLOWLIST)
    assert len(records) == 1
    assert "0512345678" not in records[0].response_text
    assert "[phone number removed]" in records[0].response_text


def test_visitor_name_and_email_never_fields_on_corpus_record_or_result():
    corpus_field_names = {f.name for f in fields(ApprovedResponseCorpusRecord)}
    result_field_names = {f.name for f in fields(ApprovedResponseMemoryResult)}
    raw_field_names = {f.name for f in fields(RawApprovedResponseCandidate)}
    for banned in ("visitor_name", "visitor_email", "reviewer_email", "reviewer_display_name"):
        assert banned not in corpus_field_names
        assert banned not in result_field_names
        assert banned not in raw_field_names


def test_original_historical_comment_never_stored_in_corpus():
    """RawApprovedResponseCandidate has no field to even carry a raw
    original_comment -- structurally impossible to leak, not merely
    unused."""
    raw_field_names = {f.name for f in fields(RawApprovedResponseCandidate)}
    corpus_field_names = {f.name for f in fields(ApprovedResponseCorpusRecord)}
    assert "original_comment" not in raw_field_names
    assert "original_comment" not in corpus_field_names


def test_identifying_context_never_treated_as_approved_response_memory():
    """The existing trusted_grounding_text/build_identifying_grounding
    mechanism (prompt.py) must never be referenced by this module family
    -- it is out of scope per contracts.py's own correction 2."""
    for module in (corpus_module, embeddings_module, approved_memory_module):
        source = inspect.getsource(module)
        assert "trusted_grounding_text" not in source
        assert "build_identifying_grounding" not in source
        assert "IdentifyingContext" not in source


# --- 12-15. Structured filtering ------------------------------------------


def _built_retriever(raw_candidates, vectors=None, **kwargs):
    embedder = FakeEmbeddingAdapter(vectors or {})
    retriever = build_approved_response_memory_retriever(
        raw_candidates, embedder, allowlist=_TRUSTED_ALLOWLIST, **kwargs
    )
    return retriever, embedder


def test_structured_intent_filtering():
    raw = [
        _raw(review_item_id="i1", primary_intent="INQUIRY", final_reply="رد بخصوص الاستفسار الأول."),
        _raw(review_item_id="i2", primary_intent="COMPLAINT", final_reply="رد بخصوص الشكوى الثانية."),
    ]
    retriever, _ = _built_retriever(raw)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="استفسار", max_results=5, intent="COMPLAINT"))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert [r.review_item_id for r in result.results] == ["i2"]


def test_structured_topic_filtering():
    raw = [
        _raw(review_item_id="i1", topic_subtype="GENERAL", final_reply="رد عام بخصوص الموضوع الأول."),
        _raw(review_item_id="i2", topic_subtype="TECHNICAL_ISSUE", final_reply="رد بخصوص مشكلة تقنية ثانية."),
    ]
    retriever, _ = _built_retriever(raw)
    result = retriever.retrieve(
        ApprovedResponseMemoryQuery(query_text="مشكلة", max_results=5, topic="TECHNICAL_ISSUE")
    )
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert [r.review_item_id for r in result.results] == ["i2"]


def test_structured_sentiment_filtering():
    raw = [
        _raw(review_item_id="i1", sentiment="POSITIVE", final_reply="رد إيجابي بخصوص الموضوع الأول."),
        _raw(review_item_id="i2", sentiment="NEGATIVE", final_reply="رد بخصوص موضوع سلبي ثاني."),
    ]
    retriever, _ = _built_retriever(raw)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="رد", max_results=5, sentiment="NEGATIVE"))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert [r.review_item_id for r in result.results] == ["i2"]


def test_filter_miss_returns_success_empty_without_touching_embedder():
    raw = [_raw(review_item_id="i1", primary_intent="INQUIRY")]
    retriever, embedder = _built_retriever(raw)
    result = retriever.retrieve(
        ApprovedResponseMemoryQuery(query_text="غير موجود", max_results=5, intent="SPAM")
    )
    assert result.outcome == SUCCESS_EMPTY
    assert result.results == ()
    assert embedder.embed_query_calls == 0  # never rank when the filter matches nothing


# --- 16-19. Top-K / ordering / outcome invariants -------------------------


def test_max_results_enforced():
    unit = np.array([1.0, 0.0, 0.0, 0.0])
    vectors = {"a": unit, "b": unit, "c": unit}
    raw = [
        _raw(review_item_id="i1", final_reply="رد يحتوي على الحرف a في وسط النص التجريبي."),
        _raw(review_item_id="i2", final_reply="رد يحتوي على الحرف b في وسط النص التجريبي."),
        _raw(review_item_id="i3", final_reply="رد يحتوي على الحرف c في وسط النص التجريبي."),
    ]
    retriever, _ = _built_retriever(raw, vectors=vectors)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="a", max_results=2))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert len(result.results) == 2


def test_deterministic_ordering_by_score_then_stable_id():
    high = np.array([1.0, 0.0, 0.0, 0.0])
    low = np.array([0.6, 0.8, 0.0, 0.0])
    vectors = {"HIGH": high, "LOWB": low, "LOWA": low, "query": high}
    raw = [
        _raw(review_item_id="z-item", final_reply="رد HIGH بخصوص الموضوع."),
        _raw(review_item_id="b-item", final_reply="رد LOWB بخصوص الموضوع."),
        _raw(review_item_id="a-item", final_reply="رد LOWA بخصوص الموضوع."),
    ]
    retriever, _ = _built_retriever(raw, vectors=vectors)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="query", max_results=3))
    assert result.outcome == SUCCESS_WITH_RESULTS
    # z-item has the strictly higher score; a-item/b-item tie and must
    # break by review_item_id ascending ("a-item" before "b-item").
    assert [r.review_item_id for r in result.results] == ["z-item", "a-item", "b-item"]
    # Re-running must produce the identical order (no set/dict-ordering luck).
    result2 = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="query", max_results=3))
    assert [r.review_item_id for r in result2.results] == [r.review_item_id for r in result.results]


def test_success_with_results_invariant_never_empty():
    raw = [_raw()]
    retriever, _ = _built_retriever(raw)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert len(result.results) >= 1


def test_empty_governed_corpus_returns_success_empty():
    retriever, embedder = _built_retriever([])
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert result.outcome == SUCCESS_EMPTY
    assert result.results == ()
    assert embedder.load_calls == 0  # never even attempt to load the model for an empty corpus


# --- 20-21. Failure outcomes ------------------------------------------------


def test_embedding_unavailable_at_build_time_returns_unavailable():
    retriever = build_approved_response_memory_retriever(
        [_raw()], UnavailableEmbeddingAdapter(), allowlist=_TRUSTED_ALLOWLIST
    )
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert result.outcome == UNAVAILABLE
    assert result.failure_detail is not None
    assert "/" not in result.failure_detail  # never a filesystem path


def test_embedding_unavailable_at_query_time_returns_unavailable():
    embedder = QueryTimeUnavailableEmbeddingAdapter()
    retriever = build_approved_response_memory_retriever([_raw()], embedder, allowlist=_TRUSTED_ALLOWLIST)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert result.outcome == UNAVAILABLE


def test_sanitized_error_behavior_never_leaks_raw_exception_text():
    embedder = ErroringEmbeddingAdapter()
    retriever = build_approved_response_memory_retriever([_raw()], embedder, allowlist=_TRUSTED_ALLOWLIST)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert result.outcome == ERROR
    assert result.failure_detail is not None
    assert "/file/path/leak" not in result.failure_detail
    assert "RuntimeError" not in result.failure_detail


# --- M5B.3a: honest TIMEOUT semantics ---------------------------------
#
# This implementation has no enforceable deadline (synchronous,
# in-process, no thread/process cancellation boundary), so it must
# NEVER retroactively label a slow-but-completed call TIMEOUT -- that
# would misleadingly imply a bound that does not exist. The tests below
# prove that removal, not just describe it.


def test_slow_synchronous_call_still_completes_and_never_returns_timeout():
    embedder = SlowEmbeddingAdapter()
    retriever = build_approved_response_memory_retriever([_raw()], embedder, allowlist=_TRUSTED_ALLOWLIST)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert result.outcome != TIMEOUT


def test_public_api_no_longer_accepts_a_timeout_budget_parameter():
    """Structural proof, not just a docstring claim: neither the
    factory nor the retriever dataclass exposes a timeout knob that
    would suggest a deadline is being enforced."""
    import inspect

    factory_params = set(inspect.signature(build_approved_response_memory_retriever).parameters)
    assert "timeout_seconds" not in factory_params

    retriever_field_names = {f.name for f in fields(ApprovedResponseMemoryRetrieverImpl)}
    assert "timeout_seconds" not in retriever_field_names


def test_approved_memory_module_never_references_timeout_outcome():
    """Static proof that TIMEOUT is not merely unreachable in practice
    but genuinely never emitted by this module's source."""
    code = _code_excluding_docstrings(inspect.getsourcefile(approved_memory_module))
    assert "TIMEOUT" not in code


def test_timeout_outcome_still_valid_in_the_shared_contract():
    """contracts.py (M5B.1) is untouched by this correction -- TIMEOUT
    remains a legitimate RetrievalOutcome for a future provider that can
    genuinely enforce a deadline."""
    from rcs.response_intelligence.retrieval.contracts import TIMEOUT as CONTRACT_TIMEOUT
    from rcs.response_intelligence.retrieval.contracts import RetrievalOutcome

    assert CONTRACT_TIMEOUT == "TIMEOUT"
    assert "TIMEOUT" in RetrievalOutcome.__args__


# --- 22-25. Embedding adapter correctness ---------------------------------


def test_exact_e5_query_and_passage_prefixes(monkeypatch):
    adapter = E5EmbeddingAdapter()
    captured: dict[str, list[str]] = {}

    def fake_embed(self, texts):
        captured["texts"] = list(texts)
        return np.zeros((len(texts), 4))

    monkeypatch.setattr(E5EmbeddingAdapter, "_embed", fake_embed)

    adapter.embed_passages(["نص أول", "نص ثاني"])
    assert captured["texts"] == ["passage: نص أول", "passage: نص ثاني"]

    adapter.embed_query("استفسار")
    assert captured["texts"] == ["query: استفسار"]


def test_e5_model_identifier_and_revision_are_pinned_and_match_m5b2():
    assert embeddings_module.E5_MODEL_IDENTIFIER == "intfloat/multilingual-e5-small"
    assert embeddings_module.E5_MODEL_REVISION == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert "minilm" not in embeddings_module.E5_MODEL_IDENTIFIER.lower()


def test_mean_pooling_correctness_on_deterministic_fixture():
    import torch

    # 2 tokens, 2 dims, batch of 1: token 1 = [2, 4], token 2 = [4, 8] (padding, masked out).
    last_hidden_state = torch.tensor([[[2.0, 4.0], [4.0, 8.0]]])
    attention_mask = torch.tensor([[1, 0]])
    pooled = _mean_pool(last_hidden_state, attention_mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 4.0]]))


def test_l2_normalization_produces_unit_vectors():
    import torch
    import torch.nn.functional as F

    pooled = torch.tensor([[3.0, 4.0]])  # norm = 5
    normalized = F.normalize(pooled, p=2, dim=1)
    assert torch.allclose(normalized, torch.tensor([[0.6, 0.8]]))
    assert torch.allclose(normalized.norm(p=2, dim=1), torch.tensor([1.0]))


def test_cosine_dot_ranking_correctness_with_normalized_vectors():
    query = np.array([1.0, 0.0, 0.0, 0.0])
    close = np.array([0.8, 0.6, 0.0, 0.0])  # unit vector, cos sim with query = 0.8
    far = np.array([0.0, 1.0, 0.0, 0.0])  # orthogonal, cos sim = 0.0
    vectors = {"CLOSE": close, "FAR": far, "query": query}
    raw = [
        _raw(review_item_id="close-item", final_reply="رد CLOSE بخصوص الموضوع."),
        _raw(review_item_id="far-item", final_reply="رد FAR بخصوص الموضوع."),
    ]
    retriever, _ = _built_retriever(raw, vectors=vectors)
    result = retriever.retrieve(ApprovedResponseMemoryQuery(query_text="query", max_results=2))
    assert [r.review_item_id for r in result.results] == ["close-item", "far-item"]
    assert result.results[0].retrieval_score == pytest.approx(0.8)
    assert result.results[1].retrieval_score == pytest.approx(0.0)


# --- 26-27. Index lifecycle -------------------------------------------------


def test_index_built_once_not_rebuilt_per_query():
    raw = [_raw()]
    retriever, embedder = _built_retriever(raw)
    for _ in range(5):
        retriever.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert embedder.embed_passages_calls == 1
    assert embedder.load_calls == 1
    assert embedder.embed_query_calls == 5


def test_explicit_rebuild_produces_independent_retriever_and_index():
    embedder = FakeEmbeddingAdapter()
    retriever_v1 = build_approved_response_memory_retriever(
        [_raw(review_item_id="only-in-v1")], embedder, allowlist=_TRUSTED_ALLOWLIST
    )
    embedder_v2 = FakeEmbeddingAdapter()
    retriever_v2 = build_approved_response_memory_retriever(
        [_raw(review_item_id="only-in-v2")], embedder_v2, allowlist=_TRUSTED_ALLOWLIST
    )
    result_v1 = retriever_v1.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    result_v2 = retriever_v2.retrieve(ApprovedResponseMemoryQuery(query_text="أي نص", max_results=5))
    assert [r.review_item_id for r in result_v1.results] == ["only-in-v1"]
    assert [r.review_item_id for r in result_v2.results] == ["only-in-v2"]
    assert retriever_v1.index is not retriever_v2.index


def test_build_index_never_loads_embedder_for_empty_records():
    embedder = FakeEmbeddingAdapter()
    index = build_index((), embedder)
    assert index.record_count == 0
    assert embedder.load_calls == 0


# --- Static isolation guards (28-35) ---------------------------------------


def _code_excluding_docstrings(path: str) -> str:
    """AST-based docstring stripper -- avoids the recurring false
    positive (hit in M5A.1, M5A.3, M5B.2) where a module's OWN docstring
    legitimately names a banned term to document its absence."""
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                for lineno in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                    idx = lineno - 1
                    if 0 <= idx < len(lines):
                        lines[idx] = "\n"
    return "".join(lines)


def _core_module_files() -> list[str]:
    return [
        inspect.getsourcefile(corpus_module),
        inspect.getsourcefile(embeddings_module),
        inspect.getsourcefile(approved_memory_module),
    ]


def test_no_generic_retriever_abstraction_introduced():
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        assert "Retriever[" not in code
        assert "class Retriever" not in code
        assert "Generic[" not in code


def test_no_official_knowledge_result_usage():
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        assert "OfficialKnowledge" not in code


def test_no_sqlalchemy_or_pymongo_in_pure_retrieval_core():
    banned = ("sqlalchemy", "pymongo", "rcs.review.models", "ReviewItem")
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        for term in banned:
            assert term not in code, f"forbidden reference {term!r} found in {path}"


def test_no_gemini_or_orchestrator_runtime_wiring():
    banned = ("google.genai", "genai_client", "orchestrator", "GenerationRequest", "generate_suggested_response")
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        for term in banned:
            assert term not in code, f"forbidden reference {term!r} found in {path}"


def test_no_external_dictionary_integration():
    for path in _core_module_files():
        code = _code_excluding_docstrings(path).lower()
        assert "external_dictionary_client" not in code


def test_no_vector_database_dependency():
    banned = ("pgvector", "pinecone", "chromadb", "elasticsearch", "faiss", "atlas", "weaviate", "qdrant")
    for path in _core_module_files():
        code = _code_excluding_docstrings(path).lower()
        for term in banned:
            assert term not in code, f"forbidden vector-db reference {term!r} found in {path}"


def test_no_secrets_in_core_modules():
    banned = ("GEMINI_API_KEY", "api_key=", "API_KEY", "SECRET", "password")
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        for term in banned:
            assert term not in code, f"unexpected secret-shaped reference {term!r} found in {path}"


def test_no_production_data_fixture_loaded_by_this_test_file():
    """This test file must never read an external fixture file (a real
    exported Excel/JSON) -- every fixture is an inline Python literal.
    Excludes this function's own body, which necessarily names the
    banned patterns as string literals to check for them."""
    code = Path(__file__).read_text(encoding="utf-8")
    body = code.split("def test_no_production_data_fixture_loaded_by_this_test_file")[0]
    for banned in ("read_excel(", "open(", "pd.read_csv(", "read_json("):
        assert banned not in body
