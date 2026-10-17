"""
Phase M5B.4 — tests for the real Official Knowledge retrieval
implementation (rcs.response_intelligence.retrieval.{official_corpus,
official_knowledge}).

No Hugging Face model is ever downloaded here: all embedding-dependent
tests use `FakeEmbeddingAdapter`, a deterministic in-test double
satisfying `embeddings.EmbeddingAdapter` (mirrors the one in
tests/test_approved_response_memory.py; not shared across files since
this codebase has no shared test-utils module). Every fixture value is
clearly synthetic Arabic placeholder text.
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from rcs.response_intelligence.retrieval import official_corpus as official_corpus_module
from rcs.response_intelligence.retrieval import official_knowledge as official_knowledge_module
from rcs.response_intelligence.retrieval.contracts import (
    ERROR,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    UNAVAILABLE,
    OfficialKnowledgeQuery,
    OfficialKnowledgeResult,
)
from rcs.response_intelligence.retrieval.embeddings import E5EmbeddingAdapter, EmbeddingModelUnavailable
from rcs.response_intelligence.retrieval.official_corpus import (
    OfficialCorpusRecord,
    OfficialSourceLoadError,
    RawOfficialSourceRecord,
    build_governed_official_corpus,
)
from rcs.response_intelligence.retrieval.official_knowledge import (
    OfficialKnowledgeIndex,
    OfficialKnowledgeRetrieverImpl,
    build_index,
    build_official_knowledge_retriever,
)

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)


def _raw(**overrides) -> RawOfficialSourceRecord:
    base = dict(
        source_id="OK-1",
        source_title="عنوان تجريبي اصطناعي",
        source_type="dictionary_entry",
        content="هذا نص تجريبي اصطناعي يوضح تعريف الكلمة المعتمد.",
        version="v1",
        effective_from=_SYNTHETIC_NOW,
        effective_to=None,
        authoritative=True,
        provenance_tag="official_approved",
        entry_id="E-1",
        lemma="كلمة",
        topic="GENERAL",
        intent="INQUIRY",
    )
    base.update(overrides)
    return RawOfficialSourceRecord(**base)


class FakeEmbeddingAdapter:
    """Deterministic double for embeddings.EmbeddingAdapter."""

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
    is_loaded = False
    embedding_dimension = None

    def load(self):
        raise EmbeddingModelUnavailable("synthetic: model unavailable in test")

    def embed_passages(self, texts):
        raise EmbeddingModelUnavailable("synthetic: model unavailable in test")

    def embed_query(self, text):
        raise EmbeddingModelUnavailable("synthetic: model unavailable in test")


class QueryTimeUnavailableEmbeddingAdapter(FakeEmbeddingAdapter):
    def embed_query(self, text):
        raise EmbeddingModelUnavailable("synthetic: query-time model unavailable")


class ErroringEmbeddingAdapter(FakeEmbeddingAdapter):
    def embed_query(self, text):
        raise RuntimeError("synthetic: unexpected internal failure with a fake /file/path/leak")


class FailingSourceProvider:
    def fetch_records(self):
        raise OfficialSourceLoadError("synthetic: source unavailable in test")


class InMemorySourceProvider:
    def __init__(self, records):
        self._records = list(records)

    def fetch_records(self):
        return tuple(self._records)


def _built_retriever(raw_candidates, vectors=None, clock=None):
    embedder = FakeEmbeddingAdapter(vectors or {})
    governed = build_governed_official_corpus(raw_candidates)
    index = build_index(governed, embedder)
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    retriever = OfficialKnowledgeRetrieverImpl(index=index, embedder=embedder, **kwargs)
    return retriever, embedder


# --- 1. No configured source ------------------------------------------


def test_no_configured_source_returns_success_empty():
    embedder = FakeEmbeddingAdapter()
    retriever = build_official_knowledge_retriever(None, embedder)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="أي نص", max_results=5))
    assert result.outcome == SUCCESS_EMPTY
    assert embedder.load_calls == 0


# --- 2 & 40. Synthetic/test/fixture source never authoritative --------


@pytest.mark.parametrize("bad_tag", [None, "synthetic", "test", "test_fixture", "draft", ""])
def test_missing_or_wrong_provenance_tag_excluded(bad_tag):
    records = build_governed_official_corpus([_raw(provenance_tag=bad_tag)])
    assert records == ()


# --- 3. Authoritative record accepted -----------------------------------


def test_authoritative_official_record_accepted():
    records = build_governed_official_corpus([_raw()])
    assert len(records) == 1
    assert records[0].source_id == "OK-1"


# --- 4. Identifying context alone is not official knowledge ------------


def test_identifying_context_alone_never_treated_as_official_knowledge():
    for module in (official_corpus_module, official_knowledge_module):
        source = inspect.getsource(module)
        assert "trusted_grounding_text" not in source
        assert "build_identifying_grounding" not in source
        assert "IdentifyingContext" not in source


# --- 5-8. Structured lookup vs semantic retrieval -----------------------


def test_exact_entry_id_lookup():
    raw = [
        _raw(source_id="OK-1", entry_id="E-1", lemma="اولى", content="محتوى الإدخال الأول."),
        _raw(source_id="OK-2", entry_id="E-2", lemma="ثانية", content="محتوى الإدخال الثاني."),
    ]
    retriever, embedder = _built_retriever(raw)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="E-2", max_results=5))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert [r.source_id for r in result.results] == ["OK-2"]
    assert embedder.embed_query_calls == 0


def test_exact_lemma_lookup():
    raw = [
        _raw(source_id="OK-1", entry_id="E-1", lemma="اولى", content="محتوى الإدخال الأول."),
        _raw(source_id="OK-2", entry_id="E-2", lemma="ثانية", content="محتوى الإدخال الثاني."),
    ]
    retriever, embedder = _built_retriever(raw)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="ثانية", max_results=5))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert [r.source_id for r in result.results] == ["OK-2"]
    assert embedder.embed_query_calls == 0


def test_exact_lookup_avoids_embedding_call():
    raw = [_raw(source_id="OK-1", entry_id="E-1")]
    retriever, embedder = _built_retriever(raw)
    retriever.retrieve(OfficialKnowledgeQuery(query_text="E-1", max_results=5))
    assert embedder.embed_query_calls == 0


def test_semantic_query_uses_embedding_adapter_when_no_exact_key_matches():
    unit = np.array([1.0, 0.0, 0.0, 0.0])
    raw = [_raw(source_id="OK-1", entry_id="E-1", lemma="كلمة", content="نص يحتوي على KEY داخل الجملة.")]
    retriever, embedder = _built_retriever(raw, vectors={"KEY": unit, "query": unit})
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="query", max_results=5))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert embedder.embed_query_calls == 1


# --- 9-10. Structured filters / top-K -----------------------------------


def test_structured_intent_and_topic_filters_respected():
    raw = [
        _raw(source_id="OK-1", intent="INQUIRY", topic="GENERAL", content="محتوى أول."),
        _raw(source_id="OK-2", intent="COMPLAINT", topic="TECHNICAL_ISSUE", content="محتوى ثانٍ."),
    ]
    retriever, _ = _built_retriever(raw)
    result = retriever.retrieve(
        OfficialKnowledgeQuery(query_text="محتوى", max_results=5, intent="COMPLAINT", topic="TECHNICAL_ISSUE")
    )
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert [r.source_id for r in result.results] == ["OK-2"]


def test_max_results_enforced():
    unit = np.array([1.0, 0.0, 0.0, 0.0])
    raw = [
        _raw(source_id="OK-1", entry_id="E-1", content="محتوى يحتوي حرف A فريد."),
        _raw(source_id="OK-2", entry_id="E-2", content="محتوى يحتوي حرف B فريد."),
        _raw(source_id="OK-3", entry_id="E-3", content="محتوى يحتوي حرف C فريد."),
    ]
    retriever, _ = _built_retriever(raw, vectors={"A": unit, "B": unit, "C": unit, "query": unit})
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="query", max_results=2))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert len(result.results) == 2


# --- 11-12. Deterministic ranking / ties --------------------------------


def test_deterministic_semantic_ranking():
    query = np.array([1.0, 0.0, 0.0, 0.0])
    close = np.array([0.8, 0.6, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0, 0.0])
    raw = [
        _raw(source_id="close-item", entry_id="E-close", content="محتوى CLOSE للمقارنة."),
        _raw(source_id="far-item", entry_id="E-far", content="محتوى FAR للمقارنة."),
    ]
    retriever, _ = _built_retriever(raw, vectors={"CLOSE": close, "FAR": far, "query": query})
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="query", max_results=2))
    assert [r.source_id for r in result.results] == ["close-item", "far-item"]
    assert result.results[0].retrieval_score == pytest.approx(0.8)
    assert result.results[1].retrieval_score == pytest.approx(0.0)


def test_deterministic_tie_break_by_source_id():
    high = np.array([1.0, 0.0, 0.0, 0.0])
    tie = np.array([0.6, 0.8, 0.0, 0.0])
    raw = [
        _raw(source_id="z-item", entry_id="E-z", content="محتوى HIGH للمقارنة."),
        _raw(source_id="b-item", entry_id="E-b", content="محتوى TIE ب."),
        _raw(source_id="a-item", entry_id="E-a", content="محتوى TIE أ."),
    ]
    retriever, _ = _built_retriever(raw, vectors={"HIGH": high, "TIE": tie, "query": high})
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="query", max_results=3))
    assert [r.source_id for r in result.results] == ["z-item", "a-item", "b-item"]


# --- 13-18. Provenance retained ------------------------------------------


def _one_result(**raw_overrides) -> OfficialKnowledgeResult:
    raw = _raw(**raw_overrides)
    retriever, _ = _built_retriever([raw])
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text=raw.entry_id, max_results=5))
    assert result.outcome == SUCCESS_WITH_RESULTS
    return result.results[0]


def test_source_id_retained():
    assert _one_result(source_id="OK-provenance-1").source_id == "OK-provenance-1"


def test_source_title_retained():
    assert _one_result(source_title="عنوان محفوظ").source_title == "عنوان محفوظ"


def test_source_type_retained():
    assert _one_result(source_type="dictionary_entry").source_type == "dictionary_entry"


def test_version_retained_when_present():
    assert _one_result(version="v2026-01").version == "v2026-01"


def test_effective_dates_retained_when_present():
    to_date = _SYNTHETIC_NOW + dt.timedelta(days=365)
    result = _one_result(effective_from=_SYNTHETIC_NOW, effective_to=to_date)
    assert result.effective_from == _SYNTHETIC_NOW
    assert result.effective_to == to_date


def test_provenance_metadata_bundle_retained_together():
    result = _one_result(
        source_id="OK-bundle", source_title="حزمة", source_type="dictionary_entry", version="v9"
    )
    assert (result.source_id, result.source_title, result.source_type, result.version) == (
        "OK-bundle",
        "حزمة",
        "dictionary_entry",
        "v9",
    )


# --- 19. Authoritative flag cannot be invented --------------------------


@pytest.mark.parametrize("bad_value", [False, None, "yes", 1])
def test_authoritative_cannot_be_invented(bad_value):
    records = build_governed_official_corpus([_raw(authoritative=bad_value)])
    assert records == ()


def test_authoritative_is_always_literal_true_in_results_never_copied_from_raw():
    result = _one_result(authoritative=True)
    assert result.authoritative is True
    # OfficialCorpusRecord itself has no `authoritative` field to copy from --
    # structurally impossible to invent a False/other value downstream.
    assert "authoritative" not in {f.name for f in fields(OfficialCorpusRecord)}


# --- 20-22. Failure/empty outcomes ---------------------------------------


def test_empty_source_returns_success_empty():
    embedder = FakeEmbeddingAdapter()
    retriever = build_official_knowledge_retriever(InMemorySourceProvider([]), embedder)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="أي نص", max_results=5))
    assert result.outcome == SUCCESS_EMPTY
    assert embedder.load_calls == 0


def test_source_load_error_returns_unavailable():
    embedder = FakeEmbeddingAdapter()
    retriever = build_official_knowledge_retriever(FailingSourceProvider(), embedder)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="أي نص", max_results=5))
    assert result.outcome == UNAVAILABLE
    assert result.failure_detail is not None
    assert "/" not in result.failure_detail


def test_embedding_model_unavailable_at_build_time_returns_unavailable():
    governed = build_governed_official_corpus([_raw()])
    index = build_index(governed, UnavailableEmbeddingAdapter())
    retriever = OfficialKnowledgeRetrieverImpl(index=index, embedder=UnavailableEmbeddingAdapter())
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="نص لا يطابق أي مفتاح", max_results=5))
    assert result.outcome == UNAVAILABLE


def test_embedding_model_unavailable_at_query_time_returns_unavailable():
    governed = build_governed_official_corpus([_raw()])
    embedder = QueryTimeUnavailableEmbeddingAdapter()
    index = build_index(governed, embedder)
    retriever = OfficialKnowledgeRetrieverImpl(index=index, embedder=embedder)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="نص لا يطابق أي مفتاح", max_results=5))
    assert result.outcome == UNAVAILABLE


# --- 23-24. Sanitized ERROR / no raw exception leakage -------------------


def test_sanitized_error_behavior_never_leaks_raw_exception_text():
    governed = build_governed_official_corpus([_raw()])
    embedder = ErroringEmbeddingAdapter()
    index = build_index(governed, embedder)
    retriever = OfficialKnowledgeRetrieverImpl(index=index, embedder=embedder)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="نص لا يطابق أي مفتاح", max_results=5))
    assert result.outcome == ERROR
    assert result.failure_detail is not None
    assert "/file/path/leak" not in result.failure_detail
    assert "RuntimeError" not in result.failure_detail


# --- 25-26. No visitor/reviewer identity fields ---------------------------


def test_no_visitor_or_reviewer_identity_fields_anywhere():
    banned = ("visitor_name", "visitor_email", "reviewer_id", "reviewer_email", "reviewer_display_name")
    for cls in (RawOfficialSourceRecord, OfficialCorpusRecord, OfficialKnowledgeResult):
        field_names = {f.name for f in fields(cls)}
        for term in banned:
            assert term not in field_names, f"{term} found on {cls.__name__}"


# --- 27-28. No merge with Approved Response Memory ------------------------


def test_no_approved_response_memory_result_used():
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        assert "ApprovedResponseMemoryResult" not in code


def test_no_fallback_to_approved_response_memory():
    """Code (not prose) must never import or call the style-memory
    retriever. `official_knowledge.py`'s own docstring legitimately
    NAMES "approved_memory.py" to explain where its design pattern
    came from -- that mention is prose, stripped here the same way as
    every other recurring docstring false positive in this codebase."""
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        assert "import approved_memory" not in code
        assert "from rcs.response_intelligence.retrieval.approved_memory" not in code
        assert "build_approved_response_memory_retriever" not in code


# --- Static isolation guards (29-34) ---------------------------------------


def _code_excluding_docstrings(path: str) -> str:
    """AST-based docstring stripper -- avoids the recurring false
    positive where a module's OWN docstring legitimately names a banned
    term to document its absence (hit repeatedly in M5A.1/M5A.3/M5B.2/
    M5B.3)."""
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
    return [inspect.getsourcefile(official_corpus_module), inspect.getsourcefile(official_knowledge_module)]


def test_no_external_dictionary_integration():
    for path in _core_module_files():
        code = _code_excluding_docstrings(path).lower()
        assert "external_dictionary_client" not in code


def test_no_external_dictionary_env_var_requirement():
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        assert "DICTIONARY_API_BASE_URL" not in code
        assert "DICTIONARY_API_KEY" not in code


def test_no_vector_database_dependency():
    banned = ("pgvector", "pinecone", "chromadb", "elasticsearch", "faiss", "atlas", "weaviate", "qdrant")
    for path in _core_module_files():
        code = _code_excluding_docstrings(path).lower()
        for term in banned:
            assert term not in code, f"forbidden vector-db reference {term!r} found in {path}"


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


def test_e5_embedding_adapter_is_reused_not_duplicated():
    """No local AutoModel/AutoTokenizer/mean-pool re-implementation --
    the module must import the shared adapter, never redefine it."""
    for path in _core_module_files():
        code = _code_excluding_docstrings(path)
        assert "AutoModel" not in code
        assert "AutoTokenizer" not in code
    knowledge_code = _code_excluding_docstrings(inspect.getsourcefile(official_knowledge_module))
    assert "from rcs.response_intelligence.retrieval.embeddings import" in knowledge_code


# --- 35. Exact E5 prefixes reused end-to-end -------------------------------


def test_semantic_path_uses_real_e5_query_prefix(monkeypatch):
    adapter = E5EmbeddingAdapter()
    captured: dict[str, list[str]] = {}

    def fake_embed(self, texts):
        captured["texts"] = list(texts)
        return np.zeros((len(texts), 4))

    monkeypatch.setattr(E5EmbeddingAdapter, "_embed", fake_embed)
    adapter.embed_query("استفسار عن كلمة")
    assert captured["texts"] == ["query: استفسار عن كلمة"]
    adapter.embed_passages(["محتوى أول", "محتوى ثانٍ"])
    assert captured["texts"] == ["passage: محتوى أول", "passage: محتوى ثانٍ"]


# --- 36-37. Index lifecycle -------------------------------------------------


def test_index_built_once_not_rebuilt_per_query():
    raw = [_raw()]
    retriever, embedder = _built_retriever(raw)
    for _ in range(5):
        retriever.retrieve(OfficialKnowledgeQuery(query_text="نص عام لا يطابق مفتاحًا", max_results=5))
    assert embedder.embed_passages_calls == 1
    assert embedder.load_calls == 1
    assert embedder.embed_query_calls == 5


def test_explicit_rebuild_produces_independent_retriever_and_index():
    embedder_v1 = FakeEmbeddingAdapter()
    retriever_v1 = build_official_knowledge_retriever(
        InMemorySourceProvider([_raw(source_id="only-in-v1", entry_id="only-in-v1")]), embedder_v1
    )
    embedder_v2 = FakeEmbeddingAdapter()
    retriever_v2 = build_official_knowledge_retriever(
        InMemorySourceProvider([_raw(source_id="only-in-v2", entry_id="only-in-v2")]), embedder_v2
    )
    result_v1 = retriever_v1.retrieve(OfficialKnowledgeQuery(query_text="only-in-v1", max_results=5))
    result_v2 = retriever_v2.retrieve(OfficialKnowledgeQuery(query_text="only-in-v2", max_results=5))
    assert [r.source_id for r in result_v1.results] == ["only-in-v1"]
    assert [r.source_id for r in result_v2.results] == ["only-in-v2"]
    assert retriever_v1.index is not retriever_v2.index


def test_build_index_never_loads_embedder_for_empty_records():
    embedder = FakeEmbeddingAdapter()
    index = build_index((), embedder)
    assert index.record_count == 0
    assert embedder.load_calls == 0


# --- 38. Structured lookup precedence --------------------------------------


def test_structured_lookup_takes_precedence_even_when_semantically_dissimilar():
    """The exact-key match wins even though its content vector is far
    from the query vector -- precedence is structural, not score-based."""
    far = np.array([0.0, 1.0, 0.0, 0.0])
    near = np.array([1.0, 0.0, 0.0, 0.0])
    raw = [
        _raw(source_id="exact-match", entry_id="E-target", content="محتوى FAR غير متعلق دلاليًا."),
        _raw(source_id="semantic-match", entry_id="E-other", content="محتوى NEAR قريب دلاليًا."),
    ]
    retriever, _ = _built_retriever(raw, vectors={"FAR": far, "NEAR": near, "E-target": near})
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="E-target", max_results=5))
    assert [r.source_id for r in result.results] == ["exact-match"]


# --- 39. Stale/conflicting source behavior ---------------------------------


def test_not_yet_effective_record_excluded():
    future = _SYNTHETIC_NOW + dt.timedelta(days=365)
    raw = [_raw(effective_from=future)]
    retriever, _ = _built_retriever(raw, clock=lambda: _SYNTHETIC_NOW)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="E-1", max_results=5))
    assert result.outcome == SUCCESS_EMPTY


def test_expired_record_excluded_by_temporal_validity():
    expired_to = _SYNTHETIC_NOW - dt.timedelta(days=1)
    raw = [_raw(effective_from=_SYNTHETIC_NOW - dt.timedelta(days=365), effective_to=expired_to)]
    retriever, _ = _built_retriever(raw, clock=lambda: _SYNTHETIC_NOW)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="E-1", max_results=5))
    assert result.outcome == SUCCESS_EMPTY


def test_currently_effective_conflicting_records_returned_deterministically_not_silently_resolved():
    """Two records share the same entry_id and are BOTH currently
    effective -- with no explicit precedence rule to break the tie,
    both are surfaced (deterministically ordered), never silently
    collapsed to one."""
    raw = [
        _raw(source_id="OK-A", entry_id="SHARED-KEY", content="محتوى النسخة أ."),
        _raw(source_id="OK-B", entry_id="SHARED-KEY", content="محتوى النسخة ب."),
    ]
    retriever, embedder = _built_retriever(raw, clock=lambda: _SYNTHETIC_NOW)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="SHARED-KEY", max_results=5))
    assert result.outcome == SUCCESS_WITH_RESULTS
    assert [r.source_id for r in result.results] == ["OK-A", "OK-B"]
    assert embedder.embed_query_calls == 0


# --- No production data fixture loaded by this test file -------------------


def test_no_production_data_fixture_loaded_by_this_test_file():
    code = Path(__file__).read_text(encoding="utf-8")
    body = code.split("def test_no_production_data_fixture_loaded_by_this_test_file")[0]
    for banned in ("read_excel(", "open(", "pd.read_csv(", "read_json("):
        assert banned not in body
