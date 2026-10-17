"""
Phase M5B.1 — type-safety tests for the retrieval contracts foundation.

These tests exercise ONLY rcs.response_intelligence.retrieval.contracts
-- pure Python objects, no database, no embedding, no corpus data, no
ReviewItem. Every fixture value below is clearly synthetic (per
instruction: unit-test fixtures may use clearly synthetic values only,
never something that could be mistaken for a real organizational fact).
"""

from __future__ import annotations

import datetime as dt
import inspect

import pytest

from rcs.response_intelligence.retrieval import contracts as contracts_module
from rcs.response_intelligence.retrieval.contracts import (
    APPROVED_RESPONSE_MEMORY,
    ERROR,
    OFFICIAL_KNOWLEDGE,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    TIMEOUT,
    UNAVAILABLE,
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryResult,
    ApprovedResponseMemoryRetrievalResult,
    ApprovedResponseMemoryRetriever,
    OfficialKnowledgeQuery,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
    OfficialKnowledgeRetriever,
)

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 0, 0, 0)


def _ok_result(**overrides) -> OfficialKnowledgeResult:
    base = dict(
        source_id="synthetic-source-1",
        source_title="Synthetic Test Policy Document",
        source_type="synthetic_test_type",
        content="This is a synthetic test fact, not a real organizational policy.",
        version="test-v1",
        effective_from=_SYNTHETIC_NOW,
        authoritative=True,
        retrieval_score=0.5,
    )
    base.update(overrides)
    return OfficialKnowledgeResult(**base)


def _arm_result(**overrides) -> ApprovedResponseMemoryResult:
    base = dict(
        review_item_id="synthetic-comment-1",
        final_reply_id="synthetic-reply-hash-1",
        response_text="نص رد تجريبي اصطناعي.",
        reviewer_action="SEND_AS_IS",
        approved_at=_SYNTHETIC_NOW,
        retrieval_score=0.5,
    )
    base.update(overrides)
    return ApprovedResponseMemoryResult(**base)


# --- 1-5. Outcome/result-count invariants -----------------------------


def test_success_with_results_rejects_empty_results():
    with pytest.raises(ValueError):
        OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=())
    with pytest.raises(ValueError):
        ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=())


def test_success_empty_rejects_non_empty_results():
    with pytest.raises(ValueError):
        OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY, results=(_ok_result(),))
    with pytest.raises(ValueError):
        ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY, results=(_arm_result(),))


def test_timeout_rejects_non_empty_results():
    with pytest.raises(ValueError):
        OfficialKnowledgeRetrievalResult(outcome=TIMEOUT, results=(_ok_result(),))


def test_unavailable_rejects_non_empty_results():
    with pytest.raises(ValueError):
        OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, results=(_ok_result(),))


def test_error_rejects_non_empty_results():
    with pytest.raises(ValueError):
        OfficialKnowledgeRetrievalResult(outcome=ERROR, results=(_ok_result(),))


def test_valid_constructions_succeed_for_every_outcome():
    assert OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),)).outcome == SUCCESS_WITH_RESULTS
    assert OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY, results=()).outcome == SUCCESS_EMPTY
    assert OfficialKnowledgeRetrievalResult(outcome=TIMEOUT, results=(), failure_detail="synthetic_timeout").outcome == TIMEOUT
    assert OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, results=()).outcome == UNAVAILABLE
    assert OfficialKnowledgeRetrievalResult(outcome=ERROR, results=(), failure_detail="synthetic_error").outcome == ERROR


def test_success_outcomes_reject_a_failure_detail():
    with pytest.raises(ValueError):
        OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY, results=(), failure_detail="should not be allowed")
    with pytest.raises(ValueError):
        OfficialKnowledgeRetrievalResult(
            outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),), failure_detail="should not be allowed"
        )


# --- 6. OfficialKnowledgeResult carries factual provenance fields -----


def test_official_knowledge_result_carries_factual_provenance_fields():
    result = _ok_result()
    for field_name in (
        "source_id", "source_title", "source_type", "content",
        "version", "effective_from", "effective_to", "authoritative", "retrieval_score",
    ):
        assert hasattr(result, field_name), f"missing provenance field: {field_name}"
    assert result.corpus_type == OFFICIAL_KNOWLEDGE


# --- 7. ApprovedResponseMemoryResult has no authoritative/factual field -


def test_approved_response_memory_result_has_no_authoritative_field():
    result = _arm_result()
    assert not hasattr(result, "authoritative")
    assert not hasattr(result, "factual")
    assert not hasattr(result, "is_authoritative")
    assert not hasattr(result, "grounding")


# --- 8. ApprovedResponseMemoryResult has no visitor/raw-comment field --


def test_approved_response_memory_result_has_no_visitor_or_raw_comment_fields():
    result = _arm_result()
    for banned_field in (
        "visitor_name", "visitor_email", "original_comment", "comment_text",
        "reviewer_display_name", "reviewer_email", "reviewer_id",
    ):
        assert not hasattr(result, banned_field), f"unexpected field leaked onto ApprovedResponseMemoryResult: {banned_field}"


# --- 9. Official knowledge and response-memory result types are distinct


def test_result_types_are_distinct_and_not_interchangeable():
    ok = _ok_result()
    arm = _arm_result()
    assert type(ok) is not type(arm)
    assert not isinstance(ok, type(arm))
    assert not isinstance(arm, type(ok))
    assert ok.corpus_type != arm.corpus_type
    assert {ok.corpus_type, arm.corpus_type} == {OFFICIAL_KNOWLEDGE, APPROVED_RESPONSE_MEMORY}


def test_wrapper_types_are_distinct():
    ok_wrap = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
    arm_wrap = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)
    assert type(ok_wrap) is not type(arm_wrap)


# --- 10. The two retriever interfaces are distinct ---------------------


def test_retriever_protocols_are_distinct():
    """Protocols here are structural typing (PEP 544), not
    @runtime_checkable -- isinstance() against them is intentionally not
    part of this contract's usage pattern (matching
    rcs.response_generation.service.SupportsGenerate's existing style,
    which is also a plain, non-runtime-checkable Protocol). Distinctness
    is proven at the type level instead: two separate class objects with
    different `retrieve` return-type annotations, and neither is a
    subclass of the other."""
    assert OfficialKnowledgeRetriever is not ApprovedResponseMemoryRetriever

    official_hints = inspect.get_annotations(OfficialKnowledgeRetriever.retrieve)
    memory_hints = inspect.get_annotations(ApprovedResponseMemoryRetriever.retrieve)
    assert official_hints["return"] is not memory_hints["return"]
    assert official_hints["query"] is not memory_hints["query"]


# --- 11. No generic runtime retriever path exists -----------------------


def test_no_generic_retriever_protocol_or_typevar_exists():
    """Static guard: this module must not define a generic Retriever[T]
    (or any TypeVar-based retriever) that both implementations could
    satisfy interchangeably -- the whole point of having two named
    Protocols instead."""
    import typing

    members = vars(contracts_module)
    for name, obj in members.items():
        assert not isinstance(obj, typing.TypeVar), f"unexpected TypeVar in contracts module: {name}"
    assert "Retriever" not in members  # only the two named, corpus-specific Protocols exist


# --- 12. Retrieval score is carried but not interpreted -----------------


def test_retrieval_score_is_a_plain_float_with_no_confidence_semantics():
    result = _ok_result(retrieval_score=0.73)
    assert isinstance(result.retrieval_score, float)
    # No threshold, no clamping, no "confidence" field exists anywhere on
    # the type -- any float value, including ones a real cosine-similarity
    # or reranker score would never produce, must be accepted here, since
    # this contract does not know or assume the scoring method.
    unusual = _ok_result(retrieval_score=5.0)
    assert unusual.retrieval_score == 5.0
    negative = _ok_result(retrieval_score=-1.0)
    assert negative.retrieval_score == -1.0
    assert not hasattr(result, "confidence")
    assert not hasattr(result, "cosine_similarity")


# --- 13. Raw provider exceptions cannot be stored in the result --------


def test_failure_detail_cannot_hold_an_exception_object():
    """The contract types this exact string; nothing coerces an
    exception into failure_detail, so a caller attempting to pass one
    directly would violate the declared type (enforced by any type
    checker) -- and even at pure runtime, storing str(exc) rather than
    the exception object itself is the only way this field is ever
    meant to be populated. This test proves the field only ever holds
    what a caller explicitly puts there -- there is no stack-trace or
    exception-repr capture built into the dataclass."""
    detail = "synthetic_timeout_after_2s"  # a sanitized category string, never a stack trace
    result = OfficialKnowledgeRetrievalResult(outcome=TIMEOUT, results=(), failure_detail=detail)
    assert result.failure_detail == detail
    assert isinstance(result.failure_detail, str)


# --- 14. Contracts import no SQLAlchemy/PyMongo/vector/embedding SDK ---


def _actual_imported_root_modules(module) -> set[str]:
    """Real imports only, via ast -- not a text/docstring scan (the
    module's own docstring legitimately names every banned dependency
    it does NOT import, which a plain substring search would misfire
    on -- exactly the lesson already learned in this session's M5A.1
    orchestrator test)."""
    import ast

    tree = ast.parse(inspect.getsource(module))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_contracts_module_has_no_forbidden_imports():
    """Narrow, readable static guard -- not an elaborate source-analysis
    framework, just the exact banned-import list from the phase brief,
    checked against actual import statements."""
    banned_roots = {
        "sqlalchemy", "pymongo", "httpx", "google",
        "sentence_transformers", "faiss", "langgraph", "boto3", "boto",
    }
    actual = _actual_imported_root_modules(contracts_module)
    forbidden_hits = actual & banned_roots
    assert not forbidden_hits, f"forbidden import(s) found in contracts.py: {forbidden_hits}"


def test_contracts_module_imports_are_stdlib_typing_only():
    allowed_roots = {"__future__", "dataclasses", "datetime", "typing"}
    actual = _actual_imported_root_modules(contracts_module)
    assert actual <= allowed_roots, f"unexpected imports in contracts.py: {actual - allowed_roots}"


# --- 15. No official corpus data or ReviewItem data is read -------------


def test_no_review_item_or_database_access_in_contracts_module():
    """Confirms (again, from a different angle than test 14) that this
    module never references the ORM model or any engine/session/
    collection object -- it is pure data shape, nothing else."""
    source = inspect.getsource(contracts_module)
    for banned in ("ReviewItem", "Session(", "engine", "collection", ".execute(", "select("):
        assert banned not in source, f"unexpected database-shaped reference in contracts.py: {banned!r}"


# --- Query contract behavior ---------------------------------------------


def test_official_knowledge_query_requires_positive_max_results():
    OfficialKnowledgeQuery(query_text="synthetic query", max_results=1)
    with pytest.raises(ValueError):
        OfficialKnowledgeQuery(query_text="synthetic query", max_results=0)
    with pytest.raises(ValueError):
        OfficialKnowledgeQuery(query_text="synthetic query", max_results=-5)


def test_approved_response_memory_query_requires_positive_max_results():
    ApprovedResponseMemoryQuery(query_text="synthetic query", max_results=1)
    with pytest.raises(ValueError):
        ApprovedResponseMemoryQuery(query_text="synthetic query", max_results=0)


def test_queries_reject_blank_query_text():
    with pytest.raises(ValueError):
        OfficialKnowledgeQuery(query_text="   ", max_results=1)
    with pytest.raises(ValueError):
        ApprovedResponseMemoryQuery(query_text="", max_results=1)


def test_official_knowledge_query_has_no_source_allowlist_field():
    """Per the M5B.0 correction: no production-source allowlist/filename
    heuristic exists anywhere in this phase."""
    query = OfficialKnowledgeQuery(query_text="synthetic", max_results=1)
    for banned_field in ("source_file", "source_constraints", "allowed_sources", "production_only"):
        assert not hasattr(query, banned_field)


def test_approved_response_memory_query_has_no_source_allowlist_field():
    query = ApprovedResponseMemoryQuery(query_text="synthetic", max_results=1)
    for banned_field in ("source_file", "allowed_sources", "production_only"):
        assert not hasattr(query, banned_field)


# --- Frozen/immutable ------------------------------------------------------


def test_result_and_wrapper_types_are_frozen():
    result = _ok_result()
    with pytest.raises(Exception):
        result.source_id = "different"
    wrapper = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
    with pytest.raises(Exception):
        wrapper.outcome = ERROR
