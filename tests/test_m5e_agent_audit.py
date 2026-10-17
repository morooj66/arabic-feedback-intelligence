"""
Phase M5E — agent execution audit trace: orchestrator-level tests using
a deterministic in-memory fake `AgentAuditRepository`. No real database
here — see tests/test_agent_audit_repository.py for the concrete
PostgreSQL/Mongo repository tests.
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
import json
from dataclasses import fields
from pathlib import Path

import pytest

from rcs.response_generation.service import GenerationRequest
from rcs.response_intelligence import audit as audit_module
from rcs.response_intelligence.audit import AgentAuditRecord, build_audit_record
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    ERROR,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    UNAVAILABLE,
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryResult,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeQuery,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
)

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)
_EMPTY_OK = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)
_VALID_DRAFT = "شكرًا لملاحظتكم حول هذا الموضوع."
_TOO_SHORT_DRAFT = "شكرًا"
_PII_DRAFT = "شكرًا لتواصلكم على الرقم 0501234567 مع المعجم."


def _req(**overrides) -> GenerationRequest:
    base = dict(
        comment_id="c-1",
        comment_text="هذا تعليق تجريبي بخصوص كلمة مقترحة.",
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        sentiment="POSITIVE",
        confidence=0.9,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def _ok_result(**overrides) -> OfficialKnowledgeResult:
    base = dict(
        source_id="OK-1", source_title="عنوان رسمي", source_type="dictionary_entry",
        content="محتوى رسمي معتمد.", version="v1", effective_from=_SYNTHETIC_NOW,
        authoritative=True, retrieval_score=0.9,
    )
    base.update(overrides)
    return OfficialKnowledgeResult(**base)


def _arm_result(**overrides) -> ApprovedResponseMemoryResult:
    base = dict(
        review_item_id="r-1", final_reply_id="hash-1", response_text="مثال أسلوب معتمد.",
        reviewer_action="SEND_AS_IS", approved_at=_SYNTHETIC_NOW, retrieval_score=0.9,
    )
    base.update(overrides)
    return ApprovedResponseMemoryResult(**base)


class FakeOfficialKnowledgeRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query: OfficialKnowledgeQuery):
        return self.result


class FakeApprovedResponseMemoryRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query: ApprovedResponseMemoryQuery):
        return self.result


class FakeGeminiClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        item = self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _draft_json(text: str) -> str:
    return json.dumps({"suggested_response": text}, ensure_ascii=False)


class FakeAuditRepository:
    """Deterministic in-memory fake satisfying AgentAuditRepository.
    Mirrors the real repositories' idempotent-on-execution_id insert
    semantics and append-only listing."""

    def __init__(self):
        self._by_id: dict[str, AgentAuditRecord] = {}
        self.record_call_count = 0

    def record(self, record: AgentAuditRecord) -> None:
        self.record_call_count += 1
        if record.execution_id in self._by_id:
            return  # idempotent: already recorded
        self._by_id[record.execution_id] = record

    def list_for_review_item(self, review_item_id: str) -> tuple[AgentAuditRecord, ...]:
        return tuple(
            sorted(
                (r for r in self._by_id.values() if r.review_item_id == review_item_id),
                key=lambda r: r.created_at,
            )
        )

    def get_by_execution_id(self, execution_id: str) -> AgentAuditRecord | None:
        return self._by_id.get(execution_id)


class FailingAuditRepository:
    def __init__(self, exc: Exception):
        self._exc = exc

    def record(self, record: AgentAuditRecord) -> None:
        raise self._exc


def _run(request, audit_repository, ok_result=_EMPTY_OK, arm_result=_EMPTY_ARM, responses=(_draft_json(_VALID_DRAFT),)):
    client = FakeGeminiClient(list(responses))
    exec_result = run_agent_pipeline(
        request, client, 0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(ok_result),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(arm_result),
        audit_repository=audit_repository,
    )
    return exec_result, client


# --- 1-5: basic recording ----------------------------------------------------


def test_1_successful_single_draft_execution_produces_one_audit_record():
    repo = FakeAuditRepository()
    _run(_req(), repo)
    assert repo.record_call_count == 1
    assert len(repo.list_for_review_item("c-1")) == 1


def test_2_refined_execution_produces_one_audit_record():
    repo = FakeAuditRepository()
    _run(_req(), repo, responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)])
    assert repo.record_call_count == 1
    assert len(repo.list_for_review_item("c-1")) == 1


def test_3_logical_generation_count_one_persisted_correctly():
    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(), repo)
    trace = repo.list_for_review_item("c-1")[0]
    assert trace.logical_generation_count == 1 == exec_result.metadata.logical_generation_count


def test_4_logical_generation_count_two_persisted_correctly():
    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(), repo, responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)])
    trace = repo.list_for_review_item("c-1")[0]
    assert trace.logical_generation_count == 2 == exec_result.metadata.logical_generation_count


def test_5_refinement_attempted_persisted_correctly():
    repo = FakeAuditRepository()
    _run(_req(), repo, responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)])
    assert repo.list_for_review_item("c-1")[0].refinement_attempted is True

    repo2 = FakeAuditRepository()
    _run(_req(comment_id="c-2"), repo2)
    assert repo2.list_for_review_item("c-2")[0].refinement_attempted is False


# --- 6-10: retrieval/grounding fields ----------------------------------------


def test_6_grounding_mode_persisted_correctly():
    repo = FakeAuditRepository()
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    _run(_req(primary_intent="INQUIRY"), repo, ok_result=official)
    assert repo.list_for_review_item("c-1")[0].grounding_mode == "FACTUAL"


def test_7_official_retrieval_outcome_persisted():
    repo = FakeAuditRepository()
    official = OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, failure_detail="synthetic")
    _run(_req(), repo, ok_result=official)
    assert repo.list_for_review_item("c-1")[0].official_retrieval_outcome == UNAVAILABLE


def test_8_official_result_count_persisted():
    repo = FakeAuditRepository()
    official = OfficialKnowledgeRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(source_id="a"), _ok_result(source_id="b"))
    )
    _run(_req(), repo, ok_result=official)
    assert repo.list_for_review_item("c-1")[0].official_result_count == 2


def test_9_style_retrieval_outcome_persisted():
    repo = FakeAuditRepository()
    approved = ApprovedResponseMemoryRetrievalResult(outcome=ERROR, failure_detail="synthetic")
    _run(_req(), repo, arm_result=approved)
    assert repo.list_for_review_item("c-1")[0].style_retrieval_outcome == ERROR


def test_10_style_result_count_persisted():
    repo = FakeAuditRepository()
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(),))
    _run(_req(), repo, arm_result=approved)
    assert repo.list_for_review_item("c-1")[0].style_result_count == 1


# --- 11-15: validation outcome fields ----------------------------------------


def test_11_initial_validation_outcome_persisted():
    repo = FakeAuditRepository()
    _run(_req(), repo, responses=[_draft_json(_PII_DRAFT)])
    assert repo.list_for_review_item("c-1")[0].initial_validation_passed is False


def test_12_initial_failure_code_persisted_when_present():
    repo = FakeAuditRepository()
    _run(_req(), repo, responses=[_draft_json(_PII_DRAFT)])
    assert repo.list_for_review_item("c-1")[0].initial_failure_code == "pii_in_draft"


def test_13_final_validation_outcome_persisted():
    repo = FakeAuditRepository()
    _run(_req(), repo, responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)])
    assert repo.list_for_review_item("c-1")[0].final_validation_passed is True


def test_14_final_failure_code_persisted_when_present():
    repo = FakeAuditRepository()
    _run(_req(), repo, responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_TOO_SHORT_DRAFT)])
    assert repo.list_for_review_item("c-1")[0].final_failure_code == "too_short"


def test_15_final_generation_status_persisted():
    repo = FakeAuditRepository()
    _run(_req(), repo, responses=[_draft_json(_PII_DRAFT)])
    assert repo.list_for_review_item("c-1")[0].generation_status == "FAILED_REVIEW_ONLY"


# --- 16-20: identity / multiple executions / append-only ---------------------


def test_16_trace_has_unique_execution_identity():
    repo = FakeAuditRepository()
    _run(_req(), repo)
    trace = repo.list_for_review_item("c-1")[0]
    assert trace.execution_id
    assert repo.get_by_execution_id(trace.execution_id) is trace


def test_17_two_genuine_executions_for_same_review_item_create_two_traces():
    repo = FakeAuditRepository()
    _run(_req(), repo)
    _run(_req(), repo)  # a second, independent generation attempt for the same item
    traces = repo.list_for_review_item("c-1")
    assert len(traces) == 2
    assert traces[0].execution_id != traces[1].execution_id


def test_18_persistence_retry_does_not_duplicate_same_execution_trace():
    repo = FakeAuditRepository()
    record = build_audit_record(
        execution_id="exec-fixed-1", review_item_id="c-1", created_at=_SYNTHETIC_NOW,
        grounding_mode="IDENTIFYING_ONLY", official_retrieval_outcome=SUCCESS_EMPTY,
        official_result_count=0, official_grounding_available=False,
        style_retrieval_outcome=SUCCESS_EMPTY, style_result_count=0,
        logical_generation_count=1, refinement_attempted=False,
        initial_validation_passed=True, initial_failure_code=None,
        final_validation_passed=True, final_failure_code=None,
        generation_status="SUCCEEDED",
    )
    repo.record(record)
    repo.record(record)  # simulated retry of the same write
    assert len(repo.list_for_review_item("c-1")) == 1


def test_19_trace_linked_to_correct_review_item():
    repo = FakeAuditRepository()
    _run(_req(comment_id="item-A"), repo)
    _run(_req(comment_id="item-B"), repo)
    assert [t.review_item_id for t in repo.list_for_review_item("item-A")] == ["item-A"]
    assert [t.review_item_id for t in repo.list_for_review_item("item-B")] == ["item-B"]


def test_20_append_only_behavior_no_repository_update_method_exists():
    from rcs.response_intelligence.audit import AgentAuditRepository

    method_names = {name for name in dir(AgentAuditRepository) if not name.startswith("_")}
    assert "update" not in method_names
    assert "delete" not in method_names
    assert "overwrite" not in method_names


# --- 21-29: forbidden content --------------------------------------------------


def test_21_to_29_no_forbidden_content_in_persisted_record():
    repo = FakeAuditRepository()
    identifying = "submission title: عنوان سري تجريبي؛ internal entry_id: ENTRY-SECRET"
    official = OfficialKnowledgeRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="محتوى معرفة رسمية سرية للغاية"),)
    )
    approved = ApprovedResponseMemoryRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="نص أسلوب سري للغاية"),)
    )
    raw_comment = "تعليق زائر حقيقي جدًا مع رقم 0501234567 وبريد leak@example.com"
    exec_result, client = _run(
        _req(comment_text=raw_comment, trusted_grounding_text=identifying, primary_intent="INQUIRY"),
        repo, ok_result=official, arm_result=approved,
    )
    trace = repo.list_for_review_item("c-1")[0]
    trace_repr = repr(trace)

    # 21: no raw visitor comment
    assert "تعليق زائر حقيقي جدًا" not in trace_repr
    # 22: no identifying context text
    assert "عنوان سري تجريبي" not in trace_repr
    assert "ENTRY-SECRET" not in trace_repr
    # 23: no Official Knowledge full content
    assert "محتوى معرفة رسمية سرية للغاية" not in trace_repr
    # 24: no style-reference full text
    assert "نص أسلوب سري للغاية" not in trace_repr
    # 25: no prompt
    assert client.prompts[0] not in trace_repr
    # PII specifically (phone/email) never leaks through either channel
    assert "0501234567" not in trace_repr
    assert "leak@example.com" not in trace_repr


def test_26_no_chain_of_thought_or_reasoning_field_exists():
    field_names = {f.name for f in fields(AgentAuditRecord)}
    for banned in ("reasoning", "chain", "thought", "rationale"):
        assert not any(banned in name.lower() for name in field_names)


def test_27_no_raw_gemini_response_persisted():
    field_names = {f.name for f in fields(AgentAuditRecord)}
    assert not any("raw_response" in name.lower() or "raw_request" in name.lower() for name in field_names)


def test_28_no_raw_exception_string_persisted():
    repo = FakeAuditRepository()
    import httpx

    _run(_req(), repo, responses=[httpx.ConnectError("synthetic failure with /secret/path leak")])
    trace = repo.list_for_review_item("c-1")[0]
    assert "/secret/path" not in repr(trace)
    assert trace.final_failure_code == "gemini_api_error"


def test_29_no_secrets_or_token_fields_exist():
    field_names = {f.name for f in fields(AgentAuditRecord)}
    for banned in ("secret", "token", "api_key", "password", "credential"):
        assert not any(banned in name.lower() for name in field_names)


# --- 30-34: success/failure/degraded behavior ---------------------------------


def test_30_gemini_success_and_trace_persistence_success():
    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(), repo)
    assert exec_result.generation_result.generation_status == "SUCCEEDED"
    assert len(repo.list_for_review_item("c-1")) == 1


def test_31_refinement_success_and_trace_persistence_success():
    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(), repo, responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)])
    assert exec_result.generation_result.generation_status == "SUCCEEDED"
    assert repo.list_for_review_item("c-1")[0].refinement_attempted is True


def test_32_validation_failure_still_produces_safe_audit_metadata():
    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(), repo, responses=[_draft_json(_PII_DRAFT)])
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    trace = repo.list_for_review_item("c-1")[0]
    assert trace.generation_status == "FAILED_REVIEW_ONLY"
    assert trace.final_failure_code == "pii_in_draft"


def test_33_gemini_provider_failure_trace_behavior():
    import httpx

    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(), repo, responses=[httpx.ConnectError("synthetic")])
    trace = repo.list_for_review_item("c-1")[0]
    assert trace.generation_status == "FAILED_REVIEW_ONLY"
    assert trace.final_failure_code == "gemini_api_error"
    assert trace.logical_generation_count == 1  # provider failure is not refinement-eligible


def test_34_retrieval_unavailable_trace_behavior():
    repo = FakeAuditRepository()
    official = OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, failure_detail="synthetic")
    exec_result, _ = _run(_req(), repo, ok_result=official)
    trace = repo.list_for_review_item("c-1")[0]
    assert trace.official_retrieval_outcome == UNAVAILABLE
    assert trace.official_grounding_available is False
    assert exec_result.generation_result.generation_status == "SUCCEEDED"  # degrades safely


# --- 35-39: audit-repository failure / atomicity / reviewer safety -----------


def test_35_audit_repository_failure_does_not_break_generation():
    failing_repo = FailingAuditRepository(RuntimeError("synthetic DB outage with /secret/conn leak"))
    exec_result = run_agent_pipeline(
        _req(), FakeGeminiClient([_draft_json(_VALID_DRAFT)]), 0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
        audit_repository=failing_repo,
    )
    assert exec_result.generation_result.generation_status == "SUCCEEDED"


def test_36_business_result_matches_documented_atomicity_policy(caplog):
    """Business result remains primary; audit failure is surfaced only
    via a sanitized log line (no raw exception text)."""
    import logging

    failing_repo = FailingAuditRepository(RuntimeError("synthetic DB outage with /secret/conn leak"))
    with caplog.at_level(logging.WARNING, logger="rcs.response_intelligence.orchestrator"):
        exec_result = run_agent_pipeline(
            _req(), FakeGeminiClient([_draft_json(_VALID_DRAFT)]), 0,
            official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
            approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
            audit_repository=failing_repo,
        )
    assert exec_result.generation_result.suggested_response == _VALID_DRAFT
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "RuntimeError" in message
    assert "/secret/conn" not in message
    assert "synthetic DB outage" not in message


def test_37_no_reviewer_approved_row_mutation(tmp_path):
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    engine = get_engine(tmp_path / "m5e_no_mutation.sqlite")
    init_db(engine)
    upsert_review_items(engine, [dict(
        comment_id="c-1", source_file="f", source_row_index=0,
        review_status="APPROVED", is_prefiltered_junk=False, ready_to_send=True, row_version=1,
        reviewer_action="SEND_AS_IS", final_reply="رد المراجع المعتمد.",
    )])
    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(), repo)
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        exec_result.generation_result, intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE"
    )
    assert detail.review_status == "APPROVED"
    assert detail.final_response == "رد المراجع المعتمد."
    # Trace was still recorded even though the business row was protected/unchanged.
    assert len(repo.list_for_review_item("c-1")) == 1


def test_38_and_39_reviewer_isolation_and_assignment_unchanged():
    """Static proof: the new audit modules import nothing from reviewer-
    isolation or assignment code -- no coupling introduced."""
    for module in (audit_module,):
        code = inspect.getsource(module)
        assert "reviewer_isolation" not in code
        assert "assign_batch" not in code
        assert "AssignmentResult" not in code


# --- 40-45: regression guards --------------------------------------------------


def test_40_generation_result_schema_unchanged():
    from rcs.response_generation.schema import GenerationResult

    field_names = {f.name for f in fields(GenerationResult)}
    assert field_names == {
        "comment_id", "routing_decision", "generation_status", "suggested_response",
        "review_required", "reason_code", "provider", "model_name", "prompt_version",
        "generated_at", "future_reviewer_actions", "eligibility_rule_id", "error",
    }


def test_41_agent_execution_metadata_remains_content_free():
    from rcs.response_intelligence.orchestrator import AgentExecutionMetadata

    content_shaped = ("prompt", "reasoning", "chain", "thought", "comment_text", "raw_")
    for f in fields(AgentExecutionMetadata):
        lowered = f.name.lower()
        for banned in content_shaped:
            assert banned not in lowered


def test_42_no_prompt_builder_regression():
    from rcs.response_generation.prompt import build_generation_prompt

    prompt = build_generation_prompt(
        comment_text="تعليق", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL",
        sentiment="POSITIVE", reason_code="thanks_praise_eligible",
    )
    assert "OFFICIAL KNOWLEDGE" in prompt
    assert "IDENTIFYING CONTEXT" not in prompt


def test_43_no_validation_gate_regression():
    from rcs.response_generation.validate import validate_draft

    ok, why = validate_draft(_PII_DRAFT)
    assert not ok and why == "pii_in_draft"


def test_44_no_eligibility_routing_regression():
    repo = FakeAuditRepository()
    exec_result, _ = _run(_req(primary_intent="COMPLAINT", sentiment="NEGATIVE"), repo)
    assert exec_result.generation_result.eligibility_rule_id == "generate_complaint"


def test_45_max_logical_gemini_calls_still_two():
    repo = FakeAuditRepository()
    _, client = _run(_req(), repo, responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_TOO_SHORT_DRAFT)])
    assert client.call_count == 2


# --- 52-56: out-of-scope guards ------------------------------------------------


def _code_excluding_docstrings(path: str) -> str:
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


def test_52_to_56_no_out_of_scope_integrations():
    import rcs.response_intelligence.orchestrator as orchestrator_module

    banned = (
        "langgraph", "react_agent", "toolcall", "external_dictionary_client",
        "pinecone", "chromadb", "weaviate", "qdrant", "pgvector", "elasticsearch", "faiss",
        "conversation_memory", "session_memory", "checkpoint",
    )
    for path in (inspect.getsourcefile(orchestrator_module), inspect.getsourcefile(audit_module)):
        code = _code_excluding_docstrings(path).lower()
        for term in banned:
            assert term not in code, f"forbidden reference {term!r} found in {path}"


def test_no_forbidden_field_names_in_audit_record():
    """Phase M5E.1: audit.py no longer enforces this at import time
    (a domain module importing successfully must never depend on a
    passing security assertion) -- the exact same policy, using the
    exact same shared constant/function (not a locally duplicated
    list), is enforced here instead, as a normal test that fails CI
    the same way any other regression would."""
    field_names = [f.name for f in fields(AgentAuditRecord)]
    audit_module.assert_no_forbidden_field_names(field_names)  # raises AssertionError on violation


def test_no_forbidden_field_names_in_orm_trace_columns():
    """Same policy, applied to the actual persisted ORM columns
    (rcs.review.models.AgentExecutionTrace) -- not just the pure
    dataclass shape, since that is what a real database row actually
    contains."""
    from sqlalchemy import inspect as sa_inspect

    from rcs.review.models import AgentExecutionTrace

    column_names = [c.name for c in sa_inspect(AgentExecutionTrace).columns]
    audit_module.assert_no_forbidden_field_names(column_names)


def test_forbidden_field_substrings_constant_is_exported_and_nonempty():
    assert len(audit_module.FORBIDDEN_FIELD_SUBSTRINGS) > 0
    assert "prompt" in audit_module.FORBIDDEN_FIELD_SUBSTRINGS
    assert "secret" in audit_module.FORBIDDEN_FIELD_SUBSTRINGS


def test_audit_module_import_has_no_assertion_side_effect():
    """Phase M5E.1: a static (AST-level) proof that audit.py contains no
    top-level, module-scope function-call statement at all -- i.e. no
    lingering `_assert_no_forbidden_fields()`-style side effect. This is
    deliberately NOT `importlib.reload()`-based: reloading a module that
    defines a class (`AgentAuditRecord`) creates a SECOND, distinct
    class object in the same process, which would silently break
    dataclass `==` comparisons (identity-sensitive) for every OTHER test
    in this session that already imported the original class -- exactly
    the kind of cross-test pollution a module-import test must never
    risk causing."""
    tree = ast.parse(inspect.getsource(audit_module))
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert top_level_calls == []
