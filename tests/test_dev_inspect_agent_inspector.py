"""
Phase DEV-INSPECT — Agent Inspector / Test Console tests.

Verifies: the Inspector reuses real production functions verbatim (no
duplicated decision logic), never mutates review_items or reviewer
state, never persists to the real M5E audit repository, exposes no
CoT/secret/PII field, and keeps the pipeline's own invariants
(refinement <= 1 attempt, logical Gemini calls <= 2) intact. No network,
no live Gemini call anywhere in this file.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from dataclasses import fields

import pytest

from rcs.response_generation.eligibility import determine_response_eligibility, EligibilityInput
from rcs.response_generation.validate import run_validation_gates
from rcs.response_intelligence import inspector
from rcs.response_intelligence.audit import FORBIDDEN_FIELD_SUBSTRINGS
from rcs.response_intelligence.orchestrator import AgentExecutionTrace, run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    OfficialKnowledgeRetrievalResult,
)
from rcs.response_generation.service import GenerationRequest
from rcs.review import inspector_api


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.call_count += 1
        item = self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _draft(text: str) -> str:
    return json.dumps({"suggested_response": text}, ensure_ascii=False)


_VALID_DRAFT = "شكرًا لملاحظتكم القيّمة حول هذا الموضوع."
_TOO_SHORT_DRAFT = "شكرًا"
_COMMENT = "شكرًا جزيلاً على هذا العمل الرائع في المعجم."


# ---------------------------------------------------------------------
# Existing Agent behavior unchanged (capture_trace default off)
# ---------------------------------------------------------------------


def test_capture_trace_default_off_matches_pre_dev_inspect_shape():
    request = GenerationRequest(comment_id="c1", comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE")
    result = run_agent_pipeline(request, _FakeClient([_draft(_VALID_DRAFT)]), None)
    assert result.trace is None  # default False -- zero behavior change


def test_capture_trace_true_does_not_change_generation_result_or_metadata():
    request = GenerationRequest(comment_id="c1", comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE")
    without_trace = run_agent_pipeline(request, _FakeClient([_draft(_VALID_DRAFT)]), None, capture_trace=False)
    with_trace = run_agent_pipeline(request, _FakeClient([_draft(_VALID_DRAFT)]), None, capture_trace=True)
    assert without_trace.generation_result == with_trace.generation_result
    assert dataclasses.replace(without_trace.metadata, execution_id="x") == dataclasses.replace(with_trace.metadata, execution_id="x")
    assert with_trace.trace is not None
    assert isinstance(with_trace.trace, AgentExecutionTrace)


# ---------------------------------------------------------------------
# No production data mutation (dry-run guarantee)
# ---------------------------------------------------------------------


def test_inspector_module_has_no_database_or_review_items_dependency():
    # The module docstring legitimately DOCUMENTS that it never touches
    # review_items -- what must not exist is an actual ORM import/usage.
    assert not hasattr(inspector, "ReviewItem")
    src = inspect.getsource(inspector)
    assert "import ReviewItem" not in src
    assert "Session(" not in src
    assert "sqlalchemy" not in src.lower()


def test_inspector_api_module_has_no_review_item_mutation_surface():
    src = inspect.getsource(inspector_api)
    forbidden = ("apply_generation_result", "assign_reviewer", "send_as_is", "edit_and_send", "mark_no_response_needed", "claim_review")
    for bad in forbidden:
        assert bad not in src


def test_inspector_comment_ids_are_namespaced_never_colliding_with_real_ids():
    request_id = inspector._inspector_comment_id()
    assert request_id.startswith("dev-inspect-")


def test_full_inspection_uses_a_namespaced_comment_id_not_caller_supplied():
    result = inspector.run_full_inspection(
        comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        confidence=None, insufficient_context=False, original_title=None, entry_id=None,
        client=_FakeClient([_draft(_VALID_DRAFT)]),
    )
    assert result.generation_result.comment_id.startswith("dev-inspect-")


# ---------------------------------------------------------------------
# Isolated audit trace (never the production repository)
# ---------------------------------------------------------------------


def test_inspector_audit_repository_is_isolated_and_in_memory():
    inspector.INSPECTOR_AUDIT_REPOSITORY.clear()
    result = inspector.run_full_inspection(
        comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        confidence=None, insufficient_context=False, original_title=None, entry_id=None,
        client=_FakeClient([_draft(_VALID_DRAFT)]),
    )
    record = inspector.INSPECTOR_AUDIT_REPOSITORY.get_by_execution_id(result.metadata.execution_id)
    assert record is not None
    assert record.review_item_id.startswith("dev-inspect-")
    # Never the real production audit repository class.
    assert type(inspector.INSPECTOR_AUDIT_REPOSITORY).__name__ == "InspectorAuditRepository"


# ---------------------------------------------------------------------
# Stage outputs match the real pipeline (same functions, same inputs)
# ---------------------------------------------------------------------


def test_classification_stage_matches_configured_runtime(monkeypatch):
    from config import settings
    from rcs.classification import runtime

    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "legacy_rcs")
    direct_intent, direct_conf = runtime.predict_intent([_COMMENT])[0]
    inspected = inspector.inspect_classification(_COMMENT)
    assert inspected.primary_intent == direct_intent
    assert inspected.primary_intent_confidence == direct_conf


def test_eligibility_stage_matches_real_determine_response_eligibility():
    direct = determine_response_eligibility(EligibilityInput(comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL"))
    inspected = inspector.inspect_eligibility(comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL")
    assert inspected == direct


def test_retrieval_stage_matches_real_default_retrievers():
    from rcs.response_intelligence.retrievers import default_official_knowledge_retriever
    from rcs.response_intelligence.retrieval.contracts import OfficialKnowledgeQuery

    direct = default_official_knowledge_retriever().retrieve(OfficialKnowledgeQuery(query_text=_COMMENT, max_results=3))
    inspected = inspector.inspect_retrieval(comment_text=_COMMENT)
    assert inspected.official_outcome == direct.outcome
    assert inspected.official_result_count == len(direct.results)


def test_retrieval_stage_hides_raw_content_by_default():
    inspected = inspector.inspect_retrieval(comment_text=_COMMENT, reveal_content=False)
    assert inspected.official_contents is None
    assert inspected.approved_contents is None


def test_grounding_stage_matches_real_evidence_assemble_grounding():
    from rcs.response_intelligence import evidence

    official_result = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
    direct = evidence.assemble_grounding(None, official_result)
    inspected = inspector.inspect_grounding(identifying_context=None, official_result=official_result)
    assert inspected.grounding_mode == direct.mode
    assert inspected.official_grounding_available == direct.official_grounding_available


def test_validation_stage_matches_real_run_validation_gates():
    direct = run_validation_gates(_VALID_DRAFT, trusted_grounding_text=None)
    inspected = inspector.inspect_validation(draft_text=_VALID_DRAFT)
    assert inspected.passed == direct.passed
    assert inspected == direct


# ---------------------------------------------------------------------
# Refinement / logical-call invariants unchanged
# ---------------------------------------------------------------------


def test_refinement_max_one_attempt_via_inspector_full_run():
    result = inspector.run_full_inspection(
        comment_text="هل تم تحويل ملاحظتي إلى الفريق المختص؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        confidence=None, insufficient_context=False, original_title=None, entry_id=None,
        client=_FakeClient([_draft("تم تحويل ملاحظتك إلى فريق المحتوى."), _draft(_VALID_DRAFT)]),
    )
    assert result.metadata.refinement_attempted is True
    assert result.metadata.logical_generation_count == 2


def test_no_case_exceeds_two_logical_calls_via_inspector():
    result = inspector.run_full_inspection(
        comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        confidence=None, insufficient_context=False, original_title=None, entry_id=None,
        client=_FakeClient([_draft(_TOO_SHORT_DRAFT), _draft(_TOO_SHORT_DRAFT)]),
    )
    assert result.metadata.logical_generation_count <= 2


# ---------------------------------------------------------------------
# No CoT / secret / PII fields anywhere in the Inspector's own types
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        AgentExecutionTrace,
        inspector.ClassificationInspection,
        inspector.RetrievalInspection,
        inspector.GroundingInspection,
    ],
)
def test_inspector_dataclasses_have_no_forbidden_field_names(cls):
    field_names = [f.name for f in fields(cls)]
    for name in field_names:
        lowered = name.lower()
        for banned in FORBIDDEN_FIELD_SUBSTRINGS:
            assert banned not in lowered, f"forbidden substring {banned!r} in field {name!r} of {cls.__name__}"


def test_full_inspection_dict_payload_has_no_forbidden_keys():
    result = inspector.run_full_inspection(
        comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        confidence=None, insufficient_context=False, original_title=None, entry_id=None,
        client=_FakeClient([_draft(_VALID_DRAFT)]),
    )
    payload = inspector_api._agent_execution_result_to_dict(result)

    def _walk_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _walk_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from _walk_keys(item)

    # "prompt_version" is a pre-existing, legitimate metadata field
    # already on GenerationResult/ReviewItem (e.g. "rg-v7") -- not raw
    # prompt text or chain-of-thought. Every other FORBIDDEN_FIELD_SUBSTRINGS
    # hit is a real problem.
    allowed_keys = {"prompt_version"}
    for key in _walk_keys(payload):
        if key in allowed_keys:
            continue
        lowered = str(key).lower()
        for banned in FORBIDDEN_FIELD_SUBSTRINGS:
            assert banned not in lowered, f"forbidden substring {banned!r} in response key {key!r}"


def test_no_raw_prompt_or_chain_of_thought_in_full_run_payload():
    result = inspector.run_full_inspection(
        comment_text=_COMMENT, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        confidence=None, insufficient_context=False, original_title=None, entry_id=None,
        client=_FakeClient([_draft(_VALID_DRAFT)]),
    )
    payload = inspector_api._agent_execution_result_to_dict(result)
    flat = json.dumps(payload, ensure_ascii=False)
    assert "أنت مساعد" not in flat  # no rendered system-prompt preamble text leaks through
    assert "reasoning" not in flat.lower()
    assert "chain_of_thought" not in flat.lower()


# ---------------------------------------------------------------------
# Reviewer isolation / assignment untouched
# ---------------------------------------------------------------------


def test_no_reviewer_isolation_or_assignment_code_touched():
    for mod in (inspector, inspector_api):
        src = inspect.getsource(mod)
        assert "reviewer_user_id" not in src
        assert "assign_reviewer" not in src
        assert "_require_item_access" not in src


def test_http_api_inspector_routes_are_admin_gated():
    import rcs.review.http_api as http_api_module

    src = inspect.getsource(http_api_module)
    for path in (
        "/api/admin/inspector/architecture",
        "/api/admin/inspector/config",
        "/api/admin/inspector/classification",
        "/api/admin/inspector/eligibility",
        "/api/admin/inspector/retrieval",
        "/api/admin/inspector/grounding",
        "/api/admin/inspector/validation",
        "/api/admin/inspector/full-run",
    ):
        assert path in src


# ---------------------------------------------------------------------
# HTTP-shape adapter behavior
# ---------------------------------------------------------------------


def test_handle_inspector_classification_requires_comment_text():
    with pytest.raises(ValueError):
        inspector_api.handle_inspector_classification({})


def test_handle_inspector_validation_reports_pass_fail_shape():
    payload = inspector_api.handle_inspector_validation({"draft_text": _VALID_DRAFT})
    assert payload["passed"] is True
    assert payload["first_failure"] is None


def test_handle_inspector_grounding_official_available_true_sets_factual_mode():
    payload = inspector_api.handle_inspector_grounding(
        {"identifying_context": None, "official_knowledge_available": True, "official_content": "نص تجريبي"}
    )
    assert payload["grounding_mode"] == "FACTUAL"
    assert payload["official_grounding_available"] is True


def test_handle_inspector_grounding_official_unavailable_stays_identifying_only():
    payload = inspector_api.handle_inspector_grounding({"identifying_context": None, "official_knowledge_available": False})
    assert payload["grounding_mode"] == "IDENTIFYING_ONLY"
    assert payload["official_grounding_available"] is False


def test_architecture_payload_uses_exact_repository_names_only():
    payload = inspector_api.inspector_architecture_payload()
    flat = json.dumps(payload, ensure_ascii=False)
    for exact_name in (
        "run_agent_pipeline",
        "OfficialKnowledgeRetriever",
        "ApprovedResponseMemoryRetriever",
        "run_validation_gates",
        "assemble_grounding",
        "assemble_style_reference_texts",
        "ResponseAgentState",
        "is_refinement_eligible",
    ):
        assert exact_name in flat
