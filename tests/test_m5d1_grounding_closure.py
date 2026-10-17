"""
Phase M5D.1 — official-grounding eligibility closure.

Corrects a defect from M5D: the orchestrator computed the caller-facing
`trusted_grounding_available` eligibility flag as `identifying_context
OR official_grounding_available`, letting identifying context alone (or
combined with Approved Response Memory) grant the "grounded inquiry"
eligibility route -- exactly what M5D's own trust rule forbids. Fixed
in `rcs.response_intelligence.orchestrator` (now sets
`trusted_grounding_available = grounding.official_grounding_available`
directly) and `rcs.response_intelligence.evidence` (the old
`eligibility_grounding_available` OR-helper is deleted, not just
unused).

These tests operate at the ELIGIBILITY-ROUTE level (the actual
`eligibility_rule_id` the orchestrator produces), not just at the
lower-level `GroundingAssembly.official_grounding_available` flag
already covered by tests/test_m5d_refinement.py -- proving the fix
reaches all the way through to real routing behavior.
"""

from __future__ import annotations

import pytest

import datetime as dt
import json
from dataclasses import fields

from rcs.response_generation.prompt import build_identifying_grounding
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest, generate_suggested_response
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
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

pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)
_EMPTY_OK = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)
_VALID_DRAFT = "شكرًا لملاحظتكم حول تفاصيل هذا الاقتراح."
_UNGROUNDED_ROUTE = "inquiry_without_trusted_grounding"
_GROUNDED_ROUTE = "generate_inquiry_with_grounding"


def _req(**overrides) -> GenerationRequest:
    base = dict(
        comment_id="c-1",
        comment_text="كيف يمكنني اقتراح كلمة جديدة للمعجم؟",
        primary_intent="INQUIRY",
        topic_subtype="GENERAL",
        sentiment="NEUTRAL",
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
        return self._responses[min(self.call_count - 1, len(self._responses) - 1)]


def _draft_json(text: str) -> str:
    return json.dumps({"suggested_response": text}, ensure_ascii=False)


def _route(request, ok_result=_EMPTY_OK, arm_result=_EMPTY_ARM) -> str:
    exec_result = run_agent_pipeline(
        request,
        FakeGeminiClient([_draft_json(_VALID_DRAFT)]),
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(ok_result),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(arm_result),
    )
    return exec_result.generation_result.eligibility_rule_id


# --- 1-6: nothing but Official Knowledge can grant the grounded route ------


def test_1_title_only_does_not_enable_grounded_route():
    identifying = build_identifying_grounding("عنوان تجريبي فقط", None)
    assert _route(_req(trusted_grounding_text=identifying)) == _UNGROUNDED_ROUTE


def test_2_entry_id_only_does_not_enable_grounded_route():
    identifying = build_identifying_grounding(None, "ENTRY-123")
    assert _route(_req(trusted_grounding_text=identifying)) == _UNGROUNDED_ROUTE


def test_3_lemma_only_does_not_enable_grounded_route():
    """build_identifying_grounding has no lemma parameter -- proven here
    via a lemma-shaped string passed the same way any identifying-context
    text would be, showing the route stays closed regardless of content."""
    assert _route(_req(trusted_grounding_text="internal lemma: كلمة")) == _UNGROUNDED_ROUTE


def test_4_full_identifying_context_only_does_not_enable_grounded_route():
    identifying = build_identifying_grounding("عنوان تجريبي", "ENTRY-123")
    assert _route(_req(trusted_grounding_text=identifying)) == _UNGROUNDED_ROUTE


def test_5_style_memory_only_does_not_enable_grounded_route():
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(),))
    assert _route(_req(), arm_result=approved) == _UNGROUNDED_ROUTE


def test_6_identifying_context_plus_style_memory_still_does_not_enable_grounded_route():
    identifying = build_identifying_grounding("عنوان تجريبي", "ENTRY-123")
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(),))
    assert _route(_req(trusted_grounding_text=identifying), arm_result=approved) == _UNGROUNDED_ROUTE


# --- 7-10: Official Knowledge outcome matrix --------------------------------


def test_7_official_knowledge_success_with_results_can_enable_grounded_route():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    assert _route(_req(), ok_result=official) == _GROUNDED_ROUTE


def test_8_official_knowledge_success_empty_cannot_enable_grounded_route():
    assert _route(_req(), ok_result=_EMPTY_OK) == _UNGROUNDED_ROUTE


def test_9_official_knowledge_unavailable_cannot_enable_grounded_route():
    official = OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, failure_detail="synthetic")
    assert _route(_req(), ok_result=official) == _UNGROUNDED_ROUTE


def test_10_official_knowledge_error_cannot_enable_grounded_route():
    official = OfficialKnowledgeRetrievalResult(outcome=ERROR, failure_detail="synthetic")
    assert _route(_req(), ok_result=official) == _UNGROUNDED_ROUTE


# --- 11: identifying context still reaches the prompt -----------------------


def test_11_identifying_context_still_preserved_in_gemini_prompt():
    """Phase M6.2: identifying context is policy-gated -- the comment
    must corroborate the title for it to reach the prompt at all. Uses a
    comment that lexically references the title, matching the new
    Context/Trust Policies step (rcs.response_intelligence.
    identifying_context_policy).

    Phase M6.2.1 correction: `entry_id` is internal metadata only and must
    never be rendered into the prompt, regardless of policy outcome --
    only `original_title` (the human-readable identifying reference) is
    ever rendered. This assertion previously required "ENTRY-XYZ" in the
    prompt, which was the exact trust-boundary violation M6.2.1 fixed."""
    client = FakeGeminiClient([_draft_json(_VALID_DRAFT)])
    run_agent_pipeline(
        _req(
            comment_text="لدي سؤال حول «عنوان تجريبي فريد ABC» تحديدًا.",
            identifying_metadata=IdentifyingMetadata("عنوان تجريبي فريد ABC", "ENTRY-XYZ"),
        ),
        client,
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert "عنوان تجريبي فريد ABC" in client.prompts[0]
    assert "ENTRY-XYZ" not in client.prompts[0]


# --- 12: strict official grounding still supports claim validation ---------


def test_12_official_knowledge_still_supports_unsupported_claim_validation():
    claim = "تم تحويل ملاحظتك إلى فريق المحتوى."
    official = OfficialKnowledgeRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="تم تحويل الملاحظات إلى فريق المحتوى دائمًا."),)
    )
    exec_result = run_agent_pipeline(
        _req(),
        FakeGeminiClient([_draft_json(claim)]),
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(official),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert exec_result.generation_result.generation_status == "SUCCEEDED"


def test_12b_style_memory_still_cannot_support_claim_validation():
    claim = "تم تحويل ملاحظتك إلى فريق المحتوى."
    approved = ApprovedResponseMemoryRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="تم تحويل الملاحظات إلى فريق المحتوى دائمًا."),)
    )
    exec_result = run_agent_pipeline(
        _req(),
        FakeGeminiClient([_draft_json(claim), _draft_json(claim)]),
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(approved),
    )
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert exec_result.generation_result.reason_code == "unsupported_operational_claim"


# --- 13: empty-retrieval baseline remains deterministic ---------------------


def test_13_empty_retrieval_baseline_matches_direct_call_deterministically():
    request = _req(primary_intent="THANKS_PRAISE", comment_text="شكرًا لكم على هذا العمل الرائع")
    direct = generate_suggested_response(request, FakeGeminiClient([_draft_json(_VALID_DRAFT)]))
    exec_result = run_agent_pipeline(
        request,
        FakeGeminiClient([_draft_json(_VALID_DRAFT)]),
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert direct == exec_result.generation_result


# --- 14: refinement max stays <= 2 logical generations ----------------------


def test_14_refinement_max_remains_two_even_with_grounding_fix():
    too_short = "شكرًا"
    exec_result = run_agent_pipeline(
        _req(),
        FakeGeminiClient([_draft_json(too_short), _draft_json(_VALID_DRAFT)]),
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert exec_result.metadata.logical_generation_count == 2


# --- 15: GenerationResult schema unchanged ----------------------------------


def test_15_generation_result_schema_unchanged():
    field_names = {f.name for f in fields(GenerationResult)}
    assert field_names == {
        "comment_id", "routing_decision", "generation_status", "suggested_response",
        "review_required", "reason_code", "provider", "model_name", "prompt_version",
        "generated_at", "future_reviewer_actions", "eligibility_rule_id", "error",
    }


# --- 16: persistence / reviewer-approved row protection unchanged ----------


def test_16_reviewer_approved_row_still_protected_after_grounding_fix(tmp_path):
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    engine = get_engine(tmp_path / "m5d1_persistence.sqlite")
    init_db(engine)
    upsert_review_items(engine, [dict(
        comment_id="c-1", source_file="f", source_row_index=0,
        review_status="APPROVED", is_prefiltered_junk=False, ready_to_send=True, row_version=1,
        reviewer_action="SEND_AS_IS", final_reply="رد المراجع المعتمد.",
    )])
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    exec_result = run_agent_pipeline(
        _req(),
        FakeGeminiClient([_draft_json(_VALID_DRAFT)]),
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(official),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        exec_result.generation_result, intent="INQUIRY", topic="GENERAL", sentiment="NEUTRAL"
    )
    assert detail.review_status == "APPROVED"
    assert detail.final_response == "رد المراجع المعتمد."
