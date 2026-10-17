"""
Phase M5D — grounding enforcement + bounded (at most one) refinement.

No Hugging Face model is ever downloaded here: retrieval uses plain
fake retrievers; Gemini uses a scripted fake client. Every fixture is
clearly synthetic Arabic placeholder text.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import inspect
import json
from dataclasses import fields
from pathlib import Path

import pytest

from rcs.response_generation import refinement as refinement_module
from rcs.response_generation import validate as validate_module
from rcs.response_generation.prompt import build_generation_prompt
from rcs.response_generation.refinement import (
    NON_REFINABLE_PRE_VALIDATION_REASON_CODES,
    NON_REFINABLE_VALIDATION_REASON_CODES,
    REFINEMENT_ELIGIBLE_REASON_CODES,
    RefinementContext,
    is_refinement_eligible,
)
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest, generate_suggested_response
from rcs.response_generation.validate import validate_draft
from rcs.response_intelligence import evidence
from rcs.response_intelligence import orchestrator as orchestrator_module
from rcs.response_intelligence import retrievers as retrievers_module
from rcs.response_intelligence.orchestrator import (
    AgentExecutionMetadata,
    generate_response_with_orchestrator,
    run_agent_pipeline,
)
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

_VALID_DRAFT = "شكرًا لملاحظتكم القيّمة حول هذا المدخل."
_TOO_SHORT_DRAFT = "شكرًا"
_BOILERPLATE_DRAFT = "أهلًا بك، نشكرك على تواصلك واهتمامك بالمعجم."
_PII_DRAFT = "شكرًا لتواصلكم على الرقم 0501234567 مع المعجم."
_LEAK_DRAFT = "شكرًا، نموذج MARBERTv2 واثق بنسبة confidence عالية."
_UNSUPPORTED_CLAIM_DRAFT = "تم تحويل ملاحظتك إلى فريق المحتوى."


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def _req(**overrides) -> GenerationRequest:
    base = dict(
        comment_id="c-1",
        comment_text="هذا تعليق تجريبي اصطناعي بخصوص الكلمة المقترحة.",
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        sentiment="POSITIVE",
        confidence=0.9,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def _ok_result(**overrides) -> OfficialKnowledgeResult:
    base = dict(
        source_id="OK-1",
        source_title="عنوان رسمي تجريبي",
        source_type="dictionary_entry",
        content="محتوى رسمي تجريبي معتمد.",
        version="v1",
        effective_from=_SYNTHETIC_NOW,
        authoritative=True,
        retrieval_score=0.9,
    )
    base.update(overrides)
    return OfficialKnowledgeResult(**base)


def _arm_result(**overrides) -> ApprovedResponseMemoryResult:
    base = dict(
        review_item_id="r-1",
        final_reply_id="hash-1",
        response_text="هذا مثال أسلوب تجريبي معتمد سابقًا.",
        reviewer_action="SEND_AS_IS",
        approved_at=_SYNTHETIC_NOW,
        retrieval_score=0.9,
    )
    base.update(overrides)
    return ApprovedResponseMemoryResult(**base)


class FakeOfficialKnowledgeRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query: OfficialKnowledgeQuery):
        self.calls.append(query)
        return self.result


class FakeApprovedResponseMemoryRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query: ApprovedResponseMemoryQuery):
        self.calls.append(query)
        return self.result


class FakeGeminiClient:
    """Scripted client. `responses` items are either a JSON string to
    return or an Exception instance to raise on that call."""

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


def _pipeline(raw_request, ok_result=_EMPTY_OK, arm_result=_EMPTY_ARM, responses=(), max_retries=0):
    client = FakeGeminiClient(list(responses)) if responses else None
    exec_result = run_agent_pipeline(
        raw_request,
        client,
        max_retries,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(ok_result),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(arm_result),
    )
    return exec_result, client


# ============================================================================
# Group A (1-11): grounding enforcement / trust boundary
# ============================================================================


def test_1_official_knowledge_can_establish_factual_grounding():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    grounding = evidence.assemble_grounding(None, official)
    assert grounding.official_grounding_available is True
    assert grounding.mode == evidence.GROUNDING_MODE_FACTUAL


def test_2_style_memory_alone_cannot_establish_factual_grounding():
    exec_result, _ = _pipeline(
        _req(primary_intent="INQUIRY"),
        arm_result=ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(),)),
        responses=[_draft_json(_VALID_DRAFT)],
    )
    assert exec_result.metadata.grounding_mode == evidence.GROUNDING_MODE_IDENTIFYING_ONLY


def test_3_identifying_context_alone_cannot_establish_factual_grounding():
    grounding = evidence.assemble_grounding("submission title: X", _EMPTY_OK)
    assert grounding.official_grounding_available is False
    assert grounding.mode == evidence.GROUNDING_MODE_IDENTIFYING_ONLY


def test_4_title_alone_cannot_establish_factual_grounding():
    from rcs.response_generation.prompt import build_identifying_grounding

    identifying = build_identifying_grounding("عنوان تجريبي فقط", None)
    grounding = evidence.assemble_grounding(identifying, _EMPTY_OK)
    assert grounding.official_grounding_available is False


def test_5_entry_id_alone_cannot_establish_factual_grounding():
    from rcs.response_generation.prompt import build_identifying_grounding

    identifying = build_identifying_grounding(None, "ENTRY-123")
    grounding = evidence.assemble_grounding(identifying, _EMPTY_OK)
    assert grounding.official_grounding_available is False


def test_6_lemma_shaped_text_alone_cannot_establish_factual_grounding():
    """`build_identifying_grounding` never even takes a lemma parameter
    -- this proves the more general point that ANY identifying-context-
    shaped text (lemma included) is treated identically: never factual."""
    lemma_shaped_text = "internal lemma: كلمة"
    grounding = evidence.assemble_grounding(lemma_shaped_text, _EMPTY_OK)
    assert grounding.official_grounding_available is False


def test_7_classification_labels_cannot_establish_factual_grounding():
    """Structural proof: assemble_grounding's signature has no
    intent/topic/sentiment parameter at all -- classification signals
    are structurally unreachable from this function."""
    sig = inspect.signature(evidence.assemble_grounding)
    assert set(sig.parameters) == {"identifying_context", "official_result"}


def test_8_old_trusted_grounding_text_field_cannot_bypass_official_knowledge():
    """A caller cannot pre-set request.trusted_grounding_text to
    something official-looking and have it survive the orchestrator's
    recomputation. The orchestrator always recomputes trusted_grounding_
    text from evidence.assemble_grounding(identifying_context,
    official_result) -- the ORIGINAL field's value is only ever used as
    the `identifying_context` INPUT, never trusted directly as official."""
    request = _req(
        primary_intent="INQUIRY",
        trusted_grounding_available=True,
        trusted_grounding_text="تم رفع البلاغ فعليًا وسيتم حله قريبًا.",  # an operational-claim-shaped string
    )
    exec_result, client = _pipeline(request, responses=[_draft_json(_UNSUPPORTED_CLAIM_DRAFT)])
    # With no real Official Knowledge, the SAME operational claim in the
    # draft must still fail validation -- the pre-set text did not
    # smuggle itself in as usable grounding for the claim gate either,
    # because assemble_grounding's fallback path passes it through only
    # as identifying_context/prompt_text, and _UNSUPPORTED_ACTION_PATTERNS
    # matching against grounding text is validate.py's own unmodified
    # logic; the point proven here is grounding_mode never becomes FACTUAL.
    assert exec_result.metadata.grounding_mode == evidence.GROUNDING_MODE_IDENTIFYING_ONLY
    assert exec_result.metadata.final_failure_code in (None, "unsupported_operational_claim")


def test_9_official_knowledge_can_enable_the_grounded_inquiry_route():
    request = _req(primary_intent="INQUIRY", topic_subtype="GENERAL", trusted_grounding_text=None)
    exec_result, _ = _pipeline(
        request,
        ok_result=OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),)),
        responses=[_draft_json(_VALID_DRAFT)],
    )
    assert exec_result.generation_result.eligibility_rule_id == "generate_inquiry_with_grounding"
    assert exec_result.metadata.grounding_mode == evidence.GROUNDING_MODE_FACTUAL


def test_10_style_memory_cannot_enable_the_grounded_inquiry_route():
    request = _req(primary_intent="INQUIRY", topic_subtype="GENERAL", trusted_grounding_text=None)
    exec_result, _ = _pipeline(
        request,
        arm_result=ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(),)),
        responses=[_draft_json(_VALID_DRAFT)],
    )
    assert exec_result.generation_result.eligibility_rule_id == "inquiry_without_trusted_grounding"
    assert exec_result.metadata.grounding_mode == evidence.GROUNDING_MODE_IDENTIFYING_ONLY


def test_11_identifying_context_alone_cannot_flip_official_grounding_available():
    """Superseded by M5D.1: this file originally documented an M5D
    compatibility exception where identifying context alone was allowed
    to satisfy the broader, caller-facing `trusted_grounding_available`
    ELIGIBILITY flag. That exception was found to violate M5D's own
    trust rule and was removed in M5D.1 (see
    tests/test_m5d1_grounding_closure.py, which proves the corrected
    eligibility-ROUTE-level behavior directly). What this test still
    proves, unchanged since M5D: identifying context alone never flips
    the strictly-official `official_grounding_available` flag."""
    request = _req(primary_intent="INQUIRY", topic_subtype="GENERAL", trusted_grounding_text="submission title: X")
    exec_result, _ = _pipeline(request, responses=[_draft_json(_VALID_DRAFT)])
    assert exec_result.metadata.grounding_mode == evidence.GROUNDING_MODE_IDENTIFYING_ONLY


# ============================================================================
# Group B (12-20): call-count / control-flow invariants
# ============================================================================


def test_12_initial_valid_draft_exactly_one_logical_call():
    exec_result, client = _pipeline(_req(), responses=[_draft_json(_VALID_DRAFT)])
    assert client.call_count == 1
    assert exec_result.metadata.logical_generation_count == 1
    assert exec_result.metadata.refinement_attempted is False


def test_13_refinable_validation_failure_exactly_two_logical_calls():
    exec_result, client = _pipeline(
        _req(), responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)]
    )
    assert client.call_count == 2
    assert exec_result.metadata.logical_generation_count == 2
    assert exec_result.metadata.refinement_attempted is True


def test_14_non_refinable_validation_failure_exactly_one_logical_call():
    exec_result, client = _pipeline(_req(), responses=[_draft_json(_PII_DRAFT)])
    assert client.call_count == 1
    assert exec_result.metadata.logical_generation_count == 1
    assert exec_result.metadata.refinement_attempted is False
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"


def test_15_refined_draft_valid_final_success():
    exec_result, _ = _pipeline(_req(), responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)])
    assert exec_result.generation_result.generation_status == "SUCCEEDED"
    assert exec_result.generation_result.suggested_response == _VALID_DRAFT
    assert exec_result.metadata.final_validation_passed is True


def test_16_refined_draft_invalid_stops_after_two():
    exec_result, client = _pipeline(
        _req(), responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_BOILERPLATE_DRAFT)]
    )
    assert client.call_count == 2
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert exec_result.metadata.final_failure_code == "generic_boilerplate"


def test_17_refinement_gemini_transport_failure_stops_after_two():
    import httpx

    exec_result, client = _pipeline(
        _req(),
        responses=[_draft_json(_BOILERPLATE_DRAFT), httpx.ConnectError("synthetic transport failure")],
    )
    assert client.call_count == 2
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert exec_result.generation_result.reason_code == "gemini_api_error"
    assert exec_result.metadata.logical_generation_count == 2


@pytest.mark.parametrize(
    "responses",
    [
        [_draft_json(_VALID_DRAFT)],
        [_draft_json(_PII_DRAFT)],
        [_draft_json(_LEAK_DRAFT)],
        [_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)],
        [_draft_json(_BOILERPLATE_DRAFT), _draft_json(_BOILERPLATE_DRAFT)],
        [_draft_json(_TOO_SHORT_DRAFT), _draft_json(_TOO_SHORT_DRAFT)],
    ],
)
def test_18_no_third_draft_under_any_path(responses):
    _exec_result, client = _pipeline(_req(), responses=responses)
    assert client.call_count <= 2


def test_19_no_recursive_refinement():
    """Static proof: run_agent_pipeline never calls itself."""
    source = inspect.getsource(orchestrator_module.run_agent_pipeline)
    tree = ast.parse(source)
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "run_agent_pipeline" not in calls


def test_20_no_while_loop_agent_retry():
    source = inspect.getsource(orchestrator_module)
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.While) for node in ast.walk(tree))


# ============================================================================
# Group C (21-27): reason-code allowlist / policy
# ============================================================================


def _trigger_all_real_validation_reason_codes() -> set[str]:
    """Cross-check REFINEMENT_ELIGIBLE_REASON_CODES against reason codes
    ACTUALLY produced by the real, unmodified run_validation_gates --
    not a hand-maintained duplicate list."""
    cases = [
        (None, "empty_output", None),
        ("قصير", "too_short", None),
        ("ط" * 700, "excessive_length", None),
        ("short but only ascii letters here", "missing_arabic", None),
        (_LEAK_DRAFT, "internal_leak", None),
        (_PII_DRAFT, "pii_in_draft", None),
        (_BOILERPLATE_DRAFT, "generic_boilerplate", None),
        ("ملاحظة دقيقة وتفصيل وافٍ حول الكلمة المقترحة في هذا السياق.", "evaluator_voice_opening", None),
        (_UNSUPPORTED_CLAIM_DRAFT, "unsupported_operational_claim", None),
        # Ambiguous/Insufficient-Context Response Handling milestone: the
        # draft attributes a specific suggestion to the visitor
        # ("اقتراحكم") with a comment that never said so.
        ("نقدّر اقتراحكم لإضافة تركيب «أنشطة لا صفية».", "unsupported_visitor_intent_claim", "أنشطة لا صفية"),
        ("نشكركم على مشاركتنا مقترحاتكم وملاحظاتكم.", "wrong_pronoun_construction", "اقتراح"),
        # reply-safety gates
        ("شكرًا لكم، سنقوم بمراجعة الملاحظة وإضافتها قريبًا.", "unsupported_commitment", "ملاحظة"),
        ("تخضع الاقتراحات لمراجعة لجنة متخصصة قبل الاعتماد.", "unsupported_policy_claim", "اقتراح"),
        ("بلغ عدد الكلمات المضافة هذا العام 1200 كلمة جديدة.", "unsupported_number", "اقتراح"),
        ("معنى كلمة «سوار» في المعجم هو حلية تلبس في المعصم.", "unsupported_definition", "ما معنى سوار"),
        ("التعريف الوارد في المعجم صحيح ولا يوجد فيه خطأ أبدا.", "unsupported_correctness_claim", "التعريف خاطئ"),
        ("يمكنكم مراجعة الصفحة https://example.org/help للمزيد.", "unapproved_url", "اقتراح"),
        ("يعتمد المعجم المصادر الأكاديمية فقط في إدراج الكلمات الجديدة.", "unsupported_assertion", "اقتراح"),
    ]
    observed = set()
    for text, expected, comment_text in cases:
        ok, why = validate_draft(text, comment_text=comment_text)
        assert not ok and why == expected, f"fixture drift: expected {expected!r}, got {why!r}"
        observed.add(why)
    return observed


def test_21_refinement_allowlist_uses_only_real_validation_reason_codes():
    real_codes = _trigger_all_real_validation_reason_codes()
    assert REFINEMENT_ELIGIBLE_REASON_CODES <= real_codes
    # And the two non-refinable-but-real codes are accounted for, not forgotten:
    assert NON_REFINABLE_VALIDATION_REASON_CODES <= real_codes
    assert REFINEMENT_ELIGIBLE_REASON_CODES.isdisjoint(NON_REFINABLE_VALIDATION_REASON_CODES)


def test_22_unknown_reason_code_never_refinement_eligible():
    assert is_refinement_eligible("FAILED_REVIEW_ONLY", "totally_made_up_reason_code") is False
    assert is_refinement_eligible("FAILED_REVIEW_ONLY", None) is False


def test_23_pii_failure_policy_is_non_refinable():
    assert "pii_in_draft" not in REFINEMENT_ELIGIBLE_REASON_CODES
    exec_result, client = _pipeline(_req(), responses=[_draft_json(_PII_DRAFT)])
    assert client.call_count == 1
    assert exec_result.generation_result.reason_code == "pii_in_draft"


def test_24_internal_leak_failure_policy_is_non_refinable():
    assert "internal_leak" not in REFINEMENT_ELIGIBLE_REASON_CODES
    exec_result, client = _pipeline(_req(), responses=[_draft_json(_LEAK_DRAFT)])
    assert client.call_count == 1
    assert exec_result.generation_result.reason_code == "internal_leak"


def test_25_malformed_schema_policy_is_non_refinable():
    assert "malformed_gemini_output" not in REFINEMENT_ELIGIBLE_REASON_CODES
    exec_result, client = _pipeline(_req(), responses=["not-json-at-all"])
    assert client.call_count == 1
    assert exec_result.generation_result.reason_code == "malformed_gemini_output"


def test_26_provider_transport_failure_policy_is_non_refinable():
    import httpx

    assert "gemini_api_error" not in REFINEMENT_ELIGIBLE_REASON_CODES
    assert "gemini_client_missing" not in REFINEMENT_ELIGIBLE_REASON_CODES
    exec_result, client = _pipeline(_req(), responses=[httpx.ConnectError("synthetic non-retryable failure")])
    assert client.call_count == 1
    assert exec_result.generation_result.reason_code == "gemini_api_error"
    assert exec_result.metadata.logical_generation_count == 1


def test_27_unsupported_claim_refinement_behavior_proven():
    assert "unsupported_operational_claim" in REFINEMENT_ELIGIBLE_REASON_CODES
    exec_result, client = _pipeline(
        _req(primary_intent="INQUIRY"),
        responses=[_draft_json(_UNSUPPORTED_CLAIM_DRAFT), _draft_json(_VALID_DRAFT)],
    )
    assert client.call_count == 2
    assert exec_result.generation_result.generation_status == "SUCCEEDED"


# ============================================================================
# Group D (28-35): no fact invention, separation preserved, no CoT
# ============================================================================


def test_28_no_official_knowledge_refinement_cannot_invent_grounding():
    exec_result, _ = _pipeline(
        _req(primary_intent="INQUIRY"),
        responses=[_draft_json(_UNSUPPORTED_CLAIM_DRAFT), _draft_json(_UNSUPPORTED_CLAIM_DRAFT)],
    )
    assert exec_result.metadata.grounding_mode == evidence.GROUNDING_MODE_IDENTIFYING_ONLY
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert exec_result.generation_result.reason_code == "unsupported_operational_claim"


def test_29_style_memory_cannot_rescue_unsupported_factual_claim():
    exec_result, client = _pipeline(
        _req(primary_intent="INQUIRY"),
        arm_result=ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(),)),
        responses=[_draft_json(_UNSUPPORTED_CLAIM_DRAFT), _draft_json(_UNSUPPORTED_CLAIM_DRAFT)],
    )
    assert "STYLE REFERENCE EXAMPLES" in client.prompts[1]
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert exec_result.generation_result.reason_code == "unsupported_operational_claim"


def test_30_identifying_context_cannot_rescue_unsupported_factual_claim():
    exec_result, _ = _pipeline(
        _req(primary_intent="INQUIRY", trusted_grounding_text="submission title: كلمة تجريبية"),
        responses=[_draft_json(_UNSUPPORTED_CLAIM_DRAFT), _draft_json(_UNSUPPORTED_CLAIM_DRAFT)],
    )
    assert exec_result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert exec_result.generation_result.reason_code == "unsupported_operational_claim"


def test_31_official_evidence_separately_labeled_in_refinement_prompt():
    exec_result, client = _pipeline(
        _req(),
        ok_result=OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="حقيقة رسمية XYZ"),)),
        responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)],
    )
    refinement_prompt = client.prompts[1]
    assert "APPROVED FACTUAL statements" in refinement_prompt
    assert "حقيقة رسمية XYZ" in refinement_prompt
    assert "REFINEMENT INSTRUCTION" in refinement_prompt


def test_32_style_examples_separately_labeled_non_factual_in_refinement():
    exec_result, client = _pipeline(
        _req(),
        arm_result=ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="نمط ABC"),)),
        responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)],
    )
    refinement_prompt = client.prompts[1]
    assert "STYLE REFERENCE EXAMPLES" in refinement_prompt
    assert "NOT factual" in refinement_prompt
    assert "نمط ABC" in refinement_prompt


def test_33_first_draft_included_only_in_refinement_prompt_not_initial():
    _exec_result, client = _pipeline(_req(), responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)])
    initial_prompt, refinement_prompt = client.prompts
    assert "REFINEMENT INSTRUCTION" not in initial_prompt
    assert "PREVIOUS DRAFT" not in initial_prompt
    assert "REFINEMENT INSTRUCTION" in refinement_prompt
    assert _BOILERPLATE_DRAFT in refinement_prompt


def test_34_no_chain_of_thought_request_in_refinement_prompt():
    _exec_result, client = _pipeline(_req(), responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)])
    refinement_prompt = client.prompts[1]
    banned_phrases = ("think step by step", "explain why", "reasoning:", "chain of thought", "show your work")
    lowered = refinement_prompt.lower()
    for phrase in banned_phrases:
        assert phrase not in lowered


def test_35_no_hidden_rationale_field_on_metadata_and_no_logging_calls():
    """Phase M5E note: `AgentExecutionMetadata` gained 7 more fields
    (execution_id + retrieval-outcome/count + generation_status) to
    serve as the audit-trace hook point M5D's own report promised --
    all still short codes/counts/booleans, never content, so this test
    still proves the field is content-free by construction (every field
    name is checked, none is a prompt/text-shaped name). `logging.` is
    no longer banned in orchestrator.py: M5E explicitly requires a
    sanitized (execution_id/review_item_id/exception-type-only) warning
    on audit-persistence failure -- see
    tests/test_m5e_agent_audit.py for the direct proof that this
    logging call never includes prompt/comment/PII content.

    Phase M6.2 note: 3 more fields were added for the Identifying
    Context Policy decision (allowed/reason/matched_reference) -- a
    closed reason code, a boolean, and a bounded matched-term string
    (never visitor comment text), so this exact-set assertion is
    intentionally updated, not silently left broken."""
    metadata_fields = {f.name for f in fields(AgentExecutionMetadata)}
    assert metadata_fields == {
        "execution_id",
        "logical_generation_count",
        "refinement_attempted",
        "initial_validation_passed",
        "initial_failure_code",
        "final_validation_passed",
        "final_failure_code",
        "grounding_mode",
        "official_grounding_available",
        "official_retrieval_outcome",
        "official_result_count",
        "style_retrieval_outcome",
        "style_result_count",
        "generation_status",
        "identifying_context_allowed",
        "identifying_context_decision_reason",
        "identifying_context_matched_reference",
    }
    content_shaped_substrings = ("prompt", "reasoning", "chain", "thought", "comment_text", "raw_")
    for name in metadata_fields:
        lowered = name.lower()
        for banned in content_shaped_substrings:
            assert banned not in lowered, f"content-shaped field name found: {name!r}"

    refinement_code = inspect.getsource(refinement_module)
    assert "logging." not in refinement_code
    assert "print(" not in inspect.getsource(orchestrator_module)
    assert "print(" not in refinement_code


# ============================================================================
# Group E (36-41): schema/persistence/lifecycle preservation
# ============================================================================


def test_36_generation_result_schema_preserved():
    field_names = {f.name for f in fields(GenerationResult)}
    assert field_names == {
        "comment_id", "routing_decision", "generation_status", "suggested_response",
        "review_required", "reason_code", "provider", "model_name", "prompt_version",
        "generated_at", "future_reviewer_actions", "eligibility_rule_id", "error",
    }


def test_37_refined_result_accepted_by_apply_generation_result(tmp_path):
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    engine = get_engine(tmp_path / "m5d_refinement_persistence.sqlite")
    init_db(engine)
    upsert_review_items(engine, [dict(
        comment_id="c-1", source_file="f", source_row_index=0,
        review_status="CLASSIFIED", is_prefiltered_junk=False, ready_to_send=False, row_version=1,
        ai_primary_intent="THANKS_PRAISE",
    )])
    exec_result, _ = _pipeline(_req(), responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)])
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        exec_result.generation_result, intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE"
    )
    assert detail.review_status == "DRAFT_READY"
    assert detail.suggested_response == _VALID_DRAFT


def test_38_immutable_human_review_semantics_preserved(tmp_path):
    """A row a reviewer already acted on must stay untouched even when
    fed a refinement-produced GenerationResult -- same protection
    apply_generation_result has always had, exercised here through the
    new code path."""
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    engine = get_engine(tmp_path / "m5d_immutability.sqlite")
    init_db(engine)
    upsert_review_items(engine, [dict(
        comment_id="c-1", source_file="f", source_row_index=0,
        review_status="APPROVED", is_prefiltered_junk=False, ready_to_send=True, row_version=1,
        reviewer_action="SEND_AS_IS", final_reply="رد المراجع النهائي المعتمد.",
    )])
    exec_result, _ = _pipeline(_req(), responses=[_draft_json(_BOILERPLATE_DRAFT), _draft_json(_VALID_DRAFT)])
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        exec_result.generation_result, intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE"
    )
    assert detail.review_status == "APPROVED"
    assert detail.final_response == "رد المراجع النهائي المعتمد."


@pytest.mark.parametrize(
    "overrides",
    [
        dict(primary_intent="COMPLAINT", sentiment="NEGATIVE", comment_text="الموقع بطيء جدًا."),
        dict(primary_intent="SUGGESTION", comment_text="أقترح إضافة كلمة جديدة."),
        dict(topic_subtype="TECHNICAL_ISSUE", comment_text="الصوت لا يعمل في هذا المدخل."),
        dict(topic_subtype="WORD_OF_YEAR_NOMINATION", comment_text="أرشح هذه الكلمة لكلمة العام."),
    ],
)
def test_39_eligibility_routing_unaffected_for_non_grounding_scenarios(overrides):
    request = _req(**overrides)
    direct = generate_suggested_response(request, FakeGeminiClient([_draft_json(_VALID_DRAFT)]))
    exec_result, _ = _pipeline(request, responses=[_draft_json(_VALID_DRAFT)])
    assert direct.eligibility_rule_id == exec_result.generation_result.eligibility_rule_id
    assert direct.routing_decision == exec_result.generation_result.routing_decision


def test_40_no_reviewer_isolation_coupling_introduced():
    for module in (orchestrator_module, refinement_module, evidence):
        code = inspect.getsource(module)
        assert "reviewer_isolation" not in code
        assert "rcs.review.sessions" not in code


def test_41_no_assignment_behavior_coupling_introduced():
    for module in (orchestrator_module, refinement_module, evidence):
        code = inspect.getsource(module)
        assert "assign_batch" not in code
        assert "AssignmentResult" not in code


# ============================================================================
# Group F (42-52): degraded behavior, DI/lifecycle, out-of-scope guards
# ============================================================================


def test_42_both_retrievers_empty_initial_valid_response_retains_baseline_behavior():
    request = _req()
    client_direct = FakeGeminiClient([_draft_json(_VALID_DRAFT)])
    direct = generate_suggested_response(request, client_direct)
    exec_result, client_orchestrated = _pipeline(request, responses=[_draft_json(_VALID_DRAFT)])
    assert direct == exec_result.generation_result
    assert client_direct.prompts == client_orchestrated.prompts


def test_43_official_retrieval_unavailable_degrades_safely():
    official = OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, failure_detail="synthetic")
    exec_result, client = _pipeline(_req(), ok_result=official, responses=[_draft_json(_VALID_DRAFT)])
    assert exec_result.generation_result.generation_status == "SUCCEEDED"
    assert exec_result.metadata.grounding_mode == evidence.GROUNDING_MODE_IDENTIFYING_ONLY
    assert client.call_count == 1


def test_44_style_retrieval_unavailable_degrades_safely():
    approved = ApprovedResponseMemoryRetrievalResult(outcome=UNAVAILABLE, failure_detail="synthetic")
    exec_result, client = _pipeline(_req(), arm_result=approved, responses=[_draft_json(_VALID_DRAFT)])
    assert exec_result.generation_result.generation_status == "SUCCEEDED"
    assert "STYLE REFERENCE EXAMPLES" not in client.prompts[0]


@pytest.fixture(autouse=True)
def _reset_retriever_cache():
    retrievers_module.reset_default_retrievers_for_testing()
    yield
    retrievers_module.reset_default_retrievers_for_testing()


def test_45_no_e5_load_or_index_rebuild_per_request_via_defaults():
    client_a = FakeGeminiClient([_draft_json(_VALID_DRAFT)])
    client_b = FakeGeminiClient([_draft_json(_VALID_DRAFT)])
    generate_response_with_orchestrator(_req(comment_id="a"), client_a)
    generate_response_with_orchestrator(_req(comment_id="b"), client_b)
    embedder = retrievers_module._get_shared_embedder()
    assert embedder.is_loaded is False  # today's empty corpora never touch it
    assert retrievers_module.default_official_knowledge_retriever() is retrievers_module.default_official_knowledge_retriever()
    assert (
        retrievers_module.default_approved_response_memory_retriever()
        is retrievers_module.default_approved_response_memory_retriever()
    )


def _all_m5d_module_files() -> list[str]:
    return [
        inspect.getsourcefile(orchestrator_module),
        inspect.getsourcefile(refinement_module),
        inspect.getsourcefile(evidence),
    ]


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


def test_46_no_external_dictionary():
    for path in _all_m5d_module_files():
        assert "external_dictionary_client" not in _code_excluding_docstrings(path).lower()


def test_47_no_vector_database():
    banned = ("pgvector", "pinecone", "chromadb", "weaviate", "qdrant", "elasticsearch", "faiss")
    for path in _all_m5d_module_files():
        code = _code_excluding_docstrings(path).lower()
        for term in banned:
            assert term not in code


def test_48_no_langgraph():
    for path in _all_m5d_module_files():
        assert "langgraph" not in _code_excluding_docstrings(path).lower()


def test_49_no_react_framework():
    for path in _all_m5d_module_files():
        code = _code_excluding_docstrings(path).lower()
        assert "react_agent" not in code
        assert "from langchain" not in code


def test_50_no_autonomous_gemini_tool_selection():
    """SupportsGenerate stays a plain (prompt, schema) -> str call --
    no tool/function-calling schema, no tool registry, anywhere in the
    M5D modules."""
    for path in _all_m5d_module_files():
        code = _code_excluding_docstrings(path)
        assert "tool_config" not in code
        assert "FunctionDeclaration" not in code
        assert "automatic_function_calling" not in code


def test_51_no_db_schema_changes():
    """Originally: no new migration and no ORM column added for M5D --
    true at the time (M5D's own diff never touched models.py). Phase
    M5E, LATER, explicitly and separately added one new, purely
    additive ORM class (`AgentExecutionTrace`) plus one Alembic
    migration -- see tests/test_agent_audit_repository.py for that
    phase's own dedicated schema tests. This test's scope is narrowed
    accordingly: it no longer bans ANY diff to models.py (that ban's
    premise became obsolete the moment a later phase legitimately
    needed new schema), but it still catches the thing that actually
    matters here -- no EXISTING schema definition may be silently
    modified or removed, only new additions are permitted."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "HEAD", "--", "src/rcs/review/models.py"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    schema_shaped = ("mapped_column", "__tablename__", "class ", "__table_args__")
    removed_schema_lines = [
        line
        for line in diff.stdout.splitlines()
        if line.startswith("-")
        and not line.startswith("---")
        and any(marker in line for marker in schema_shaped)
    ]
    assert removed_schema_lines == [], f"existing schema definitions were removed/modified: {removed_schema_lines}"


def test_52_no_frontend_coupling_in_backend_m5d_modules():
    """A real `git status` audit (done manually per the M5D final
    report) is the authoritative proof that no `web/` file changed --
    a unit test cannot distinguish this session's changes from
    pre-existing uncommitted ones via git alone. This test instead
    proves the narrower, machine-checkable fact: none of the new/
    modified M5D backend modules reference the frontend at all."""
    for path in _all_m5d_module_files():
        code = _code_excluding_docstrings(path)
        assert "web/" not in code
        assert "ReviewerWorkspace" not in code
