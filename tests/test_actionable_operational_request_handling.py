"""
Tests for how the Response Agent handles a visitor's explicit request
for an action or operational outcome.

Root cause under audit: when a visitor explicitly asks the institution
to DO something (send content, contact them, cooperate/partner, visit,
add/correct an entry, approve a suggestion), the Response Agent had no
explicit situation hint for this category and the deterministic
validator's org-action-claim phrase list (`validate.
_UNSUPPORTED_ACTION_PATTERNS`) only covered phrasings evidenced from
earlier complaint/inquiry review findings ("تم تحويل", "سيتم التواصل",
...) -- not the active-voice/completed forms most likely for THESE
specific action verbs ("سنرسل", "تم إرسال", "سنعتمد", "تم إضافة",
"سنعدل", "سنتعاون", ...). Existing general Hard Rules already banned
"promises"/"completed actions"/"will act" in principle (see prompt.py's
Hard rules), but -- consistent with every prior human-review finding in
this codebase -- a scattered/implicit rule does not reliably generalize
to a new named situation without an explicit hint.

Fix (both narrow, no new taxonomy, no new Gemini call, no new gate,
reuses the EXISTING "unsupported_operational_claim" reason code):
  1. prompt.py -- one added Hard Rule plus one situation hint naming
     this category directly (primary mechanism; not independently
     testable without a live Gemini call, verified structurally below).
  2. validate.py's `_UNSUPPORTED_ACTION_PATTERNS` -- extended with the
     missing send/contact/cooperate/add/correct/approve execution-claim
     phrasings, still a small, bounded, evidence-anchored list (no
     stemming, no generic verb-prefix regex), reusing the EXISTING
     `_unsupported_claim_gate` and its EXISTING reason code.

Fast, deterministic -- no real model load, no network, no Gemini call.
"""

from __future__ import annotations

import pytest
import datetime as dt

from config import settings
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS
from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES
from rcs.response_generation.service import GenerationRequest
from rcs.response_generation.validate import validate_draft
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
)
from scripts.offline_eval.response_agent.fakes import (
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
    FakeOfficialKnowledgeRetriever,
)

_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def _empty_ok():
    return FakeOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY))


def _run(request, client, ok=None, arm=None):
    return run_agent_pipeline(
        request, client=client,
        official_knowledge_retriever=ok or _empty_ok(),
        approved_response_retriever=arm or FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )


def _req(comment_text, primary_intent, topic_subtype, sentiment="NEUTRAL"):
    return GenerationRequest(
        comment_id="x", comment_text=comment_text, primary_intent=primary_intent,
        topic_subtype=topic_subtype, sentiment=sentiment, confidence=0.4,
    )


def _draft_json(text: str) -> str:
    import json

    return json.dumps({"suggested_response": text}, ensure_ascii=False)


# --- 1. Prompt-level fix is present (structural) ---------------------------


def test_hard_rule_against_claiming_or_promising_execution_is_present():
    assert (
        "never state or imply that you, this draft, or the institution has performed, "
        "is performing, or will perform that action"
    ) in SYSTEM_INSTRUCTIONS


def test_actionable_request_situation_hint_is_present():
    assert "Actionable / operational request" in SYSTEM_INSTRUCTIONS
    assert "never say or imply it has been done, is being done, or will be done" in SYSTEM_INSTRUCTIONS


def test_prompt_version_bumped_to_rg_v12():
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"


def test_prior_hard_rules_untouched():
    """This change is response-behavior only -- the visitor-intent-
    attribution rule and the factual-grounding rule must remain exactly
    as they were."""
    assert "Do not attribute a specific action, request, or proposal to the visitor" in SYSTEM_INSTRUCTIONS
    assert (
        "Do not confirm, validate, or correct a visitor's linguistic, scientific, or factual claim"
        in SYSTEM_INSTRUCTIONS
    )


# --- 2. Validator backstop -- direct unit tests -----------------------------
# Diagnostic set A-H, each as a REQUEST-IS-CLEAR case where the
# draft nonetheless (wrongly) claims execution/commitment.


def test_A_file_delivery_request_rejects_claimed_or_promised_send():
    ok1, why1 = validate_draft("تم إرسال الملف إليكم.", comment_text="أرسلوا لي الملف")
    assert ok1 is False and why1 == "unsupported_operational_claim"
    ok2, why2 = validate_draft("سنرسل لكم نسخة Excel قريبًا.", comment_text="أحتاج نسخة Excel")
    assert ok2 is False and why2 == "unsupported_operational_claim"


def test_B_contact_request_rejects_claimed_or_promised_contact():
    ok1, why1 = validate_draft("سيتم التواصل معكم قريبًا.", comment_text="تواصلوا معي")
    assert ok1 is False and why1 == "unsupported_operational_claim"
    ok2, why2 = validate_draft("سنتواصل معكم على الرقم المذكور.", comment_text="أتمنى التواصل معي على الرقم 0501234567")
    assert ok2 is False and why2 == "unsupported_operational_claim"


def test_C_cooperation_request_does_not_invent_partnership_process():
    ok, why = validate_draft("سنتعاون معكم في هذا الشأن.", comment_text="أرغب بالتعاون معكم")
    assert ok is False and why == "unsupported_operational_claim"


def test_D_visit_request_does_not_invent_booking_or_contact_instructions():
    ok, why = validate_draft("يمكنك التواصل عبر مكتبنا لتحديد موعد الزيارة.", comment_text="هل أستطيع زيارتكم؟")
    assert ok is False and why == "unsupported_operational_claim"


def test_E_add_word_request_does_not_claim_word_will_be_added():
    ok1, why1 = validate_draft("تم إضافة الكلمة إلى المعجم.", comment_text="أضيفوا هذه الكلمة")
    assert ok1 is False and why1 == "unsupported_operational_claim"
    ok2, why2 = validate_draft("سنضيف هذه الكلمة إلى المعجم.", comment_text="أضيفوا هذه الكلمة")
    assert ok2 is False and why2 == "unsupported_operational_claim"


def test_F_correction_request_does_not_claim_definition_will_be_changed():
    ok1, why1 = validate_draft("تم تعديل التعريف كما طلبتم.", comment_text="عدّلوا التعريف")
    assert ok1 is False and why1 == "unsupported_operational_claim"
    ok2, why2 = validate_draft("سنعدل التعريف قريبًا.", comment_text="عدّلوا التعريف")
    assert ok2 is False and why2 == "unsupported_operational_claim"


def test_G_approval_request_does_not_claim_acceptance():
    ok1, why1 = validate_draft("تم اعتماد اقتراحكم.", comment_text="أرجو اعتماد الاقتراح")
    assert ok1 is False and why1 == "unsupported_operational_claim"
    ok2, why2 = validate_draft("سنعتمد اقتراحكم.", comment_text="أرجو اعتماد الاقتراح")
    assert ok2 is False and why2 == "unsupported_operational_claim"


def test_H_procedure_question_without_grounding_rejects_invented_channel():
    ok, why = validate_draft("يمكنكم التواصل عبر البريد الإلكتروني الرسمي لتقديم الطلب.", comment_text="كيف أقدم طلب؟")
    assert ok is False and why == "unsupported_operational_claim"


def test_safe_neutral_acknowledgement_never_flagged():
    """A neutral acknowledgement of the SAME requests, with no
    execution/commitment claim, must pass -- proves this is not a
    blanket ban on replying to actionable requests at all."""
    for comment, draft in (
        ("أرسلوا لي الملف", "نقدّر تواصلكم بخصوص طلب الملف."),
        ("تواصلوا معي", "نقدّر تواصلكم ورغبتكم بالتواصل معنا."),
        ("أرغب بالتعاون معكم", "نقدّر رغبتكم بالتعاون معنا."),
        ("أضيفوا هذه الكلمة", "نقدّر ملاحظتكم بخصوص الكلمة المقترحة."),
    ):
        ok, why = validate_draft(draft, comment_text=comment)
        assert ok is True and why is None, f"unexpected failure for {comment!r}: {why}"


def test_reason_code_reused_not_invented():
    """No new reason code was introduced -- the existing
    "unsupported_operational_claim" code (already refinement-eligible)
    is reused for the extended pattern set."""
    assert "unsupported_operational_claim" in REFINEMENT_ELIGIBLE_REASON_CODES


# --- 3. Official Knowledge interaction --------------------------------------


_PROCEDURE_TEXT = "يمكن تقديم طلب إضافة كلمة جديدة من خلال النموذج الرسمي المتاح على الموقع."


def _ok_with_procedure():
    return FakeOfficialKnowledgeRetriever(
        OfficialKnowledgeRetrievalResult(
            outcome=SUCCESS_WITH_RESULTS,
            results=(
                OfficialKnowledgeResult(
                    source_id="OK-PROC-1", source_title="إجراءات التقديم", source_type="procedure",
                    content=_PROCEDURE_TEXT, version="v1",
                    effective_from=dt.datetime(2020, 1, 1),
                    authoritative=True, retrieval_score=1.0,
                ),
            ),
        )
    )


def test_case1_grounded_procedure_may_be_stated_without_claiming_execution():
    """CASE 1: OK contains the relevant procedure -- the Agent may state
    it, but must not additionally claim the visitor's own request was
    executed."""
    draft = f"{_PROCEDURE_TEXT} يسعدنا اطلاعكم على هذا الإجراء."
    ok, why = validate_draft(draft, comment_text="كيف أقدم طلبًا لإضافة كلمة جديدة؟", trusted_grounding_text=_PROCEDURE_TEXT)
    assert ok is True and why is None


def test_case1_e2e_grounded_procedure_reaches_official_knowledge_block():
    client = FakeGeminiClient([_draft_json(f"{_PROCEDURE_TEXT} نتمنى أن يفيدكم هذا التوضيح.")])
    result = _run(
        _req("كيف أقدم طلبًا لإضافة كلمة جديدة؟", "INQUIRY", "GENERAL"),
        client,
        ok=_ok_with_procedure(),
    )
    assert result.metadata.official_grounding_available is True
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert "OFFICIAL KNOWLEDGE" in client.prompts[0]
    assert _PROCEDURE_TEXT in client.prompts[0]


def test_case2_no_grounding_procedure_must_not_be_invented():
    """CASE 2: no trusted Official Knowledge -- a draft that invents a
    procedure/channel must fail; a neutral acknowledgement must pass.
    (The validator is a bounded, evidence-anchored backstop -- it
    catches this via the pre-existing "يمكنك التواصل"/"التواصل عبر"
    channel-invention patterns; broader invented-procedure wording is
    the prompt Hard Rule's job, since it cannot be exhaustively
    enumerated as fixed patterns here.)"""
    ok_bad, why_bad = validate_draft(
        "يمكنكم التواصل عبر النموذج الرسمي المتاح على الموقع لتقديم الطلب.",
        comment_text="كيف أقدم طلبًا لإضافة كلمة جديدة؟", trusted_grounding_text=None,
    )
    assert ok_bad is False and why_bad == "unsupported_operational_claim"

    ok_safe, why_safe = validate_draft(
        "نقدّر تساؤلكم بخصوص كيفية تقديم طلب إضافة كلمة جديدة.",
        comment_text="كيف أقدم طلبًا لإضافة كلمة جديدة؟", trusted_grounding_text=None,
    )
    assert ok_safe is True and why_safe is None


def test_case2_e2e_no_grounding_never_invents_procedure():
    client = FakeGeminiClient([_draft_json("نقدّر تساؤلكم بخصوص كيفية تقديم طلب إضافة كلمة جديدة.")])
    result = _run(_req("كيف أقدم طلبًا لإضافة كلمة جديدة؟", "INQUIRY", "GENERAL"), client)
    assert result.metadata.official_grounding_available is False
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert "(none — do not invent facts)" in client.prompts[0]


# --- 4. Validation / refinement ---------------------------------------------


def test_bounded_refinement_recovers_from_unsupported_execution_claim():
    """One bounded refinement, exactly as the existing architecture
    already supports -- no new retry loop."""
    bad_draft = _draft_json("تم إرسال الملف إليكم.")
    safe_draft = _draft_json("نقدّر تواصلكم بخصوص طلب الملف.")
    client = FakeGeminiClient([bad_draft, safe_draft])
    result = _run(_req("أرسلوا لي الملف", "COMPLAINT", "GENERAL"), client)
    assert result.metadata.initial_validation_passed is False
    assert result.metadata.initial_failure_code == "unsupported_operational_claim"
    assert result.metadata.refinement_attempted is True
    assert result.metadata.logical_generation_count == 2
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert result.generation_result.suggested_response == "نقدّر تواصلكم بخصوص طلب الملف."
    assert client.call_count == 2


def test_no_more_than_one_refinement_attempt():
    """If the refined draft ALSO fails, the existing hard invariant
    (no third attempt, no loop) still applies -- final result is
    FAILED_REVIEW_ONLY, not a repeated retry."""
    bad_draft = _draft_json("تم إرسال الملف إليكم.")
    still_bad_draft = _draft_json("سنرسل لكم الملف قريبًا.")
    client = FakeGeminiClient([bad_draft, still_bad_draft])
    result = _run(_req("أرسلوا لي الملف", "COMPLAINT", "GENERAL"), client)
    assert result.metadata.refinement_attempted is True
    assert result.metadata.logical_generation_count == 2
    assert client.call_count == 2
    assert result.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert result.generation_result.reason_code == "unsupported_operational_claim"


def test_no_additional_gemini_call_for_a_clean_actionable_reply():
    client = FakeGeminiClient([_draft_json("نقدّر رغبتكم بالتعاون معنا.")])
    _run(_req("أرغب بالتعاون معكم", "SUGGESTION", "GENERAL"), client)
    assert client.call_count == 1


# --- 5. Non-operational replies must not regress ----------------------------


def test_K_non_actionable_lexical_suggestion_remains_natural():
    """A plain word-addition suggestion, with no operational EXECUTION
    claim, must still pass -- ordinary suggestion acknowledgements must
    not be flattened."""
    ok, why = validate_draft(
        "نقدّر اقتراحكم لإضافة كلمة «سلس» إلى المعجم.",
        comment_text="أقترح إضافة كلمة «سلس» إلى المعجم",
    )
    assert ok is True and why is None


def test_L_praise_control_remains_natural():
    ok, why = validate_draft("نقدّر كلماتكم الطيبة عن المعجم.", comment_text="شكرًا لكم على هذا العمل الرائع")
    assert ok is True and why is None


def test_explicit_visitor_intent_still_acknowledged_accurately():
    """The EXPLICIT request itself (not an invented one) may still be
    named in the draft -- only the EXECUTION claim is forbidden. This
    also proves the rg-v11 visitor-intent-attribution gate does not
    over-fire here: the comment itself states the request, so naming it
    is accurate, not invented."""
    ok, why = validate_draft("نشكركم على طلبكم إرسال نسخة من الملف.", comment_text="أرجو إرسال نسخة من الملف")
    assert ok is True and why is None


# --- 6. Visitor-intent-attribution and identifying-context policy unchanged


def test_ambiguous_intent_gate_unchanged():
    from rcs.response_intelligence.identifying_context_policy import (
        REASON_NO_CORROBORATING_REFERENCE,
        IdentifyingMetadata,
        evaluate_identifying_context,
    )

    assert "unsupported_visitor_intent_claim" in REFINEMENT_ELIGIBLE_REASON_CODES
    decision = evaluate_identifying_context(
        comment_text="أحسنتم", identifying_metadata=IdentifyingMetadata("إِطَاعَة", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_ambiguous_intent_bad_draft_still_rejected():
    ok, why = validate_draft(
        "نقدّر اقتراحكم لإضافة تركيب «أنشطة لا صفية».", comment_text="أنشطة لا صفية"
    )
    assert ok is False and why == "unsupported_visitor_intent_claim"


# --- 7. Structural: no new Gemini call / classifier / routing / DB ---------


def test_no_new_gate_function_was_added():
    """Reuses the existing _unsupported_claim_gate/_UNSUPPORTED_ACTION_
    PATTERNS mechanism -- no new gate function was introduced."""
    import inspect

    import rcs.response_generation.validate as validate_module

    gate_functions = {
        name for name, obj in inspect.getmembers(validate_module)
        if inspect.isfunction(obj) and name.endswith("_gate")
    }
    assert gate_functions == {
        "_content_shape_gate", "_internal_leak_gate", "_pii_gate", "_boilerplate_gate",
        "_evaluator_voice_gate", "_unsupported_claim_gate", "_visitor_intent_attribution_gate", "_pronoun_gate",
    }


def test_refinement_allowlist_is_exactly_the_documented_set():
    assert REFINEMENT_ELIGIBLE_REASON_CODES == frozenset(
        {
            "unsupported_operational_claim",
            "generic_boilerplate",
            "empty_output",
            "too_short",
            "excessive_length",
            "missing_arabic",
            "evaluator_voice_opening",
            "unsupported_visitor_intent_claim",
            "wrong_pronoun_construction",
            # reply-safety gates (claim_gates): repairable by one bounded rewrite that is re-validated by the same gates
            "unsupported_assertion",
            "unsupported_commitment",
            "unsupported_policy_claim",
            "unsupported_definition",
            "unsupported_correctness_claim",
            "unsupported_number",
            "unapproved_url",
        }
    )
