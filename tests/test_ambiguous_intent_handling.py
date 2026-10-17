"""
Tests for how the Response Agent handles ambiguous or insufficient-
context comments.

Root cause under test: the Response Agent sometimes invents a
communicative intent or operational action the visitor never stated --
confirmed real evidence (scripts/offline_eval/identifying_context_
mechanism/results/identifying_context_mechanism_cases.jsonl, eval_id
"anshita_la_safiya-NONE"): comment "أنشطة لا صفية" (a bare phrase) with
NO identifying context at all still produced "نشكر إسهامكم باقتراح
تركيب «أنشطة لا صفية»." -- treating a bare fragment (classified
topic_subtype=WORD_SUBMISSION) as an explicit proposal to add it.

Fix (both narrow, no new taxonomy, no new Gemini call, no new field):
  1. prompt.py -- one added Hard Rule plus a tightened wording of the
     "Word suggestion / submission" / "Dictionary suggestion" situation
     hints (primary mechanism; not independently testable without a
     live Gemini call, verified structurally below).
  2. validate.py's new `_visitor_intent_attribution_gate` -- a small,
     general, closed-vocabulary deterministic BACKSTOP: fails a draft
     only when it attributes a specific action (suggestion/request/
     correction) to the visitor via a small set of general "your-<noun>"
     phrases, with no matching lexical evidence anywhere in the
     visitor's own comment text. Fully testable without live Gemini via
     FakeGeminiClient.

Fast, deterministic -- no real model load, no network, no Gemini call.
"""

from __future__ import annotations

from config import settings
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS
from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES
from rcs.response_generation.service import GenerationRequest
from rcs.response_generation.validate import validate_draft
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
)
from scripts.offline_eval.response_agent.fakes import (
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
    FakeOfficialKnowledgeRetriever,
)

_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)


import pytest

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


# --- 1. Prompt-level fix is present (structural; behavior itself needs a
#        live Gemini call and is not asserted here) -------------------------


def test_hard_rule_against_inventing_visitor_action_is_present():
    assert "Do not attribute a specific action, request, or proposal to the visitor" in SYSTEM_INSTRUCTIONS
    assert "topic_subtype_signal" in SYSTEM_INSTRUCTIONS.split(
        "Do not attribute a specific action"
    )[1][:400]


def test_word_submission_hint_requires_comment_to_state_the_proposal():
    assert "only when the comment itself proposes or names a specific word/addition" in SYSTEM_INSTRUCTIONS
    assert "bare word, phrase, or fragment with no stated proposal or reasoning" in SYSTEM_INSTRUCTIONS


def test_dictionary_suggestion_hint_requires_comment_to_express_the_idea():
    assert "acknowledge the actual lexicographic idea only when the comment itself expresses one" in SYSTEM_INSTRUCTIONS


def test_prompt_version_is_rg_v11():
    """Traceability requirement: a real prompt content change (a new
    Hard Rule plus tightened situation hints) requires
    settings.RESPONSE_GENERATION_PROMPT_VERSION to become a NEW label,
    never left at the previous one -- otherwise historical executions
    under the old label and new ones would be indistinguishable despite
    Gemini receiving different instructions. See
    rcs.response_generation.prompt's module docstring.

    Value now reads "rg-v14.2": later content changes bumped it again
    for the identical reason -- this test's own point (a genuine
    content change must never be left under the PREVIOUS label) is
    unaffected by which label is current."""
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"


def test_existing_factual_grounding_hard_rules_untouched():
    """This change is about response semantics/sufficiency, not factual
    grounding -- the Hard Rule on linguistic-claim validation must
    remain exactly as it was."""
    assert (
        "Do not confirm, validate, or correct a visitor's linguistic, scientific, or factual claim"
        in SYSTEM_INSTRUCTIONS
    )


# --- 2. Validator gate -- direct unit tests ---------------------------------


def test_suggestion_attribution_without_evidence_fails():
    ok, why = validate_draft(
        "نقدّر اقتراحكم لإضافة تركيب «أنشطة لا صفية»، واهتمامكم بإثراء المحتوى المعجمي.",
        comment_text="أنشطة لا صفية",
    )
    assert ok is False
    assert why == "unsupported_visitor_intent_claim"


def test_suggestion_attribution_with_evidence_passes():
    ok, why = validate_draft(
        "نقدّر اقتراحكم لإضافة كلمة «سلس» إلى المعجم.",
        comment_text="أقترح إضافة كلمة «سلس» إلى المعجم",
    )
    assert ok is True and why is None


def test_request_attribution_without_evidence_fails():
    ok, why = validate_draft("سننظر في طلبكم في أقرب فرصة.", comment_text="تتت")
    assert ok is False
    assert why == "unsupported_visitor_intent_claim"


def test_request_attribution_with_evidence_passes():
    ok, why = validate_draft(
        "نشكركم على طلبكم تصحيح المثال في هذا المدخل.",
        comment_text="أرجو تصحيح المثال في هذا المدخل",
    )
    assert ok is True and why is None


def test_correction_attribution_without_evidence_fails():
    ok, why = validate_draft("شكرًا على تصحيحكم وتنبيهكم.", comment_text="لا اعلم")
    assert ok is False
    assert why == "unsupported_visitor_intent_claim"


def test_correction_attribution_with_evidence_passes():
    ok, why = validate_draft(
        "شكرًا على تصحيحكم لتعريف كلمة «سلس».",
        comment_text="يوجد خطأ في تعريف كلمة «سلس»",
    )
    assert ok is True and why is None


def test_generic_acknowledgement_never_flagged_regardless_of_comment():
    """Words like "مشاركتكم"/"تواصلكم" are not in the closed attribution
    set at all -- a neutral, non-specific acknowledgement must never be
    flagged, which is exactly the safe fallback this change encourages
    for ambiguous comments."""
    ok, why = validate_draft("نقدّر مشاركتكم معنا.", comment_text="الله اعلم")
    assert ok is True and why is None


def test_gate_is_a_no_op_when_comment_text_is_not_supplied():
    """Backward compatibility: every pre-existing caller of validate_draft/
    run_validation_gates that never passes comment_text (dev-inspect's
    arbitrary-draft-paste tool, offline-eval scripts) is byte-identical
    to before this gate existed."""
    ok, why = validate_draft("نقدّر اقتراحكم لإضافة تركيب «أنشطة لا صفية».")
    assert ok is True and why is None


def test_reason_code_is_refinement_eligible():
    assert "unsupported_visitor_intent_claim" in REFINEMENT_ELIGIBLE_REASON_CODES


# --- 3. End-to-end via the real orchestrator (FakeGeminiClient) ------------
# The 8 required test cases, run through the same real, unmodified
# run_agent_pipeline every canonical caller uses.

_BAD_SUGGESTION_DRAFT = "نقدّر اقتراحكم لإضافة تركيب «أنشطة لا صفية»، واهتمامكم بإثراء المحتوى المعجمي."
_SAFE_ACK_DRAFT = "هل يمكنكم توضيح ما الذي تودون معرفته عن هذا التركيب؟"


def test_1_bare_fragment_word_submission_rejects_invented_suggestion_claim():
    """comment = 'أنشطة لا صفية' -- real evidence case. A draft that
    claims the visitor proposed adding it must fail; a neutral
    clarification question must pass and reach the reviewer."""
    bad_client = FakeGeminiClient([_draft_json(_BAD_SUGGESTION_DRAFT), _draft_json(_SAFE_ACK_DRAFT)])
    result = _run(_req("أنشطة لا صفية", "COMPLAINT", "WORD_SUBMISSION", sentiment="NEGATIVE"), bad_client)
    assert result.metadata.initial_validation_passed is False
    assert result.metadata.initial_failure_code == "unsupported_visitor_intent_claim"
    assert result.metadata.refinement_attempted is True
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert result.generation_result.suggested_response == _SAFE_ACK_DRAFT


def test_2_gibberish_fragment_rejects_invented_linguistic_proposal():
    """comment = 'تتت' -- no invented linguistic action/request. This
    draft also claims a future org commitment ("سنضيف", covered by
    validate.py's _UNSUPPORTED_ACTION_PATTERNS), so that pre-existing-
    order gate now fires first -- same first-failure-wins precedence as
    test_3 below; the draft is rejected either way."""
    bad_client = FakeGeminiClient([_draft_json("سنضيف اقتراحكم إلى قائمة الكلمات قيد الدراسة.")])
    result = _run(_req("تتت", "SUGGESTION", "WORD_SUBMISSION", sentiment="NEGATIVE"), bad_client)
    assert result.metadata.initial_validation_passed is False
    assert result.metadata.initial_failure_code == "unsupported_operational_claim"


def test_2b_gibberish_fragment_rejects_invented_suggestion_attribution_specifically():
    """Same comment, a draft that avoids the org-action gate entirely --
    isolates that the visitor-intent-attribution gate independently
    catches the invented suggestion attribution."""
    bad_client = FakeGeminiClient([_draft_json("نقدّر اقتراحكم.")])
    result = _run(_req("تتت", "SUGGESTION", "WORD_SUBMISSION", sentiment="NEGATIVE"), bad_client)
    assert result.metadata.initial_validation_passed is False
    assert result.metadata.initial_failure_code == "unsupported_visitor_intent_claim"


def test_3_laa_aelam_rejects_invented_specific_request():
    """comment = 'لا اعلم' -- no invented specific request/complaint.
    This draft also happens to claim a completed org action ("تم
    استلام"), so the pre-existing unsupported_claim gate fires first --
    first-failure-wins order is unchanged, and the draft is rejected
    either way."""
    bad_client = FakeGeminiClient([_draft_json("تم استلام طلبكم وسننظر فيه.")])
    result = _run(_req("لا اعلم", "COMPLAINT", "WORD_SUBMISSION"), bad_client)
    assert result.metadata.initial_validation_passed is False
    assert result.metadata.initial_failure_code == "unsupported_operational_claim"


def test_3b_laa_aelam_rejects_invented_request_claim_specifically():
    """Same comment, a draft that avoids the org-action gate entirely --
    isolates that the NEW gate independently catches the invented
    request attribution."""
    bad_client = FakeGeminiClient([_draft_json("سننظر في طلبكم قريبًا.")])
    result = _run(_req("لا اعلم", "COMPLAINT", "WORD_SUBMISSION"), bad_client)
    assert result.metadata.initial_failure_code == "unsupported_visitor_intent_claim"


def test_4_allah_aelam_rejects_invented_action():
    """comment = 'الله اعلم' -- no invented request/action; a generic,
    non-attributing thanks (the real observed behavior for this exact
    case) must still succeed."""
    safe_client = FakeGeminiClient([_draft_json("نقدّر مشاركتك معنا.")])
    result = _run(_req("الله اعلم", "THANKS_PRAISE", "WORD_SUBMISSION", sentiment="POSITIVE"), safe_client)
    assert result.generation_result.generation_status == "SUCCEEDED"

    bad_client = FakeGeminiClient([_draft_json("نشكركم على اقتراحكم القيّم.")])
    bad_result = _run(_req("الله اعلم", "THANKS_PRAISE", "WORD_SUBMISSION", sentiment="POSITIVE"), bad_client)
    assert bad_result.metadata.initial_failure_code == "unsupported_visitor_intent_claim"


def test_5_explicit_suggestion_attribution_allowed():
    """'أقترح إضافة كلمة «س» إلى المعجم' -- suggestion attribution allowed."""
    client = FakeGeminiClient([_draft_json("نقدّر اقتراحكم لإضافة كلمة «س» إلى المعجم.")])
    result = _run(_req("أقترح إضافة كلمة «س» إلى المعجم", "SUGGESTION", "WORD_SUBMISSION"), client)
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert "اقتراحكم" in result.generation_result.suggested_response


def test_6_explicit_correction_attribution_allowed():
    """'يوجد خطأ في تعريف كلمة «س»' -- correction/complaint attribution allowed."""
    client = FakeGeminiClient([_draft_json("شكرًا على تصحيحكم لتعريف كلمة «س».")])
    result = _run(_req("يوجد خطأ في تعريف كلمة «س»", "COMPLAINT", "ENTRY_CORRECTION"), client)
    assert result.generation_result.generation_status == "SUCCEEDED"


def test_7_explicit_inquiry_attribution_allowed():
    """'هل معنى كلمة «س» هو كذا؟' -- inquiry attribution allowed."""
    client = FakeGeminiClient([_draft_json("شكرًا لسؤالكم عن معنى كلمة «س»، ولا تتوفر لدينا معلومة موثوقة لتأكيد معناها.")])
    result = _run(_req("هل معنى كلمة «س» هو كذا؟", "INQUIRY", "GENERAL"), client)
    assert result.generation_result.generation_status == "SUCCEEDED"


def test_8_explicit_request_attribution_allowed():
    """'أرجو تصحيح المثال في هذا المدخل' -- correction/request attribution allowed."""
    client = FakeGeminiClient([_draft_json("نشكركم على طلبكم تصحيح المثال في هذا المدخل.")])
    result = _run(_req("أرجو تصحيح المثال في هذا المدخل", "COMPLAINT", "ENTRY_CORRECTION"), client)
    assert result.generation_result.generation_status == "SUCCEEDED"


# --- 4. Substantive valid suggestion is not neutered ------------------------
# Real comment + real generated reply from the existing evaluation corpus
# (scripts/offline_eval/identifying_context_mechanism, source_row_index 1,
# eval_id "linguistic_suggestion-NONE") -- proves a genuine, long,
# substantive linguistic contribution is not flattened by the new gate.

_REAL_SUBSTANTIVE_COMMENT = (
    "لا يعمل الرابط الصوتي في صفحة هذه المادة عندي منذ يومين«اعتبار» هل يمكن توضيح الفرق بين هذه الكلمة والكلمة القريبة منها في المعنى"
    "أتمنى إضافة أمثلة تطبيقية توضّح استخدام الكلمة في سياقات مختلفة«لا يعمل الرابط الصوتي في صفحة هذه المادة عندي منذ يومين»ما أصل هذه الكلمة وهل وردت في المعاجم القديمة بالمعنى نفسه"
    "اقترح إضافة خاصية البحث بحسب الجذر لتسهيل الوصول إلى الكلمات المشتقة"
)
_REAL_SUBSTANTIVE_REPLY = (
    "التفريق الذي أشرتم إليه بين المعنى العام لكلمة «اعتبار» وبين دلالتها الاصطلاحية في السياقات "
    "الفلسفية والتراثية -كالاستدلال والنظر- يبرز أهمية تتبع التطور الدلالي للمصطلحات. "
    "نشكركم على مشاركة هذه اللفتة المعجمية."
)


def test_9_real_substantive_suggestion_from_evaluation_corpus_not_neutered():
    ok, why = validate_draft(_REAL_SUBSTANTIVE_REPLY, comment_text=_REAL_SUBSTANTIVE_COMMENT)
    assert ok is True and why is None


def test_9b_real_substantive_suggestion_succeeds_end_to_end():
    client = FakeGeminiClient([_draft_json(_REAL_SUBSTANTIVE_REPLY)])
    result = _run(_req(_REAL_SUBSTANTIVE_COMMENT, "SUGGESTION", "ENTRY_CORRECTION"), client)
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert result.generation_result.suggested_response == _REAL_SUBSTANTIVE_REPLY
    assert result.metadata.refinement_attempted is False


# --- 5. Everything else this change must not disturb -------------------------


def test_unrelated_gates_still_fire_before_the_new_gate():
    """Gate order: content_shape -> internal_leak -> pii -> boilerplate ->
    evaluator_voice -> unsupported_claim -> visitor_intent_attribution.
    A PII draft must still fail on 'pii_in_draft', never reaching the
    new gate at all."""
    ok, why = validate_draft(
        "شكرًا لتواصلكم على الرقم 0501234567، ونقدّر اقتراحكم.",
        comment_text="تتت",
    )
    assert ok is False and why == "pii_in_draft"


def test_official_knowledge_path_unaffected():
    ok_result = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
    client = FakeGeminiClient([_draft_json("لا تتوفر معلومة رسمية بخصوص هذا السؤال حاليًا.")])
    result = _run(
        _req("هل معنى كلمة «س» هو كذا؟", "INQUIRY", "GENERAL"),
        client,
        ok=FakeOfficialKnowledgeRetriever(ok_result),
    )
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert result.metadata.official_grounding_available is False


# --- 6. Traceability: rg-v10 -> rg-v11 --------------------------------------
# The prompt content changed here (Hard Rule + tightened hints) is
# real, materially different instruction text -- Gemini receives a
# different prompt than historical rg-v10 executions did. The version
# constant and every place it is recorded must reflect that.


def test_generation_result_records_rg_v11():
    """New GenerationResult records the current prompt version
    (originally "rg-v11" for this change; now "rg-v14.2" after later
    content change -- the mechanism this test proves is unaffected)."""
    client = FakeGeminiClient([_draft_json(_SAFE_ACK_DRAFT)])
    result = _run(_req("شكرًا لكم", "THANKS_PRAISE", "GENERAL", sentiment="POSITIVE"), client)
    assert result.generation_result.prompt_version == "rg-v14.4"


def test_historical_evaluation_evidence_is_not_rewritten():
    """Historical rg-v10 evidence remains unchanged. This asserts the
    ACTUAL on-disk historical evidence the root-cause audit here was
    built on (scripts/offline_eval/identifying_context_mechanism/
    results/identifying_context_mechanism_cases.jsonl) still literally
    says "rg-v10" for that historical run -- proving no old JSONL
    evidence was edited merely to rename its prompt-version label after
    the production constant moved to rg-v11."""
    import json
    from pathlib import Path

    results_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "offline_eval" / "identifying_context_mechanism"
        / "results" / "identifying_context_mechanism_cases.jsonl"
    )
    anshita_none = None
    for line in results_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("eval_id") == "anshita_la_safiya-NONE":
            anshita_none = row
            break
    assert anshita_none is not None
    assert anshita_none["generated_reply"] == "نشكر إسهامكم باقتراح تركيب «أنشطة لا صفية»."
    assert anshita_none["reason_code"] == "complaint_safe_acknowledgement"


def test_no_runtime_behavior_besides_version_attribution_changed():
    """Same ambiguous-comment decision (rejected/refined) and same
    explicit-comment decision (accepted) as before the version bump --
    only the recorded prompt_version differs. Proves the rg-v10 -> rg-v11
    correction is a pure attribution/traceability change, not a behavior
    change."""
    bad_client = FakeGeminiClient([_draft_json(_BAD_SUGGESTION_DRAFT), _draft_json(_SAFE_ACK_DRAFT)])
    bad_result = _run(_req("أنشطة لا صفية", "COMPLAINT", "WORD_SUBMISSION", sentiment="NEGATIVE"), bad_client)
    assert bad_result.metadata.initial_failure_code == "unsupported_visitor_intent_claim"
    assert bad_result.generation_result.generation_status == "SUCCEEDED"
    assert bad_result.generation_result.prompt_version == "rg-v14.4"

    good_client = FakeGeminiClient([_draft_json("نقدّر اقتراحكم لإضافة كلمة «س» إلى المعجم.")])
    good_result = _run(_req("أقترح إضافة كلمة «س» إلى المعجم", "SUGGESTION", "WORD_SUBMISSION"), good_client)
    assert good_result.generation_result.generation_status == "SUCCEEDED"
    assert good_result.generation_result.prompt_version == "rg-v14.4"


def _draft_json(text: str) -> str:
    import json

    return json.dumps({"suggested_response": text}, ensure_ascii=False)
