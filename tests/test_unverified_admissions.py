"""Unverified admissions of a dictionary error or a technical failure (live rg-v14.2 pilot #2 and #3). Narrow rule: apology / admission verb + an error or failure noun that no
hedge directly qualifies. Neutral acknowledgment and empathy for the visitor's experience stay valid."""
import json

import pytest

from rcs.response_generation.claim_gates import unverified_admission_of_error, _sentences
from rcs.response_generation.refinement import REFINEMENT_GUIDANCE, is_refinement_eligible
from rcs.response_generation.validate import collect_reply_findings, validate_draft


def flagged(text):
    return any(unverified_admission_of_error(s) for s in _sentences(text))


C2 = "تشكيل عنوان البطاقة غير صحيح في الصفحة"
C3 = "أرى أن التعريف المذكور للكلمة في هذه الصفحة غير دقيق وأرجو من القائمين على المعجم مراجعته وتصحيحه"
R2 = "نعتذر عن الخطأ الوارد في تشكيل عنوان البطاقة، ونقدّر تنبيهكم عليه."
R3 = "نرحب بملاحظتكم، ونعتذر عن عدم الدقة التي أشرتم إليها."


def test_exact_pilot_replies_2_and_3_are_rejected_as_unverified_admissions():
    assert validate_draft(R2, comment_text=C2) == (False, "unsupported_correctness_claim")
    assert validate_draft(R3, comment_text=C3) == (False, "unsupported_correctness_claim")


@pytest.mark.parametrize("text", [
    "نعتذر عن الخطأ في المدخل", "نأسف للخطأ الذي وقع في التعريف", "ونعتذر عن عدم الدقة في الشرح", "نعتذر منكم عن هذا الخطأ", "عذرا على الغلط في المثال",   # dictionary errors
    "نقر بوجود خطأ في المدخل", "نؤكد وجود خطأ في التعريف", "تبين لنا أن الخطأ واضح",                                                         # admissions
    "نعتذر عن الهفوة في الشرح", "نعتذر عن التقصير في المراجعة", "نعتذر عن القصور في الأمثلة",
])
def test_dictionary_error_admissions_are_flagged(text):
    assert flagged(text)


@pytest.mark.parametrize("text", [                                       # adjustment 1: a technical failure is NOT automatically safe to admit
    "نعتذر عن تعطل البحث", "نأسف للخلل الذي حدث في الموقع", "نعتذر عن البطء في الصفحة", "نعتذر عن توقف التطبيق", "ونعتذر عن عدم ظهور النتائج",
    "نقر بوجود خلل في الموقع", "نعتذر عن انقطاع الخدمة",
])
def test_technical_failure_admissions_are_flagged(text):
    assert flagged(text)


@pytest.mark.parametrize("text", [                                       # neutral acknowledgment of the reported experience - no cause confirmed
    "نعتذر عمّا واجهتموه من صعوبة في البحث", "نأسف لما واجهتموه", "نعتذر عن الإزعاج", "نعتذر عن عدم تمكنكم من فتح الصفحة", "نتفهم انزعاجكم من هذه التجربة",
    "نقدّر ملاحظتكم بشأن الخطأ في العنوان", "نشكركم على التنبيه إلى وجود خطأ", "نقدّر إبلاغكم عن بطء البحث", "نشكركم على ملاحظتكم حول تعطل الصفحة",
    "نقدّر تنبيهكم بشأن تشكيل عنوان البطاقة.", "شكرًا لكم، وجزاكم الله خيرًا",
])
def test_neutral_acknowledgments_and_empathy_pass(text):
    assert not flagged(text)


@pytest.mark.parametrize("text", [                                       # a hedge that directly qualifies THAT noun
    "نعتذر إن كان هناك خطأ", "نعتذر إن وقع خطأ في المدخل", "نعتذر عن أي خطأ قد يكون وقع", "نعتذر عن الخطأ المحتمل", "نأسف إذا كان في الشرح عدم دقة",
    "نعتذر لو كان هناك تعطل في البحث", "إذا تبين وجود خطأ نعتذر",
])
def test_conditional_apologies_with_a_hedge_on_the_claim_pass(text):
    assert not flagged(text)


@pytest.mark.parametrize("text", [                                       # adjustment 2: a hedge word elsewhere in the sentence exempts nothing
    "قد نعتذر عن الخطأ", "نعتذر عن الخطأ إذا أردتم", "نعتذر عن الخطأ في المدخل وقد نراجعه", "ربما نعتذر عن عدم الدقة التي أشرتم إليها", "لو سمحتم نعتذر عن تعطل البحث",
    "إذا أمكن نعتذر عن الخطأ الوارد في العنوان",
])
def test_a_hedge_that_does_not_qualify_the_claim_does_not_exempt_the_sentence(text):
    assert flagged(text)


def test_with_approved_grounding_the_rule_does_not_apply():
    assert validate_draft(R2, comment_text=C2, trusted_grounding_text="تم التحقق من وجود خطأ في تشكيل عنوان البطاقة وتصحيحه في المدخل.")[1] != "unsupported_correctness_claim"


def test_repair_is_allowed_and_carries_the_new_guidance():
    assert is_refinement_eligible("FAILED_REVIEW_ONLY", "unsupported_correctness_claim")
    g = REFINEMENT_GUIDANCE["unsupported_correctness_claim"]
    assert "do not apologise for or admit an error" in g and "inconvenience" in g


def test_main_prompt_complaint_hint_no_longer_invites_admissions():
    from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS

    assert "apologize when appropriate" not in SYSTEM_INSTRUCTIONS       # narrowed in rg-v14.3; the validator remains the enforcing layer
    assert "never admit a fault" in SYSTEM_INSTRUCTIONS


def test_reviewer_findings_classify_the_admission_as_attestable_not_blocking():
    f = collect_reply_findings(R2, comment_text=C2)
    assert "unsupported_correctness_claim" in f.attestable and not f.blocking


def test_a_permitted_reviewer_who_verified_the_error_can_still_send_it_with_attestation(tmp_path):
    from rcs.review.service import FactAttestation
    from rcs.review.workflow import AttestationRequired
    from tests.test_reviewer_attestation import REVIEWER, _engine, _seed

    svc = _seed(_engine(tmp_path), status_text=C2)
    with pytest.raises(AttestationRequired) as exc:
        svc.edit_and_send("c1", REVIEWER, R2)
    assert exc.value.finding_codes == ("unsupported_correctness_claim",)
    detail = svc.edit_and_send("c1", REVIEWER, R2, attestation=FactAttestation(True, ("unsupported_correctness_claim",)))
    assert detail.final_response == R2


def test_saved_live_pilot_drafts_are_revalidated_offline():
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data/evaluation/reply_quality/live_pilot_results.json"
    if not p.exists():
        pytest.skip("local pilot results not available")
    d = json.loads(p.read_text())
    rejected_now, passing_now = [], []
    for r in d["results"]:
        for k, c in enumerate(r["gemini_calls"], 1):
            ok, why = validate_draft(c["draft_returned"], comment_text=r["comment_sanitized"], trusted_grounding_text=None)
            (passing_now if ok else rejected_now).append((r["n"], k, why))
    assert (2, 1, "unsupported_correctness_claim") in rejected_now and (3, 2, "unsupported_correctness_claim") in rejected_now
    assert (7, 1, "wrong_pronoun_construction") in rejected_now                                 # «على مشاركتنا»: caught by the rg-v14.4 pronoun gate
    for n, k in ((1, 1), (4, 1), (5, 1), (6, 1), (9, 1), (10, 1), (8, 2), (12, 2)):           # previously accepted replies and the two recovered by the suggestion fix
        assert (n, k) in [(a, b) for a, b, _ in passing_now], (n, k)
