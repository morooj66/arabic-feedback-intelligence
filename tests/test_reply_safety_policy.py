"""Reply-safety and classification-safety policy tests. Runs on the PRODUCTION defaults (no legacy fixture), no network, no live model.

Covers: Arabic promise variants, fabricated definitions / policies / URLs / verdicts, reviewer-edited replies (blocked before any send, never
rewritten, re-validation), safe replies that must pass, fail-closed behaviour when grounding is unavailable, future readiness for an approved
knowledge source, and non-suppression of comments on a single uncertain SPAM / OFF_TOPIC label."""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

import pytest

from config import settings
from rcs.response_generation.eligibility import (
    EligibilityInput,
    GENERATE_DRAFT,
    NO_RESPONSE_RECOMMENDED,
    REVIEW_ONLY,
    asks_for_facts,
    determine_response_eligibility,
)
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest
from rcs.response_generation.validate import validate_draft, validate_reviewer_edit
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
)
from rcs.review.auth import create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.identity import ReviewerIdentity
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.review.workflow import DraftValidationError
from scripts.offline_eval.response_agent.fakes import (
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
    FakeOfficialKnowledgeRetriever,
)
from scripts.offline_eval.system_audit.reply_safety_battery import SAFE, UNSAFE

REVIEWER = ReviewerIdentity("rev-1", "مراجع تجريبي")
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)


# ------------------------------------------------------------------------------------------------ 1. validator battery
@pytest.mark.parametrize("category,draft,comment", UNSAFE, ids=[f"{c}-{i}" for i, (c, _, _) in enumerate(UNSAFE)])
def test_unsafe_draft_is_rejected_when_no_grounding_exists(category, draft, comment):
    ok, why = validate_draft(draft, comment_text=comment, trusted_grounding_text=None)
    assert ok is False, f"{category} draft was accepted: {draft}"
    assert why


@pytest.mark.parametrize("category,draft,comment", UNSAFE, ids=[f"{c}-{i}" for i, (c, _, _) in enumerate(UNSAFE)])
def test_unsafe_reviewer_edit_is_rejected_too(category, draft, comment):
    ok, why = validate_reviewer_edit(draft, comment_text=comment, trusted_grounding_text=None)
    assert ok is False, f"{category} reviewer edit was accepted: {draft}"
    assert why


@pytest.mark.parametrize("category,draft,comment", SAFE, ids=[f"{c}-{i}" for i, (c, _, _) in enumerate(SAFE)])
def test_safe_reply_passes_generation_and_reviewer_profiles(category, draft, comment):
    assert validate_draft(draft, comment_text=comment) == (True, None)
    assert validate_reviewer_edit(draft, comment_text=comment) == (True, None)


def test_battery_is_larger_than_the_original_fourteen_cases():
    assert len(UNSAFE) >= 40 and len(SAFE) >= 10


def test_reason_codes_are_specific():
    cases = {
        "سنقوم بمراجعة الملاحظة وإضافتها.": "unsupported_commitment",
        "يقبل المعجم الإضافات من الزوار بعد أسبوعين من تقديمها.": "unsupported_policy_claim",
        "يمكنكم مراجعة https://example.org/help للمزيد.": "unapproved_url",
        "التعريف الوارد في المعجم صحيح ولا يوجد فيه خطأ.": "unsupported_correctness_claim",
        "تعني كلمة «قارورة» الوعاء الزجاجي الصغير الذي يحفظ العطور.": "unsupported_definition",
        "بلغ عدد الكلمات المضافة هذا العام 1200 كلمة.": "unsupported_number",
    }
    for draft, expected in cases.items():
        assert validate_draft(draft, comment_text="تعليق") == (False, expected), draft


def test_diacritics_and_spelling_variants_do_not_evade_the_gates():
    assert validate_draft("سَنَقُومُ بِتَصْحِيحِ الْمَدْخَلِ.", comment_text="x")[0] is False
    assert validate_draft("سيُصَحَّحُ التَّعْرِيفُ فِي التَّحْدِيثِ الْقَادِمِ.", comment_text="x")[0] is False
    assert validate_draft("سوف نُراجع الطلب", comment_text="x")[0] is False        # missing hamza / punctuation variants


# ------------------------------------------------------------------------------- 2. grounding fail-closed / future-ready
_DEFINITION = "السوار حلية تلبس في المعصم وتجمع على أساور."


def test_definition_without_grounding_is_rejected_and_with_matching_grounding_is_accepted():
    draft = "معنى كلمة «سوار» في المعجم هو حلية تلبس في المعصم."
    assert validate_draft(draft, comment_text="ما معنى سوار", trusted_grounding_text=None) == (False, "unsupported_definition")
    assert validate_draft(draft, comment_text="ما معنى سوار", trusted_grounding_text="(none — do not invent facts)")[0] is False
    ok, why = validate_draft(draft, comment_text="ما معنى سوار", trusted_grounding_text=_DEFINITION)
    assert (ok, why) == (True, None)


def test_definition_that_contradicts_the_grounding_is_rejected_even_when_grounding_exists():
    draft = "معنى كلمة «سوار» في المعجم هو وعاء زجاجي صغير لحفظ العطور."
    assert validate_draft(draft, comment_text="ما معنى سوار", trusted_grounding_text=_DEFINITION) == (False, "unsupported_definition")


def test_policy_stated_in_grounding_may_be_repeated_but_not_extended():
    grounding = "تخضع الاقتراحات لمراجعة لجنة متخصصة."
    assert validate_draft("تخضع الاقتراحات لمراجعة لجنة متخصصة.", comment_text="x", trusted_grounding_text=grounding)[0] is True
    assert validate_draft("تخضع الاقتراحات لمراجعة لجنة متخصصة خلال أسبوعين.", comment_text="x", trusted_grounding_text=grounding)[0] is False


def test_unapproved_url_is_allowed_only_if_it_appears_in_the_grounding():
    draft = "يمكنكم مراجعة https://example.org/help للمزيد."
    assert validate_draft(draft, comment_text="x")[0] is False
    assert validate_draft(draft, comment_text="x", trusted_grounding_text="الدليل متاح عبر https://example.org/help")[0] is True


def _ok_with(text: str) -> FakeOfficialKnowledgeRetriever:
    return FakeOfficialKnowledgeRetriever(
        OfficialKnowledgeRetrievalResult(
            outcome=SUCCESS_WITH_RESULTS,
            results=(
                OfficialKnowledgeResult(
                    source_id="OK-1", source_title="مصدر معتمد", source_type="definition", content=text, version="v1",
                    effective_from=dt.datetime(2020, 1, 1), authoritative=True, retrieval_score=1.0,
                ),
            ),
        )
    )


def _empty_ok() -> FakeOfficialKnowledgeRetriever:
    return FakeOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY))


def _run(request, client, ok):
    return run_agent_pipeline(
        request, client=client, official_knowledge_retriever=ok, approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    )


def _draft_json(text: str) -> str:
    return json.dumps({"suggested_response": text}, ensure_ascii=False)


def _req(comment, intent, topic="GENERAL", comment_id="c"):
    return GenerationRequest(comment_id=comment_id, comment_text=comment, primary_intent=intent, topic_subtype=topic, sentiment="NEUTRAL", confidence=0.4)


def test_inquiry_without_official_knowledge_goes_to_a_human_and_makes_zero_model_calls():
    client = FakeGeminiClient([_draft_json("معنى كلمة «سوار» هو حلية تلبس في المعصم.")])
    result = _run(_req("ما معنى كلمة سوار؟", "INQUIRY"), client, _empty_ok()).generation_result
    assert result.routing_decision == REVIEW_ONLY and result.generation_status == "SKIPPED"
    assert result.suggested_response is None
    assert result.reason_code == "factual_question_without_grounding"
    assert client.call_count == 0


def test_question_mislabelled_as_suggestion_is_still_not_answered_without_grounding():
    """The Intent model's INQUIRY recall is low, so the gate keys on the text, not only on the label."""
    client = FakeGeminiClient([_draft_json("معنى الكلمة هو كذا وكذا.")])
    result = _run(_req("هل يوجد في المعجم كلمة مرادفة لكلمة سوار", "SUGGESTION"), client, _empty_ok()).generation_result
    assert result.routing_decision == REVIEW_ONLY and client.call_count == 0


def test_future_approved_knowledge_source_re_enables_a_grounded_answer():
    client = FakeGeminiClient([_draft_json("معنى كلمة «سوار» في المعجم هو حلية تلبس في المعصم.")])
    result = _run(_req("ما معنى كلمة سوار؟", "INQUIRY"), client, _ok_with(_DEFINITION)).generation_result
    assert result.routing_decision == GENERATE_DRAFT and result.generation_status == "SUCCEEDED"
    assert client.call_count == 1


def test_safe_drafting_is_preserved_for_comments_that_need_no_unavailable_facts():
    for intent, topic, comment, reply in (
        ("THANKS_PRAISE", "GENERAL", "جزاكم الله خيرا على هذا الجهد", "جزاكم الله خيرًا على دعائكم الطيب."),
        ("SUGGESTION", "DICTIONARY_SUGGESTION", "أتمنى إضافة خاصية البحث بالجذر", "نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر."),
        ("COMPLAINT", "TECHNICAL_ISSUE", "الموقع ما يفتح عندي", "نأسف لما واجهتموه، ونرجو تزويدنا بتفاصيل أكثر عن الصفحة التي تعذر فتحها."),
    ):
        client = FakeGeminiClient([_draft_json(reply)])
        result = _run(_req(comment, intent, topic), client, _empty_ok()).generation_result
        assert result.generation_status == "SUCCEEDED", (intent, result.reason_code, result.error)
        assert client.call_count == 1


def test_a_model_draft_that_invents_a_promise_is_failed_closed_with_no_draft():
    client = FakeGeminiClient([_draft_json("شكرًا لكم، سنقوم بمراجعة اقتراحكم وإضافته قريبًا.")] * 2)
    result = _run(_req("أتمنى إضافة خاصية البحث بالجذر", "SUGGESTION", "DICTIONARY_SUGGESTION"), client, _empty_ok()).generation_result
    assert result.generation_status == "FAILED_REVIEW_ONLY"
    assert result.suggested_response is None
    assert result.reason_code in ("unsupported_commitment", "unsupported_operational_claim")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ما معنى كلمة سوار", True),
        ("كيف أضيف كلمة جديدة", True),
        ("هل يمكن التعاون في النشر", True),
        ("متى تضاف الكلمات الجديدة", True),
        ("لماذا حذفت الكلمة", True),
        ("وش معنى هالكلمة؟", True),
        ("جزاكم الله خيرا", False),
        ("أقترح إضافة خاصية البحث بالجذر", False),
        ("الموقع لا يفتح عندي", False),
    ],
)
def test_asks_for_facts_detection(text, expected):
    assert asks_for_facts(text) is expected


# ------------------------------------------------------------------------ 3. suppression on one uncertain model label
def _elig(**kw):
    base = dict(comment_text="نص كاف للمعالجة والفهم", primary_intent="SUGGESTION", topic_subtype="GENERAL")
    base.update(kw)
    return determine_response_eligibility(EligibilityInput(**base))


def test_spam_label_alone_goes_to_human_review_not_silent_suppression():
    r = _elig(primary_intent="SPAM")
    assert (r.routing_decision, r.reason_code, r.rule_id) == (REVIEW_ONLY, "uncertain_exclusion_requires_review", "uncertain_exclusion_review")


def test_off_topic_label_alone_goes_to_human_review():
    r = _elig(primary_intent="THANKS_PRAISE", topic_subtype="OFF_TOPIC_PERSONAL")
    assert r.routing_decision == REVIEW_ONLY and r.reason_code == "uncertain_exclusion_requires_review"


def test_operational_exclude_derived_from_one_label_does_not_suppress():
    r = _elig(primary_intent="SUGGESTION", classification_routing_decision="EXCLUDE")
    assert r.routing_decision == REVIEW_ONLY


def test_two_independent_models_agreeing_keeps_the_recommendation_which_a_reviewer_can_override():
    r = _elig(primary_intent="SPAM", topic_subtype="OFF_TOPIC_PERSONAL")
    assert (r.routing_decision, r.reason_code) == (NO_RESPONSE_RECOMMENDED, "spam")


def test_deterministic_corroboration_keeps_the_recommendation():
    r = _elig(primary_intent="SPAM", exclusion_corroborated=True)
    assert r.routing_decision == NO_RESPONSE_RECOMMENDED


def test_blank_or_meaningless_text_is_still_not_answered():
    assert _elig(comment_text="  ").routing_decision == NO_RESPONSE_RECOMMENDED
    assert _elig(comment_text="و").routing_decision == NO_RESPONSE_RECOMMENDED


def test_policy_flags_can_be_switched_off_to_restore_the_previous_behaviour(monkeypatch):
    monkeypatch.setattr(settings, "RESPONSE_REQUIRE_CORROBORATION_FOR_EXCLUSION", False)
    monkeypatch.setattr(settings, "RESPONSE_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS", False)
    assert _elig(primary_intent="SPAM").routing_decision == NO_RESPONSE_RECOMMENDED
    assert _elig(primary_intent="INQUIRY", comment_text="كيف أضيف كلمة جديدة؟").reason_code == "inquiry_without_trusted_grounding"


def test_defaults_are_the_safe_ones():
    assert settings.RESPONSE_REQUIRE_CORROBORATION_FOR_EXCLUSION is True
    assert settings.RESPONSE_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS is True


def _engine(tmp_path):
    engine = get_engine(tmp_path / "safety.sqlite")
    init_db(engine)
    return engine


def _seed(engine, comment_id, text="نص تعليق للاختبار", **overrides):
    now = dt.datetime(2026, 1, 15, 12, 0, 0)
    row = {
        "comment_id": comment_id, "source_file": "t.xlsx", "source_row_index": 1, "original_comment": text, "ingested_at": now,
        "comment_created_at": now, "review_status": "CLASSIFIED", "ready_to_send": False, "primary_intent": "SPAM", "topic_subtype": "GENERAL",
        "ai_primary_intent": "SPAM", "ai_topic_subtype": "GENERAL", "ai_confidence": 0.41, "row_version": 1,
    }
    row.update(overrides)
    upsert_review_items(engine, [row])


def test_uncertain_spam_is_kept_visible_with_its_original_prediction_and_stays_actionable(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-spam", text="أقترح إضافة خاصية البحث بالجذر في المعجم")
    svc = ReviewWorkflowService(engine)
    client = FakeGeminiClient([])
    result = _run(_req("أقترح إضافة خاصية البحث بالجذر في المعجم", "SPAM", comment_id="c-spam"), client, _empty_ok()).generation_result
    assert result.routing_decision == REVIEW_ONLY and client.call_count == 0
    detail = svc.apply_generation_result(result, intent="SPAM", topic="GENERAL", confidence=0.41, needs_review=True)
    assert detail.review_status == "REVIEW_ONLY"                       # visible in the pending queue, not NO_RESPONSE_RECOMMENDED / hidden
    assert detail.review_status in settings.QUEUE_PENDING_STATUSES
    item = svc.repo.get("c-spam")
    assert item.ai_primary_intent == "SPAM" and item.ai_topic_subtype == "GENERAL" and item.ai_confidence == pytest.approx(0.41)
    assert item.original_comment == "أقترح إضافة خاصية البحث بالجذر في المعجم"
    sent = svc.edit_and_send("c-spam", REVIEWER, "نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر.")   # a human can still answer it
    assert sent.review_status == "EDITED_AND_APPROVED"
    assert svc.repo.get("c-spam").ai_primary_intent == "SPAM"          # the model's original label is never overwritten


# --------------------------------------------------------------- 4. reviewer-edited replies: block, preserve, re-validate
def _drafted(tmp_path, comment_id="c-edit"):
    engine = _engine(tmp_path)
    _seed(engine, comment_id, text="أتمنى إضافة خاصية البحث بالجذر", primary_intent="SUGGESTION", ai_primary_intent="SUGGESTION")
    svc = ReviewWorkflowService(engine)
    draft = "نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر."
    svc.apply_generation_result(
        GenerationResult(
            comment_id=comment_id, routing_decision="GENERATE_DRAFT", generation_status="SUCCEEDED", suggested_response=draft,
            review_required=True, reason_code="suggestion_eligible", provider="gemini", model_name="m", prompt_version="rg-v14.2",
            generated_at="2026-08-25T12:00:00+00:00",
        )
    )
    return engine, svc, draft


def test_unsafe_reviewer_edit_is_blocked_before_any_state_change(tmp_path):
    engine, svc, draft = _drafted(tmp_path)
    before = svc.repo.get("c-edit")
    with pytest.raises(DraftValidationError) as exc:
        svc.edit_and_send("c-edit", REVIEWER, "شكرًا لكم، سنقوم بإضافة الخاصية في التحديث القادم.")
    assert exc.value.reason_code == "unsupported_commitment" and exc.value.message_ar
    after = svc.repo.get("c-edit")
    assert after.review_status == before.review_status == "DRAFT_READY"
    assert after.suggested_final_reply == draft                          # the generated draft is preserved
    assert after.reviewer_action is None and not after.final_reply
    assert after.row_version == before.row_version                       # nothing was written
    assert after.outbound_status != "SENT"


def test_corrected_reply_is_revalidated_and_saved_exactly_as_written(tmp_path):
    engine, svc, draft = _drafted(tmp_path)
    with pytest.raises(DraftValidationError):
        svc.edit_and_send("c-edit", REVIEWER, "سنضيف الخاصية قريبا.")
    ok, reason, message = svc.validate_reply_text("c-edit", "نشكر لكم اقتراحكم، ونرحب بمشاركتكم.")
    assert ok is True and reason is None and message is None
    corrected = "  نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر.  "
    detail = svc.edit_and_send("c-edit", REVIEWER, corrected)
    assert detail.edited_response == corrected.strip() or detail.edited_response == corrected   # stored as written (no rewriting)
    assert "خاصية البحث بالجذر" in detail.final_response


def test_reviewer_text_is_never_silently_rewritten(tmp_path):
    engine, svc, draft = _drafted(tmp_path)
    text = "نشكر لكم اقتراحكم القيّم، وجزاكم الله خيرًا."
    detail = svc.edit_and_send("c-edit", REVIEWER, text)
    assert detail.final_response == text


def test_validate_reply_text_stores_nothing(tmp_path):
    engine, svc, draft = _drafted(tmp_path)
    v = svc.repo.get("c-edit").row_version
    assert svc.validate_reply_text("c-edit", "تمت إضافة الكلمة إلى المعجم.")[0] is False
    assert svc.repo.get("c-edit").row_version == v


def test_stale_generated_draft_cannot_be_sent_as_is_when_it_fails_the_current_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OFFICIAL_KNOWLEDGE_SOURCE_PATH", None)
    engine = _engine(tmp_path)
    _seed(engine, "c-stale", primary_intent="SUGGESTION", ai_primary_intent="SUGGESTION")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(
        GenerationResult(
            comment_id="c-stale", routing_decision="GENERATE_DRAFT", generation_status="SUCCEEDED",
            suggested_response="شكرًا لكم، سنقوم بمراجعة الملاحظة وإضافتها.", review_required=True, reason_code="suggestion_eligible",
            provider="gemini", model_name="m", prompt_version="rg-v13", generated_at="2026-08-25T12:00:00+00:00",
        )
    )
    with pytest.raises(DraftValidationError):
        svc.send_as_is("c-stale", REVIEWER)
    assert svc.repo.get("c-stale").review_status == "DRAFT_READY"


def test_safe_generated_draft_still_sends_as_is(tmp_path):
    engine, svc, draft = _drafted(tmp_path)
    assert svc.send_as_is("c-edit", REVIEWER).review_status == "APPROVED"


def test_nothing_is_ever_sent_automatically(tmp_path):
    engine, svc, draft = _drafted(tmp_path)
    svc.send_as_is("c-edit", REVIEWER)
    assert svc.repo.get("c-edit").outbound_status in ("PENDING", "NOT_APPLICABLE")       # approval does not dispatch
    assert settings.AUTOMATIC_ACK_ENABLED is False


# ------------------------------------------------------------------------------------------------- 5. HTTP behaviour
def _json(url, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), err.headers


def test_http_returns_422_with_the_reason_and_keeps_the_draft(tmp_path):
    engine, svc, draft = _drafted(tmp_path)
    user = create_user(engine, email="r@example.org", password="passwordR123", display_name="R", role="REVIEWER")
    from sqlalchemy.orm import Session
    from rcs.review.models import ReviewItem

    with Session(engine) as s:
        s.get(ReviewItem, "c-edit").reviewer_user_id = user.user_id
        s.commit()
    server, base = serve_in_thread(engine)
    try:
        cookie = _json(f"{base}/api/login", method="POST", data={"email": "r@example.org", "password": "passwordR123"})[2].get("Set-Cookie").split(";")[0]
        detail = _json(f"{base}/api/items/c-edit", cookie=cookie)[1]
        status, payload, _ = _json(
            f"{base}/api/items/c-edit/edit-and-send", method="POST", cookie=cookie,
            data={"expected_version": detail["row_version"], "edited_response": "يقبل المعجم الإضافات بعد أسبوعين من تقديمها."},
        )
        assert status == 422
        assert payload["error"] == "draft_validation_failed" and payload["reason_code"] == "unsupported_policy_claim"
        assert payload["message_ar"] and payload["draft_preserved"] is True and payload["sent"] is False
        status, check, _ = _json(f"{base}/api/items/c-edit/validate-reply", method="POST", cookie=cookie, data={"reply": "نشكر لكم اقتراحكم."})
        assert status == 200 and check["valid"] is True
        status, check, _ = _json(f"{base}/api/items/c-edit/validate-reply", method="POST", cookie=cookie, data={"reply": "سنصلح الخلل قريبا."})
        assert status == 200 and check["valid"] is False and check["reason_code"] == "unsupported_commitment"
        again = _json(f"{base}/api/items/c-edit", cookie=cookie)[1]
        assert again["review_status"] == "DRAFT_READY" and again["row_version"] == detail["row_version"]
        status, ok, _ = _json(
            f"{base}/api/items/c-edit/edit-and-send", method="POST", cookie=cookie,
            data={"expected_version": detail["row_version"], "edited_response": "نشكر لكم اقتراحكم القيّم حول هذه الخاصية."},
        )
        assert status == 200 and ok["review_status"] == "EDITED_AND_APPROVED"
    finally:
        server.shutdown()
