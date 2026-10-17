"""Reviewer workflow backend tests. No live Gemini, no external-platform HTTP."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from config import settings
from rcs.response_generation.outbound import UnconfiguredOutboundSender as LegacyUnconfigured
from rcs.response_generation.schema import GenerationResult
from rcs.review.identity import ReviewerIdentity
from rcs.review.models import ReviewItem
from rcs.review.outbound import RecordingOutboundSender, UnconfiguredOutboundSender
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.review.workflow import InvalidTransition, VersionConflict, next_status


REVIEWER = ReviewerIdentity("rev-1", "مراجع تجريبي")


def _engine(tmp_path):
    engine = get_engine(tmp_path / "review.sqlite")
    init_db(engine)
    return engine


def _seed(engine, comment_id: str, **overrides) -> None:
    now = dt.datetime(2026, 1, 15, 12, 0, 0)
    row = {
        "comment_id": comment_id,
        "source_file": "visitors-reports.xlsx",
        "source_row_index": 1,
        "original_comment": "شكرًا لكم على المعجم",
        "ingested_at": now,
        "comment_created_at": now,
        "review_status": "CLASSIFIED",
        "ready_to_send": False,
        "primary_intent": "THANKS_PRAISE",
        "topic_subtype": "GENERAL",
        "ai_primary_intent": "THANKS_PRAISE",
        "ai_topic_subtype": "GENERAL",
        "ai_confidence": 0.9,
        "row_version": 1,
    }
    row.update(overrides)
    upsert_review_items(engine, [row])


def _gen(comment_id: str, **overrides) -> GenerationResult:
    base = dict(
        comment_id=comment_id,
        routing_decision="GENERATE_DRAFT",
        generation_status="SUCCEEDED",
        suggested_response="شكرًا لتواصلكم معنا، نسعد بثقتكم بالمعجم.",
        review_required=True,
        reason_code="thanks_praise_eligible",
        provider="gemini",
        model_name="gemini-3.6-flash",
        prompt_version="rg-v1",
        generated_at="2026-08-25T12:00:00+00:00",
    )
    base.update(overrides)
    return GenerationResult(**base)


def test_illegal_transition_rejected():
    with pytest.raises(InvalidTransition):
        next_status("UNPROCESSED", "SEND_AS_IS")


def test_persist_generation_and_queue_draft(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-draft")
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(_gen("c-draft"), intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE")
    assert detail.review_status == "DRAFT_READY"
    assert detail.suggested_response
    page = svc.list_review_queue()
    assert page.total == 1
    assert page.items[0].comment_id == "c-draft"


def test_review_only_in_queue(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-ro", primary_intent="INQUIRY", ai_primary_intent="INQUIRY")
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        _gen(
            "c-ro",
            routing_decision="REVIEW_ONLY",
            generation_status="SKIPPED",
            suggested_response=None,
            reason_code="inquiry_without_trusted_grounding",
            provider=None,
            model_name=None,
        ),
        intent="INQUIRY",
    )
    assert detail.review_status == "REVIEW_ONLY"
    assert detail.suggested_response is None
    assert "EDIT_AND_SEND" in detail.allowed_actions
    assert "SEND_AS_IS" not in detail.allowed_actions
    assert svc.list_review_queue().total == 1


def test_no_response_recommended_is_pending_in_default_queue(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-spam", primary_intent="SPAM", ai_primary_intent="SPAM")
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        _gen(
            "c-spam",
            routing_decision="NO_RESPONSE_RECOMMENDED",
            generation_status="SKIPPED",
            suggested_response=None,
            reason_code="spam",
            provider=None,
            model_name=None,
        ),
        intent="SPAM",
    )
    assert detail.review_status == "NO_RESPONSE_RECOMMENDED"
    # A model recommendation is pending human review, not completed work.
    assert svc.list_review_queue().total == 1
    assert svc.workflow_counts()["pending"] == 1
    assert svc.workflow_counts()["no_reply"] == 0
    assert detail.allowed_actions == ("EDIT_AND_SEND", "NO_RESPONSE_NEEDED")


def test_no_response_recommended_can_be_overridden_with_a_reply_without_a_draft(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-ovr", primary_intent="COMPLAINT", ai_primary_intent="SPAM")
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        _gen("c-ovr", routing_decision="NO_RESPONSE_RECOMMENDED", generation_status="SKIPPED",
             suggested_response=None, reason_code="spam", provider=None, model_name=None),
        intent="SPAM",
    )
    assert not detail.suggested_response
    reviewer = ReviewerIdentity(reviewer_id="r1", display_name="R1", user_id="u1")
    # No draft exists: SEND_AS_IS is refused, nothing is generated or sent.
    with pytest.raises(InvalidTransition):
        svc.send_as_is("c-ovr", reviewer, expected_version=detail.row_version)
    # Stale version is still rejected (optimistic concurrency preserved).
    with pytest.raises(VersionConflict):
        svc.edit_and_send("c-ovr", reviewer, "رد المراجع", expected_version=detail.row_version + 5)
    done = svc.edit_and_send("c-ovr", reviewer, "رد المراجع", expected_version=detail.row_version)
    assert done.review_status == "EDITED_AND_APPROVED"
    assert done.reviewer_action == "EDIT_AND_SEND"
    assert done.final_response == "رد المراجع"
    assert svc.workflow_counts()["pending"] == 0


def test_no_response_recommended_confirmed_becomes_final_no_reply(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-conf", primary_intent="SPAM", ai_primary_intent="SPAM")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(
        _gen("c-conf", routing_decision="NO_RESPONSE_RECOMMENDED", generation_status="SKIPPED",
             suggested_response=None, reason_code="spam", provider=None, model_name=None),
        intent="SPAM",
    )
    reviewer = ReviewerIdentity(reviewer_id="r1", display_name="R1", user_id="u1")
    done = svc.mark_no_response_needed("c-conf", reviewer, reason="محتوى غير ذي صلة")
    assert done.review_status == "NO_REPLY"
    assert svc.workflow_counts()["no_reply"] == 1


def test_spam_draft_can_be_marked_no_response_needed(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-spam-draft", primary_intent="SPAM", ai_primary_intent="SPAM")
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        _gen("c-spam-draft", reason_code="spam", suggested_response="شكرًا لتواصلكم، تم استلام رسالتكم."),
        intent="SPAM",
    )
    assert detail.review_status == "DRAFT_READY"
    assert detail.suggested_response
    assert "NO_RESPONSE_NEEDED" in detail.allowed_actions
    marked = svc.mark_no_response_needed("c-spam-draft", REVIEWER, reason="spam")
    assert marked.review_status == "NO_REPLY"
    assert marked.ready_to_send is False


def test_send_as_is(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c1")
    svc = ReviewWorkflowService(engine)
    suggestion = _gen("c1").suggested_response
    svc.apply_generation_result(_gen("c1"))
    detail = svc.send_as_is("c1", REVIEWER)
    assert detail.review_status == "APPROVED"
    assert detail.reviewer_action == "SEND_AS_IS"
    assert detail.final_response == suggestion
    assert detail.suggested_response == suggestion
    assert detail.ready_to_send is True
    assert detail.reviewed_at is not None
    assert detail.outbound_status == "PENDING"


def test_edit_and_send_preserves_suggestion(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c2")
    svc = ReviewWorkflowService(engine)
    original = _gen("c2").suggested_response
    svc.apply_generation_result(_gen("c2"))
    edited = "شكرًا لتواصلكم معنا ولاهتمامكم."
    detail = svc.edit_and_send("c2", REVIEWER, edited)
    assert detail.review_status == "EDITED_AND_APPROVED"
    assert detail.edited_response == edited
    assert detail.final_response == edited
    assert detail.suggested_response == original
    with Session(engine) as session:
        item = session.get(ReviewItem, "c2")
        assert item.ai_suggested_final_reply == original
        assert item.suggested_final_reply == original


def test_no_response_needed(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c3")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c3"))
    detail = svc.mark_no_response_needed("c3", REVIEWER, reason="لا يستدعي ردًا")
    assert detail.review_status == "NO_REPLY"
    assert detail.final_response is None
    assert detail.ready_to_send is False
    assert detail.no_response_reason == "لا يستدعي ردًا"
    assert detail.outbound_status == "NOT_APPLICABLE"


def test_edit_and_send_rejects_empty_response(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-empty")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c-empty"))
    with pytest.raises(InvalidTransition):
        svc.edit_and_send("c-empty", REVIEWER, "  \n")


def test_send_as_is_rejects_missing_suggestion(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-nosug")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(
        _gen("c-nosug", routing_decision="REVIEW_ONLY", generation_status="SKIPPED", suggested_response=None, reason_code="x", provider=None, model_name=None)
    )
    with pytest.raises(InvalidTransition):
        svc.send_as_is("c-nosug", REVIEWER)


def test_approve_stays_pending_without_dispatch(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-pend")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c-pend"))
    detail = svc.send_as_is("c-pend", REVIEWER)
    assert detail.outbound_status == "PENDING"
    assert detail.review_status != "SENT"
    with Session(engine) as session:
        item = session.get(ReviewItem, "c-pend")
        assert item.outbound_status == "PENDING"
        assert item.review_status == "APPROVED"
    engine = _engine(tmp_path)
    _seed(engine, "c-manual", primary_intent="INQUIRY")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(
        _gen("c-manual", routing_decision="REVIEW_ONLY", generation_status="SKIPPED", suggested_response=None, reason_code="inquiry_without_trusted_grounding", provider=None, model_name=None)
    )
    with pytest.raises(InvalidTransition):
        svc.send_as_is("c-manual", REVIEWER)
    detail = svc.edit_and_send("c-manual", REVIEWER, "يمكنكم استخدام نموذج إضافة كلمة في الموقع.")
    assert detail.review_status == "EDITED_AND_APPROVED"
    assert detail.suggested_response is None
    assert detail.final_response.startswith("يمكنكم")


def test_invalid_transition_from_approved(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c4")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c4"))
    svc.send_as_is("c4", REVIEWER)
    with pytest.raises(InvalidTransition):
        svc.mark_no_response_needed("c4", REVIEWER)


def test_idempotent_send_as_is(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c5")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c5"))
    first = svc.send_as_is("c5", REVIEWER)
    second = svc.send_as_is("c5", REVIEWER)
    assert first.row_version == second.row_version
    assert second.final_response == first.final_response


def test_generation_apply_idempotent(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c6")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c6"))
    again = svc.apply_generation_result(_gen("c6", suggested_response="نص مختلف يجب ألا يُكتب"))
    assert again.suggested_response == _gen("c6").suggested_response


def test_unconfigured_legacy_sender_still_raises():
    with pytest.raises(NotImplementedError):
        LegacyUnconfigured().send_approved_response("c", "hello")


def test_outbound_not_configured_does_not_lose_final(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c7")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c7"))
    approved = svc.send_as_is("c7", REVIEWER)
    detail, result = svc.dispatch_outbound("c7", UnconfiguredOutboundSender())
    assert result.ok is False
    assert result.error_code == "CONNECTOR_NOT_CONFIGURED"
    assert detail.review_status == "SEND_FAILED"
    assert detail.final_response == approved.final_response
    with Session(engine) as session:
        assert session.get(ReviewItem, "c7").final_reply == approved.final_response


def test_send_failure_preserves_text_and_retry(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c8")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c8"))
    svc.send_as_is("c8", REVIEWER)
    fail = RecordingOutboundSender(fail=True)
    detail, result = svc.dispatch_outbound("c8", fail)
    assert result.ok is False
    assert detail.final_response
    ok_sender = RecordingOutboundSender(fail=False)
    detail2, result2 = svc.dispatch_outbound("c8", ok_sender)
    assert result2.ok is True
    assert detail2.review_status == "SENT"
    assert detail2.final_response == detail.final_response
    assert len(ok_sender.calls) == 1


def test_queue_filter_sort_pagination(tmp_path):
    engine = _engine(tmp_path)
    svc = ReviewWorkflowService(engine)
    _seed(engine, "old-thanks", ingested_at=dt.datetime(2026, 1, 1), primary_intent="THANKS_PRAISE")
    _seed(
        engine,
        "new-complaint",
        ingested_at=dt.datetime(2026, 2, 1),
        original_comment="الموقع لا يعمل بشكل جيد",
        primary_intent="COMPLAINT",
        ai_primary_intent="COMPLAINT",
        topic_subtype="GENERAL",
    )
    svc.apply_generation_result(_gen("old-thanks"), intent="THANKS_PRAISE")
    svc.apply_generation_result(
        _gen("new-complaint", reason_code="complaint_safe_acknowledgement"),
        intent="COMPLAINT",
    )
    oldest = svc.list_review_queue(newest_first=False)
    assert oldest.items[0].comment_id == "new-complaint"
    assert oldest.items[0].review_priority == 1
    page = svc.list_review_queue(limit=1, offset=0)
    assert page.total == 2
    assert len(page.items) == 1
    found = svc.list_review_queue(search="c-does-not-exist")
    assert found.total == 0
    found2 = svc.list_review_queue(search="old-thanks")
    assert found2.total == 1


def test_version_conflict(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c9")
    svc = ReviewWorkflowService(engine)
    d = svc.apply_generation_result(_gen("c9"))
    svc.send_as_is("c9", REVIEWER, expected_version=d.row_version)
    with pytest.raises(VersionConflict):
        svc.send_as_is("c9", ReviewerIdentity("other"), expected_version=d.row_version)


def test_claim_and_audit_fields(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c10")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c10"), sentiment="POSITIVE")
    claimed = svc.claim_review("c10", REVIEWER)
    assert claimed.review_status == "IN_REVIEW"
    assert claimed.reviewer_id == "rev-1"
    assert claimed.reviewer_started_at is not None
    again = svc.claim_review("c10", REVIEWER)
    assert again.review_status == "IN_REVIEW"
    with pytest.raises(InvalidTransition):
        svc.claim_review("c10", ReviewerIdentity("rev-2"))


def test_reviewer_detail_exposes_non_pii_entry_context(tmp_path):
    """Reviewer entry-context fix: original_title/entry_id/platform/
    original_type must reach the reviewer-facing detail DTO, so a
    reviewer can understand vague references like "هذا المدخل"."""
    engine = _engine(tmp_path)
    _seed(
        engine,
        "c-context",
        original_title="حَسَبَ",
        entry_id="entry-4471",
        platform="الرياض",
        original_type="التعليق على المدخل",
        visitor_name="زائر حقيقي",
        visitor_email="visitor@example.com",
    )
    svc = ReviewWorkflowService(engine)
    detail = svc.get_review_item("c-context")
    assert detail.original_title == "حَسَبَ"
    assert detail.entry_id == "entry-4471"
    assert detail.platform == "الرياض"
    assert detail.original_type == "التعليق على المدخل"


def test_reviewer_detail_never_exposes_visitor_name_or_email(tmp_path):
    from rcs.review.dto import detail_to_dict

    engine = _engine(tmp_path)
    _seed(
        engine,
        "c-pii",
        visitor_name="زائر حقيقي",
        visitor_email="secret.visitor@example.com",
    )
    svc = ReviewWorkflowService(engine)
    detail = svc.get_review_item("c-pii")
    assert not hasattr(detail, "visitor_name")
    assert not hasattr(detail, "visitor_email")
    payload = detail_to_dict(detail)
    assert "visitor_name" not in payload
    assert "visitor_email" not in payload
    assert "secret.visitor@example.com" not in str(payload)
    assert "زائر حقيقي" not in str(payload)


def test_reviewer_detail_handles_missing_context_cleanly(tmp_path):
    """A row with no title/entry_id/platform/type (all None, the common
    case for e.g. رسالة تواصل rows) must not break detail retrieval."""
    engine = _engine(tmp_path)
    _seed(engine, "c-no-context", original_title=None, entry_id=None, platform=None, original_type=None)
    svc = ReviewWorkflowService(engine)
    detail = svc.get_review_item("c-no-context")
    assert detail.original_title is None
    assert detail.entry_id is None
    assert detail.platform is None
    assert detail.original_type is None


def test_repository_round_trip(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c11")
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(_gen("c11"), sentiment="POSITIVE", confidence=0.91)
    detail, internal = svc.get_review_item("c11", internal=True)
    assert detail.sentiment == "POSITIVE"
    assert internal.generation_model_name == "gemini-3.6-flash"
    assert "gemini" not in (detail.original_comment or "").lower()


def test_priority_does_not_use_sentiment_alone(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, "c-pos", ai_confidence=0.99)
    svc = ReviewWorkflowService(engine)
    d = svc.apply_generation_result(_gen("c-pos"), intent="THANKS_PRAISE", sentiment="NEGATIVE")
    assert d.review_priority == 3
    assert "sentiment" not in (d.priority_reason or "")


def test_queue_sort_modes_and_counts(tmp_path):
    engine = _engine(tmp_path)
    svc = ReviewWorkflowService(engine)
    _seed(engine, "old-thanks", ingested_at=dt.datetime(2026, 1, 1), primary_intent="THANKS_PRAISE")
    _seed(
        engine,
        "new-complaint",
        ingested_at=dt.datetime(2026, 2, 1),
        original_comment="الموقع لا يعمل بشكل جيد",
        primary_intent="COMPLAINT",
        ai_primary_intent="COMPLAINT",
        topic_subtype="GENERAL",
    )
    svc.apply_generation_result(_gen("old-thanks"), intent="THANKS_PRAISE")
    svc.apply_generation_result(
        _gen("new-complaint", reason_code="complaint_safe_acknowledgement"),
        intent="COMPLAINT",
    )
    newest = svc.list_review_queue(sort="newest")
    assert newest.items[0].comment_id == "new-complaint"
    oldest = svc.list_review_queue(sort="oldest")
    assert oldest.items[0].comment_id == "old-thanks"
    by_priority = svc.list_review_queue(sort="priority")
    assert by_priority.items[0].comment_id == "new-complaint"
    _seed(engine, "older-p3", ingested_at=dt.datetime(2025, 12, 1), primary_intent="THANKS_PRAISE")
    _seed(engine, "newer-p3", ingested_at=dt.datetime(2026, 3, 1), primary_intent="THANKS_PRAISE")
    svc.apply_generation_result(_gen("older-p3"), intent="THANKS_PRAISE")
    svc.apply_generation_result(_gen("newer-p3"), intent="THANKS_PRAISE")
    tied = [i.comment_id for i in svc.list_review_queue(sort="priority").items if i.review_priority == 3]
    assert tied.index("older-p3") < tied.index("newer-p3")
    counts = svc.workflow_counts()
    assert counts["pending"] == 4
    assert counts["all"] == 4


def test_operational_analytics_from_persisted_timestamps(tmp_path):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 26, 12, 0, 0)
    _seed(
        engine,
        "in-range",
        ingested_at=now - dt.timedelta(days=2),
        comment_created_at=now - dt.timedelta(days=2),
        review_status="DRAFT_READY",
        review_priority=1,
        primary_intent="COMPLAINT",
        topic_subtype="TECHNICAL_ISSUE",
        sentiment="NEGATIVE",
    )
    _seed(
        engine,
        "done",
        ingested_at=now - dt.timedelta(days=3),
        comment_created_at=now - dt.timedelta(days=3),
        review_status="APPROVED",
        reviewer_action="SEND_AS_IS",
        reviewer_id="rev-1",
        reviewer_started_at=now - dt.timedelta(hours=2),
        reviewed_at=now - dt.timedelta(hours=1),
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        sentiment="POSITIVE",
    )
    svc = ReviewWorkflowService(engine)
    snap = svc.operational_analytics(range_key="week", reviewer_id="rev-1", now=now)
    assert snap["kpis"]["incoming"] == 2
    assert snap["kpis"]["reviewed"] == 1
    assert snap["kpis"]["pending"] == 1
    assert snap["kpis"]["urgent_pending"] == 1
    assert snap["kpis"]["avg_review_seconds"] == 3600
    assert snap["outcomes"]["SEND_AS_IS"] == 1
    assert snap["today"]["reviewed_by_me"] == 1
    assert snap["quality"]["send_as_is_rate"] == 1
    intent_keys = [row["key"] for row in snap["classification"]["intent"]]
    topic_keys = [row["key"] for row in snap["classification"]["topic"]]
    sentiment_keys = [row["key"] for row in snap["classification"]["sentiment"]]
    assert intent_keys == list(settings.PRIMARY_INTENTS)
    assert topic_keys == list(settings.TOPIC_SUBTYPES)
    assert sentiment_keys == list(settings.SENTIMENTS)
    assert "WORD_OF_YEAR" not in topic_keys
    assert sum(1 for row in snap["classification"]["intent"] if row["count"] == 0) >= 1
    assert any(point["incoming"] or point["reviewed"] for point in snap["trend"])


def test_legacy_topic_is_unmapped_not_an_eighth_class(tmp_path):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 26, 12, 0, 0)
    _seed(
        engine,
        "legacy-woy",
        ingested_at=now,
        comment_created_at=now,
        review_status="DRAFT_READY",
        primary_intent="SUGGESTION",
        topic_subtype="WORD_OF_YEAR",
        sentiment="NEUTRAL",
    )
    svc = ReviewWorkflowService(engine)
    snap = svc.operational_analytics(range_key="week", now=now)
    topic_keys = [row["key"] for row in snap["classification"]["topic"]]
    assert topic_keys == list(settings.TOPIC_SUBTYPES)
    unmapped = snap["data_quality"]["unmapped_classification"]["topic"]
    assert unmapped == [{"key": "WORD_OF_YEAR", "count": 1}]


def test_all_three_reviewer_actions_aggregate(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 26, 15, 0, 0)
    # send_as_is/edit_and_send/mark_no_response_needed's persistence
    # (including reviewed_at) lives entirely in
    # PostgresReviewItemRepository, which has its own _now() -- that is
    # the only clock this test needs to control. service.py has no
    # `_now` of its own (dispatch_outbound, its one method that used to,
    # is not exercised by this test and now delegates its own clock
    # reads to the repository too).
    monkeypatch.setattr("rcs.review.repositories.postgres._now", lambda: now)
    svc = ReviewWorkflowService(engine)
    for cid in ("a-send", "a-edit", "a-none"):
        _seed(engine, cid, ingested_at=now - dt.timedelta(hours=3), comment_created_at=now - dt.timedelta(hours=3))
        svc.apply_generation_result(_gen(cid))
    svc.send_as_is("a-send", REVIEWER)
    svc.edit_and_send("a-edit", REVIEWER, "رد معدّل بعد المراجعة.")
    svc.mark_no_response_needed("a-none", REVIEWER, reason="لا يتطلب ردًا")
    snap = svc.operational_analytics(range_key="week", now=now + dt.timedelta(minutes=1))
    assert snap["outcomes"]["SEND_AS_IS"] == 1
    assert snap["outcomes"]["EDIT_AND_SEND"] == 1
    assert snap["outcomes"]["NO_RESPONSE_NEEDED"] == 1
    assert snap["kpis"]["reviewed"] == 3


def test_analytics_range_and_today_hourly_trend(tmp_path):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 26, 16, 0, 0)
    _seed(
        engine,
        "today-in",
        ingested_at=now.replace(hour=10, minute=0, second=0),
        comment_created_at=now.replace(hour=10),
        primary_intent="INQUIRY",
        topic_subtype="GENERAL",
        sentiment="NEUTRAL",
    )
    _seed(
        engine,
        "old-out",
        ingested_at=now - dt.timedelta(days=20),
        comment_created_at=now - dt.timedelta(days=20),
        review_status="APPROVED",
        reviewer_action="SEND_AS_IS",
        reviewed_at=now - dt.timedelta(days=20),
        primary_intent="SPAM",
        topic_subtype="OFF_TOPIC_PERSONAL",
        sentiment="NEGATIVE",
    )
    svc = ReviewWorkflowService(engine)
    today = svc.operational_analytics(range_key="today", now=now)
    assert today["kpis"]["incoming"] == 1
    assert today["kpis"]["reviewed"] == 0
    inquiry = next(row for row in today["classification"]["intent"] if row["key"] == "INQUIRY")
    spam = next(row for row in today["classification"]["intent"] if row["key"] == "SPAM")
    assert inquiry["count"] == 1
    assert spam["count"] == 0
    assert any(" " in point["bucket"] for point in today["trend"])
    custom = svc.operational_analytics(
        range_key="custom",
        now=now,
        custom_start=dt.datetime(2026, 8, 1),
        custom_end=dt.datetime(2026, 8, 31, 23, 59, 59),
    )
    assert custom["kpis"]["incoming"] == 2
    inverted = svc.operational_analytics(
        range_key="custom",
        now=now,
        custom_start=dt.datetime(2026, 8, 30),
        custom_end=dt.datetime(2026, 8, 1),
    )
    assert inverted["kpis"]["incoming"] == 0

