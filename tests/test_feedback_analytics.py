"""Phase 3 — feedback analytics: aggregation, breakdowns, small-sample
marking, edit buckets, and duplicate-group non-inflation."""

from __future__ import annotations

import datetime as dt

from rcs.review.feedback import list_feedback_examples
from rcs.review.feedback_analytics import (
    MIN_RELIABLE_SAMPLE_SIZE,
    compute_feedback_analytics,
    edit_bucket,
)
from rcs.review.identity import ReviewerIdentity
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService
from sqlalchemy.orm import Session

REVIEWER = ReviewerIdentity("rev-fa", "مراجع التحليلات")


def _base_item(**overrides) -> ReviewItem:
    short_id = overrides.pop("comment_id", "1")
    defaults = dict(
        comment_id="fa-" + short_id,
        source_file="analytics_test.xlsx",
        source_row_index=0,
        original_comment="تعليق حقيقي للاختبار",
        review_status="DRAFT_READY",
        ai_primary_intent="SUGGESTION",
        ai_topic_subtype="WORD_SUBMISSION",
        ai_sentiment="NEUTRAL",
        suggested_final_reply="مسودة الذكاء الاصطناعي الأصلية.",
        ai_suggested_final_reply="مسودة الذكاء الاصطناعي الأصلية.",
        generation_status="SUCCEEDED",
        generation_routing_decision="GENERATE_DRAFT",
        generation_provider="gemini",
        generation_model_name="gemini-3.6-flash",
        generation_prompt_version="rg-v7",
        source_reference="analytics_test.xlsx:0",
    )
    defaults.update(overrides)
    return ReviewItem(**defaults)


def _engine(tmp_path, name="analytics.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def test_empty_dataset_does_not_crash_and_reports_none_rates():
    analytics = compute_feedback_analytics([])
    assert analytics.overall.reviewed_count == 0
    assert analytics.overall.accepted_as_is_rate is None
    assert analytics.overall.small_sample is True
    assert analytics.by_intent == []


def test_overall_counts_and_breakdowns(tmp_path):
    engine = _engine(tmp_path)
    svc = ReviewWorkflowService(engine)
    rows = [
        _base_item(comment_id="1", source_row_index=1, ai_primary_intent="SUGGESTION", ai_topic_subtype="WORD_SUBMISSION"),
        _base_item(comment_id="2", source_row_index=2, ai_primary_intent="SUGGESTION", ai_topic_subtype="WORD_SUBMISSION"),
        _base_item(comment_id="3", source_row_index=3, ai_primary_intent="INQUIRY", ai_topic_subtype="GENERAL"),
    ]
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()

    svc.send_as_is("fa-1", REVIEWER)
    svc.edit_and_send("fa-2", REVIEWER, "رد معدل مختلف تمامًا عن المسودة الأصلية بشكل واضح.")
    svc.mark_no_response_needed("fa-3", REVIEWER, reason="غير مناسب")

    examples = list_feedback_examples(engine)
    analytics = compute_feedback_analytics(examples)

    assert analytics.overall.reviewed_count == 3
    assert analytics.overall.accepted_as_is_count == 1
    assert analytics.overall.edited_count == 1
    assert analytics.overall.no_response_count == 1
    assert analytics.overall.small_sample is True  # 3 < MIN_RELIABLE_SAMPLE_SIZE

    by_intent = {g.key: g for g in analytics.by_intent}
    assert by_intent["SUGGESTION"].reviewed_count == 2
    assert by_intent["INQUIRY"].reviewed_count == 1

    by_topic = {g.key: g for g in analytics.by_topic}
    assert by_topic["WORD_SUBMISSION"].reviewed_count == 2
    assert by_topic["GENERAL"].reviewed_count == 1

    by_prompt = {g.key: g for g in analytics.by_prompt_version}
    assert by_prompt["rg-v7"].reviewed_count == 3


def test_small_sample_flag_threshold(tmp_path):
    engine = _engine(tmp_path)
    svc = ReviewWorkflowService(engine)
    n = MIN_RELIABLE_SAMPLE_SIZE
    with Session(engine) as session:
        session.add_all([_base_item(comment_id=str(i), source_row_index=i) for i in range(n)])
        session.commit()
    for i in range(n):
        svc.send_as_is(f"fa-{i}", REVIEWER)

    analytics = compute_feedback_analytics(list_feedback_examples(engine))
    assert analytics.overall.reviewed_count == n
    assert analytics.overall.small_sample is False  # exactly at the threshold, not below it

    # One fewer -> flagged small.
    engine2 = _engine(tmp_path, name="analytics2.sqlite")
    svc2 = ReviewWorkflowService(engine2)
    with Session(engine2) as session:
        session.add_all([_base_item(comment_id=str(i), source_row_index=i) for i in range(n - 1)])
        session.commit()
    for i in range(n - 1):
        svc2.send_as_is(f"fa-{i}", REVIEWER)
    analytics2 = compute_feedback_analytics(list_feedback_examples(engine2))
    assert analytics2.overall.small_sample is True


def test_prompt_version_breakdown_isolates_versions(tmp_path):
    engine = _engine(tmp_path)
    svc = ReviewWorkflowService(engine)
    with Session(engine) as session:
        session.add_all(
            [
                _base_item(comment_id="v7a", source_row_index=1, generation_prompt_version="rg-v7"),
                _base_item(comment_id="v7b", source_row_index=2, generation_prompt_version="rg-v7"),
                _base_item(comment_id="v8", source_row_index=3, generation_prompt_version="rg-v8-candidate-context"),
            ]
        )
        session.commit()
    svc.send_as_is("fa-v7a", REVIEWER)
    svc.send_as_is("fa-v7b", REVIEWER)
    svc.edit_and_send("fa-v8", REVIEWER, "رد آخر مختلف كليًا.")

    analytics = compute_feedback_analytics(list_feedback_examples(engine))
    by_prompt = {g.key: g for g in analytics.by_prompt_version}
    assert by_prompt["rg-v7"].reviewed_count == 2
    assert by_prompt["rg-v7"].accepted_as_is_count == 2
    assert by_prompt["rg-v8-candidate-context"].reviewed_count == 1
    assert by_prompt["rg-v8-candidate-context"].edited_count == 1


def test_duplicate_group_does_not_inflate_analytics(tmp_path):
    engine = _engine(tmp_path)
    shared_text = "نص مكرر لاختبار عدم تضخيم التحليلات"
    canonical = _base_item(comment_id="dup-canon", source_row_index=0, original_comment=shared_text)
    canonical.dedupe_canonical_id = "fa-dup-canon"
    canonical.content_dedupe_key = "shared"
    dups = []
    for i in range(1, 4):
        d = _base_item(
            comment_id=f"dup-{i}",
            source_row_index=i,
            original_comment=shared_text,
            review_status="DUPLICATE_CONTENT",
            suggested_final_reply=None,
            ai_suggested_final_reply=None,
            generation_status=None,
            generation_provider=None,
            generation_model_name=None,
            generation_prompt_version=None,
        )
        d.dedupe_canonical_id = "fa-dup-canon"
        d.content_dedupe_key = "shared"
        dups.append(d)
    with Session(engine) as session:
        session.add(canonical)
        session.add_all(dups)
        session.commit()

    svc = ReviewWorkflowService(engine)
    svc.send_as_is("fa-dup-canon", REVIEWER)

    analytics = compute_feedback_analytics(list_feedback_examples(engine))
    assert analytics.overall.reviewed_count == 1  # not 4
    assert analytics.overall.accepted_as_is_count == 1
    assert analytics.overall.accepted_as_is_rate == 1.0


def test_edit_bucket_deterministic_thresholds():
    assert edit_bucket(None) is None
    assert edit_bucket(1.0) == "unchanged"
    assert edit_bucket(0.95) == "unchanged"
    assert edit_bucket(0.9499999) == "light_edit"
    assert edit_bucket(0.70) == "light_edit"
    assert edit_bucket(0.6999999) == "substantial_edit"
    assert edit_bucket(0.0) == "substantial_edit"
