"""Phase 2 — Human Feedback Capture Layer.

Covers acceptance criteria A-G: SEND_AS_IS / EDIT_AND_SEND /
NO_RESPONSE_NEEDED derivation, exact-duplicate canonical-only counting,
restart persistence, provenance immutability, and privacy (no visitor
PII in the feedback view).
"""

from __future__ import annotations

import datetime as dt

from rcs.review.feedback import (
    FEEDBACK_CSV_COLUMNS,
    FeedbackExample,
    export_feedback_dataset,
    levenshtein_distance,
    list_feedback_examples,
    similarity_ratio,
    summarize_feedback,
)
from rcs.review.identity import ReviewerIdentity
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService
from sqlalchemy.orm import Session

REVIEWER = ReviewerIdentity("rev-fb", "مراجع التغذية الراجعة")


def _base_item(**overrides) -> ReviewItem:
    short_id = overrides.pop("comment_id", "1")
    defaults = dict(
        comment_id="c-" + short_id,
        source_file="feedback_test.xlsx",
        source_row_index=0,
        original_comment="تعليق حقيقي للاختبار",
        review_status="DRAFT_READY",
        ai_primary_intent="SUGGESTION",
        ai_topic_subtype="WORD_SUBMISSION",
        ai_sentiment="NEUTRAL",
        primary_intent="SUGGESTION",
        topic_subtype="WORD_SUBMISSION",
        sentiment="NEUTRAL",
        suggested_final_reply="مسودة الذكاء الاصطناعي الأصلية.",
        ai_suggested_final_reply="مسودة الذكاء الاصطناعي الأصلية.",
        generation_status="SUCCEEDED",
        generation_routing_decision="GENERATE_DRAFT",
        generation_provider="gemini",
        generation_model_name="gemini-3.6-flash",
        generation_prompt_version="rg-v7",
        source_reference="feedback_test.xlsx:0",
    )
    defaults.update(overrides)
    return ReviewItem(**defaults)


def _engine(tmp_path, name="feedback.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _seed(engine, item: ReviewItem):
    with Session(engine) as session:
        session.add(item)
        session.commit()


def test_send_as_is_feedback_derivation(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, _base_item(comment_id="send"))
    svc = ReviewWorkflowService(engine)
    svc.send_as_is("c-send", REVIEWER)

    examples = list_feedback_examples(engine)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.reviewer_action == "SEND_AS_IS"
    assert ex.draft_accepted_as_is is True
    assert ex.was_edited is False
    assert ex.no_response_needed is False
    assert ex.final_response == ex.ai_draft == "مسودة الذكاء الاصطناعي الأصلية."
    assert ex.edit_distance is None
    assert ex.edit_ratio is None


def test_edit_and_send_feedback_derivation_and_deterministic_edit_metrics(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, _base_item(comment_id="edit"))
    svc = ReviewWorkflowService(engine)
    edited = "ردّ المراجع البديل والمختلف عن المسودة."
    svc.edit_and_send("c-edit", REVIEWER, edited)

    examples = list_feedback_examples(engine)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.reviewer_action == "EDIT_AND_SEND"
    assert ex.draft_accepted_as_is is False
    assert ex.was_edited is True
    # AI draft remains exactly the original, unmodified by the edit.
    assert ex.ai_draft == "مسودة الذكاء الاصطناعي الأصلية."
    assert ex.reviewer_edit == edited
    assert ex.final_response == edited

    expected_distance = levenshtein_distance(ex.ai_draft, ex.final_response)
    expected_ratio = similarity_ratio(ex.ai_draft, ex.final_response)
    assert ex.edit_distance == expected_distance
    assert ex.edit_ratio == expected_ratio
    assert ex.edit_distance > 0
    assert 0.0 <= ex.edit_ratio <= 1.0

    # Determinism: recomputing from the same persisted state gives the same numbers.
    again = list_feedback_examples(engine)[0]
    assert again.edit_distance == ex.edit_distance
    assert again.edit_ratio == ex.edit_ratio


def test_levenshtein_and_ratio_known_values():
    # Manually verifiable small examples — documents the exact method used.
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("abc", "abc") == 0
    assert similarity_ratio("abc", "abc") == 1.0
    assert similarity_ratio("abc", "xyz") == 0.0


def test_no_response_needed_feedback_derivation(tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, _base_item(comment_id="noresp"))
    svc = ReviewWorkflowService(engine)
    svc.mark_no_response_needed("c-noresp", REVIEWER, reason="غير مناسب للرد")

    examples = list_feedback_examples(engine)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.reviewer_action == "NO_RESPONSE_NEEDED"
    assert ex.no_response_needed is True
    assert ex.draft_accepted_as_is is False
    assert ex.was_edited is False
    assert ex.final_response is None
    assert ex.edit_distance is None
    assert ex.edit_ratio is None


def test_exact_duplicate_group_produces_one_feedback_example(tmp_path):
    engine = _engine(tmp_path)
    shared_text = "نص مكرر تمامًا عبر أربعة صفوف مصدر مختلفة"
    canonical = _base_item(comment_id="dup-canon", source_row_index=0, original_comment=shared_text)
    canonical.dedupe_canonical_id = "c-dup-canon"
    canonical.content_dedupe_key = "shared-key"

    duplicates = []
    for i in range(1, 4):
        dup = _base_item(
            comment_id=f"dup-{i}",
            source_row_index=i,
            original_comment=shared_text,
            review_status="DUPLICATE_CONTENT",
            suggested_final_reply=None,
            ai_suggested_final_reply=None,
            generation_status=None,
            generation_routing_decision=None,
            generation_provider=None,
            generation_model_name=None,
            generation_prompt_version=None,
            ai_primary_intent=None,
            ai_topic_subtype=None,
            ai_sentiment=None,
        )
        dup.dedupe_canonical_id = "c-dup-canon"
        dup.content_dedupe_key = "shared-key"
        dup.outbound_status = "NOT_APPLICABLE"
        duplicates.append(dup)

    with Session(engine) as session:
        session.add(canonical)
        session.add_all(duplicates)
        session.commit()

    svc = ReviewWorkflowService(engine)
    svc.send_as_is("c-dup-canon", REVIEWER)

    examples = list_feedback_examples(engine)
    # Exactly one feedback example for the whole 4-row group — the 3
    # duplicates never received a reviewer_action and are excluded both
    # by that fact and by the explicit dedupe_canonical_id guard.
    assert len(examples) == 1
    assert examples[0].comment_id == "c-dup-canon"

    summary = summarize_feedback(examples)
    assert summary.completed_total == 1
    assert summary.send_as_is_count == 1
    assert summary.acceptance_as_is_rate == 1.0


def test_feedback_reconstruction_survives_restart(tmp_path):
    db_path = tmp_path / "restart.sqlite"
    engine1 = get_engine(db_path)
    init_db(engine1)
    _seed(engine1, _base_item(comment_id="restart"))
    svc1 = ReviewWorkflowService(engine1)
    svc1.edit_and_send("c-restart", REVIEWER, "رد بعد التعديل قبل إعادة التشغيل.")
    before = list_feedback_examples(engine1)

    # Simulate a fresh process: brand-new engine bound to the same file.
    engine2 = get_engine(db_path)
    init_db(engine2)
    after = list_feedback_examples(engine2)

    assert len(before) == len(after) == 1
    assert before[0] == after[0]  # dataclass equality: every field identical


def test_generation_provenance_preserved_exactly(tmp_path):
    engine = _engine(tmp_path)
    _seed(
        engine,
        _base_item(
            comment_id="prov",
            generation_provider="gemini",
            generation_model_name="gemini-3.6-flash",
            generation_prompt_version="rg-v7",
        ),
    )
    svc = ReviewWorkflowService(engine)
    svc.send_as_is("c-prov", REVIEWER)

    ex = list_feedback_examples(engine)[0]
    assert ex.generation_provider == "gemini"
    assert ex.generation_model == "gemini-3.6-flash"
    assert ex.generation_prompt_version == "rg-v7"


def test_feedback_dataset_has_no_visitor_pii(tmp_path):
    engine = _engine(tmp_path)
    phone_bearing_comment = "الجوال: 0501234567, نص الرسالة: أقترح كلمة جديدة للمعجم"
    item = _base_item(
        comment_id="pii",
        original_comment=phone_bearing_comment,
        visitor_email="visitor-real-address@example.com",
        visitor_name="زائر حقيقي بالاسم",
    )
    _seed(engine, item)
    svc = ReviewWorkflowService(engine)
    svc.send_as_is("c-pii", REVIEWER)

    ex = list_feedback_examples(engine)[0]

    # No visitor identity fields exist on the dataclass/CSV contract at all.
    assert not hasattr(ex, "visitor_email")
    assert not hasattr(ex, "visitor_name")
    assert not hasattr(ex, "visitor_phone")
    assert "visitor_email" not in FEEDBACK_CSV_COLUMNS
    assert "visitor_name" not in FEEDBACK_CSV_COLUMNS

    # The visitor's own embedded phone number is redacted in comment_text.
    assert "0501234567" not in (ex.comment_text or "")

    out = export_feedback_dataset(engine, tmp_path / "feedback.csv")
    csv_bytes = out.read_bytes()
    assert b"0501234567" not in csv_bytes
    assert b"visitor-real-address@example.com" not in csv_bytes
