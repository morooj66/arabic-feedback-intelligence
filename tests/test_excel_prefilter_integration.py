"""Excel E2E: deterministic prefilter -> frozen classification -> rg-v7 generation.

Uses a scripted GenAI-like client (no network) so these stay fast/offline,
but exercises the SAME production code path (process_uploaded_excel) that
the live Gemini run uses — only the client is swapped, per
apply_live_generation's documented seam.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from rcs.review.excel_review_flow import process_uploaded_excel, source_digest, write_fixture_workbook
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.identity import ReviewerIdentity
from rcs.review.service import ReviewWorkflowService
from sqlalchemy.orm import Session


class _RecordingClient:
    """Records every prompt it's asked to generate for, so tests can
    assert Gemini was never called for a deterministically-excluded row."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.calls.append(prompt)
        return json.dumps({"suggested_response": "ردٌ تجريبي مخصص لهذا التعليق تحديدًا."}, ensure_ascii=False)


def _engine(tmp_path: Path):
    engine = get_engine(tmp_path / "prefilter_flow.sqlite")
    init_db(engine)
    return engine


def _fixture(path: Path) -> Path:
    return write_fixture_workbook(
        path,
        [
            {
                # Matches settings.KNOWN_JUNK_STRINGS verbatim -> exact_match_known_bot_string
                "original_comment": "jhnjkn njnun Phone Number: 111111111111j",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "زائر",
                "visitor_email": "junk@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": dt.datetime(2026, 8, 1, 10, 0, 0),
                "platform": "الرياض",
            },
            {
                "original_comment": "شكرًا جزيلاً على هذا المعجم الرائع والمفيد جدًا لطلابنا",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "زائر",
                "visitor_email": "real@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": dt.datetime(2026, 8, 2, 10, 0, 0),
                "platform": "الرياض",
            },
        ],
    )


def _keyboard_mash_fixture(path: Path) -> Path:
    return write_fixture_workbook(
        path,
        [{
            "original_comment": "asdf qwer zxcv",
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "زائر تجريبي",
            "visitor_email": "synthetic@example.invalid",
            "original_type": "رسالة تواصل",
            "comment_created_at": dt.datetime(2026, 10, 17, 10, 0, 0),
            "platform": "الرياض",
        }],
    )


def test_keyboard_mash_is_excluded_before_classification_and_generation(tmp_path):
    source = _keyboard_mash_fixture(tmp_path / "keyboard-mash.xlsx")
    engine = _engine(tmp_path)
    client = _RecordingClient()

    summary = process_uploaded_excel(engine, source, limit=10, client=client)

    assert summary.excluded_rows == 1
    assert summary.classification_processed == 0
    assert summary.generated_drafts == 0
    assert client.calls == []
    with Session(engine) as session:
        row = session.query(ReviewItem).one()
        assert row.prefilter_reason == "keyboard_mash_sequence"
        assert row.review_status == "EXCLUDED"
        assert row.routing_decision == "EXCLUDE"
        assert row.ai_primary_intent is None


def test_deterministic_junk_is_excluded_before_classification_and_never_reaches_gemini(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    client = _RecordingClient()

    summary = process_uploaded_excel(engine, source, limit=10, client=client)

    assert summary.excluded_rows == 1
    assert summary.classification_processed == 1  # only the real comment was sent to the frozen classifiers

    with Session(engine) as session:
        junk = (
            session.query(ReviewItem)
            .filter(ReviewItem.original_comment == "jhnjkn njnun Phone Number: 111111111111j")
            .one()
        )
        assert junk.is_prefiltered_junk is True
        assert junk.prefilter_reason == "exact_match_known_bot_string"
        assert junk.review_status == "EXCLUDED"
        assert junk.primary_intent == "SPAM"
        assert junk.routing_decision == "EXCLUDE"
        assert junk.suggested_final_reply is None
        assert junk.ai_primary_intent is None  # never touched by the frozen classifiers at all

        real = (
            session.query(ReviewItem)
            .filter(ReviewItem.original_comment.like("شكرًا جزيلاً%"))
            .one()
        )
        # Whatever the frozen (weak) classifiers decided for the real
        # comment, it must have been routed to exactly one terminal
        # generation state — never left stuck, never excluded like junk.
        assert real.review_status in ("DRAFT_READY", "REVIEW_ONLY", "NO_RESPONSE_RECOMMENDED")
        assert real.primary_intent is not None
        assert real.primary_intent != "SPAM"

    # The recording client must never have been asked to draft using a
    # prompt containing the junk row's text (it may be called zero or one
    # times for the real row, depending on the frozen classifiers'
    # eligibility routing for that row — either is fine here).
    for prompt in client.calls:
        assert "jhnjkn njnun" not in prompt


def test_normal_valid_comments_reach_classification_and_generation(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    client = _RecordingClient()

    summary = process_uploaded_excel(engine, source, limit=10, client=client)

    assert summary.total_rows == 2
    assert summary.inserted_rows == 2
    assert summary.classification_processed == 1
    assert summary.failed_rows == 0
    # The real (non-junk) row must reach exactly one eligibility outcome —
    # which one depends on the frozen classifiers' own (documented-weak)
    # prediction, not on this integration wiring.
    assert summary.generated_drafts + summary.no_response_recommended + summary.review_only == 1


def test_repeated_import_is_idempotent_and_preserves_reviewer_decision(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    before = source_digest(source)
    client = _RecordingClient()

    process_uploaded_excel(engine, source, limit=10, client=client)
    svc = ReviewWorkflowService(engine)
    reviewer = ReviewerIdentity("rev-1", "مراجع")

    with Session(engine) as session:
        real_item = (
            session.query(ReviewItem).filter(ReviewItem.original_comment.like("شكرًا جزيلاً%")).one()
        )
        real_id, real_status = real_item.comment_id, real_item.review_status

    # Take whichever reviewer action the frozen classifiers' actual
    # eligibility outcome allows — the point under test is that the
    # decision survives a duplicate import, not which outcome occurred.
    if real_status == "DRAFT_READY":
        detail = svc.send_as_is(real_id, reviewer)
    elif real_status == "REVIEW_ONLY":
        detail = svc.edit_and_send(real_id, reviewer, "رد يدوي من المراجع بعد المعاينة.")
    else:
        detail = svc.mark_no_response_needed(real_id, reviewer, reason="لا يستدعي ردًا")

    calls_before_second_run = len(client.calls)
    second = process_uploaded_excel(engine, source, limit=10, client=client)

    assert source_digest(source) == before
    assert second.inserted_rows == 0
    assert second.duplicate_rows == 2
    assert second.excluded_rows == 0
    assert second.classification_processed == 0
    assert second.generated_drafts == 0
    assert len(client.calls) == calls_before_second_run  # no new Gemini calls on the duplicate import

    with Session(engine) as session:
        assert session.query(ReviewItem).count() == 2  # no duplicate rows created
        item = session.get(ReviewItem, real_id)
        assert item.reviewer_action == detail.reviewer_action
        assert item.review_status == detail.review_status
        assert item.final_reply == detail.final_response


def test_source_workbook_untouched_by_full_pipeline(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    before = source_digest(source)
    process_uploaded_excel(engine, source, limit=10, client=_RecordingClient())
    assert source_digest(source) == before
