"""Phase 4.1 cleanup — Dashboard active-session scoping and reviewer-facing
PII redaction.

Covers:
  A. active-session analytics counts exclude historical/other-file rows
  B. queue (dashboard "attention" list) excludes rows from other source files
  D. reviewer-facing comment text redacts phone numbers
  E. persisted original_comment is never modified by redaction
  F. urgent_pending is derived from real review_priority data, not a fixed
     value — a session with no priority-1 rows reports urgent_pending == 0
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from rcs.review.dto import to_detail, to_queue_item
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService


def _item(**overrides) -> ReviewItem:
    short_id = overrides.pop("comment_id")
    defaults = dict(
        comment_id=short_id,
        source_file="active.xlsx",
        source_row_index=0,
        source_reference="active.xlsx:0",
        original_comment="تعليق حقيقي للاختبار",
        review_status="DRAFT_READY",
        review_priority=3,
        priority_reason="routine",
        ingested_at=dt.datetime(2026, 8, 20, 10, 0, 0),
        row_version=1,
        content_dedupe_key="key-" + short_id,
        dedupe_canonical_id=short_id,
    )
    defaults.update(overrides)
    return ReviewItem(**defaults)


def _engine(tmp_path, name="scoping.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def test_active_session_analytics_excludes_historical_rows(tmp_path):
    """A: analytics scoped to source_file must not count another file's
    (or historical, unscoped) pending rows."""
    engine = _engine(tmp_path)
    with Session(engine) as session:
        session.add_all(
            [
                _item(comment_id="a1", source_file="active.xlsx", review_status="DRAFT_READY", review_priority=3),
                _item(comment_id="a2", source_file="active.xlsx", review_status="DRAFT_READY", review_priority=1, priority_reason="complaint_or_technical"),
                _item(comment_id="h1", source_file="historical.xlsx", review_status="DRAFT_READY", review_priority=1, priority_reason="complaint_or_technical"),
                _item(comment_id="h2", source_file="historical.xlsx", review_status="DRAFT_READY", review_priority=1, priority_reason="complaint_or_technical"),
            ]
        )
        session.commit()

    svc = ReviewWorkflowService(engine)
    scoped = svc.operational_analytics(source_file="active.xlsx")
    assert scoped["kpis"]["pending"] == 2
    assert scoped["kpis"]["urgent_pending"] == 1

    unscoped = svc.operational_analytics()
    assert unscoped["kpis"]["pending"] == 4
    assert unscoped["kpis"]["urgent_pending"] == 3


def test_dashboard_queue_excludes_other_source_files(tmp_path):
    """B: the same source_file-scoped queue call the dashboard's
    'attention' list uses must never surface another workbook's rows."""
    engine = _engine(tmp_path)
    with Session(engine) as session:
        session.add_all(
            [
                _item(comment_id="a1", source_file="active.xlsx"),
                _item(comment_id="a2", source_file="active.xlsx"),
                _item(comment_id="h1", source_file="historical.xlsx"),
            ]
        )
        session.commit()

    svc = ReviewWorkflowService(engine)
    page = svc.list_review_queue(source_file="active.xlsx", limit=5)
    ids = {i.comment_id for i in page.items}
    assert ids == {"a1", "a2"}
    assert "h1" not in ids


def test_urgent_pending_reflects_real_priority_not_a_fixed_value(tmp_path):
    """F: with zero priority-1 rows in scope, urgent_pending must be 0 —
    proving the KPI is derived from review_priority, not a hardcoded/demo
    badge that always shows a nonzero count."""
    engine = _engine(tmp_path)
    with Session(engine) as session:
        session.add_all(
            [
                _item(comment_id="a1", review_status="DRAFT_READY", review_priority=3, priority_reason="routine"),
                _item(comment_id="a2", review_status="DRAFT_READY", review_priority=2, priority_reason="inquiry_or_review_only"),
            ]
        )
        session.commit()

    svc = ReviewWorkflowService(engine)
    scoped = svc.operational_analytics(source_file="active.xlsx")
    assert scoped["kpis"]["urgent_pending"] == 0
    assert scoped["kpis"]["pending"] == 2


def test_reviewer_facing_comment_text_redacts_phone_number(tmp_path):
    """D + E: the phone number must not appear in the reviewer-facing
    queue preview or item detail, while the persisted DB row keeps the
    verbatim original text untouched."""
    engine = _engine(tmp_path)
    raw_text = "الرجاء التواصل معي على 0555123456 لمناقشة اقتراحي بخصوص المعجم"
    with Session(engine) as session:
        session.add(
            _item(
                comment_id="p1",
                original_comment=raw_text,
                review_status="DRAFT_READY",
            )
        )
        session.commit()

    svc = ReviewWorkflowService(engine)

    page = svc.list_review_queue(source_file="active.xlsx", limit=5)
    (queue_item,) = page.items
    assert "0555123456" not in (queue_item.comment_preview or "")

    detail = svc.get_review_item("p1")
    assert "0555123456" not in (detail.original_comment or "")
    assert detail.original_comment  # redaction must not blank the text out

    with Session(engine) as session:
        row = session.get(ReviewItem, "p1")
        assert row is not None
        assert row.original_comment == raw_text  # verbatim, untouched in the DB


def test_dto_redaction_helpers_do_not_mutate_input(tmp_path):
    """D + E at the DTO layer directly: to_queue_item/to_detail return a
    redacted copy without touching the ORM instance's own attribute."""
    raw_text = "اتصل بي على الرقم 0501234567 من فضلك"
    item = _item(comment_id="d1", original_comment=raw_text, review_status="DRAFT_READY")

    queue_dto = to_queue_item(item)
    detail_dto = to_detail(item)

    assert "0501234567" not in (queue_dto.comment_preview or "")
    assert "0501234567" not in (detail_dto.original_comment or "")
    # the ORM object's own attribute is untouched by the DTO conversion
    assert item.original_comment == raw_text
