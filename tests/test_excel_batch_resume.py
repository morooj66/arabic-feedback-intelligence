"""Resumable batch processing for a large workbook.

Verifies: batch N processes only new/unprocessed rows, a repeated call
continues from the remainder, already-reviewed rows are never reset or
re-sent to Gemini, and progress reaches zero remaining once every
meaningful row has been processed.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from rcs.review.excel_review_flow import (
    process_uploaded_excel_batch,
    source_digest,
    write_fixture_workbook,
)
from rcs.review.identity import ReviewerIdentity
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService
from sqlalchemy.orm import Session


class _RecordingClient:
    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.calls.append(prompt)
        return json.dumps({"suggested_response": f"ردٌ رقم {len(self.calls)} خاص بهذا التعليق."}, ensure_ascii=False)


# Varied real-shaped sentences (praise/suggestion/inquiry/complaint), cycled
# by index — the frozen classifiers are weak and documented-noisy
# (FINAL_HUMAN_TEST_REPORT.md); repeating one near-identical template risks
# every row landing on the same (possibly NO_RESPONSE_RECOMMENDED) outcome,
# which would make it impossible to test the DRAFT_READY/SEND_AS_IS path.
_TEMPLATES = [
    "شكرًا جزيلاً لكم على هذا الجهد الرائع في خدمة اللغة العربية",
    "أقترح إضافة كلمة جميلة إلى المعجم لفائدة الطلاب والباحثين",
    "هل يمكن توضيح معنى كلمة استبانة وأصلها اللغوي من فضلكم؟",
    "لاحظت أن معنى هذه الكلمة غير دقيق ويحتاج إلى تصحيح لغوي",
    "الموقع لا يعمل عند البحث عن بعض الكلمات، أرجو إصلاح المشكلة",
]


def _fixture_with_n_rows(path: Path, n: int) -> Path:
    rows = [
        {
            "original_comment": f"{_TEMPLATES[i % len(_TEMPLATES)]} (تعليق رقم {i})",
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": f"زائر{i}",
            "visitor_email": f"v{i}@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": dt.datetime(2026, 8, 1, 10, 0, 0) + dt.timedelta(minutes=i),
            "platform": "الرياض",
        }
        for i in range(n)
    ]
    return write_fixture_workbook(path, rows)


def _engine(tmp_path: Path):
    engine = get_engine(tmp_path / "batch_flow.sqlite")
    init_db(engine)
    return engine


def test_batch_processes_only_configured_size_per_call(tmp_path):
    source = _fixture_with_n_rows(tmp_path / "big.xlsx", 10)
    engine = _engine(tmp_path)
    client = _RecordingClient()

    summary = process_uploaded_excel_batch(engine, source, batch_size=4, client=client)

    assert summary.total_file_rows == 10
    assert summary.batch_processed == 4
    assert summary.processed_total == 4
    assert summary.remaining == 6


def test_repeated_batches_resume_from_remainder_until_complete(tmp_path):
    source = _fixture_with_n_rows(tmp_path / "big.xlsx", 10)
    engine = _engine(tmp_path)
    client = _RecordingClient()

    first = process_uploaded_excel_batch(engine, source, batch_size=4, client=client)
    assert first.batch_processed == 4
    assert first.remaining == 6
    first_ids = set(first.comment_ids)

    second = process_uploaded_excel_batch(engine, source, batch_size=4, client=client)
    assert second.batch_processed == 4
    assert second.processed_total == 8
    assert second.remaining == 2
    second_ids = set(second.comment_ids)
    assert first_ids.isdisjoint(second_ids)  # never reprocesses the first batch's rows

    third = process_uploaded_excel_batch(engine, source, batch_size=4, client=client)
    assert third.batch_processed == 2  # only 2 rows left
    assert third.processed_total == 10
    assert third.remaining == 0

    # Nothing left to do — a further call is a safe no-op.
    fourth = process_uploaded_excel_batch(engine, source, batch_size=4, client=client)
    assert fourth.batch_processed == 0
    assert fourth.remaining == 0

    with Session(engine) as session:
        assert session.query(ReviewItem).count() == 10  # no duplicate rows across batches


def test_reviewer_decisions_and_gemini_drafts_survive_across_batches(tmp_path):
    source = _fixture_with_n_rows(tmp_path / "big.xlsx", 8)
    engine = _engine(tmp_path)
    client = _RecordingClient()
    before = source_digest(source)

    first = process_uploaded_excel_batch(engine, source, batch_size=4, client=client)
    svc = ReviewWorkflowService(engine)
    reviewer = ReviewerIdentity("rev-batch", "مراجع الدفعات")

    with Session(engine) as session:
        drafted = (
            session.query(ReviewItem)
            .filter(ReviewItem.comment_id.in_(first.comment_ids), ReviewItem.review_status == "DRAFT_READY")
            .all()
        )
        any_first_batch_item = (
            session.query(ReviewItem).filter(ReviewItem.comment_id.in_(first.comment_ids)).first()
        )
    if drafted:
        reviewed_id = drafted[0].comment_id
        reviewed_draft = drafted[0].suggested_final_reply
        svc.send_as_is(reviewed_id, reviewer)
        expected_action, expected_status = "SEND_AS_IS", "APPROVED"
    else:
        # Frozen-classifier variability: no row in this batch happened to
        # land on GENERATE_DRAFT. Still verify persistence-across-batches
        # for whatever state the row actually reached.
        reviewed_id = any_first_batch_item.comment_id
        reviewed_draft = any_first_batch_item.suggested_final_reply
        svc.mark_no_response_needed(reviewed_id, reviewer, reason="لا يستدعي ردًا")
        expected_action, expected_status = "NO_RESPONSE_NEEDED", "NO_REPLY"

    calls_before_next_batch = len(client.calls)
    process_uploaded_excel_batch(engine, source, batch_size=4, client=client)
    process_uploaded_excel_batch(engine, source, batch_size=4, client=client)  # drain to completion

    assert source_digest(source) == before

    with Session(engine) as session:
        item = session.get(ReviewItem, reviewed_id)
        assert item.reviewer_action == expected_action
        assert item.review_status == expected_status
        assert item.suggested_final_reply == reviewed_draft  # never overwritten by a later batch
        original_text = item.original_comment

    # Later batches only draft for the NEW rows they contain — the already
    # reviewed/generated row from batch 1 must never trigger another
    # Gemini call using its own comment text.
    for prompt in client.calls[calls_before_next_batch:]:
        assert original_text not in prompt
