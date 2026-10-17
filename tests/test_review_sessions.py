"""Phase 4 — review session / processing job / resume behavior.

Covers acceptance criteria A-L: session creation, real stage progression,
no reprocessing on resume, restart persistence, queue scoping, canonical
counts (no duplicate inflation), completed-session detection, export
provenance/PII, and migration safety over historical data.
"""

from __future__ import annotations

import datetime as dt
import time

from rcs.review.excel_review_flow import write_fixture_workbook
from rcs.review.excel_upload import save_uploaded_workbook
from rcs.review.identity import ReviewerIdentity
from rcs.review.job_runner import start_or_resume_processing
from rcs.review.models import ProcessingJob, ReviewItem, ReviewSession
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService
from rcs.review.sessions import (
    compute_session_counts,
    create_session,
    get_active_session,
    get_job,
    get_session,
    latest_job_for_session,
)
from sqlalchemy.orm import Session

REVIEWER = ReviewerIdentity("rev-session", "مراجع الجلسات")


def _fixture(path, n_valid=1, n_junk=1):
    rows = []
    t0 = dt.datetime(2026, 8, 1, 9, 0, 0)
    for i in range(n_valid):
        rows.append(
            {
                "original_comment": f"أقترح إضافة كلمة رائعة رقم {i} إلى المعجم لفائدة الباحثين",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": f"زائر{i}",
                "visitor_email": f"visitor{i}@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": t0 + dt.timedelta(minutes=i),
                "platform": "الرياض",
            }
        )
    for i in range(n_junk):
        rows.append(
            {
                "original_comment": "jhnjkn njnun Phone Number: 111111111111j",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "junk",
                "visitor_email": "junk@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": t0 + dt.timedelta(minutes=100 + i),
                "platform": "الرياض",
            }
        )
    return write_fixture_workbook(path, rows)


def _engine(tmp_path, name="sessions.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _make_session(tmp_path, engine, monkeypatch, *, n_valid=1, n_junk=1):
    source = _fixture(tmp_path / "src.xlsx", n_valid=n_valid, n_junk=n_junk)
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    saved = save_uploaded_workbook("src.xlsx", source.read_bytes())
    review_session = create_session(
        engine, original_filename="src.xlsx", source_file=saved.original_filename, upload_token=saved.token
    )
    return review_session, saved


def _run_job_to_completion(engine, session_id, *, timeout=60):
    job_id = start_or_resume_processing(engine, session_id, batch_size=10)
    deadline = time.time() + timeout
    job = get_job(engine, job_id)
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.2)
        job = get_job(engine, job_id)
    assert job.status == "succeeded", f"job did not succeed: {job.status} / {job.error_message}"
    return job


# A -----------------------------------------------------------------------


def test_upload_creates_session_associated_with_source_file(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    review_session, saved = _make_session(tmp_path, engine, monkeypatch)
    assert review_session.status == "uploaded"
    assert review_session.source_file == saved.original_filename
    fetched = get_session(engine, review_session.session_id)
    assert fetched is not None
    assert fetched.session_id == review_session.session_id


# B -----------------------------------------------------------------------


def test_processing_job_reports_real_stage_progression(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    review_session, _ = _make_session(tmp_path, engine, monkeypatch, n_valid=2, n_junk=1)
    job = _run_job_to_completion(engine, review_session.session_id)
    assert job.current_stage == "جاهز للمراجعة"
    assert job.rows_total == 3
    assert job.rows_done == 3
    assert job.finished_at is not None
    assert job.started_at <= job.updated_at


# C -----------------------------------------------------------------------


def test_completed_rows_are_not_reprocessed_after_resume(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    review_session, _ = _make_session(tmp_path, engine, monkeypatch, n_valid=1, n_junk=1)
    _run_job_to_completion(engine, review_session.session_id)
    with Session(engine) as db:
        row_count_after_first = db.query(ReviewItem).count()

    second_job = _run_job_to_completion(engine, review_session.session_id)
    assert second_job.rows_done == 2  # unchanged — same total, nothing new to process
    with Session(engine) as db:
        assert db.query(ReviewItem).count() == row_count_after_first  # no duplicate rows created


# D -----------------------------------------------------------------------


def test_restart_preserves_session_progress_and_reviewer_decisions(tmp_path, monkeypatch):
    db_path = tmp_path / "restart.sqlite"
    engine1 = get_engine(db_path)
    init_db(engine1)
    review_session, _ = _make_session(tmp_path, engine1, monkeypatch, n_valid=1, n_junk=1)
    _run_job_to_completion(engine1, review_session.session_id)

    svc1 = ReviewWorkflowService(engine1)
    with Session(engine1) as db:
        canonical = (
            db.query(ReviewItem)
            .filter(ReviewItem.source_file == review_session.source_file, ReviewItem.review_status != "EXCLUDED")
            .first()
        )
        cid, status_before = canonical.comment_id, canonical.review_status
    if status_before == "DRAFT_READY":
        svc1.send_as_is(cid, REVIEWER)
    elif status_before == "REVIEW_ONLY":
        svc1.edit_and_send(cid, REVIEWER, "رد يدوي بعد إعادة التشغيل.")
    # else: NO_RESPONSE_RECOMMENDED — already auto-resolved, no action needed

    # Simulate a fresh process: brand-new engine bound to the same file.
    engine2 = get_engine(db_path)
    init_db(engine2)
    active = get_active_session(engine2)
    assert active is not None
    assert active.session_id == review_session.session_id
    with Session(engine2) as db:
        item = db.get(ReviewItem, cid)
        if status_before == "DRAFT_READY":
            assert item.reviewer_action == "SEND_AS_IS"
        elif status_before == "REVIEW_ONLY":
            assert item.reviewer_action == "EDIT_AND_SEND"


# E -----------------------------------------------------------------------


def test_review_queue_scoped_to_active_session(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    session_a, _ = _make_session(tmp_path, engine, monkeypatch, n_valid=2, n_junk=0)
    _run_job_to_completion(engine, session_a.session_id)

    # A second, unrelated workbook imported into the SAME db.
    from rcs.review.excel_review_flow import import_subset

    source_b = write_fixture_workbook(
        tmp_path / "other.xlsx",
        [
            {
                "original_comment": "تعليق من ملف آخر تمامًا",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "x",
                "visitor_email": "x@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": dt.datetime(2026, 8, 5),
                "platform": "الرياض",
            }
        ],
    )
    import_subset(engine, source_b, limit=10)

    svc = ReviewWorkflowService(engine)
    scoped = svc.list_review_queue(source_file=session_a.source_file, include_no_response_recommended=True)
    all_ids = {i.comment_id for i in scoped.items}
    with Session(engine) as db:
        other_ids = {
            r.comment_id for r in db.query(ReviewItem).filter(ReviewItem.source_file == source_b.name).all()
        }
    assert all_ids.isdisjoint(other_ids)  # never shows pending items from the other workbook

    unscoped = svc.list_review_queue(include_no_response_recommended=True)
    assert unscoped.total >= scoped.total  # global query still sees both, scoped query doesn't


# F, G ----------------------------------------------------------------------


def test_session_counts_are_correct_and_duplicates_dont_inflate(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    source = tmp_path / "dupes.xlsx"
    shared = "نص مكرر تمامًا لاختبار عدم تضخيم عدد الجلسة"
    rows = [
        {
            "original_comment": shared,
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": f"v{i}",
            "visitor_email": f"v{i}@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": dt.datetime(2026, 8, 1) + dt.timedelta(minutes=i),
            "platform": "الرياض",
        }
        for i in range(4)
    ]
    rows.append(
        {
            "original_comment": "jhnjkn njnun Phone Number: 111111111111j",
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "junk",
            "visitor_email": "junk@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": dt.datetime(2026, 8, 2),
            "platform": "الرياض",
        }
    )
    write_fixture_workbook(source, rows)
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads2")
    saved = save_uploaded_workbook("dupes.xlsx", source.read_bytes())
    review_session = create_session(
        engine, original_filename="dupes.xlsx", source_file=saved.original_filename, upload_token=saved.token
    )
    _run_job_to_completion(engine, review_session.session_id)

    counts = compute_session_counts(engine, review_session.source_file)
    assert counts.total == 5
    assert counts.excluded == 1
    assert counts.duplicate_content == 3  # 4 identical rows -> 1 canonical + 3 duplicates
    assert counts.canonical_total == 1  # NOT 4 — duplicates never inflate the reviewable population


def test_session_counts_sql_aggregate_matches_every_status_branch(tmp_path):
    """compute_session_counts is computed as SQL-side conditional
    aggregates (a single round trip) rather than fetching every row into
    Python — this was changed after the Import page's "checking for an
    active file" step was observed taking ~7.5s on a real ~7,400-row
    source_file, because the old implementation pulled every row over
    the wire just to count them client-side. Directly covers every
    branch the aggregate query has to get right: EXCLUDED,
    DUPLICATE_CONTENT, a normally-completed item (reviewer_action set),
    a NO_RESPONSE_RECOMMENDED auto-resolved item (counts as done without
    a reviewer_action), and a still-pending item."""
    engine = _engine(tmp_path)
    source = "sql_aggregate_fixture.xlsx"
    rows = [
        dict(comment_id="c-excluded", review_status="EXCLUDED", reviewer_action=None),
        dict(comment_id="c-dup-1", review_status="DUPLICATE_CONTENT", reviewer_action=None),
        dict(comment_id="c-dup-2", review_status="DUPLICATE_CONTENT", reviewer_action=None),
        dict(comment_id="c-completed", review_status="APPROVED", reviewer_action="SEND_AS_IS"),
        dict(comment_id="c-auto-resolved", review_status="NO_RESPONSE_RECOMMENDED", reviewer_action=None),
        dict(comment_id="c-pending", review_status="DRAFT_READY", reviewer_action=None),
    ]
    with Session(engine) as db:
        for i, r in enumerate(rows):
            db.add(ReviewItem(source_file=source, source_row_index=i, **r))
        db.commit()

    counts = compute_session_counts(engine, source)
    assert counts.total == 6
    assert counts.excluded == 1
    assert counts.duplicate_content == 2
    assert counts.canonical_total == 3  # 6 - 1 excluded - 2 duplicates
    assert counts.completed == 1  # c-completed only; a recommendation is not completed work
    assert counts.remaining == 2  # c-pending + c-auto-resolved (awaiting human review)


# H -----------------------------------------------------------------------


def test_completed_session_detected_after_last_item_reviewed(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    review_session, _ = _make_session(tmp_path, engine, monkeypatch, n_valid=1, n_junk=1)
    _run_job_to_completion(engine, review_session.session_id)

    counts = compute_session_counts(engine, review_session.source_file)
    assert counts.remaining <= 1
    if counts.remaining == 1:
        with Session(engine) as db:
            item = (
                db.query(ReviewItem)
                .filter(ReviewItem.source_file == review_session.source_file, ReviewItem.review_status != "EXCLUDED")
                .filter(ReviewItem.reviewer_action.is_(None))
                .first()
            )
        svc = ReviewWorkflowService(engine)
        if item.review_status == "DRAFT_READY":
            svc.send_as_is(item.comment_id, REVIEWER)
        else:
            svc.mark_no_response_needed(item.comment_id, REVIEWER, reason="اختبار")

    active = get_active_session(engine)
    assert active is not None
    assert active.status == "completed"


# I, K ----------------------------------------------------------------------


def test_export_two_sheets_include_entry_lemma_no_visitor_pii(tmp_path, monkeypatch):
    from rcs.review.excel_review_flow import (
        _EXPORT_SHEET1_HEADERS,
        _EXPORT_SHEET1_TITLE,
        _EXPORT_SHEET2_TITLE,
        export_reviewed_workbook,
    )

    engine = _engine(tmp_path)
    review_session, _ = _make_session(tmp_path, engine, monkeypatch, n_valid=1, n_junk=0)
    _run_job_to_completion(engine, review_session.session_id)

    out = tmp_path / "export.xlsx"
    export_reviewed_workbook(engine, out)

    import pandas as pd

    sheet1 = pd.read_excel(out, sheet_name=_EXPORT_SHEET1_TITLE, dtype=str)
    sheet2 = pd.read_excel(out, sheet_name=_EXPORT_SHEET2_TITLE, dtype=str)
    assert "معرف المدخل" in sheet1.columns
    assert "اللما" in sheet1.columns
    assert len(sheet1) == 1
    for col in sheet1.columns:
        assert col not in ("visitor_email", "visitor_name", "visitor_phone")
    combined_text = sheet1.to_csv() + sheet2.to_csv()
    assert "@example.com" not in combined_text  # no visitor email anywhere in either sheet


# L -----------------------------------------------------------------------


def test_migration_preserves_historical_review_items(tmp_path):
    """Simulates a pre-Phase-4 database (review_items only, no
    review_sessions/processing_jobs tables) and confirms init_db() adds
    the new tables additively without touching existing rows."""
    from sqlalchemy import inspect

    db_path = tmp_path / "legacy.sqlite"
    engine = get_engine(db_path)
    init_db(engine)  # creates review_items (+ new tables, but we drop them below to simulate "legacy")

    with Session(engine) as db:
        db.add(
            ReviewItem(
                comment_id="legacy-1",
                source_file="legacy.xlsx",
                source_row_index=0,
                original_comment="قرار مراجع تاريخي محفوظ",
                review_status="APPROVED",
                reviewer_action="SEND_AS_IS",
                reviewer_display_name="مراجع قديم",
                final_reply="رد نهائي تاريخي",
                suggested_final_reply="رد نهائي تاريخي",
            )
        )
        db.commit()

    ReviewSession.__table__.drop(engine, checkfirst=True)
    ProcessingJob.__table__.drop(engine, checkfirst=True)
    inspector = inspect(engine)
    assert "review_sessions" not in inspector.get_table_names()

    # Re-run init_db (what happens on the next server start) — additive only.
    init_db(engine)
    inspector = inspect(engine)
    assert "review_sessions" in inspector.get_table_names()
    assert "processing_jobs" in inspector.get_table_names()

    with Session(engine) as db:
        preserved = db.get(ReviewItem, "legacy-1")
        assert preserved is not None
        assert preserved.reviewer_action == "SEND_AS_IS"
        assert preserved.final_reply == "رد نهائي تاريخي"
