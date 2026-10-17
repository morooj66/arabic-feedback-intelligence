"""
Phase 4 — review session / processing job persistence.

A ReviewSession answers "which uploaded workbook is the reviewer
currently working on" — the missing concept Phase 1-3 never needed
because every acceptance run was manually scoped. A ProcessingJob is a
lightweight, persisted progress record for one background processing run
of a session (see rcs.review.job_runner for the thread that drives it).

Neither table duplicates comment data: session/job rows hold only
identity + status; every count is computed live from review_items.

The actual persistence lives behind ReviewSessionRepository /
ProcessingJobRepository (rcs.review.repositories.contracts), reached
through the persistence bundle (rcs.review.persistence) rather than a
concrete Postgres/Mongo class. Every function below keeps its exact
original signature and behavior — callers (http_api.py, job_runner.py,
service.py, tests) are unchanged — and is now a thin delegation through
that bundle instead of opening a SQLAlchemy Session directly.
"""

from __future__ import annotations

from rcs.review.models import ProcessingJob, ReviewSession
from rcs.review.persistence import resolve_bundle
from rcs.review.repositories.contracts import SessionCounts

# Stage labels — the exact Arabic strings the UI displays, defined once
# here so backend and frontend never drift.
STAGE_UPLOAD = "رفع الملف"
STAGE_PREPARE = "تهيئة البيانات"
STAGE_CLASSIFY = "تصنيف التعليقات"
STAGE_GENERATE = "إنشاء الردود المقترحة"
STAGE_READY = "جاهز للمراجعة"
STAGES_IN_ORDER = (STAGE_UPLOAD, STAGE_PREPARE, STAGE_CLASSIFY, STAGE_GENERATE, STAGE_READY)

# Shown via get_active_session() — includes "completed" so a finished
# file keeps displaying (with its download action) until a NEW upload
# supersedes it as the most recent session.
ACTIVE_SESSION_STATUSES = ("uploaded", "processing", "ready", "completed")

# Eligible to (re)start a background processing job — deliberately
# excludes "completed" (nothing left to process) and "failed" (a human
# should decide whether to re-upload rather than silently retry).
RESUMABLE_SESSION_STATUSES = ("uploaded", "processing", "ready")

__all__ = [
    "STAGE_UPLOAD",
    "STAGE_PREPARE",
    "STAGE_CLASSIFY",
    "STAGE_GENERATE",
    "STAGE_READY",
    "STAGES_IN_ORDER",
    "ACTIVE_SESSION_STATUSES",
    "RESUMABLE_SESSION_STATUSES",
    "SessionCounts",
    "create_session",
    "get_session",
    "get_active_session",
    "set_session_status",
    "compute_session_counts",
    "create_job",
    "update_job",
    "get_job",
    "latest_job_for_session",
]


def create_session(engine, *, original_filename: str, source_file: str, upload_token: str) -> ReviewSession:
    return resolve_bundle(engine).review_sessions.create(
        original_filename=original_filename, source_file=source_file, upload_token=upload_token
    )


def _maybe_mark_completed(engine, row: ReviewSession) -> ReviewSession:
    """Lazy status transition: a "ready" session whose canonical review
    population has zero remaining work is now complete. Checked on every
    read rather than pushed from ReviewWorkflowService, so the reviewer
    workflow stays completely unaware of the session concept (Phase 4
    backend-boundary requirement — Excel/session is an adapter concern,
    not something classification/review/feedback logic depends on)."""
    if row.status != "ready":
        return row
    counts = compute_session_counts(engine, row.source_file)
    if counts.canonical_total > 0 and counts.remaining == 0:
        set_session_status(engine, row.session_id, "completed")
        row.status = "completed"
    return row


def get_session(engine, session_id: str) -> ReviewSession | None:
    row = resolve_bundle(engine).review_sessions.get(session_id)
    if row is not None:
        row = _maybe_mark_completed(engine, row)
    return row


def get_active_session(engine) -> ReviewSession | None:
    """Most recently created session — the single "active file" concept
    the import UI needs. Includes "completed" sessions on purpose: once a
    reviewer finishes a file it should keep showing (in its completed
    view, with a download action) until a NEW upload supersedes it as the
    most recent session — never silently disappear. There is
    deliberately no notion of multiple simultaneous active sessions in
    this phase (smallest safe scope)."""
    row = resolve_bundle(engine).review_sessions.get_active()
    if row is not None:
        row = _maybe_mark_completed(engine, row)
    return row


def set_session_status(engine, session_id: str, status: str) -> None:
    resolve_bundle(engine).review_sessions.set_status(session_id, status)


def compute_session_counts(engine, source_file: str) -> SessionCounts:
    """Read-only aggregation over review_items for one source_file —
    exactly the same canonical semantics used everywhere else in this
    project (a DUPLICATE_CONTENT row is never counted as separate
    reviewable work). See ReviewSessionRepository.compute_counts (both
    backends) for the single-round-trip implementation this delegates to
    — that shape (not fetching every row and counting in Python) is a
    deliberate, previously-fixed performance requirement, not a detail
    to casually change."""
    return resolve_bundle(engine).review_sessions.compute_counts(source_file)


def create_job(engine, *, session_id: str) -> ProcessingJob:
    return resolve_bundle(engine).processing_jobs.create(session_id)


def update_job(
    engine,
    job_id: str,
    *,
    status: str | None = None,
    current_stage: str | None = None,
    rows_done: int | None = None,
    rows_total: int | None = None,
    error_message: str | None = None,
    finished: bool = False,
) -> None:
    resolve_bundle(engine).processing_jobs.update(
        job_id,
        status=status,
        current_stage=current_stage,
        rows_done=rows_done,
        rows_total=rows_total,
        error_message=error_message,
        finished=finished,
    )


def get_job(engine, job_id: str) -> ProcessingJob | None:
    return resolve_bundle(engine).processing_jobs.get(job_id)


def latest_job_for_session(engine, session_id: str) -> ProcessingJob | None:
    return resolve_bundle(engine).processing_jobs.latest_for_session(session_id)
