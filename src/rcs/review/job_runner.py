"""
Phase 4 — the smallest background job mechanism that gives real,
non-fake processing progress without an external queue (no Celery, no
Redis, no message bus): a single in-process Python thread per active
session, looping the EXISTING rcs.review.excel_review_flow.
process_uploaded_excel_batch() until the file is fully processed,
persisting progress to a ProcessingJob row after every batch.

Resumability: process_uploaded_excel_batch() is already idempotent and
comment_id-stable (Phase 4 batch/resume work) — starting a fresh thread
for a session that was interrupted (browser closed, backend restarted)
simply continues from wherever review_items state actually is. This
module adds an in-memory registry only to avoid starting two threads for
the same session concurrently within one running server process; it is
not a source of truth (the DB rows are) and is safely rebuilt as empty
on every process restart.
"""

from __future__ import annotations

import logging
import threading

from rcs.review.excel_review_flow import DEFAULT_BATCH_SIZE, process_uploaded_excel_batch
from rcs.review.excel_upload import resolve_uploaded_workbook
from rcs.review.persistence import resolve_bundle
from rcs.review.sessions import (
    STAGE_CLASSIFY,
    STAGE_GENERATE,
    STAGE_PREPARE,
    STAGE_READY,
    create_job,
    get_session,
    set_session_status,
    update_job,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active_threads: dict[str, threading.Thread] = {}  # session_id -> Thread


def _run(engine, session_id: str, job_id: str, *, batch_size: int) -> None:
    try:
        review_session = get_session(engine, session_id)
        if review_session is None:
            update_job(engine, job_id, status="failed", error_message="session_not_found", finished=True)
            return
        source_path = resolve_uploaded_workbook(review_session.upload_token)

        set_session_status(engine, session_id, "processing")
        update_job(engine, job_id, current_stage=STAGE_PREPARE)

        while True:
            update_job(engine, job_id, current_stage=STAGE_CLASSIFY)
            summary = process_uploaded_excel_batch(engine, source_path, batch_size=batch_size)
            update_job(
                engine,
                job_id,
                current_stage=STAGE_GENERATE,
                rows_done=summary.processed_total,
                # rows that are not real comments (too short) are expected skips, not pending work: the bar must be able to reach 100%
                rows_total=summary.total_file_rows - summary.invalid_rows_skipped,
            )
            if summary.remaining <= 0:
                break
            if summary.batch_processed == 0:
                # Nothing new was picked up this iteration even though
                # remaining > 0 — stop rather than spin forever; the
                # session stays "processing" for a human to investigate,
                # never silently marked ready with incomplete data.
                update_job(
                    engine,
                    job_id,
                    status="failed",
                    error_message="no_progress_stopped",
                    finished=True,
                )
                return

        update_job(engine, job_id, status="succeeded", current_stage=STAGE_READY, finished=True)
        set_session_status(engine, session_id, "ready")
    except Exception as exc:  # noqa: BLE001 — job must record the failure, never crash silently
        # The full exception (e.g. a DB error's SQL statement and bound
        # parameters, which can include visitor PII) goes to the server
        # log only — job.error_message is reviewer/admin-facing (returned
        # by GET /api/excel/jobs/{id} and shown verbatim by the frontend),
        # so it carries just the exception type name, matching the
        # sanitized-message convention already used for audit-trace
        # failures (rcs.response_intelligence.orchestrator).
        logger.warning(
            "excel processing job failed: session_id=%s job_id=%s error_type=%s",
            session_id,
            job_id,
            type(exc).__name__,
            exc_info=True,
        )
        update_job(engine, job_id, status="failed", error_message=f"processing_failed:{type(exc).__name__}", finished=True)
        set_session_status(engine, session_id, "uploaded")  # safe to retry from "start processing" again
    finally:
        with _lock:
            _active_threads.pop(session_id, None)


def start_or_resume_processing(engine, session_id: str, *, batch_size: int = DEFAULT_BATCH_SIZE) -> str:
    """Starts a new background job for this session, unless one is
    already actively running in this process (returns that job's id
    instead). Always safe to call repeatedly — this is exactly the
    "متابعة المعالجة" resume path after an interruption."""
    with _lock:
        existing_thread = _active_threads.get(session_id)
        if existing_thread is not None and existing_thread.is_alive():
            from rcs.review.sessions import latest_job_for_session

            job = latest_job_for_session(engine, session_id)
            if job is not None and job.status == "running":
                return job.job_id

        job = create_job(engine, session_id=session_id)
        thread = threading.Thread(
            target=_run, args=(engine, session_id, job.job_id), kwargs={"batch_size": batch_size}, daemon=True
        )
        _active_threads[session_id] = thread
        thread.start()
        return job.job_id


def resume_orphaned_jobs(engine) -> list[str]:
    """Phase 7 startup recovery.

    _active_threads is rebuilt empty on every process start (by
    design — it's a registry, not a source of truth, see module
    docstring). On a host that can restart the process mid-job (a
    container platform redeploy, a crash), any ProcessingJob a previous
    process left at status="running" is therefore, unconditionally, an
    orphan the moment a NEW process starts: no thread anywhere will ever
    finish it, and the frontend would poll a "running" job forever.

    Called once at server startup: marks each such job "failed" with an
    honest reason (never silently left ambiguous), then starts a fresh
    resumable job for the same session. process_uploaded_excel_batch()
    is idempotent and comment_id-stable, so this never reprocesses
    already-completed rows — only the genuinely remaining ones continue.
    Best-effort per session: one session's resolution failure (e.g. its
    uploaded file is gone after an ephemeral-disk restart) is logged and
    skipped, never crashes startup for every other session.
    """
    orphaned = resolve_bundle(engine).processing_jobs.list_running()
    session_ids = [job.session_id for job in orphaned]
    for job in orphaned:
        update_job(engine, job.job_id, status="failed", error_message="orphaned_by_process_restart", finished=True)

    resumed: list[str] = []
    for session_id in session_ids:
        try:
            start_or_resume_processing(engine, session_id)
            resumed.append(session_id)
        except Exception as exc:  # noqa: BLE001 — one bad session must never block the others or crash startup
            print(f"resume_orphaned_jobs: could not resume session {session_id}: {type(exc).__name__}: {exc}")
    return resumed
