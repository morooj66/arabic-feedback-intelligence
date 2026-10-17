"""Phase 7 — production-safety fixes: CORS allowlist, Secure/SameSite
cookie behavior, login rate limiting, and bounded transient-DB retry."""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from config import settings
from rcs.review.auth import (
    _rate_limit_attempts,
    check_login_rate_limit,
    create_user,
    record_login_attempt,
)
from rcs.review.db_retry import with_transient_retry
from rcs.review.excel_review_flow import write_fixture_workbook
from rcs.review.excel_upload import save_uploaded_workbook
from rcs.review.http_api import serve_in_thread
from rcs.review.job_runner import resume_orphaned_jobs, start_or_resume_processing
from rcs.review.models import ProcessingJob, ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.sessions import create_job, create_session, get_job, latest_job_for_session, update_job


def _req(url: str, *, method="GET", data=None, headers=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers
    except urllib.error.HTTPError as err:
        return err.code, err.headers


def _engine(tmp_path, name="deploy_safety.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


# --- CORS allowlist -------------------------------------------------------


def test_cors_rejects_an_origin_not_on_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", ("https://reviewer.example.com",))
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        status, headers = _req(f"{base}/api/health", headers={"Origin": "https://evil-arbitrary-site.example"})
        assert status == 200  # the request itself isn't blocked server-side (browsers enforce CORS)
        assert headers.get("Access-Control-Allow-Origin") is None
    finally:
        server.shutdown()


def test_cors_allows_an_origin_on_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", ("https://reviewer.example.com",))
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        status, headers = _req(f"{base}/api/health", headers={"Origin": "https://reviewer.example.com"})
        assert status == 200
        assert headers.get("Access-Control-Allow-Origin") == "https://reviewer.example.com"
        assert headers.get("Access-Control-Allow-Credentials") == "true"
    finally:
        server.shutdown()


def test_local_dev_default_allowlist_permits_the_vite_dev_server_origin(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    import importlib

    import config.settings as settings_module

    importlib.reload(settings_module)
    try:
        assert "http://localhost:5173" in settings_module.ALLOWED_ORIGINS
    finally:
        importlib.reload(settings_module)  # restore whatever the real environment has


# --- Secure/SameSite cookie behavior ---------------------------------------


def test_login_cookie_is_lax_without_secure_on_plain_local_request(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role="REVIEWER")
    server, base = serve_in_thread(engine)
    try:
        status, headers = _req(
            f"{base}/api/login", method="POST", data={"email": "a@x.com", "password": "realpassword123"}
        )
        assert status == 200
        cookie_header = headers.get("Set-Cookie")
        assert "SameSite=Lax" in cookie_header
        assert "Secure" not in cookie_header
        assert "HttpOnly" in cookie_header
    finally:
        server.shutdown()


def test_login_cookie_is_samesite_none_and_secure_behind_a_forwarded_https_proxy(tmp_path):
    """Simulates Render terminating TLS and forwarding X-Forwarded-Proto:
    https — the shape of the real production deployment."""
    engine = _engine(tmp_path)
    create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role="REVIEWER")
    server, base = serve_in_thread(engine)
    try:
        status, headers = _req(
            f"{base}/api/login",
            method="POST",
            data={"email": "a@x.com", "password": "realpassword123"},
            headers={"X-Forwarded-Proto": "https"},
        )
        assert status == 200
        cookie_header = headers.get("Set-Cookie")
        assert "SameSite=None" in cookie_header
        assert "Secure" in cookie_header
        assert "HttpOnly" in cookie_header
    finally:
        server.shutdown()


# --- login rate limiting ----------------------------------------------------


def test_rate_limit_blocks_after_repeated_failures_from_the_same_key():
    key = "test-rate-limit-key-1"
    _rate_limit_attempts.pop(key, None)
    for _ in range(5):
        assert check_login_rate_limit(key) is True
        record_login_attempt(key)
    assert check_login_rate_limit(key) is False


def test_rate_limit_is_per_key_not_global():
    _rate_limit_attempts.pop("key-a", None)
    _rate_limit_attempts.pop("key-b", None)
    for _ in range(5):
        record_login_attempt("key-a")
    assert check_login_rate_limit("key-a") is False
    assert check_login_rate_limit("key-b") is True


def test_login_endpoint_returns_429_after_repeated_failures_from_the_same_client(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role="REVIEWER")
    server, base = serve_in_thread(engine)
    try:
        last_status = None
        for _ in range(8):
            last_status, _ = _req(
                f"{base}/api/login", method="POST", data={"email": "a@x.com", "password": "wrong-password"}
            )
        assert last_status == 429
    finally:
        server.shutdown()


# --- bounded transient-DB retry ---------------------------------------------


def test_transient_retry_retries_a_recognized_connection_error_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OperationalError("SELECT 1", {}, Exception("SSL error: unexpected eof while reading"))
        return "ok"

    assert with_transient_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_transient_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_flaky():
        calls["n"] += 1
        raise OperationalError("SELECT 1", {}, Exception("SSL error: unexpected eof while reading"))

    with pytest.raises(OperationalError):
        with_transient_retry(always_flaky, max_attempts=3)
    assert calls["n"] == 3


def test_transient_retry_never_retries_a_non_transient_error():
    calls = {"n": 0}

    def real_bug():
        calls["n"] += 1
        raise OperationalError("INSERT ...", {}, Exception("UNIQUE constraint failed: users.email"))

    with pytest.raises(OperationalError):
        with_transient_retry(real_bug)
    assert calls["n"] == 1  # never retried — this would fail identically every time


# --- orphaned background job recovery (Phase 7 Step 5) ---------------------


def test_resume_orphaned_jobs_marks_the_stale_job_failed_and_completes_the_work(tmp_path, monkeypatch):
    """Simulates a container restart mid-job: a ProcessingJob left at
    status="running" with no live thread (this test never starts one for
    it) must be marked failed with an honest reason, and the session's
    remaining work must still get done via a fresh job — without
    reprocessing anything already-completed."""
    engine = get_engine(tmp_path / "orphan.sqlite")
    init_db(engine)
    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")

    rows = [
        {
            "original_comment": f"أقترح إضافة كلمة رقم {i} إلى المعجم لفائدة الباحثين",
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": f"زائر{i}",
            "visitor_email": f"visitor{i}@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": dt.datetime(2026, 8, 1, 9, i, 0),
            "platform": "الرياض",
        }
        for i in range(2)
    ]
    source = write_fixture_workbook(tmp_path / "src.xlsx", rows)
    saved = save_uploaded_workbook("src.xlsx", source.read_bytes())
    review_session = create_session(
        engine, original_filename="src.xlsx", source_file=saved.original_filename, upload_token=saved.token
    )

    # A "running" job with no thread — exactly what a fresh process sees
    # after a restart interrupted a previous one.
    orphan_job = create_job(engine, session_id=review_session.session_id)
    update_job(engine, orphan_job.job_id, current_stage="تصنيف التعليقات", rows_done=0, rows_total=2)

    resumed = resume_orphaned_jobs(engine)
    assert resumed == [review_session.session_id]

    with Session(engine) as db:
        stale = db.get(ProcessingJob, orphan_job.job_id)
        assert stale.status == "failed"
        assert stale.error_message == "orphaned_by_process_restart"

    latest = latest_job_for_session(engine, review_session.session_id)
    assert latest.job_id != orphan_job.job_id  # a genuinely new job, not the stale one resurrected

    deadline = time.time() + 60
    job = get_job(engine, latest.job_id)
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.2)
        job = get_job(engine, latest.job_id)
    assert job.status == "succeeded", f"resumed job did not succeed: {job.status} / {job.error_message}"

    with Session(engine) as db:
        n = db.execute(select(ProcessingJob.job_id)).all()
        total_items = db.execute(select(ReviewItem.comment_id)).all()
    assert len(total_items) == 2  # both rows processed exactly once, nothing duplicated
    assert len(n) == 2  # the orphaned job + the one real resumed job — no extra job rows created


def test_resume_orphaned_jobs_is_a_noop_when_nothing_is_running(tmp_path):
    engine = get_engine(tmp_path / "no_orphan.sqlite")
    init_db(engine)
    assert resume_orphaned_jobs(engine) == []


def test_processing_job_failure_stores_a_sanitized_error_message_not_the_raw_exception(tmp_path, monkeypatch, caplog):
    """Regression: an unhandled exception during Excel processing (a DB
    error's SQL statement + bound parameters, an outage message, anything)
    must never reach job.error_message verbatim — that field is returned
    by GET /api/excel/jobs/{id} and rendered directly by the frontend's
    Import page, so a raw exception there previously leaked internal
    schema details and visitor PII straight into the admin-facing UI.
    Only the exception type name is reviewer/admin-facing; the full
    exception goes to the server log only, same convention already used
    for audit-trace failures (rcs.response_intelligence.orchestrator)."""
    import logging

    engine = get_engine(tmp_path / "sanitized_error.sqlite")
    init_db(engine)
    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")

    rows = [
        {
            "original_comment": "أقترح إضافة كلمة إلى المعجم",
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "زائر",
            "visitor_email": "leaked-visitor@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": dt.datetime(2026, 8, 1, 9, 0, 0),
            "platform": "الرياض",
        }
    ]
    source = write_fixture_workbook(tmp_path / "src2.xlsx", rows)
    saved = save_uploaded_workbook("src2.xlsx", source.read_bytes())
    review_session = create_session(
        engine, original_filename="src2.xlsx", source_file=saved.original_filename, upload_token=saved.token
    )

    secret_looking_text = "leaked-visitor@example.com /secret/conn synthetic DB outage"

    def _boom(*args, **kwargs):
        raise RuntimeError(secret_looking_text)

    monkeypatch.setattr("rcs.review.job_runner.process_uploaded_excel_batch", _boom)

    with caplog.at_level(logging.WARNING, logger="rcs.review.job_runner"):
        job_id = start_or_resume_processing(engine, review_session.session_id)
        deadline = time.time() + 30
        job = get_job(engine, job_id)
        while job.status == "running" and time.time() < deadline:
            time.sleep(0.1)
            job = get_job(engine, job_id)

    assert job.status == "failed"
    assert job.error_message == "processing_failed:RuntimeError"
    assert secret_looking_text not in job.error_message

    assert len(caplog.records) == 1
    logged = caplog.records[0].getMessage()
    assert "RuntimeError" in logged
    assert secret_looking_text not in logged
