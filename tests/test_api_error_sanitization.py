"""Production API responses must never carry internal exception text, stack traces, database details or secrets - while curated validation messages and the existing status codes survive.
Every test plants a recognisable 'canary' inside the exception message and asserts it appears neither in the response nor in the logs."""
from __future__ import annotations

import http.client
import json
import logging
import re

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from rcs.review import http_api
from rcs.review.auth import create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService
from rcs.review.workflow import InvalidTransition, VersionConflict

CANARY = "CANARY-SECRET-hunter2-SELECT password_hash FROM users WHERE email='victim@corp.example'"
BANNED_IN_BODY = ("CANARY", "hunter2", "SELECT", "password_hash", "victim@corp", "Traceback", "File \"", ".py", "sqlalchemy", "psycopg", "/Users/", "RuntimeError", "OperationalError", "IntegrityError")


@pytest.fixture(params=["development", "production"])
def server(request, tmp_path, monkeypatch):
    monkeypatch.setenv("RCS_ENV", request.param)
    monkeypatch.setattr(http_api.settings, "ALLOWED_ORIGINS", ("https://app.test",))
    engine = get_engine(tmp_path / "e.sqlite"); init_db(engine)
    create_user(engine, email="admin@x.org", password="admin-password-1", display_name="A", role="ADMIN")
    rev = create_user(engine, email="rev@x.org", password="rev-password-1", display_name="R", role="REVIEWER")
    with Session(engine) as s:                                                           # a reviewer may only act on items assigned to them
        s.add(ReviewItem(comment_id="it-1", source_file="f.xlsx", source_row_index=1, original_comment="نص تجريبي", review_status="DRAFT_READY", reviewer_user_id=rev.user_id,
                         suggested_final_reply="شكرًا لكم."))
        s.commit()
    srv, base = serve_in_thread(engine)
    host, port = base.replace("http://", "").split(":")

    class Ctx:
        pass

    c = Ctx(); c.host, c.port, c.mode = host, int(port), request.param
    c.cookie_admin = _login(c, "admin@x.org", "admin-password-1")
    c.cookie_rev = _login(c, "rev@x.org", "rev-password-1")
    yield c
    srv.shutdown()


def _req(c, method, path, *, body=None, cookie=None, headers=None):
    h = {"Origin": "https://app.test"} if c.mode == "production" else {}
    if body is not None:
        h["Content-Type"] = "application/json"
    if cookie:
        h["Cookie"] = cookie
    h.update(headers or {})
    conn = http.client.HTTPConnection(c.host, c.port, timeout=15)
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=h)
    r = conn.getresponse(); raw = r.read().decode("utf-8"); conn.close()
    try:
        return r.status, json.loads(raw or "{}"), raw
    except ValueError:
        return r.status, {}, raw


def _login(c, email, password):
    conn = http.client.HTTPConnection(c.host, c.port, timeout=15)
    h = {"Content-Type": "application/json"}
    if c.mode == "production":
        h["Origin"] = "https://app.test"
    conn.request("POST", "/api/login", body=json.dumps({"email": email, "password": password}), headers=h)
    r = conn.getresponse(); r.read(); cookie = r.getheader("Set-Cookie").split(";")[0]; conn.close()
    return cookie


def assert_clean(raw: str):
    for banned in BANNED_IN_BODY:
        assert banned not in raw, f"response leaks {banned!r}: {raw[:200]}"


# ---------------------------------------------------------------- unhandled errors (the verified leaks)
@pytest.mark.parametrize("exc", [
    RuntimeError(CANARY),
    OperationalError("SELECT * FROM users WHERE email=:e", {"e": "victim@corp.example"}, Exception(CANARY)),
    IntegrityError("INSERT INTO review_items VALUES (:p)", {"p": CANARY}, Exception("duplicate key")),
    KeyError(CANARY),
])
def test_unhandled_get_error_returns_only_a_reference(server, monkeypatch, exc):
    monkeypatch.setattr(ReviewWorkflowService, "get_review_item", lambda self, cid: (_ for _ in ()).throw(exc))
    status, body, raw = _req(server, "GET", "/api/items/anything", cookie=server.cookie_admin)
    assert status in (500, 503)
    assert_clean(raw)
    if status == 500:
        assert set(body) == {"error", "reference"} and body["error"] == "server_error" and re.fullmatch(r"[0-9a-f]{8}", body["reference"])


@pytest.mark.parametrize("exc", [RuntimeError(CANARY), KeyError(CANARY), Exception(CANARY)])
def test_unhandled_post_error_returns_only_a_reference(server, monkeypatch, exc):
    monkeypatch.setattr(ReviewWorkflowService, "claim_review", lambda self, *a, **k: (_ for _ in ()).throw(exc))
    status, body, raw = _req(server, "POST", "/api/items/it-1/claim", body={"expected_version": 1}, cookie=server.cookie_rev)
    assert status == 500
    assert_clean(raw)
    assert set(body) == {"error", "reference"}


def test_persistence_outage_stays_a_generic_503(server, monkeypatch):
    err = OperationalError("SELECT 1", {}, Exception("connection to server at 10.1.2.3 refused"))
    monkeypatch.setattr(ReviewWorkflowService, "get_review_item", lambda self, cid: (_ for _ in ()).throw(err))
    status, body, raw = _req(server, "GET", "/api/items/x", cookie=server.cookie_admin)
    assert status == 503 and body == {"error": "service_unavailable"} and "10.1.2" not in raw


# ---------------------------------------------------------------- logging stays diagnostic but sanitised
def test_unhandled_error_is_logged_with_type_and_locations_but_never_the_message(server, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(ReviewWorkflowService, "claim_review", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError(CANARY)))
    status, body, _raw = _req(server, "POST", "/api/items/it-1/claim", body={"expected_version": 1}, cookie=server.cookie_rev)
    text = "\n".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)
    assert f"ref={body['reference']}" in text                                           # the client's reference finds exactly this line
    assert "type=RuntimeError" in text and "POST /api/items/it-1/claim" in text and "http_api.py" in text
    for secret in ("CANARY", "hunter2", "password_hash", "victim@corp", "Traceback"):
        assert secret not in text
    assert all(r.exc_info is None for r in caplog.records if "unhandled API error" in r.getMessage())     # no raw traceback object / text is attached


def test_the_log_line_never_includes_the_request_body_or_cookie(server, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(ReviewWorkflowService, "claim_review", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    _req(server, "POST", "/api/items/it-1/claim", body={"expected_version": 1, "note": "BODY-CANARY-visitor-text"}, cookie=server.cookie_rev)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "BODY-CANARY" not in text and "rcs_session" not in text and server.cookie_rev.split("=")[1] not in text


# ---------------------------------------------------------------- validation errors: safe ones survive, others are generic
def test_curated_validation_message_is_preserved(server):
    status, body, _ = _req(server, "POST", "/api/admin/inspector/classification", body={}, cookie=server.cookie_admin)
    assert status == 400 and body == {"error": "invalid_request", "detail": "comment_text is required"}


def test_library_value_errors_are_not_echoed(server):
    status, body, raw = _req(server, "POST", "/api/excel/process", body={"upload_token": "0" * 32, "batch_size": "not-a-number-CANARY"}, cookie=server.cookie_admin)
    assert status in (400, 422)
    assert "invalid literal" not in raw and "CANARY" not in raw and "int()" not in raw


def test_an_uncurated_value_error_message_is_dropped_but_keeps_its_status(server, monkeypatch):
    monkeypatch.setattr(ReviewWorkflowService, "claim_review", lambda self, *a, **k: (_ for _ in ()).throw(ValueError(CANARY)))
    status, body, raw = _req(server, "POST", "/api/items/it-1/claim", body={"expected_version": 1}, cookie=server.cookie_rev)
    assert status == 400 and body == {"error": "invalid_request"}
    assert_clean(raw)


def test_unknown_sort_value_is_still_reported(server):
    status, body, _ = _req(server, "GET", "/api/queue?sort=nonsense", cookie=server.cookie_admin)
    assert status in (200, 400)
    if status == 400:
        assert "CANARY" not in json.dumps(body)


def test_weak_password_message_is_preserved_and_unexpected_text_is_not(server, monkeypatch):
    monkeypatch.setattr(http_api.settings, "PASSWORD_RESET_LINK_BASE", "https://app.test")
    from rcs.review.email_sender import FakeEmailSender, set_email_sender
    set_email_sender(FakeEmailSender())
    try:
        status, body, _ = _req(server, "POST", "/api/password-reset/confirm", body={"token": "t" * 40, "password": "short"})
        assert status == 400 and body["error"] == "weak_password" and body["detail"] == "password must be at least 8 characters"
        monkeypatch.setattr(http_api, "confirm_password_reset", lambda *a, **k: (_ for _ in ()).throw(ValueError(CANARY)))
        status, body, raw = _req(server, "POST", "/api/password-reset/confirm", body={"token": "t" * 40, "password": "long-enough-password"})
        assert status == 400 and body["error"] == "weak_password" and "CANARY" not in raw
    finally:
        set_email_sender(None)


# ---------------------------------------------------------------- workflow conflicts
def test_version_conflict_keeps_409_without_internal_numbers(server, monkeypatch):
    monkeypatch.setattr(ReviewWorkflowService, "claim_review", lambda self, *a, **k: (_ for _ in ()).throw(VersionConflict("row_version 41 != expected 7")))
    status, body, raw = _req(server, "POST", "/api/items/it-1/claim", body={"expected_version": 7}, cookie=server.cookie_rev)
    assert status == 409 and body["error"] == "version_conflict"
    assert "41" not in raw and "row_version" not in raw


def test_claim_conflict_never_reveals_which_reviewer_holds_the_item(server, monkeypatch):
    monkeypatch.setattr(ReviewWorkflowService, "claim_review", lambda self, *a, **k: (_ for _ in ()).throw(InvalidTransition("claimed by reviewer-0a1b2c3d-other-user-id")))
    status, body, raw = _req(server, "POST", "/api/items/it-1/claim", body={"expected_version": 1}, cookie=server.cookie_rev)
    assert status == 409 and body["error"] == "claimed"
    assert "0a1b2c3d" not in raw and "other-user" not in raw


@pytest.mark.parametrize("message,expected_detail", [
    ("EDIT_AND_SEND requires non-empty edited_response", "EDIT_AND_SEND requires non-empty edited_response"),   # curated: preserved
    ("SEND_AS_IS requires a suggested_response", "SEND_AS_IS requires a suggested_response"),
    ("illegal transition: DRAFT_READY + SEND_AS_IS", "illegal transition: DRAFT_READY + SEND_AS_IS"),
    ("unknown destination status SOMETHING-INTERNAL", "this action is not allowed in the item's current state"),   # uncurated: replaced
    (CANARY, "this action is not allowed in the item's current state"),
])
def test_invalid_transition_details_are_curated(server, monkeypatch, message, expected_detail):
    monkeypatch.setattr(ReviewWorkflowService, "send_as_is", lambda self, *a, **k: (_ for _ in ()).throw(InvalidTransition(message)))
    status, body, raw = _req(server, "POST", "/api/items/it-1/send-as-is", body={"expected_version": 1}, cookie=server.cookie_rev)
    assert status == 409 and body["error"] == "invalid_transition" and body["detail"] == expected_detail
    assert "CANARY" not in raw


# ---------------------------------------------------------------- the safe, meaningful errors and all status semantics are intact
def test_upload_errors_keep_their_codes_and_arabic_messages(server):
    h = {"Content-Type": "application/octet-stream", "X-Filename": "bad.exe"}
    if server.mode == "production":
        h["Origin"] = "https://app.test"
    conn = http.client.HTTPConnection(server.host, server.port, timeout=15)
    conn.request("POST", "/api/excel/upload", body=b"not excel", headers={**h, "Cookie": server.cookie_admin})
    r = conn.getresponse(); body = json.loads(r.read()); conn.close()
    assert r.status == 400 and body["error"] == "unsupported_file_type" and "xlsx" in body["detail"]


def test_status_semantics_are_unchanged(server):
    assert _req(server, "GET", "/api/queue")[0] == 401                                              # unauthenticated
    assert _req(server, "GET", "/api/admin/users", cookie=server.cookie_rev)[0] == 403              # forbidden
    assert _req(server, "GET", "/api/items/does-not-exist", cookie=server.cookie_admin)[0] in (403, 404)
    assert _req(server, "GET", "/api/no-such-route", cookie=server.cookie_admin)[0] == 404
    assert _req(server, "POST", "/api/login", body={"email": "admin@x.org", "password": "wrong"})[0] == 401


def test_sanitizer_helpers_are_closed_by_default():
    assert http_api._curated("anything unexpected", http_api._SAFE_VALIDATION_MESSAGES, http_api._SAFE_VALIDATION_PATTERNS) is None
    assert http_api._validation_error_body(ValueError("invalid literal for int() with base 10: 'x'")) == {"error": "invalid_request"}
    assert http_api._code_locations(RuntimeError("no traceback")) == ""
    try:
        raise RuntimeError(CANARY)
    except RuntimeError as e:
        loc = http_api._code_locations(e)
    assert "test_api_error_sanitization.py" in loc and "CANARY" not in loc
