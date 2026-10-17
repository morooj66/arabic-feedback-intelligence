"""Phase 6 — real authentication and role-based access.

Covers auth.py unit behavior plus the HTTP-level security checks listed
in the Phase 6 spec (Step 9, items 1-13, 16, 19, 20 — 14/15/17/18 are
live-verification concerns covered in the Phase 6 report, and 17/18 are
already covered by the Phase 4.1 dashboard/redaction and export tests,
not duplicated here).
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

from sqlalchemy.orm import Session

from rcs.review.auth import (
    ADMIN,
    REVIEWER,
    authenticate,
    create_user,
    end_session,
    hash_password,
    resolve_session,
    session_key,
    start_session,
    verify_password,
)
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem, User, UserSession
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService


def _json(url: str, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), err.headers


def _engine(tmp_path, name="auth.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


# --- unit-level: password hashing ------------------------------------------


def test_password_hash_is_never_plaintext_and_verifies_correctly():
    h = hash_password("a-real-password-123")
    assert h != "a-real-password-123"
    assert verify_password("a-real-password-123", h)
    assert not verify_password("wrong-password", h)


def test_create_user_rejects_short_password_and_duplicate_email(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="a@x.com", password="longenough123", display_name="A", role=REVIEWER)
    try:
        create_user(engine, email="short@x.com", password="short", display_name="B", role=REVIEWER)
        assert False, "expected ValueError for a too-short password"
    except ValueError:
        pass
    try:
        create_user(engine, email="a@x.com", password="longenough123", display_name="dup", role=REVIEWER)
        assert False, "expected ValueError for a duplicate email"
    except ValueError:
        pass
    try:
        create_user(engine, email="c@x.com", password="longenough123", display_name="C", role="SUPERUSER")
        assert False, "expected ValueError for an invalid role"
    except ValueError:
        pass


# --- unit-level: authenticate / sessions ------------------------------------


def test_authenticate_rejects_unknown_email_wrong_password_and_disabled_account(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    assert authenticate(engine, email="nobody@x.com", password="whatever") is None
    assert authenticate(engine, email="rev@x.com", password="wrongpassword") is None
    ok = authenticate(engine, email="rev@x.com", password="realpassword123")
    assert ok is not None and ok.email == "rev@x.com"

    with Session(engine) as db:
        row = db.get(User, ok.user_id)
        row.is_active = False
        db.commit()
    assert authenticate(engine, email="rev@x.com", password="realpassword123") is None


def test_session_lifecycle_start_resolve_end(tmp_path):
    engine = _engine(tmp_path)
    user = create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    token = start_session(engine, user_id=user.user_id)
    resolved = resolve_session(engine, token)
    assert resolved is not None
    assert resolved.user_id == user.user_id
    assert resolved.role == REVIEWER

    end_session(engine, token)
    assert resolve_session(engine, token) is None  # real invalidation, not just "wait for expiry"


def test_expired_session_token_is_rejected(tmp_path):
    engine = _engine(tmp_path)
    user = create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    token = start_session(engine, user_id=user.user_id)
    with Session(engine) as db:
        row = db.get(UserSession, session_key(token))
        row.expires_at = dt.datetime.utcnow() - dt.timedelta(seconds=1)
        db.commit()
    assert resolve_session(engine, token) is None


def test_only_the_hash_of_the_session_token_is_stored(tmp_path):
    engine = _engine(tmp_path)
    user = create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    token = start_session(engine, user_id=user.user_id)
    with Session(engine) as db:
        stored = [r.session_token for r in db.query(UserSession).all()]
    assert stored == [session_key(token)] and token not in stored and len(stored[0]) == 64
    assert resolve_session(engine, token).user_id == user.user_id
    assert resolve_session(engine, stored[0]) is None                      # the stored value is NOT a usable cookie
    end_session(engine, stored[0])                                         # ...and cannot be used to log someone out either
    assert resolve_session(engine, token) is not None
    end_session(engine, token)
    assert resolve_session(engine, token) is None


def test_garbage_or_unknown_session_token_is_rejected(tmp_path):
    engine = _engine(tmp_path)
    assert resolve_session(engine, None) is None
    assert resolve_session(engine, "totally-made-up-token") is None


def test_deactivating_a_user_immediately_invalidates_their_live_session(tmp_path):
    """A session token that was valid when issued must stop working the
    moment the account is deactivated — no stale access survives a
    deactivation, even mid-session."""
    engine = _engine(tmp_path)
    user = create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    token = start_session(engine, user_id=user.user_id)
    assert resolve_session(engine, token) is not None
    with Session(engine) as db:
        db.get(User, user.user_id).is_active = False
        db.commit()
    assert resolve_session(engine, token) is None


# --- HTTP-level security checks (Phase 6 Step 9) ----------------------------


def test_protected_endpoint_without_auth_returns_401(tmp_path):
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        assert _json(f"{base}/api/queue")[0] == 401
        assert _json(f"{base}/api/counts")[0] == 401
        assert _json(f"{base}/api/session")[0] == 401
    finally:
        server.shutdown()


def test_reviewer_cannot_call_admin_endpoint_403_not_401(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    server, base = serve_in_thread(engine)
    try:
        _, _, headers = _json(f"{base}/api/login", method="POST", data={"email": "rev@x.com", "password": "realpassword123"})
        cookie = headers.get("Set-Cookie").split(";")[0]
        status, payload, _ = _json(f"{base}/api/admin/users", cookie=cookie)
        assert status == 403  # authenticated but unauthorized, distinct from 401
        assert payload["error"] == "forbidden"
    finally:
        server.shutdown()


def test_reviewer_cannot_view_unscoped_global_analytics_but_can_view_scoped(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    server, base = serve_in_thread(engine)
    try:
        _, _, headers = _json(f"{base}/api/login", method="POST", data={"email": "rev@x.com", "password": "realpassword123"})
        cookie = headers.get("Set-Cookie").split(";")[0]
        unscoped = _json(f"{base}/api/analytics?range=week", cookie=cookie)
        assert unscoped[0] == 403
        scoped = _json(f"{base}/api/analytics?range=week&source_file=some.xlsx", cookie=cookie)
        assert scoped[0] == 200  # the dashboard's own scoped view stays available to REVIEWER
    finally:
        server.shutdown()


def test_admin_can_view_both_scoped_and_unscoped_analytics_and_admin_users(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    server, base = serve_in_thread(engine)
    try:
        _, _, headers = _json(f"{base}/api/login", method="POST", data={"email": "admin@x.com", "password": "realpassword123"})
        cookie = headers.get("Set-Cookie").split(";")[0]
        assert _json(f"{base}/api/analytics?range=week", cookie=cookie)[0] == 200
        status, payload, _ = _json(f"{base}/api/admin/users", cookie=cookie)
        assert status == 200
        assert any(u["email"] == "admin@x.com" for u in payload["users"])
        assert "password_hash" not in json.dumps(payload)
    finally:
        server.shutdown()


def test_client_cannot_forge_role_via_login_body(tmp_path):
    """Posting a role in the login body must be silently ignored — role
    only ever comes from the users table, never the request."""
    engine = _engine(tmp_path)
    create_user(engine, email="rev@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    server, base = serve_in_thread(engine)
    try:
        status, payload, _ = _json(
            f"{base}/api/login",
            method="POST",
            data={"email": "rev@x.com", "password": "realpassword123", "role": "ADMIN"},
        )
        assert status == 200
        assert payload["role"] == "REVIEWER"  # the forged "role": "ADMIN" in the body had no effect
    finally:
        server.shutdown()


def test_client_cannot_forge_reviewer_identity_on_decision_submission(tmp_path):
    """A forged reviewer_id/user_id in the action body must be ignored —
    the persisted identity always comes from the authenticated session."""
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 1, 1)
    real_user = create_user(engine, email="real@x.com", password="realpassword123", display_name="Real Reviewer", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            {
                "comment_id": "forge-test",
                "source_file": "t",
                "source_row_index": 1,
                "original_comment": "شكرًا",
                "ingested_at": now,
                "review_status": "DRAFT_READY",
                "suggested_final_reply": "رد مقترح",
                "ready_to_send": False,
                "row_version": 1,
                # Phase 8: a REVIEWER can only act on an item already
                # assigned to them — pre-assign so this test still
                # exercises identity-forgery prevention, not the
                # (separately tested) assignment gate.
                "reviewer_user_id": real_user.user_id,
            }
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        _, _, headers = _json(f"{base}/api/login", method="POST", data={"email": "real@x.com", "password": "realpassword123"})
        cookie = headers.get("Set-Cookie").split(";")[0]
        status, sent, _ = _json(
            f"{base}/api/items/forge-test/send-as-is",
            method="POST",
            # attempts to forge a completely different identity in the body
            data={"expected_version": 1, "reviewer_id": "someone-else", "user_id": "not-real-either", "display_name": "Impersonator"},
            cookie=cookie,
        )
        assert status == 200
        assert sent["review_status"] == "APPROVED"
    finally:
        server.shutdown()

    with Session(engine) as db:
        row = db.get(ReviewItem, "forge-test")
        assert row.reviewer_user_id == real_user.user_id  # the REAL authenticated identity, not the forged body
        assert row.reviewer_id == real_user.user_id
        assert row.reviewer_display_name == "Real Reviewer"


def test_new_authenticated_reviewer_action_persists_reviewer_user_id(tmp_path):
    """Step 5/16: a NEW decision made through a real authenticated
    session must record reviewer_user_id pointing at users.user_id."""
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 1, 1)
    upsert_review_items(
        engine,
        [
            {
                "comment_id": "identity-persist-test",
                "source_file": "t",
                "source_row_index": 1,
                "original_comment": "تعليق",
                "ingested_at": now,
                "review_status": "REVIEW_ONLY",
                "ready_to_send": False,
                "row_version": 1,
            }
        ],
    )
    user = create_user(engine, email="id-test@x.com", password="realpassword123", display_name="Identity Tester", role=REVIEWER)
    svc = ReviewWorkflowService(engine)
    from rcs.review.identity import ReviewerIdentity

    reviewer = ReviewerIdentity(reviewer_id=user.user_id, display_name=user.display_name, user_id=user.user_id)
    svc.mark_no_response_needed("identity-persist-test", reviewer, reason="لا يحتاج", expected_version=1)

    with Session(engine) as db:
        row = db.get(ReviewItem, "identity-persist-test")
        assert row.reviewer_user_id == user.user_id


def test_historical_rows_keep_null_reviewer_user_id_and_are_never_backfilled(tmp_path):
    """A row that already had a reviewer_id from before Phase 6 (or any
    row never touched by an authenticated Phase 6 action) must keep
    reviewer_user_id NULL — never fabricated."""
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 1, 1)
    upsert_review_items(
        engine,
        [
            {
                "comment_id": "historical-row",
                "source_file": "t",
                "source_row_index": 1,
                "original_comment": "تعليق قديم",
                "ingested_at": now,
                "review_status": "APPROVED",
                "reviewer_id": "demo:old-reviewer",
                "reviewer_display_name": "مراجع قديم",
                "row_version": 1,
            }
        ],
    )
    with Session(engine) as db:
        row = db.get(ReviewItem, "historical-row")
        assert row.reviewer_user_id is None
        assert row.reviewer_id == "demo:old-reviewer"  # untouched
        assert row.reviewer_display_name == "مراجع قديم"  # untouched


def test_password_and_hash_never_appear_in_login_or_session_response(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="rev@x.com", password="a-secret-password-123", display_name="R", role=REVIEWER)
    server, base = serve_in_thread(engine)
    try:
        _, login_payload, headers = _json(
            f"{base}/api/login", method="POST", data={"email": "rev@x.com", "password": "a-secret-password-123"}
        )
        raw = json.dumps(login_payload)
        assert "a-secret-password-123" not in raw
        assert "password_hash" not in raw
        cookie = headers.get("Set-Cookie").split(";")[0]
        _, session_payload, _ = _json(f"{base}/api/session", cookie=cookie)
        raw2 = json.dumps(session_payload)
        assert "a-secret-password-123" not in raw2
        assert "password_hash" not in raw2
    finally:
        server.shutdown()


def test_no_public_self_registration_endpoint_exists(tmp_path):
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        for path in ("/api/register", "/api/signup", "/api/users", "/api/demo/login"):
            status, payload, _ = _json(f"{base}{path}", method="POST", data={"email": "new@x.com", "password": "whatever123"})
            # Never 200/201 — either rejected as unauthenticated (routes
            # requiring auth are checked before "not found") or a plain
            # 404. Either way, nothing here can create an account.
            assert status in (401, 404), f"{path} should not exist as a public endpoint, got {status}"
            assert payload.get("error") in ("unauthenticated", "not_found")
        # logging in with an email nobody created must fail, not silently create an account
        status, payload, _ = _json(
            f"{base}/api/login", method="POST", data={"email": "never-created@x.com", "password": "whatever123"}
        )
        assert status == 401
        assert payload["error"] == "invalid_credentials"
    finally:
        server.shutdown()
