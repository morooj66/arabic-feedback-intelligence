"""Self-service password reset: token security, enumeration safety, rate limits, session revocation, SQL + Mongo parity, HTTP integration. No network, no real email."""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import urllib.error
import urllib.request

import pytest
from sqlalchemy.orm import Session

from config import settings
from rcs.review import auth
from rcs.review.auth import (
    authenticate,
    check_password_policy,
    confirm_password_reset,
    create_user,
    hash_reset_token,
    password_reset_available,
    request_password_reset,
    resolve_session,
    start_session,
)
from rcs.review.email_sender import FakeEmailSender, OutgoingEmail, UnconfiguredEmailSender, get_email_sender, set_email_sender
from rcs.review.http_api import serve_in_thread
from rcs.review.models import PasswordResetToken, User
from rcs.review.persistence import bundle_from_engine, bundle_from_mongo
from rcs.review.repository import get_engine, init_db

import test_mongo_repository_unit as mongo_fake

BASE = "https://rcs.example.test"
OLD_PW, NEW_PW = "old-password-123", "brand-new-password-456"


def _update_many(self, filt, update, **_):                      # the shared in-memory fake lacks update_many; add the one operation these repositories use
    n = 0
    for doc in self._docs:
        if mongo_fake._matches(doc, filt):
            mongo_fake._apply_update(doc, update)
            n += 1
    return mongo_fake._FakeUpdateResult(matched_count=n, modified_count=n)


mongo_fake._FakeCollection.update_many = _update_many


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    auth._reset_attempts.clear(); auth._rate_limit_attempts.clear()
    monkeypatch.setattr(auth, "_run_async", lambda fn: fn())    # deterministic: send inline in tests
    monkeypatch.setattr(settings, "PASSWORD_RESET_LINK_BASE", BASE)
    sender = FakeEmailSender()
    set_email_sender(sender)
    yield sender
    set_email_sender(None)
    auth._reset_attempts.clear(); auth._rate_limit_attempts.clear()


@pytest.fixture(params=["sql", "mongo"])
def backend(request, tmp_path):
    if request.param == "sql":
        engine = get_engine(tmp_path / "pr.sqlite"); init_db(engine)
        return request.param, bundle_from_engine(engine), engine
    db = mongo_fake._FakeDatabase()
    return request.param, bundle_from_mongo(db), db


def _user(bundle, email="alice@x.com", pw=OLD_PW, role="REVIEWER"):
    return create_user(bundle, email=email, password=pw, display_name="A", role=role)


def _token_from(sender, index=-1):
    m = re.search(r"/reset-password#token=([A-Za-z0-9_\-]+)", sender.outbox[index].body)
    assert m, sender.outbox[index].body
    return m.group(1)


def _deactivate(kind, bundle, user_id):
    if kind == "sql":
        with Session(bundle.users.engine) as s:
            s.get(User, user_id).is_active = False
            s.commit()
    else:
        bundle.users.collection.update_one({"_id": user_id}, {"$set": {"is_active": False}})


def _stored_tokens(kind, bundle):
    if kind == "sql":
        with Session(bundle.users.engine) as s:
            return [(r.token_hash, r.user_id, r.created_at, r.expires_at, r.used_at) for r in s.query(PasswordResetToken).all()]
    return [(d["_id"], d["user_id"], d["created_at"], d["expires_at"], d["used_at"]) for d in bundle.password_reset_tokens.collection._docs]


# ---------------------------------------------------------------- token storage, lifetime, single use
def test_only_the_token_hash_is_stored_and_the_link_uses_a_fragment(backend, _isolated):
    kind, bundle, _h = backend
    _user(bundle)
    request_password_reset(bundle, "alice@x.com")
    token = _token_from(_isolated)
    body = _isolated.outbox[0].body
    assert f"{BASE}/reset-password#token=" in body and "?token=" not in body
    rows = _stored_tokens(kind, bundle)
    assert len(rows) == 1 and rows[0][0] == hash_reset_token(token) and rows[0][0] != token
    assert all(token not in str(v) for v in rows[0])                                   # the raw token is nowhere in storage
    assert len(token) >= 40                                                            # 256 bits of randomness


def test_tokens_expire_within_thirty_minutes(backend, _isolated, monkeypatch):
    kind, bundle, _h = backend
    _user(bundle)
    request_password_reset(bundle, "alice@x.com")
    _hash, _uid, created, expires, _used = _stored_tokens(kind, bundle)[0]
    assert settings.PASSWORD_RESET_TOKEN_TTL_MINUTES <= 30
    assert expires - created <= dt.timedelta(minutes=30)
    real_now = auth._now()
    monkeypatch.setattr(auth, "_now", lambda: real_now + dt.timedelta(minutes=31))
    assert confirm_password_reset(bundle, _token_from(_isolated), NEW_PW) is False       # expired
    assert authenticate(bundle, email="alice@x.com", password=OLD_PW) is not None        # nothing changed


def test_a_token_works_once_only(backend, _isolated):
    _kind, bundle, _h = backend
    _user(bundle)
    request_password_reset(bundle, "alice@x.com")
    token = _token_from(_isolated)
    assert confirm_password_reset(bundle, token, NEW_PW) is True
    assert confirm_password_reset(bundle, token, "another-password-789") is False
    assert authenticate(bundle, email="alice@x.com", password=NEW_PW) is not None
    assert authenticate(bundle, email="alice@x.com", password=OLD_PW) is None


def test_repository_consume_is_atomic_single_use(backend):
    _kind, bundle, _h = backend
    now = dt.datetime.utcnow()
    bundle.password_reset_tokens.create(token_hash="h1", user_id="u1", created_at=now, expires_at=now + dt.timedelta(minutes=10))
    assert bundle.password_reset_tokens.consume("h1", now) == "u1"
    assert bundle.password_reset_tokens.consume("h1", now) is None


@pytest.mark.parametrize("bad", ["", "x", "not-a-real-token", "A" * 300, None, 12345])
def test_invalid_tokens_are_rejected(backend, bad):
    _kind, bundle, _h = backend
    _user(bundle)
    assert confirm_password_reset(bundle, bad, NEW_PW) is False
    assert authenticate(bundle, email="alice@x.com", password=OLD_PW) is not None


def test_the_stored_hash_cannot_be_used_as_a_token(backend, _isolated):
    kind, bundle, _h = backend
    _user(bundle)
    request_password_reset(bundle, "alice@x.com")
    stored_hash = _stored_tokens(kind, bundle)[0][0]
    assert confirm_password_reset(bundle, stored_hash, NEW_PW) is False                  # a database read is not a working link


def test_a_new_request_supersedes_the_previous_link(backend, _isolated):
    _kind, bundle, _h = backend
    _user(bundle)
    request_password_reset(bundle, "alice@x.com"); request_password_reset(bundle, "alice@x.com")
    old, new = _token_from(_isolated, 0), _token_from(_isolated, 1)
    assert confirm_password_reset(bundle, old, NEW_PW) is False
    assert confirm_password_reset(bundle, new, NEW_PW) is True


# ---------------------------------------------------------------- password policy
def test_policy_is_the_existing_one_plus_the_bcrypt_limit():
    check_password_policy("12345678")
    with pytest.raises(ValueError, match="at least 8"):
        check_password_policy("short")
    with pytest.raises(ValueError, match="at most 72"):
        check_password_policy("é" * 40)


def test_a_weak_password_does_not_burn_the_link(backend, _isolated):
    _kind, bundle, _h = backend
    _user(bundle)
    request_password_reset(bundle, "alice@x.com")
    token = _token_from(_isolated)
    with pytest.raises(ValueError):
        confirm_password_reset(bundle, token, "short")
    assert confirm_password_reset(bundle, token, NEW_PW) is True


# ---------------------------------------------------------------- accounts: unknown, inactive
def test_unknown_and_inactive_accounts_get_no_token_and_no_email(backend, _isolated):
    kind, bundle, _h = backend
    u = _user(bundle, "inactive@x.com")
    _deactivate(kind, bundle, u.user_id)
    request_password_reset(bundle, "nobody@x.com"); request_password_reset(bundle, "inactive@x.com")
    assert _isolated.outbox == [] and _stored_tokens(kind, bundle) == []


def test_deactivating_after_the_request_blocks_the_reset(backend, _isolated):
    kind, bundle, _h = backend
    u = _user(bundle)
    request_password_reset(bundle, "alice@x.com")
    _deactivate(kind, bundle, u.user_id)
    assert confirm_password_reset(bundle, _token_from(_isolated), NEW_PW) is False


# ---------------------------------------------------------------- sessions
def test_a_successful_reset_revokes_every_session_of_that_user_only(backend, _isolated):
    _kind, bundle, _h = backend
    a, b = _user(bundle), _user(bundle, "bob@x.com")
    sa1, sa2, sb = start_session(bundle, user_id=a.user_id), start_session(bundle, user_id=a.user_id), start_session(bundle, user_id=b.user_id)
    request_password_reset(bundle, "alice@x.com")
    assert confirm_password_reset(bundle, _token_from(_isolated), NEW_PW) is True
    assert resolve_session(bundle, sa1) is None and resolve_session(bundle, sa2) is None
    assert resolve_session(bundle, sb) is not None


def test_roles_and_activation_are_untouched(backend, _isolated):
    _kind, bundle, _h = backend
    u = _user(bundle, role="ADMIN")
    request_password_reset(bundle, "alice@x.com"); confirm_password_reset(bundle, _token_from(_isolated), NEW_PW)
    after = bundle.users.get_by_id(u.user_id)
    assert after.role == "ADMIN" and after.is_active


# ---------------------------------------------------------------- email abstraction / fail closed
def test_default_sender_is_unconfigured_and_fails_closed():
    set_email_sender(None)
    assert isinstance(get_email_sender(), UnconfiguredEmailSender) and password_reset_available() is False
    with pytest.raises(Exception):
        get_email_sender().send(OutgoingEmail("a@b.c", "s", "b"))


def test_link_base_is_also_required(monkeypatch):
    monkeypatch.setattr(settings, "PASSWORD_RESET_LINK_BASE", "")
    assert password_reset_available() is False


# ---------------------------------------------------------------- HTTP
def _post(base, path, data, xff=None):
    headers = {"Content-Type": "application/json"}
    if xff:
        headers["X-Forwarded-For"] = xff
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


@pytest.fixture()
def http(backend):
    kind, bundle, _h = backend
    _user(bundle); _user(bundle, "inactive@x.com")
    _deactivate(kind, bundle, bundle.users.get_by_email("inactive@x.com").user_id)
    srv, base = serve_in_thread(bundle)
    yield kind, bundle, base
    srv.shutdown()


def test_http_full_flow_reset_then_login(http, _isolated):
    _kind, bundle, base = http
    old_cookie_token = start_session(bundle, user_id=bundle.users.get_by_email("alice@x.com").user_id)
    assert _post(base, "/api/password-reset/request", {"email": "alice@x.com"}) == (200, {"ok": True})
    token = _token_from(_isolated)
    assert _post(base, "/api/password-reset/confirm", {"token": token, "password": NEW_PW}) == (200, {"ok": True})
    assert _post(base, "/api/login", {"email": "alice@x.com", "password": OLD_PW})[0] == 401
    assert _post(base, "/api/login", {"email": "alice@x.com", "password": NEW_PW})[0] == 200
    assert resolve_session(bundle, old_cookie_token) is None                              # the pre-reset session died
    assert _post(base, "/api/password-reset/confirm", {"token": token, "password": NEW_PW}) == (400, {"error": "invalid_or_expired_token"})


def test_http_responses_do_not_reveal_whether_an_account_exists(http, _isolated):
    _kind, _bundle, base = http
    answers = {e: _post(base, "/api/password-reset/request", {"email": e}) for e in ("alice@x.com", "nobody@x.com", "inactive@x.com", "ALICE@X.COM ")}
    assert len(set((s, json.dumps(b, sort_keys=True)) for s, b in answers.values())) == 1
    assert len(_isolated.outbox) == 2                                                    # only the two real, active requests produced mail (alice, ALICE)
    bad = [_post(base, "/api/password-reset/confirm", {"token": t, "password": NEW_PW}) for t in ("a" * 43, "b" * 10)]
    assert bad[0] == bad[1] == (400, {"error": "invalid_or_expired_token"})


def test_http_fails_closed_without_a_configured_sender_and_identically_for_every_email(http):
    _kind, bundle, base = http
    set_email_sender(None)
    a = _post(base, "/api/password-reset/request", {"email": "alice@x.com"})
    b = _post(base, "/api/password-reset/request", {"email": "nobody@x.com"})
    assert a == b == (503, {"error": "password_reset_unavailable"})
    assert _stored_tokens(_kind, bundle) == []
    assert _post(base, "/api/password-reset/confirm", {"token": "x" * 40, "password": NEW_PW})[0] == 503


def test_http_rejects_malformed_requests(http):
    _kind, _bundle, base = http
    for payload in ({}, {"email": ""}, {"email": 5}, {"email": "no-at-sign"}, {"email": "a@" + "b" * 300}):
        assert _post(base, "/api/password-reset/request", payload)[0] == 400
    assert _post(base, "/api/password-reset/confirm", {"token": 1, "password": 2})[0] == 400


def test_http_weak_password_is_a_400_and_keeps_the_token_usable(http, _isolated):
    _kind, _bundle, base = http
    _post(base, "/api/password-reset/request", {"email": "alice@x.com"})
    token = _token_from(_isolated)
    status, body = _post(base, "/api/password-reset/confirm", {"token": token, "password": "short"})
    assert status == 400 and body["error"] == "weak_password"
    assert _post(base, "/api/password-reset/confirm", {"token": token, "password": NEW_PW})[0] == 200


def test_http_request_rate_limit_per_email_applies_equally_to_unknown_emails(http, _isolated):
    _kind, _bundle, base = http
    for email in ("alice@x.com", "nobody@x.com"):
        codes = [_post(base, "/api/password-reset/request", {"email": email})[0] for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429], (email, codes)
    assert len(_isolated.outbox) == 3                                                    # only alice's first three requests sent mail


def test_http_request_rate_limit_per_address_and_spoofed_header_is_ignored(http, monkeypatch):
    _kind, _bundle, base = http
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", ())
    codes = [_post(base, "/api/password-reset/request", {"email": f"user{i}@x.com"}, xff=f"9.9.9.{i}")[0] for i in range(12)]
    assert codes[:10] == [200] * 10 and codes[10:] == [429, 429]


def test_http_token_guessing_is_rate_limited(http):
    _kind, _bundle, base = http
    codes = [_post(base, "/api/password-reset/confirm", {"token": f"guess{i:03d}" * 5, "password": NEW_PW})[0] for i in range(22)]
    assert codes[:20] == [400] * 20 and codes[20:] == [429, 429]


def test_http_health_reports_availability(http):
    _kind, _bundle, base = http
    with urllib.request.urlopen(base + "/api/health", timeout=10) as r:
        assert json.loads(r.read())["password_reset"] is True
    set_email_sender(None)
    with urllib.request.urlopen(base + "/api/health", timeout=10) as r:
        assert json.loads(r.read())["password_reset"] is False


def test_secrets_never_reach_the_logs_even_when_sending_fails(http, caplog, monkeypatch):
    _kind, bundle, base = http

    class Boom:
        configured = True

        def send(self, message):
            raise RuntimeError(f"smtp rejected {message.to} body={message.body}")        # a hostile exception text that echoes everything

    set_email_sender(Boom())
    caplog.set_level(logging.DEBUG)
    status, _b = _post(base, "/api/password-reset/request", {"email": "alice@x.com"})
    assert status == 200                                                                  # a delivery failure is not visible to the caller
    tokens = [r[0] for r in _stored_tokens(_kind, bundle)]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "alice@x.com" not in text and "reset-password" not in text and not any(t in text for t in tokens)
    assert "error_type=RuntimeError" in text


def test_passwords_never_reach_the_logs(http, _isolated, caplog):
    _kind, _bundle, base = http
    caplog.set_level(logging.DEBUG)
    _post(base, "/api/password-reset/request", {"email": "alice@x.com"})
    _post(base, "/api/password-reset/confirm", {"token": _token_from(_isolated), "password": NEW_PW})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert NEW_PW not in text and _token_from(_isolated) not in text


# ---------------------------------------------------------------- migration (isolated SQLite file, never a real database)
def test_migration_creates_exactly_the_model_table_and_downgrades_cleanly(tmp_path):
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect

    path = Path(__file__).resolve().parents[1] / "migrations/versions/b3d9f0a1c7e2_password_reset_tokens.py"
    spec = importlib.util.spec_from_file_location("mig_b3d9", path)
    mig = importlib.util.module_from_spec(spec); spec.loader.exec_module(mig)
    engine = create_engine(f"sqlite:///{tmp_path / 'mig.sqlite'}")
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mig.upgrade()
    insp = inspect(engine)
    cols = {c["name"]: c for c in insp.get_columns("password_reset_tokens")}
    assert set(cols) == {c.name for c in PasswordResetToken.__table__.columns} == {"token_hash", "user_id", "created_at", "expires_at", "used_at"}
    assert {n for n, c in cols.items() if not c["nullable"]} == {"token_hash", "user_id", "created_at", "expires_at"}
    assert {i["name"] for i in insp.get_indexes("password_reset_tokens")} == {"ix_password_reset_tokens_user_created"}
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mig.downgrade()
    assert "password_reset_tokens" not in inspect(engine).get_table_names()
