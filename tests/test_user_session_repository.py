"""
Phase M4D.2 — UserSessionRepository contract + parity tests.

Postgres and Mongo scenarios are placed side by side per behavior, same
convention as the other repository test files. The Mongo tests run
against the same in-memory fake used in test_mongo_repository_unit.py --
they prove THIS CODE's logic, not MongoDB's own server-side primary-key
uniqueness guarantee. See test_mongo_repository_live.py for what
remains BLOCKED pending a working Atlas connection.

Security: every token/hash used below is a synthetic literal, never a
real credential or a real session issued by this app.

Architecture note load-bearing for several tests: the repository layer
has NO expiry or active-account logic at all -- get_by_token returns
the raw row regardless of whether it is expired, and the repository
never touches the `users` collection/table. Both checks live in
rcs.review.auth.resolve_session (untouched business logic). The
`_resolve` helper below is a small, test-local mirror of that exact
logic, used only to prove the Mongo repository primitives compose
correctly -- auth.py itself is not modified to accept a Mongo backend
(no runtime selector exists, per instruction).
"""

from __future__ import annotations

import datetime as dt

import pytest
from pymongo.errors import DuplicateKeyError
from sqlalchemy.exc import IntegrityError

from rcs.review.repositories import contracts
from rcs.review.repositories.mongo import MongoUserRepository, MongoUserSessionRepository
from rcs.review.repositories.postgres import PostgresUserRepository, PostgresUserSessionRepository
from rcs.review.repository import get_engine, init_db

from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake

_SYNTHETIC_HASH = "$2b$12$synthetichashsynthetichashsynthetichashsyn"


def _engine(tmp_path, name="sessions.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _pg_repo(tmp_path):
    return PostgresUserSessionRepository(_engine(tmp_path))


def _pg_repo_pair(tmp_path):
    engine = _engine(tmp_path)
    return PostgresUserSessionRepository(engine), PostgresUserRepository(engine)


def _mongo_repo():
    return MongoUserSessionRepository(_FakeDatabase())


def _mongo_repo_pair():
    db = _FakeDatabase()
    return MongoUserSessionRepository(db), MongoUserRepository(db)


def _session_kwargs(**overrides):
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    kwargs = dict(
        session_token="tok-synthetic-1",
        user_id="u-1",
        created_at=now,
        expires_at=now + dt.timedelta(hours=12),
    )
    kwargs.update(overrides)
    return kwargs


def _user_kwargs(**overrides):
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    kwargs = dict(
        user_id="u-1", email="a@x.com", password_hash=_SYNTHETIC_HASH, display_name="A",
        role="REVIEWER", is_active=True, created_at=now, updated_at=now,
    )
    kwargs.update(overrides)
    return kwargs


def _resolve(session_repo, user_repo, token, *, now):
    """Test-local mirror of rcs.review.auth.resolve_session's exact
    logic, minus token generation and cookie plumbing -- used only to
    prove the repository primitives (session lookup + expiry check +
    user lookup + active check) compose correctly for BOTH backends."""
    if not token:
        return None
    sess = session_repo.get_by_token(token)
    if sess is None or sess.expires_at <= now:
        return None
    user = user_repo.get_by_id(sess.user_id)
    if user is None or not user.is_active:
        return None
    return user


# A. Contract/signature ------------------------------------------------


def test_contract_declares_user_session_repository_methods():
    for name in ("create", "get_by_token", "delete_by_token", "delete_expired"):
        assert hasattr(contracts.UserSessionRepository, name)


def test_both_backends_implement_user_session_repository():
    for name in ("create", "get_by_token", "delete_by_token", "delete_expired"):
        assert hasattr(PostgresUserSessionRepository, name)
        assert hasattr(MongoUserSessionRepository, name)


# B. Create session -------------------------------------------------------


def test_create_returns_full_row_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    created = repo.create(**_session_kwargs())
    assert created.session_token == "tok-synthetic-1"
    assert created.user_id == "u-1"


def test_create_returns_full_row_mongo():
    repo = _mongo_repo()
    created = repo.create(**_session_kwargs())
    assert created.session_token == "tok-synthetic-1"
    assert created.user_id == "u-1"


# C. Get valid session -----------------------------------------------------


def test_get_by_token_returns_the_session_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_session_kwargs())
    found = repo.get_by_token("tok-synthetic-1")
    assert found is not None
    assert found.user_id == "u-1"


def test_get_by_token_returns_the_session_mongo():
    repo = _mongo_repo()
    repo.create(**_session_kwargs())
    found = repo.get_by_token("tok-synthetic-1")
    assert found is not None
    assert found.user_id == "u-1"


# D. Missing token ----------------------------------------------------------


def test_get_by_token_missing_returns_none_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    assert repo.get_by_token("does-not-exist") is None


def test_get_by_token_missing_returns_none_mongo():
    repo = _mongo_repo()
    assert repo.get_by_token("does-not-exist") is None


# E. Expired session: repository still returns the row (no filtering) -----


def test_get_by_token_returns_expired_row_unfiltered_postgres(tmp_path):
    """The repository itself does NOT reject expired sessions -- that
    check belongs to auth.py's resolve_session, proven separately in M
    below. get_by_token is a plain lookup."""
    repo = _pg_repo(tmp_path)
    past = dt.datetime(2020, 1, 1)
    repo.create(**_session_kwargs(expires_at=past))
    found = repo.get_by_token("tok-synthetic-1")
    assert found is not None
    assert found.expires_at == past


def test_get_by_token_returns_expired_row_unfiltered_mongo():
    repo = _mongo_repo()
    past = dt.datetime(2020, 1, 1)
    repo.create(**_session_kwargs(expires_at=past))
    found = repo.get_by_token("tok-synthetic-1")
    assert found is not None
    assert found.expires_at == past


# F. Boundary: expires_at == now is treated as expired (strict <=) --------


def test_expiry_boundary_equal_to_now_is_rejected_postgres(tmp_path):
    session_repo, user_repo = _pg_repo_pair(tmp_path)
    user_repo.create(**_user_kwargs())
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    session_repo.create(**_session_kwargs(expires_at=now))
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is None
    # one microsecond earlier: still valid
    session_repo.delete_by_token("tok-synthetic-1")
    session_repo.create(**_session_kwargs(expires_at=now + dt.timedelta(microseconds=1)))
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is not None


def test_expiry_boundary_equal_to_now_is_rejected_mongo():
    session_repo, user_repo = _mongo_repo_pair()
    user_repo.create(**_user_kwargs())
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    session_repo.create(**_session_kwargs(expires_at=now))
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is None
    session_repo.delete_by_token("tok-synthetic-1")
    session_repo.create(**_session_kwargs(expires_at=now + dt.timedelta(microseconds=1)))
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is not None


# G. Logout/delete ----------------------------------------------------------


def test_delete_by_token_removes_session_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_session_kwargs())
    repo.delete_by_token("tok-synthetic-1")
    assert repo.get_by_token("tok-synthetic-1") is None


def test_delete_by_token_removes_session_mongo():
    repo = _mongo_repo()
    repo.create(**_session_kwargs())
    repo.delete_by_token("tok-synthetic-1")
    assert repo.get_by_token("tok-synthetic-1") is None


def test_delete_by_token_unknown_token_is_a_silent_noop_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.delete_by_token("never-existed")  # must not raise


def test_delete_by_token_unknown_token_is_a_silent_noop_mongo():
    repo = _mongo_repo()
    repo.delete_by_token("never-existed")  # must not raise


# H. Purge expired -----------------------------------------------------------


def test_delete_expired_removes_only_expired_and_returns_count_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    repo.create(**_session_kwargs(session_token="expired-1", expires_at=now - dt.timedelta(hours=1)))
    repo.create(**_session_kwargs(session_token="expired-2", expires_at=now))  # inclusive boundary
    repo.create(**_session_kwargs(session_token="alive-1", expires_at=now + dt.timedelta(hours=1)))
    count = repo.delete_expired(now)
    assert count == 2
    assert repo.get_by_token("alive-1") is not None
    assert repo.get_by_token("expired-1") is None
    assert repo.get_by_token("expired-2") is None


def test_delete_expired_removes_only_expired_and_returns_count_mongo():
    repo = _mongo_repo()
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    repo.create(**_session_kwargs(session_token="expired-1", expires_at=now - dt.timedelta(hours=1)))
    repo.create(**_session_kwargs(session_token="expired-2", expires_at=now))
    repo.create(**_session_kwargs(session_token="alive-1", expires_at=now + dt.timedelta(hours=1)))
    count = repo.delete_expired(now)
    assert count == 2
    assert repo.get_by_token("alive-1") is not None
    assert repo.get_by_token("expired-1") is None
    assert repo.get_by_token("expired-2") is None


# I. Non-expired session survives purge (see also H) -------------------------


def test_purge_with_nothing_expired_is_a_noop_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    repo.create(**_session_kwargs(expires_at=now + dt.timedelta(hours=1)))
    assert repo.delete_expired(now) == 0
    assert repo.get_by_token("tok-synthetic-1") is not None


def test_purge_with_nothing_expired_is_a_noop_mongo():
    repo = _mongo_repo()
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    repo.create(**_session_kwargs(expires_at=now + dt.timedelta(hours=1)))
    assert repo.delete_expired(now) == 0
    assert repo.get_by_token("tok-synthetic-1") is not None


# J. Duplicate token (blind insert relies on the store's own PK) ----------


def test_duplicate_token_raises_integrity_error_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_session_kwargs(user_id="u-1"))
    with pytest.raises(IntegrityError):
        repo.create(**_session_kwargs(user_id="u-2"))


def test_duplicate_token_raises_duplicate_key_error_mongo():
    repo = _mongo_repo()
    repo.create(**_session_kwargs(user_id="u-1"))
    with pytest.raises(DuplicateKeyError):
        repo.create(**_session_kwargs(user_id="u-2"))


# K. user_id round-trip ------------------------------------------------------


def test_user_id_round_trips_exactly_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_session_kwargs(user_id="specific-user-id-123"))
    assert repo.get_by_token("tok-synthetic-1").user_id == "specific-user-id-123"


def test_user_id_round_trips_exactly_mongo():
    repo = _mongo_repo()
    repo.create(**_session_kwargs(user_id="specific-user-id-123"))
    assert repo.get_by_token("tok-synthetic-1").user_id == "specific-user-id-123"


# L. Datetime/BSON parity (both columns are "utc" convention) --------------


def test_created_and_expires_at_round_trip_exactly_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    created_at = dt.datetime(2026, 3, 4, 8, 30, 0)
    expires_at = created_at + dt.timedelta(hours=12)
    repo.create(**_session_kwargs(created_at=created_at, expires_at=expires_at))
    found = repo.get_by_token("tok-synthetic-1")
    assert found.created_at == created_at
    assert found.expires_at == expires_at


def test_created_and_expires_at_round_trip_exactly_mongo():
    repo = _mongo_repo()
    created_at = dt.datetime(2026, 3, 4, 8, 30, 0)
    expires_at = created_at + dt.timedelta(hours=12)
    repo.create(**_session_kwargs(created_at=created_at, expires_at=expires_at))
    found = repo.get_by_token("tok-synthetic-1")
    assert found.created_at == created_at
    assert found.expires_at == expires_at


# M/N. Active-account revalidation, including deactivation mid-session ----


def test_resolve_fails_when_account_deactivated_mid_session_postgres(tmp_path):
    session_repo, user_repo = _pg_repo_pair(tmp_path)
    created = user_repo.create(**_user_kwargs())
    session_repo.create(**_session_kwargs(user_id=created.user_id))
    now = dt.datetime(2026, 1, 1, 10, 0, 0)
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is not None
    # deactivate mid-session, using the SAME engine (a separate write path,
    # exactly like manage_users.py would)
    from sqlalchemy.orm import Session as SASession
    from rcs.review.models import User as UserModel
    with SASession(session_repo.engine) as db:
        db.get(UserModel, created.user_id).is_active = False
        db.commit()
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is None


def test_resolve_fails_when_account_deactivated_mid_session_mongo():
    session_repo, user_repo = _mongo_repo_pair()
    created = user_repo.create(**_user_kwargs())
    session_repo.create(**_session_kwargs(user_id=created.user_id))
    now = dt.datetime(2026, 1, 1, 10, 0, 0)
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is not None
    user_repo.collection.update_one({"_id": created.user_id}, {"$set": {"is_active": False}})
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is None


# O. Unknown user referenced by session --------------------------------------


def test_resolve_fails_cleanly_when_session_references_no_user_postgres(tmp_path):
    session_repo, user_repo = _pg_repo_pair(tmp_path)
    session_repo.create(**_session_kwargs(user_id="ghost-user"))
    now = dt.datetime(2026, 1, 1, 10, 0, 0)
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is None


def test_resolve_fails_cleanly_when_session_references_no_user_mongo():
    session_repo, user_repo = _mongo_repo_pair()
    session_repo.create(**_session_kwargs(user_id="ghost-user"))
    now = dt.datetime(2026, 1, 1, 10, 0, 0)
    assert _resolve(session_repo, user_repo, "tok-synthetic-1", now=now) is None


# P. Mongo mutations never use blind application-level write retry --------


@pytest.mark.parametrize("method,kwargs", [
    ("create", _session_kwargs()),
    ("delete_by_token", {"session_token": "whatever"}),
    ("delete_expired", {"now": dt.datetime(2026, 1, 1)}),
])
def test_mongo_mutations_never_use_transient_retry(monkeypatch, method, kwargs):
    import rcs.review.repositories.mongo as mongo_module

    def _boom(*_a, **_kw):
        raise AssertionError(f"with_transient_retry must not be used by UserSessionRepository.{method}")

    monkeypatch.setattr(mongo_module, "with_transient_retry", _boom)
    repo = MongoUserSessionRepository(_FakeDatabase())
    getattr(repo, method)(**kwargs)  # must not raise the AssertionError above


# Q. Mongo read lookup may use the safe transient retry layer ---------------


def test_mongo_get_by_token_uses_transient_retry(monkeypatch):
    import rcs.review.repositories.mongo as mongo_module

    calls = {"n": 0}
    real = mongo_module.with_transient_retry

    def _counting(fn):
        calls["n"] += 1
        return real(fn)

    monkeypatch.setattr(mongo_module, "with_transient_retry", _counting)
    repo = MongoUserSessionRepository(_FakeDatabase())
    repo.get_by_token("whatever")
    assert calls["n"] == 1
