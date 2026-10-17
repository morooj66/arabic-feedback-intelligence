"""
Production login/DB boundary hardening — proves that a persistence-
infrastructure failure (Mongo unreachable, Mongo authentication/
authorization rejected, Postgres unreachable) reaching the HTTP layer
during login/session/logout/health NEVER:

  - is reported to the end user as "wrong email/password"
  - leaks a raw driver exception, connection string, hostname, or
    exception class name into the browser-facing JSON body

and instead always produces one stable, sanitized `{"error":
"service_unavailable"}` (503).

No live MongoDB/Postgres connection is used anywhere in this file --
every failing repository here is a small hand-built fake that raises a
real pymongo/SQLAlchemy exception type on demand, wired into a genuine
RepositoryBundle exactly as the served application's composition root
(rcs.review.persistence.build_repository_bundle) would build one -- the
whole point being to prove the HTTP adapter's own sanitization, not to
re-prove pymongo/SQLAlchemy's own behavior.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from pymongo.errors import ConnectionFailure, OperationFailure
from sqlalchemy.exc import OperationalError as PostgresOperationalError

from rcs.review.auth import create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.persistence import RepositoryBundle, bundle_from_engine
from rcs.review.repository import get_engine, init_db


def _json_request(url: str, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload
    except urllib.error.HTTPError as err:
        raw = err.read()
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        return err.code, payload


class _RaisingUsers:
    """Stands in for UserRepository -- every method raises the given
    exception, exactly as a real Mongo/Postgres repository would when
    the underlying connection/authentication fails."""

    def __init__(self, exc_factory):
        self._exc_factory = exc_factory

    def get_by_id(self, user_id):
        raise self._exc_factory()

    def get_by_email(self, email):
        raise self._exc_factory()

    def list_all(self):
        raise self._exc_factory()

    def list_active_reviewers(self):
        raise self._exc_factory()

    def create(self, **kwargs):
        raise self._exc_factory()


class _RaisingUserSessions:
    """Stands in for UserSessionRepository -- same failure-injection
    shape as _RaisingUsers, for the session half of the login/logout
    path."""

    def __init__(self, exc_factory):
        self._exc_factory = exc_factory

    def create(self, **kwargs):
        raise self._exc_factory()

    def get_by_token(self, session_token):
        raise self._exc_factory()

    def delete_by_token(self, session_token):
        raise self._exc_factory()

    def delete_expired(self, now):
        raise self._exc_factory()


def _failing_bundle(exc_factory) -> RepositoryBundle:
    """A RepositoryBundle whose users/user_sessions always raise --
    every other field is None, matching this file's scope (login/
    session/logout/health never touch review_items/analytics/etc)."""
    return RepositoryBundle(
        review_items=None,  # type: ignore[arg-type]
        users=_RaisingUsers(exc_factory),
        user_sessions=_RaisingUserSessions(exc_factory),
        review_sessions=None,  # type: ignore[arg-type]
        processing_jobs=None,  # type: ignore[arg-type]
        analytics=None,  # type: ignore[arg-type]
        audit=None,  # type: ignore[arg-type]
    )


_MONGO_AUTH_FAILURE = lambda: OperationFailure("Authentication failed.", code=18)  # noqa: E731
_MONGO_UNREACHABLE = lambda: ConnectionFailure("connection refused")  # noqa: E731
_POSTGRES_UNREACHABLE = lambda: PostgresOperationalError("connection refused", None, None)  # noqa: E731

_FAILURE_CASES = [
    pytest.param(_MONGO_AUTH_FAILURE, id="mongo-auth-failure"),
    pytest.param(_MONGO_UNREACHABLE, id="mongo-unreachable"),
    pytest.param(_POSTGRES_UNREACHABLE, id="postgres-unreachable"),
]


def _assert_no_driver_detail_leaked(payload: dict) -> None:
    """The entire point of this file: whatever the raw exception said,
    none of it reaches the response body -- only the fixed sanitized
    code. Checked structurally (exact keys/value), not just by
    substring, so a future accidental `"detail": str(exc)` addition
    fails this immediately."""
    assert payload == {"error": "service_unavailable"}
    blob = json.dumps(payload)
    for forbidden in ("OperationFailure", "ConnectionFailure", "OperationalError", "Authentication failed", "pymongo", "mongodb://", "postgresql://", "Traceback"):
        assert forbidden not in blob


@pytest.mark.parametrize("exc_factory", _FAILURE_CASES)
def test_login_never_reports_persistence_failure_as_invalid_credentials(exc_factory):
    """The exact requirement: a DB-infrastructure failure during login
    must never be reported as "wrong email/password" (401
    invalid_credentials) -- it is a different, sanitized 503."""
    bundle = _failing_bundle(exc_factory)
    server, base = serve_in_thread(bundle)
    try:
        status, payload = _json_request(
            f"{base}/api/login", method="POST", data={"email": "someone@example.test", "password": "whatever123"}
        )
        assert status == 503
        assert status != 401
        _assert_no_driver_detail_leaked(payload)
    finally:
        server.shutdown()


@pytest.mark.parametrize("exc_factory", _FAILURE_CASES)
def test_session_restore_returns_sanitized_503_on_persistence_failure(exc_factory):
    bundle = _failing_bundle(exc_factory)
    server, base = serve_in_thread(bundle)
    try:
        status, payload = _json_request(f"{base}/api/session", cookie="rcs_session=some-opaque-token")
        assert status == 503
        assert status != 401
        _assert_no_driver_detail_leaked(payload)
    finally:
        server.shutdown()


@pytest.mark.parametrize("exc_factory", _FAILURE_CASES)
def test_logout_returns_sanitized_503_on_persistence_failure(exc_factory):
    bundle = _failing_bundle(exc_factory)
    server, base = serve_in_thread(bundle)
    try:
        status, payload = _json_request(f"{base}/api/logout", method="POST", cookie="rcs_session=some-opaque-token")
        assert status == 503
        _assert_no_driver_detail_leaked(payload)
    finally:
        server.shutdown()


@pytest.mark.parametrize("exc_factory", _FAILURE_CASES)
def test_health_reports_not_ready_without_leaking_driver_detail(exc_factory):
    bundle = _failing_bundle(exc_factory)
    server, base = serve_in_thread(bundle)
    try:
        status, payload = _json_request(f"{base}/api/health")
        assert status == 503
        assert payload["ok"] is True  # process is alive and answering
        assert payload["ready"] is False  # persistence dependency is not usable
        blob = json.dumps(payload)
        for forbidden in ("OperationFailure", "ConnectionFailure", "OperationalError", "Authentication failed", "pymongo"):
            assert forbidden not in blob
    finally:
        server.shutdown()


def test_health_reports_ready_true_on_a_genuinely_healthy_backend(tmp_path):
    """The negative case: a real, working SQLite-backed engine reports
    ready=True and 200 -- proves the new probe does not just always say
    "not ready", and that it works against a raw engine as well as a
    RepositoryBundle (both are valid values of self.engine, see
    rcs.review.persistence.resolve_bundle)."""
    engine = get_engine(tmp_path / "health.sqlite")
    init_db(engine)
    create_user(engine, email="health-probe@example.test", password="HealthProbe123!", display_name="H", role="ADMIN")
    server, base = serve_in_thread(engine)
    try:
        status, payload = _json_request(f"{base}/api/health")
        assert status == 200
        assert payload == {"ok": True, "ready": True, "demo_auth": False, "entra": False, "password_reset": False, "account_requests": True, "demo": False}
    finally:
        server.shutdown()


def test_a_genuinely_unrelated_bug_is_not_misclassified_as_service_unavailable():
    """Negative control for the classifier itself: an exception that has
    nothing to do with persistence infrastructure (a plain bug) must
    keep going through the existing generic 500 path, not be swallowed
    into the new 503 sanitization -- proves is_persistence_unavailable
    is narrow, not "catch everything and call it service_unavailable"."""
    bundle = _failing_bundle(lambda: RuntimeError("boom: totally unrelated bug"))
    server, base = serve_in_thread(bundle)
    try:
        status, payload = _json_request(
            f"{base}/api/login", method="POST", data={"email": "someone@example.test", "password": "whatever123"}
        )
        assert status == 500
        assert payload["error"] == "server_error"
    finally:
        server.shutdown()


def test_bundle_from_engine_still_works_end_to_end_after_the_health_probe_change(tmp_path):
    """Guards against the health-check probe accidentally becoming the
    only code path exercising UserRepository.get_by_id for a fresh
    Postgres/SQLite bundle -- a plain sanity check that the probe's
    lookup of a nonexistent id is a harmless no-op, not a schema/query
    mismatch that would only show up under Mongo."""
    engine = get_engine(tmp_path / "probe.sqlite")
    init_db(engine)
    bundle = bundle_from_engine(engine)
    assert bundle.users.get_by_id("__rcs_health_probe__") is None
