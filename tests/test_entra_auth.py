"""Phase 9 — Microsoft Entra ID SSO.

Entra only verifies identity; the existing users/user_sessions tables and
REVIEWER/ADMIN roles remain the sole source of authorization. No live
network/JWKS/token calls are ever made here — _exchange_code_for_id_token
and _validate_id_token (the two functions that would otherwise reach
Microsoft) are monkeypatched in every test that exercises the callback.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlparse

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from rcs.review import entra_auth
from rcs.review.auth import ADMIN, REVIEWER, authenticate, create_user
from rcs.review.entra_auth import (
    EntraNotConfigured,
    EntraStateInvalid,
    EntraTokenInvalid,
    EntraUserNotMapped,
    build_authorize_url,
    complete_login,
    resolve_local_user_for_entra_login,
)
from rcs.review.http_api import serve_in_thread
from rcs.review.models import User, UserSession
from rcs.review.persistence import bundle_from_mongo
from rcs.review.repository import get_engine, init_db
from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake


def _engine(tmp_path, name="entra.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _configure_entra(monkeypatch):
    monkeypatch.setattr(settings, "ENTRA_TENANT_ID", "test-tenant")
    monkeypatch.setattr(settings, "ENTRA_CLIENT_ID", "test-client")
    monkeypatch.setattr(settings, "ENTRA_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(settings, "ENTRA_CONFIGURED", True)
    monkeypatch.setattr(settings, "ENTRA_FRONTEND_RETURN_URL", "http://localhost:5173/")


def _unconfigure_entra(monkeypatch):
    monkeypatch.setattr(settings, "ENTRA_CONFIGURED", False)


def _mock_callback_success(monkeypatch, *, email: str, extra_claims: dict | None = None):
    """Bypasses the real token exchange + JWKS signature validation —
    exactly the two calls that would otherwise reach Microsoft — with a
    fake-but-realistic claims set, so the rest of the real flow (state
    consumption, nonce check inside complete_login's caller, local-user
    resolution, session creation) runs unmodified."""
    calls = {"exchange": 0, "validate": 0}

    def fake_exchange(code, code_verifier):
        calls["exchange"] += 1
        assert code == "fake-auth-code"
        assert code_verifier  # a real PKCE verifier was generated and threaded through
        return "fake.id.token"

    def fake_validate(id_token, expected_nonce):
        calls["validate"] += 1
        assert id_token == "fake.id.token"
        claims = {"email": email, "nonce": expected_nonce, "sub": "entra-sub-123"}
        claims.update(extra_claims or {})
        return claims

    monkeypatch.setattr(entra_auth, "_exchange_code_for_id_token", fake_exchange)
    monkeypatch.setattr(entra_auth, "_validate_id_token", fake_validate)
    return calls


# --- unit-level: resolve_local_user_for_entra_login --------------------------


def test_resolve_local_user_matches_active_user_by_normalized_email(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="Reviewer@Example.com", password="realpassword123", display_name="R", role=REVIEWER)
    user = resolve_local_user_for_entra_login(engine, "reviewer@example.com")
    assert user.email == "reviewer@example.com"
    assert user.role == REVIEWER


def test_resolve_local_user_rejects_unmapped_email_without_creating_one(tmp_path):
    engine = _engine(tmp_path)
    before = _user_count(engine)
    with pytest.raises(EntraUserNotMapped):
        resolve_local_user_for_entra_login(engine, "nobody@example.com")
    assert _user_count(engine) == before  # never auto-creates


def test_resolve_local_user_rejects_inactive_user(tmp_path):
    engine = _engine(tmp_path)
    user = create_user(engine, email="disabled@example.com", password="realpassword123", display_name="D", role=REVIEWER)
    with Session(engine) as db:
        row = db.get(User, user.user_id)
        row.is_active = False
        db.commit()
    with pytest.raises(EntraUserNotMapped):
        resolve_local_user_for_entra_login(engine, "disabled@example.com")


def _user_count(engine) -> int:
    with Session(engine) as db:
        return len(db.execute(select(User.user_id)).all())


# --- Phase M4D.3: entra_auth.py now resolves the local user through --------
# UserRepository.get_by_email instead of a direct SQLAlchemy query --------


def test_resolve_local_user_delegates_to_user_repository(tmp_path, monkeypatch):
    """Proves the seam itself: resolve_local_user_for_entra_login must
    call PostgresUserRepository.get_by_email, not run its own SQL."""
    import rcs.review.entra_auth as entra_auth_module
    from rcs.review.repositories.postgres import PostgresUserRepository

    engine = _engine(tmp_path)
    create_user(engine, email="reviewer@example.com", password="realpassword123", display_name="R", role=REVIEWER)

    calls: list[str] = []
    real_get_by_email = PostgresUserRepository.get_by_email

    def _spy(self, email):
        calls.append(email)
        return real_get_by_email(self, email)

    monkeypatch.setattr(PostgresUserRepository, "get_by_email", _spy)
    user = entra_auth_module.resolve_local_user_for_entra_login(engine, "reviewer@example.com")
    assert calls == ["reviewer@example.com"]
    assert user.email == "reviewer@example.com"


def test_resolve_local_user_email_match_is_exact_not_case_insensitive(tmp_path):
    """Normalization is the caller's job (_extract_verified_email runs
    .strip().lower() before this is ever reached) -- the lookup itself,
    now going through UserRepository.get_by_email, must still be a
    plain exact-string match, matching rcs.review.auth's own behavior
    (see the M4D.1 report's identical finding for authenticate())."""
    engine = _engine(tmp_path)
    create_user(engine, email="lower@example.com", password="realpassword123", display_name="L", role=REVIEWER)
    with pytest.raises(EntraUserNotMapped):
        resolve_local_user_for_entra_login(engine, "LOWER@example.com")
    assert resolve_local_user_for_entra_login(engine, "lower@example.com").email == "lower@example.com"


def test_resolve_local_user_works_against_a_mongo_backed_bundle_not_only_postgres():
    """Production login/DB boundary hardening — the actual regression
    this closes: resolve_local_user_for_entra_login used to construct
    PostgresUserRepository(engine) directly, which breaks the instant
    `engine` is a Mongo-backed RepositoryBundle instead of a SQLAlchemy
    engine (exactly what the served application hands every handler
    under PERSISTENCE_BACKEND=mongo — see rcs.review.persistence and
    scripts/serve_reviewer_api.py, which always passes the bundle, never
    a bare engine, to serve()). No live MongoDB connection — bundle_from_
    mongo wraps the same in-memory fake every other Mongo contract test
    in this project uses."""
    bundle = bundle_from_mongo(_FakeDatabase())
    create_user(bundle, email="mongo.sso@example.test", password="realpassword123", display_name="Mongo SSO", role=REVIEWER)

    user = resolve_local_user_for_entra_login(bundle, "mongo.sso@example.test")
    assert user.email == "mongo.sso@example.test"
    assert user.role == REVIEWER

    with pytest.raises(EntraUserNotMapped):
        resolve_local_user_for_entra_login(bundle, "nobody@example.test")


def test_callback_succeeds_end_to_end_against_a_mongo_backed_bundle(monkeypatch):
    """HTTP-level counterpart of the test above: the full /api/auth/
    microsoft/start -> callback -> session-cookie -> /api/session flow,
    running against a served application whose self.engine is a
    Mongo-backed RepositoryBundle -- the exact topology
    scripts/serve_reviewer_api.py produces under PERSISTENCE_BACKEND=
    mongo, which every other Entra HTTP test in this file (all of which
    pass a raw SQLAlchemy engine to serve_in_thread) never exercised."""
    _configure_entra(monkeypatch)
    bundle = bundle_from_mongo(_FakeDatabase())
    create_user(bundle, email="mongo.sso.http@example.test", password="realpassword123", display_name="Mongo SSO HTTP", role=REVIEWER)
    _mock_callback_success(monkeypatch, email="mongo.sso.http@example.test")
    server, base = serve_in_thread(bundle)
    try:
        _, start_location, _ = _raw_get(base, "/api/auth/microsoft/start")
        state = urllib.parse.parse_qs(urlparse(start_location).query)["state"][0]
        status, location, set_cookie = _raw_get(
            base, f"/api/auth/microsoft/callback?code=fake-auth-code&state={state}"
        )
        assert status == 302
        assert location == "http://localhost:5173/"
        assert set_cookie and "rcs_session=" in set_cookie
        cookie_value = set_cookie.split(";")[0]

        status2, payload, _ = _json_get(base, "/api/session", cookie=cookie_value)
        assert status2 == 200
        assert payload["email"] == "mongo.sso.http@example.test"
        assert payload["role"] == "REVIEWER"
    finally:
        server.shutdown()


def test_entra_auth_module_contains_no_direct_user_sql():
    """Static guard: after the M4D.3 cleanup, entra_auth.py must not
    construct its own SQLAlchemy query against the User table -- every
    lookup goes through UserRepository."""
    import inspect

    import rcs.review.entra_auth as entra_auth_module

    source = inspect.getsource(entra_auth_module)
    assert "select(User)" not in source
    assert "db.get(User" not in source
    assert "session.get(User" not in source


# --- unit-level: complete_login orchestration ---------------------------------


def test_complete_login_requires_configuration(tmp_path):
    engine = _engine(tmp_path)
    with pytest.raises(EntraNotConfigured):
        complete_login(engine, code="x", state="y", error=None)


def test_complete_login_rejects_unknown_state(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    with pytest.raises(EntraStateInvalid):
        complete_login(engine, code="fake-auth-code", state="never-issued", error=None)


def test_complete_login_rejects_provider_error_without_touching_token_exchange(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    calls = _mock_callback_success(monkeypatch, email="whoever@example.com")
    url = build_authorize_url()
    state = urllib.parse.parse_qs(urlparse(url).query)["state"][0]
    with pytest.raises(EntraStateInvalid):
        complete_login(engine, code=None, state=state, error="access_denied")
    assert calls["exchange"] == 0  # never reached the token endpoint


def test_complete_login_state_is_single_use(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    create_user(engine, email="a@example.com", password="realpassword123", display_name="A", role=REVIEWER)
    _mock_callback_success(monkeypatch, email="a@example.com")
    url = build_authorize_url()
    state = urllib.parse.parse_qs(urlparse(url).query)["state"][0]
    complete_login(engine, code="fake-auth-code", state=state, error=None)
    with pytest.raises(EntraStateInvalid):
        complete_login(engine, code="fake-auth-code", state=state, error=None)  # replay rejected


def test_complete_login_succeeds_for_existing_active_user_and_ignores_role_claim(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    admin = create_user(engine, email="real.admin@example.com", password="realpassword123", display_name="Real Admin", role=ADMIN)
    # An attacker-controlled-looking claim in the token must never be trusted for role.
    _mock_callback_success(monkeypatch, email="real.admin@example.com", extra_claims={"roles": ["REVIEWER"], "jobTitle": "Reviewer"})
    url = build_authorize_url()
    state = urllib.parse.parse_qs(urlparse(url).query)["state"][0]
    user = complete_login(engine, code="fake-auth-code", state=state, error=None)
    assert user.user_id == admin.user_id
    assert user.role == ADMIN  # from the local DB, never from the token


def test_token_exchange_failure_is_reported_as_entra_token_invalid(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)

    def failing_exchange(code, code_verifier):
        raise entra_auth.EntraTokenInvalid("token endpoint request failed: simulated network error")

    monkeypatch.setattr(entra_auth, "_exchange_code_for_id_token", failing_exchange)
    url = build_authorize_url()
    state = urllib.parse.parse_qs(urlparse(url).query)["state"][0]
    with pytest.raises(EntraTokenInvalid):
        complete_login(engine, code="fake-auth-code", state=state, error=None)


# --- HTTP-level: the two new endpoints ----------------------------------------


def _raw_get(base: str, path: str, *, cookie: str | None = None):
    """http.client, not urllib.request — urllib auto-follows redirects,
    which would hide the 302 Location/Set-Cookie headers these endpoints
    exist to produce."""
    parsed = urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    headers = {"Cookie": cookie} if cookie else {}
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    resp.read()
    location = resp.getheader("Location")
    set_cookie = resp.getheader("Set-Cookie")
    conn.close()
    return resp.status, location, set_cookie


def test_start_endpoint_redirects_to_microsoft_with_pkce_and_state(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        status, location, _ = _raw_get(base, "/api/auth/microsoft/start")
        assert status == 302
        assert location.startswith("https://login.microsoftonline.com/test-tenant/oauth2/v2.0/authorize?")
        params = urllib.parse.parse_qs(urlparse(location).query)
        assert params["client_id"] == ["test-client"]
        assert params["response_type"] == ["code"]
        assert params["code_challenge_method"] == ["S256"]
        assert "state" in params and "nonce" in params and "code_challenge" in params
    finally:
        server.shutdown()


def test_start_endpoint_redirects_with_error_when_not_configured(tmp_path, monkeypatch):
    _unconfigure_entra(monkeypatch)
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        status, location, _ = _raw_get(base, "/api/auth/microsoft/start")
        assert status == 302
        assert location == "http://localhost:5173/?error=not_configured"
    finally:
        server.shutdown()


def test_callback_success_creates_a_real_app_session_for_an_existing_reviewer(tmp_path, monkeypatch):
    """End-to-end (minus the real Microsoft hop): start -> callback ->
    session cookie -> GET /api/session resolves the authenticated user.
    Also pins the callback's redirect_uri and final Location to the same
    origin as ENTRA_FRONTEND_RETURN_URL — the invariant that keeps the
    cookie this test asserts on actually reachable by the real browser's
    SPA once the Vite dev proxy is in the picture (a plain http.client
    round trip like this one doesn't enforce browser cookie-domain
    scoping, so that invariant has to be asserted explicitly here rather
    than relying on this test alone to catch a host mismatch)."""
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    create_user(engine, email="sso.reviewer@example.com", password="realpassword123", display_name="SSO Reviewer", role=REVIEWER)
    _mock_callback_success(monkeypatch, email="sso.reviewer@example.com")
    server, base = serve_in_thread(engine)
    try:
        status, start_location, _ = _raw_get(base, "/api/auth/microsoft/start")
        state = urllib.parse.parse_qs(urlparse(start_location).query)["state"][0]
        redirect_uri = urllib.parse.parse_qs(urlparse(start_location).query)["redirect_uri"][0]

        status, location, set_cookie = _raw_get(
            base, f"/api/auth/microsoft/callback?code=fake-auth-code&state={state}"
        )
        assert status == 302
        assert location == "http://localhost:5173/"
        assert set_cookie and "rcs_session=" in set_cookie
        cookie_value = set_cookie.split(";")[0]

        # The host Microsoft would actually redirect the browser to
        # (redirect_uri) must be same-origin as the host the browser ends
        # up on afterwards (location) — otherwise the cookie set on the
        # callback response is scoped to a host the SPA never talks to.
        assert _origin(redirect_uri) == _origin(location)

        status2, payload, _ = _json_get(base, "/api/session", cookie=cookie_value)
        assert status2 == 200
        assert payload["email"] == "sso.reviewer@example.com"
        assert payload["role"] == "REVIEWER"
        assert payload["display_name"] == "SSO Reviewer"
    finally:
        server.shutdown()


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    return (parsed.scheme, parsed.hostname, parsed.port)


def test_redirect_uri_is_same_origin_as_frontend_return_url_so_the_vite_proxy_keeps_the_cookie_visible(
    tmp_path, monkeypatch
):
    """Regression guard for the 127.0.0.1-vs-localhost host mismatch: the
    Entra callback is a real top-level browser navigation straight to
    ENTRA_REDIRECT_URI (never through the SPA's own fetch(), so Vite's
    dev proxy never touches it from the browser's point of view). If that
    host differs from where the SPA itself runs (ENTRA_FRONTEND_RETURN_URL),
    the session cookie the callback sets is scoped to a host the SPA never
    calls /api/session from, and the user lands back on the app silently
    logged out even though a real session row was created server-side.
    This asserts the two share one origin, and that the callback path
    falls under Vite's "/api" proxy prefix (see web/vite.config.ts) so the
    browser's direct navigation there still reaches this backend."""
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        _, start_location, _ = _raw_get(base, "/api/auth/microsoft/start")
        redirect_uri = urllib.parse.parse_qs(urlparse(start_location).query)["redirect_uri"][0]
    finally:
        server.shutdown()

    assert _origin(redirect_uri) == _origin(settings.ENTRA_FRONTEND_RETURN_URL), (
        f"ENTRA_REDIRECT_URI ({redirect_uri}) must be same-origin as "
        f"ENTRA_FRONTEND_RETURN_URL ({settings.ENTRA_FRONTEND_RETURN_URL}) — otherwise the "
        "browser's session cookie from the callback is invisible to the SPA."
    )
    assert urlparse(redirect_uri).path.startswith("/api/")


def test_callback_rejects_unmapped_account_without_creating_a_user_or_session(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    before = _user_count(engine)
    _mock_callback_success(monkeypatch, email="stranger@example.com")
    server, base = serve_in_thread(engine)
    try:
        _, start_location, _ = _raw_get(base, "/api/auth/microsoft/start")
        state = urllib.parse.parse_qs(urlparse(start_location).query)["state"][0]
        status, location, set_cookie = _raw_get(
            base, f"/api/auth/microsoft/callback?code=fake-auth-code&state={state}"
        )
        assert status == 302
        assert location == "http://localhost:5173/?error=account_not_registered"
        assert set_cookie is None
    finally:
        server.shutdown()
    assert _user_count(engine) == before
    with Session(engine) as db:
        assert db.execute(select(UserSession.session_token)).all() == []


def test_callback_rejects_inactive_account(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    user = create_user(engine, email="was.active@example.com", password="realpassword123", display_name="X", role=REVIEWER)
    with Session(engine) as db:
        row = db.get(User, user.user_id)
        row.is_active = False
        db.commit()
    _mock_callback_success(monkeypatch, email="was.active@example.com")
    server, base = serve_in_thread(engine)
    try:
        _, start_location, _ = _raw_get(base, "/api/auth/microsoft/start")
        state = urllib.parse.parse_qs(urlparse(start_location).query)["state"][0]
        status, location, set_cookie = _raw_get(
            base, f"/api/auth/microsoft/callback?code=fake-auth-code&state={state}"
        )
        assert status == 302
        assert location == "http://localhost:5173/?error=account_not_registered"
        assert set_cookie is None
    finally:
        server.shutdown()


def test_callback_rejects_state_mismatch(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        status, location, set_cookie = _raw_get(
            base, "/api/auth/microsoft/callback?code=fake-auth-code&state=totally-made-up"
        )
        assert status == 302
        assert location == "http://localhost:5173/?error=sso_cancelled_or_expired"
        assert set_cookie is None
    finally:
        server.shutdown()


def test_callback_handles_user_cancellation_and_provider_errors_safely(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        _, start_location, _ = _raw_get(base, "/api/auth/microsoft/start")
        state = urllib.parse.parse_qs(urlparse(start_location).query)["state"][0]
        status, location, set_cookie = _raw_get(
            base, f"/api/auth/microsoft/callback?error=access_denied&state={state}"
        )
        assert status == 302
        assert location == "http://localhost:5173/?error=sso_cancelled_or_expired"
        assert set_cookie is None
        # Never leaks the raw provider error code/description into the redirect.
        assert "access_denied" not in location
    finally:
        server.shutdown()


def test_callback_handles_provider_failure_during_token_exchange_safely(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)

    def failing_exchange(code, code_verifier):
        raise entra_auth.EntraTokenInvalid("token endpoint request failed: simulated network error")

    monkeypatch.setattr(entra_auth, "_exchange_code_for_id_token", failing_exchange)
    server, base = serve_in_thread(engine)
    try:
        _, start_location, _ = _raw_get(base, "/api/auth/microsoft/start")
        state = urllib.parse.parse_qs(urlparse(start_location).query)["state"][0]
        status, location, set_cookie = _raw_get(
            base, f"/api/auth/microsoft/callback?code=fake-auth-code&state={state}"
        )
        assert status == 302
        assert location == "http://localhost:5173/?error=sso_failed"
        assert set_cookie is None
        assert "simulated network error" not in location  # never leaks internal error text
    finally:
        server.shutdown()


class _FakeSigningKey:
    def __init__(self, public_key):
        self.key = public_key


class _FakeJwksClient:
    def __init__(self, public_key):
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._public_key)


@pytest.fixture()
def rsa_keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _sign_claims(private_key, claims: dict) -> str:
    import jwt as _jwt

    return _jwt.encode(claims, private_key, algorithm="RS256")


class TestRealIdTokenValidation:
    """Exercises the *real* _validate_id_token — real RS256 signature
    verification and real iss/aud/nonce/exp checks — against a locally
    generated key pair. Only _get_jwks_client is faked (to hand back our
    test key instead of fetching Microsoft's real JWKS), so this proves
    the actual security-critical validation logic is correct, not just
    that the orchestration trusts whatever a mock returns."""

    def _base_claims(self, *, nonce="expected-nonce"):
        import time

        now = int(time.time())
        return {
            "iss": f"https://login.microsoftonline.com/{settings.ENTRA_TENANT_ID}/v2.0",
            "aud": settings.ENTRA_CLIENT_ID,
            "exp": now + 300,
            "iat": now,
            "nonce": nonce,
            "email": "real.crypto@example.com",
            "sub": "entra-sub-xyz",
        }

    def test_valid_signed_token_is_accepted(self, monkeypatch, rsa_keypair):
        _configure_entra(monkeypatch)
        private_key, public_key = rsa_keypair
        monkeypatch.setattr(entra_auth, "_get_jwks_client", lambda: _FakeJwksClient(public_key))
        token = _sign_claims(private_key, self._base_claims())
        claims = entra_auth._validate_id_token(token, "expected-nonce")
        assert claims["email"] == "real.crypto@example.com"

    def test_wrong_audience_is_rejected(self, monkeypatch, rsa_keypair):
        _configure_entra(monkeypatch)
        private_key, public_key = rsa_keypair
        monkeypatch.setattr(entra_auth, "_get_jwks_client", lambda: _FakeJwksClient(public_key))
        claims = self._base_claims()
        claims["aud"] = "someone-elses-client-id"
        token = _sign_claims(private_key, claims)
        with pytest.raises(EntraTokenInvalid):
            entra_auth._validate_id_token(token, "expected-nonce")

    def test_wrong_issuer_is_rejected(self, monkeypatch, rsa_keypair):
        _configure_entra(monkeypatch)
        private_key, public_key = rsa_keypair
        monkeypatch.setattr(entra_auth, "_get_jwks_client", lambda: _FakeJwksClient(public_key))
        claims = self._base_claims()
        claims["iss"] = "https://attacker.example.com/v2.0"
        token = _sign_claims(private_key, claims)
        with pytest.raises(EntraTokenInvalid):
            entra_auth._validate_id_token(token, "expected-nonce")

    def test_expired_token_is_rejected(self, monkeypatch, rsa_keypair):
        _configure_entra(monkeypatch)
        private_key, public_key = rsa_keypair
        monkeypatch.setattr(entra_auth, "_get_jwks_client", lambda: _FakeJwksClient(public_key))
        claims = self._base_claims()
        claims["exp"] = claims["iat"] - 10  # already expired
        token = _sign_claims(private_key, claims)
        with pytest.raises(EntraTokenInvalid):
            entra_auth._validate_id_token(token, "expected-nonce")

    def test_nonce_mismatch_is_rejected(self, monkeypatch, rsa_keypair):
        _configure_entra(monkeypatch)
        private_key, public_key = rsa_keypair
        monkeypatch.setattr(entra_auth, "_get_jwks_client", lambda: _FakeJwksClient(public_key))
        token = _sign_claims(private_key, self._base_claims(nonce="some-other-nonce"))
        with pytest.raises(EntraTokenInvalid):
            entra_auth._validate_id_token(token, "expected-nonce")

    def test_token_signed_by_a_different_key_is_rejected(self, monkeypatch, rsa_keypair):
        """The attacker-controlled scenario: a token that is otherwise
        perfectly well-formed but signed by a key that isn't in
        Microsoft's real JWKS — proves signature verification is real,
        not skipped."""
        from cryptography.hazmat.primitives.asymmetric import rsa

        _configure_entra(monkeypatch)
        _real_private, real_public = rsa_keypair
        attacker_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        monkeypatch.setattr(entra_auth, "_get_jwks_client", lambda: _FakeJwksClient(real_public))
        token = _sign_claims(attacker_private, self._base_claims())
        with pytest.raises(EntraTokenInvalid):
            entra_auth._validate_id_token(token, "expected-nonce")


def test_existing_password_login_is_unaffected_by_entra_being_configured(tmp_path, monkeypatch):
    _configure_entra(monkeypatch)
    engine = _engine(tmp_path)
    create_user(engine, email="password.user@example.com", password="realpassword123", display_name="Password User", role=ADMIN)
    user = authenticate(engine, email="password.user@example.com", password="realpassword123")
    assert user is not None
    assert user.role == ADMIN


def _json_get(base: str, path: str, *, cookie: str | None = None):
    req = urllib.request.Request(f"{base}{path}", headers={"Cookie": cookie} if cookie else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), err.headers
