"""Phase 9 — Microsoft Entra ID SSO: identity verification only.

Entra never creates or authorizes anyone. This module's entire job is to
answer one question — "is this really the person behind this Microsoft
account?" — via a standard server-side OAuth 2.0 Authorization Code +
PKCE / OIDC flow, and then hand back a *verified email address*.
Everything after that (does this email belong to an existing, active
local user, and what role do they have) is decided exclusively from the
existing `users` table (see resolve_local_user_for_entra_login) — the
same table and the same rcs.review.auth.start_session()/session-cookie
mechanism app-owned password login already uses. No new user is ever
created here, and role is never taken from anything Microsoft sends.

The local-user lookup itself goes through the same persistence
composition root (rcs.review.persistence.resolve_bundle) every other
runtime auth path uses — this module used to be a documented exception
that instantiated PostgresUserRepository directly, which is exactly the
kind of implicit backend selection the production login/DB boundary
audit exists to close (see the audit report). There is no
Mongo-vs-Postgres decision anywhere in this file.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from urllib.request import Request

import jwt
from jwt import PyJWKClient

from config import settings
from rcs.review.models import User
from rcs.review.persistence import resolve_bundle

_AUTHORIZE_URL_TMPL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
_TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
_JWKS_URL_TMPL = "https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys"
_ISSUER_TMPL = "https://login.microsoftonline.com/{tenant}/v2.0"

_STATE_TTL_SECONDS = 600  # 10 minutes — long enough for a real login, short enough to bound the in-memory store
_HTTP_TIMEOUT_SECONDS = 10


class EntraError(RuntimeError):
    """Base class — every subtype maps to one clean, non-technical Arabic
    message at the HTTP layer. Never exposes provider error text, tokens,
    or secrets in its own message."""


class EntraNotConfigured(EntraError):
    pass


class EntraStateInvalid(EntraError):
    """Missing/expired/already-used state, or Microsoft reported an error
    (user cancelled, provider failure) on the callback."""


class EntraTokenInvalid(EntraError):
    """Token exchange failed, or the returned ID token failed signature/
    issuer/audience/nonce/expiry validation."""


class EntraUserNotMapped(EntraError):
    """The verified email has no active local user — by design, never
    auto-creates one."""


@dataclass(frozen=True)
class PendingLogin:
    nonce: str
    code_verifier: str
    created_at: float


# In-memory, single-process store — exactly the same tradeoff already
# made for rcs.review.auth's login rate limiter: a login flow only ever
# needs to survive a few minutes on the same process, a restart clearing
# it just means an in-flight SSO redirect has to be retried, and it must
# never add a database round-trip to every OAuth step.
_pending_lock = threading.Lock()
_pending_logins: dict[str, PendingLogin] = {}

_jwks_lock = threading.Lock()
_jwks_client: PyJWKClient | None = None
_jwks_client_tenant: str | None = None


def _now() -> float:
    return time.monotonic()


def _purge_expired_locked() -> None:
    cutoff = _now() - _STATE_TTL_SECONDS
    expired = [state for state, pending in _pending_logins.items() if pending.created_at < cutoff]
    for state in expired:
        del _pending_logins[state]


def _require_configured() -> None:
    if not settings.ENTRA_CONFIGURED:
        raise EntraNotConfigured("Microsoft Entra SSO is not configured on this server")


def _generate_pkce_pair() -> tuple[str, str]:
    """(code_verifier, code_challenge) — RFC 7636 S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url() -> str:
    """Starts a new login attempt: generates state/nonce/PKCE, stores them
    server-side keyed by state, and returns the full Microsoft
    authorization URL to redirect the browser to."""
    _require_configured()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier, challenge = _generate_pkce_pair()

    with _pending_lock:
        _purge_expired_locked()
        _pending_logins[state] = PendingLogin(nonce=nonce, code_verifier=verifier, created_at=_now())

    params = {
        "client_id": settings.ENTRA_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": settings.ENTRA_REDIRECT_URI,
        "response_mode": "query",
        "scope": "openid profile email",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    base = _AUTHORIZE_URL_TMPL.format(tenant=urllib.parse.quote(settings.ENTRA_TENANT_ID, safe=""))
    return f"{base}?{urllib.parse.urlencode(params)}"


def _consume_pending_login(state: str | None) -> PendingLogin:
    """Single-use: a state can only ever be redeemed once, whether the
    callback succeeds or fails, so a captured/replayed callback URL can't
    be reused."""
    if not state:
        raise EntraStateInvalid("missing state")
    with _pending_lock:
        _purge_expired_locked()
        pending = _pending_logins.pop(state, None)
    if pending is None:
        raise EntraStateInvalid("unknown, expired, or already-used state")
    return pending


def _exchange_code_for_id_token(code: str, code_verifier: str) -> str:
    """The one server-to-server call that uses the client secret — never
    reachable from, or visible to, the frontend."""
    body = urllib.parse.urlencode(
        {
            "client_id": settings.ENTRA_CLIENT_ID,
            "client_secret": settings.ENTRA_CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings.ENTRA_REDIRECT_URI,
            "code_verifier": code_verifier,
        }
    ).encode("ascii")
    url = _TOKEN_URL_TMPL.format(tenant=urllib.parse.quote(settings.ENTRA_TENANT_ID, safe=""))
    req = Request(url, data=body, method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            import json as _json

            payload = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise EntraTokenInvalid(f"token endpoint request failed: {exc}") from exc
    except ValueError as exc:
        raise EntraTokenInvalid("token endpoint returned a non-JSON response") from exc

    id_token = payload.get("id_token")
    if not id_token:
        raise EntraTokenInvalid("token endpoint response did not include an id_token")
    return id_token


def _get_jwks_client() -> PyJWKClient:
    """Cached across calls (PyJWKClient has its own internal TTL cache for
    the fetched keys) — only rebuilt if the tenant changes, which never
    happens at runtime outside of tests."""
    global _jwks_client, _jwks_client_tenant
    with _jwks_lock:
        if _jwks_client is None or _jwks_client_tenant != settings.ENTRA_TENANT_ID:
            url = _JWKS_URL_TMPL.format(tenant=urllib.parse.quote(settings.ENTRA_TENANT_ID, safe=""))
            _jwks_client = PyJWKClient(url, timeout=_HTTP_TIMEOUT_SECONDS)
            _jwks_client_tenant = settings.ENTRA_TENANT_ID
        return _jwks_client


def _validate_id_token(id_token: str, expected_nonce: str) -> dict:
    """Full OIDC validation: RS256 signature against Microsoft's published
    JWKS, issuer, audience, expiry (PyJWT's default checks), and nonce.
    Raises EntraTokenInvalid on any failure — deliberately generic, never
    echoes the raw token or provider-internal detail back to the caller."""
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.ENTRA_CLIENT_ID,
            issuer=_ISSUER_TMPL.format(tenant=settings.ENTRA_TENANT_ID),
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except jwt.PyJWTError as exc:
        raise EntraTokenInvalid(f"id_token failed validation: {exc}") from exc

    if claims.get("nonce") != expected_nonce:
        raise EntraTokenInvalid("id_token nonce did not match this login attempt")
    return claims


def _extract_verified_email(claims: dict) -> str:
    """Only the `email` claim is trusted for identity matching — it is
    the claim Microsoft's docs describe as the account's verified email,
    unlike `preferred_username`/UPN, which is not guaranteed to be a real
    or verified email address. Matches "safest available claim" from the
    Phase 9 spec exactly, not the most convenient one."""
    email = claims.get("email")
    if not email or not isinstance(email, str) or "@" not in email:
        raise EntraTokenInvalid("id_token did not include a verified email claim")
    return email.strip().lower()


def resolve_local_user_for_entra_login(engine, email: str) -> User:
    """The only place Entra identity turns into local authorization.
    Looks up an existing, active user by normalized email — never
    creates one, never trusts a role claim from Microsoft. Raises
    EntraUserNotMapped if there's no match, exactly as if the account
    doesn't exist as far as this app is concerned.

    Production login/DB boundary hardening: the raw `users`-row lookup
    goes through resolve_bundle(engine).users.get_by_email — the same
    composition-root seam rcs.review.auth already uses for local
    password login — never a directly-instantiated PostgresUserRepository.
    `engine` here is exactly what the served application's HTTP handler
    holds as `self.engine`, which under PERSISTENCE_BACKEND=mongo is a
    RepositoryBundle, not a SQLAlchemy engine; instantiating
    PostgresUserRepository(engine) directly (the previous, Entra-only
    exemption from this seam) would either silently query Postgres while
    every other login path used Mongo, or — the actual failure mode,
    since the served app always hands handlers a bundle regardless of
    backend — crash outright with a raw SQLAlchemy error the first time
    anyone tried Microsoft SSO. The active-check and EntraUserNotMapped
    raise stay here exactly as before, since they are Entra-specific
    authorization logic, not persistence."""
    user = resolve_bundle(engine).users.get_by_email(email)
    if user is None or not user.is_active:
        raise EntraUserNotMapped(email)
    return user


def complete_login(engine, *, code: str | None, state: str | None, error: str | None) -> User:
    """Full callback-side flow: validates state, rejects a provider-
    reported error (user cancelled, provider failure) cleanly, exchanges
    the code, validates the id_token, and resolves the local user. Raises
    one of the EntraError subtypes on any failure — the HTTP layer maps
    each to a specific, non-technical Arabic message and never re-raises
    the underlying exception text to the client."""
    _require_configured()
    pending = _consume_pending_login(state)
    if error:
        raise EntraStateInvalid(f"provider reported an error: {error}")
    if not code:
        raise EntraStateInvalid("missing authorization code")

    id_token = _exchange_code_for_id_token(code, pending.code_verifier)
    claims = _validate_id_token(id_token, pending.nonce)
    email = _extract_verified_email(claims)
    return resolve_local_user_for_entra_login(engine, email)
