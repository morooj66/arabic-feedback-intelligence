"""Phase 6 — real, backend-authoritative authentication and roles.

App-owned auth, not Supabase Auth (see Phase 6 report for the
justification): a `users` table with bcrypt-hashed passwords, and a
`user_sessions` table holding opaque, high-entropy session tokens — the
cookie itself carries no identity data, so nothing about it can be
forged. Role is a plain column on `users`, set only by
scripts/manage_users.py; there is no HTTP path that lets a client set or
change its own role.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass

import bcrypt

from rcs.review.db_retry import with_transient_retry
from rcs.review.models import User
from rcs.review.persistence import resolve_bundle

logger = logging.getLogger(__name__)

REVIEWER = "REVIEWER"
ADMIN = "ADMIN"
ROLES = (REVIEWER, ADMIN)

SESSION_LIFETIME = dt.timedelta(hours=12)
SESSION_COOKIE_NAME = "rcs_session"


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str
    display_name: str
    role: str


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        # malformed stored hash — never a match, never a crash
        return False


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def new_user_id() -> str:
    return uuid.uuid4().hex


def check_password_policy(password: str) -> None:
    """The one password policy (create_user and password reset): at least 8 characters, and at most 72 bytes of UTF-8 because bcrypt cannot hash more.
    Raises ValueError with a stable message."""
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    if len(password.encode("utf-8")) > 72:
        raise ValueError("password must be at most 72 bytes")


def create_user(
    engine,
    *,
    email: str,
    password: str,
    display_name: str,
    role: str,
) -> User:
    """The actual `users`-row persistence (duplicate-email check +
    insert) lives in the selected backend's UserRepository.create; this
    function keeps every business-rule validation (role, email shape,
    password length) and every id/hash/timestamp computation exactly
    where they were, then delegates the write through the persistence
    bundle (see rcs.review.persistence) rather than a concrete backend
    class."""
    if role not in ROLES:
        raise ValueError(f"invalid role: {role!r} — must be one of {ROLES}")
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("a valid email is required")
    check_password_policy(password)
    now = _now()
    return resolve_bundle(engine).users.create(
        user_id=new_user_id(),
        email=email,
        password_hash=hash_password(password),
        display_name=display_name.strip() or email,
        role=role,
        is_active=True,
        created_at=now,
        updated_at=now,
    )


def authenticate(engine, *, email: str, password: str) -> User | None:
    """Returns the User row on success. Returns None for: unknown email,
    wrong password, or an inactive account — deliberately the same
    outcome for all three, so the caller can return one generic failure
    message (never revealing which reason applied).

    The `users`-row lookup goes through the selected backend's
    UserRepository.get_by_email, reached via the persistence bundle; the
    active-flag check and password verification stay here exactly as
    before — they are business logic, not persistence."""
    email = (email or "").strip().lower()
    if not email or not password:
        return None
    users = resolve_bundle(engine).users

    def _lookup() -> User | None:
        user = users.get_by_email(email)
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    return with_transient_retry(_lookup)


def session_key(token: str) -> str:
    """What is STORED for a session: the SHA-256 of the cookie token. The raw token exists only in the browser's cookie, so a read of the sessions table (backup,
    replica, SQL injection elsewhere) cannot be turned into a usable login. The token is 256 random bits, so an unsalted hash is sufficient."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def start_session(engine, *, user_id: str) -> str:
    """The actual `user_sessions`-row persistence lives in the selected
    backend's UserSessionRepository.create, reached via the persistence
    bundle; token generation and timestamp computation stay here exactly
    as before -- they are pure logic, not persistence. Both existing
    callers (http_api.py's /api/login and its Entra-callback path) call
    this exact same function unchanged, so the Entra integration point
    keeps working with zero changes to entra_auth.py, and independently
    of which persistence backend is selected."""
    token = new_session_token()
    now = _now()
    resolve_bundle(engine).user_sessions.create(
        session_token=session_key(token), user_id=user_id, created_at=now, expires_at=now + SESSION_LIFETIME
    )
    return token


def resolve_session(engine, token: str | None) -> AuthenticatedUser | None:
    """Looks up the session token, checks expiry, checks the user is
    still active (an account deactivated mid-session is rejected on its
    very next request — no stale access after deactivation), and returns
    the authenticated identity. Everything about the identity — role
    included — comes from this server-side lookup, never from anything
    the client sent.

    The session-row lookup goes through the selected backend's
    UserSessionRepository.get_by_token, reached via the persistence
    bundle; the expiry check (`expires_at <= now`) and the User-side
    active-account revalidation both stay here exactly as before -- a
    session document never caches or replaces the active check, and a
    Mongo TTL index (a housekeeping mechanism, see
    scripts/bootstrap_mongo_indexes.py) can never substitute for this
    immediate, synchronous rejection either."""
    if not token:
        return None
    now = _now()
    bundle = resolve_bundle(engine)

    def _lookup() -> AuthenticatedUser | None:
        sess = bundle.user_sessions.get_by_token(session_key(token))
        if sess is None or sess.expires_at <= now:
            return None
        user = bundle.users.get_by_id(sess.user_id)
        if user is None or not user.is_active:
            return None
        return AuthenticatedUser(
            user_id=user.user_id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
        )

    # Every authenticated request depends on this one read-only lookup —
    # the highest-value place to absorb a transient pooler drop, since it
    # runs before any response bytes have been written.
    return with_transient_retry(_lookup)


def end_session(engine, token: str | None) -> None:
    """Delegates to the selected backend's UserSessionRepository.
    delete_by_token -- idempotent no-op on an unknown/already-gone
    token, exactly as the original delete(...).where(...) was."""
    if not token:
        return
    resolve_bundle(engine).user_sessions.delete_by_token(session_key(token))


# --- self-service password reset ----------------------------------------------------------------------------------------------------------------------
#
# request: for an existing ACTIVE user, supersede older links, create one random single-use token, store ONLY its SHA-256 hash with a <= 30 minute expiry, and email
# the raw token inside a link (URL fragment, so it is never sent to a server or a Referer). Nothing observable differs for an unknown or inactive address.
# confirm: validate the password policy, atomically consume the token hash (unknown / expired / used all look the same), replace the password hash, revoke EVERY
# session of that user and supersede any remaining tokens. Raw tokens and passwords are never logged.
RESET_REQUEST_LIMIT_PER_EMAIL = 3
RESET_REQUEST_LIMIT_PER_ADDRESS = 10
RESET_CONFIRM_FAILURE_LIMIT_PER_ADDRESS = 20
_RESET_WINDOW_SECONDS = 1800
_reset_lock = threading.Lock()
_reset_attempts: dict[str, deque] = defaultdict(deque)
_run_async = lambda fn: threading.Thread(target=fn, daemon=True).start()  # noqa: E731  (tests replace this with a synchronous runner)


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def password_reset_available(sender=None) -> bool:
    """True only when an email sender is configured AND a link base URL is set AND the backend has the token repository. Otherwise reset fails closed."""
    from rcs.review.email_sender import get_email_sender
    from config import settings

    s = sender if sender is not None else get_email_sender()
    return bool(getattr(s, "configured", False) and settings.PASSWORD_RESET_LINK_BASE)


def _reset_check(keys: list[tuple[str, int]]) -> bool:
    now = time.monotonic()
    with _reset_lock:
        for key, limit in keys:
            attempts = _reset_attempts.get(key)
            if not attempts:
                continue
            while attempts and now - attempts[0] > _RESET_WINDOW_SECONDS:
                attempts.popleft()
            if len(attempts) >= limit:
                return False
        return True


def _reset_record(keys: list[tuple[str, int]]) -> None:
    now = time.monotonic()
    with _reset_lock:
        if len(_reset_attempts) > _MAX_TRACKED_KEYS:
            for k in [k for k, d in _reset_attempts.items() if not d or now - d[-1] > _RESET_WINDOW_SECONDS]:
                _reset_attempts.pop(k, None)
        for key, _limit in keys:
            _reset_attempts[key].append(now)


def _reset_request_keys(address: str, email: str | None) -> list[tuple[str, int]]:
    e = (email or "").strip().lower()[:254]
    return [(f"reset-ip:{address}", RESET_REQUEST_LIMIT_PER_ADDRESS), (f"reset-email:{e}", RESET_REQUEST_LIMIT_PER_EMAIL)]


def reset_request_allowed(address: str, email: str | None) -> bool:
    """Counts every request, whether or not the account exists, so the limiter itself can never reveal which addresses are registered."""
    return _reset_check(_reset_request_keys(address, email))


def record_reset_request(address: str, email: str | None) -> None:
    _reset_record(_reset_request_keys(address, email))


def reset_confirm_allowed(address: str) -> bool:
    return _reset_check([(f"reset-confirm-ip:{address}", RESET_CONFIRM_FAILURE_LIMIT_PER_ADDRESS)])


def record_reset_confirm_failure(address: str) -> None:
    _reset_record([(f"reset-confirm-ip:{address}", RESET_CONFIRM_FAILURE_LIMIT_PER_ADDRESS)])


def request_password_reset(engine, email: str, *, sender=None) -> None:
    """Always returns None. Sending runs off the request path so response time does not reveal whether an email was sent."""
    from rcs.review.email_sender import OutgoingEmail, get_email_sender
    from config import settings

    bundle = resolve_bundle(engine)
    mailer = sender if sender is not None else get_email_sender()
    user = bundle.users.get_by_email((email or "").strip().lower())
    if user is None or not user.is_active or bundle.password_reset_tokens is None:
        return None
    now = _now()
    bundle.password_reset_tokens.invalidate_for_user(user.user_id, now)
    token = secrets.token_urlsafe(32)
    bundle.password_reset_tokens.create(
        token_hash=hash_reset_token(token),
        user_id=user.user_id,
        created_at=now,
        expires_at=now + dt.timedelta(minutes=settings.PASSWORD_RESET_TOKEN_TTL_MINUTES),
    )
    link = f"{settings.PASSWORD_RESET_LINK_BASE}/reset-password#token={token}"
    message = OutgoingEmail(
        to=user.email,
        subject="إعادة تعيين كلمة المرور",
        body=(
            "تلقينا طلبًا لإعادة تعيين كلمة المرور لحسابك في نظام مراجعة تعليقات الزوار.\n\n"
            f"افتح الرابط التالي لتعيين كلمة مرور جديدة (صالح لمدة {settings.PASSWORD_RESET_TOKEN_TTL_MINUTES} دقيقة ولمرة واحدة فقط):\n{link}\n\n"
            "إذا لم تطلب ذلك فتجاهل هذه الرسالة، ولن يتغير شيء في حسابك."
        ),
    )

    def _send() -> None:
        try:
            mailer.send(message)
        except Exception as exc:  # noqa: BLE001 - never log the address, the link or the exception text (it could echo either)
            logger.warning("password reset email was not sent: error_type=%s", type(exc).__name__)

    _run_async(_send)
    return None


def confirm_password_reset(engine, token: str, new_password: str) -> bool:
    """True on success. False for ANY unusable token (unknown, expired, already used, inactive or missing account) - indistinguishable to the caller.
    Raises ValueError (before the token is touched) if the new password violates the policy, so a typo does not burn the link."""
    check_password_policy(new_password)
    if not isinstance(token, str) or not token or len(token) > 200:
        return False
    bundle = resolve_bundle(engine)
    if bundle.password_reset_tokens is None:
        return False
    now = _now()
    user_id = bundle.password_reset_tokens.consume(hash_reset_token(token), now)
    if user_id is None:
        return False
    user = bundle.users.get_by_id(user_id)
    if user is None or not user.is_active:
        return False
    if not bundle.users.set_password_hash(user_id, hash_password(new_password), now):
        return False
    bundle.user_sessions.delete_for_user(user_id)
    bundle.password_reset_tokens.invalidate_for_user(user_id, now)
    logger.info("password reset completed: user_id=%s", user_id)
    return True


# --- login rate limiting ----------------------------------------------
#
# Smallest suitable approach for a single persistent process (no Redis):
# an in-memory, per-key sliding window guarded by one lock. Deliberately
# in-process rather than DB-backed — it's a coarse abuse guard, not an
# audit record, and it must not add a database round-trip to every login
# attempt or survive a restart (a restart clearing it is fine; a bad
# actor doesn't get materially more attempts from a rare redeploy).
_RATE_LIMIT_MAX_ATTEMPTS = 5
_RATE_LIMIT_WINDOW_SECONDS = 300  # 5 minutes
_rate_limit_lock = threading.Lock()
_rate_limit_attempts: dict[str, deque] = defaultdict(deque)


def check_login_rate_limit(key: str) -> bool:
    """True if this key (e.g. client IP) is still allowed to attempt a
    login. Call before checking credentials."""
    now = time.monotonic()
    with _rate_limit_lock:
        attempts = _rate_limit_attempts[key]
        while attempts and now - attempts[0] > _RATE_LIMIT_WINDOW_SECONDS:
            attempts.popleft()
        return len(attempts) < _RATE_LIMIT_MAX_ATTEMPTS


def record_login_attempt(key: str) -> None:
    """Record one FAILED login attempt against this key. Never called on
    success — only repeated failures count toward the limit."""
    with _rate_limit_lock:
        _rate_limit_attempts[key].append(time.monotonic())


# --- client address and per-account login limits ----------------------------------------------------------------------------------------------------
#
# Behind a reverse proxy every request arrives from the proxy's address, so a limiter keyed on the socket peer alone makes five failures by ANYONE lock out EVERY
# user. Two fixes, both explicit:
#   1. resolve_client_ip() believes X-Forwarded-For only when the direct peer is in the configured trusted-proxy list (settings.TRUSTED_PROXIES, empty by default),
#      reading the header right to left and returning the first address that is not itself a trusted proxy. From any other peer the header is ignored, so a client
#      cannot pick its own address by sending one.
#   2. Failures are counted per (email, address) pair, with two wider caps: per address (password spraying across emails) and per email (distributed guessing).
#      One user's typos therefore never block a different user, and each cap still bounds brute force.
LOGIN_LIMIT_PER_ACCOUNT_ADDRESS = _RATE_LIMIT_MAX_ATTEMPTS      # one email from one address
LOGIN_LIMIT_PER_ADDRESS = 30                                     # all emails from one address
LOGIN_LIMIT_PER_EMAIL = 20                                       # one email from all addresses
_MAX_TRACKED_KEYS = 20000


def _parse_ip(text: str):
    import ipaddress

    s = (text or "").strip().strip("[]")
    if s.count(":") == 1 and "." in s:                            # "1.2.3.4:5678"
        s = s.split(":", 1)[0]
    try:
        return ipaddress.ip_address(s)
    except ValueError:
        return None


def _trusted_networks(trusted) -> list:
    import ipaddress

    nets = []
    for item in trusted or ():
        try:
            nets.append(ipaddress.ip_network(str(item).strip(), strict=False))
        except ValueError:
            continue                                              # a malformed entry trusts nothing
    return nets


def resolve_client_ip(peer: str | None, forwarded_for: str | None, trusted_proxies=()) -> str:
    """The address to rate-limit by. `peer` is the socket address; `forwarded_for` the raw X-Forwarded-For value (or None)."""
    peer_ip = _parse_ip(peer or "")
    if peer_ip is None:
        return "unknown"
    nets = _trusted_networks(trusted_proxies)
    if not nets or not any(peer_ip in n for n in nets) or not forwarded_for:
        return str(peer_ip)                                       # not a trusted proxy: the header is attacker-controlled, ignore it
    for raw in reversed([p for p in forwarded_for.split(",") if p.strip()]):
        hop = _parse_ip(raw)
        if hop is None:
            return str(peer_ip)                                   # malformed header: fall back to the proxy, never to a guessed value
        if not any(hop in n for n in nets):
            return str(hop)
    return str(peer_ip)                                           # every hop was a trusted proxy


def _login_keys(address: str, email: str | None) -> list[tuple[str, int]]:
    e = (email or "").strip().lower()[:254]
    keys = [(f"ip:{address}", LOGIN_LIMIT_PER_ADDRESS)]
    if e:
        keys += [(f"acct:{e}|{address}", LOGIN_LIMIT_PER_ACCOUNT_ADDRESS), (f"email:{e}", LOGIN_LIMIT_PER_EMAIL)]
    return keys


def _prune_locked(now: float) -> None:
    if len(_rate_limit_attempts) <= _MAX_TRACKED_KEYS:
        return
    for k in [k for k, d in _rate_limit_attempts.items() if not d or now - d[-1] > _RATE_LIMIT_WINDOW_SECONDS]:
        _rate_limit_attempts.pop(k, None)


def login_attempt_allowed(address: str, email: str | None) -> bool:
    """False when ANY applicable cap is exhausted. Call before checking credentials."""
    now = time.monotonic()
    with _rate_limit_lock:
        for key, limit in _login_keys(address, email):
            attempts = _rate_limit_attempts.get(key)
            if not attempts:
                continue
            while attempts and now - attempts[0] > _RATE_LIMIT_WINDOW_SECONDS:
                attempts.popleft()
            if len(attempts) >= limit:
                return False
        return True


def record_failed_login(address: str, email: str | None) -> None:
    """Record one FAILED login against every applicable key. Never called on success."""
    now = time.monotonic()
    with _rate_limit_lock:
        _prune_locked(now)
        for key, _limit in _login_keys(address, email):
            _rate_limit_attempts[key].append(now)


def purge_expired_sessions(engine) -> int:
    """Housekeeping only — never required for correctness (resolve_session
    already rejects expired tokens), just keeps the table from growing
    unbounded. Safe to call anytime, e.g. opportunistically on login.
    Not currently called anywhere in the runtime app or scripts (audited
    fact, not assumed) -- kept and migrated because it is one of this
    phase's four named target functions regardless.

    Delegates to the selected backend's UserSessionRepository.
    delete_expired -- same inclusive `expires_at <= now` predicate,
    same rowcount-as-int return."""
    return resolve_bundle(engine).user_sessions.delete_expired(_now())
