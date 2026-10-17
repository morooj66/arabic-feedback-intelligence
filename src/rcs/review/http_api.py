"""
Small stdlib HTTP adapter over ReviewWorkflowService.

Chosen over FastAPI/Flask so the reviewer UI can call the existing domain
service without adding a web framework to the ML/runtime requirements.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import re
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from config import settings
from rcs.review.dto import assignable_item_to_dict, detail_to_dict, queue_item_to_dict
from rcs.review.auth import (
    ADMIN,
    REVIEWER,
    AuthenticatedUser,
    authenticate,
    confirm_password_reset,
    end_session,
    login_attempt_allowed,
    password_reset_available,
    record_failed_login,
    record_reset_confirm_failure,
    record_reset_request,
    request_password_reset,
    reset_confirm_allowed,
    reset_request_allowed,
    resolve_client_ip,
    resolve_session,
    start_session,
)
from rcs.review.auth import SESSION_COOKIE_NAME as COOKIE_NAME
from rcs.review.clock import iso_app, parse_range_bound
from rcs.review.db_errors import is_persistence_unavailable
from rcs.review.db_retry import with_transient_retry
from rcs.review.entra_auth import (
    EntraError,
    EntraNotConfigured,
    EntraStateInvalid,
    EntraTokenInvalid,
    EntraUserNotMapped,
    build_authorize_url,
    complete_login as complete_entra_login,
)
from rcs.review.excel_review_flow import DEFAULT_BATCH_SIZE, export_reviewed_workbook, process_uploaded_excel_batch
from rcs.review.import_identity import ImportIdentityCollision, resolve_source_file
from rcs.review.excel_upload import (
    UploadRejected,
    resolve_uploaded_workbook,
    save_uploaded_workbook,
    validate_workbook,
)
from rcs.review.identity import ReviewerIdentity, is_production_env
from rcs.review import playground_api
from rcs.response_intelligence.playground import playground_available
from rcs.review.inspector_api import (
    handle_inspector_classification,
    handle_inspector_eligibility,
    handle_inspector_full_run,
    handle_inspector_grounding,
    handle_inspector_retrieval,
    handle_inspector_validation,
    inspector_architecture_payload,
    inspector_config_payload,
)
from rcs.review.job_runner import start_or_resume_processing
from rcs.demo import state as demo_state
from rcs.review import account_requests, messaging
from rcs.review.persistence import resolve_bundle
from rcs.review.service import ReviewWorkflowService
from rcs.review.sessions import (
    RESUMABLE_SESSION_STATUSES,
    compute_session_counts,
    create_session,
    get_active_session,
    get_job,
    get_session,
    latest_job_for_session,
)
from rcs.review.queue_filters import FilterError, parse_queue_query
from rcs.review.workflow import AttestationNotPermitted, AttestationRequired, DraftValidationError, Forbidden, InvalidTransition, ItemNotFound, VersionConflict

# Never logs a token, cookie, password, or driver exception text that
# might carry connection-string detail -- only short, fixed diagnostic
# strings (exception class names, endpoint identifiers). See
# _log_persistence_failure below, the one call site that writes to this.
logger = logging.getLogger(__name__)


def _log_persistence_failure(where: str, exc: BaseException) -> None:
    logger.warning("persistence unavailable during %s: %s", where, type(exc).__name__)


# --- error sanitisation ------------------------------------------------------------------------------------------------------------------------------------
# An exception's text is NOT safe to return to a client: library and database errors can contain SQL fragments, bound parameters (visitor data), file paths, connection details or another user's
# identifier. Rules enforced below:
#   * an unhandled error answers {"error": "server_error", "reference": "<8 hex>"} and nothing else; the reference ties the response to one server log line;
#   * a message is returned as "detail" only if it is one of OUR curated validation messages (allowlist) - everything else is dropped or replaced by a fixed sentence;
#   * logs record the exception TYPE and code locations (file:line function), never the message, the traceback text or the request body.
_SAFE_VALIDATION_MESSAGES = frozenset({
    "comment_text is required",
    "reviewer_id is required",
    "a valid email is required",
    "password must be at least 8 characters",
    "password must be at most 72 bytes",
})
_SAFE_VALIDATION_PATTERNS = (re.compile(r"^unknown sort: '[A-Za-z0-9_ .-]{0,40}'$"),)
_SAFE_TRANSITION_MESSAGES = frozenset({
    "SEND_AS_IS requires a suggested_response",
    "EDIT_AND_SEND requires non-empty edited_response",
})
_SAFE_TRANSITION_PATTERNS = (re.compile(r"^illegal transition: [A-Z_]{2,40} \+ [A-Z_]{2,40}$"),)
_VERSION_CONFLICT_DETAIL = "the item changed since it was loaded; reload and retry"
_CLAIMED_DETAIL = "this item is claimed by another reviewer"
_TRANSITION_DETAIL = "this action is not allowed in the item's current state"


def _curated(message: str, exact: frozenset, patterns: tuple) -> str | None:
    """The message itself if (and only if) it is one of the curated strings; otherwise None."""
    return message if (message in exact or any(p.match(message) for p in patterns)) else None


def _validation_error_body(exc: BaseException) -> dict:
    body = {"error": "invalid_request"}
    detail = _curated(str(exc), _SAFE_VALIDATION_MESSAGES, _SAFE_VALIDATION_PATTERNS)
    if detail:
        body["detail"] = detail
    return body


def _transition_error_body(exc: BaseException) -> dict:
    message = str(exc)
    if "claimed by" in message:
        return {"error": "claimed", "detail": _CLAIMED_DETAIL}                  # never reveals WHICH reviewer holds the item
    return {"error": "invalid_transition", "detail": _curated(message, _SAFE_TRANSITION_MESSAGES, _SAFE_TRANSITION_PATTERNS) or _TRANSITION_DETAIL}


def _code_locations(exc: BaseException, limit: int = 6) -> str:
    """'file.py:123 func <- ...' for the innermost frames. Built from the traceback STRUCTURE only: no exception text, no source lines, no local variables."""
    frames = traceback.extract_tb(exc.__traceback__)[-limit:]
    return " <- ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}" for f in reversed(frames))


def _log_unhandled(method: str, path: str, exc: BaseException) -> str:
    reference = secrets.token_hex(4)
    cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else "-"
    logger.error("unhandled API error ref=%s %s %s type=%s cause=%s at=%s", reference, method, path, type(exc).__name__, cause, _code_locations(exc))
    return reference


def _as_reviewer_identity(user: AuthenticatedUser) -> ReviewerIdentity:
    """The rest of the app (ReviewWorkflowService, dto.py, etc.) still
    speaks ReviewerIdentity — a small provider-agnostic shape that
    predates Phase 6. This is the one conversion point from a real,
    server-verified AuthenticatedUser to that shape. reviewer_id and
    user_id are the same value going forward (the real user_id); the
    dedicated reviewer_user_id column exists specifically to distinguish
    these authenticated writes from historical/demo reviewer_id strings."""
    return ReviewerIdentity(
        reviewer_id=user.user_id, display_name=user.display_name, user_id=user.user_id, role=user.role
    )


def _qs_multi(qs: dict[str, list[str]], key: str) -> tuple[str, ...] | None:
    """Comma-separated multi-select query param (e.g. `?primary_intent=
    SUGGESTION,COMPLAINT`) -> a tuple of the non-blank values, or None
    when the param is absent/blank -- matches `list_queue`/
    `list_assignable`'s own "None/empty means no filter" convention. An
    unrecognized value is never an error here: it simply matches zero
    rows once it reaches the repository's `IN`/`$in` clause, exactly
    like any other filter in this file."""
    raw = (qs.get(key) or [None])[0]
    if not raw:
        return None
    values = tuple(v.strip() for v in raw.split(",") if v.strip())
    return values or None


def _entra_login_error_url(code: str) -> str:
    """The frontend root is the login page whenever there's no session —
    a plain query-param round trip, no separate "error return URL" env
    var needed. `code` is a short, non-technical machine token
    (LoginPage.tsx maps it to Arabic copy) — never raw provider error
    text, a token, or a secret."""
    base = settings.ENTRA_FRONTEND_RETURN_URL.rstrip("/")
    return f"{base}/?error={quote(code)}"



def _attestation_from_body(body: dict):
    """`{"attestation": {"confirmed": true, "finding_codes": [...]}}` -> FactAttestation, or None when absent / malformed (treated as no attestation)."""
    from rcs.review.service import FactAttestation

    raw = body.get("attestation")
    if not isinstance(raw, dict):
        return None
    codes = raw.get("finding_codes")
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        codes = []
    return FactAttestation(confirmed=raw.get("confirmed") is True, finding_codes=tuple(codes))


class RequestRejected(Exception):
    """A request refused before any business logic ran (body too large, wrong media type, foreign origin). Carries the HTTP status and a stable error code."""

    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status, self.code = status, code


_UPLOAD_CONTENT_TYPES = ("application/octet-stream", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


class ReviewerAPIHandler(BaseHTTPRequestHandler):
    engine = None
    server_version = "RCSReviewerAPI/0.1"
    timeout = settings.REQUEST_SOCKET_TIMEOUT_SECONDS          # per blocking socket operation; a stalled connection is dropped instead of holding a thread forever

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _cors(self) -> None:
        # Explicit allowlist (settings.ALLOWED_ORIGINS), never a reflected
        # arbitrary Origin — Phase 7 production-safety fix. Credentialed
        # CORS (cookies) must never be paired with "allow anything".
        origin = self.headers.get("Origin")
        if origin and origin in settings.ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Expose-Headers", "Content-Disposition, X-Export-Rows, X-Export-Matched, X-Export-Requested, X-Export-Shortfall")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")      # X-Filename: the Excel upload names its file here (a custom header, so it needs a preflight)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self, status: int, payload: dict[str, Any], *, set_cookie: str | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if set_cookie is not None:
            self.send_header("Set-Cookie", set_cookie)
        self.end_headers()
        self.wfile.write(body)

    def _content_length(self) -> int:
        raw = (self.headers.get("Content-Length") or "0").strip()
        if not raw.isdigit():
            raise RequestRejected(400, "invalid_content_length")
        return int(raw)

    def _read_json(self) -> dict[str, Any]:
        length = self._content_length()
        if length <= 0:
            return {}
        if length > settings.MAX_JSON_BODY_BYTES:
            raise RequestRejected(413, "payload_too_large")            # refused before a single body byte is read
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise RequestRejected(400, "invalid_json") from None
        return data if isinstance(data, dict) else {}

    def _own_origin(self) -> str:
        host = (self.headers.get("Host") or "").strip()
        scheme = "https" if self._is_secure_request() else "http"
        return f"{scheme}://{host}" if host else ""

    def _check_state_changing_request(self, path: str) -> None:
        """CSRF defence for every POST, in two layers.

        1. Media type: JSON endpoints accept ONLY `application/json`, the Excel upload only a binary/xlsx type. A cross-site HTML form can send only
           text/plain, urlencoded or multipart bodies without a CORS preflight, so refusing those closes the "simple request" path regardless of cookie policy.
        2. Origin: browsers attach `Origin` to every cross-origin POST. In production (or with RCS_ENFORCE_ORIGIN=1) it must be this server's own origin or one of
           ALLOWED_ORIGINS; a request without Origin but flagged `Sec-Fetch-Site: cross-site` is refused too. Non-browser clients (no Origin, no Sec-Fetch-Site) are
           not a CSRF vector and pass.
        """
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        has_body = (self.headers.get("Content-Length") or "0").strip() not in ("", "0")
        if path == "/api/excel/upload":
            if content_type not in _UPLOAD_CONTENT_TYPES:
                raise RequestRejected(415, "unsupported_media_type")
        elif content_type != "application/json" and (has_body or content_type):
            raise RequestRejected(415, "unsupported_media_type")
        if not (is_production_env() or settings.ENFORCE_ORIGIN_CHECK):
            return
        origin = (self.headers.get("Origin") or "").strip()
        if origin:
            if origin != self._own_origin() and origin not in settings.ALLOWED_ORIGINS:
                raise RequestRejected(403, "origin_not_allowed")
        elif (self.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
            raise RequestRejected(403, "origin_not_allowed")

    def _read_raw_body(self, *, max_bytes: int) -> bytes | None:
        """Read a raw (non-JSON) POST body, e.g. an uploaded file. Returns
        None (having already responded 413) if Content-Length is missing,
        zero, or exceeds `max_bytes` — never buffers more than that cap."""
        length = self._content_length()
        if length <= 0:
            self._json(400, {"error": "empty_body"})
            return None
        if length > max_bytes:
            self._json(413, {"error": "file_too_large"})
            return None
        return self.rfile.read(length)

    def _binary(self, status: int, data: bytes, *, content_type: str, filename: str) -> None:
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        for key, value in (getattr(self, "_export_headers", None) or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str, *, set_cookie: str | None = None) -> None:
        self.send_response(302)
        self._cors()
        self.send_header("Location", location)
        if set_cookie is not None:
            self.send_header("Set-Cookie", set_cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _cookie(self, name: str) -> str | None:
        header = self.headers.get("Cookie") or ""
        for part in header.split(";"):
            if "=" not in part:
                continue
            key, value = part.strip().split("=", 1)
            if key == name:
                return unquote(value)
        return None

    def _is_secure_request(self) -> bool:
        """RCS_ENV=production (set on the deployed backend), or a proxy
        in front of us (Render terminates TLS and forwards plain HTTP)
        says the original request was HTTPS. False for local dev, which
        must keep working without HTTPS."""
        if is_production_env():
            return True
        return (self.headers.get("X-Forwarded-Proto") or "").lower() == "https"

    def _cookie_attrs(self) -> str:
        """Frontend and backend are different origins in the real
        deployment (Vercel + Render) — a cross-site fetch() only carries
        a cookie with SameSite=None, which browsers require to be paired
        with Secure. Local dev stays same-origin (the Vite dev server
        proxies /api to the backend), where Lax without Secure is both
        sufficient and required (no HTTPS locally)."""
        if self._is_secure_request():
            return "SameSite=None; Secure; HttpOnly"
        return "SameSite=Lax; HttpOnly"

    def _current_user(self) -> AuthenticatedUser | None:
        """The real, server-verified identity — resolved by looking the
        session-cookie token up in the user_sessions table (checking
        expiry and that the account is still active) — never decoded
        from anything the client sent. This is the only source of truth
        for role."""
        return resolve_session(self.engine, self._cookie(COOKIE_NAME))

    def _identity(self) -> ReviewerIdentity | None:
        """Back-compat shape for ReviewWorkflowService/dto.py, which
        predate Phase 6 and only know ReviewerIdentity. Always derived
        from _current_user() — never anything client-supplied."""
        user = self._current_user()
        return _as_reviewer_identity(user) if user is not None else None

    def _require_admin(self) -> AuthenticatedUser | None:
        """Returns the current user if authenticated AND role == ADMIN,
        else None. Caller is responsible for the 401 vs 403 distinction
        (401 if _current_user() is None, 403 if authenticated but not
        admin) — kept as two checks at each call site rather than folded
        in here, so the correct status code is always explicit."""
        user = self._current_user()
        if user is None or user.role != ADMIN:
            return None
        return user

    def _require_admin_or_respond(self) -> AuthenticatedUser | None:
        """Writes the 401/403 response itself and returns None when the
        caller isn't an authenticated ADMIN — one call instead of the
        401-then-403 two-check pattern, for the many ADMIN-only Excel/
        import and assignment endpoints. Single _current_user() lookup
        (not two)."""
        user = self._current_user()
        if user is None:
            self._json(401, {"error": "unauthenticated"})
            return None
        if user.role != ADMIN:
            self._json(403, {"error": "forbidden"})
            return None
        return user

    def _playground_admin_or_respond(self, *, status_route: bool = False) -> AuthenticatedUser | None:
        """Internal AI Evaluation Playground gate: authentication first
        (401), ADMIN role second (403), then the feature switch. When the
        playground is disabled (the default, and always in production) every
        route except the status probe answers 404 and does nothing."""
        user = self._require_admin_or_respond()
        if user is None:
            return None
        if not status_route and not playground_available():
            self._json(404, {"error": "playground_disabled"})
            return None
        return user

    def _account_requests_respond(self, call) -> None:
        try:
            self._json(200, call())
        except account_requests.AccountRequestError as exc:
            payload = {"error": exc.code}
            if exc.detail:
                payload["detail"] = exc.detail
            self._json(exc.status, payload)

    def _messaging_respond(self, call) -> None:
        """Runs one messaging operation for the already-authenticated user and answers with its result, or with the MessagingError's status and code."""
        try:
            self._json(200, call())
        except messaging.MessagingError as exc:
            payload = {"error": exc.code}
            if exc.detail:
                payload["detail"] = exc.detail
            self._json(exc.status, payload)

    def _svc(self) -> ReviewWorkflowService:
        return ReviewWorkflowService(self.engine)

    def _require_item_access(self, comment_id: str, user: AuthenticatedUser) -> None:
        """Phase 8 — reviewer isolation. ADMIN always passes (global
        visibility, unchanged). A REVIEWER passes only if the item is
        currently assigned to them (reviewer_user_id == their own
        user_id) — an item assigned to no one and an item assigned to
        someone else are rejected identically (Forbidden -> 403), so a
        REVIEWER can never distinguish "unassigned" from "someone else's"
        by response shape. Raises ItemNotFound (-> 404) for a genuinely
        nonexistent comment_id, exactly like every other lookup in this
        file. Always a fresh, direct read — never trusts anything cached
        or client-supplied."""
        if user.role == ADMIN:
            return
        item = self._svc().repo.get(comment_id)
        if item is None:
            raise ItemNotFound(comment_id)
        if item.reviewer_user_id != user.user_id:
            raise Forbidden(comment_id)

    def _session_payload(self, user: AuthenticatedUser) -> dict[str, Any]:
        return {
            "user_id": user.user_id,
            "display_name": user.display_name,
            "email": user.email,
            "role": user.role,
            "auth_mode": "session",
        }

    def _session_payload_with_counts(self, review_session) -> dict[str, Any]:
        counts = compute_session_counts(self.engine, review_session.source_file)
        job = latest_job_for_session(self.engine, review_session.session_id)
        return {
            "session_id": review_session.session_id,
            "original_filename": review_session.original_filename,
            "source_file": review_session.source_file,
            "status": review_session.status,
            "created_at": iso_app(review_session.created_at),
            "updated_at": iso_app(review_session.updated_at),
            "counts": {
                "total": counts.total,
                "excluded": counts.excluded,
                "duplicate_content": counts.duplicate_content,
                "canonical_total": counts.canonical_total,
                "completed": counts.completed,
                "remaining": counts.remaining,
            },
            "latest_job_id": job.job_id if job else None,
        }

    def _job_payload(self, job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "session_id": job.session_id,
            "status": job.status,
            "current_stage": job.current_stage,
            "rows_done": job.rows_done,
            "rows_total": job.rows_total,
            "started_at": iso_app(job.started_at),
            "updated_at": iso_app(job.updated_at),
            "finished_at": iso_app(job.finished_at),
            "error_message": job.error_message,
        }

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)
        try:
            if path == "/api/health":
                # "ok" means only "this process is alive and answering
                # HTTP" -- process liveness. "ready" additionally proves
                # the configured persistence backend is actually
                # reachable and authenticated, via one cheap, harmless
                # read (an id no real user will ever have) through the
                # exact same repository contract every request uses --
                # never a raw ping/admin command, never backend-specific.
                # A process that looks "ok" but is not "ready" (Mongo
                # unreachable or its credentials rejected, Postgres
                # unreachable) must not be mistaken for healthy by an
                # operator or a load balancer -- hence the paired 503.
                # Never includes exception text, hostnames, or any
                # driver detail in the response; see db_errors.py.
                ready = True
                try:
                    resolve_bundle(self.engine).users.get_by_id("__rcs_health_probe__")
                except Exception as exc:
                    ready = False
                    _log_persistence_failure("health check", exc)
                self._json(
                    200 if ready else 503,
                    {"ok": True, "ready": ready, "demo_auth": False, "entra": settings.ENTRA_CONFIGURED, "password_reset": password_reset_available(), "account_requests": account_requests.ENABLED, "demo": demo_state.is_active()},
                )
                return
            if path == "/api/auth/microsoft/start":
                try:
                    authorize_url = build_authorize_url()
                except EntraNotConfigured:
                    self._redirect(_entra_login_error_url("not_configured"))
                    return
                self._redirect(authorize_url)
                return
            if path == "/api/auth/microsoft/callback":
                error_param = (qs.get("error") or [None])[0]
                code = (qs.get("code") or [None])[0]
                state = (qs.get("state") or [None])[0]
                try:
                    user = complete_entra_login(self.engine, code=code, state=state, error=error_param)
                    token = start_session(self.engine, user_id=user.user_id)
                except EntraNotConfigured:
                    self._redirect(_entra_login_error_url("not_configured"))
                    return
                except EntraStateInvalid:
                    self._redirect(_entra_login_error_url("sso_cancelled_or_expired"))
                    return
                except EntraTokenInvalid:
                    self._redirect(_entra_login_error_url("sso_failed"))
                    return
                except EntraUserNotMapped:
                    self._redirect(_entra_login_error_url("account_not_registered"))
                    return
                except EntraError:
                    self._redirect(_entra_login_error_url("sso_failed"))
                    return
                except Exception as exc:
                    # A persistence failure here (Mongo auth/connection
                    # rejected, Postgres unreachable) is not an Entra
                    # error at all -- but this endpoint is a real
                    # top-level browser navigation, not a JSON fetch(),
                    # so it must still redirect back into the SPA (with
                    # a sanitized error code) rather than fall through to
                    # the generic JSON 500 handler below, which would
                    # dump raw JSON in the middle of the OAuth redirect.
                    if not is_persistence_unavailable(exc):
                        raise
                    _log_persistence_failure("/api/auth/microsoft/callback", exc)
                    self._redirect(_entra_login_error_url("service_unavailable"))
                    return
                cookie = f"{COOKIE_NAME}={token}; Path=/; {self._cookie_attrs()}"
                self._redirect(settings.ENTRA_FRONTEND_RETURN_URL, set_cookie=cookie)
                return
            if path == "/api/session":
                user = self._current_user()
                if user is None:
                    self._json(401, {"error": "unauthenticated"})
                    return
                self._json(200, self._session_payload(user))
                return
            if path == "/api/admin/users":
                admin = self._require_admin()
                if self._current_user() is None:
                    self._json(401, {"error": "unauthenticated"})
                    return
                if admin is None:
                    self._json(403, {"error": "forbidden"})
                    return
                rows = self._svc().user_repo.list_all()
                self._json(
                    200,
                    {
                        "users": [
                            {
                                "user_id": u.user_id,
                                "email": u.email,
                                "display_name": u.display_name,
                                "role": u.role,
                                "is_active": u.is_active,
                                "created_at": iso_app(u.created_at),
                            }
                            for u in rows
                        ]
                    },
                )
                return
            user = self._current_user()
            if user is None:
                self._json(401, {"error": "unauthenticated"})
                return
            ident = _as_reviewer_identity(user)
            # Phase 8: a REVIEWER's counts/queue/analytics are scoped to
            # their own assigned items -- forced to their own user_id
            # unconditionally, never overridable by any request param.
            # ADMIN has no such forced scope (None reproduces the
            # original global behavior), but MAY explicitly ask to view
            # one specific reviewer's items (Completed Reviews filtered
            # by reviewer) via ?reviewer_user_id=; this is read-only and
            # changes nothing about who can claim/act on anything.
            reviewer_scope = (
                user.user_id if user.role != ADMIN else (qs.get("reviewer_user_id") or [None])[0] or None
            )
            if path == "/api/admin/account-requests":
                self._account_requests_respond(lambda: {"requests": account_requests.list_pending(self.engine, user)})
                return
            if path == "/api/messages/conversations":
                self._messaging_respond(lambda: {"conversations": messaging.list_conversations(self.engine, user)})
                return
            if path == "/api/messages/unread":
                self._messaging_respond(lambda: {"unread": messaging.unread_total(self.engine, user)})
                return
            if path == "/api/messages/conversation":
                self._messaging_respond(lambda: messaging.get_conversation(self.engine, user, reviewer_user_id=(qs.get("reviewer_user_id") or [None])[0]))
                return
            if path == "/api/counts":
                self._json(200, self._svc().workflow_counts(reviewer_user_id=reviewer_scope))
                return
            if path == "/api/analytics":
                range_key = (qs.get("range") or ["week"])[0] or "week"
                start_raw = (qs.get("start") or [None])[0]
                end_raw = (qs.get("end") or [None])[0]
                custom_start = parse_range_bound(start_raw, end_of_day=False)
                custom_end = parse_range_bound(end_raw, end_of_day=True)
                source_file = (qs.get("source_file") or [None])[0]
                if not source_file and user.role != ADMIN:
                    # Unscoped/global analytics is the "broader/global
                    # analytics" ADMIN capability (Phase 6 Step 5) — a
                    # REVIEWER only ever gets the scoped (source_file=)
                    # view the Dashboard already uses.
                    self._json(403, {"error": "forbidden"})
                    return
                # Computed BEFORE any response bytes are written, so a
                # retry here can never corrupt an in-flight response —
                # this is the endpoint where the Supabase pooler's
                # transient SSL drops have actually been observed.
                analytics_payload = with_transient_retry(
                    lambda: self._svc().operational_analytics(
                        range_key=range_key,
                        reviewer_id=ident.reviewer_id,
                        custom_start=custom_start,
                        custom_end=custom_end,
                        source_file=source_file or None,
                        reviewer_user_id=reviewer_scope,
                    )
                )
                self._json(
                    200,
                    analytics_payload,
                )
                return
            if path == "/api/admin/reviewers":
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                summaries = self._svc().reviewer_summaries()
                self._json(
                    200,
                    {
                        "reviewers": [
                            {
                                "user_id": s.user_id,
                                "email": s.email,
                                "display_name": s.display_name,
                                "assigned_total": s.assigned_total,
                                "pending": s.pending,
                                "completed": s.completed,
                            }
                            for s in summaries
                        ]
                    },
                )
                return
            if path.startswith("/api/admin/items/") and path.endswith("/history"):
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                comment_id = unquote(path[len("/api/admin/items/") : -len("/history")])
                events = self._svc().item_history(comment_id)
                self._json(
                    200,
                    {
                        "comment_id": comment_id,
                        "events": [
                            {
                                "event_id": e.event_id,
                                "event_type": e.event_type,
                                "actor_user_id": e.actor_user_id,
                                "actor_role": e.actor_role,
                                "from_status": e.from_status,
                                "to_status": e.to_status,
                                "from_assignee_user_id": e.from_assignee_user_id,
                                "to_assignee_user_id": e.to_assignee_user_id,
                                "overrode_recommendation": e.overrode_recommendation,
                                "created_at": e.created_at.isoformat() if e.created_at else None,
                            }
                            for e in events
                        ],
                    },
                )
                return
            if path == "/api/admin/assignable-items":
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                unassigned_raw = (qs.get("unassigned_only") or ["false"])[0].strip().lower()
                if unassigned_raw not in ("1", "true", "yes", "0", "false", "no", ""):
                    raise FilterError("unassigned_only", f"unrecognised value {unassigned_raw!r}")
                unassigned_only = unassigned_raw in ("1", "true", "yes")
                q = parse_queue_query(qs, default_limit=50, default_sort="oldest")
                if q.reviewer_user_id:
                    if unassigned_only:
                        raise FilterError("reviewer_user_id", "cannot be combined with unassigned_only")
                    if self._svc().user_repo.get_by_id(q.reviewer_user_id) is None:
                        raise FilterError("reviewer_user_id", "unknown reviewer")
                page = self._svc().list_assignable_items(
                    unassigned_only=unassigned_only,
                    source_file=q.source_file,
                    offset=q.offset,
                    limit=q.limit,
                    primary_intents=q.primary_intents,
                    topic_subtypes=q.topic_subtypes,
                    search=q.search,
                    sentiments=q.sentiments,
                    priorities=q.priorities,
                    created_from=q.created_from,
                    created_to=q.created_to,
                    reviewer_user_id=q.reviewer_user_id,
                    statuses=q.statuses,
                    sort=q.sort,
                )
                self._json(
                    200,
                    {
                        "items": [assignable_item_to_dict(i) for i in page.items],
                        "total": page.total,
                        "offset": page.offset,
                        "limit": page.limit,
                    },
                )
                return
            if path == "/api/admin/source-summary":
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                summaries = self._svc().source_file_summaries()
                self._json(
                    200,
                    {
                        "sources": [
                            {
                                "source_file": s.source_file,
                                "total_reviewable": s.total_reviewable,
                                "unassigned": s.unassigned,
                                "assigned": s.assigned,
                                "completed": s.completed,
                            }
                            for s in summaries
                        ]
                    },
                )
                return
            if path == "/api/queue":
                q = parse_queue_query(qs)
                # A REVIEWER's own id always wins (reviewer_scope above); a
                # reviewer-supplied reviewer_user_id is ignored, never an
                # override. Only ADMIN's value is used, so only it is checked.
                if q.reviewer_user_id and user.role == ADMIN:
                    if self._svc().user_repo.get_by_id(q.reviewer_user_id) is None:
                        raise FilterError("reviewer_user_id", "unknown reviewer")
                page = self._svc().list_review_queue(
                    statuses=q.statuses,
                    search=q.search,
                    sort=q.sort,
                    offset=q.offset,
                    limit=q.limit,
                    source_file=q.source_file,
                    reviewer_user_id=reviewer_scope,
                    primary_intents=q.primary_intents,
                    topic_subtypes=q.topic_subtypes,
                    reviewer_actions=q.reviewer_actions,
                    sentiments=q.sentiments,
                    priorities=q.priorities,
                    created_from=q.created_from,
                    created_to=q.created_to,
                )
                self._json(
                    200,
                    {
                        "items": [queue_item_to_dict(i) for i in page.items],
                        "total": page.total,
                        "offset": page.offset,
                        "limit": page.limit,
                    },
                )
                return
            if path.startswith("/api/items/"):
                comment_id = unquote(path[len("/api/items/") :])
                self._require_item_access(comment_id, user)
                detail = self._svc().get_review_item(comment_id)
                self._json(200, detail_to_dict(detail))
                return
            if path == "/api/excel/export/completed":
                # Completed-only export, server-side across ALL matching persisted decisions (not the displayed page). Read-only.
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                want_all = (qs.get("all") or [""])[0].strip().lower() in ("1", "true", "yes")
                raw_count = (qs.get("count") or [None])[0]
                if want_all == (raw_count is not None):
                    self._json(400, {"error": "invalid_request", "detail": "send exactly one of: count (positive integer) or all=true"})
                    return
                want = None
                if not want_all:
                    if not raw_count.isdigit() or int(raw_count) < 1:
                        self._json(400, {"error": "invalid_request", "detail": "count must be a positive integer"})
                        return
                    want = int(raw_count)
                forbidden = [k for k in ("filter", "offset", "limit") if k in qs]
                if forbidden:
                    self._json(400, {"error": "invalid_request", "detail": f"unsupported parameter: {forbidden[0]}"})
                    return
                fq = {k: v for k, v in qs.items() if k not in ("all", "count")}
                fq["filter"] = ["completed"]                                  # completed-only; an explicit review_status can only narrow it further
                q = parse_queue_query(fq, default_limit=500, default_sort="oldest")
                if q.reviewer_user_id and self._svc().user_repo.get_by_id(q.reviewer_user_id) is None:
                    raise FilterError("reviewer_user_id", "unknown reviewer")
                ids: list[str] = []
                matched = 0
                offset = 0
                while True:
                    page = self._svc().list_review_queue(
                        statuses=q.statuses, search=q.search, sort=q.sort, offset=offset, limit=500, source_file=q.source_file,
                        reviewer_user_id=q.reviewer_user_id, primary_intents=q.primary_intents, topic_subtypes=q.topic_subtypes,
                        reviewer_actions=q.reviewer_actions, sentiments=q.sentiments, priorities=q.priorities,
                        created_from=q.created_from, created_to=q.created_to,
                    )
                    matched = page.total
                    ids.extend(i.comment_id for i in page.items)
                    offset += len(page.items)
                    if not page.items or offset >= page.total or (want is not None and len(ids) >= want):
                        break
                if want is not None:
                    ids = ids[:want]
                if not ids:
                    self._json(409, {"error": "no_results_to_export"})
                    return
                import tempfile
                from pathlib import Path as _Path

                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
                out_name = f"completed_reviews_{stamp}.xlsx"
                with tempfile.TemporaryDirectory() as tmp:
                    data = export_reviewed_workbook(self.engine, _Path(tmp) / out_name, ids).read_bytes()
                self._export_headers = {"X-Export-Rows": str(len(ids)), "X-Export-Matched": str(matched), "X-Export-Requested": "all" if want is None else str(want),
                                        "X-Export-Shortfall": str(0 if want is None else max(0, want - len(ids)))}
                try:
                    self._binary(200, data, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=out_name)
                finally:
                    self._export_headers = None
                return
            if path == "/api/excel/export":
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                source_file = (qs.get("source_file") or [None])[0]
                explicit_ids = _qs_multi(qs, "comment_ids")
                # A GET query string has a practical length ceiling (many
                # intermediaries truncate well under 8KB) -- an explicit
                # id list large enough to risk that is rejected outright
                # rather than silently truncated, so "export my selection"
                # stays honest about its safe size instead of quietly
                # dropping rows past some invisible limit.
                _MAX_EXPLICIT_EXPORT_IDS = 200
                if explicit_ids is not None and len(explicit_ids) > _MAX_EXPLICIT_EXPORT_IDS:
                    self._json(
                        400,
                        {
                            "error": "too_many_ids",
                            "detail": f"comment_ids supports at most {_MAX_EXPLICIT_EXPORT_IDS} ids per export request",
                        },
                    )
                    return
                comment_ids = None
                if explicit_ids:
                    comment_ids = list(explicit_ids)
                elif source_file:
                    comment_ids = self._svc().repo.comment_ids_for_source_file(source_file)
                    if not comment_ids:
                        self._json(409, {"error": "no_results_to_export"})
                        return
                elif self._svc().repo.count_all() == 0:
                    self._json(409, {"error": "no_results_to_export"})
                    return
                import tempfile
                from pathlib import Path as _Path

                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
                out_name = f"reviewed_responses_{stamp}.xlsx"
                with tempfile.TemporaryDirectory() as tmp:
                    out_path = export_reviewed_workbook(self.engine, _Path(tmp) / out_name, comment_ids)
                    data = out_path.read_bytes()
                self._binary(
                    200,
                    data,
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    filename=out_name,
                )
                return
            if path == "/api/excel/sessions/active":
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                active = get_active_session(self.engine)
                if active is None:
                    self._json(200, {"active": None})
                    return
                self._json(200, {"active": self._session_payload_with_counts(active)})
                return
            if path.startswith("/api/excel/sessions/") and not path.endswith(("/process", "/resume")):
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                session_id = unquote(path[len("/api/excel/sessions/") :])
                found = get_session(self.engine, session_id)
                if found is None:
                    self._json(404, {"error": "not_found"})
                    return
                self._json(200, self._session_payload_with_counts(found))
                return
            if path.startswith("/api/excel/jobs/"):
                if user.role != ADMIN:
                    self._json(403, {"error": "forbidden"})
                    return
                job_id = unquote(path[len("/api/excel/jobs/") :])
                job = get_job(self.engine, job_id)
                if job is None:
                    self._json(404, {"error": "not_found"})
                    return
                self._json(200, self._job_payload(job))
                return
            if path == "/api/admin/inspector/architecture":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, inspector_architecture_payload())
                return
            if path == "/api/admin/inspector/config":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, inspector_config_payload())
                return
            if path == "/api/admin/playground/status":
                if self._playground_admin_or_respond(status_route=True) is None:
                    return
                self._json(200, playground_api.handle_status())
                return
            if path == "/api/admin/playground/cases":
                if self._playground_admin_or_respond() is None:
                    return
                self._json(200, playground_api.handle_list_cases())
                return
            if path == "/api/admin/playground/metrics":
                if self._playground_admin_or_respond() is None:
                    return
                self._json(200, playground_api.handle_metrics())
                return
            self._json(404, {"error": "not_found"})
        except FilterError as exc:
            self._json(400, {"error": "invalid_filter", "param": exc.param, "detail": exc.detail})
        except ItemNotFound:
            self._json(404, {"error": "not_found"})
        except Forbidden:
            self._json(403, {"error": "forbidden"})
        except Exception as exc:
            if is_persistence_unavailable(exc):
                _log_persistence_failure(f"GET {path}", exc)
                self._json(503, {"error": "service_unavailable"})
                return
            self._json(500, {"error": "server_error", "reference": _log_unhandled("GET", path, exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            self._check_state_changing_request(path)
            if path == "/api/login":
                client_ip = resolve_client_ip(
                    self.client_address[0] if self.client_address else None, self.headers.get("X-Forwarded-For"), settings.TRUSTED_PROXIES
                )
                body = self._read_json()
                email = str(body.get("email") or "")
                if not login_attempt_allowed(client_ip, email):
                    self._json(429, {"error": "too_many_attempts"})
                    return
                password = str(body.get("password") or "")
                user = authenticate(self.engine, email=email, password=password)
                if user is None:
                    # Deliberately identical response for unknown email,
                    # wrong password, and a disabled account — never
                    # reveal which reason applied.
                    record_failed_login(client_ip, email)
                    self._json(401, {"error": "invalid_credentials"})
                    return
                token = start_session(self.engine, user_id=user.user_id)
                cookie = f"{COOKIE_NAME}={token}; Path=/; {self._cookie_attrs()}"
                self._json(
                    200,
                    self._session_payload(
                        AuthenticatedUser(
                            user_id=user.user_id, email=user.email, display_name=user.display_name, role=user.role
                        )
                    ),
                    set_cookie=cookie,
                )
                return
            if path == "/api/account-requests":
                # Public. Never a login: the account does not exist until an ADMIN approves it, the role is never read from the body, and the answer is the same
                # whether or not the address is already known (see rcs.review.account_requests).
                client_ip = resolve_client_ip(
                    self.client_address[0] if self.client_address else None, self.headers.get("X-Forwarded-For"), settings.TRUSTED_PROXIES
                )
                body = self._read_json()
                self._account_requests_respond(lambda: (account_requests.submit_request(
                    self.engine, email=body.get("email"), display_name=body.get("display_name"), password=body.get("password"), client_address=client_ip) or {"ok": True}))
                return
            if path == "/api/password-reset/request":
                # Public. Identical answer for registered, unregistered and inactive addresses; fails closed (503, also identical for every address) when no
                # email provider is configured. Every request is rate limited per address and per email whether or not the account exists.
                client_ip = resolve_client_ip(
                    self.client_address[0] if self.client_address else None, self.headers.get("X-Forwarded-For"), settings.TRUSTED_PROXIES
                )
                body = self._read_json()
                email = body.get("email")
                if not isinstance(email, str) or not email.strip() or len(email) > 254 or "@" not in email:
                    self._json(400, {"error": "invalid_request"})
                    return
                if not password_reset_available():
                    self._json(503, {"error": "password_reset_unavailable"})
                    return
                if not reset_request_allowed(client_ip, email):
                    self._json(429, {"error": "too_many_attempts"})
                    return
                record_reset_request(client_ip, email)
                request_password_reset(self.engine, email)
                self._json(200, {"ok": True})
                return
            if path == "/api/password-reset/confirm":
                client_ip = resolve_client_ip(
                    self.client_address[0] if self.client_address else None, self.headers.get("X-Forwarded-For"), settings.TRUSTED_PROXIES
                )
                if not password_reset_available():
                    self._json(503, {"error": "password_reset_unavailable"})
                    return
                if not reset_confirm_allowed(client_ip):
                    self._json(429, {"error": "too_many_attempts"})
                    return
                body = self._read_json()
                token, new_password = body.get("token"), body.get("password")
                if not isinstance(token, str) or not isinstance(new_password, str):
                    self._json(400, {"error": "invalid_request"})
                    return
                try:
                    ok = confirm_password_reset(self.engine, token, new_password)
                except ValueError as exc:
                    self._json(400, {"error": "weak_password", "detail": _curated(str(exc), _SAFE_VALIDATION_MESSAGES, ()) or "password does not meet the policy"})
                    return
                if not ok:
                    record_reset_confirm_failure(client_ip)
                    self._json(400, {"error": "invalid_or_expired_token"})
                    return
                self._json(200, {"ok": True})
                return
            if path == "/api/logout":
                end_session(self.engine, self._cookie(COOKIE_NAME))
                self._json(
                    200,
                    {"ok": True},
                    set_cookie=f"{COOKIE_NAME}=; Path=/; Max-Age=0; {self._cookie_attrs()}",
                )
                return
            if path == "/api/excel/upload":
                # Phase 8 follow-up: import is now ADMIN-owned end to end
                # — a REVIEWER only ever reviews items already assigned
                # to them, never uploads/processes a workbook.
                if self._require_admin_or_respond() is None:
                    return
                raw = self._read_raw_body(max_bytes=settings.EXCEL_UPLOAD_MAX_BYTES)
                if raw is None:
                    return  # _read_raw_body already sent the error response
                qs = parse_qs(urlparse(self.path).query)
                raw_filename = self.headers.get("X-Filename") or (qs.get("filename") or [None])[0]
                filename = unquote(raw_filename) if raw_filename else None
                try:
                    saved = save_uploaded_workbook(filename, raw)
                except UploadRejected as exc:
                    self._json(400, {"error": exc.code, "detail": str(exc)})
                    return
                validation = validate_workbook(saved.path)
                self._json(
                    200,
                    {
                        "upload_token": saved.token,
                        "filename": saved.original_filename,
                        "size_bytes": saved.size_bytes,
                        "valid": validation.ok,
                        "total_rows": validation.total_rows,
                        "errors": validation.errors,
                    },
                )
                return
            if path == "/api/excel/process":
                if self._require_admin_or_respond() is None:
                    return
                body = self._read_json()
                token = body.get("upload_token") or ""
                try:
                    source_path = resolve_uploaded_workbook(token)
                except UploadRejected as exc:
                    self._json(400, {"error": exc.code, "detail": str(exc)})
                    return
                validation = validate_workbook(source_path)
                if not validation.ok:
                    self._json(422, {"error": "invalid_workbook", "detail": "؛ ".join(validation.errors)})
                    return
                batch_size = body.get("batch_size")
                try:
                    source_path = resolve_source_file(self.engine, source_path)
                    summary = process_uploaded_excel_batch(
                        self.engine,
                        source_path,
                        batch_size=int(batch_size) if batch_size else DEFAULT_BATCH_SIZE,
                    )
                except ImportIdentityCollision:
                    self._json(409, {"error": "import_identity_collision", "detail": "اسم الملف مستخدم لدفعة أخرى بمحتوى مختلف؛ لم يُستورد شيء."})
                    return
                payload = dataclasses.asdict(summary)
                payload.pop("comment_ids", None)  # internal detail, not part of the UI-facing summary contract
                self._json(200, payload)
                return
            if path == "/api/excel/sessions":
                if self._require_admin_or_respond() is None:
                    return
                body = self._read_json()
                token = body.get("upload_token") or ""
                try:
                    source_path = resolve_uploaded_workbook(token)
                except UploadRejected as exc:
                    self._json(400, {"error": exc.code, "detail": str(exc)})
                    return
                validation = validate_workbook(source_path)
                if not validation.ok:
                    self._json(422, {"error": "invalid_workbook", "detail": "؛ ".join(validation.errors)})
                    return
                original_name = source_path.name
                try:
                    # A different workbook under an existing name becomes its own batch (content-addressed name); the identical file keeps the old name.
                    source_path = resolve_source_file(self.engine, source_path)
                except ImportIdentityCollision:
                    self._json(409, {"error": "import_identity_collision", "detail": "اسم الملف مستخدم لدفعة أخرى بمحتوى مختلف؛ لم يُستورد شيء."})
                    return
                review_session = create_session(
                    self.engine,
                    original_filename=original_name,
                    source_file=source_path.name,
                    upload_token=token,
                )
                job_id = start_or_resume_processing(self.engine, review_session.session_id)
                self._json(
                    200,
                    {"session": self._session_payload_with_counts(review_session), "job_id": job_id},
                )
                return
            if path == "/api/admin/assign":
                admin = self._require_admin()
                if self._current_user() is None:
                    self._json(401, {"error": "unauthenticated"})
                    return
                if admin is None:
                    self._json(403, {"error": "forbidden"})
                    return
                body = self._read_json()
                comment_ids = body.get("comment_ids")
                target_user_id = str(body.get("reviewer_user_id") or "")
                if not isinstance(comment_ids, list) or not comment_ids or not all(
                    isinstance(c, str) and c for c in comment_ids
                ):
                    self._json(400, {"error": "invalid_request", "detail": "comment_ids must be a non-empty list of strings"})
                    return
                target = self._svc().user_repo.get_by_id(target_user_id)
                if target is None or target.role != REVIEWER or not target.is_active:
                    self._json(400, {"error": "invalid_reviewer", "detail": "reviewer_user_id must name an active REVIEWER"})
                    return
                result = self._svc().assign_reviewer(
                    comment_ids, target_user_id, actor_user_id=admin.user_id, actor_role=admin.role
                )
                self._json(
                    200,
                    {
                        "assigned": list(result.assigned),
                        "skipped_already_decided": list(result.skipped_already_decided),
                        "skipped_not_reviewable": list(result.skipped_not_reviewable),
                        "not_found": list(result.not_found),
                    },
                )
                return
            if path == "/api/admin/assign-by-filter":
                admin = self._require_admin()
                if self._current_user() is None:
                    self._json(401, {"error": "unauthenticated"})
                    return
                if admin is None:
                    self._json(403, {"error": "forbidden"})
                    return
                body = self._read_json()
                target_user_id = str(body.get("reviewer_user_id") or "")
                target = self._svc().user_repo.get_by_id(target_user_id)
                if target is None or target.role != REVIEWER or not target.is_active:
                    self._json(400, {"error": "invalid_reviewer", "detail": "reviewer_user_id must name an active REVIEWER"})
                    return
                want_all = body.get("all") is True
                raw_count = body.get("count")
                if want_all == (raw_count is not None):
                    self._json(400, {"error": "invalid_request", "detail": "send exactly one of: count (positive integer) or all=true"})
                    return
                if not want_all and (isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1):
                    self._json(400, {"error": "invalid_request", "detail": "count must be a positive integer"})
                    return
                filters = body.get("filters") or {}
                if not isinstance(filters, dict) or any(k in filters for k in ("reviewer_user_id", "offset", "limit", "filter")):
                    self._json(400, {"error": "invalid_request", "detail": "filters must be an object of assignment filter parameters"})
                    return
                qs = {str(k): [",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)] for k, v in filters.items() if v not in (None, "", [])}
                try:
                    q = parse_queue_query(qs, default_limit=50, default_sort="oldest")
                except FilterError as exc:
                    self._json(400, {"error": "invalid_filter", "param": exc.param, "detail": exc.detail})
                    return
                result = self._svc().assign_by_filter(
                    target_user_id, count=None if want_all else raw_count, source_file=q.source_file, primary_intents=q.primary_intents,
                    topic_subtypes=q.topic_subtypes, search=q.search, sentiments=q.sentiments, priorities=q.priorities,
                    created_from=q.created_from, created_to=q.created_to, statuses=q.statuses, sort=q.sort,
                    actor_user_id=admin.user_id, actor_role=admin.role,
                )
                self._json(200, {
                    "requested": result.requested, "assigned": result.assigned, "shortfall": result.shortfall,
                    "matched_before": result.matched_before, "remaining_eligible": result.remaining_eligible,
                })
                return
            if path.startswith("/api/excel/sessions/") and path.endswith(("/process", "/resume")):
                if self._require_admin_or_respond() is None:
                    return
                suffix = "/process" if path.endswith("/process") else "/resume"
                session_id = unquote(path[len("/api/excel/sessions/") : -len(suffix)])
                review_session = get_session(self.engine, session_id)
                if review_session is None:
                    self._json(404, {"error": "not_found"})
                    return
                if review_session.status not in RESUMABLE_SESSION_STATUSES:
                    self._json(409, {"error": "session_not_active", "detail": review_session.status})
                    return
                job_id = start_or_resume_processing(self.engine, session_id)
                self._json(200, {"job_id": job_id})
                return
            if path == "/api/admin/inspector/classification":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, handle_inspector_classification(self._read_json()))
                return
            if path == "/api/admin/inspector/eligibility":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, handle_inspector_eligibility(self._read_json()))
                return
            if path == "/api/admin/inspector/retrieval":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, handle_inspector_retrieval(self._read_json()))
                return
            if path == "/api/admin/inspector/grounding":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, handle_inspector_grounding(self._read_json()))
                return
            if path == "/api/admin/inspector/validation":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, handle_inspector_validation(self._read_json()))
                return
            if path == "/api/admin/inspector/full-run":
                if self._require_admin_or_respond() is None:
                    return
                self._json(200, handle_inspector_full_run(self._read_json()))
                return
            if path.startswith("/api/admin/playground/"):
                pg_user = self._playground_admin_or_respond()
                if pg_user is None:
                    return
                pg_body = self._read_json()
                if path == "/api/admin/playground/run":
                    self._json(200, playground_api.handle_run(pg_body, pg_user.email))
                    return
                if path == "/api/admin/playground/compare":
                    self._json(200, playground_api.handle_compare(pg_body, pg_user.email))
                    return
                if path == "/api/admin/playground/cases":
                    self._json(200, playground_api.handle_save_case(pg_body, pg_user.email))
                    return
                if path == "/api/admin/playground/cases/delete":
                    self._json(200, playground_api.handle_delete_case(pg_body, pg_user.email))
                    return
                self._json(404, {"error": "not_found"})
                return
            user = self._current_user()
            if user is None:
                self._json(401, {"error": "unauthenticated"})
                return
            if path in ("/api/admin/account-requests/approve", "/api/admin/account-requests/reject"):
                ar_body = self._read_json()
                rid = ar_body.get("request_id")
                fn = account_requests.approve if path.endswith("/approve") else account_requests.reject
                self._account_requests_respond(lambda: fn(self.engine, user, rid))
                return
            if path in ("/api/messages/send", "/api/messages/read"):
                msg_body = self._read_json()
                rid = msg_body.get("reviewer_user_id")
                rid = rid if isinstance(rid, str) else None
                if path == "/api/messages/send":
                    self._messaging_respond(lambda: messaging.send_message(self.engine, user, reviewer_user_id=rid, body=msg_body.get("body"), client_message_id=msg_body.get("client_message_id")))
                else:
                    self._messaging_respond(lambda: messaging.mark_conversation_read(self.engine, user, reviewer_user_id=rid))
                return
            ident = _as_reviewer_identity(user)
            body = self._read_json()
            expected = body.get("expected_version")
            if expected is not None:
                expected = int(expected)
            if path.endswith("/claim") and path.startswith("/api/items/"):
                comment_id = unquote(path[len("/api/items/") : -len("/claim")])
                self._require_item_access(comment_id, user)
                detail = self._svc().claim_review(comment_id, ident, expected_version=expected)
                self._json(200, detail_to_dict(detail))
                return
            if path.endswith("/send-as-is") and path.startswith("/api/items/"):
                comment_id = unquote(path[len("/api/items/") : -len("/send-as-is")])
                self._require_item_access(comment_id, user)
                detail = self._svc().send_as_is(comment_id, ident, expected_version=expected, attestation=_attestation_from_body(body))
                self._json(200, detail_to_dict(detail))
                return
            if path.endswith("/edit-and-send") and path.startswith("/api/items/"):
                comment_id = unquote(path[len("/api/items/") : -len("/edit-and-send")])
                self._require_item_access(comment_id, user)
                detail = self._svc().edit_and_send(
                    comment_id,
                    ident,
                    body.get("edited_response") or "",
                    expected_version=expected,
                    attestation=_attestation_from_body(body),
                )
                self._json(200, detail_to_dict(detail))
                return
            if path.endswith("/validate-reply") and path.startswith("/api/items/"):
                # Validate-only: nothing is stored, sent, or rewritten. Lets a reviewer re-check a corrected reply before submitting it.
                comment_id = unquote(path[len("/api/items/") : -len("/validate-reply")])
                self._require_item_access(comment_id, user)
                svc = self._svc()
                ok, reason, message_ar = svc.validate_reply_text(comment_id, body.get("reply") or "")
                a = svc.assess_reply(comment_id, body.get("reply") or "", ident)
                self._json(
                    200,
                    {
                        "valid": ok, "reason_code": reason, "message_ar": message_ar,
                        "blocking_findings": list(a.blocking), "attestable_findings": list(a.attestable),
                        "attestation_required": a.attestation_required, "attestation_permitted": a.may_attest,
                    },
                )
                return
            if path.endswith("/no-response") and path.startswith("/api/items/"):
                comment_id = unquote(path[len("/api/items/") : -len("/no-response")])
                self._require_item_access(comment_id, user)
                detail = self._svc().mark_no_response_needed(
                    comment_id,
                    ident,
                    reason=body.get("reason"),
                    expected_version=expected,
                )
                self._json(200, detail_to_dict(detail))
                return
            self._json(404, {"error": "not_found"})
        except RequestRejected as exc:
            self._json(exc.status, {"error": exc.code})
        except ValueError as exc:
            self._json(400, _validation_error_body(exc))
        except ItemNotFound:
            self._json(404, {"error": "not_found"})
        except Forbidden:
            self._json(403, {"error": "forbidden"})
        except VersionConflict:
            self._json(409, {"error": "version_conflict", "detail": _VERSION_CONFLICT_DETAIL})
        except AttestationNotPermitted as exc:
            self._json(403, {"error": "attestation_not_permitted", "attestable_findings": list(exc.finding_codes), "draft_preserved": True, "sent": False})
        except AttestationRequired as exc:
            self._json(
                422,
                {"error": "attestation_required", "attestable_findings": list(exc.finding_codes), "messages_ar": list(exc.messages_ar),
                 "draft_preserved": True, "sent": False},
            )
        except DraftValidationError as exc:
            # Nothing was sent or changed; the stored draft is intact and the reviewer can correct the text and resubmit.
            self._json(
                422,
                {"error": "draft_validation_failed", "reason_code": exc.reason_code, "message_ar": exc.message_ar, "detail": exc.message_ar,
                 "blocking_findings": list(exc.blocking), "attestable_findings": list(exc.attestable), "draft_preserved": True, "sent": False},
            )
        except InvalidTransition as exc:
            self._json(409, _transition_error_body(exc))
        except Exception as exc:
            if is_persistence_unavailable(exc):
                _log_persistence_failure(f"POST {path}", exc)
                self._json(503, {"error": "service_unavailable"})
                return
            self._json(500, {"error": "server_error", "reference": _log_unhandled("POST", path, exc)})


def make_handler(engine) -> type[ReviewerAPIHandler]:
    class Bound(ReviewerAPIHandler):
        pass

    Bound.engine = engine
    return Bound


def serve(engine, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if is_production_env() and not settings.TRUSTED_PROXIES:
        logger.warning("RCS_TRUSTED_PROXIES is empty in production: behind a reverse proxy all clients share the proxy address for login rate limiting")
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    return server


def serve_in_thread(engine, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = serve(engine, host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    return server, f"http://{bound_host}:{bound_port}"
