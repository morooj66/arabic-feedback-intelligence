"""Self-service "new account request" with admin approval.

Rules (each is covered by tests/test_account_requests.py):
  * A request is NOT a login. Until an ADMIN approves it there is no user row, so the requester cannot authenticate (the login answer is the same generic failure as
    for any unknown account).
  * The requester never supplies a role. Approval always creates a REVIEWER. ADMIN accounts are provisioned only by scripts/manage_users.py.
  * No account enumeration: submitting an e-mail that already has an account, or an earlier request (pending, approved or rejected), looks exactly like a fresh
    submission (`ok`), and nothing is stored or changed.
  * The password the requester chose is checked against the one password policy and stored only as a bcrypt hash; on approval that hash becomes the account's.
  * Rate limited per client address and per e-mail, with a global cap on pending requests, so the public endpoint cannot be used to flood the admin queue.
  * Only authenticated ADMINs can list, approve or reject (enforced by the HTTP layer through the admin check, and again by the functions below).
Limitations (documented, not hidden): the e-mail address is NOT verified (no mail is sent to the requester), and the requester is not notified of the decision."""

from __future__ import annotations

import datetime as dt
import threading
import time
import uuid
from collections import defaultdict, deque

from rcs.review.auth import ADMIN, REVIEWER, AuthenticatedUser, check_password_policy, hash_password, new_user_id
from rcs.review.persistence import resolve_bundle

REQUEST_LIMIT_PER_ADDRESS = 5
REQUEST_LIMIT_PER_EMAIL = 3
PENDING_CAP = 500
_WINDOW_SECONDS = 3600
MAX_NAME_CHARS = 100
MAX_EMAIL_CHARS = 254
PENDING_LIST_LIMIT = 200

ENABLED = True            # the demo mode switches this off (rcs.demo.activation): accounts there are invite-only
_lock = threading.Lock()
_attempts: dict[str, deque] = defaultdict(deque)


class AccountRequestError(Exception):
    def __init__(self, status: int, code: str, detail: str = ""):
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail


def reset_limits_for_tests() -> None:
    with _lock:
        _attempts.clear()


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


def _allowed_and_record(address: str, email: str) -> bool:
    """Every submission counts (valid or not, new account or not), so the limiter reveals nothing about which addresses exist."""
    horizon = time.monotonic() - _WINDOW_SECONDS
    with _lock:
        for key, cap in ((f"a:{address}", REQUEST_LIMIT_PER_ADDRESS), (f"e:{email}", REQUEST_LIMIT_PER_EMAIL)):
            q = _attempts[key]
            while q and q[0] < horizon:
                q.popleft()
            if len(q) >= cap:
                return False
        for key in (f"a:{address}", f"e:{email}"):
            _attempts[key].append(time.monotonic())
    return True


def submit_request(engine_or_bundle, *, email, display_name, password, client_address: str) -> None:
    """Returns None on every well-formed submission, whatever happened behind it. Raises AccountRequestError only for malformed input (400) or rate limiting (429)."""
    if not ENABLED:
        raise AccountRequestError(503, "account_requests_unavailable")
    if not isinstance(email, str) or not isinstance(display_name, str) or not isinstance(password, str):
        raise AccountRequestError(400, "invalid_request")
    email = email.strip().lower()
    name = " ".join(display_name.split())
    if not email or "@" not in email or len(email) > MAX_EMAIL_CHARS or any(c.isspace() for c in email) or email.count("@") != 1 or email.startswith("@") or email.endswith("@"):
        raise AccountRequestError(400, "invalid_email", "أدخل بريدًا إلكترونيًا صالحًا.")
    if not name or len(name) > MAX_NAME_CHARS:
        raise AccountRequestError(400, "invalid_name", "أدخل الاسم (حتى 100 حرف).")
    try:
        check_password_policy(password)
    except ValueError:
        raise AccountRequestError(400, "invalid_password", "كلمة المرور يجب ألا تقل عن 8 أحرف.") from None
    if not _allowed_and_record(client_address or "unknown", email):
        raise AccountRequestError(429, "too_many_requests")
    bundle = resolve_bundle(engine_or_bundle)
    repo = bundle.account_requests
    if repo is None:
        raise AccountRequestError(503, "account_requests_unavailable")
    password_hash = hash_password(password)                                          # always computed, so response TIME does not reveal whether the e-mail is known
    if bundle.users.get_by_email(email) is not None:
        return                                                                       # already has an account: indistinguishable from success, nothing stored
    if repo.count_pending() >= PENDING_CAP:
        return                                                                       # queue full: same answer; the admin queue is protected from flooding
    repo.create(request_id=uuid.uuid4().hex, email=email, display_name=name, password_hash=password_hash, created_at=_now())


def _require_admin(actor: AuthenticatedUser) -> None:
    if actor is None or actor.role != ADMIN:
        raise AccountRequestError(403, "forbidden")


def _public(r) -> dict:
    return {"request_id": r.request_id, "email": r.email, "display_name": r.display_name, "created_at": r.created_at.isoformat() + "Z"}


def list_pending(engine_or_bundle, actor: AuthenticatedUser) -> list[dict]:
    _require_admin(actor)
    repo = resolve_bundle(engine_or_bundle).account_requests
    if repo is None:
        raise AccountRequestError(503, "account_requests_unavailable")
    return [_public(r) for r in repo.list_pending(PENDING_LIST_LIMIT)]


def approve(engine_or_bundle, actor: AuthenticatedUser, request_id: str) -> dict:
    """Creates the REVIEWER account (active) from the stored request. The decision is an atomic PENDING -> APPROVED, so two admins cannot both approve it."""
    _require_admin(actor)
    bundle = resolve_bundle(engine_or_bundle)
    repo = bundle.account_requests
    if repo is None:
        raise AccountRequestError(503, "account_requests_unavailable")
    req = repo.get(request_id) if isinstance(request_id, str) else None
    if req is None:
        raise AccountRequestError(404, "not_found")
    if req.status != "PENDING":
        raise AccountRequestError(409, "already_decided")
    if bundle.users.get_by_email(req.email) is not None:
        repo.decide(request_id, "REJECTED", actor.user_id, _now())
        raise AccountRequestError(409, "account_exists", "يوجد حساب بهذا البريد بالفعل؛ أُغلق الطلب.")
    if repo.decide(request_id, "APPROVED", actor.user_id, _now()) is None:
        raise AccountRequestError(409, "already_decided")
    now = _now()
    user = bundle.users.create(user_id=new_user_id(), email=req.email, password_hash=req.password_hash, display_name=req.display_name, role=REVIEWER, is_active=True,
                               created_at=now, updated_at=now)
    return {"request_id": request_id, "user_id": user.user_id, "email": user.email, "role": user.role}


def reject(engine_or_bundle, actor: AuthenticatedUser, request_id: str) -> dict:
    _require_admin(actor)
    repo = resolve_bundle(engine_or_bundle).account_requests
    if repo is None:
        raise AccountRequestError(503, "account_requests_unavailable")
    if not isinstance(request_id, str) or repo.get(request_id) is None:
        raise AccountRequestError(404, "not_found")
    if repo.decide(request_id, "REJECTED", actor.user_id, _now()) is None:
        raise AccountRequestError(409, "already_decided")
    return {"request_id": request_id, "status": "REJECTED"}
