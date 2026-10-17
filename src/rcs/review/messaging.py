"""Internal admin <-> reviewer text messaging. One conversation per reviewer (keyed by the reviewer's user id), readable and writable only by that reviewer and by admins.

Every function takes the server-verified `AuthenticatedUser` (never anything client-supplied) and enforces the role rules HERE, so the HTTP layer cannot forget them:
  * REVIEWER: always their own conversation. A different `reviewer_user_id` -> Forbidden (they can never tell whether another conversation exists).
  * ADMIN: must name an existing REVIEWER; an unknown id, or a user who is not a reviewer, -> NotFound.
Text only; no attachments. Retried sends are idempotent through a client-generated `client_message_id`."""

from __future__ import annotations

import datetime as dt
import re
import uuid

from rcs.review.auth import ADMIN, REVIEWER, AuthenticatedUser
from rcs.review.persistence import RepositoryBundle, resolve_bundle
from rcs.review.repositories.contracts import DirectMessageRecord

MAX_BODY_CHARS = 2000
HISTORY_LIMIT = 200
PREVIEW_CHARS = 120
_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class MessagingError(Exception):
    """status: the HTTP status to answer with; code: a short machine-readable error."""

    def __init__(self, status: int, code: str, detail: str = ""):
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


def _iso(value: dt.datetime | None) -> str | None:
    return None if value is None else value.isoformat() + "Z"


def _repo(bundle: RepositoryBundle):
    if bundle.messages is None:
        raise MessagingError(503, "messaging_unavailable")
    return bundle.messages


def resolve_conversation(bundle: RepositoryBundle, actor: AuthenticatedUser, reviewer_user_id: str | None, *, for_send: bool = False) -> str:
    """The reviewer user id of the conversation `actor` is allowed to use, or a MessagingError."""
    if actor.role == REVIEWER:
        if reviewer_user_id and reviewer_user_id != actor.user_id:
            raise MessagingError(403, "forbidden")
        return actor.user_id
    if actor.role != ADMIN:
        raise MessagingError(403, "forbidden")
    target = bundle.users.get_by_id(reviewer_user_id) if reviewer_user_id else None
    if target is None or target.role != REVIEWER:
        raise MessagingError(404, "not_found")
    if for_send and not target.is_active:
        raise MessagingError(409, "reviewer_inactive")
    return target.user_id


def validate_body(body) -> str:
    if not isinstance(body, str):
        raise MessagingError(400, "invalid_message", "الرسالة يجب أن تكون نصًا.")
    text = body.replace("\r\n", "\n").strip()
    if not text:
        raise MessagingError(400, "empty_message", "لا يمكن إرسال رسالة فارغة.")
    if len(text) > MAX_BODY_CHARS:
        raise MessagingError(400, "message_too_long", f"الحد الأقصى {MAX_BODY_CHARS} حرفًا.")
    if _CONTROL.search(text):
        raise MessagingError(400, "invalid_message", "الرسالة تحتوي على محارف غير مسموحة.")
    return text


def _message_dict(m: DirectMessageRecord, names: dict[str, str]) -> dict:
    return {
        "message_id": m.message_id, "reviewer_user_id": m.reviewer_user_id, "sender_user_id": m.sender_user_id, "sender_role": m.sender_role,
        "sender_name": names.get(m.sender_user_id, ""), "body": m.body, "created_at": _iso(m.created_at), "read_at": _iso(m.read_at),
    }


def _names(bundle: RepositoryBundle, ids: set[str]) -> dict[str, str]:
    out = {}
    for uid in ids:
        u = bundle.users.get_by_id(uid)
        out[uid] = (u.display_name or u.email) if u else ""
    return out


def send_message(engine_or_bundle, actor: AuthenticatedUser, *, reviewer_user_id: str | None, body, client_message_id) -> dict:
    bundle = resolve_bundle(engine_or_bundle)
    repo = _repo(bundle)
    conv = resolve_conversation(bundle, actor, reviewer_user_id, for_send=True)
    text = validate_body(body)
    if not isinstance(client_message_id, str) or not _CLIENT_ID.match(client_message_id):
        raise MessagingError(400, "invalid_client_message_id", "معرّف الإرسال غير صالح.")
    record, created = repo.add(
        message_id=uuid.uuid4().hex, reviewer_user_id=conv, sender_user_id=actor.user_id, sender_role=actor.role, body=text,
        client_message_id=client_message_id, created_at=_now(),
    )
    out = _message_dict(record, _names(bundle, {record.sender_user_id}))
    out["duplicate"] = not created
    return out


def get_conversation(engine_or_bundle, actor: AuthenticatedUser, *, reviewer_user_id: str | None) -> dict:
    bundle = resolve_bundle(engine_or_bundle)
    repo = _repo(bundle)
    conv = resolve_conversation(bundle, actor, reviewer_user_id)
    msgs = repo.list_conversation(conv, HISTORY_LIMIT)
    names = _names(bundle, {m.sender_user_id for m in msgs} | {conv})
    other = "ADMIN" if actor.role == REVIEWER else "REVIEWER"
    return {"reviewer_user_id": conv, "reviewer_name": names.get(conv, ""), "messages": [_message_dict(m, names) for m in msgs], "unread": repo.unread_count(conv, other)}


def mark_conversation_read(engine_or_bundle, actor: AuthenticatedUser, *, reviewer_user_id: str | None) -> dict:
    bundle = resolve_bundle(engine_or_bundle)
    repo = _repo(bundle)
    conv = resolve_conversation(bundle, actor, reviewer_user_id)
    marked = repo.mark_read(conv, "ADMIN" if actor.role == REVIEWER else "REVIEWER", _now())     # only what the OTHER side sent is marked: you never read your own messages
    return {"reviewer_user_id": conv, "marked": marked}


def list_conversations(engine_or_bundle, actor: AuthenticatedUser) -> list[dict]:
    """ADMIN only: every active reviewer (plus any inactive reviewer who has messages), newest activity first."""
    if actor.role != ADMIN:
        raise MessagingError(403, "forbidden")
    bundle = resolve_bundle(engine_or_bundle)
    repo = _repo(bundle)
    summaries = repo.summaries()
    reviewers = {u.user_id: u for u in bundle.users.list_active_reviewers()}
    for rid in summaries:
        if rid not in reviewers:
            u = bundle.users.get_by_id(rid)
            if u is not None and u.role == REVIEWER:
                reviewers[rid] = u
    out = []
    for rid, u in reviewers.items():
        s = summaries.get(rid)
        out.append({
            "reviewer_user_id": rid, "reviewer_name": u.display_name or u.email,
            "last_message": None if s is None else {"body": s.last_message.body[:PREVIEW_CHARS], "sender_role": s.last_message.sender_role, "created_at": _iso(s.last_message.created_at)},
            "unread": 0 if s is None else s.unread_from_reviewer,
        })
    out.sort(key=lambda c: c["reviewer_name"])
    with_messages = sorted((c for c in out if c["last_message"]), key=lambda c: c["last_message"]["created_at"], reverse=True)
    return with_messages + [c for c in out if not c["last_message"]]


def unread_total(engine_or_bundle, actor: AuthenticatedUser) -> int:
    bundle = resolve_bundle(engine_or_bundle)
    repo = _repo(bundle)
    if actor.role == REVIEWER:
        return repo.unread_count(actor.user_id, "ADMIN")
    if actor.role == ADMIN:
        return sum(s.unread_from_reviewer for s in repo.summaries().values())
    raise MessagingError(403, "forbidden")
