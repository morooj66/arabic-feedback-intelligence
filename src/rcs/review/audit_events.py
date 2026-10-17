"""Append-only review-item audit events (shared by both persistence backends).

An event records WHO did WHAT to a review item and the status / assignee
transition it caused. It deliberately carries no visitor text, no reply
text and no free-form reason: only identifiers, enumerated codes and
timestamps, so the log can be shown to admins without PII exposure.

Repositories write an event in the SAME transaction as the state change
it describes (PostgreSQL: one Session commit; MongoDB: one multi-document
transaction). A failed event write aborts the whole action -- there is no
best-effort path anywhere.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

EVENT_ASSIGNED = "ASSIGNED"
EVENT_REASSIGNED = "REASSIGNED"
EVENT_CLAIMED = "CLAIMED"
EVENT_SEND_AS_IS = "SEND_AS_IS"
EVENT_EDIT_AND_SEND = "EDIT_AND_SEND"
EVENT_NO_RESPONSE_NEEDED = "NO_RESPONSE_NEEDED"
# A send whose reply contains factual claims that could not be verified against approved grounding and that the acting reviewer explicitly
# attested (see rcs.review.service). Same transaction, same columns as the plain events: actor, role, status transition, timestamp.
EVENT_SEND_AS_IS_ATTESTED = "SEND_AS_IS_ATTESTED"
EVENT_EDIT_AND_SEND_ATTESTED = "EDIT_AND_SEND_ATTESTED"

EVENT_TYPES = (
    EVENT_ASSIGNED,
    EVENT_REASSIGNED,
    EVENT_CLAIMED,
    EVENT_SEND_AS_IS,
    EVENT_EDIT_AND_SEND,
    EVENT_NO_RESPONSE_NEEDED,
    EVENT_SEND_AS_IS_ATTESTED,
    EVENT_EDIT_AND_SEND_ATTESTED,
)

RECOMMENDED_STATUS = "NO_RESPONSE_RECOMMENDED"

# Actor recorded when a caller supplies none (scripts, seeds, legacy
# callers). The event is still written -- never skipped.
SYSTEM_ACTOR = "system"


@dataclass(frozen=True)
class ReviewEvent:
    event_id: str
    review_item_id: str
    event_type: str
    actor_user_id: str
    actor_role: str | None
    from_status: str | None
    to_status: str | None
    from_assignee_user_id: str | None
    to_assignee_user_id: str | None
    overrode_recommendation: bool
    created_at: dt.datetime


def build_event_fields(
    *,
    review_item_id: str,
    event_type: str,
    actor_user_id: str | None,
    actor_role: str | None,
    from_status: str | None,
    to_status: str | None,
    from_assignee_user_id: str | None,
    to_assignee_user_id: str | None,
    created_at: dt.datetime,
) -> dict:
    """Validated column values for one event (a plain dict, so the
    SQLAlchemy and Mongo writers share one definition). Raises ValueError
    for an unknown event type, which aborts the surrounding transaction.

    `overrode_recommendation` is true exactly when a reviewer wrote a
    reply (EDIT_AND_SEND) on an item that was still a model
    NO_RESPONSE_RECOMMENDED recommendation."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown audit event type: {event_type!r}")
    if not review_item_id:
        raise ValueError("audit event requires review_item_id")
    return {
        "event_id": uuid.uuid4().hex,
        "review_item_id": review_item_id,
        "event_type": event_type,
        "actor_user_id": (actor_user_id or SYSTEM_ACTOR),
        "actor_role": actor_role,
        "from_status": from_status,
        "to_status": to_status,
        "from_assignee_user_id": from_assignee_user_id,
        "to_assignee_user_id": to_assignee_user_id,
        "overrode_recommendation": bool(
            event_type in (EVENT_EDIT_AND_SEND, EVENT_EDIT_AND_SEND_ATTESTED) and from_status == RECOMMENDED_STATUS
        ),
        "created_at": created_at,
    }


def event_from_fields(fields: dict) -> ReviewEvent:
    return ReviewEvent(
        event_id=fields["event_id"],
        review_item_id=fields["review_item_id"],
        event_type=fields["event_type"],
        actor_user_id=fields["actor_user_id"],
        actor_role=fields.get("actor_role"),
        from_status=fields.get("from_status"),
        to_status=fields.get("to_status"),
        from_assignee_user_id=fields.get("from_assignee_user_id"),
        to_assignee_user_id=fields.get("to_assignee_user_id"),
        overrode_recommendation=bool(fields.get("overrode_recommendation")),
        created_at=fields["created_at"],
    )
