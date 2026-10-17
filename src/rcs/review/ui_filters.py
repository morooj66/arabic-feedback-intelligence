"""Map reviewer-UI filter keys onto existing review_status values.

Does not invent workflow states. Used by the thin HTTP adapter only.
"""

from __future__ import annotations

from config import settings

# Sidebar / queue filters → backend statuses
FILTER_STATUSES: dict[str, tuple[str, ...]] = {
    "pending": settings.QUEUE_PENDING_STATUSES,
    "awaiting": ("DRAFT_READY", "REVIEW_ONLY", "NO_RESPONSE_RECOMMENDED"),
    "in_review": ("IN_REVIEW",),
    "approved": ("APPROVED", "EDITED_AND_APPROVED", "SENDING", "SENT"),
    # Final human decisions only; a model recommendation is pending work.
    "no_reply": ("NO_REPLY",),
    "followup": ("SEND_FAILED",),
    # Admin "المراجعات المنجزة" (Completed Reviews) destination — every
    # terminal/decided status, composed from the exact tuples above
    # rather than a new list, so there is one place ("approved" ∪
    # "no_reply" ∪ "followup") that defines what "decided" means.
    "completed": ("APPROVED", "EDITED_AND_APPROVED", "SENDING", "SENT", "NO_REPLY", "SEND_FAILED"),
    "all": (
        "DRAFT_READY",
        "REVIEW_ONLY",
        "IN_REVIEW",
        "APPROVED",
        "EDITED_AND_APPROVED",
        "NO_REPLY",
        "NO_RESPONSE_RECOMMENDED",
        "SENDING",
        "SENT",
        "SEND_FAILED",
    ),
}


def statuses_for_filter(key: str | None) -> tuple[str, ...] | None:
    if not key:
        return None
    return FILTER_STATUSES.get(key.strip().lower())
