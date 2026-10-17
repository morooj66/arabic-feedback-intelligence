"""Read-only operational analytics from persisted review_items columns.

Does not change workflow, classification, or generation. Aggregates only
fields already stored: ingested_at, reviewed_at, reviewer_started_at,
reviewer_action, review_status, review_priority, primary_intent,
topic_subtype, sentiment, reviewer_id.

Phase M4C.5: the actual query logic (SQL, Session-bound) now lives in
PostgresAnalyticsRepository.compute_operational_analytics -- this module
keeps only the pure, backend-neutral helpers (range/bucket/date math, no
DB access) that both PostgresAnalyticsRepository and
MongoAnalyticsRepository import, plus a thin backward-compatible wrapper
for compute_operational_analytics's one remaining direct caller
(test_db_portability.py).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy.orm import Session

from config import settings
from rcs.review.clock import now_naive

PENDING = settings.QUEUE_PENDING_STATUSES
OUTCOME_ACTIONS = ("SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED")
OVERDUE_DAYS = 7


def _naive(now: dt.datetime | None) -> dt.datetime:
    return now_naive(now)


def range_bounds(
    key: str,
    now: dt.datetime,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> tuple[dt.datetime | None, dt.datetime]:
    key = (key or "week").strip().lower()
    if key == "today":
        begin = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return begin, now
    if key == "week":
        return now - dt.timedelta(days=7), now
    if key == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), now
    if key == "custom":
        return start, end or now
    return None, now


def _in_window(value: dt.datetime | None, start: dt.datetime | None, end: dt.datetime) -> bool:
    if value is None:
        return False
    if start is not None and value < start:
        return False
    if value > end:
        return False
    return True


def _bucket_age(ingested: dt.datetime | None, now: dt.datetime) -> str | None:
    if ingested is None:
        return None
    days = (now - ingested).total_seconds() / 86400
    if days < 1:
        return "under_day"
    if days < 3:
        return "one_two"
    if days < 8:
        return "three_seven"
    return "over_seven"


def _complete_counts(canonical: tuple[str, ...], rows: list[tuple[Any, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = {str(key): int(n) for key, n in rows if key}
    out = [{"key": key, "count": counts.pop(key, 0)} for key in canonical]
    unmapped = [{"key": key, "count": n} for key, n in sorted(counts.items())]
    return out, unmapped


def compute_operational_analytics(
    session: Session,
    *,
    range_key: str = "week",
    now: dt.datetime | None = None,
    reviewer_id: str | None = None,
    custom_start: dt.datetime | None = None,
    custom_end: dt.datetime | None = None,
    source_file: str | None = None,
    reviewer_user_id: str | None = None,
) -> dict[str, Any]:
    """Phase M4C.5: the actual query logic now lives in
    PostgresAnalyticsRepository.compute_operational_analytics (verbatim
    move); this free function is kept as a thin, behavior-identical
    wrapper for its one remaining direct caller (test_db_portability.py,
    which opens its own Session against an engine it already has) --
    service.py now calls the repository directly via
    ReviewWorkflowService.analytics_repo. `session.get_bind()` recovers
    the Engine the repository needs; the repository opens its own new
    Session against it, which is behaviorally equivalent for a read-only
    query (the portability test always commits before reading)."""
    from rcs.review.repositories.postgres import PostgresAnalyticsRepository

    return PostgresAnalyticsRepository(session.get_bind()).compute_operational_analytics(
        range_key=range_key,
        now=now,
        reviewer_id=reviewer_id,
        custom_start=custom_start,
        custom_end=custom_end,
        source_file=source_file,
        reviewer_user_id=reviewer_user_id,
    )
