"""Strict query-string parsing for the queue / assignment / completed lists.

Every value is validated against the frozen taxonomy and workflow
vocabularies, and an unknown value raises FilterError (HTTP 400) instead
of being passed to the repository where it would silently match nothing
-- or, for a bad tab or sort key, silently fall back to a default list.
Parsing is pure: it touches no persistence and changes no taxonomy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from config import settings
from rcs.review.clock import parse_range_bound
from rcs.review.ui_filters import FILTER_STATUSES

# Public sort keys. `newest`/`oldest` (and their explicit `ingested_*`
# aliases) order by INGESTION time; `comment_*` order by the visitor
# comment's own `comment_created_at`. Every ordering ends with the
# unique comment_id so OFFSET/LIMIT pages are stable.
SORT_KEYS = (
    "priority",
    "newest",
    "oldest",
    "ingested_newest",
    "ingested_oldest",
    "comment_newest",
    "comment_oldest",
)
PRIORITY_VALUES = (1, 2, 3)
MAX_LIMIT = 500


class FilterError(ValueError):
    """A client-supplied filter/sort/paging value is not recognised."""

    def __init__(self, param: str, detail: str):
        super().__init__(f"{param}: {detail}")
        self.param = param
        self.detail = detail


@dataclass(frozen=True)
class QueueQuery:
    filter_key: str | None
    statuses: tuple[str, ...] | None  # tab statuses ∩ explicit review_status
    search: str | None
    sort: str
    offset: int
    limit: int
    source_file: str | None
    primary_intents: tuple[str, ...] | None
    topic_subtypes: tuple[str, ...] | None
    sentiments: tuple[str, ...] | None
    priorities: tuple[int, ...] | None
    reviewer_actions: tuple[str, ...] | None
    reviewer_user_id: str | None
    created_from: dt.datetime | None
    created_to: dt.datetime | None


def _first(qs: dict[str, list[str]], key: str) -> str | None:
    raw = (qs.get(key) or [None])[0]
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _multi(qs: dict[str, list[str]], key: str, allowed: tuple[Any, ...], *, cast=str) -> tuple | None:
    raw = _first(qs, key)
    if raw is None:
        return None
    values: list = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = cast(piece)
        except (TypeError, ValueError):
            raise FilterError(key, f"unrecognised value {piece!r}") from None
        if value not in allowed:
            raise FilterError(key, f"unrecognised value {piece!r}")
        if value not in values:
            values.append(value)
    return tuple(values) or None


def _int(qs, key: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = _first(qs, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise FilterError(key, "must be an integer") from None
    if value < minimum or (maximum is not None and value > maximum):
        raise FilterError(key, "out of range")
    return value


def _date(qs, key: str, *, end_of_day: bool) -> dt.datetime | None:
    raw = _first(qs, key)
    if raw is None:
        return None
    try:
        return parse_range_bound(raw, end_of_day=end_of_day)
    except ValueError:
        raise FilterError(key, "must be an ISO date (YYYY-MM-DD) or datetime") from None


def parse_queue_query(
    qs: dict[str, list[str]], *, default_limit: int = 25, default_sort: str = "priority"
) -> QueueQuery:
    filter_key = _first(qs, "filter")
    if filter_key is not None:
        filter_key = filter_key.lower()
        if filter_key not in FILTER_STATUSES:
            raise FilterError("filter", f"unrecognised value {filter_key!r}")
    sort = (_first(qs, "sort") or default_sort).lower()
    if sort not in SORT_KEYS:
        raise FilterError("sort", f"unrecognised value {sort!r}")

    tab_statuses = FILTER_STATUSES[filter_key] if filter_key else None
    explicit = _multi(qs, "review_status", settings.REVIEW_STATUSES)
    if tab_statuses is not None and explicit is not None:
        statuses: tuple[str, ...] | None = tuple(s for s in explicit if s in tab_statuses)
    else:
        statuses = tab_statuses if tab_statuses is not None else explicit

    created_from = _date(qs, "created_from", end_of_day=False)
    created_to = _date(qs, "created_to", end_of_day=True)
    if created_from and created_to and created_from > created_to:
        raise FilterError("created_from", "must not be after created_to")

    return QueueQuery(
        filter_key=filter_key,
        statuses=statuses,
        search=_first(qs, "search"),
        sort=sort,
        offset=_int(qs, "offset", 0, minimum=0),
        limit=_int(qs, "limit", default_limit, minimum=1, maximum=MAX_LIMIT),
        source_file=_first(qs, "source_file"),
        primary_intents=_multi(qs, "primary_intent", settings.PRIMARY_INTENTS),
        topic_subtypes=_multi(qs, "topic_subtype", settings.TOPIC_SUBTYPES),
        sentiments=_multi(qs, "sentiment", settings.SENTIMENTS),
        priorities=_multi(qs, "priority", PRIORITY_VALUES, cast=int),
        reviewer_actions=_multi(
            qs, "reviewer_action", tuple(settings.WORKFLOW_REVIEWER_ACTIONS) + tuple(settings.REVIEWER_ACTIONS)
        ),
        reviewer_user_id=_first(qs, "reviewer_user_id"),
        created_from=created_from,
        created_to=created_to,
    )
