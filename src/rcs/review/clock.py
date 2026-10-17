"""Application clock for reviewer timestamps (Asia/Riyadh).

Stored naive datetimes are treated as wall-clock times in APP_TIMEZONE.
Do not apply ad-hoc hour offsets in callers.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from config import settings

APP_TZ_NAME = getattr(settings, "APP_TIMEZONE", "Asia/Riyadh")
APP_TZ = ZoneInfo(APP_TZ_NAME)


def now_naive(now: dt.datetime | None = None) -> dt.datetime:
    if now is None:
        return dt.datetime.now(APP_TZ).replace(tzinfo=None)
    if now.tzinfo is not None:
        return now.astimezone(APP_TZ).replace(tzinfo=None)
    return now


def as_app(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(APP_TZ)
    return value.replace(tzinfo=APP_TZ)


def iso_app(value: dt.datetime | None) -> str | None:
    aware = as_app(value)
    return aware.isoformat() if aware is not None else None


def parse_range_bound(raw: str | None, *, end_of_day: bool) -> dt.datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        day = dt.date.fromisoformat(text)
        clock = dt.time(23, 59, 59) if end_of_day else dt.time.min
        return dt.datetime.combine(day, clock)
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return now_naive(parsed)
