"""
Safe, structured logging/reporting helpers for Phase M3 migration tooling.

Nothing in this package should ever be handed a raw secret or PII value
to print -- callers pass already-safe data (counts, ids, status
strings) or use short_ref() to turn a sensitive value into a
non-reversible short reference first. See tables.NEVER_LOG_FIELDS for
the fields this applies to.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time


def short_ref(value: str | None) -> str:
    """Non-reversible 8-hex-char reference for a sensitive value (a
    session/upload token, etc.) -- stable enough to correlate "this one
    row" across log lines without ever exposing or allowing recovery of
    the original value."""
    if not value:
        return "<none>"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def log(event: str, **fields) -> None:
    record = {"ts": round(time.time(), 3), "event": event, **fields}
    print(json.dumps(record, ensure_ascii=False, default=str), file=sys.stderr)


_SECRET_SUBSTRINGS = ("mongodb+srv://", "mongodb://", "postgresql://", "postgresql+psycopg://")


def assert_no_secret_leak(text: str) -> None:
    """Defensive guard for report-writing code: raises if a report string
    accidentally contains what looks like a connection URI. Not a
    substitute for routing sensitive fields through short_ref() in the
    first place -- a last-resort tripwire."""
    lowered = text.lower()
    for marker in _SECRET_SUBSTRINGS:
        if marker in lowered:
            raise ValueError("refusing to write a report containing what looks like a connection URI")
