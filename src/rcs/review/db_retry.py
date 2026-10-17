"""Phase 7 — bounded retry for known-transient database connection
failures (the Supabase session pooler's documented intermittent
"SSL error: unexpected eof while reading" drops).

Deliberately narrow:
  - retries only a specific, recognized class of connection-level error
    text, never every OperationalError (a real constraint violation or
    bad-SQL error must never be retried — it will just fail identically
    every time and retrying only hides the real problem).
  - small, bounded attempt count with a short backoff — never unbounded.
  - meant for READ-ONLY call sites only. A write whose commit outcome is
    ambiguous (the connection dropped after the server executed it but
    before the client saw the acknowledgement) must NOT be blindly
    retried — that risks a double-write. This module is used for GET
    request dispatch (http_api.py) and read-only auth lookups
    (resolve_session/authenticate) — never for POST/mutation handlers.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from sqlalchemy.exc import OperationalError

T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 0.2

_TRANSIENT_MARKERS = (
    "ssl error",
    "server closed the connection",
    "connection reset",
    "connection refused",
    "could not connect",
    "consuming input failed",
)


def _is_transient(exc: OperationalError) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def with_transient_retry(fn: Callable[[], T], *, max_attempts: int = MAX_ATTEMPTS) -> T:
    """Calls fn() with no arguments, retrying it (from scratch — fn must
    be idempotent/read-only) up to max_attempts times on a recognized
    transient connection error, with a short linear backoff. Re-raises
    immediately on any other exception, or once attempts are exhausted."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except OperationalError as exc:
            if attempt >= max_attempts or not _is_transient(exc):
                raise
            time.sleep(BASE_DELAY_SECONDS * attempt)
