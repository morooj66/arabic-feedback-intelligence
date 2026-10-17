"""
Phase M4B — bounded retry for known-transient pymongo connection
failures.

Mirrors rcs.review.db_retry's discipline exactly but targets pymongo's
exception hierarchy instead of SQLAlchemy's -- deliberately NOT reused
from db_retry.py, which is explicitly Postgres/Supabase-pooler-specific
(matches literal "ssl error" / "server closed the connection" text from
that one connection path) and has no meaning for a different database
driver's own transient-failure vocabulary.

Narrow by design:
  - retries only pymongo.errors.AutoReconnect / NetworkTimeout /
    ConnectionFailure -- never OperationFailure (a real server-side
    rejection: auth, a bad operator, a validation error -- will fail
    identically every retry), never WriteError/DuplicateKeyError (a real
    data conflict), and never rcs.review.workflow.VersionConflict /
    InvalidTransition (domain exceptions -- retrying would silently
    paper over an optimistic-concurrency conflict the caller must see).
  - intended for READ-ONLY call sites (get/list/count/aggregate) or a
    single atomic findOneAndUpdate keyed by an explicit filter (naturally
    idempotent: a retried CAS either still matches and applies once more
    with the same net effect, or no longer matches because it already
    applied -- either way it never double-applies a field change). It is
    NOT intended for the multi-document assign_batch transaction, whose
    own retry/abort semantics are governed by pymongo's
    ClientSession.with_transaction() instead.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from pymongo.errors import AutoReconnect, ConnectionFailure, NetworkTimeout

T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 0.2

_RETRYABLE = (AutoReconnect, NetworkTimeout, ConnectionFailure)


def with_transient_retry(fn: Callable[[], T], *, max_attempts: int = MAX_ATTEMPTS) -> T:
    """Calls fn() with no arguments, retrying it (from scratch) up to
    max_attempts times on a recognized transient connection error, with a
    short linear backoff. Re-raises immediately on any other exception,
    or once attempts are exhausted."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except _RETRYABLE:
            if attempt >= max_attempts:
                raise
            time.sleep(BASE_DELAY_SECONDS * attempt)
