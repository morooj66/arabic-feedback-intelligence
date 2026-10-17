"""
Explicit, intentionally-invoked MongoDB index bootstrap.

Importing this module (or importing/starting the served application)
never creates or modifies an index — this script only ever does anything
when a human runs it directly, against a database THEY chose via
MONGODB_URI/MONGODB_DATABASE. There is no automatic index creation
anywhere else in this codebase.

Usage:
    MONGODB_URI=... MONGODB_DATABASE=... .venv/bin/python scripts/bootstrap_mongo_indexes.py [--dry-run]

--dry-run prints the exact index plan (collection, keys, options) and
exits without connecting to anything — use this to review the plan
before pointing it at a real database. Without --dry-run, the script
connects, creates every index in the plan (create_index is idempotent:
re-running this script against a database that already has these
indexes is always a safe no-op), and prints a short per-index result
line. It never drops or modifies an existing index with a different
definition — a name collision with an incompatible existing index
raises loudly (pymongo's own OperationFailure) rather than silently
replacing it.

The plan itself lives in INDEX_PLAN below as plain data (not buried in
argparse/CLI logic) specifically so it can be imported and asserted on
by tests without ever opening a network connection.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class IndexSpec:
    collection: str
    keys: tuple[tuple[str, int], ...]  # pymongo direction ints: 1 ascending, -1 descending
    name: str
    unique: bool = False
    expire_after_seconds: int | None = None


# The exact query-driven index plan — one entry per access pattern this
# codebase's Mongo repositories (rcs.review.repositories.mongo) actually
# issue today. Adding a new query pattern later means adding one entry
# here, never touching business-logic code.
INDEX_PLAN: tuple[IndexSpec, ...] = (
    # users.get_by_email / auth.authenticate + create()'s duplicate-email
    # check — PostgreSQL already enforces this as a real unique
    # constraint; Mongo currently has none (see MongoUserRepository's
    # docstring), so two genuinely concurrent create() calls with the
    # same email can both succeed without this index.
    IndexSpec(collection="users", keys=(("email", 1),), name="uniq_users_email", unique=True),
    # review_items.list_queue's reviewer-scoped filter (REVIEWER role)
    # and counts_by_status(reviewer_user_id=...).
    IndexSpec(
        collection="review_items",
        keys=(("reviewer_user_id", 1), ("review_status", 1)),
        name="review_items_reviewer_status",
    ),
    # list_assignable / source-summary / queue's source_file scoping.
    IndexSpec(collection="review_items", keys=(("source_file", 1),), name="review_items_source_file"),
    # review_item_events (append-only audit log): list_events(comment_id)
    # and per-actor lookups.
    IndexSpec(
        collection="review_item_events",
        keys=(("review_item_id", 1), ("created_at", 1)),
        name="review_item_events_item_created",
    ),
    IndexSpec(
        collection="review_item_events",
        keys=(("actor_user_id", 1), ("created_at", 1)),
        name="review_item_events_actor_created",
    ),
    # list_queue's own status filter and counts_by_status() with no
    # reviewer_user_id (ADMIN's global view).
    IndexSpec(collection="review_items", keys=(("review_status", 1),), name="review_items_review_status"),
    # find_existing_dedup_canonical's exact filter shape (source_file +
    # content_dedupe_key together — a compound index, not two separate
    # single-field ones, since the query always supplies both).
    IndexSpec(
        collection="review_items",
        keys=(("source_file", 1), ("content_dedupe_key", 1)),
        name="review_items_dedupe_scope",
    ),
    # latest_for_session / job lookups scoped to one review session.
    IndexSpec(collection="processing_jobs", keys=(("session_id", 1),), name="processing_jobs_session_id"),
    # Session expiry: a real Mongo TTL index as the housekeeping
    # mechanism, replacing the application-level delete_expired() sweep
    # (auth.purge_expired_sessions) as the source of automatic cleanup.
    # resolve_session()'s own expiry check stays the authoritative
    # rejection either way — this index is a housekeeping optimization,
    # never the security boundary.
    IndexSpec(
        collection="user_sessions",
        keys=(("expires_at", 1),),
        name="ttl_user_sessions_expires_at",
        expire_after_seconds=0,
    ),
    # Password-reset tokens: housekeeping TTL (consume() re-checks expiry itself, so this is never the security boundary) and the per-user lookup used to
    # supersede older links.
    IndexSpec(collection="password_reset_tokens", keys=(("expires_at", 1),), name="ttl_password_reset_tokens_expires_at", expire_after_seconds=0),
    IndexSpec(collection="password_reset_tokens", keys=(("user_id", 1),), name="password_reset_tokens_user_id"),
    # Internal messaging: the conversation read (one reviewer, newest first). Idempotency needs no index (the _id is a hash of sender + client id).
    IndexSpec(collection="direct_messages", keys=(("reviewer_user_id", 1), ("created_at", -1)), name="direct_messages_conversation_created"),
    # Account requests: the pending list. E-mail uniqueness needs no index (sentinel _id "email:<address>").
    IndexSpec(collection="account_requests", keys=(("status", 1), ("created_at", 1)), name="account_requests_status_created"),
)


def _print_plan(plan: tuple[IndexSpec, ...]) -> None:
    for spec in plan:
        keys_repr = ", ".join(f"{k}:{d:+d}" for k, d in spec.keys)
        extra = []
        if spec.unique:
            extra.append("unique")
        if spec.expire_after_seconds is not None:
            extra.append(f"TTL={spec.expire_after_seconds}s")
        extra_repr = f" ({', '.join(extra)})" if extra else ""
        print(f"  {spec.collection}.{spec.name}: {{{keys_repr}}}{extra_repr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the index plan and exit — never connects to anything."
    )
    args = parser.parse_args()

    print(f"Mongo index plan ({len(INDEX_PLAN)} indexes):")
    _print_plan(INDEX_PLAN)

    if args.dry_run:
        print("\n--dry-run: no connection attempted, nothing created.")
        return

    from config import settings
    from rcs.review.demo_seed import load_dotenv_if_present
    from rcs.review.repositories.mongo_client import get_mongo_database

    load_dotenv_if_present()
    if not settings.MONGODB_URI or not settings.MONGODB_DATABASE:
        raise SystemExit(
            "MONGODB_URI and MONGODB_DATABASE must both be set to run this bootstrap "
            "(use --dry-run to review the plan without connecting to anything)."
        )

    print(f"\nConnecting to database={settings.MONGODB_DATABASE!r} (URI not printed)...")
    client, db = get_mongo_database(settings.MONGODB_URI, settings.MONGODB_DATABASE)
    try:
        for spec in INDEX_PLAN:
            kwargs = {"name": spec.name, "unique": spec.unique}
            if spec.expire_after_seconds is not None:
                kwargs["expireAfterSeconds"] = spec.expire_after_seconds
            result_name = db[spec.collection].create_index(list(spec.keys), **kwargs)
            print(f"  ok: {spec.collection}.{result_name}")
    finally:
        client.close()

    print("\nDone. Re-running this script against the same database is always safe (idempotent).")


if __name__ == "__main__":
    main()
