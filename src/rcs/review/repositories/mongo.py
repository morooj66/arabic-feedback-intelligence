"""
Phase M4B — MongoDB implementation of the repository contracts defined
in contracts.py (review_items / review_sessions / processing_jobs only
-- users/user_sessions/auth are explicitly out of scope, see the M4B
report).

Not wired into the running application in this phase: no backend
selector exists yet (that is a later, separate step gated on this
implementation passing its full contract test suite against a live,
credentialed Atlas connection).

Returns duck-typed rcs.review.dto-compatible items (a plain
types.SimpleNamespace exposing the same attribute names ReviewItem/
ReviewSession/ProcessingJob expose) rather than importing those
SQLAlchemy ORM classes -- dto.py's to_detail()/to_queue_item()/etc. only
ever do attribute access, so this satisfies the exact same duck-typed
surface PostgresReviewItemRepository already provides, with zero changes
to dto.py or contracts.py.

Datetime handling deliberately duplicates (does not import) the small
per-column timezone-convention facts already established and audited in
rcs.migration.tables.DATETIME_TZ_CONVENTION -- rcs.migration documents
itself as "never imported by the served application", and this module
IS meant to eventually run inside the served application, so the two
packages must stay independent even though they encode the same real,
previously-audited facts about which columns are naive-UTC, naive-
Asia/Riyadh, or a raw unconverted source value.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import uuid
from types import SimpleNamespace

import pandas as pd

from pymongo import InsertOne, ReturnDocument, UpdateOne

from config import settings
from rcs.review import clock
from rcs.review.audit_events import (
    EVENT_ASSIGNED,
    EVENT_CLAIMED,
    EVENT_EDIT_AND_SEND,
    EVENT_EDIT_AND_SEND_ATTESTED,
    EVENT_SEND_AS_IS_ATTESTED,
    EVENT_NO_RESPONSE_NEEDED,
    EVENT_REASSIGNED,
    EVENT_SEND_AS_IS,
    build_event_fields,
    event_from_fields,
)
from rcs.review.analytics import OUTCOME_ACTIONS, OVERDUE_DAYS, PENDING, _bucket_age, _complete_counts, range_bounds
from rcs.review.models import ReviewItem as _ReviewItemModel
from rcs.review.priority import compute_priority
from rcs.review.repositories.contracts import (
    AssignmentResult,
    FilterAssignmentResult,
    DedupCandidate,
    ReviewerAssignmentCounts,
    SessionCounts,
    SourceFileStatusFields,
)
from rcs.review.repositories.mongo_retry import with_transient_retry
from rcs.review.workflow import InvalidTransition, ItemNotFound, VersionConflict, next_status

# Same eligibility universe as PostgresReviewItemRepository (duplicated,
# not imported, to keep this module's only dependency on the workflow/
# clock layer -- never on the Postgres repository implementation).
_NOT_ASSIGNABLE_STATUSES = ("EXCLUDED", "DUPLICATE_CONTENT")

# rcs.review.models is part of the served app's own domain layer (not
# rcs.migration, which this module deliberately never imports -- see the
# module docstring). Used here ONLY for column-name reflection at import
# time (no engine, no Session, no DB connection) so a freshly-inserted
# ingest document always has every field explicitly present -- matching
# the M3-established "never missing, always explicit null" contract --
# without hand-maintaining a duplicate 81-name list that could drift.
_REVIEW_ITEM_FIELD_NAMES = tuple(c.name for c in _ReviewItemModel.__table__.columns)

# Verbatim duplicate of repository.py::INGEST_OWNED_FIELDS (not imported,
# same reasoning as _NOT_ASSIGNABLE_STATUSES above -- repository.py pulls
# in SQLAlchemy Session/create_engine at module level, which this module
# stays free of).
_INGEST_OWNED_FIELDS = (
    "source_file",
    "source_row_index",
    "original_comment",
    "original_title",
    "original_type",
    "entry_id",
    "lemma",
    "visitor_name",
    "visitor_email",
    "platform",
    "comment_created_at",
    "is_exact_duplicate_of",
    "ingested_at",
    "comment_age_days",
    "age_group",
    "source_reference",
)


_QUEUE_SORT_SPECS = {
    "newest": [("ingested_at", -1), ("_id", 1)],
    "oldest": [("ingested_at", 1), ("_id", 1)],
    "comment_newest": [("comment_created_at", -1), ("ingested_at", -1), ("_id", 1)],
    "comment_oldest": [("comment_created_at", 1), ("ingested_at", 1), ("_id", 1)],
    "priority_newest": [("review_priority", 1), ("ingested_at", -1), ("_id", 1)],
    "priority_oldest": [("review_priority", 1), ("ingested_at", 1), ("_id", 1)],
}


def _apply_classification_filters(
    query: dict,
    *,
    primary_intents=None,
    topic_subtypes=None,
    sentiments=None,
    priorities=None,
    created_from=None,
    created_to=None,
) -> None:
    """Mongo twin of postgres._apply_classification_filters (same
    semantics: OR within a dimension, AND across, NULL never matches a
    filtered dimension, comment-date range inclusive on `comment_created_at`)."""
    if primary_intents:
        query["primary_intent"] = {"$in": list(primary_intents)}
    if topic_subtypes:
        query["topic_subtype"] = {"$in": list(topic_subtypes)}
    if sentiments:
        query["sentiment"] = {"$in": list(sentiments)}
    if priorities:
        query["review_priority"] = {"$in": list(priorities)}
    if created_from is not None or created_to is not None:
        bounds: dict = {}
        if created_from is not None:
            bounds["$gte"] = _query_bound("comment_created_at", created_from)
        if created_to is not None:
            bounds["$lte"] = _query_bound("comment_created_at", created_to)
        query["comment_created_at"] = bounds


def _assignable_statuses() -> tuple[str, ...]:
    from config import settings

    return settings.QUEUE_PENDING_STATUSES


# --- datetime conversion (see module docstring) ---------------------------

_DATETIME_CONVENTION = {
    "review_items": {
        "comment_created_at": "source",
        "ingested_at": "utc",
        "processed_at": "utc",
        "updated_at": "riyadh",
        "generated_at": "riyadh",
        "reviewer_started_at": "riyadh",
        "reviewed_at": "riyadh",
        "outbound_attempted_at": "riyadh",
        "outbound_sent_at": "riyadh",
    },
    "review_sessions": {"created_at": "utc", "updated_at": "utc"},
    "processing_jobs": {"started_at": "utc", "updated_at": "utc", "finished_at": "utc"},
    # Phase M4D.1: rcs.review.auth populates both columns via
    # dt.datetime.utcnow() directly (not rcs.review.clock.now_naive(),
    # which is Riyadh) -- confirmed by reading auth.py's own `_now()`.
    "users": {"created_at": "utc", "updated_at": "utc"},
    # Phase M4D.2: same auth.py `_now()` (dt.datetime.utcnow()) populates
    # both user_sessions columns -- also independently confirmed by
    # rcs.migration.tables.DATETIME_TZ_CONVENTION's own pre-existing
    # ("user_sessions", "created_at"/"expires_at"): "utc" entries.
    "user_sessions": {"created_at": "utc", "expires_at": "utc"},
    # Self-service password reset: written by rcs.review.auth with dt.datetime.utcnow() like the two tables above.
    "password_reset_tokens": {"created_at": "utc", "expires_at": "utc", "used_at": "utc"},
    # Internal messaging: naive UTC like the auth tables.
    "direct_messages": {"created_at": "utc", "read_at": "utc"},
    "account_requests": {"created_at": "utc", "decided_at": "utc"},
}


def _now() -> dt.datetime:
    return clock.now_naive()


def _to_bson_value(value, convention: str | None):
    """naive-UTC / naive-source values round-trip through pymongo's
    default (naive-treated-as-UTC) codec unchanged -- only "riyadh"
    needs the naive wall-clock value tagged with its real zone so pymongo
    stores the correct UTC instant."""
    if value is None or convention is None:
        return value
    if convention == "riyadh":
        return clock.as_app(value)
    return value


def _from_bson_value(value, convention: str | None):
    if value is None or convention is None:
        return value
    if convention == "riyadh":
        return clock.now_naive(value.replace(tzinfo=dt.timezone.utc))
    return value


def _now_bson(convention: str) -> dt.datetime:
    return _to_bson_value(_now(), convention)


def _query_bound(field: str, value: dt.datetime, *, table: str = "review_items") -> dt.datetime:
    """Phase M4C.5 -- convert a naive application-timezone boundary
    datetime (as produced by rcs.review.analytics.range_bounds/now_naive)
    into the naive-UTC value directly comparable, in a $match/count_
    documents filter, against that field's raw BSON-decoded
    representation (pymongo decodes every stored datetime as naive-UTC
    by default, whatever the field's own timezone convention is).

    Distinct from _to_bson_value: that function prepares a VALUE for a
    WRITE (pymongo accepts an aware datetime there and itself converts
    it to UTC on encode); this one prepares a comparison BOUND for a
    READ filter, so it must already be expressed in the same naive-UTC
    terms the stored/decoded values are in -- passing an aware datetime
    into a query filter would still be correct for a real server (pymongo
    encodes it the same way), but the naive-UTC form is what the
    in-memory unit-test fake's plain Python comparisons need, and is
    exactly the value a real document's raw field already holds."""
    convention = _DATETIME_CONVENTION.get(table, {}).get(field)
    if convention == "riyadh":
        return clock.as_app(value).astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value


def _idempotency_key(comment_id: str, action: str, body: str) -> str:
    payload = f"{comment_id}|{action}|{body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _doc_to_item(doc: dict | None, *, id_field: str, table: str) -> SimpleNamespace | None:
    if doc is None:
        return None
    fields = dict(doc)
    id_value = fields.pop("_id")
    conventions = _DATETIME_CONVENTION.get(table, {})
    for key, convention in conventions.items():
        if key in fields:
            fields[key] = _from_bson_value(fields[key], convention)
    fields[id_field] = id_value
    return SimpleNamespace(**fields)


def _generation_apply_action(result) -> str:
    """Verbatim duplicate of service.py's / postgres.py's module-level
    helper of the same name -- see postgres.py's docstring for why this
    module never imports from either."""
    if result.routing_decision == "GENERATE_DRAFT" and result.generation_status == "SUCCEEDED":
        return "apply_generate_draft"
    if result.routing_decision == "NO_RESPONSE_RECOMMENDED":
        return "apply_no_response_recommended"
    return "apply_review_only"


_GENERATION_ALREADY_LANDED_STATUSES = (
    "DRAFT_READY",
    "REVIEW_ONLY",
    "NO_RESPONSE_RECOMMENDED",
    "IN_REVIEW",
    "APPROVED",
    "EDITED_AND_APPROVED",
    "NO_REPLY",
    "SENT",
)


def _apply_claim_step(doc: dict, *, reviewer_id: str, now_riyadh) -> tuple[str, object]:
    """Mirrors service.py/postgres.py's `_ensure_claim` exactly: if
    already IN_REVIEW, only a reviewer-mismatch is checked (raises
    InvalidTransition); otherwise the status transitions via the "claim"
    edge. Returns (status_after_claim, reviewer_started_at)."""
    if doc["review_status"] == "IN_REVIEW":
        if doc.get("reviewer_id") not in (None, reviewer_id):
            raise InvalidTransition(f"claimed by {doc.get('reviewer_id')}")
        return doc["review_status"], doc.get("reviewer_started_at") or now_riyadh
    return next_status(doc["review_status"], "claim"), doc.get("reviewer_started_at") or now_riyadh


def _is_locked(doc: dict) -> bool:
    """Exact mirror of repository.py::upsert_ingest_items' lock check:
    `bool(reviewer_action)` (not `is not None` -- an empty string would
    not count as locked, matching the original exactly) OR review_status
    outside ("UNPROCESSED", None)."""
    return bool(doc.get("reviewer_action")) or doc.get("review_status") not in ("UNPROCESSED", None)


def _build_new_ingest_document(row: dict) -> dict:
    """Every ReviewItem column explicitly present -- None for anything
    not supplied by this ingest row (matching the M3/M4B "never missing,
    always explicit null" contract) -- plus row_version=1, the one
    non-null ORM default an insert would otherwise pick up. Only accepts
    row keys that are real ReviewItem columns, mirroring Postgres's
    `if hasattr(ReviewItem, k)` defensive filter."""
    doc = {name: None for name in _REVIEW_ITEM_FIELD_NAMES if name != "comment_id"}
    doc["row_version"] = 1
    conventions = _DATETIME_CONVENTION.get("review_items", {})
    for key, value in row.items():
        if key == "comment_id" or key not in doc:
            continue
        convention = conventions.get(key)
        doc[key] = _to_bson_value(value, convention) if convention else value
    doc["_id"] = row["comment_id"]
    return doc


def _ingest_owned_field_set(row: dict) -> dict:
    """The $set payload for an existing-unlocked row -- ONLY
    _INGEST_OWNED_FIELDS, exactly mirroring repository.py's update loop.
    Never touches row_version, never touches any review/classification/
    generation field."""
    conventions = _DATETIME_CONVENTION.get("review_items", {})
    fields = {}
    for key in _INGEST_OWNED_FIELDS:
        if key in row:
            convention = conventions.get(key)
            fields[key] = _to_bson_value(row[key], convention) if convention else row[key]
    return fields


class MongoReviewItemRepository:
    def __init__(self, db):
        self.db = db
        self.collection = db["review_items"]

    def _to_item(self, doc):
        return _doc_to_item(doc, id_field="comment_id", table="review_items")

    def _atomic_update(self, comment_id: str, expected_version: int | None, update: dict, session=None):
        filter_ = {"_id": comment_id}
        if expected_version is not None:
            filter_["row_version"] = expected_version
        result = self.collection.find_one_and_update(
            filter_, update, return_document=ReturnDocument.AFTER, session=session
        )
        if result is not None:
            return result
        if expected_version is None:
            raise ItemNotFound(comment_id)
        if self.collection.find_one({"_id": comment_id}, {"_id": 1}, session=session) is None:
            raise ItemNotFound(comment_id)
        raise VersionConflict(f"row_version changed, expected {expected_version}")

    def _audited(self, comment_id, event_type, *, actor_user_id, actor_role, action):
        """Runs `action(session)` and its audit event inside ONE MongoDB
        multi-document transaction (requires a replica set / Atlas; on a
        standalone server `start_transaction` raises -- loudly, never a
        silent fallback). The event is written only if the action actually
        changed the document (row_version moved), so idempotent repeats
        write nothing. Any exception -- version conflict, invalid
        transition, or a failed event insert -- aborts the transaction,
        discarding the state change too."""
        result = {}

        def _callback(session):
            before = self.collection.find_one({"_id": comment_id}, session=session)
            if before is None:
                raise ItemNotFound(comment_id)
            item = action(session)
            result["item"] = item
            if int(item.row_version or 1) != int(before.get("row_version") or 1):
                fields = build_event_fields(
                    review_item_id=comment_id,
                    event_type=event_type,
                    actor_user_id=actor_user_id,
                    actor_role=actor_role,
                    from_status=before.get("review_status"),
                    to_status=item.review_status,
                    from_assignee_user_id=before.get("reviewer_user_id"),
                    to_assignee_user_id=item.reviewer_user_id,
                    created_at=_now_bson("riyadh"),
                )
                self.db["review_item_events"].insert_one({"_id": fields["event_id"], **fields}, session=session)

        with self.db.client.start_session() as session:
            session.with_transaction(_callback)
        return result["item"]

    def list_events(self, comment_id: str) -> list:
        docs = with_transient_retry(
            lambda: list(self.db["review_item_events"].find({"review_item_id": comment_id}).sort("created_at", 1))
        )
        return [event_from_fields(d) for d in docs]

    # --- reads (retryable -- read-only) ------------------------------------

    def get(self, comment_id: str):
        doc = with_transient_retry(lambda: self.collection.find_one({"_id": comment_id}))
        return self._to_item(doc)

    def _find_ordered(self, query: dict, sort_spec: list, offset: int, limit: int) -> list:
        """Sorted page with the leading sort key's NULLs LAST in both
        directions (PostgreSQL parity -- MongoDB natively sorts null/
        missing lowest) and the unique `_id` as the final key, so paging is
        stable. When the query already filters on the leading field every
        match is non-null and one sorted find suffices; otherwise the
        non-null rows are paged first, then the null rows."""
        lead = sort_spec[0][0]
        if lead in query:
            return list(self.collection.find(query).sort(sort_spec).skip(offset).limit(limit))
        non_null = {**query, lead: {"$ne": None}}
        n_non_null = self.collection.count_documents(non_null)
        out: list = []
        if offset < n_non_null:
            out = list(self.collection.find(non_null).sort(sort_spec).skip(offset).limit(limit))
        remaining = limit - len(out)
        if remaining > 0:
            nulls = {**query, lead: None}
            out += list(
                self.collection.find(nulls).sort(sort_spec[1:]).skip(max(0, offset - n_non_null)).limit(remaining)
            )
        return out

    def list_queue(
        self,
        *,
        statuses,
        source_file,
        reviewer_user_id,
        search,
        sort,
        offset,
        limit,
        primary_intents=None,
        topic_subtypes=None,
        reviewer_actions=None,
        sentiments=None,
        priorities=None,
        created_from=None,
        created_to=None,
    ):
        query: dict = {"review_status": {"$in": list(statuses)}}
        if source_file:
            query["source_file"] = source_file
        if reviewer_user_id:
            query["reviewer_user_id"] = reviewer_user_id
        _apply_classification_filters(
            query,
            primary_intents=primary_intents,
            topic_subtypes=topic_subtypes,
            sentiments=sentiments,
            priorities=priorities,
            created_from=created_from,
            created_to=created_to,
        )
        if reviewer_actions:
            query["reviewer_action"] = {"$in": list(reviewer_actions)}
        if search:
            pattern = re.escape(search)
            query["$or"] = [{"_id": {"$regex": pattern}}, {"original_comment": {"$regex": pattern}}]
        if sort not in _QUEUE_SORT_SPECS:
            raise ValueError(f"unknown queue sort: {sort!r}")
        sort_spec = _QUEUE_SORT_SPECS[sort]

        def _run():
            total = self.collection.count_documents(query)
            page = self._find_ordered(query, sort_spec, offset, limit)
            return page, total

        page, total = with_transient_retry(_run)
        return [self._to_item(d) for d in page], int(total)

    def counts_by_status(self, reviewer_user_id: str | None) -> dict[str, int]:
        match = {"reviewer_user_id": reviewer_user_id} if reviewer_user_id else {}
        pipeline = [{"$match": match}, {"$group": {"_id": "$review_status", "count": {"$sum": 1}}}]
        rows = with_transient_retry(lambda: list(self.collection.aggregate(pipeline)))
        return {str(r["_id"] or ""): int(r["count"]) for r in rows}

    def list_assignable(
        self,
        *,
        unassigned_only: bool,
        source_file: str | None,
        offset: int,
        limit: int,
        primary_intents: tuple[str, ...] | None = None,
        topic_subtypes: tuple[str, ...] | None = None,
        search: str | None = None,
        sentiments: tuple[str, ...] | None = None,
        priorities: tuple[int, ...] | None = None,
        created_from=None,
        created_to=None,
        reviewer_user_id: str | None = None,
        statuses: tuple[str, ...] | None = None,
        sort: str = "oldest",
    ):
        eligible = list(_assignable_statuses())
        if statuses:
            eligible = [st for st in eligible if st in statuses]
        query: dict = {"review_status": {"$in": eligible}, "reviewer_action": None}
        if unassigned_only:
            query["reviewer_user_id"] = None
        if reviewer_user_id:
            # unassigned_only AND a named assignee can never both hold
            # (PostgreSQL ANDs the two predicates), so match nothing.
            query["reviewer_user_id"] = {"$in": []} if unassigned_only else reviewer_user_id
        if source_file:
            query["source_file"] = source_file
        _apply_classification_filters(
            query,
            primary_intents=primary_intents,
            topic_subtypes=topic_subtypes,
            sentiments=sentiments,
            priorities=priorities,
            created_from=created_from,
            created_to=created_to,
        )
        if search:
            pattern = re.escape(search)
            query["$or"] = [{"_id": {"$regex": pattern}}, {"original_comment": {"$regex": pattern}}]

        def _run():
            total = self.collection.count_documents(query)
            if sort not in _QUEUE_SORT_SPECS:
                raise ValueError(f"unknown queue sort: {sort!r}")
            page = self._find_ordered(query, _QUEUE_SORT_SPECS[sort], offset, limit)
            return page, total

        page, total = with_transient_retry(_run)
        return [self._to_item(d) for d in page], int(total)

    def distinct_source_files(self) -> list[str]:
        return sorted(with_transient_retry(lambda: self.collection.distinct("source_file")))

    def unassigned_counts_by_source_file(self) -> dict[str, int]:
        pipeline = [
            {"$match": {"review_status": {"$nin": list(_NOT_ASSIGNABLE_STATUSES)}, "reviewer_user_id": None}},
            {"$group": {"_id": "$source_file", "count": {"$sum": 1}}},
        ]
        rows = with_transient_retry(lambda: list(self.collection.aggregate(pipeline)))
        return {r["_id"]: int(r["count"]) for r in rows}

    def count_all(self) -> int:
        return with_transient_retry(lambda: self.collection.count_documents({}))

    def list_ordered(self, comment_ids: list[str] | None):
        query = {"_id": {"$in": list(comment_ids)}} if comment_ids is not None else {}
        docs = with_transient_retry(
            lambda: list(self.collection.find(query).sort([("source_row_index", 1), ("_id", 1)]))
        )
        return [self._to_item(d) for d in docs]

    def get_many(self, comment_ids: list[str]) -> dict:
        if not comment_ids:
            return {}
        docs = with_transient_retry(lambda: list(self.collection.find({"_id": {"$in": list(comment_ids)}})))
        return {d["_id"]: self._to_item(d) for d in docs}

    def comment_ids_for_source_file(self, source_file: str) -> list[str]:
        docs = with_transient_retry(
            lambda: list(self.collection.find({"source_file": source_file}, {"_id": 1}))
        )
        return [d["_id"] for d in docs]

    def reviewer_assignment_counts(self) -> dict[str, ReviewerAssignmentCounts]:
        total_pipeline = [
            {"$match": {"reviewer_user_id": {"$ne": None}}},
            {"$group": {"_id": "$reviewer_user_id", "count": {"$sum": 1}}},
        ]
        pending_pipeline = [
            {"$match": {"reviewer_user_id": {"$ne": None}, "reviewer_action": None}},
            {"$group": {"_id": "$reviewer_user_id", "count": {"$sum": 1}}},
        ]
        totals = with_transient_retry(lambda: list(self.collection.aggregate(total_pipeline)))
        pendings = with_transient_retry(lambda: list(self.collection.aggregate(pending_pipeline)))
        pending_map = {r["_id"]: int(r["count"]) for r in pendings}
        return {
            r["_id"]: ReviewerAssignmentCounts(total=int(r["count"]), pending=pending_map.get(r["_id"], 0))
            for r in totals
        }

    def get_dedup_candidates(self, comment_ids: list[str]) -> list[DedupCandidate]:
        if not comment_ids:
            return []
        projection = {
            "source_file": 1, "source_row_index": 1, "original_comment": 1,
            "review_status": 1, "reviewer_action": 1, "dedupe_canonical_id": 1,
        }
        docs = with_transient_retry(
            lambda: list(self.collection.find({"_id": {"$in": list(comment_ids)}}, projection))
        )
        return [
            DedupCandidate(
                comment_id=d["_id"], source_file=d.get("source_file"), source_row_index=d.get("source_row_index"),
                original_comment=d.get("original_comment"), review_status=d.get("review_status"),
                reviewer_action=d.get("reviewer_action"), dedupe_canonical_id=d.get("dedupe_canonical_id"),
            )
            for d in docs
        ]

    def find_existing_dedup_canonical(self, source_file: str, content_dedupe_key: str) -> str | None:
        doc = with_transient_retry(
            lambda: self.collection.find_one(
                {
                    "source_file": source_file,
                    "content_dedupe_key": content_dedupe_key,
                    "dedupe_canonical_id": {"$ne": None},
                },
                {"dedupe_canonical_id": 1},
            )
        )
        return doc["dedupe_canonical_id"] if doc is not None else None

    def set_dedupe_canonical_if_unset(self, comment_id: str, content_dedupe_key: str | None) -> bool:
        """Atomic CAS, never retried (same reasoning as _atomic_update
        above): a conditional write must never be blindly re-issued."""
        result = self.collection.update_one(
            {"_id": comment_id, "dedupe_canonical_id": None},
            {"$set": {"content_dedupe_key": content_dedupe_key, "dedupe_canonical_id": comment_id}},
        )
        if result.matched_count == 0:
            if self.collection.find_one({"_id": comment_id}, {"_id": 1}) is None:
                raise ItemNotFound(comment_id)
            return False  # already had a canonical -- never re-touched
        return True

    def mark_dedupe_duplicate(self, comment_id: str, content_dedupe_key: str, canonical_id: str) -> None:
        result = self.collection.update_one(
            {"_id": comment_id},
            {
                "$set": {
                    "content_dedupe_key": content_dedupe_key,
                    "dedupe_canonical_id": canonical_id,
                    "review_status": "DUPLICATE_CONTENT",
                    "outbound_status": "NOT_APPLICABLE",
                }
            },
        )
        if result.matched_count == 0:
            raise ItemNotFound(comment_id)

    def review_statuses(self, comment_ids: list[str]) -> dict[str, str]:
        if not comment_ids:
            return {}
        projection = {"review_status": 1}
        docs = with_transient_retry(
            lambda: list(self.collection.find({"_id": {"$in": list(comment_ids)}}, projection))
        )
        return {d["_id"]: d.get("review_status") for d in docs}

    def source_file_status_fields(self, source_file: str) -> list[SourceFileStatusFields]:
        projection = {"review_status": 1, "generation_routing_decision": 1, "generation_status": 1}
        docs = with_transient_retry(
            lambda: list(self.collection.find({"source_file": source_file}, projection))
        )
        return [
            SourceFileStatusFields(
                review_status=d.get("review_status"),
                generation_routing_decision=d.get("generation_routing_decision"),
                generation_status=d.get("generation_status"),
            )
            for d in docs
        ]

    # --- single-item mutations (atomic CAS; never retried -- see mongo_retry) --

    def apply_generation_result(
        self,
        result,
        *,
        intent=None,
        topic=None,
        sentiment=None,
        confidence=None,
        needs_review=None,
    ):
        doc = self.collection.find_one({"_id": result.comment_id})
        if doc is None:
            raise ItemNotFound(result.comment_id)
        if doc.get("reviewer_action") or doc.get("ready_to_send"):
            return self._to_item(doc)
        if doc.get("generation_routing_decision") and doc["review_status"] in _GENERATION_ALREADY_LANDED_STATUSES:
            return self._to_item(doc)

        action = _generation_apply_action(result)
        fields: dict = {
            "review_status": next_status(doc["review_status"], action),
            "generation_routing_decision": result.routing_decision,
            "generation_status": result.generation_status,
            "generation_reason_code": result.reason_code,
            "generation_provider": result.provider,
            "generation_model_name": result.model_name,
            "generation_prompt_version": result.prompt_version,
        }
        if result.generated_at:
            try:
                generated_at = dt.datetime.fromisoformat(result.generated_at.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                generated_at = _now()
            fields["generated_at"] = _to_bson_value(generated_at, "riyadh")
        if result.suggested_response:
            fields["suggested_final_reply"] = result.suggested_response
            fields["ai_suggested_final_reply"] = result.suggested_response
        primary_intent = doc.get("primary_intent")
        if intent:
            primary_intent = doc.get("primary_intent") or intent
            fields["primary_intent"] = primary_intent
            fields["ai_primary_intent"] = doc.get("ai_primary_intent") or intent
        topic_subtype = doc.get("topic_subtype")
        if topic:
            topic_subtype = doc.get("topic_subtype") or topic
            fields["topic_subtype"] = topic_subtype
            fields["ai_topic_subtype"] = doc.get("ai_topic_subtype") or topic
        if sentiment:
            fields["sentiment"] = sentiment
            fields["ai_sentiment"] = doc.get("ai_sentiment") or sentiment
        needs_review_value = doc.get("needs_review")
        if needs_review is not None:
            needs_review_value = needs_review
            fields["needs_review"] = needs_review
        confidence_value = doc.get("ai_confidence")
        if confidence is not None:
            confidence_value = confidence
            fields["ai_confidence"] = confidence
        fields["source_reference"] = doc.get("source_reference") or doc.get("source_file")

        prio, reason = compute_priority(
            primary_intent=primary_intent,
            topic_subtype=topic_subtype,
            generation_routing=fields["generation_routing_decision"],
            needs_review=needs_review_value,
            confidence=confidence_value,
        )
        fields["review_priority"] = prio
        fields["priority_reason"] = reason
        if fields["review_status"] == "NO_RESPONSE_RECOMMENDED":
            fields["outbound_status"] = "NOT_APPLICABLE"
        fields["updated_at"] = _now_bson("riyadh")

        update = {"$set": fields, "$inc": {"row_version": 1}}
        return self._to_item(self._atomic_update(result.comment_id, None, update))

    def claim(self, comment_id, *, reviewer_id, display_name, user_id, expected_version, actor_role=None):
        return self._audited(
            comment_id,
            EVENT_CLAIMED,
            actor_user_id=user_id or reviewer_id,
            actor_role=actor_role,
            action=lambda session: self._claim_core(comment_id, reviewer_id=reviewer_id, display_name=display_name, user_id=user_id, expected_version=expected_version, session=session),
        )

    def _claim_core(self, comment_id, *, reviewer_id, display_name, user_id, expected_version, session=None):
        doc = self.collection.find_one({"_id": comment_id}, session=session)
        if doc is None:
            raise ItemNotFound(comment_id)
        current_version = int(doc.get("row_version") or 1)
        if expected_version is not None and expected_version != current_version:
            raise VersionConflict(f"row_version {current_version} != expected {expected_version}")
        if doc["review_status"] == "IN_REVIEW" and doc.get("reviewer_id") == reviewer_id:
            return self._to_item(doc)
        if doc["review_status"] == "IN_REVIEW" and doc.get("reviewer_id") not in (None, reviewer_id):
            raise InvalidTransition(f"claimed by {doc.get('reviewer_id')}")
        now_riyadh = _now_bson("riyadh")
        update = {
            "$set": {
                "review_status": next_status(doc["review_status"], "claim"),
                "reviewer_id": reviewer_id,
                "reviewer_display_name": display_name,
                "reviewer_user_id": user_id,
                "reviewer_started_at": doc.get("reviewer_started_at") or now_riyadh,
                "updated_at": now_riyadh,
            },
            "$inc": {"row_version": 1},
        }
        return self._to_item(self._atomic_update(comment_id, expected_version, update, session))

    def send_as_is(self, comment_id, *, reviewer_id, display_name, user_id, expected_version, actor_role=None, attested=False):
        return self._audited(
            comment_id,
            EVENT_SEND_AS_IS_ATTESTED if attested else EVENT_SEND_AS_IS,
            actor_user_id=user_id or reviewer_id,
            actor_role=actor_role,
            action=lambda session: self._send_as_is_core(comment_id, reviewer_id=reviewer_id, display_name=display_name, user_id=user_id, expected_version=expected_version, session=session),
        )

    def _send_as_is_core(self, comment_id, *, reviewer_id, display_name, user_id, expected_version, session=None):
        doc = self.collection.find_one({"_id": comment_id}, session=session)
        if doc is None:
            raise ItemNotFound(comment_id)
        current_version = int(doc.get("row_version") or 1)
        if expected_version is not None and expected_version != current_version:
            raise VersionConflict(f"row_version {current_version} != expected {expected_version}")
        if doc.get("reviewer_action") == "SEND_AS_IS" and doc["review_status"] == "APPROVED":
            return self._to_item(doc)
        suggestion = (doc.get("suggested_final_reply") or "").strip()
        if not suggestion:
            raise InvalidTransition("SEND_AS_IS requires a suggested_response")
        now_riyadh = _now_bson("riyadh")
        status_after_claim, reviewer_started_at = _apply_claim_step(doc, reviewer_id=reviewer_id, now_riyadh=now_riyadh)
        final_status = next_status(status_after_claim, "SEND_AS_IS")
        update = {
            "$set": {
                "review_status": final_status,
                "reviewer_action": "SEND_AS_IS",
                "reviewer_id": reviewer_id,
                "reviewer_display_name": display_name,
                "reviewer_user_id": user_id,
                "reviewer_started_at": reviewer_started_at,
                "final_reply": suggestion,
                "ready_to_send": True,
                "reviewed_at": now_riyadh,
                "outbound_status": "PENDING",
                "outbound_idempotency_key": _idempotency_key(comment_id, "SEND_AS_IS", suggestion),
                "updated_at": now_riyadh,
            },
            "$inc": {"row_version": 1},
        }
        return self._to_item(self._atomic_update(comment_id, expected_version, update, session))

    def edit_and_send(self, comment_id, *, reviewer_id, display_name, user_id, edited_response, expected_version, actor_role=None, attested=False):
        return self._audited(
            comment_id,
            EVENT_EDIT_AND_SEND_ATTESTED if attested else EVENT_EDIT_AND_SEND,
            actor_user_id=user_id or reviewer_id,
            actor_role=actor_role,
            action=lambda session: self._edit_and_send_core(comment_id, reviewer_id=reviewer_id, display_name=display_name, user_id=user_id, edited_response=edited_response, expected_version=expected_version, session=session),
        )

    def _edit_and_send_core(self, comment_id, *, reviewer_id, display_name, user_id, edited_response, expected_version, session=None):
        edited = (edited_response or "").strip()
        if not edited:
            raise InvalidTransition("EDIT_AND_SEND requires non-empty edited_response")
        doc = self.collection.find_one({"_id": comment_id}, session=session)
        if doc is None:
            raise ItemNotFound(comment_id)
        current_version = int(doc.get("row_version") or 1)
        if expected_version is not None and expected_version != current_version:
            raise VersionConflict(f"row_version {current_version} != expected {expected_version}")
        original = doc.get("suggested_final_reply")
        if (
            doc.get("reviewer_action") == "EDIT_AND_SEND"
            and doc["review_status"] == "EDITED_AND_APPROVED"
            and doc.get("final_reply") == edited
        ):
            return self._to_item(doc)
        now_riyadh = _now_bson("riyadh")
        status_after_claim, reviewer_started_at = _apply_claim_step(doc, reviewer_id=reviewer_id, now_riyadh=now_riyadh)
        final_status = next_status(status_after_claim, "EDIT_AND_SEND")
        update = {
            "$set": {
                "review_status": final_status,
                "reviewer_action": "EDIT_AND_SEND",
                "reviewer_id": reviewer_id,
                "reviewer_display_name": display_name,
                "reviewer_user_id": user_id,
                "reviewer_started_at": reviewer_started_at,
                "reviewer_edit": edited,
                "final_reply": edited,
                "suggested_final_reply": original,
                "ready_to_send": True,
                "reviewed_at": now_riyadh,
                "outbound_status": "PENDING",
                "outbound_idempotency_key": _idempotency_key(comment_id, "EDIT_AND_SEND", edited),
                "updated_at": now_riyadh,
            },
            "$inc": {"row_version": 1},
        }
        return self._to_item(self._atomic_update(comment_id, expected_version, update, session))

    def mark_no_response_needed(self, comment_id, *, reviewer_id, display_name, user_id, reason, expected_version, actor_role=None):
        return self._audited(
            comment_id,
            EVENT_NO_RESPONSE_NEEDED,
            actor_user_id=user_id or reviewer_id,
            actor_role=actor_role,
            action=lambda session: self._mark_no_response_needed_core(comment_id, reviewer_id=reviewer_id, display_name=display_name, user_id=user_id, reason=reason, expected_version=expected_version, session=session),
        )

    def _mark_no_response_needed_core(self, comment_id, *, reviewer_id, display_name, user_id, reason, expected_version, session=None):
        doc = self.collection.find_one({"_id": comment_id}, session=session)
        if doc is None:
            raise ItemNotFound(comment_id)
        current_version = int(doc.get("row_version") or 1)
        if expected_version is not None and expected_version != current_version:
            raise VersionConflict(f"row_version {current_version} != expected {expected_version}")
        if doc.get("reviewer_action") == "NO_RESPONSE_NEEDED" and doc["review_status"] == "NO_REPLY":
            return self._to_item(doc)
        now_riyadh = _now_bson("riyadh")
        update = {
            "$set": {
                "review_status": next_status(doc["review_status"], "NO_RESPONSE_NEEDED"),
                "reviewer_action": "NO_RESPONSE_NEEDED",
                "reviewer_id": reviewer_id,
                "reviewer_display_name": display_name,
                "reviewer_user_id": user_id,
                "no_response_reason": reason,
                "ready_to_send": False,
                "final_reply": None,
                "reviewed_at": now_riyadh,
                "outbound_status": "NOT_APPLICABLE",
                "updated_at": now_riyadh,
            },
            "$inc": {"row_version": 1},
        }
        return self._to_item(self._atomic_update(comment_id, expected_version, update, session))

    def begin_outbound_dispatch(self, comment_id: str, expected_version: int | None) -> ReviewItem:
        doc = self.collection.find_one({"_id": comment_id})
        if doc is None:
            raise ItemNotFound(comment_id)
        current_version = int(doc.get("row_version") or 1)
        if expected_version is not None and expected_version != current_version:
            raise VersionConflict(f"row_version {current_version} != expected {expected_version}")
        if doc["review_status"] == "SENT" and doc.get("outbound_idempotency_key"):
            return self._to_item(doc)
        idempotency_key = doc.get("outbound_idempotency_key") or _idempotency_key(
            comment_id, doc.get("reviewer_action") or "dispatch", doc.get("final_reply") or ""
        )
        now_riyadh = _now_bson("riyadh")
        update = {
            "$set": {
                "review_status": next_status(doc["review_status"], "dispatch"),
                "outbound_status": "SENDING",
                "outbound_attempted_at": now_riyadh,
                "outbound_idempotency_key": idempotency_key,
                "updated_at": now_riyadh,
            },
            "$inc": {"row_version": 1},
        }
        return self._to_item(self._atomic_update(comment_id, expected_version, update))

    def complete_outbound_dispatch(
        self,
        comment_id: str,
        *,
        ok: bool,
        external_id: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ReviewItem:
        doc = self.collection.find_one({"_id": comment_id})
        if doc is None:
            raise ItemNotFound(comment_id)
        now_riyadh = _now_bson("riyadh")
        if ok:
            fields = {
                "review_status": next_status(doc["review_status"], "mark_sent"),
                "outbound_status": "SENT",
                "outbound_sent_at": now_riyadh,
                "outbound_external_id": external_id,
                "outbound_error_code": None,
                "outbound_error_message": None,
            }
        else:
            fields = {
                "review_status": next_status(doc["review_status"], "mark_failed"),
                "outbound_status": "FAILED",
                "outbound_error_code": error_code,
                "outbound_error_message": error_message,
            }
        fields["ready_to_send"] = fields["review_status"] in settings.READY_TO_SEND_STATUSES or fields["review_status"] == "SEND_FAILED"
        fields["updated_at"] = now_riyadh
        update = {"$set": fields, "$inc": {"row_version": 1}}
        return self._to_item(self._atomic_update(comment_id, None, update))

    # --- batch assignment: real multi-document transaction ------------------

    def assign_batch(
        self,
        comment_ids: list[str],
        reviewer_user_id: str,
        *,
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ) -> AssignmentResult:
        """Whole-batch-or-nothing, matching PostgresReviewItemRepository's
        one-Session/one-commit contract exactly: a MongoDB multi-document
        ACID transaction wraps the entire eligibility-check-then-
        conditional-update loop. `session.with_transaction()` (pymongo's
        own helper) commits once at the end and aborts -- discarding every
        write staged in the transaction -- if the callback raises for any
        reason, including an unexpected mid-batch failure. Never retried
        by mongo_retry (that module is explicitly read-only/CAS-only);
        with_transaction has its own narrower, transaction-specific retry
        for TransientTransactionError, which is a different, safe concern
        (retrying the whole transaction from scratch, never partially)."""
        assigned: list[str] = []
        skipped_decided: list[str] = []
        skipped_not_reviewable: list[str] = []
        not_found: list[str] = []
        assignable_statuses = _assignable_statuses()

        def _callback(session):
            assigned.clear()
            skipped_decided.clear()
            skipped_not_reviewable.clear()
            not_found.clear()
            for comment_id in comment_ids:
                doc = self.collection.find_one({"_id": comment_id}, session=session)
                if doc is None:
                    not_found.append(comment_id)
                    continue
                if doc.get("reviewer_action") is not None:
                    skipped_decided.append(comment_id)
                    continue
                if doc["review_status"] not in assignable_statuses:
                    skipped_not_reviewable.append(comment_id)
                    continue
                if doc.get("reviewer_user_id") != reviewer_user_id:
                    prev_assignee = doc.get("reviewer_user_id")
                    self.collection.update_one(
                        {"_id": comment_id},
                        {"$set": {"reviewer_user_id": reviewer_user_id, "updated_at": _now_bson("riyadh")}, "$inc": {"row_version": 1}},
                        session=session,
                    )
                    fields = build_event_fields(
                        review_item_id=comment_id,
                        event_type=EVENT_ASSIGNED if prev_assignee is None else EVENT_REASSIGNED,
                        actor_user_id=actor_user_id,
                        actor_role=actor_role,
                        from_status=doc["review_status"],
                        to_status=doc["review_status"],
                        from_assignee_user_id=prev_assignee,
                        to_assignee_user_id=reviewer_user_id,
                        created_at=_now_bson("riyadh"),
                    )
                    self.db["review_item_events"].insert_one({"_id": fields["event_id"], **fields}, session=session)
                assigned.append(comment_id)

        client = self.db.client
        with client.start_session() as session:
            session.with_transaction(_callback)

        return AssignmentResult(
            assigned=tuple(assigned),
            skipped_already_decided=tuple(skipped_decided),
            skipped_not_reviewable=tuple(skipped_not_reviewable),
            not_found=tuple(not_found),
        )

    def assign_by_filter(
        self,
        reviewer_user_id: str,
        *,
        count: int | None,
        source_file: str | None = None,
        primary_intents=None,
        topic_subtypes=None,
        search: str | None = None,
        sentiments=None,
        priorities=None,
        created_from=None,
        created_to=None,
        statuses=None,
        sort: str = "oldest",
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ) -> FilterAssignmentResult:
        """Same contract as PostgresReviewItemRepository.assign_by_filter. One multi-document transaction; every claim is a compare-and-set
        update_one filtered on "still unassigned and undecided" so concurrent callers can never both claim a document. UNVERIFIED against a live
        MongoDB (unit-tested with an in-memory fake only)."""
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
            raise ValueError("count must be a positive integer or None")
        if sort not in _QUEUE_SORT_SPECS:
            raise ValueError(f"unknown queue sort: {sort!r}")
        eligible = list(_assignable_statuses())
        if statuses:
            eligible = [st for st in eligible if st in statuses]
        query: dict = {"review_status": {"$in": eligible}, "reviewer_action": None, "reviewer_user_id": None}
        if source_file:
            query["source_file"] = source_file
        _apply_classification_filters(query, primary_intents=primary_intents, topic_subtypes=topic_subtypes, sentiments=sentiments,
                                      priorities=priorities, created_from=created_from, created_to=created_to)
        if search:
            pattern = re.escape(search)
            query["$or"] = [{"_id": {"$regex": pattern}}, {"original_comment": {"$regex": pattern}}]
        spec = _QUEUE_SORT_SPECS[sort]
        state = {"claimed": 0, "matched": 0}
        max_rounds = 25

        def _callback(session):
            state["claimed"] = 0
            state["matched"] = self.collection.count_documents(query, session=session)
            for _ in range(max_rounds):
                want = None if count is None else count - state["claimed"]
                if want is not None and want <= 0:
                    break
                total_left = self.collection.count_documents(query, session=session)
                page = self._find_ordered(query, spec, 0, want if want is not None else max(total_left, 1))
                if not page:
                    break
                won = 0
                for doc in page:
                    res = self.collection.update_one(
                        {"_id": doc["_id"], "reviewer_user_id": None, "reviewer_action": None},
                        {"$set": {"reviewer_user_id": reviewer_user_id, "updated_at": _now_bson("riyadh")}, "$inc": {"row_version": 1}},
                        session=session,
                    )
                    if res.modified_count != 1:
                        continue                                  # somebody else claimed or decided it first
                    won += 1
                    fields = build_event_fields(
                        review_item_id=doc["_id"], event_type=EVENT_ASSIGNED, actor_user_id=actor_user_id, actor_role=actor_role,
                        from_status=doc["review_status"], to_status=doc["review_status"], from_assignee_user_id=None,
                        to_assignee_user_id=reviewer_user_id, created_at=_now_bson("riyadh"),
                    )
                    self.db["review_item_events"].insert_one({"_id": fields["event_id"], **fields}, session=session)
                state["claimed"] += won
                if won == 0:
                    break

        with self.db.client.start_session() as session:
            session.with_transaction(_callback)
        remaining = with_transient_retry(lambda: self.collection.count_documents(query))
        assigned = state["claimed"]
        return FilterAssignmentResult(state["matched"], count, assigned, 0 if count is None else max(0, count - assigned), int(remaining))

    def upsert_ingest_batch(self, rows: list[dict]) -> dict:
        """Phase M4C.1 -- corrected read-partition-write algorithm (see
        the M4C.1 plan-correction pass). One multi-document transaction:
        ONE read for all requested IDs, partition NEW/EXISTING_UNLOCKED/
        EXISTING_LOCKED in memory, insert NEW, conditionally update
        EXISTING_UNLOCKED with a write-time filter that re-checks the
        lock condition, and issue no operation at all for
        EXISTING_LOCKED.

        Never wrapped in mongo_retry.with_transient_retry -- only
        pymongo's own session.with_transaction() retry applies (see the
        M4C.1 retry-matrix correction). A genuine duplicate-key error on
        a concurrently-inserted NEW row, or any other non-transient
        error, is never swallowed here -- it propagates out of this
        method exactly as repository.py::upsert_ingest_items' unhandled
        exceptions already do, letting job_runner.py's existing handler
        mark the job failed (see the M4C.1 call-path correction: there
        is no automatic retry of the current batch at any layer above
        this one)."""
        if not rows:
            return {"inserted": 0, "updated": 0, "skipped": 0}
        rows_by_id = {row["comment_id"]: row for row in rows}
        ids = list(rows_by_id.keys())
        counts = {"inserted": 0, "updated": 0, "skipped": 0}

        def _callback(session):
            # Reset on every (re-)execution -- with_transaction may call
            # this callback more than once; nothing here may accumulate
            # across invocations.
            counts["inserted"] = 0
            counts["updated"] = 0
            counts["skipped"] = 0

            existing_docs = list(
                self.collection.find(
                    {"_id": {"$in": ids}},
                    {"_id": 1, "reviewer_action": 1, "review_status": 1},
                    session=session,
                )
            )
            existing_by_id = {d["_id"]: d for d in existing_docs}

            new_ids = [i for i in ids if i not in existing_by_id]
            unlocked_ids = [i for i, d in existing_by_id.items() if not _is_locked(d)]

            ops = [InsertOne(_build_new_ingest_document(rows_by_id[i])) for i in new_ids]
            for i in unlocked_ids:
                ops.append(
                    UpdateOne(
                        # Write-time re-check of the SAME lock condition
                        # used for the in-memory partition above -- the
                        # primary protection against a row that becomes
                        # locked between the read and this write is a
                        # genuine MongoDB WriteConflict (aborts and
                        # retries the whole callback); this filter is the
                        # second, independent layer, and the only one
                        # that matters if a transaction is ever bypassed.
                        {"_id": i, "reviewer_action": {"$in": [None, ""]}, "review_status": {"$in": [None, "UNPROCESSED"]}},
                        {"$set": _ingest_owned_field_set(rows_by_id[i])},
                    )
                )

            if ops:
                # ordered=False is a throughput choice only -- inside a
                # transaction, any real (non-transient) error still
                # aborts the whole transaction; there is no partial
                # commit either way.
                self.collection.bulk_write(ops, ordered=False, session=session)

            counts["inserted"] = len(new_ids)
            counts["updated"] = len(unlocked_ids)
            counts["skipped"] = len(existing_by_id) - len(unlocked_ids)

        client = self.db.client
        with client.start_session() as session:
            session.with_transaction(_callback)

        return dict(counts)

    def load_for_prefilter(self, comment_ids: list[str] | None = None) -> pd.DataFrame:
        """Phase M4C.2 -- exact column set as PostgresReviewItemRepository.
        load_for_prefilter, as a DataFrame with the same column names/
        order. No .sort() -- the original SQL has no ORDER BY, so no
        ordering is invented here either."""
        _PREFILTER_COLUMNS = (
            "comment_id", "original_comment", "visitor_email",
            "visitor_name", "comment_created_at", "is_exact_duplicate_of",
        )
        projection = {
            "_id": 1, "original_comment": 1, "visitor_email": 1,
            "visitor_name": 1, "comment_created_at": 1, "is_exact_duplicate_of": 1,
        }
        query = {"_id": {"$in": list(comment_ids)}} if comment_ids is not None else {}
        docs = with_transient_retry(lambda: list(self.collection.find(query, projection)))
        records = [
            {
                "comment_id": d["_id"],
                "original_comment": d.get("original_comment"),
                "visitor_email": d.get("visitor_email"),
                "visitor_name": d.get("visitor_name"),
                "comment_created_at": d.get("comment_created_at"),
                "is_exact_duplicate_of": d.get("is_exact_duplicate_of"),
            }
            for d in docs
        ]
        df = pd.DataFrame(records, columns=list(_PREFILTER_COLUMNS))
        df["comment_created_at"] = pd.to_datetime(df["comment_created_at"])
        return df

    def apply_prefilter_batch(self, results: list[dict]) -> dict:
        """Phase M4C.2 -- one transaction per call, matching Postgres's
        one-Session/one-commit whole-call atomicity. ONE upfront read to
        detect any missing comment_id (raises KeyError and attempts NO
        write at all if found -- an even stricter match to "nothing
        persisted on a missing item" than Postgres's own commit-at-the-
        end behavior, with an identical observable outcome). `now` is
        computed ONCE, shared across the whole batch, exactly as
        Postgres. No review_status guard -- exactly as Postgres. Never
        touches row_version."""
        if not results:
            return {"excluded": 0, "passed": 0}
        now = dt.datetime.utcnow()
        counts = {"excluded": 0, "passed": 0}

        def _callback(session):
            counts["excluded"] = 0
            counts["passed"] = 0

            ids = [row["comment_id"] for row in results]
            existing_ids = {
                d["_id"] for d in self.collection.find({"_id": {"$in": ids}}, {"_id": 1}, session=session)
            }
            missing = [i for i in ids if i not in existing_ids]
            if missing:
                raise KeyError(f"comment_id {missing[0]} not found in review_items")

            ops = []
            for row in results:
                is_junk = bool(row["is_prefiltered_junk"])
                reason = row["prefilter_reason"]
                if is_junk:
                    fields = {
                        "is_prefiltered_junk": True,
                        "prefilter_reason": reason,
                        "processed_at": now,
                        "primary_intent": "SPAM",
                        "routing_decision": "EXCLUDE",
                        "review_status": "EXCLUDED",
                        "exclusion_reason": reason,
                    }
                    counts["excluded"] += 1
                else:
                    fields = {
                        "is_prefiltered_junk": False,
                        "prefilter_reason": reason,
                        "processed_at": now,
                        "primary_intent": None,
                        "routing_decision": None,
                        "review_status": "UNPROCESSED",
                        "exclusion_reason": None,
                    }
                    counts["passed"] += 1
                ops.append(UpdateOne({"_id": row["comment_id"]}, {"$set": fields}))

            if ops:
                self.collection.bulk_write(ops, ordered=False, session=session)

        client = self.db.client
        with client.start_session() as session:
            session.with_transaction(_callback)

        return dict(counts)

    def load_for_classification(self) -> pd.DataFrame:
        """Phase M4C.3 -- exact eligibility/column set as
        PostgresReviewItemRepository.load_for_classification. `False`
        (exact BSON boolean equality) for is_prefiltered_junk mirrors
        Postgres's `.is_(False)` -- strict IS FALSE, excluding null/
        missing, NOT `$ne: True` (which would incorrectly include null/
        missing documents). No .sort() -- no ordering in the original."""
        _CLASSIFICATION_COLUMNS = (
            "comment_id", "original_comment", "original_title", "original_type",
            "age_group", "platform", "entry_id", "lemma",
        )
        projection = {
            "_id": 1, "original_comment": 1, "original_title": 1, "original_type": 1,
            "age_group": 1, "platform": 1, "entry_id": 1, "lemma": 1,
        }
        query = {"review_status": "UNPROCESSED", "is_prefiltered_junk": False}
        docs = with_transient_retry(lambda: list(self.collection.find(query, projection)))
        records = [
            {
                "comment_id": d["_id"],
                "original_comment": d.get("original_comment"),
                "original_title": d.get("original_title"),
                "original_type": d.get("original_type"),
                "age_group": d.get("age_group"),
                "platform": d.get("platform"),
                "entry_id": d.get("entry_id"),
                "lemma": d.get("lemma"),
            }
            for d in docs
        ]
        return pd.DataFrame(records, columns=list(_CLASSIFICATION_COLUMNS))

    def apply_classification_batch(
        self,
        results: list[dict],
        *,
        model_name: str,
        model_version: str | None,
        prompt_version: str,
        pipeline_version: str,
        inference_run_id: str,
    ) -> dict:
        """Phase M4C.3 -- one transaction per call, matching Postgres's
        one-Session/one-commit whole-call atomicity. ONE upfront read
        (projecting only ai_primary_intent/review_status, the two fields
        the immutability guard needs) both detects any missing
        comment_id (raises KeyError, no write attempted at all -- same
        reasoning as apply_prefilter_batch) and supplies the skip
        decision for every row from one consistent, transaction-
        snapshotted read. `now` computed ONCE, shared across the whole
        batch. Never touches row_version, reviewer fields, or generation/
        outbound fields."""
        if not results:
            return {"classified": 0, "failed": 0, "skipped_already_classified": 0}
        now = dt.datetime.utcnow()
        counts = {"classified": 0, "failed": 0, "skipped_already_classified": 0}

        def _callback(session):
            counts["classified"] = 0
            counts["failed"] = 0
            counts["skipped_already_classified"] = 0

            ids = [r["comment_id"] for r in results]
            existing_docs = list(
                self.collection.find(
                    {"_id": {"$in": ids}},
                    {"_id": 1, "ai_primary_intent": 1, "review_status": 1},
                    session=session,
                )
            )
            existing_by_id = {d["_id"]: d for d in existing_docs}
            missing = [i for i in ids if i not in existing_by_id]
            if missing:
                raise KeyError(f"comment_id {missing[0]} not found in review_items")

            ops = []
            for result in results:
                comment_id = result["comment_id"]
                doc = existing_by_id[comment_id]
                if doc.get("ai_primary_intent") is not None or doc.get("review_status") != "UNPROCESSED":
                    counts["skipped_already_classified"] += 1
                    continue

                fields = {
                    "model_name": model_name,
                    "model_version": model_version,
                    "prompt_version": prompt_version,
                    "pipeline_version": pipeline_version,
                    "inference_run_id": inference_run_id,
                    "processed_at": now,
                }
                if result["success"]:
                    fields.update(
                        {
                            "ai_primary_intent": result["ai_primary_intent"],
                            "ai_topic_subtype": result["ai_topic_subtype"],
                            "ai_subcategory": result["ai_subcategory"],
                            "ai_secondary_intent": result["ai_secondary_intent"],
                            "ai_routing_decision": result["ai_routing_decision"],
                            "ai_classification_reason": result["ai_classification_reason"],
                            "ai_confidence": result["ai_confidence"],
                            "ai_classification_source": result.get("ai_classification_source"),
                            "ai_needs_review": result.get("ai_needs_review"),
                            "primary_intent": result["ai_primary_intent"],
                            "topic_subtype": result["ai_topic_subtype"],
                            "subcategory": result["ai_subcategory"],
                            "secondary_intent": result["ai_secondary_intent"],
                            "classification_source": result.get("ai_classification_source"),
                            "needs_review": result.get("ai_needs_review"),
                            "routing_decision": result["ai_routing_decision"],
                            "exclusion_reason": (
                                result.get("routing_reason_code")
                                if result["ai_routing_decision"] == "EXCLUDE"
                                else None
                            ),
                            "review_status": "CLASSIFIED",
                        }
                    )
                    if result.get("ai_sentiment"):
                        fields["ai_sentiment"] = result["ai_sentiment"]
                        fields["sentiment"] = result["ai_sentiment"]
                    counts["classified"] += 1
                else:
                    fields.update(
                        {
                            "ai_routing_decision": "REVIEW_ONLY",
                            "ai_classification_reason": result["ai_classification_reason"],
                            # review_status intentionally left as "UNPROCESSED" -- retry-eligible.
                        }
                    )
                    counts["failed"] += 1
                ops.append(UpdateOne({"_id": comment_id}, {"$set": fields}))

            if ops:
                self.collection.bulk_write(ops, ordered=False, session=session)

        client = self.db.client
        with client.start_session() as session:
            session.with_transaction(_callback)

        return dict(counts)


class MongoReviewSessionRepository:
    def __init__(self, db):
        self.db = db
        self.collection = db["review_sessions"]
        self.review_items = db["review_items"]

    def _to_item(self, doc):
        return _doc_to_item(doc, id_field="session_id", table="review_sessions")

    def create(self, *, original_filename: str, source_file: str, upload_token: str):
        session_id = uuid.uuid4().hex
        now = dt.datetime.utcnow()
        doc = {
            "_id": session_id,
            "original_filename": original_filename,
            "source_file": source_file,
            "upload_token": upload_token,
            "status": "uploaded",
            "created_at": now,
            "updated_at": now,
        }
        self.collection.insert_one(doc)
        return self._to_item(doc)

    def get(self, session_id: str):
        doc = with_transient_retry(lambda: self.collection.find_one({"_id": session_id}))
        return self._to_item(doc)

    def get_active(self):
        active_statuses = ("uploaded", "processing", "ready", "completed")
        doc = with_transient_retry(
            lambda: self.collection.find_one({"status": {"$in": list(active_statuses)}}, sort=[("created_at", -1)])
        )
        return self._to_item(doc)

    def set_status(self, session_id: str, status: str) -> None:
        self.collection.update_one({"_id": session_id}, {"$set": {"status": status, "updated_at": dt.datetime.utcnow()}})

    def compute_counts(self, source_file: str) -> SessionCounts:
        """Single aggregation pipeline over review_items -- mirrors the
        exact conditional-$sum shape of PostgresReviewSessionRepository.
        compute_counts, which was itself a deliberate fix for a real,
        previously-observed ~7.5s regression. Never fetch rows here."""
        pipeline = [
            {"$match": {"source_file": source_file}},
            {"$group": {
                "_id": None,
                "total": {"$sum": 1},
                "excluded": {"$sum": {"$cond": [{"$eq": ["$review_status", "EXCLUDED"]}, 1, 0]}},
                "duplicate_content": {"$sum": {"$cond": [{"$eq": ["$review_status", "DUPLICATE_CONTENT"]}, 1, 0]}},
                "completed": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$not": [{"$in": ["$review_status", ["EXCLUDED", "DUPLICATE_CONTENT"]]}]},
                                    {"$ne": ["$reviewer_action", None]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "auto_resolved": {"$sum": {"$cond": [{"$eq": ["$review_status", "NO_RESPONSE_RECOMMENDED"]}, 1, 0]}},
            }},
        ]
        rows = with_transient_retry(lambda: list(self.review_items.aggregate(pipeline)))
        if not rows:
            return SessionCounts(total=0, excluded=0, duplicate_content=0, canonical_total=0, completed=0, remaining=0)
        r = rows[0]
        total = int(r["total"])
        excluded = int(r["excluded"])
        duplicate_content = int(r["duplicate_content"])
        # NO_RESPONSE_RECOMMENDED is pending human review, not completed
        # work: it is deliberately not added to `completed`.
        completed_or_auto = int(r["completed"])
        canonical_total = total - excluded - duplicate_content
        remaining = max(0, canonical_total - completed_or_auto)
        return SessionCounts(
            total=total,
            excluded=excluded,
            duplicate_content=duplicate_content,
            canonical_total=canonical_total,
            completed=completed_or_auto,
            remaining=remaining,
        )


class MongoProcessingJobRepository:
    def __init__(self, db):
        self.db = db
        self.collection = db["processing_jobs"]

    def _to_item(self, doc):
        return _doc_to_item(doc, id_field="job_id", table="processing_jobs")

    def create(self, session_id: str):
        from rcs.review.sessions import STAGE_PREPARE

        job_id = uuid.uuid4().hex
        now = dt.datetime.utcnow()
        doc = {
            "_id": job_id,
            "session_id": session_id,
            "status": "running",
            "current_stage": STAGE_PREPARE,
            "rows_done": 0,
            "rows_total": None,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "error_message": None,
        }
        self.collection.insert_one(doc)
        return self._to_item(doc)

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        current_stage: str | None = None,
        rows_done: int | None = None,
        rows_total: int | None = None,
        error_message: str | None = None,
        finished: bool = False,
    ) -> None:
        fields: dict = {}
        if status is not None:
            fields["status"] = status
        if current_stage is not None:
            fields["current_stage"] = current_stage
        if rows_done is not None:
            fields["rows_done"] = rows_done
        if rows_total is not None:
            fields["rows_total"] = rows_total
        if error_message is not None:
            fields["error_message"] = error_message
        now = dt.datetime.utcnow()
        fields["updated_at"] = now
        if finished:
            fields["finished_at"] = now
        self.collection.update_one({"_id": job_id}, {"$set": fields})

    def get(self, job_id: str):
        doc = with_transient_retry(lambda: self.collection.find_one({"_id": job_id}))
        return self._to_item(doc)

    def latest_for_session(self, session_id: str):
        doc = with_transient_retry(
            lambda: self.collection.find_one({"session_id": session_id}, sort=[("started_at", -1)])
        )
        return self._to_item(doc)

    def list_running(self):
        docs = with_transient_retry(lambda: list(self.collection.find({"status": "running"})))
        return [self._to_item(d) for d in docs]


class MongoAnalyticsRepository:
    """Phase M4C.5 -- server-side aggregation counterpart to
    PostgresAnalyticsRepository.compute_operational_analytics. Read-only:
    every method here is count_documents/$match+$group/find, never a
    write, so the established bounded transient read retry layer
    (with_transient_retry) is used throughout, exactly like every other
    read method in this file.

    Two deliberately-preserved (not "improved") findings from the audit,
    see the M4C.5 report:

    1. `ingested_at` is a "utc"-convention column while the analytics
       `now`/`start`/`end` boundaries are naive-Riyadh wall-clock values
       -- the ORIGINAL Postgres code compares them with zero conversion,
       an embedded ~3-hour mismatch that predates this migration. This
       implementation reproduces that exact comparison (via
       _query_bound, which passes "utc"-convention boundaries through
       unconverted) rather than silently correcting it.
    2. `reviewed_at`/`reviewer_started_at` are "riyadh"-convention
       columns -- boundaries ARE converted (_query_bound) before use in
       a filter, and raw fetched values are converted back
       (_from_bson_value) before Python arithmetic, so duration/bucket
       calculations operate on the same naive-Riyadh values Postgres's
       own session.execute(...) rows would yield.

    Average-duration and trend-bucket calculations fetch the same
    bounded (already server-side filtered) row set Postgres's own
    session.execute(...).all() fetches and finish the arithmetic in
    Python -- not "loading the entire collection": PostgresAnalytics
    Repository itself computes these in Python today (round(sum/len,1),
    strftime bucketing), so mirroring that here is exact-parity, not a
    shortcut. Every count/group-by metric is genuinely server-side
    (count_documents / $group)."""

    def __init__(self, db):
        self.db = db
        self.collection = db["review_items"]

    @staticmethod
    def _unbound(field: str, value):
        if value is None:
            return None
        convention = _DATETIME_CONVENTION.get("review_items", {}).get(field)
        return _from_bson_value(value, convention)

    def compute_operational_analytics(
        self,
        *,
        range_key: str = "week",
        now: dt.datetime | None = None,
        reviewer_id: str | None = None,
        custom_start: dt.datetime | None = None,
        custom_end: dt.datetime | None = None,
        source_file: str | None = None,
        reviewer_user_id: str | None = None,
    ) -> dict:
        now = clock.now_naive(now)
        start, end = range_bounds(range_key, now, start=custom_start, end=custom_end)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        overdue_cut = now - dt.timedelta(days=OVERDUE_DAYS)

        scope: dict = {"source_file": source_file} if source_file else {}
        if reviewer_user_id:
            scope["reviewer_user_id"] = reviewer_user_id

        pending_q = {"review_status": {"$in": list(PENDING)}, **scope}
        pending = with_transient_retry(lambda: self.collection.count_documents(pending_q))
        urgent = with_transient_retry(
            lambda: self.collection.count_documents({**pending_q, "review_priority": 1})
        )
        overdue = with_transient_retry(
            lambda: self.collection.count_documents({
                **pending_q,
                "ingested_at": {"$ne": None, "$lte": _query_bound("ingested_at", overdue_cut)},
            })
        )
        oldest_doc = with_transient_retry(
            lambda: self.collection.find_one(
                {**pending_q, "ingested_at": {"$ne": None}}, {"ingested_at": 1}, sort=[("ingested_at", 1)]
            )
        )
        oldest_pending = self._unbound("ingested_at", oldest_doc["ingested_at"]) if oldest_doc else None
        oldest_age = int((now - oldest_pending).total_seconds()) if oldest_pending else None

        ingested_range: dict = {"$ne": None, "$lte": _query_bound("ingested_at", end)}
        if start is not None:
            ingested_range["$gte"] = _query_bound("ingested_at", start)
        reviewed_range: dict = {"$ne": None, "$lte": _query_bound("reviewed_at", end)}
        if start is not None:
            reviewed_range["$gte"] = _query_bound("reviewed_at", start)

        incoming = with_transient_retry(
            lambda: self.collection.count_documents({"ingested_at": ingested_range, **scope})
        )
        reviewed = with_transient_retry(
            lambda: self.collection.count_documents({"reviewed_at": reviewed_range, **scope})
        )

        duration_docs = with_transient_retry(
            lambda: list(self.collection.find(
                {"reviewer_started_at": {"$ne": None}, "reviewed_at": reviewed_range, **scope},
                {"reviewer_started_at": 1, "reviewed_at": 1},
            ))
        )
        durations = []
        for doc in duration_docs:
            started = self._unbound("reviewer_started_at", doc.get("reviewer_started_at"))
            done = self._unbound("reviewed_at", doc.get("reviewed_at"))
            if started and done and done >= started:
                durations.append((done - started).total_seconds())
        avg_seconds = round(sum(durations) / len(durations), 1) if durations else None

        action_rows = with_transient_retry(
            lambda: list(self.collection.aggregate([
                {"$match": {"reviewer_action": {"$in": list(OUTCOME_ACTIONS)}, "reviewed_at": reviewed_range, **scope}},
                {"$group": {"_id": "$reviewer_action", "count": {"$sum": 1}}},
            ]))
        )
        outcomes = {key: 0 for key in OUTCOME_ACTIONS}
        for row in action_rows:
            outcomes[str(row["_id"])] = int(row["count"])
        outcome_total = sum(outcomes.values())

        today_reviewed_range = {
            "$ne": None,
            "$gte": _query_bound("reviewed_at", today_start),
            "$lte": _query_bound("reviewed_at", now),
        }
        today_action_rows = with_transient_retry(
            lambda: list(self.collection.aggregate([
                {"$match": {"reviewer_action": {"$in": list(OUTCOME_ACTIONS)}, "reviewed_at": today_reviewed_range, **scope}},
                {"$group": {"_id": "$reviewer_action", "count": {"$sum": 1}}},
            ]))
        )
        today_actions = {key: 0 for key in OUTCOME_ACTIONS}
        for row in today_action_rows:
            today_actions[str(row["_id"])] = int(row["count"])
        reviewed_today = with_transient_retry(
            lambda: self.collection.count_documents({"reviewed_at": today_reviewed_range, **scope})
        )
        reviewed_by_me = 0
        if reviewer_id:
            reviewed_by_me = with_transient_retry(
                lambda: self.collection.count_documents(
                    {"reviewed_at": today_reviewed_range, "reviewer_id": reviewer_id, **scope}
                )
            )

        pending_docs = with_transient_retry(lambda: list(self.collection.find(pending_q, {"ingested_at": 1})))
        health = {"under_day": 0, "one_two": 0, "three_seven": 0, "over_seven": 0}
        dated = 0
        for doc in pending_docs:
            stamp = self._unbound("ingested_at", doc.get("ingested_at"))
            bucket = _bucket_age(stamp, now)
            if not bucket:
                continue
            health[bucket] += 1
            dated += 1

        class_query = {"ingested_at": ingested_range, **scope}
        intent_rows = with_transient_retry(
            lambda: list(self.collection.aggregate([
                {"$match": {"primary_intent": {"$ne": None}, **class_query}},
                {"$group": {"_id": "$primary_intent", "count": {"$sum": 1}}},
            ]))
        )
        topic_rows = with_transient_retry(
            lambda: list(self.collection.aggregate([
                {"$match": {"topic_subtype": {"$ne": None}, **class_query}},
                {"$group": {"_id": "$topic_subtype", "count": {"$sum": 1}}},
            ]))
        )
        sentiment_rows = with_transient_retry(
            lambda: list(self.collection.aggregate([
                {"$match": {"sentiment": {"$ne": None}, **class_query}},
                {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}},
            ]))
        )

        trend = self._trend_series(
            start, end, now, range_key, source_file=source_file, reviewer_user_id=reviewer_user_id
        )
        quality_sample = outcome_total
        quality = {
            "sample": quality_sample,
            "send_as_is_rate": round(outcomes["SEND_AS_IS"] / quality_sample, 3) if quality_sample else None,
            "edited_rate": round(outcomes["EDIT_AND_SEND"] / quality_sample, 3) if quality_sample else None,
            "no_reply_rate": round(outcomes["NO_RESPONSE_NEEDED"] / quality_sample, 3) if quality_sample else None,
        }

        no_reply_all = with_transient_retry(
            lambda: self.collection.count_documents(
                {"review_status": "NO_REPLY", **scope}
            )
        )

        intent, intent_unmapped = _complete_counts(
            settings.PRIMARY_INTENTS, [(row["_id"], row["count"]) for row in intent_rows]
        )
        topic, topic_unmapped = _complete_counts(
            settings.TOPIC_SUBTYPES, [(row["_id"], row["count"]) for row in topic_rows]
        )
        sentiment, sentiment_unmapped = _complete_counts(
            settings.SENTIMENTS, [(row["_id"], row["count"]) for row in sentiment_rows]
        )
        unmapped = {"intent": intent_unmapped, "topic": topic_unmapped, "sentiment": sentiment_unmapped}
        has_unmapped = any(unmapped[dim] for dim in unmapped)

        return {
            "range": range_key,
            "range_start": clock.iso_app(start) if start else None,
            "range_end": clock.iso_app(end) or end.isoformat(),
            "kpis": {
                "incoming": incoming,
                "reviewed": reviewed,
                "pending": pending,
                "urgent_pending": urgent,
                "overdue_pending": overdue,
                "no_reply": no_reply_all,
                "avg_review_seconds": avg_seconds,
                "avg_review_sample": len(durations),
            },
            "today": {
                "reviewed_all": reviewed_today,
                "reviewed_by_me": reviewed_by_me,
                "actions": today_actions,
            },
            "outcomes": outcomes,
            "queue_health": {
                **health,
                "dated": dated,
                "oldest_pending_age_seconds": oldest_age,
            },
            "quality": quality,
            "classification": {"intent": intent, "topic": topic, "sentiment": sentiment},
            "data_quality": {
                "unmapped_classification": unmapped,
                "note": "legacy_unmapped_labels_excluded_from_canonical_taxonomy" if has_unmapped else None,
            },
            "trend": trend,
            "availability": {
                "avg_review_time": "ok" if durations else "insufficient_timestamps",
                "outcomes": "ok" if outcome_total else "no_completed_actions",
                "trend": "ok" if any(p["incoming"] or p["reviewed"] for p in trend) else "empty_window",
                "classification": "ok" if any(row["count"] for row in intent_rows) else "no_labels",
            },
        }

    def _trend_series(self, start, end, now, range_key, *, source_file=None, reviewer_user_id=None):
        scope: dict = {"source_file": source_file} if source_file else {}
        if reviewer_user_id:
            scope["reviewer_user_id"] = reviewer_user_id
        hourly = (range_key or "").lower() == "today"
        if (range_key or "").lower() == "custom" and start is not None:
            hourly = (end - start) <= dt.timedelta(days=2)
        fmt = "%Y-%m-%d %H" if hourly else "%Y-%m-%d"

        ingested_range: dict = {"$ne": None, "$lte": _query_bound("ingested_at", end)}
        if start is not None:
            ingested_range["$gte"] = _query_bound("ingested_at", start)
        reviewed_range: dict = {"$ne": None, "$lte": _query_bound("reviewed_at", end)}
        if start is not None:
            reviewed_range["$gte"] = _query_bound("reviewed_at", start)

        incoming_docs = with_transient_retry(
            lambda: list(self.collection.find({"ingested_at": ingested_range, **scope}, {"ingested_at": 1}))
        )
        reviewed_docs = with_transient_retry(
            lambda: list(self.collection.find({"reviewed_at": reviewed_range, **scope}, {"reviewed_at": 1}))
        )
        incoming_map: dict[str, int] = {}
        for doc in incoming_docs:
            stamp = self._unbound("ingested_at", doc["ingested_at"])
            key = stamp.strftime(fmt)
            incoming_map[key] = incoming_map.get(key, 0) + 1
        reviewed_map: dict[str, int] = {}
        for doc in reviewed_docs:
            stamp = self._unbound("reviewed_at", doc["reviewed_at"])
            key = stamp.strftime(fmt)
            reviewed_map[key] = reviewed_map.get(key, 0) + 1
        keys: list[str] = []
        window_start = start or (now - dt.timedelta(days=13))
        if hourly:
            cursor = window_start.replace(minute=0, second=0, microsecond=0)
            while cursor <= end:
                keys.append(cursor.strftime("%Y-%m-%d %H"))
                cursor += dt.timedelta(hours=1)
                if len(keys) > 48:
                    break
        else:
            cursor = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
            last = end.replace(hour=0, minute=0, second=0, microsecond=0)
            while cursor <= last:
                keys.append(cursor.strftime("%Y-%m-%d"))
                cursor += dt.timedelta(days=1)
                if len(keys) > 62:
                    break
        return [
            {"bucket": key, "incoming": incoming_map.get(key, 0), "reviewed": reviewed_map.get(key, 0)}
            for key in keys
        ]


class MongoUserRepository:
    """Phase M4D.1 -- `users` collection, `_id` = the existing stable
    `user_id` string (never an ObjectId -- matches the already-verified
    migrated staging shape, per instruction). Reads use the established
    bounded transient read-retry layer; `create` does not (it is a
    mutation, and its own duplicate-email check is itself the reason a
    blind retry would be unsafe here -- retrying could re-run the check
    against a since-changed collection state and still race the same
    way, see the M4D.1 report's email-uniqueness finding).

    Critical, explicitly NOT claimed as fixed: unlike PostgreSQL (whose
    `users.email` column has a real unique index as a hard safety net
    beneath this same read-then-insert check), this collection has NO
    unique index yet. Two genuinely concurrent create() calls with the
    same email can both pass the check and both insert -- true
    concurrent-safe uniqueness requires a real Mongo unique index on
    `email` before any cutover; see the M4D.1 report's index
    recommendations. This is disclosed, not silently assumed safe."""

    def __init__(self, db):
        self.db = db
        self.collection = db["users"]

    def _to_item(self, doc):
        return _doc_to_item(doc, id_field="user_id", table="users")

    def get_by_id(self, user_id: str):
        doc = with_transient_retry(lambda: self.collection.find_one({"_id": user_id}))
        return self._to_item(doc)

    def get_by_email(self, email: str):
        doc = with_transient_retry(lambda: self.collection.find_one({"email": email}))
        return self._to_item(doc)

    def list_all(self):
        docs = with_transient_retry(lambda: list(self.collection.find({}).sort([("created_at", 1)])))
        return [self._to_item(d) for d in docs]

    def list_active_reviewers(self):
        docs = with_transient_retry(
            lambda: list(self.collection.find({"role": "REVIEWER", "is_active": True}).sort([("display_name", 1)]))
        )
        return [self._to_item(d) for d in docs]

    def create(
        self,
        *,
        user_id: str,
        email: str,
        password_hash: str,
        display_name: str,
        role: str,
        is_active: bool,
        created_at: dt.datetime,
        updated_at: dt.datetime,
    ):
        existing = self.collection.find_one({"email": email})
        if existing is not None:
            raise ValueError(f"a user with email {email!r} already exists")
        doc = {
            "_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "display_name": display_name,
            "role": role,
            "is_active": is_active,
            "created_at": _to_bson_value(created_at, "utc"),
            "updated_at": _to_bson_value(updated_at, "utc"),
        }
        self.collection.insert_one(doc)
        return self._to_item(doc)

    def set_password_hash(self, user_id: str, password_hash: str, updated_at: dt.datetime) -> bool:
        result = self.collection.update_one({"_id": user_id}, {"$set": {"password_hash": password_hash, "updated_at": _to_bson_value(updated_at, "utc")}})
        return result.matched_count == 1


class MongoUserSessionRepository:
    """Phase M4D.2 -- `user_sessions` collection, `_id` = the existing
    `session_token` string (never an ObjectId) -- session_token is
    already the natural PK in Postgres, so this preserves that identity
    exactly, per instruction. Reads use the established bounded
    transient read-retry layer; mutations (create/delete_by_token/
    delete_expired) do not -- see UserSessionRepository's docstring on
    why a blind insert with no existing-token check is the correct
    verbatim-preserving behavior here."""

    def __init__(self, db):
        self.db = db
        self.collection = db["user_sessions"]

    def _to_item(self, doc):
        return _doc_to_item(doc, id_field="session_token", table="user_sessions")

    def create(self, *, session_token: str, user_id: str, created_at: dt.datetime, expires_at: dt.datetime):
        doc = {
            "_id": session_token,
            "user_id": user_id,
            "created_at": _to_bson_value(created_at, "utc"),
            "expires_at": _to_bson_value(expires_at, "utc"),
        }
        self.collection.insert_one(doc)
        return self._to_item(doc)

    def get_by_token(self, session_token: str):
        doc = with_transient_retry(lambda: self.collection.find_one({"_id": session_token}))
        return self._to_item(doc)

    def delete_by_token(self, session_token: str) -> None:
        self.collection.delete_one({"_id": session_token})

    def delete_expired(self, now: dt.datetime) -> int:
        result = self.collection.delete_many({"expires_at": {"$lte": _to_bson_value(now, "utc")}})
        return result.deleted_count

    def delete_for_user(self, user_id: str) -> int:
        return self.collection.delete_many({"user_id": user_id}).deleted_count


class MongoPasswordResetTokenRepository:
    """`password_reset_tokens` collection, `_id` = the token HASH (never the raw token). `consume` is one atomic find_one_and_update whose filter requires
    used_at == None and expires_at > now, so two concurrent redemptions cannot both succeed."""

    def __init__(self, db):
        self.db = db
        self.collection = db["password_reset_tokens"]

    def create(self, *, token_hash: str, user_id: str, created_at: dt.datetime, expires_at: dt.datetime) -> None:
        self.collection.insert_one(
            {"_id": token_hash, "user_id": user_id, "created_at": _to_bson_value(created_at, "utc"), "expires_at": _to_bson_value(expires_at, "utc"), "used_at": None}
        )

    def consume(self, token_hash: str, now: dt.datetime) -> str | None:
        stamp = _to_bson_value(now, "utc")
        doc = self.collection.find_one_and_update(
            {"_id": token_hash, "used_at": None, "expires_at": {"$gt": stamp}}, {"$set": {"used_at": stamp}}, return_document=ReturnDocument.BEFORE
        )
        return None if doc is None else doc["user_id"]

    def invalidate_for_user(self, user_id: str, now: dt.datetime) -> int:
        result = self.collection.update_many({"user_id": user_id, "used_at": None}, {"$set": {"used_at": _to_bson_value(now, "utc")}})
        return result.modified_count

    def delete_expired(self, now: dt.datetime) -> int:
        return self.collection.delete_many({"expires_at": {"$lte": _to_bson_value(now, "utc")}}).deleted_count


def _trace_doc_to_record(doc: dict):
    from rcs.response_intelligence.audit import AgentAuditRecord

    return AgentAuditRecord(
        execution_id=doc["_id"],
        review_item_id=doc["review_item_id"],
        created_at=doc["created_at"],
        grounding_mode=doc["grounding_mode"],
        official_retrieval_outcome=doc["official_retrieval_outcome"],
        official_result_count=doc["official_result_count"],
        official_grounding_available=doc["official_grounding_available"],
        style_retrieval_outcome=doc["style_retrieval_outcome"],
        style_result_count=doc["style_result_count"],
        logical_generation_count=doc["logical_generation_count"],
        refinement_attempted=doc["refinement_attempted"],
        initial_validation_passed=doc["initial_validation_passed"],
        initial_failure_code=doc.get("initial_failure_code"),
        final_validation_passed=doc["final_validation_passed"],
        final_failure_code=doc.get("final_failure_code"),
        generation_status=doc["generation_status"],
    )


class MongoAgentAuditRepository:
    """Phase M5E. Deferred/not-wired status: identical to every other
    class in this module (see the module docstring) -- no backend
    selector exists yet, and this is exercised only by unit tests
    against a hand-built fake (never a live Atlas connection), matching
    `MongoProcessingJobRepository`'s own established pattern. Returns
    real `AgentAuditRecord` instances (not a SimpleNamespace duck-type)
    since there is no dto.py consumer requiring that shape for this
    entity -- only `rcs.response_intelligence.audit.AgentAuditRepository`
    callers, which expect exactly `AgentAuditRecord`.

    `record()` is idempotent on `execution_id` (`_id`): a duplicate
    insert raises `pymongo.errors.DuplicateKeyError`, caught and treated
    as "already recorded" -- never a second document, mirroring
    `PostgresAgentAuditRepository`'s same guarantee."""

    def __init__(self, db):
        self.db = db
        self.collection = db["agent_execution_traces"]

    def record(self, record) -> None:
        from pymongo.errors import DuplicateKeyError

        doc = {
            "_id": record.execution_id,
            "review_item_id": record.review_item_id,
            "created_at": record.created_at,
            "grounding_mode": record.grounding_mode,
            "official_retrieval_outcome": record.official_retrieval_outcome,
            "official_result_count": record.official_result_count,
            "official_grounding_available": record.official_grounding_available,
            "style_retrieval_outcome": record.style_retrieval_outcome,
            "style_result_count": record.style_result_count,
            "logical_generation_count": record.logical_generation_count,
            "refinement_attempted": record.refinement_attempted,
            "initial_validation_passed": record.initial_validation_passed,
            "initial_failure_code": record.initial_failure_code,
            "final_validation_passed": record.final_validation_passed,
            "final_failure_code": record.final_failure_code,
            "generation_status": record.generation_status,
        }
        try:
            self.collection.insert_one(doc)
        except DuplicateKeyError:
            pass

    def list_for_review_item(self, review_item_id: str) -> tuple:
        cursor = self.collection.find({"review_item_id": review_item_id}).sort("created_at", 1)
        return tuple(_trace_doc_to_record(doc) for doc in cursor)

    def get_by_execution_id(self, execution_id: str):
        doc = self.collection.find_one({"_id": execution_id})
        return _trace_doc_to_record(doc) if doc is not None else None


def _message_from_doc(d):
    from rcs.review.repositories.contracts import DirectMessageRecord

    return DirectMessageRecord(
        d["_id"], d["reviewer_user_id"], d["sender_user_id"], d["sender_role"], d["body"], d["client_message_id"],
        _from_bson_value(d["created_at"], "utc"), _from_bson_value(d.get("read_at"), "utc") if d.get("read_at") is not None else None,
    )


class MongoDirectMessageRepository:
    """`direct_messages` collection. `_id` is a deterministic hash of (sender_user_id, client_message_id), so a retried or racing identical send hits the
    primary-key duplicate error and returns the stored row -- no secondary unique index is needed for idempotency. `message_id` is that `_id`."""

    def __init__(self, db):
        self.db = db
        self.collection = db["direct_messages"]

    @staticmethod
    def _key(sender_user_id: str, client_message_id: str) -> str:
        return hashlib.sha256(f"{sender_user_id}|{client_message_id}".encode("utf-8")).hexdigest()

    def add(self, *, message_id, reviewer_user_id, sender_user_id, sender_role, body, client_message_id, created_at):
        from pymongo.errors import DuplicateKeyError

        _id = self._key(sender_user_id, client_message_id)
        doc = {"_id": _id, "reviewer_user_id": reviewer_user_id, "sender_user_id": sender_user_id, "sender_role": sender_role, "body": body,
               "client_message_id": client_message_id, "created_at": _to_bson_value(created_at, "utc"), "read_at": None}
        try:
            self.collection.insert_one(doc)
            return _message_from_doc(doc), True
        except DuplicateKeyError:
            return _message_from_doc(self.collection.find_one({"_id": _id})), False

    def list_conversation(self, reviewer_user_id, limit):
        docs = list(self.collection.find({"reviewer_user_id": reviewer_user_id}).sort([("created_at", -1), ("_id", -1)]).limit(limit))
        return [_message_from_doc(d) for d in reversed(docs)]

    def mark_read(self, reviewer_user_id, sender_role, read_at):
        stamp = _to_bson_value(read_at, "utc")
        n = 0
        for d in list(self.collection.find({"reviewer_user_id": reviewer_user_id, "sender_role": sender_role, "read_at": None})):
            n += self.collection.update_one({"_id": d["_id"], "read_at": None}, {"$set": {"read_at": stamp}}).modified_count
        return n

    def unread_count(self, reviewer_user_id, sender_role):
        return self.collection.count_documents({"reviewer_user_id": reviewer_user_id, "sender_role": sender_role, "read_at": None})

    def summaries(self):
        from rcs.review.repositories.contracts import ConversationSummary

        out = {}
        for rid in self.collection.distinct("reviewer_user_id"):
            last = self.collection.find_one({"reviewer_user_id": rid}, sort=[("created_at", -1), ("_id", -1)])
            out[rid] = ConversationSummary(rid, _message_from_doc(last), self.unread_count(rid, "REVIEWER"), self.unread_count(rid, "ADMIN"))
        return out


def _account_request_from_doc(d):
    from rcs.review.repositories.contracts import AccountRequestRecord

    return AccountRequestRecord(
        d["_id"], d["email"], d["display_name"], d["password_hash"], d["status"], _from_bson_value(d["created_at"], "utc"),
        _from_bson_value(d["decided_at"], "utc") if d.get("decided_at") is not None else None, d.get("decided_by_user_id"),
    )


class MongoAccountRequestRepository:
    """`account_requests` collection. One document per e-mail: `_id` is the request id, and a second document with the same e-mail is prevented by also keeping
    a sentinel document `_id = "email:<address>"` -- inserting it is the atomic uniqueness check (primary-key duplicate), so no secondary index is needed."""

    def __init__(self, db):
        self.db = db
        self.collection = db["account_requests"]

    def create(self, *, request_id, email, display_name, password_hash, created_at):
        from pymongo.errors import DuplicateKeyError

        try:
            self.collection.insert_one({"_id": f"email:{email}", "kind": "email_claim", "request_id": request_id})
        except DuplicateKeyError:
            return False
        self.collection.insert_one({"_id": request_id, "kind": "request", "email": email, "display_name": display_name, "password_hash": password_hash, "status": "PENDING",
                                    "created_at": _to_bson_value(created_at, "utc"), "decided_at": None, "decided_by_user_id": None})
        return True

    def get(self, request_id):
        d = self.collection.find_one({"_id": request_id, "kind": "request"})
        return None if d is None else _account_request_from_doc(d)

    def list_pending(self, limit):
        docs = list(self.collection.find({"kind": "request", "status": "PENDING"}).sort([("created_at", 1), ("_id", 1)]).limit(limit))
        return [_account_request_from_doc(d) for d in docs]

    def count_pending(self):
        return self.collection.count_documents({"kind": "request", "status": "PENDING"})

    def decide(self, request_id, status, decided_by_user_id, decided_at):
        before = self.collection.find_one_and_update(
            {"_id": request_id, "kind": "request", "status": "PENDING"},
            {"$set": {"status": status, "decided_at": _to_bson_value(decided_at, "utc"), "decided_by_user_id": decided_by_user_id}}, return_document=ReturnDocument.BEFORE,
        )
        return None if before is None else _account_request_from_doc(before)
