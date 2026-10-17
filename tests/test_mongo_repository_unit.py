"""
Phase M4B — pure/unit tests for the Mongo repository implementations.

NO Atlas connection, NO network call, NO pymongo server round-trip.
Every test runs against a small, purpose-built in-memory fake
(_FakeDatabase below) that implements exactly the pymongo surface
rcs.review.repositories.mongo actually calls (find_one, find with sort/
skip/limit, count_documents, distinct, aggregate for the two pipeline
shapes this code produces, find_one_and_update, update_one, insert_one,
and a session/with_transaction pair that snapshots-and-restores on
exception).

This is NOT a general MongoDB emulator and does NOT prove MongoDB's own
server-side ACID/aggregation-engine behavior -- it proves that THIS
CODE (a) builds the right query/pipeline shapes, (b) interprets results
correctly, and (c) correctly aborts/rolls back its OWN staged writes
when assign_batch's transaction callback raises partway through. The
genuine, server-enforced transactional guarantee can only be verified
against a live replica-set-backed Mongo (Atlas) -- see
test_mongo_repository_live.py, which is BLOCKED in this environment
(see that file's skip reason).
"""

from __future__ import annotations

import datetime as dt
import re as _re

import pytest
from pymongo import InsertOne, UpdateOne
from pymongo.errors import DuplicateKeyError

from rcs.review.repositories.contracts import SessionCounts
from rcs.review.repositories.mongo import (
    MongoProcessingJobRepository,
    MongoReviewItemRepository,
    MongoReviewSessionRepository,
    MongoUserRepository,
    MongoUserSessionRepository,
)
from rcs.review.workflow import VersionConflict


# --- minimal in-memory pymongo-shaped fake ---------------------------------

def _matches(doc: dict, filt: dict) -> bool:
    if "$or" in filt:
        clauses = filt["$or"]
        rest = {k: v for k, v in filt.items() if k != "$or"}
        return any(_matches(doc, c) for c in clauses) and _matches(doc, rest)
    for key, cond in filt.items():
        val = doc.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and val not in cond["$in"]:
                return False
            if "$nin" in cond and val in cond["$nin"]:
                return False
            if "$ne" in cond and val == cond["$ne"]:
                return False
            if "$regex" in cond and (val is None or not _re.search(cond["$regex"], val)):
                return False
            if "$gte" in cond and (val is None or val < cond["$gte"]):
                return False
            if "$lte" in cond and (val is None or val > cond["$lte"]):
                return False
            if "$gt" in cond and (val is None or val <= cond["$gt"]):
                return False
            if "$lt" in cond and (val is None or val >= cond["$lt"]):
                return False
        else:
            if val != cond:
                return False
    return True


def _apply_update(doc: dict, update: dict) -> None:
    for key, value in update.get("$set", {}).items():
        doc[key] = value
    for key, value in update.get("$inc", {}).items():
        doc[key] = (doc.get(key) or 0) + value


def _eval_expr(expr, doc):
    if isinstance(expr, str) and expr.startswith("$"):
        return doc.get(expr[1:])
    return expr


def _eval_bool(expr, doc) -> bool:
    if isinstance(expr, dict):
        if "$eq" in expr:
            a, b = expr["$eq"]
            return _eval_expr(a, doc) == _eval_expr(b, doc)
        if "$ne" in expr:
            a, b = expr["$ne"]
            return _eval_expr(a, doc) != _eval_expr(b, doc)
        if "$in" in expr:
            a, b = expr["$in"]
            return _eval_expr(a, doc) in b
        if "$and" in expr:
            return all(_eval_bool(e, doc) for e in expr["$and"])
        if "$not" in expr:
            inner = expr["$not"]
            target = inner[0] if isinstance(inner, list) else inner
            return not _eval_bool(target, doc)
    raise ValueError(f"unsupported expr in fake aggregate: {expr!r}")


def _eval_sum_expr(sum_expr, doc):
    """`sum_expr` is the inner value of a `{"$sum": <inner>}` accumulator."""
    inner = sum_expr["$sum"] if isinstance(sum_expr, dict) and "$sum" in sum_expr else sum_expr
    if inner == 1:
        return 1
    if isinstance(inner, dict) and "$cond" in inner:
        cond, then_, else_ = inner["$cond"]
        return then_ if _eval_bool(cond, doc) else else_
    raise ValueError(f"unsupported $sum expr in fake aggregate: {sum_expr!r}")


class _FakeDeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count


class _FakeUpdateResult:
    def __init__(self, matched_count: int, modified_count: int):
        self.matched_count = matched_count
        self.modified_count = modified_count


class _FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key_or_list, direction=None):
        """Matches real pymongo's Cursor.sort: either a single key plus
        a direction (`.sort("field", 1)`), or a list of (key, direction)
        pairs (`.sort([("a", 1), ("b", -1)])`)."""
        if direction is not None:
            spec = [(key_or_list, direction)]
        elif isinstance(key_or_list, tuple):
            spec = [key_or_list]
        else:
            spec = key_or_list
        for key, direction in reversed(spec):
            # (is_none, value) rather than a bare value: real MongoDB
            # treats a missing/null sort field as its own consistent
            # rank rather than raising, and Python's own `None < None`
            # has no such ordering -- the tuple's leading bool lets two
            # None entries compare equal (via `==`) without ever
            # invoking `<` on the Nones themselves.
            self._docs.sort(key=lambda d: (d.get(key) is None, d.get(key)), reverse=(direction == -1))
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(dict(d) for d in self._docs)


class _FakeCollection:
    def __init__(self):
        self._docs: list[dict] = []

    def insert_one(self, doc, **_):
        if any(d["_id"] == doc["_id"] for d in self._docs):
            raise DuplicateKeyError(f"E11000 duplicate key error, _id: {doc['_id']!r}")
        self._docs.append(dict(doc))

    def bulk_write(self, ops, ordered=True, **_):
        """Minimal fake: applies InsertOne/UpdateOne in order. A
        duplicate-key InsertOne raises immediately (mirroring: inside a
        real transaction, any such error aborts the whole transaction --
        there is no partial-tolerance even under ordered=False)."""
        for op in ops:
            if isinstance(op, InsertOne):
                self.insert_one(op._doc)
            elif isinstance(op, UpdateOne):
                self.update_one(op._filter, op._doc)
            else:
                raise TypeError(f"fake bulk_write does not support op type: {type(op)!r}")

    def find_one(self, filt=None, projection=None, sort=None, **_):
        filt = filt or {}
        candidates = [d for d in self._docs if _matches(d, filt)]
        if sort:
            spec = [sort] if isinstance(sort, tuple) else sort
            for key, direction in reversed(spec):
                candidates.sort(key=lambda d: d.get(key), reverse=(direction == -1))
        if not candidates:
            return None
        doc = dict(candidates[0])
        if projection:
            doc = {k: v for k, v in doc.items() if k in projection or k == "_id"}
        return doc

    def find(self, filt=None, projection=None, **_):
        filt = filt or {}
        docs = [d for d in self._docs if _matches(d, filt)]
        if projection:
            docs = [{k: v for k, v in d.items() if k in projection or k == "_id"} for d in docs]
        return _FakeCursor(docs)

    def count_documents(self, filt, **_):
        return sum(1 for d in self._docs if _matches(d, filt or {}))

    def distinct(self, field, **_):
        return list({d.get(field) for d in self._docs if field in d})

    def find_one_and_update(self, filt, update, return_document=None, **_):
        for doc in self._docs:
            if _matches(doc, filt):
                _apply_update(doc, update)
                return dict(doc)
        return None

    def update_one(self, filt, update, **_):
        for doc in self._docs:
            if _matches(doc, filt):
                _apply_update(doc, update)
                return _FakeUpdateResult(matched_count=1, modified_count=1)
        return _FakeUpdateResult(matched_count=0, modified_count=0)

    def delete_one(self, filt, **_):
        for i, doc in enumerate(self._docs):
            if _matches(doc, filt):
                del self._docs[i]
                return _FakeDeleteResult(1)
        return _FakeDeleteResult(0)

    def delete_many(self, filt, **_):
        keep = [d for d in self._docs if not _matches(d, filt)]
        removed = len(self._docs) - len(keep)
        self._docs = keep
        return _FakeDeleteResult(removed)

    def aggregate(self, pipeline, **_):
        docs = [dict(d) for d in self._docs]
        for stage in pipeline:
            if "$match" in stage:
                docs = [d for d in docs if _matches(d, stage["$match"])]
            elif "$group" in stage:
                group = stage["$group"]
                buckets: dict = {}
                for d in docs:
                    key = _eval_expr(group["_id"], d)
                    bucket = buckets.setdefault(key, {"_id": key})
                    for field, expr in group.items():
                        if field == "_id":
                            continue
                        bucket[field] = bucket.get(field, 0) + _eval_sum_expr(expr, d)
                docs = list(buckets.values())
        return docs


class _FakeSession:
    def __init__(self, db):
        self._db = db

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def with_transaction(self, callback):
        snapshot = {name: [dict(d) for d in coll._docs] for name, coll in self._db._collections.items()}
        try:
            callback(self)
        except Exception:
            for name, docs in snapshot.items():
                self._db._collections[name]._docs = docs
            raise


class _FakeClient:
    def __init__(self, db):
        self._db = db

    def start_session(self):
        return _FakeSession(self._db)


class _FakeDatabase:
    def __init__(self):
        self._collections: dict[str, _FakeCollection] = {}
        self.client = _FakeClient(self)

    def __getitem__(self, name):
        return self._collections.setdefault(name, _FakeCollection())


# --- fixtures ---------------------------------------------------------------

def _item_doc(comment_id: str, *, source_row_index: int = 0, **overrides) -> dict:
    doc = dict(
        _id=comment_id,
        source_file="contract.xlsx",
        source_row_index=source_row_index,
        original_comment=f"تعليق {comment_id}",
        review_status="DRAFT_READY",
        suggested_final_reply="رد مقترح",
        reviewer_action=None,
        reviewer_id=None,
        reviewer_display_name=None,
        reviewer_user_id=None,
        reviewer_started_at=None,
        reviewer_edit=None,
        final_reply=None,
        no_response_reason=None,
        dedupe_canonical_id=None,
        primary_intent=None,
        review_priority=3,
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0),
        reviewed_at=None,
        ready_to_send=False,
        outbound_status=None,
        outbound_idempotency_key=None,
        row_version=1,
    )
    doc.update(overrides)
    return doc


@pytest.fixture
def db():
    return _FakeDatabase()


# 1: get item by ID
def test_get_returns_item_by_id(db):
    db["review_items"].insert_one(_item_doc("c1"))
    repo = MongoReviewItemRepository(db)
    item = repo.get("c1")
    assert item.comment_id == "c1"
    assert repo.get("missing") is None


# 2 + 3: reviewer-scoped queue / ADMIN vs reviewer isolation support
def test_list_queue_admin_sees_all_reviewer_sees_own(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("for-a", reviewer_user_id="u-a"))
    coll.insert_one(_item_doc("for-b", reviewer_user_id="u-b"))
    repo = MongoReviewItemRepository(db)

    admin_items, admin_total = repo.list_queue(
        statuses=("DRAFT_READY",), source_file=None, reviewer_user_id=None,
        search=None, sort="priority_oldest", offset=0, limit=50,
    )
    assert admin_total == 2
    assert {i.comment_id for i in admin_items} == {"for-a", "for-b"}

    reviewer_items, reviewer_total = repo.list_queue(
        statuses=("DRAFT_READY",), source_file=None, reviewer_user_id="u-a",
        search=None, sort="priority_oldest", offset=0, limit=50,
    )
    assert reviewer_total == 1
    assert {i.comment_id for i in reviewer_items} == {"for-a"}


# 4: source-file filtering
def test_list_queue_filters_by_source_file(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("in-file", source_file="a.xlsx"))
    coll.insert_one(_item_doc("other-file", source_file="b.xlsx"))
    repo = MongoReviewItemRepository(db)
    items, total = repo.list_queue(
        statuses=("DRAFT_READY",), source_file="a.xlsx", reviewer_user_id=None,
        search=None, sort="priority_oldest", offset=0, limit=50,
    )
    assert total == 1
    assert items[0].comment_id == "in-file"


# 5: status filtering
def test_list_queue_filters_by_status(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("draft", review_status="DRAFT_READY"))
    coll.insert_one(_item_doc("excluded", review_status="EXCLUDED"))
    repo = MongoReviewItemRepository(db)
    items, total = repo.list_queue(
        statuses=("DRAFT_READY",), source_file=None, reviewer_user_id=None,
        search=None, sort="priority_oldest", offset=0, limit=50,
    )
    assert total == 1
    assert items[0].comment_id == "draft"


# 6: deterministic ordering
def test_list_queue_orders_priority_then_ingested_at(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("p2-early", review_priority=2, ingested_at=dt.datetime(2026, 1, 1)))
    coll.insert_one(_item_doc("p1", review_priority=1, ingested_at=dt.datetime(2026, 1, 2)))
    coll.insert_one(_item_doc("p2-late", review_priority=2, ingested_at=dt.datetime(2026, 1, 3)))
    repo = MongoReviewItemRepository(db)
    items, _ = repo.list_queue(
        statuses=("DRAFT_READY",), source_file=None, reviewer_user_id=None,
        search=None, sort="priority_oldest", offset=0, limit=50,
    )
    assert [i.comment_id for i in items] == ["p1", "p2-early", "p2-late"]


# 7: assignment result buckets
def test_assign_batch_reports_correct_buckets(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("eligible"))
    coll.insert_one(_item_doc("decided", reviewer_action="SEND_AS_IS", review_status="APPROVED"))
    coll.insert_one(_item_doc("not-reviewable", review_status="EXCLUDED"))
    repo = MongoReviewItemRepository(db)
    result = repo.assign_batch(["eligible", "decided", "not-reviewable", "missing"], "u-1")
    assert result.assigned == ("eligible",)
    assert result.skipped_already_decided == ("decided",)
    assert result.skipped_not_reviewable == ("not-reviewable",)
    assert result.not_found == ("missing",)


# 8: same-reviewer assignment idempotency
def test_assign_batch_same_reviewer_is_idempotent(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("c1"))
    repo = MongoReviewItemRepository(db)
    repo.assign_batch(["c1"], "u-1")
    version_after_first = repo.get("c1").row_version
    repo.assign_batch(["c1"], "u-1")
    version_after_second = repo.get("c1").row_version
    assert version_after_first == version_after_second


# 9: batch rollback on injected mid-batch failure
def test_assign_batch_rolls_back_on_injected_mid_batch_failure(db, monkeypatch):
    coll = db["review_items"]
    coll.insert_one(_item_doc("c1"))
    coll.insert_one(_item_doc("c2"))
    coll.insert_one(_item_doc("c3"))
    repo = MongoReviewItemRepository(db)

    call_count = {"n": 0}
    real_find_one = coll.find_one

    def _find_one_then_fail_on_second(filt=None, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-batch failure")
        return real_find_one(filt, *args, **kwargs)

    monkeypatch.setattr(coll, "find_one", _find_one_then_fail_on_second)

    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        repo.assign_batch(["c1", "c2", "c3"], "u-1")

    for cid in ("c1", "c2", "c3"):
        item = repo.get(cid)
        assert item.reviewer_user_id is None, f"{cid} should not be persisted after rollback"
        assert item.row_version == 1


# 10: stale row_version conflict
def test_claim_raises_version_conflict_on_stale_expected_version(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("c1", row_version=1))
    repo = MongoReviewItemRepository(db)
    repo.claim("c1", reviewer_id="r1", display_name="R1", user_id="u1", expected_version=1)  # -> row_version 2
    with pytest.raises(VersionConflict):
        repo.claim("c1", reviewer_id="r2", display_name="R2", user_id="u2", expected_version=1)


# 11: atomic row_version increment
def test_claim_atomically_applies_fields_and_increments_row_version(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("c1", row_version=5))
    repo = MongoReviewItemRepository(db)
    item = repo.claim("c1", reviewer_id="r1", display_name="R1", user_id="u1", expected_version=5)
    assert item.row_version == 6
    assert item.reviewer_id == "r1"
    assert item.review_status == "IN_REVIEW"


# 12: review-session CRUD
def test_review_session_create_get_set_status(db):
    repo = MongoReviewSessionRepository(db)
    created = repo.create(original_filename="f.xlsx", source_file="f.xlsx", upload_token="tok")
    fetched = repo.get(created.session_id)
    assert fetched.status == "uploaded"
    repo.set_status(created.session_id, "processing")
    assert repo.get(created.session_id).status == "processing"
    assert repo.get_active().session_id == created.session_id


# 13: session count semantics
def test_compute_counts_matches_expected_semantics(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("s1", source_file="s.xlsx", review_status="EXCLUDED"))
    coll.insert_one(_item_doc("s2", source_file="s.xlsx", review_status="DUPLICATE_CONTENT"))
    coll.insert_one(_item_doc("s3", source_file="s.xlsx", review_status="DRAFT_READY"))
    coll.insert_one(_item_doc("s4", source_file="s.xlsx", review_status="APPROVED", reviewer_action="SEND_AS_IS"))
    coll.insert_one(_item_doc("s5", source_file="s.xlsx", review_status="NO_RESPONSE_RECOMMENDED"))
    repo = MongoReviewSessionRepository(db)
    counts = repo.compute_counts("s.xlsx")
    assert counts == SessionCounts(total=5, excluded=1, duplicate_content=1, canonical_total=3, completed=1, remaining=2)


# 14: processing-job CRUD
def test_processing_job_create_update_get_latest(db):
    repo = MongoProcessingJobRepository(db)
    job = repo.create("session-1")
    assert job.status == "running"
    repo.update(job.job_id, status="succeeded", rows_done=10, finished=True)
    fetched = repo.get(job.job_id)
    assert fetched.status == "succeeded"
    assert fetched.rows_done == 10
    assert fetched.finished_at is not None
    assert repo.latest_for_session("session-1").job_id == job.job_id
    assert repo.list_running() == []


# 15: ordered export reads
def test_list_ordered_by_source_row_index(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("high", source_row_index=8))
    coll.insert_one(_item_doc("low", source_row_index=2))
    coll.insert_one(_item_doc("mid", source_row_index=5))
    repo = MongoReviewItemRepository(db)
    items = repo.list_ordered(None)
    assert [i.comment_id for i in items] == ["low", "mid", "high"]


# 16: canonical dedupe reads
def test_get_many_for_canonical_lookup(db):
    coll = db["review_items"]
    coll.insert_one(_item_doc("canonical", primary_intent="THANKS_PRAISE"))
    coll.insert_one(_item_doc("dup", review_status="DUPLICATE_CONTENT", dedupe_canonical_id="canonical"))
    repo = MongoReviewItemRepository(db)
    canonicals = repo.get_many(["canonical"])
    assert set(canonicals.keys()) == {"canonical"}
    assert canonicals["canonical"].primary_intent == "THANKS_PRAISE"


# --- Production login/DB boundary hardening — UserRepository / -------------
# UserSessionRepository contract parity (auth.py/sessions.py's actual
# runtime callers: create/get_by_id/get_by_email/list_all/
# list_active_reviewers, and create/get_by_token/delete_by_token/
# delete_expired) against the exact same in-memory fake every other
# Mongo contract test in this file uses. No live MongoDB connection.


# 17: user create + lookup by id/email
def test_user_create_and_lookup_by_id_and_email(db):
    repo = MongoUserRepository(db)
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    created = repo.create(
        user_id="u-1",
        email="reviewer@example.test",
        password_hash="hashed",
        display_name="مراجع",
        role="REVIEWER",
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    assert created.user_id == "u-1"
    assert repo.get_by_id("u-1").email == "reviewer@example.test"
    assert repo.get_by_email("reviewer@example.test").user_id == "u-1"
    assert repo.get_by_id("does-not-exist") is None
    assert repo.get_by_email("nobody@example.test") is None


# 18: duplicate email rejected, exactly like Postgres's unique-index-backed check
def test_user_create_rejects_duplicate_email(db):
    repo = MongoUserRepository(db)
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    repo.create(
        user_id="u-1", email="dup@example.test", password_hash="h", display_name="A",
        role="REVIEWER", is_active=True, created_at=now, updated_at=now,
    )
    with pytest.raises(ValueError):
        repo.create(
            user_id="u-2", email="dup@example.test", password_hash="h2", display_name="B",
            role="ADMIN", is_active=True, created_at=now, updated_at=now,
        )


# 19: list_all / list_active_reviewers filtering
def test_user_list_all_and_list_active_reviewers(db):
    repo = MongoUserRepository(db)
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    repo.create(user_id="a-1", email="admin@example.test", password_hash="h", display_name="Admin", role="ADMIN", is_active=True, created_at=now, updated_at=now)
    repo.create(user_id="r-1", email="active-reviewer@example.test", password_hash="h", display_name="Active Reviewer", role="REVIEWER", is_active=True, created_at=now, updated_at=now)
    repo.create(user_id="r-2", email="disabled-reviewer@example.test", password_hash="h", display_name="Disabled Reviewer", role="REVIEWER", is_active=False, created_at=now, updated_at=now)

    assert {u.user_id for u in repo.list_all()} == {"a-1", "r-1", "r-2"}
    active = repo.list_active_reviewers()
    assert [u.user_id for u in active] == ["r-1"]  # excludes disabled reviewer and the admin


# 20: session create/get/delete + expiry-based bulk delete
def test_user_session_create_get_delete_and_expire(db):
    repo = MongoUserSessionRepository(db)
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    expires = now + dt.timedelta(hours=12)
    created = repo.create(session_token="tok-1", user_id="u-1", created_at=now, expires_at=expires)
    assert created.session_token == "tok-1"

    fetched = repo.get_by_token("tok-1")
    assert fetched is not None and fetched.user_id == "u-1" and fetched.expires_at == expires
    assert repo.get_by_token("unknown-token") is None

    repo.delete_by_token("tok-1")
    assert repo.get_by_token("tok-1") is None
    repo.delete_by_token("tok-1")  # idempotent -- deleting an already-gone token is a silent no-op

    repo.create(session_token="expired", user_id="u-1", created_at=now, expires_at=now - dt.timedelta(minutes=1))
    repo.create(session_token="still-valid", user_id="u-1", created_at=now, expires_at=now + dt.timedelta(hours=1))
    deleted_count = repo.delete_expired(now)
    assert deleted_count == 1
    assert repo.get_by_token("expired") is None
    assert repo.get_by_token("still-valid") is not None
