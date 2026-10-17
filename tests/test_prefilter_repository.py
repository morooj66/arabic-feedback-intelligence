"""
Phase M4C.2 — prefilter repository contract + parity tests.

Postgres and Mongo scenarios are placed side by side per behavior, same
convention as test_ingest_repository.py. The Mongo tests run against the
same in-memory fake used in test_mongo_repository_unit.py -- they prove
THIS CODE's logic, not MongoDB's own server-enforced transactional
guarantee. See test_mongo_repository_live.py for what remains BLOCKED
pending a working Atlas connection.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.review.repositories import contracts
from rcs.review.repositories.mongo import MongoReviewItemRepository
from rcs.review.repositories.postgres import PostgresReviewItemRepository
from rcs.review.repository import get_engine, init_db, upsert_review_items

from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake


def _engine(tmp_path, name="prefilter.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _seed_existing(engine, comment_id: str, **overrides) -> None:
    row = dict(
        comment_id=comment_id,
        source_file="f.xlsx",
        source_row_index=0,
        original_comment="نص تجريبي",
        review_status="UNPROCESSED",
        ready_to_send=False,
        row_version=1,
        is_prefiltered_junk=None,
        prefilter_reason=None,
        processed_at=None,
        primary_intent=None,
        routing_decision=None,
        exclusion_reason=None,
    )
    row.update(overrides)
    upsert_review_items(engine, [row])


def _mongo_doc(comment_id: str, **overrides) -> dict:
    doc = dict(
        _id=comment_id,
        original_comment="نص تجريبي",
        visitor_email=None,
        visitor_name=None,
        comment_created_at=dt.datetime(2026, 1, 1),
        is_exact_duplicate_of=None,
        review_status="UNPROCESSED",
        reviewer_action=None,
        row_version=1,
        is_prefiltered_junk=None,
        prefilter_reason=None,
        processed_at=None,
        primary_intent=None,
        routing_decision=None,
        exclusion_reason=None,
    )
    doc.update(overrides)
    return doc


# Contract/signature coverage.
def test_contract_declares_prefilter_methods():
    assert hasattr(contracts.ReviewItemRepository, "load_for_prefilter")
    assert hasattr(contracts.ReviewItemRepository, "apply_prefilter_batch")
    assert hasattr(PostgresReviewItemRepository, "load_for_prefilter")
    assert hasattr(PostgresReviewItemRepository, "apply_prefilter_batch")
    assert hasattr(MongoReviewItemRepository, "load_for_prefilter")
    assert hasattr(MongoReviewItemRepository, "apply_prefilter_batch")


def test_load_for_prefilter_column_shape_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    df = PostgresReviewItemRepository(engine).load_for_prefilter(["c1"])
    assert list(df.columns) == [
        "comment_id", "original_comment", "visitor_email",
        "visitor_name", "comment_created_at", "is_exact_duplicate_of",
    ]
    assert len(df) == 1


def test_load_for_prefilter_column_shape_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    df = MongoReviewItemRepository(db).load_for_prefilter(["c1"])
    assert list(df.columns) == [
        "comment_id", "original_comment", "visitor_email",
        "visitor_name", "comment_created_at", "is_exact_duplicate_of",
    ]
    assert len(df) == 1


# A. Non-junk result (both backends).
def test_non_junk_result_fields_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": False, "prefilter_reason": None}])
    assert result == {"excluded": 0, "passed": 1}
    item = repo.get("c1")
    assert item.is_prefiltered_junk is False
    assert item.prefilter_reason is None
    assert item.processed_at is not None
    assert item.primary_intent is None
    assert item.routing_decision is None
    assert item.review_status == "UNPROCESSED"
    assert item.exclusion_reason is None
    assert item.row_version == 1


def test_non_junk_result_fields_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": False, "prefilter_reason": None}])
    assert result == {"excluded": 0, "passed": 1}
    item = repo.get("c1")
    assert item.is_prefiltered_junk is False
    assert item.prefilter_reason is None
    assert item.processed_at is not None
    assert item.primary_intent is None
    assert item.routing_decision is None
    assert item.review_status == "UNPROCESSED"
    assert item.exclusion_reason is None
    assert item.row_version == 1


# B. Junk result (both backends).
def test_junk_result_fields_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "known_junk_string"}])
    assert result == {"excluded": 1, "passed": 0}
    item = repo.get("c1")
    assert item.is_prefiltered_junk is True
    assert item.prefilter_reason == "known_junk_string"
    assert item.processed_at is not None
    assert item.primary_intent == "SPAM"
    assert item.routing_decision == "EXCLUDE"
    assert item.review_status == "EXCLUDED"
    assert item.exclusion_reason == "known_junk_string"
    assert item.row_version == 1


def test_junk_result_fields_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "known_junk_string"}])
    assert result == {"excluded": 1, "passed": 0}
    item = repo.get("c1")
    assert item.is_prefiltered_junk is True
    assert item.prefilter_reason == "known_junk_string"
    assert item.processed_at is not None
    assert item.primary_intent == "SPAM"
    assert item.routing_decision == "EXCLUDE"
    assert item.review_status == "EXCLUDED"
    assert item.exclusion_reason == "known_junk_string"
    assert item.row_version == 1


# C. Caller supplies an item already advanced beyond UNPROCESSED --
# intentional PARITY testing of the current, unguarded behavior. Not an
# endorsement of the business design (see the M4C.2 audit note).
def test_no_status_guard_reclassifies_advanced_item_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", review_status="DRAFT_READY", primary_intent="THANKS_PRAISE")
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"}])
    assert result == {"excluded": 1, "passed": 0}
    item = repo.get("c1")
    assert item.review_status == "EXCLUDED"  # overwritten despite being DRAFT_READY -- no guard, matches Postgres exactly
    assert item.primary_intent == "SPAM"


def test_no_status_guard_reclassifies_advanced_item_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1", review_status="DRAFT_READY", primary_intent="THANKS_PRAISE"))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"}])
    assert result == {"excluded": 1, "passed": 0}
    item = repo.get("c1")
    assert item.review_status == "EXCLUDED"
    assert item.primary_intent == "SPAM"


# D. Reapplying the same result -- business-idempotent (same field values,
# same counts) but processed_at IS rewritten on every call. Not a true
# no-op: documented explicitly, not silently called "no-op".
def test_reapply_same_result_rewrites_processed_at_postgres(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    row = {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"}

    monkeypatch.setattr("rcs.review.repositories.postgres.dt", _FixedDt(dt.datetime(2026, 1, 1, 0, 0, 0)))
    first = repo.apply_prefilter_batch([row])
    ts1 = repo.get("c1").processed_at

    monkeypatch.setattr("rcs.review.repositories.postgres.dt", _FixedDt(dt.datetime(2026, 1, 2, 0, 0, 0)))
    second = repo.apply_prefilter_batch([row])
    ts2 = repo.get("c1").processed_at

    assert first == second == {"excluded": 1, "passed": 0}  # business-idempotent counts
    assert ts1 != ts2  # but the timestamp DOES change -- not a true no-op
    assert repo.get("c1").row_version == 1  # still never bumped


def test_reapply_same_result_rewrites_processed_at_mongo(monkeypatch):
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    row = {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"}

    monkeypatch.setattr("rcs.review.repositories.mongo.dt", _FixedDt(dt.datetime(2026, 1, 1, 0, 0, 0)))
    first = repo.apply_prefilter_batch([row])
    ts1 = repo.get("c1").processed_at

    monkeypatch.setattr("rcs.review.repositories.mongo.dt", _FixedDt(dt.datetime(2026, 1, 2, 0, 0, 0)))
    second = repo.apply_prefilter_batch([row])
    ts2 = repo.get("c1").processed_at

    assert first == second == {"excluded": 1, "passed": 0}
    assert ts1 != ts2
    assert repo.get("c1").row_version == 1


class _FixedDt:
    """Minimal stand-in for the `dt` module so `dt.datetime.utcnow()`
    returns a fixed, caller-chosen value without touching the real clock
    module used elsewhere."""

    def __init__(self, fixed: dt.datetime):
        class _FixedDatetime:
            @staticmethod
            def utcnow():
                return fixed

        self.datetime = _FixedDatetime


# Missing-item semantics: raises, whole call aborts, nothing persisted.
def test_missing_item_raises_and_aborts_whole_batch_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    with pytest.raises(KeyError):
        repo.apply_prefilter_batch([
            {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "missing", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        ])
    item = repo.get("c1")
    assert item.is_prefiltered_junk is None  # untouched -- whole call aborted, nothing committed
    assert item.review_status == "UNPROCESSED"


def test_missing_item_raises_and_aborts_whole_batch_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    with pytest.raises(KeyError):
        repo.apply_prefilter_batch([
            {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "missing", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        ])
    item = repo.get("c1")
    assert item.is_prefiltered_junk is None
    assert item.review_status == "UNPROCESSED"


# row_version: explicit regression -- never bumped, insert-then-reapply.
def test_row_version_never_bumped_by_prefilter_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", row_version=1)
    repo = PostgresReviewItemRepository(engine)
    for _ in range(3):
        repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"}])
    assert repo.get("c1").row_version == 1


def test_row_version_never_bumped_by_prefilter_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1", row_version=1))
    repo = MongoReviewItemRepository(db)
    for _ in range(3):
        repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"}])
    assert repo.get("c1").row_version == 1


# Transaction tests: one item / multiple items success.
def test_multiple_item_success_postgres(tmp_path):
    engine = _engine(tmp_path)
    for cid in ("c1", "c2", "c3"):
        _seed_existing(engine, cid)
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_prefilter_batch([
        {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        {"comment_id": "c2", "is_prefiltered_junk": False, "prefilter_reason": None},
        {"comment_id": "c3", "is_prefiltered_junk": True, "prefilter_reason": "known_junk_string"},
    ])
    assert result == {"excluded": 2, "passed": 1}


def test_multiple_item_success_mongo():
    db = _FakeDatabase()
    for cid in ("c1", "c2", "c3"):
        db["review_items"].insert_one(_mongo_doc(cid))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_prefilter_batch([
        {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        {"comment_id": "c2", "is_prefiltered_junk": False, "prefilter_reason": None},
        {"comment_id": "c3", "is_prefiltered_junk": True, "prefilter_reason": "known_junk_string"},
    ])
    assert result == {"excluded": 2, "passed": 1}


# Injected mid-batch failure -> whole batch rolls back (both backends).
# The Mongo assertion proves THIS CODE's callback correctly leaves
# nothing behind -- NOT proof of MongoDB's own server-enforced ACID
# guarantee (see test_mongo_repository_live.py for what that requires).
def test_injected_mid_batch_failure_rolls_back_postgres(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    for cid in ("c1", "c2", "c3"):
        _seed_existing(engine, cid)
    repo = PostgresReviewItemRepository(engine)

    from sqlalchemy.orm import Session as _Session

    real_get = _Session.get
    call_count = {"n": 0}

    def _get_then_fail_on_third(self, model, ident, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("simulated mid-batch failure")
        return real_get(self, model, ident, *a, **kw)

    monkeypatch.setattr(_Session, "get", _get_then_fail_on_third)
    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        repo.apply_prefilter_batch([
            {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "c2", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "c3", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        ])
    monkeypatch.undo()
    for cid in ("c1", "c2", "c3"):
        assert PostgresReviewItemRepository(engine).get(cid).is_prefiltered_junk is None


def test_injected_mid_batch_failure_rolls_back_mongo(monkeypatch):
    db = _FakeDatabase()
    for cid in ("c1", "c2", "c3"):
        db["review_items"].insert_one(_mongo_doc(cid))
    repo = MongoReviewItemRepository(db)

    coll = db["review_items"]
    real_bulk_write = coll.bulk_write

    def _bulk_write_fails(ops, ordered=True, **kw):
        raise RuntimeError("simulated mid-batch failure")

    monkeypatch.setattr(coll, "bulk_write", _bulk_write_fails)
    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        repo.apply_prefilter_batch([
            {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "c2", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "c3", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        ])
    monkeypatch.undo()
    for cid in ("c1", "c2", "c3"):
        assert repo.get(cid).is_prefiltered_junk is None


# Callback re-execution -> counters/results do not leak (Mongo).
def test_callback_reexecution_does_not_leak_counters_mongo(monkeypatch):
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    db["review_items"].insert_one(_mongo_doc("c2"))
    repo = MongoReviewItemRepository(db)

    coll = db["review_items"]
    call_count = {"n": 0}
    real_find = coll.find

    def _find_fail_first_attempt_only(filt=None, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated transient error on first attempt")
        return real_find(filt, *a, **kw)

    monkeypatch.setattr(coll, "find", _find_fail_first_attempt_only)
    with pytest.raises(RuntimeError, match="simulated transient error"):
        repo.apply_prefilter_batch([
            {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "c2", "is_prefiltered_junk": False, "prefilter_reason": None},
        ])
    monkeypatch.undo()
    assert repo.get("c1").is_prefiltered_junk is None  # untouched by the discarded first attempt

    final = repo.apply_prefilter_batch([
        {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        {"comment_id": "c2", "is_prefiltered_junk": False, "prefilter_reason": None},
    ])
    assert final == {"excluded": 1, "passed": 1}  # not 2/2 (which would mean leaked/double-counted)


# Same batch re-run behavior: idempotent result, still bounded counts.
def test_same_batch_rerun_behavior_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    row = [{"comment_id": "c1", "is_prefiltered_junk": False, "prefilter_reason": None}]
    r1 = repo.apply_prefilter_batch(row)
    r2 = repo.apply_prefilter_batch(row)
    assert r1 == r2 == {"excluded": 0, "passed": 1}


def test_same_batch_rerun_behavior_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    row = [{"comment_id": "c1", "is_prefiltered_junk": False, "prefilter_reason": None}]
    r1 = repo.apply_prefilter_batch(row)
    r2 = repo.apply_prefilter_batch(row)
    assert r1 == r2 == {"excluded": 0, "passed": 1}


# Type/datetime parity: processed_at is a real datetime, shared across
# every row in one call (matching the "now computed once" original).
def test_processed_at_is_real_datetime_shared_across_batch_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    _seed_existing(engine, "c2")
    repo = PostgresReviewItemRepository(engine)
    repo.apply_prefilter_batch([
        {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        {"comment_id": "c2", "is_prefiltered_junk": False, "prefilter_reason": None},
    ])
    ts1 = repo.get("c1").processed_at
    ts2 = repo.get("c2").processed_at
    assert isinstance(ts1, dt.datetime)
    assert ts1 == ts2


def test_processed_at_is_real_datetime_shared_across_batch_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    db["review_items"].insert_one(_mongo_doc("c2"))
    repo = MongoReviewItemRepository(db)
    repo.apply_prefilter_batch([
        {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        {"comment_id": "c2", "is_prefiltered_junk": False, "prefilter_reason": None},
    ])
    ts1 = repo.get("c1").processed_at
    ts2 = repo.get("c2").processed_at
    assert isinstance(ts1, dt.datetime)
    assert ts1 == ts2


# Ensure the prefilter WRITE path never uses mongo_retry.with_transient_retry.
def test_apply_prefilter_batch_never_uses_mongo_retry_wrapper(monkeypatch):
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)

    def _fail_if_called(fn, *, max_attempts=None):
        raise AssertionError("apply_prefilter_batch must never call with_transient_retry")

    monkeypatch.setattr("rcs.review.repositories.mongo.with_transient_retry", _fail_if_called)
    result = repo.apply_prefilter_batch([{"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"}])
    assert result == {"excluded": 1, "passed": 0}
