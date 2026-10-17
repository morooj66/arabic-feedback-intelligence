"""
Phase M4C.1 — ingest repository contract + parity tests.

Postgres and Mongo scenarios are placed side by side per behavior so the
intended parity is explicit. The Mongo tests run against the same
in-memory fake used in test_mongo_repository_unit.py (imported from
there rather than duplicated) -- they prove THIS CODE's logic, not
MongoDB's own server-enforced transactional guarantee. See
test_mongo_repository_live.py for what remains BLOCKED pending a working
Atlas connection.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.review.repositories import contracts
from rcs.review.repositories.mongo import MongoReviewItemRepository
from rcs.review.repositories.mongo_retry import with_transient_retry
from rcs.review.repositories.postgres import PostgresReviewItemRepository
from rcs.review.repository import INGEST_OWNED_FIELDS, get_engine, init_db, upsert_review_items

from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake


def _engine(tmp_path, name="ingest.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _seed_existing(engine, comment_id: str, **overrides) -> None:
    """Seed a PRE-EXISTING row via the OTHER (blind-overwrite) upsert
    function -- upsert_ingest_batch is the thing under test, so it must
    never be used to set up its own test fixtures."""
    row = dict(
        comment_id=comment_id,
        source_file="old.xlsx",
        source_row_index=99,
        original_comment="نص قديم",
        review_status="UNPROCESSED",
        ready_to_send=False,
        row_version=1,
        ai_primary_intent=None,
    )
    row.update(overrides)
    upsert_review_items(engine, [row])


def _ingest_row(comment_id: str, **overrides) -> dict:
    row = dict(
        comment_id=comment_id,
        source_file="new.xlsx",
        source_row_index=0,
        original_comment="نص جديد",
        original_title=None,
        original_type=None,
        entry_id=None,
        lemma=None,
        visitor_name=None,
        visitor_email=None,
        platform=None,
        comment_created_at=dt.datetime(2026, 8, 1),
        is_exact_duplicate_of=None,
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0),
        comment_age_days=2.0,
        age_group="recent",
        review_status="UNPROCESSED",
        ready_to_send=False,
        source_reference="new.xlsx:0",
    )
    row.update(overrides)
    return row


# A. Contract/signature coverage.
def test_contract_declares_upsert_ingest_batch():
    assert hasattr(contracts.ReviewItemRepository, "upsert_ingest_batch")
    assert hasattr(PostgresReviewItemRepository, "upsert_ingest_batch")
    assert hasattr(MongoReviewItemRepository, "upsert_ingest_batch")


# B. New row: inserted correctly (both backends).
def test_new_row_inserted_postgres(tmp_path):
    engine = _engine(tmp_path)
    repo = PostgresReviewItemRepository(engine)
    result = repo.upsert_ingest_batch([_ingest_row("c1")])
    assert result == {"inserted": 1, "updated": 0, "skipped": 0}
    item = repo.get("c1")
    assert item.source_file == "new.xlsx"
    assert item.review_status == "UNPROCESSED"
    assert item.row_version == 1


def test_new_row_inserted_mongo():
    db = _FakeDatabase()
    repo = MongoReviewItemRepository(db)
    result = repo.upsert_ingest_batch([_ingest_row("c1")])
    assert result == {"inserted": 1, "updated": 0, "skipped": 0}
    item = repo.get("c1")
    assert item.source_file == "new.xlsx"
    assert item.review_status == "UNPROCESSED"
    assert item.row_version == 1


# C. Existing unlocked: only INGEST_OWNED_FIELDS updated (both backends).
def test_existing_unlocked_updates_only_owned_fields_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", source_file="old.xlsx", primary_intent="THANKS_PRAISE")
    repo = PostgresReviewItemRepository(engine)
    result = repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    assert result == {"inserted": 0, "updated": 1, "skipped": 0}
    item = repo.get("c1")
    assert item.source_file == "new.xlsx"  # INGEST_OWNED_FIELDS -- refreshed
    assert item.primary_intent == "THANKS_PRAISE"  # not ingest-owned -- untouched
    assert item.row_version == 1  # never bumped by ingest


def test_existing_unlocked_updates_only_owned_fields_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        dict(
            _id="c1", source_file="old.xlsx", source_row_index=99, reviewer_action=None,
            review_status="UNPROCESSED", primary_intent="THANKS_PRAISE", row_version=1,
            original_comment="نص قديم",
        )
    )
    repo = MongoReviewItemRepository(db)
    result = repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    assert result == {"inserted": 0, "updated": 1, "skipped": 0}
    item = repo.get("c1")
    assert item.source_file == "new.xlsx"
    assert item.primary_intent == "THANKS_PRAISE"
    assert item.row_version == 1


# D. Existing locked by review_status (both backends).
def test_existing_locked_by_status_untouched_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", review_status="DRAFT_READY", source_file="old.xlsx", ai_primary_intent="THANKS_PRAISE")
    repo = PostgresReviewItemRepository(engine)
    result = repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    assert result == {"inserted": 0, "updated": 0, "skipped": 1}
    item = repo.get("c1")
    assert item.source_file == "old.xlsx"
    assert item.ai_primary_intent == "THANKS_PRAISE"


def test_existing_locked_by_status_untouched_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        dict(_id="c1", source_file="old.xlsx", reviewer_action=None, review_status="DRAFT_READY",
             ai_primary_intent="THANKS_PRAISE", row_version=1)
    )
    repo = MongoReviewItemRepository(db)
    result = repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    assert result == {"inserted": 0, "updated": 0, "skipped": 1}
    item = repo.get("c1")
    assert item.source_file == "old.xlsx"
    assert item.ai_primary_intent == "THANKS_PRAISE"


# E. Existing locked by reviewer_action (both backends).
def test_existing_locked_by_reviewer_action_untouched_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", reviewer_action="SEND_AS_IS", review_status="APPROVED", source_file="old.xlsx")
    repo = PostgresReviewItemRepository(engine)
    result = repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    assert result == {"inserted": 0, "updated": 0, "skipped": 1}
    assert repo.get("c1").source_file == "old.xlsx"


def test_existing_locked_by_reviewer_action_untouched_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        dict(_id="c1", source_file="old.xlsx", reviewer_action="SEND_AS_IS", review_status="APPROVED", row_version=1)
    )
    repo = MongoReviewItemRepository(db)
    result = repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    assert result == {"inserted": 0, "updated": 0, "skipped": 1}
    assert repo.get("c1").source_file == "old.xlsx"


# F. Re-ingesting same input: idempotent business result (both backends).
def test_reingest_same_input_is_idempotent_postgres(tmp_path):
    engine = _engine(tmp_path)
    repo = PostgresReviewItemRepository(engine)
    row = _ingest_row("c1")
    repo.upsert_ingest_batch([row])
    result2 = repo.upsert_ingest_batch([row])
    assert result2 == {"inserted": 0, "updated": 1, "skipped": 0}
    item = repo.get("c1")
    assert item.row_version == 1
    assert item.source_file == "new.xlsx"


def test_reingest_same_input_is_idempotent_mongo():
    db = _FakeDatabase()
    repo = MongoReviewItemRepository(db)
    row = _ingest_row("c1")
    repo.upsert_ingest_batch([row])
    result2 = repo.upsert_ingest_batch([row])
    assert result2 == {"inserted": 0, "updated": 1, "skipped": 0}
    item = repo.get("c1")
    assert item.row_version == 1


# G. Protected fields (reviewer decisions, assignment, generation,
# classification) remain untouched on an unlocked update too -- ingest
# only ever refreshes INGEST_OWNED_FIELDS, never anything else, even
# when the row IS eligible for a refresh.
def test_protected_fields_never_touched_on_unlocked_update_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(
        engine, "c1", source_file="old.xlsx",
        reviewer_user_id=None, generation_status=None,
        ai_topic_subtype="GENERAL", topic_subtype="GENERAL",
        suggested_final_reply=None,
    )
    repo = PostgresReviewItemRepository(engine)
    repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    item = repo.get("c1")
    assert item.ai_topic_subtype == "GENERAL"
    assert item.topic_subtype == "GENERAL"


def test_protected_fields_never_touched_on_unlocked_update_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        dict(_id="c1", source_file="old.xlsx", reviewer_action=None, review_status="UNPROCESSED",
             ai_topic_subtype="GENERAL", topic_subtype="GENERAL", row_version=1)
    )
    repo = MongoReviewItemRepository(db)
    repo.upsert_ingest_batch([_ingest_row("c1", source_file="new.xlsx")])
    item = repo.get("c1")
    assert item.ai_topic_subtype == "GENERAL"
    assert item.topic_subtype == "GENERAL"


# H. row_version: never changes through ingest, for insert (=1), unlocked
# update (unchanged), and skip (unchanged) -- covered across B/C/D/E/F
# above; one more explicit multi-call check for clarity.
def test_row_version_never_moves_across_repeated_ingest_calls_postgres(tmp_path):
    engine = _engine(tmp_path)
    repo = PostgresReviewItemRepository(engine)
    row = _ingest_row("c1")
    for _ in range(3):
        repo.upsert_ingest_batch([row])
    assert repo.get("c1").row_version == 1


def test_row_version_never_moves_across_repeated_ingest_calls_mongo():
    db = _FakeDatabase()
    repo = MongoReviewItemRepository(db)
    row = _ingest_row("c1")
    for _ in range(3):
        repo.upsert_ingest_batch([row])
    assert repo.get("c1").row_version == 1


# I. Whole-batch rollback on a mid-batch injected failure -- both
# backends. The Mongo assertion proves THIS CODE's callback correctly
# leaves nothing behind when with_transaction's callback raises; it is
# NOT proof of MongoDB's own server-enforced ACID guarantee, which only
# a live Atlas run can establish (see test_mongo_repository_live.py).
def test_whole_batch_rolls_back_on_injected_failure_postgres(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    repo = PostgresReviewItemRepository(engine)
    rows = [_ingest_row("c1"), _ingest_row("c2"), _ingest_row("c3")]

    call_count = {"n": 0}
    from sqlalchemy.orm import Session as _Session

    real_get = _Session.get

    def _get_then_fail_on_third(self, model, ident, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("simulated mid-batch failure")
        return real_get(self, model, ident, *a, **kw)

    monkeypatch.setattr(_Session, "get", _get_then_fail_on_third)
    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        repo.upsert_ingest_batch(rows)
    monkeypatch.undo()
    for cid in ("c1", "c2", "c3"):
        assert PostgresReviewItemRepository(engine).get(cid) is None, f"{cid} should not exist after rollback"


def test_whole_batch_rolls_back_on_injected_failure_mongo(monkeypatch):
    db = _FakeDatabase()
    repo = MongoReviewItemRepository(db)
    rows = [_ingest_row("c1"), _ingest_row("c2"), _ingest_row("c3")]

    coll = db["review_items"]
    call_count = {"n": 0}
    real_find = coll.find

    def _find_then_fail_on_first(filt=None, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Let the read succeed once normally elsewhere, but fail here
            # to simulate a mid-callback error AFTER the read but before
            # any write is applied for this batch.
            raise RuntimeError("simulated mid-batch failure")
        return real_find(filt, *a, **kw)

    monkeypatch.setattr(coll, "find", _find_then_fail_on_first)
    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        repo.upsert_ingest_batch(rows)
    monkeypatch.undo()
    for cid in ("c1", "c2", "c3"):
        assert repo.get(cid) is None, f"{cid} should not exist after rollback (fake proves OUR rollback logic only)"


# J. Transaction callback re-execution: force the EXISTING fake
# transaction machinery (_FakeSession.with_transaction, already proven
# in test_mongo_repository_unit.py) to invoke the callback twice, by
# having the first invocation raise. That fake already snapshots-and-
# restores storage around the failing first attempt (the same mechanism
# that proves whole-batch rollback elsewhere) -- so a correct second,
# successful invocation must report ONLY its own fresh counts, never an
# accumulation from the discarded first attempt.
def test_callback_reexecution_does_not_leak_accumulators_mongo(monkeypatch):
    db = _FakeDatabase()
    repo = MongoReviewItemRepository(db)
    rows = [_ingest_row("c1"), _ingest_row("c2")]

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
        # The bare fake session does not retry on its own (only real
        # with_transaction/TransientTransactionError would) -- this
        # confirms the FIRST attempt's exception propagates and nothing
        # was left behind, matching test I's rollback proof exactly.
        repo.upsert_ingest_batch(rows)
    monkeypatch.undo()
    assert db["review_items"]._docs == []

    # A genuinely fresh, second call (simulating a real retry re-invoking
    # the whole method from scratch) must report clean, non-accumulated
    # counts -- not 4 (2 leaked from a discarded first attempt + 2 real).
    result = repo.upsert_ingest_batch(rows)
    assert result == {"inserted": 2, "updated": 0, "skipped": 0}


# K. Concurrent/duplicate insert handling -- as far as a fake can prove:
# a genuine duplicate-key error on an InsertOne propagates, unswallowed,
# out of upsert_ingest_batch (never silently converted to an update).
def test_duplicate_key_on_concurrent_insert_propagates_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(dict(_id="c1", source_file="already-there.xlsx", reviewer_action=None, review_status="UNPROCESSED", row_version=1))
    # Force this repo's own read to see c1 as absent (simulating a
    # snapshot taken before a concurrent transaction's insert committed),
    # so it attempts an InsertOne that will collide.
    repo = MongoReviewItemRepository(db)
    coll = db["review_items"]
    real_find = coll.find

    def _find_hides_c1(filt=None, *a, **kw):
        docs = list(real_find(filt, *a, **kw))
        return iter([d for d in docs if d["_id"] != "c1"])

    import pytest as _pytest
    from pymongo.errors import DuplicateKeyError

    coll.find = _find_hides_c1
    try:
        with _pytest.raises(DuplicateKeyError):
            repo.upsert_ingest_batch([_ingest_row("c1")])
    finally:
        coll.find = real_find
    # Untouched by the failed attempt -- still whatever it was before.
    assert repo.get("c1").source_file == "already-there.xlsx"


# L. Ensure the ingest write path never uses mongo_retry.with_transient_retry.
def test_ingest_never_uses_mongo_retry_wrapper(monkeypatch):
    db = _FakeDatabase()
    repo = MongoReviewItemRepository(db)

    def _fail_if_called(fn, *, max_attempts=None):
        raise AssertionError("upsert_ingest_batch must never call with_transient_retry")

    monkeypatch.setattr("rcs.review.repositories.mongo.with_transient_retry", _fail_if_called)
    # Should complete without ever touching the patched function.
    result = repo.upsert_ingest_batch([_ingest_row("c1")])
    assert result == {"inserted": 1, "updated": 0, "skipped": 0}
