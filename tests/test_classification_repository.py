"""
Phase M4C.3 — frozen-classification repository contract + parity tests.

Postgres and Mongo scenarios are placed side by side per behavior, same
convention as test_ingest_repository.py / test_prefilter_repository.py.
The Mongo tests run against the same in-memory fake used in
test_mongo_repository_unit.py -- they prove THIS CODE's logic, not
MongoDB's own server-enforced transactional guarantee. See
test_mongo_repository_live.py for what remains BLOCKED pending a working
Atlas connection.

Persistence extraction only -- no frozen model, taxonomy, or routing
logic is touched or exercised here (results are hand-constructed dicts,
never a real classifier call).
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.review.repositories import contracts
from rcs.review.repositories.mongo import MongoReviewItemRepository
from rcs.review.repositories.postgres import PostgresReviewItemRepository
from rcs.review.repository import get_engine, init_db, upsert_review_items

from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake

_MODEL_KW = dict(
    model_name="tfidf_logistic_intent+marbertv2_topic+camelbert_sentiment",
    model_version=None,
    prompt_version="frozen-2026-08-25",
    pipeline_version="0.1.0",
    inference_run_id="run-1",
)


def _engine(tmp_path, name="classification.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _seed_existing(engine, comment_id: str, **overrides) -> None:
    row = dict(
        comment_id=comment_id,
        source_file="f.xlsx",
        source_row_index=0,
        original_comment="نص تجريبي",
        age_group="recent",
        review_status="UNPROCESSED",
        is_prefiltered_junk=False,
        ready_to_send=False,
        row_version=1,
        ai_primary_intent=None,
        model_name=None,
        reviewer_edit=None,
        suggested_final_reply=None,
        generation_status=None,
    )
    row.update(overrides)
    upsert_review_items(engine, [row])


def _mongo_doc(comment_id: str, **overrides) -> dict:
    """A hand-built fixture standing in for a real, fully-populated
    ingest document (every field explicitly present -- a real document,
    produced via upsert_ingest_batch, always has this shape; a fixture
    that omits a field would let a SimpleNamespace-vs-real-document gap
    hide behind an incomplete test double)."""
    doc = dict(
        _id=comment_id,
        original_comment="نص تجريبي",
        original_title=None,
        original_type=None,
        age_group="recent",
        platform=None,
        entry_id=None,
        lemma=None,
        review_status="UNPROCESSED",
        is_prefiltered_junk=False,
        reviewer_action=None,
        row_version=1,
        ai_primary_intent=None,
        ai_topic_subtype=None,
        ai_subcategory=None,
        ai_secondary_intent=None,
        ai_routing_decision=None,
        ai_classification_reason=None,
        ai_confidence=None,
        ai_classification_source=None,
        ai_needs_review=None,
        primary_intent=None,
        topic_subtype=None,
        subcategory=None,
        secondary_intent=None,
        classification_source=None,
        needs_review=None,
        routing_decision=None,
        exclusion_reason=None,
        model_name=None,
        model_version=None,
        prompt_version=None,
        pipeline_version=None,
        inference_run_id=None,
        processed_at=None,
        reviewer_edit=None,
        suggested_final_reply=None,
        generation_status=None,
    )
    doc.update(overrides)
    return doc


def _success_result(comment_id: str, **overrides) -> dict:
    result = dict(
        comment_id=comment_id,
        success=True,
        ai_primary_intent="THANKS_PRAISE",
        ai_topic_subtype="GENERAL",
        ai_subcategory=None,
        ai_secondary_intent=None,
        ai_routing_decision="GENERATE_DRAFT",
        ai_classification_reason="frozen_model_inference",
        ai_confidence=0.92,
        routing_reason_code=None,
    )
    result.update(overrides)
    return result


def _failure_result(comment_id: str, **overrides) -> dict:
    result = dict(comment_id=comment_id, success=False, ai_classification_reason="schema_validation_failed")
    result.update(overrides)
    return result


# A. Contract/signature.
def test_contract_declares_classification_methods():
    assert hasattr(contracts.ReviewItemRepository, "load_for_classification")
    assert hasattr(contracts.ReviewItemRepository, "apply_classification_batch")
    assert hasattr(PostgresReviewItemRepository, "load_for_classification")
    assert hasattr(PostgresReviewItemRepository, "apply_classification_batch")
    assert hasattr(MongoReviewItemRepository, "load_for_classification")
    assert hasattr(MongoReviewItemRepository, "apply_classification_batch")


# B. Eligibility query -- is_prefiltered_junk must be exactly False
# (never None/missing, never True); review_status must be exactly
# UNPROCESSED.
def test_eligibility_excludes_null_and_true_junk_and_advanced_status_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "eligible", is_prefiltered_junk=False, review_status="UNPROCESSED")
    _seed_existing(engine, "junk-null", is_prefiltered_junk=None, review_status="UNPROCESSED")
    _seed_existing(engine, "junk-true", is_prefiltered_junk=True, review_status="EXCLUDED")
    _seed_existing(engine, "advanced", is_prefiltered_junk=False, review_status="DRAFT_READY")
    df = PostgresReviewItemRepository(engine).load_for_classification()
    assert list(df["comment_id"]) == ["eligible"]


def test_eligibility_excludes_null_and_true_junk_and_advanced_status_mongo():
    db = _FakeDatabase()
    coll = db["review_items"]
    coll.insert_one(_mongo_doc("eligible", is_prefiltered_junk=False, review_status="UNPROCESSED"))
    coll.insert_one(_mongo_doc("junk-null", is_prefiltered_junk=None, review_status="UNPROCESSED"))
    coll.insert_one(_mongo_doc("junk-true", is_prefiltered_junk=True, review_status="EXCLUDED"))
    coll.insert_one(_mongo_doc("advanced", is_prefiltered_junk=False, review_status="DRAFT_READY"))
    df = MongoReviewItemRepository(db).load_for_classification()
    assert list(df["comment_id"]) == ["eligible"]


def test_load_for_classification_column_shape_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    df = MongoReviewItemRepository(db).load_for_classification()
    assert list(df.columns) == [
        "comment_id", "original_comment", "original_title", "original_type",
        "age_group", "platform", "entry_id", "lemma",
    ]


# C. Successful classification -- full field matrix.
def test_successful_classification_fields_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    assert result == {"classified": 1, "failed": 0, "skipped_already_classified": 0}
    item = repo.get("c1")
    assert item.ai_primary_intent == "THANKS_PRAISE"
    assert item.ai_topic_subtype == "GENERAL"
    assert item.ai_subcategory is None
    assert item.ai_secondary_intent is None
    assert item.ai_routing_decision == "GENERATE_DRAFT"
    assert item.ai_classification_reason == "frozen_model_inference"
    assert item.ai_confidence == 0.92
    assert item.primary_intent == "THANKS_PRAISE"
    assert item.topic_subtype == "GENERAL"
    assert item.routing_decision == "GENERATE_DRAFT"
    assert item.exclusion_reason is None
    assert item.review_status == "CLASSIFIED"
    assert item.model_name == _MODEL_KW["model_name"]
    assert item.prompt_version == _MODEL_KW["prompt_version"]
    assert item.pipeline_version == _MODEL_KW["pipeline_version"]
    assert item.inference_run_id == _MODEL_KW["inference_run_id"]
    assert item.processed_at is not None
    assert item.row_version == 1


def test_successful_classification_fields_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    assert result == {"classified": 1, "failed": 0, "skipped_already_classified": 0}
    item = repo.get("c1")
    assert item.ai_primary_intent == "THANKS_PRAISE"
    assert item.ai_topic_subtype == "GENERAL"
    assert item.ai_subcategory is None
    assert item.ai_secondary_intent is None
    assert item.ai_routing_decision == "GENERATE_DRAFT"
    assert item.ai_classification_reason == "frozen_model_inference"
    assert item.ai_confidence == 0.92
    assert item.primary_intent == "THANKS_PRAISE"
    assert item.topic_subtype == "GENERAL"
    assert item.routing_decision == "GENERATE_DRAFT"
    assert item.exclusion_reason is None
    assert item.review_status == "CLASSIFIED"
    assert item.model_name == _MODEL_KW["model_name"]
    assert item.inference_run_id == _MODEL_KW["inference_run_id"]
    assert item.processed_at is not None
    assert item.row_version == 1


# Success + EXCLUDE routing -> exclusion_reason set from routing_reason_code.
def test_successful_classification_exclude_routing_sets_exclusion_reason_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    repo.apply_classification_batch(
        [_success_result("c1", ai_routing_decision="EXCLUDE", routing_reason_code="not_meaningful")], **_MODEL_KW
    )
    assert repo.get("c1").exclusion_reason == "not_meaningful"


def test_successful_classification_exclude_routing_sets_exclusion_reason_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    repo.apply_classification_batch(
        [_success_result("c1", ai_routing_decision="EXCLUDE", routing_reason_code="not_meaningful")], **_MODEL_KW
    )
    assert repo.get("c1").exclusion_reason == "not_meaningful"


# D. Failure classification -- exact field matrix.
def test_failure_classification_fields_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_classification_batch([_failure_result("c1")], **_MODEL_KW)
    assert result == {"classified": 0, "failed": 1, "skipped_already_classified": 0}
    item = repo.get("c1")
    assert item.ai_primary_intent is None  # never set on failure
    assert item.ai_topic_subtype is None
    assert item.ai_routing_decision == "REVIEW_ONLY"
    assert item.ai_classification_reason == "schema_validation_failed"
    assert item.review_status == "UNPROCESSED"  # NOT advanced -- retry-eligible
    assert item.model_name == _MODEL_KW["model_name"]  # metadata IS written even on failure
    assert item.processed_at is not None
    assert item.row_version == 1


def test_failure_classification_fields_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_classification_batch([_failure_result("c1")], **_MODEL_KW)
    assert result == {"classified": 0, "failed": 1, "skipped_already_classified": 0}
    item = repo.get("c1")
    assert item.ai_primary_intent is None
    assert item.ai_topic_subtype is None
    assert item.ai_routing_decision == "REVIEW_ONLY"
    assert item.ai_classification_reason == "schema_validation_failed"
    assert item.review_status == "UNPROCESSED"
    assert item.model_name == _MODEL_KW["model_name"]
    assert item.processed_at is not None
    assert item.row_version == 1


# E. Already-classified row skipped entirely (no field touched, not even processed_at).
def test_already_classified_row_skipped_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", ai_primary_intent="THANKS_PRAISE", review_status="CLASSIFIED", model_name="old-model")
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_classification_batch([_success_result("c1", ai_primary_intent="COMPLAINT")], **_MODEL_KW)
    assert result == {"classified": 0, "failed": 0, "skipped_already_classified": 1}
    item = repo.get("c1")
    assert item.ai_primary_intent == "THANKS_PRAISE"  # untouched -- immutable
    assert item.model_name == "old-model"  # untouched


def test_already_classified_row_skipped_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        _mongo_doc("c1", ai_primary_intent="THANKS_PRAISE", review_status="CLASSIFIED", model_name="old-model")
    )
    repo = MongoReviewItemRepository(db)
    result = repo.apply_classification_batch([_success_result("c1", ai_primary_intent="COMPLAINT")], **_MODEL_KW)
    assert result == {"classified": 0, "failed": 0, "skipped_already_classified": 1}
    item = repo.get("c1")
    assert item.ai_primary_intent == "THANKS_PRAISE"
    assert item.model_name == "old-model"


# F. Advanced-status row skipped, even with ai_primary_intent still None
# (e.g. a human reviewer somehow acted before generation exists, or the
# row was independently excluded after this batch was queued).
def test_advanced_status_row_skipped_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", ai_primary_intent=None, review_status="EXCLUDED", model_name=None)
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    assert result == {"classified": 0, "failed": 0, "skipped_already_classified": 1}
    item = repo.get("c1")
    assert item.ai_primary_intent is None
    assert item.review_status == "EXCLUDED"
    assert item.model_name is None


def test_advanced_status_row_skipped_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1", ai_primary_intent=None, review_status="EXCLUDED", model_name=None))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    assert result == {"classified": 0, "failed": 0, "skipped_already_classified": 1}
    item = repo.get("c1")
    assert item.ai_primary_intent is None
    assert item.review_status == "EXCLUDED"
    assert item.model_name is None


# G. Mixed batch: write (success) + write (failure) + skip, all in one call.
def test_mixed_batch_write_and_skip_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")  # eligible -> success
    _seed_existing(engine, "c2")  # eligible -> failure
    _seed_existing(engine, "c3", ai_primary_intent="COMPLAINT", review_status="CLASSIFIED")  # already classified -> skip
    repo = PostgresReviewItemRepository(engine)
    result = repo.apply_classification_batch(
        [_success_result("c1"), _failure_result("c2"), _success_result("c3")], **_MODEL_KW
    )
    assert result == {"classified": 1, "failed": 1, "skipped_already_classified": 1}


def test_mixed_batch_write_and_skip_mongo():
    db = _FakeDatabase()
    coll = db["review_items"]
    coll.insert_one(_mongo_doc("c1"))
    coll.insert_one(_mongo_doc("c2"))
    coll.insert_one(_mongo_doc("c3", ai_primary_intent="COMPLAINT", review_status="CLASSIFIED"))
    repo = MongoReviewItemRepository(db)
    result = repo.apply_classification_batch(
        [_success_result("c1"), _failure_result("c2"), _success_result("c3")], **_MODEL_KW
    )
    assert result == {"classified": 1, "failed": 1, "skipped_already_classified": 1}


# H. Missing-item behavior: raises, whole call aborts, nothing persisted.
def test_missing_item_raises_and_aborts_whole_batch_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    with pytest.raises(KeyError):
        repo.apply_classification_batch([_success_result("c1"), _success_result("missing")], **_MODEL_KW)
    item = repo.get("c1")
    assert item.ai_primary_intent is None
    assert item.review_status == "UNPROCESSED"


def test_missing_item_raises_and_aborts_whole_batch_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    with pytest.raises(KeyError):
        repo.apply_classification_batch([_success_result("c1"), _success_result("missing")], **_MODEL_KW)
    item = repo.get("c1")
    assert item.ai_primary_intent is None
    assert item.review_status == "UNPROCESSED"


# I. Same-batch rerun: NOT a uniform no-op -- depends on whether the
# prior attempt succeeded (immutability guard skips -> true no-op, zero
# field changes including processed_at) or failed (review_status stays
# UNPROCESSED, retry-eligible -> re-applies, processed_at DOES move).
def test_rerun_after_success_is_a_true_noop_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    ts1 = repo.get("c1").processed_at
    repo.apply_classification_batch([_success_result("c1")], model_name="different-model", model_version=None, prompt_version="v2", pipeline_version="p2", inference_run_id="run-2")
    ts2 = repo.get("c1").processed_at
    assert ts1 == ts2  # untouched -- true no-op
    assert repo.get("c1").model_name == _MODEL_KW["model_name"]  # NOT overwritten with "different-model"


def test_rerun_after_success_is_a_true_noop_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    ts1 = repo.get("c1").processed_at
    repo.apply_classification_batch([_success_result("c1")], model_name="different-model", model_version=None, prompt_version="v2", pipeline_version="p2", inference_run_id="run-2")
    ts2 = repo.get("c1").processed_at
    assert ts1 == ts2
    assert repo.get("c1").model_name == _MODEL_KW["model_name"]


def test_rerun_after_failure_reprocesses_and_moves_timestamp_postgres(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)

    monkeypatch.setattr("rcs.review.repositories.postgres.dt", _FixedDt(dt.datetime(2026, 1, 1)))
    r1 = repo.apply_classification_batch([_failure_result("c1")], **_MODEL_KW)
    ts1 = repo.get("c1").processed_at

    monkeypatch.setattr("rcs.review.repositories.postgres.dt", _FixedDt(dt.datetime(2026, 1, 2)))
    r2 = repo.apply_classification_batch([_failure_result("c1")], **_MODEL_KW)
    ts2 = repo.get("c1").processed_at

    assert r1 == r2 == {"classified": 0, "failed": 1, "skipped_already_classified": 0}  # NOT skipped
    assert ts1 != ts2  # timestamp DOES move -- not a no-op
    assert repo.get("c1").row_version == 1


def test_rerun_after_failure_reprocesses_and_moves_timestamp_mongo(monkeypatch):
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)

    monkeypatch.setattr("rcs.review.repositories.mongo.dt", _FixedDt(dt.datetime(2026, 1, 1)))
    r1 = repo.apply_classification_batch([_failure_result("c1")], **_MODEL_KW)
    ts1 = repo.get("c1").processed_at

    monkeypatch.setattr("rcs.review.repositories.mongo.dt", _FixedDt(dt.datetime(2026, 1, 2)))
    r2 = repo.apply_classification_batch([_failure_result("c1")], **_MODEL_KW)
    ts2 = repo.get("c1").processed_at

    assert r1 == r2 == {"classified": 0, "failed": 1, "skipped_already_classified": 0}
    assert ts1 != ts2
    assert repo.get("c1").row_version == 1


class _FixedDt:
    """Minimal stand-in for the `dt` module so `dt.datetime.utcnow()`
    returns a fixed, caller-chosen value without touching the real
    clock module used elsewhere."""

    def __init__(self, fixed: dt.datetime):
        class _FixedDatetime:
            @staticmethod
            def utcnow():
                return fixed

        self.datetime = _FixedDatetime


# J. processed_at semantics: real datetime, shared across every row in one call.
def test_processed_at_shared_across_batch_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    _seed_existing(engine, "c2")
    repo = PostgresReviewItemRepository(engine)
    repo.apply_classification_batch([_success_result("c1"), _failure_result("c2")], **_MODEL_KW)
    ts1 = repo.get("c1").processed_at
    ts2 = repo.get("c2").processed_at
    assert isinstance(ts1, dt.datetime)
    assert ts1 == ts2


def test_processed_at_shared_across_batch_mongo():
    db = _FakeDatabase()
    coll = db["review_items"]
    coll.insert_one(_mongo_doc("c1"))
    coll.insert_one(_mongo_doc("c2"))
    repo = MongoReviewItemRepository(db)
    repo.apply_classification_batch([_success_result("c1"), _failure_result("c2")], **_MODEL_KW)
    ts1 = repo.get("c1").processed_at
    ts2 = repo.get("c2").processed_at
    assert isinstance(ts1, dt.datetime)
    assert ts1 == ts2


# K. model metadata parity -- written identically on both backends, both paths.
def test_model_metadata_parity_on_both_paths_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    _seed_existing(engine, "c2")
    repo = PostgresReviewItemRepository(engine)
    repo.apply_classification_batch([_success_result("c1"), _failure_result("c2")], **_MODEL_KW)
    for cid in ("c1", "c2"):
        item = repo.get(cid)
        assert item.model_name == _MODEL_KW["model_name"]
        assert item.model_version == _MODEL_KW["model_version"]
        assert item.prompt_version == _MODEL_KW["prompt_version"]
        assert item.pipeline_version == _MODEL_KW["pipeline_version"]
        assert item.inference_run_id == _MODEL_KW["inference_run_id"]


def test_model_metadata_parity_on_both_paths_mongo():
    db = _FakeDatabase()
    coll = db["review_items"]
    coll.insert_one(_mongo_doc("c1"))
    coll.insert_one(_mongo_doc("c2"))
    repo = MongoReviewItemRepository(db)
    repo.apply_classification_batch([_success_result("c1"), _failure_result("c2")], **_MODEL_KW)
    for cid in ("c1", "c2"):
        item = repo.get(cid)
        assert item.model_name == _MODEL_KW["model_name"]
        assert item.model_version == _MODEL_KW["model_version"]
        assert item.prompt_version == _MODEL_KW["prompt_version"]
        assert item.pipeline_version == _MODEL_KW["pipeline_version"]
        assert item.inference_run_id == _MODEL_KW["inference_run_id"]


# L. row_version unchanged across insert/success/failure/rerun.
def test_row_version_never_bumped_by_classification_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", row_version=1)
    repo = PostgresReviewItemRepository(engine)
    repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    assert repo.get("c1").row_version == 1


def test_row_version_never_bumped_by_classification_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1", row_version=1))
    repo = MongoReviewItemRepository(db)
    repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    assert repo.get("c1").row_version == 1


# M. Protected reviewer/generation fields unchanged.
def test_protected_reviewer_and_generation_fields_untouched_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1", reviewer_edit="pre-existing edit", suggested_final_reply="pre-existing draft", generation_status="SUCCEEDED")
    repo = PostgresReviewItemRepository(engine)
    repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    item = repo.get("c1")
    assert item.reviewer_edit == "pre-existing edit"
    assert item.suggested_final_reply == "pre-existing draft"
    assert item.generation_status == "SUCCEEDED"


def test_protected_reviewer_and_generation_fields_untouched_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        _mongo_doc("c1", reviewer_edit="pre-existing edit", suggested_final_reply="pre-existing draft", generation_status="SUCCEEDED")
    )
    repo = MongoReviewItemRepository(db)
    repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    item = repo.get("c1")
    assert item.reviewer_edit == "pre-existing edit"
    assert item.suggested_final_reply == "pre-existing draft"
    assert item.generation_status == "SUCCEEDED"


# N. Whole-batch rollback on injected mid-batch failure.
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
        repo.apply_classification_batch(
            [_success_result("c1"), _success_result("c2"), _success_result("c3")], **_MODEL_KW
        )
    monkeypatch.undo()
    for cid in ("c1", "c2", "c3"):
        assert PostgresReviewItemRepository(engine).get(cid).ai_primary_intent is None


def test_injected_mid_batch_failure_rolls_back_mongo(monkeypatch):
    db = _FakeDatabase()
    for cid in ("c1", "c2", "c3"):
        db["review_items"].insert_one(_mongo_doc(cid))
    repo = MongoReviewItemRepository(db)

    coll = db["review_items"]

    def _bulk_write_fails(ops, ordered=True, **kw):
        raise RuntimeError("simulated mid-batch failure")

    monkeypatch.setattr(coll, "bulk_write", _bulk_write_fails)
    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        repo.apply_classification_batch(
            [_success_result("c1"), _success_result("c2"), _success_result("c3")], **_MODEL_KW
        )
    monkeypatch.undo()
    for cid in ("c1", "c2", "c3"):
        assert repo.get(cid).ai_primary_intent is None


# O. Transaction callback re-execution does not leak counters (Mongo).
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
        repo.apply_classification_batch([_success_result("c1"), _failure_result("c2")], **_MODEL_KW)
    monkeypatch.undo()
    assert repo.get("c1").ai_primary_intent is None  # untouched by the discarded first attempt

    final = repo.apply_classification_batch([_success_result("c1"), _failure_result("c2")], **_MODEL_KW)
    assert final == {"classified": 1, "failed": 1, "skipped_already_classified": 0}  # not doubled/leaked


# P. Ensure the classification WRITE path never uses mongo_retry.with_transient_retry.
def test_apply_classification_batch_never_uses_mongo_retry_wrapper(monkeypatch):
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)

    def _fail_if_called(fn, *, max_attempts=None):
        raise AssertionError("apply_classification_batch must never call with_transient_retry")

    monkeypatch.setattr("rcs.review.repositories.mongo.with_transient_retry", _fail_if_called)
    result = repo.apply_classification_batch([_success_result("c1")], **_MODEL_KW)
    assert result == {"classified": 1, "failed": 0, "skipped_already_classified": 0}
