"""
Phase M4C.4 — generation-result persistence repository contract + parity
tests.

Postgres and Mongo scenarios are placed side by side per behavior, same
convention as test_ingest_repository.py / test_prefilter_repository.py /
test_classification_repository.py. The Mongo tests run against the same
in-memory fake used in test_mongo_repository_unit.py -- they prove THIS
CODE's logic, not MongoDB's own server-enforced atomic-update guarantee.
See test_mongo_repository_live.py for what remains BLOCKED pending a
working Atlas connection.

Persistence extraction only -- no response-generation (Gemini) logic,
frozen classification, or routing/taxonomy business rule is touched or
exercised here (results are hand-constructed GenerationResult objects,
never a real Gemini call). `compute_priority`/`next_status` are called
via their real, unmodified implementations (business logic this phase
must reproduce, not own).

Important audited finding (see the M4C.4 report): unlike claim/
send_as_is/edit_and_send/mark_no_response_needed, the current
apply_generation_result has never taken an `expected_version` parameter
-- there is no caller-supplied optimistic-concurrency check for this
method in either backend. It still unconditionally bumps row_version by
1 on every non-skipped write (proven below), but "stale row_version" /
"version conflict" are not reachable inputs for this method as it
exists today, so no such test is asserted here -- inventing one would
test behavior the code does not have.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.response_generation.schema import GenerationResult
from rcs.review.models import ReviewItem
from rcs.review.repositories import contracts
from rcs.review.repositories.mongo import MongoReviewItemRepository
from rcs.review.repositories.postgres import PostgresReviewItemRepository
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.workflow import InvalidTransition, ItemNotFound

from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake

_ALL_COLUMNS = tuple(c.name for c in ReviewItem.__table__.columns)


def _engine(tmp_path, name="generation.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _seed_existing(engine, comment_id: str, **overrides) -> None:
    row = dict(
        comment_id=comment_id,
        source_file="f.xlsx",
        source_row_index=0,
        original_comment="نص تجريبي",
        review_status="CLASSIFIED",
        is_prefiltered_junk=False,
        ready_to_send=False,
        reviewer_action=None,
        row_version=1,
        ai_primary_intent="THANKS_PRAISE",
        primary_intent=None,
        topic_subtype=None,
        sentiment=None,
        generation_routing_decision=None,
        generation_status=None,
        outbound_status=None,
        review_priority=None,
        priority_reason=None,
        updated_at=None,
        source_reference=None,
    )
    row.update(overrides)
    upsert_review_items(engine, [row])


def _mongo_doc(comment_id: str, **overrides) -> dict:
    """Every declared ReviewItem column, defaulted to None, so this
    fixture can never be missing a field a real ingested document would
    have (the exact SimpleNamespace-vs-incomplete-fixture gap caught and
    fixed while writing test_classification_repository.py)."""
    doc = {name: None for name in _ALL_COLUMNS if name != "comment_id"}
    doc.update(
        _id=comment_id,
        source_file="f.xlsx",
        source_row_index=0,
        original_comment="نص تجريبي",
        review_status="CLASSIFIED",
        is_prefiltered_junk=False,
        ready_to_send=False,
        row_version=1,
        ai_primary_intent="THANKS_PRAISE",
    )
    doc.update(overrides)
    return doc


def _gen_result(comment_id: str, **overrides) -> GenerationResult:
    defaults = dict(
        comment_id=comment_id,
        routing_decision="GENERATE_DRAFT",
        generation_status="SUCCEEDED",
        suggested_response="شكراً لتواصلكم معنا",
        review_required=False,
        reason_code="ok",
        provider="gemini",
        model_name="gemini-1.5-flash",
        prompt_version="gen-v1",
        generated_at="2026-01-01T10:00:00Z",
    )
    defaults.update(overrides)
    return GenerationResult(**defaults)


# A. Contract/signature ------------------------------------------------


def test_contract_declares_apply_generation_result():
    assert hasattr(contracts.ReviewItemRepository, "apply_generation_result")


def test_both_backends_implement_apply_generation_result():
    assert hasattr(PostgresReviewItemRepository, "apply_generation_result")
    assert hasattr(MongoReviewItemRepository, "apply_generation_result")


# B. Successful GENERATE_DRAFT result -----------------------------------


def test_success_generate_draft_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c1")
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(_gen_result("c1"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "DRAFT_READY"
    assert item.generation_routing_decision == "GENERATE_DRAFT"
    assert item.generation_status == "SUCCEEDED"
    assert item.generation_reason_code == "ok"
    assert item.generation_provider == "gemini"
    assert item.generation_model_name == "gemini-1.5-flash"
    assert item.generation_prompt_version == "gen-v1"
    assert item.suggested_final_reply == "شكراً لتواصلكم معنا"
    assert item.ai_suggested_final_reply == "شكراً لتواصلكم معنا"
    assert item.generated_at == dt.datetime(2026, 1, 1, 10, 0, 0)
    assert item.row_version == 2
    assert item.outbound_status is None


def test_success_generate_draft_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c1"))
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(_gen_result("c1"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "DRAFT_READY"
    assert item.generation_routing_decision == "GENERATE_DRAFT"
    assert item.suggested_final_reply == "شكراً لتواصلكم معنا"
    assert item.ai_suggested_final_reply == "شكراً لتواصلكم معنا"
    # Not asserted as an exact value here: `generated_at` is a "riyadh"-
    # convention column (see _DATETIME_CONVENTION), and the in-memory
    # fake stores Python objects verbatim rather than round-tripping
    # through real BSON's aware-datetime-to-UTC encoding the way a real
    # Atlas connection would -- same reason the pre-existing claim/
    # send_as_is unit tests never assert an exact reviewer_started_at/
    # updated_at value either. See test_mongo_repository_live.py for the
    # real-server BSON parity proof (currently BLOCKED).
    assert item.generated_at is not None
    assert item.row_version == 2


# C. NO_RESPONSE_RECOMMENDED result -> outbound_status NOT_APPLICABLE --


def test_no_response_recommended_sets_outbound_not_applicable_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c2")
    repo = PostgresReviewItemRepository(engine)
    result = _gen_result("c2", routing_decision="NO_RESPONSE_RECOMMENDED", generation_status="SKIPPED", suggested_response=None)
    item = repo.apply_generation_result(result, intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "NO_RESPONSE_RECOMMENDED"
    assert item.outbound_status == "NOT_APPLICABLE"
    assert item.suggested_final_reply is None


def test_no_response_recommended_sets_outbound_not_applicable_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c2"))
    repo = MongoReviewItemRepository(db)
    result = _gen_result("c2", routing_decision="NO_RESPONSE_RECOMMENDED", generation_status="SKIPPED", suggested_response=None)
    item = repo.apply_generation_result(result, intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "NO_RESPONSE_RECOMMENDED"
    assert item.outbound_status == "NOT_APPLICABLE"


# D. REVIEW_ONLY / failure fallback result -------------------------------


def test_review_only_result_leaves_outbound_status_untouched_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c3")
    repo = PostgresReviewItemRepository(engine)
    result = _gen_result("c3", routing_decision="REVIEW_ONLY", generation_status="FAILED_REVIEW_ONLY", suggested_response=None)
    item = repo.apply_generation_result(result, intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "REVIEW_ONLY"
    assert item.outbound_status is None


def test_review_only_result_leaves_outbound_status_untouched_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c3"))
    repo = MongoReviewItemRepository(db)
    result = _gen_result("c3", routing_decision="REVIEW_ONLY", generation_status="FAILED_REVIEW_ONLY", suggested_response=None)
    item = repo.apply_generation_result(result, intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "REVIEW_ONLY"
    assert item.outbound_status is None


# E. Eligible from UNPROCESSED (not just CLASSIFIED) ---------------------


def test_unprocessed_item_is_also_eligible_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c4", review_status="UNPROCESSED", ai_primary_intent=None)
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(_gen_result("c4"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "DRAFT_READY"


def test_unprocessed_item_is_also_eligible_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c4", review_status="UNPROCESSED", ai_primary_intent=None))
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(_gen_result("c4"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "DRAFT_READY"


# F. Skip guard A: reviewer already acted / already sendable -------------


@pytest.mark.parametrize("field,value", [("reviewer_action", "SEND_AS_IS"), ("ready_to_send", True)])
def test_skip_when_reviewer_already_acted_postgres(tmp_path, field, value):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c5", review_status="APPROVED", **{field: value})
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(_gen_result("c5"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.row_version == 1
    assert item.generation_routing_decision is None


@pytest.mark.parametrize("field,value", [("reviewer_action", "SEND_AS_IS"), ("ready_to_send", True)])
def test_skip_when_reviewer_already_acted_mongo(field, value):
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c5", review_status="APPROVED", **{field: value}))
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(_gen_result("c5"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.row_version == 1
    assert item.generation_routing_decision is None


# G. Skip guard B: generation already landed on an advanced status -------


def test_skip_when_generation_already_landed_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c6", review_status="SENT", generation_routing_decision="GENERATE_DRAFT")
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(
        _gen_result("c6", suggested_response="نص مختلف يجب ألا يُكتب"),
        intent=None, topic=None, sentiment=None, confidence=None, needs_review=None,
    )
    assert item.row_version == 1
    assert item.suggested_final_reply is None


def test_skip_when_generation_already_landed_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        _mongo_doc("c6", review_status="SENT", generation_routing_decision="GENERATE_DRAFT")
    )
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(
        _gen_result("c6", suggested_response="نص مختلف يجب ألا يُكتب"),
        intent=None, topic=None, sentiment=None, confidence=None, needs_review=None,
    )
    assert item.row_version == 1
    assert item.suggested_final_reply is None


# H. Missing item ---------------------------------------------------------


def test_missing_item_postgres(tmp_path):
    engine = _engine(tmp_path)
    repo = PostgresReviewItemRepository(engine)
    with pytest.raises(ItemNotFound):
        repo.apply_generation_result(_gen_result("ghost"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)


def test_missing_item_mongo():
    db = _FakeDatabase()
    repo = MongoReviewItemRepository(db)
    with pytest.raises(ItemNotFound):
        repo.apply_generation_result(_gen_result("ghost"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)


# I. Reapply-after-success is a true no-op (skip guard B kicks in) -------


def test_reapply_after_success_is_noop_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c7")
    repo = PostgresReviewItemRepository(engine)
    first = repo.apply_generation_result(_gen_result("c7"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert first.row_version == 2
    second = repo.apply_generation_result(
        _gen_result("c7", suggested_response="نص مختلف"),
        intent=None, topic=None, sentiment=None, confidence=None, needs_review=None,
    )
    assert second.row_version == 2
    assert second.suggested_final_reply == "شكراً لتواصلكم معنا"


def test_reapply_after_success_is_noop_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c7"))
    repo = MongoReviewItemRepository(db)
    first = repo.apply_generation_result(_gen_result("c7"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert first.row_version == 2
    second = repo.apply_generation_result(
        _gen_result("c7", suggested_response="نص مختلف"),
        intent=None, topic=None, sentiment=None, confidence=None, needs_review=None,
    )
    assert second.row_version == 2
    assert second.suggested_final_reply == "شكراً لتواصلكم معنا"


# J. Protected fields are never touched -----------------------------------


def test_protected_reviewer_and_assignment_fields_untouched_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(
        engine, "c8",
        reviewer_id="u1", reviewer_display_name="Sara", reviewer_user_id="u1",
        final_reply=None, reviewed_at=None,
    )
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(_gen_result("c8"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.reviewer_id == "u1"
    assert item.reviewer_display_name == "Sara"
    assert item.reviewer_user_id == "u1"
    assert item.final_reply is None
    assert item.reviewed_at is None


def test_protected_reviewer_and_assignment_fields_untouched_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        _mongo_doc("c8", reviewer_id="u1", reviewer_display_name="Sara", reviewer_user_id="u1")
    )
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(_gen_result("c8"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.reviewer_id == "u1"
    assert item.reviewer_display_name == "Sara"
    assert item.reviewer_user_id == "u1"


# K. Coalesce vs. overwrite semantics --------------------------------------


def test_intent_topic_coalesce_only_if_unset_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c9", primary_intent="COMPLAINT", topic_subtype="TECHNICAL_ISSUE")
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(
        _gen_result("c9"), intent="THANKS_PRAISE", topic="GENERAL", sentiment=None, confidence=None, needs_review=None
    )
    assert item.primary_intent == "COMPLAINT"
    assert item.topic_subtype == "TECHNICAL_ISSUE"


def test_intent_topic_coalesce_only_if_unset_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        _mongo_doc("c9", primary_intent="COMPLAINT", topic_subtype="TECHNICAL_ISSUE")
    )
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(
        _gen_result("c9"), intent="THANKS_PRAISE", topic="GENERAL", sentiment=None, confidence=None, needs_review=None
    )
    assert item.primary_intent == "COMPLAINT"
    assert item.topic_subtype == "TECHNICAL_ISSUE"


def test_sentiment_confidence_needs_review_always_overwrite_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c10", sentiment="NEGATIVE", ai_confidence=0.2, needs_review=True)
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(
        _gen_result("c10"), intent=None, topic=None, sentiment="POSITIVE", confidence=0.95, needs_review=False
    )
    assert item.sentiment == "POSITIVE"
    assert item.ai_confidence == 0.95
    assert item.needs_review is False


def test_sentiment_confidence_needs_review_always_overwrite_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(
        _mongo_doc("c10", sentiment="NEGATIVE", ai_confidence=0.2, needs_review=True)
    )
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(
        _gen_result("c10"), intent=None, topic=None, sentiment="POSITIVE", confidence=0.95, needs_review=False
    )
    assert item.sentiment == "POSITIVE"
    assert item.ai_confidence == 0.95
    assert item.needs_review is False


# L. row_version bumps by exactly 1, never on skip -------------------------


def test_row_version_bump_is_exactly_one_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c11", row_version=7)
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(_gen_result("c11"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.row_version == 8


def test_row_version_bump_is_exactly_one_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c11", row_version=7))
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(_gen_result("c11"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.row_version == 8


# M. Invalid transition (current status has no matching edge) propagates -


def test_invalid_transition_propagates_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c12", review_status="SEND_FAILED")
    repo = PostgresReviewItemRepository(engine)
    with pytest.raises(InvalidTransition):
        repo.apply_generation_result(_gen_result("c12"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)


def test_invalid_transition_propagates_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c12", review_status="SEND_FAILED"))
    repo = MongoReviewItemRepository(db)
    with pytest.raises(InvalidTransition):
        repo.apply_generation_result(_gen_result("c12"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)


# N. mongo_retry is never used for this mutation ---------------------------


def test_mongo_retry_never_used_for_apply_generation_result(monkeypatch):
    import rcs.review.repositories.mongo as mongo_module

    def _boom(*_a, **_kw):
        raise AssertionError("with_transient_retry must not be used by apply_generation_result")

    monkeypatch.setattr(mongo_module, "with_transient_retry", _boom)
    db = _FakeDatabase()
    db["review_items"].insert_one(_mongo_doc("c13"))
    repo = MongoReviewItemRepository(db)
    item = repo.apply_generation_result(_gen_result("c13"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "DRAFT_READY"


# O. updated_at moves on a real write, is untouched on skip ----------------


def test_updated_at_moves_on_write_postgres(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c14")
    import rcs.review.repositories.postgres as pg_module

    fixed = dt.datetime(2026, 5, 5, 12, 0, 0)
    monkeypatch.setattr(pg_module, "_now", lambda: fixed)
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(_gen_result("c14", generated_at=""), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.updated_at == fixed


def test_updated_at_untouched_on_skip_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_existing(engine, "c15", review_status="SENT", generation_routing_decision="GENERATE_DRAFT", updated_at=None)
    repo = PostgresReviewItemRepository(engine)
    item = repo.apply_generation_result(_gen_result("c15"), intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.updated_at is None


# P. Service orchestration wrapper still returns a DTO ----------------------


def test_service_wrapper_delegates_and_returns_detail(tmp_path):
    from rcs.review.service import ReviewWorkflowService

    engine = _engine(tmp_path)
    _seed_existing(engine, "c16")
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(_gen_result("c16"), intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE")
    assert detail.comment_id == "c16"
    assert detail.review_status == "DRAFT_READY"
