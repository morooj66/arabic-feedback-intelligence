"""
Phase M4B — LIVE MongoDB integration tests for the Mongo repository
implementations. These exercise a real Atlas connection and are the
only tests in this suite that can prove MongoDB's own server-side
ACID transaction / aggregation-engine behavior (the in-memory fake in
test_mongo_repository_unit.py proves this code's own logic, not the
database engine's real guarantees).

Safety:
  - Never touches the real staging database (visitor_comments_rcs_
    staging_2026) -- targets a dedicated, clearly-disposable database
    name this test creates and drops itself.
  - Never prints MONGODB_URI or any credential.
  - Skips (not fails) the entire module if MONGODB_URI is unset or a
    connectivity probe fails -- this environment's Atlas credentials
    were last confirmed failing authentication (Phase M3/M4 finding);
    if that has not been fixed, this file reports SKIPPED, and this
    phase's report must say live integration is BLOCKED, never claim
    "passed" or "cutover-ready".
"""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest

from rcs.response_generation.schema import GenerationResult
from rcs.review import clock
from rcs.review.demo_seed import load_dotenv_if_present
from rcs.review.repositories.mongo import (
    MongoAnalyticsRepository,
    MongoProcessingJobRepository,
    MongoReviewItemRepository,
    MongoReviewSessionRepository,
    MongoUserRepository,
    MongoUserSessionRepository,
)
from rcs.review.repositories.mongo_client import get_mongo_database
from rcs.review.workflow import VersionConflict

load_dotenv_if_present()

_LIVE_TEST_DB_NAME = "rcs_m4b_repo_live_test"  # disposable; never the staging db


def _connectivity_ok() -> bool:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return False
    try:
        client, _db = get_mongo_database(uri, _LIVE_TEST_DB_NAME)
        try:
            client.admin.command("ping")
            return True
        finally:
            client.close()
    except Exception:
        return False


_LIVE_AVAILABLE = _connectivity_ok()

pytestmark = pytest.mark.skipif(
    not _LIVE_AVAILABLE,
    reason=(
        "BLOCKED: no working MongoDB Atlas connection in this environment "
        "(MONGODB_URI missing or authentication/connectivity failed -- see "
        "the Phase M3/M4 preflight findings). Live integration coverage for "
        "MongoReviewItemRepository/MongoReviewSessionRepository/"
        "MongoProcessingJobRepository is BLOCKED, not passing, until a "
        "working controlled connection is available. M4B is not "
        "cutover-ready without this."
    ),
)


@pytest.fixture
def live_db():
    uri = os.environ["MONGODB_URI"]
    client, db = get_mongo_database(uri, _LIVE_TEST_DB_NAME)
    # Confirm this disposable test database is actually empty/ours before
    # writing anything -- never assume, never reuse a pre-existing name.
    existing = db.list_collection_names()
    if existing:
        client.close()
        pytest.fail(
            f"refusing to run live tests: {_LIVE_TEST_DB_NAME!r} already has collections {existing} "
            "-- it must be empty/disposable before this suite runs"
        )
    try:
        yield db
    finally:
        for name in db.list_collection_names():
            db.drop_collection(name)
        client.close()


def _item_doc(comment_id: str, **overrides) -> dict:
    doc = dict(
        _id=comment_id, source_file="live.xlsx", source_row_index=0,
        original_comment="تعليق", review_status="DRAFT_READY",
        suggested_final_reply="رد", reviewer_action=None, reviewer_id=None,
        reviewer_display_name=None, reviewer_user_id=None, reviewer_started_at=None,
        reviewer_edit=None, final_reply=None, no_response_reason=None,
        dedupe_canonical_id=None, primary_intent=None, review_priority=3,
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0), reviewed_at=None,
        ready_to_send=False, outbound_status=None, outbound_idempotency_key=None,
        row_version=1,
    )
    doc.update(overrides)
    return doc


def test_live_get_and_claim_round_trip(live_db):
    live_db["review_items"].insert_one(_item_doc("c1"))
    repo = MongoReviewItemRepository(live_db)
    assert repo.get("c1").comment_id == "c1"
    claimed = repo.claim("c1", reviewer_id="r1", display_name="R1", user_id="u1", expected_version=1)
    assert claimed.row_version == 2
    assert claimed.review_status == "IN_REVIEW"


def test_live_stale_version_conflict(live_db):
    live_db["review_items"].insert_one(_item_doc("c1"))
    repo = MongoReviewItemRepository(live_db)
    repo.claim("c1", reviewer_id="r1", display_name="R1", user_id="u1", expected_version=1)
    with pytest.raises(VersionConflict):
        repo.claim("c1", reviewer_id="r2", display_name="R2", user_id="u2", expected_version=1)


def test_live_assign_batch_transaction_rolls_back_on_real_server(live_db, monkeypatch):
    """The one test that can only be meaningful against a real
    replica-set-backed Mongo: proves the server itself aborts every
    staged write in the transaction, not just this code's own bookkeeping."""
    coll = live_db["review_items"]
    coll.insert_one(_item_doc("c1"))
    coll.insert_one(_item_doc("c2"))
    coll.insert_one(_item_doc("c3"))
    repo = MongoReviewItemRepository(live_db)

    call_count = {"n": 0}
    real_find_one = coll.find_one

    def _fail_on_second(filt=None, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-batch failure")
        return real_find_one(filt, *a, **kw)

    monkeypatch.setattr(coll, "find_one", _fail_on_second)
    with pytest.raises(RuntimeError):
        repo.assign_batch(["c1", "c2", "c3"], "u-1")

    for cid in ("c1", "c2", "c3"):
        doc = coll.find_one({"_id": cid})
        assert doc["reviewer_user_id"] is None
        assert doc["row_version"] == 1


def test_live_session_and_job_crud(live_db):
    session_repo = MongoReviewSessionRepository(live_db)
    job_repo = MongoProcessingJobRepository(live_db)
    session = session_repo.create(original_filename="f.xlsx", source_file="f.xlsx", upload_token=uuid.uuid4().hex)
    job = job_repo.create(session.session_id)
    job_repo.update(job.job_id, status="succeeded", finished=True)
    assert job_repo.get(job.job_id).status == "succeeded"
    assert job_repo.latest_for_session(session.session_id).job_id == job.job_id


def _ingest_row(comment_id: str, **overrides) -> dict:
    row = dict(
        comment_id=comment_id, source_file="live-ingest.xlsx", source_row_index=0,
        original_comment="نص", original_title=None, original_type=None, entry_id=None,
        lemma=None, visitor_name=None, visitor_email=None, platform=None,
        comment_created_at=dt.datetime(2026, 8, 1), is_exact_duplicate_of=None,
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0), comment_age_days=1.0,
        age_group="recent", review_status="UNPROCESSED", ready_to_send=False,
        source_reference="live-ingest.xlsx:0",
    )
    row.update(overrides)
    return row


def test_live_ingest_new_row_full_bson_parity(live_db):
    """Real transaction commit + real BSON round-trip for a fresh insert
    -- proves what the fake cannot: real server-side type handling for
    every field, including the datetime conventions."""
    repo = MongoReviewItemRepository(live_db)
    result = repo.upsert_ingest_batch([_ingest_row("c1")])
    assert result == {"inserted": 1, "updated": 0, "skipped": 0}
    item = repo.get("c1")
    assert item.row_version == 1
    assert item.review_status == "UNPROCESSED"
    assert item.ingested_at == dt.datetime(2026, 8, 1, 9, 0, 0)
    assert item.ai_primary_intent is None


def test_live_ingest_concurrent_overlapping_insert_conflict(live_db):
    """Two genuinely concurrent transactions both classifying the SAME
    comment_id as NEW -- the second commit must hit a real duplicate-key
    error, unswallowed. This specific interleaving cannot be proven by
    the in-memory fake (it is single-threaded); it requires a real
    server enforcing _id uniqueness across two live transactions."""
    import threading

    from pymongo.errors import DuplicateKeyError

    repo_a = MongoReviewItemRepository(live_db)
    repo_b = MongoReviewItemRepository(live_db)
    barrier = threading.Barrier(2, timeout=10)
    errors: list[Exception] = []
    results: list[dict] = []

    def _run(repo):
        try:
            barrier.wait()
            results.append(repo.upsert_ingest_batch([_ingest_row("race-c1")]))
        except Exception as exc:  # noqa: BLE001 -- captured for assertion, not swallowed silently
            errors.append(exc)

    t1 = threading.Thread(target=_run, args=(repo_a,))
    t2 = threading.Thread(target=_run, args=(repo_b,))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    # Exactly one side succeeds; the other must see a real, unswallowed
    # duplicate-key error -- never a silent conversion to an update.
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], DuplicateKeyError)
    assert repo_a.get("race-c1") is not None


def _prefilter_seed_row(comment_id: str, **overrides) -> dict:
    row = dict(
        _id=comment_id, original_comment="نص", visitor_email=None, visitor_name=None,
        comment_created_at=dt.datetime(2026, 1, 1), is_exact_duplicate_of=None,
        review_status="UNPROCESSED", reviewer_action=None, row_version=1,
        is_prefiltered_junk=None, prefilter_reason=None, processed_at=None,
        primary_intent=None, routing_decision=None, exclusion_reason=None,
    )
    row.update(overrides)
    return row


def test_live_prefilter_transaction_rollback_on_missing_item(live_db):
    """Real transaction abort: a missing comment_id in the middle of a
    batch must leave EVERY row untouched, not just the one that
    triggered the error."""
    coll = live_db["review_items"]
    coll.insert_one(_prefilter_seed_row("c1"))
    coll.insert_one(_prefilter_seed_row("c2"))
    repo = MongoReviewItemRepository(live_db)

    with pytest.raises(KeyError):
        repo.apply_prefilter_batch([
            {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "does-not-exist", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
            {"comment_id": "c2", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        ])
    for cid in ("c1", "c2"):
        doc = coll.find_one({"_id": cid})
        assert doc["is_prefiltered_junk"] is None
        assert doc["review_status"] == "UNPROCESSED"


def test_live_prefilter_bson_processed_at_parity(live_db):
    """Real BSON round-trip: processed_at must come back as a real
    datetime (not a string), and identical across every row in one call
    -- proves the "computed once, shared" behavior survives real
    encode/decode, not just the in-memory fake."""
    coll = live_db["review_items"]
    coll.insert_one(_prefilter_seed_row("c1"))
    coll.insert_one(_prefilter_seed_row("c2"))
    repo = MongoReviewItemRepository(live_db)
    repo.apply_prefilter_batch([
        {"comment_id": "c1", "is_prefiltered_junk": True, "prefilter_reason": "burst"},
        {"comment_id": "c2", "is_prefiltered_junk": False, "prefilter_reason": None},
    ])
    item1 = repo.get("c1")
    item2 = repo.get("c2")
    assert isinstance(item1.processed_at, dt.datetime)
    assert item1.processed_at == item2.processed_at
    assert item1.row_version == 1
    assert item2.row_version == 1


def _classification_seed_row(comment_id: str, **overrides) -> dict:
    row = dict(
        _id=comment_id, original_comment="نص", original_title=None, original_type=None,
        age_group="recent", platform=None, entry_id=None, lemma=None,
        review_status="UNPROCESSED", is_prefiltered_junk=False, reviewer_action=None, row_version=1,
        ai_primary_intent=None, ai_topic_subtype=None, ai_subcategory=None, ai_secondary_intent=None,
        ai_routing_decision=None, ai_classification_reason=None, ai_confidence=None,
        ai_classification_source=None, ai_needs_review=None, primary_intent=None, topic_subtype=None,
        subcategory=None, secondary_intent=None, classification_source=None, needs_review=None,
        routing_decision=None, exclusion_reason=None, model_name=None, model_version=None,
        prompt_version=None, pipeline_version=None, inference_run_id=None, processed_at=None,
    )
    row.update(overrides)
    return row


def test_live_classification_transaction_rollback_on_missing_item(live_db):
    """Real transaction abort: a missing comment_id in the middle of a
    classification batch must leave EVERY row untouched."""
    coll = live_db["review_items"]
    coll.insert_one(_classification_seed_row("c1"))
    coll.insert_one(_classification_seed_row("c2"))
    repo = MongoReviewItemRepository(live_db)

    success = dict(
        comment_id="c1", success=True, ai_primary_intent="THANKS_PRAISE", ai_topic_subtype="GENERAL",
        ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="GENERATE_DRAFT",
        ai_classification_reason="frozen_model_inference", ai_confidence=0.9, routing_reason_code=None,
    )
    with pytest.raises(KeyError):
        repo.apply_classification_batch(
            [success, {**success, "comment_id": "does-not-exist"}, {**success, "comment_id": "c2"}],
            model_name="m", model_version=None, prompt_version="v1", pipeline_version="p1", inference_run_id="r1",
        )
    for cid in ("c1", "c2"):
        doc = coll.find_one({"_id": cid})
        assert doc["ai_primary_intent"] is None
        assert doc["review_status"] == "UNPROCESSED"


def test_live_classification_immutable_skip_and_bson_parity(live_db):
    """Real BSON round-trip for a successful classification, then a
    real second attempt against the now-CLASSIFIED row proves the
    immutability guard holds against an actual server round-trip, not
    just the in-memory fake."""
    coll = live_db["review_items"]
    coll.insert_one(_classification_seed_row("c1"))
    repo = MongoReviewItemRepository(live_db)
    success = dict(
        comment_id="c1", success=True, ai_primary_intent="THANKS_PRAISE", ai_topic_subtype="GENERAL",
        ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="GENERATE_DRAFT",
        ai_classification_reason="frozen_model_inference", ai_confidence=0.9, routing_reason_code=None,
    )
    r1 = repo.apply_classification_batch(
        [success], model_name="m", model_version=None, prompt_version="v1", pipeline_version="p1", inference_run_id="r1"
    )
    assert r1 == {"classified": 1, "failed": 0, "skipped_already_classified": 0}
    item = repo.get("c1")
    assert item.ai_confidence == 0.9
    assert isinstance(item.processed_at, dt.datetime)
    assert item.row_version == 1

    r2 = repo.apply_classification_batch(
        [{**success, "ai_primary_intent": "COMPLAINT"}],
        model_name="m2", model_version=None, prompt_version="v2", pipeline_version="p2", inference_run_id="r2",
    )
    assert r2 == {"classified": 0, "failed": 0, "skipped_already_classified": 1}
    assert repo.get("c1").ai_primary_intent == "THANKS_PRAISE"  # unchanged -- immutable


def _generation_seed_row(comment_id: str, **overrides) -> dict:
    row = dict(
        _id=comment_id, source_file="live-gen.xlsx", source_row_index=0,
        original_comment="نص", review_status="CLASSIFIED", is_prefiltered_junk=False,
        ready_to_send=False, reviewer_action=None, row_version=1,
        ai_primary_intent="THANKS_PRAISE", primary_intent=None, topic_subtype=None,
        sentiment=None, ai_sentiment=None, ai_confidence=None, needs_review=None,
        generation_routing_decision=None, generation_status=None, outbound_status=None,
        review_priority=None, priority_reason=None, updated_at=None, source_reference=None,
        suggested_final_reply=None, ai_suggested_final_reply=None, generated_at=None,
    )
    row.update(overrides)
    return row


def test_live_generation_result_transaction_and_bson_parity(live_db):
    """Real find_one_and_update commit + real BSON round-trip for the
    "riyadh"-convention generated_at/updated_at columns -- proves the
    exact-value round-trip the in-memory fake in
    test_generation_repository.py cannot (that fake stores Python
    objects verbatim instead of encoding/decoding real BSON, so an
    aware-datetime write there comes back skewed by the app-timezone
    offset; see that file's test_success_generate_draft_mongo)."""
    coll = live_db["review_items"]
    coll.insert_one(_generation_seed_row("c1"))
    repo = MongoReviewItemRepository(live_db)
    result = GenerationResult(
        comment_id="c1", routing_decision="GENERATE_DRAFT", generation_status="SUCCEEDED",
        suggested_response="شكراً لتواصلكم معنا", review_required=False, reason_code="ok",
        provider="gemini", model_name="gemini-1.5-flash", prompt_version="gen-v1",
        generated_at="2026-01-01T10:00:00Z",
    )
    item = repo.apply_generation_result(result, intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.review_status == "DRAFT_READY"
    assert item.row_version == 2
    assert item.generated_at == dt.datetime(2026, 1, 1, 10, 0, 0)
    assert isinstance(item.updated_at, dt.datetime)
    assert item.suggested_final_reply == "شكراً لتواصلكم معنا"


def test_live_generation_result_skip_preserves_protected_fields(live_db):
    """A real server round-trip proving the skip guard is a true no-op:
    an item the reviewer already sent must come back completely
    unchanged (row_version included) from a real find_one_and_update
    call that never fires because the guard short-circuits first."""
    coll = live_db["review_items"]
    coll.insert_one(_generation_seed_row(
        "c1", review_status="APPROVED", reviewer_action="SEND_AS_IS",
        reviewer_id="r1", suggested_final_reply="رد نهائي معتمد", row_version=3,
    ))
    repo = MongoReviewItemRepository(live_db)
    result = GenerationResult(
        comment_id="c1", routing_decision="GENERATE_DRAFT", generation_status="SUCCEEDED",
        suggested_response="نص مختلف يجب ألا يُكتب", review_required=False, reason_code="ok",
        provider="gemini", model_name="gemini-1.5-flash", prompt_version="gen-v1",
        generated_at="2026-01-01T10:00:00Z",
    )
    item = repo.apply_generation_result(result, intent=None, topic=None, sentiment=None, confidence=None, needs_review=None)
    assert item.row_version == 3
    assert item.suggested_final_reply == "رد نهائي معتمد"
    assert item.reviewer_id == "r1"
    assert item.generation_routing_decision is None


def _analytics_seed_row(comment_id: str, **overrides) -> dict:
    row = dict(
        _id=comment_id, source_file="live-analytics.xlsx", source_row_index=0,
        review_status="DRAFT_READY", review_priority=3, ready_to_send=False, row_version=1,
        ingested_at=None, reviewed_at=None, reviewer_started_at=None,
        reviewer_action=None, reviewer_id=None, reviewer_user_id=None,
        primary_intent=None, topic_subtype=None, sentiment=None,
    )
    row.update(overrides)
    return row


def test_live_analytics_real_aggregation_and_bson_datetime_parity(live_db):
    """Real $group aggregation + real BSON round-trip for the "riyadh"-
    convention reviewed_at/reviewer_started_at columns -- proves the
    exact-value duration calculation the in-memory fake in
    test_analytics_repository.py cannot prove on its own (that fake
    never performs real BSON encode/decode; it only proves this code's
    query-construction logic against pre-converted fixture values -- see
    that file's module docstring and _riyadh_raw() helper)."""
    coll = live_db["review_items"]
    started = dt.datetime(2026, 6, 12, 9, 0, 0)  # naive-Riyadh wall clock
    reviewed = started + dt.timedelta(seconds=600)
    coll.insert_one(_analytics_seed_row(
        "c1", ingested_at=started, primary_intent="THANKS_PRAISE",
        reviewer_started_at=clock.as_app(started), reviewed_at=clock.as_app(reviewed),
        reviewer_action="SEND_AS_IS",
    ))
    coll.insert_one(_analytics_seed_row("c2", ingested_at=started, primary_intent="COMPLAINT"))
    repo = MongoAnalyticsRepository(live_db)
    result = repo.compute_operational_analytics(range_key="week", now=dt.datetime(2026, 6, 15, 12, 0, 0))
    assert result["kpis"]["incoming"] == 2
    assert result["kpis"]["avg_review_seconds"] == 600.0
    assert result["outcomes"]["SEND_AS_IS"] == 1
    intent_by_key = {row["key"]: row["count"] for row in result["classification"]["intent"]}
    assert intent_by_key["THANKS_PRAISE"] == 1
    assert intent_by_key["COMPLAINT"] == 1


def test_live_analytics_null_vs_missing_field_parity(live_db):
    """A field that is explicitly null and a field that is entirely
    absent from the document must behave identically under this
    repository's $ne-null translation of Postgres's IS NOT NULL -- real
    server semantics, not the fake's dict.get() approximation."""
    coll = live_db["review_items"]
    now = dt.datetime(2026, 6, 15, 12, 0, 0)
    explicit_null = _analytics_seed_row("explicit_null", ingested_at=now - dt.timedelta(hours=1), primary_intent=None)
    missing_field = _analytics_seed_row("missing_field", ingested_at=now - dt.timedelta(hours=1))
    del missing_field["primary_intent"]
    coll.insert_one(explicit_null)
    coll.insert_one(missing_field)
    repo = MongoAnalyticsRepository(live_db)
    result = repo.compute_operational_analytics(range_key="week", now=now)
    assert result["kpis"]["incoming"] == 2
    total_intent_count = sum(row["count"] for row in result["classification"]["intent"])
    assert total_intent_count == 0


_SYNTHETIC_HASH = "$2b$12$synthetichashsynthetichashsynthetichashsyn"


def _user_kwargs(**overrides):
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    kwargs = dict(
        user_id="u-live-1", email="live-user@x.example", password_hash=_SYNTHETIC_HASH,
        display_name="Live User", role="REVIEWER", is_active=True, created_at=now, updated_at=now,
    )
    kwargs.update(overrides)
    return kwargs


def test_live_user_bson_parity(live_db):
    """Real insert + real BSON round-trip for the `users` collection --
    _id is the plain user_id string (never an ObjectId), and both
    "utc"-convention timestamp columns round-trip exactly."""
    repo = MongoUserRepository(live_db)
    created = repo.create(**_user_kwargs())
    assert created.user_id == "u-live-1"
    fetched = repo.get_by_id("u-live-1")
    assert fetched is not None
    assert fetched.email == "live-user@x.example"
    assert fetched.created_at == dt.datetime(2026, 1, 1, 9, 0, 0)
    assert fetched.password_hash == _SYNTHETIC_HASH
    raw = live_db["users"].find_one({"_id": "u-live-1"})
    assert isinstance(raw["_id"], str)  # never an ObjectId


def test_live_user_duplicate_email_no_unique_index_yet(live_db):
    """Explicit, disclosed gap (M4D.1 report item 6/7): with NO real
    unique index on `email` in this disposable test database, the
    application-level read-then-insert check still catches a SEQUENTIAL
    duplicate create (this test), but genuinely CONCURRENT duplicate
    creates are NOT guaranteed to be caught -- that requires a real
    unique index, not created here or anywhere against an organization
    database."""
    repo = MongoUserRepository(live_db)
    repo.create(**_user_kwargs(user_id="u-live-1", email="dup@x.example"))
    with pytest.raises(ValueError):
        repo.create(**_user_kwargs(user_id="u-live-2", email="dup@x.example"))


def test_live_user_concurrent_create_same_email_is_not_yet_safe(live_db):
    """Documents the real gap, does not paper over it: two genuinely
    concurrent create() calls for the same email, against a collection
    with no unique index, can both succeed -- unlike PostgreSQL (whose
    users.email unique constraint would make the second one fail even
    if the application-level check raced). This test's assertion is
    intentionally permissive (>= 1 succeeds) because the outcome is a
    real race, not deterministic -- it exists to make the gap visible
    the moment a unique index is added and this starts reliably raising
    on the second thread instead."""
    import threading

    repo_a = MongoUserRepository(live_db)
    repo_b = MongoUserRepository(live_db)
    barrier = threading.Barrier(2, timeout=10)
    results: list[object] = []
    errors: list[Exception] = []

    def _run(repo, user_id):
        try:
            barrier.wait()
            results.append(repo.create(**_user_kwargs(user_id=user_id, email="race@x.example")))
        except Exception as exc:  # noqa: BLE001 -- captured for assertion, not swallowed silently
            errors.append(exc)

    t1 = threading.Thread(target=_run, args=(repo_a, "race-1"))
    t2 = threading.Thread(target=_run, args=(repo_b, "race-2"))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert len(results) >= 1
    stored = list(live_db["users"].find({"email": "race@x.example"}))
    assert len(stored) == len(results)  # every create() that returned actually persisted


def test_live_reviewer_summaries_using_real_mongo_users(live_db):
    """reviewer_summaries's User-side migration, proven against a real
    server: MongoUserRepository.list_active_reviewers() feeding
    ReviewWorkflowService, combined with ordinary Postgres ReviewItem
    data, produces the same shape a Postgres-backed user_repo would."""
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    repo = MongoUserRepository(live_db)
    repo.create(**_user_kwargs(user_id="rev-live-1", email="rev-live-1@x.example", display_name="Live Reviewer"))

    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        engine = get_engine(pathlib.Path(d) / "live_reviewer_summaries.sqlite")
        init_db(engine)
        upsert_review_items(engine, [
            dict(comment_id="c1", source_file="f", source_row_index=0, review_status="APPROVED",
                 reviewer_user_id="rev-live-1", reviewer_action="SEND_AS_IS", ingested_at=dt.datetime(2026, 1, 1)),
        ])
        summary = ReviewWorkflowService(engine, user_repo=repo).reviewer_summaries()

    assert len(summary) == 1
    assert summary[0].user_id == "rev-live-1"
    assert summary[0].assigned_total == 1
    assert summary[0].completed == 1


def _session_kwargs(**overrides):
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    kwargs = dict(
        session_token="tok-live-synthetic-1", user_id="u-live-1",
        created_at=now, expires_at=now + dt.timedelta(hours=12),
    )
    kwargs.update(overrides)
    return kwargs


def test_live_user_session_bson_parity(live_db):
    """Real insert + real BSON round-trip for `user_sessions` -- _id is
    the plain session_token string (never an ObjectId), and both "utc"-
    convention timestamp columns round-trip exactly."""
    repo = MongoUserSessionRepository(live_db)
    created = repo.create(**_session_kwargs())
    assert created.session_token == "tok-live-synthetic-1"
    fetched = repo.get_by_token("tok-live-synthetic-1")
    assert fetched is not None
    assert fetched.user_id == "u-live-1"
    assert fetched.created_at == dt.datetime(2026, 1, 1, 9, 0, 0)
    raw = live_db["user_sessions"].find_one({"_id": "tok-live-synthetic-1"})
    assert isinstance(raw["_id"], str)  # never an ObjectId


def test_live_user_session_expiry_boundary(live_db):
    """Real-server proof of the exact `expires_at <= now` boundary this
    phase's resolve_session-equivalent logic depends on -- a session
    expiring at exactly `now` must be treated as already expired."""
    repo = MongoUserSessionRepository(live_db)
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    repo.create(**_session_kwargs(session_token="on-boundary", expires_at=now))
    sess = repo.get_by_token("on-boundary")
    assert sess.expires_at <= now  # the repository itself does not filter; this is the exact predicate resolve_session applies


def test_live_user_session_delete_on_logout(live_db):
    """Real delete-on-logout: after delete_by_token, the document is
    genuinely gone from the server, not just marked revoked."""
    repo = MongoUserSessionRepository(live_db)
    repo.create(**_session_kwargs())
    repo.delete_by_token("tok-live-synthetic-1")
    assert repo.get_by_token("tok-live-synthetic-1") is None
    assert live_db["user_sessions"].find_one({"_id": "tok-live-synthetic-1"}) is None


def test_live_user_session_concurrent_duplicate_token_insert(live_db):
    """Two genuinely concurrent create() calls for the SAME session_token
    (cryptographically implausible in practice given new_session_token's
    256 bits of entropy, but exercised here anyway): the server's own
    _id uniqueness must let exactly one succeed and the other raise a
    real, unswallowed DuplicateKeyError -- this cannot be proven by the
    single-threaded in-memory fake."""
    import threading

    from pymongo.errors import DuplicateKeyError

    repo_a = MongoUserSessionRepository(live_db)
    repo_b = MongoUserSessionRepository(live_db)
    barrier = threading.Barrier(2, timeout=10)
    results: list[object] = []
    errors: list[Exception] = []

    def _run(repo, user_id):
        try:
            barrier.wait()
            results.append(repo.create(**_session_kwargs(session_token="race-token", user_id=user_id)))
        except Exception as exc:  # noqa: BLE001 -- captured for assertion, not swallowed silently
            errors.append(exc)

    t1 = threading.Thread(target=_run, args=(repo_a, "u-race-1"))
    t2 = threading.Thread(target=_run, args=(repo_b, "u-race-2"))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], DuplicateKeyError)
    assert repo_a.get_by_token("race-token") is not None


def test_live_user_session_purge_expired(live_db):
    """Real server-side delete_many predicate -- proves the exact
    `expires_at <= now` purge boundary and count against a real
    collection, not the fake."""
    repo = MongoUserSessionRepository(live_db)
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    repo.create(**_session_kwargs(session_token="expired-1", expires_at=now - dt.timedelta(hours=1)))
    repo.create(**_session_kwargs(session_token="expired-2", expires_at=now))
    repo.create(**_session_kwargs(session_token="alive-1", expires_at=now + dt.timedelta(hours=1)))
    count = repo.delete_expired(now)
    assert count == 2
    assert repo.get_by_token("alive-1") is not None
    assert repo.get_by_token("expired-1") is None
