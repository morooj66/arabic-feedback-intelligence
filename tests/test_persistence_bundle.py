"""
Persistence-decoupling milestone — proves the central composition
boundary (rcs.review.persistence) actually does what it claims:

  A/B. each backend's bundle is homogeneous (never mixed)
  C/D. backend selection fails closed, never silently falls back
  E.   auth/service operate against contracts, never a concrete
       Postgres import, once given a bundle
  F/G/H/I. cross-backend parity for reviewer_summaries, exact dedup,
       auth/session, and review-session/job lifecycles
  J.   swapping which Mongo "database" a bundle points at changes only
       the data reached, never application behavior
  K.   the sanitized-processing-failure fix has not regressed
  L.   nothing in this file ever touches a real network/organization
       database — every Mongo-shaped bundle here is the existing
       in-memory fake; every Postgres-shaped bundle is a disposable
       tmp_path SQLite file

No live MongoDB connection is used or required anywhere in this file.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pytest

from config import settings
from rcs.review.auth import authenticate, create_user, end_session, resolve_session, start_session
from rcs.review.excel_review_flow import run_content_deduplication
from rcs.review.persistence import (
    MissingPersistenceConfig,
    RepositoryBundle,
    UnsupportedPersistenceBackend,
    bundle_from_engine,
    bundle_from_mongo,
    build_repository_bundle,
    resolve_bundle,
)
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.review.sessions import create_job, create_session, get_job, latest_job_for_session, update_job
from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake


def _sqlite_engine(tmp_path, name="bundle.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


# --------------------------------------------------------------- A / B


def test_postgres_bundle_is_entirely_postgres(tmp_path):
    bundle = bundle_from_engine(_sqlite_engine(tmp_path))
    for field in bundle.__dataclass_fields__:
        assert type(getattr(bundle, field)).__name__.startswith("Postgres"), field


def test_mongo_bundle_is_entirely_mongo():
    bundle = bundle_from_mongo(_FakeDatabase())
    for field in bundle.__dataclass_fields__:
        assert type(getattr(bundle, field)).__name__.startswith("Mongo"), field


# --------------------------------------------------------------- C / D


def test_unsupported_backend_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "PERSISTENCE_BACKEND", "sqlite_but_pretend_its_something_else")
    with pytest.raises(UnsupportedPersistenceBackend):
        build_repository_bundle(engine_factory=lambda: (_ for _ in ()).throw(AssertionError("never called")))


def test_mongo_missing_config_fails_closed_with_no_postgres_fallback(monkeypatch):
    monkeypatch.setattr(settings, "PERSISTENCE_BACKEND", "mongo")
    monkeypatch.setattr(settings, "MONGODB_URI", None)
    monkeypatch.setattr(settings, "MONGODB_DATABASE", None)
    called = {"engine_factory": False}

    def _poison_factory():
        called["engine_factory"] = True
        raise AssertionError("must never be called for backend=mongo")

    with pytest.raises(MissingPersistenceConfig):
        build_repository_bundle(engine_factory=_poison_factory)
    assert called["engine_factory"] is False  # no silent Postgres fallback


def test_mongo_partial_config_still_fails_closed(monkeypatch):
    """Only one of MONGODB_URI/MONGODB_DATABASE set -- still a hard
    failure, never "half-configured but somehow working"."""
    monkeypatch.setattr(settings, "PERSISTENCE_BACKEND", "mongo")
    monkeypatch.setattr(settings, "MONGODB_URI", "mongodb://example.invalid/")
    monkeypatch.setattr(settings, "MONGODB_DATABASE", None)
    with pytest.raises(MissingPersistenceConfig):
        build_repository_bundle()


# --------------------------------------------------------------- E


@pytest.mark.parametrize(
    "module_name",
    [
        "rcs.review.auth",
        "rcs.review.sessions",
        "rcs.review.job_runner",
        "rcs.review.service",
        "rcs.review.excel_review_flow",
        # Production login/DB boundary hardening: rcs.review.entra_auth
        # used to be a documented, deliberate exception to this guard
        # (PostgresUserRepository instantiated directly for the
        # Microsoft SSO path) -- closed, so it now holds to the same
        # rule as every other auth-adjacent module.
        "rcs.review.entra_auth",
    ],
)
def test_business_modules_never_import_concrete_postgres_classes(module_name):
    """Structural proof for milestone requirement E: these modules must
    receive/use repository contracts through rcs.review.persistence,
    never choose a concrete backend class themselves. Reading the
    module's own source (not just its runtime imports, which Python
    would cache regardless) is the honest way to prove this — a lazy
    `from rcs.review.repositories.postgres import ...` hidden inside a
    function body would still show up in the source text."""
    import importlib

    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    assert "repositories.postgres" not in source, f"{module_name} still references a concrete Postgres repository"


def test_auth_and_service_operate_on_a_directly_supplied_bundle(tmp_path):
    """A caller can hand these modules a RepositoryBundle directly
    (what the served application does under PERSISTENCE_BACKEND=mongo)
    instead of a raw engine, and every existing business function keeps
    working unchanged -- proving they depend on the contract, not on
    "an engine happens to be a SQLAlchemy Engine"."""
    engine = _sqlite_engine(tmp_path)
    bundle = resolve_bundle(engine)
    assert resolve_bundle(bundle) is bundle  # already a bundle -- passthrough, not rewrapped

    create_user(bundle, email="bundle-direct@example.test", password="BundleDirect123!", display_name="Bundle Direct", role="ADMIN")
    assert authenticate(bundle, email="bundle-direct@example.test", password="BundleDirect123!") is not None
    svc = ReviewWorkflowService(bundle)
    assert svc.workflow_counts() is not None


# --------------------------------------------------------------- F: reviewer_summaries parity


def _seed_reviewer_summary_fixture(bundle, engine_or_none):
    """Same 3-row fixture (2 assigned to one reviewer -- one decided, one
    not -- 1 unassigned) built directly through the bundle, so both
    backends start from an identical state."""
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    reviewer = create_user(bundle, email="parity-reviewer@example.test", password="ParityPass123!", display_name="Parity Reviewer", role="REVIEWER")
    rows = [
        {"comment_id": "p-1", "source_file": "f", "source_row_index": 0, "original_comment": "one", "ingested_at": now, "review_status": "DRAFT_READY", "ready_to_send": False, "row_version": 1, "reviewer_user_id": reviewer.user_id},
        {"comment_id": "p-2", "source_file": "f", "source_row_index": 1, "original_comment": "two", "ingested_at": now, "review_status": "APPROVED", "ready_to_send": True, "row_version": 2, "reviewer_user_id": reviewer.user_id, "reviewer_action": "SEND_AS_IS"},
        {"comment_id": "p-3", "source_file": "f", "source_row_index": 2, "original_comment": "three", "ingested_at": now, "review_status": "DRAFT_READY", "ready_to_send": False, "row_version": 1},
    ]
    if engine_or_none is not None:
        upsert_review_items(engine_or_none, rows)
    else:
        for row in rows:
            bundle.review_items.collection.insert_one({**{k: None for k in ("reviewer_action",)}, "_id": row["comment_id"], **{k: v for k, v in row.items() if k != "comment_id"}})
    return reviewer


def test_reviewer_summaries_parity_between_postgres_and_mongo(tmp_path):
    pg_engine = _sqlite_engine(tmp_path)
    pg_bundle = bundle_from_engine(pg_engine)
    pg_reviewer = _seed_reviewer_summary_fixture(pg_bundle, pg_engine)
    pg_summary = ReviewWorkflowService(pg_bundle).reviewer_summaries()

    mongo_bundle = bundle_from_mongo(_FakeDatabase())
    mongo_reviewer = _seed_reviewer_summary_fixture(mongo_bundle, None)
    mongo_summary = ReviewWorkflowService(mongo_bundle).reviewer_summaries()

    assert len(pg_summary) == 1
    assert len(mongo_summary) == 1
    pg_row, mongo_row = pg_summary[0], mongo_summary[0]
    assert (pg_row.assigned_total, pg_row.pending, pg_row.completed) == (2, 1, 1)
    assert (mongo_row.assigned_total, mongo_row.pending, mongo_row.completed) == (2, 1, 1)
    assert pg_reviewer.email == mongo_reviewer.email == pg_row.email == mongo_row.email


# --------------------------------------------------------------- G: exact-dedup parity


_DEDUP_ROWS = [
    {"comment_id": "d-1", "source_file": "batch", "source_row_index": 0, "original_comment": "same text", "review_status": "UNPROCESSED"},
    {"comment_id": "d-2", "source_file": "batch", "source_row_index": 1, "original_comment": "same text", "review_status": "UNPROCESSED"},
    {"comment_id": "d-3", "source_file": "batch", "source_row_index": 2, "original_comment": "different text", "review_status": "UNPROCESSED"},
    {"comment_id": "d-4", "source_file": "batch", "source_row_index": 3, "original_comment": "   ", "review_status": "UNPROCESSED"},  # blank -> singleton
]


def _seed_dedup_rows(bundle, engine_or_none, rows):
    now = dt.datetime(2026, 1, 1)
    if engine_or_none is not None:
        upsert_review_items(engine_or_none, [
            {**r, "ingested_at": now, "ready_to_send": False, "row_version": 1} for r in rows
        ])
    else:
        for r in rows:
            bundle.review_items.collection.insert_one({
                "_id": r["comment_id"], "source_file": r["source_file"], "source_row_index": r["source_row_index"],
                "original_comment": r["original_comment"], "review_status": r["review_status"],
                "reviewer_action": None, "dedupe_canonical_id": None, "content_dedupe_key": None,
                "outbound_status": None, "ingested_at": now, "ready_to_send": False, "row_version": 1,
            })


def test_exact_dedup_parity_between_postgres_and_mongo(tmp_path):
    comment_ids = [r["comment_id"] for r in _DEDUP_ROWS]

    pg_engine = _sqlite_engine(tmp_path)
    pg_bundle = bundle_from_engine(pg_engine)
    _seed_dedup_rows(pg_bundle, pg_engine, _DEDUP_ROWS)
    pg_result = run_content_deduplication(pg_bundle, comment_ids)

    mongo_bundle = bundle_from_mongo(_FakeDatabase())
    _seed_dedup_rows(mongo_bundle, None, _DEDUP_ROWS)
    mongo_result = run_content_deduplication(mongo_bundle, comment_ids)

    assert pg_result == mongo_result == {"canonical": 3, "duplicate_content": 1, "skipped_excluded": 0}

    pg_d1, pg_d2 = pg_bundle.review_items.get("d-1"), pg_bundle.review_items.get("d-2")
    mongo_d1, mongo_d2 = mongo_bundle.review_items.get("d-1"), mongo_bundle.review_items.get("d-2")
    # d-1 (lowest source_row_index among the "same text" group) is canonical on both backends.
    assert pg_d1.dedupe_canonical_id == "d-1" and pg_d2.dedupe_canonical_id == "d-1" and pg_d2.review_status == "DUPLICATE_CONTENT"
    assert mongo_d1.dedupe_canonical_id == "d-1" and mongo_d2.dedupe_canonical_id == "d-1" and mongo_d2.review_status == "DUPLICATE_CONTENT"

    pg_d4, mongo_d4 = pg_bundle.review_items.get("d-4"), mongo_bundle.review_items.get("d-4")
    assert pg_d4.dedupe_canonical_id == "d-4" and mongo_d4.dedupe_canonical_id == "d-4"  # blank content: its own singleton


def test_exact_dedup_idempotent_on_both_backends(tmp_path):
    """Re-running dedup over already-resolved rows must be a pure no-op
    on both backends -- the exact idempotency guarantee the original
    inline-SQL version had."""
    comment_ids = [r["comment_id"] for r in _DEDUP_ROWS]

    pg_engine = _sqlite_engine(tmp_path)
    pg_bundle = bundle_from_engine(pg_engine)
    _seed_dedup_rows(pg_bundle, pg_engine, _DEDUP_ROWS)
    first = run_content_deduplication(pg_bundle, comment_ids)
    second = run_content_deduplication(pg_bundle, comment_ids)
    assert second == {"canonical": 0, "duplicate_content": 0, "skipped_excluded": 0}
    assert first != second

    mongo_bundle = bundle_from_mongo(_FakeDatabase())
    _seed_dedup_rows(mongo_bundle, None, _DEDUP_ROWS)
    mongo_first = run_content_deduplication(mongo_bundle, comment_ids)
    mongo_second = run_content_deduplication(mongo_bundle, comment_ids)
    assert mongo_second == {"canonical": 0, "duplicate_content": 0, "skipped_excluded": 0}
    assert mongo_first != mongo_second


# --------------------------------------------------------------- H: auth/session parity


def test_auth_and_session_parity_between_postgres_and_mongo(tmp_path):
    pg_bundle = bundle_from_engine(_sqlite_engine(tmp_path))
    mongo_bundle = bundle_from_mongo(_FakeDatabase())

    for bundle in (pg_bundle, mongo_bundle):
        user = create_user(bundle, email="parity-auth@example.test", password="ParityAuthPass123!", display_name="Parity Auth", role="REVIEWER")
        assert authenticate(bundle, email="parity-auth@example.test", password="wrong") is None
        assert authenticate(bundle, email="parity-auth@example.test", password="ParityAuthPass123!") is not None

        token = start_session(bundle, user_id=user.user_id)
        resolved = resolve_session(bundle, token)
        assert resolved is not None
        assert resolved.role == "REVIEWER"
        assert resolved.email == "parity-auth@example.test"

        end_session(bundle, token)
        assert resolve_session(bundle, token) is None  # logout invalidates immediately, both backends

        assert resolve_session(bundle, "not-a-real-token") is None  # unknown token, both backends


# --------------------------------------------------------------- I: review-session/job parity


def test_review_session_and_job_parity_between_postgres_and_mongo(tmp_path):
    pg_bundle = bundle_from_engine(_sqlite_engine(tmp_path))
    mongo_bundle = bundle_from_mongo(_FakeDatabase())

    for bundle in (pg_bundle, mongo_bundle):
        session = create_session(bundle, original_filename="f.xlsx", source_file="f.xlsx", upload_token="tok")
        job = create_job(bundle, session_id=session.session_id)
        assert job.status == "running"
        update_job(bundle, job.job_id, status="succeeded", finished=True)
        refreshed = get_job(bundle, job.job_id)
        assert refreshed.status == "succeeded"
        latest = latest_job_for_session(bundle, session.session_id)
        assert latest.job_id == job.job_id


# --------------------------------------------------------------- J: database-name swap changes only the target


def test_switching_mongo_database_changes_only_the_data_reached_not_behavior():
    """Two independent Mongo "databases" (the fake's equivalent of
    MONGODB_DATABASE=A vs. MONGODB_DATABASE=B against the same
    deployment) -- the SAME create_user/authenticate code path behaves
    identically against either, and a user created in one is never
    visible through the other. This is the structural proof that the
    database name is pure configuration, never business logic."""
    bundle_a = bundle_from_mongo(_FakeDatabase())
    bundle_b = bundle_from_mongo(_FakeDatabase())

    create_user(bundle_a, email="only-in-a@example.test", password="OnlyInAPass123!", display_name="Only In A", role="ADMIN")

    assert authenticate(bundle_a, email="only-in-a@example.test", password="OnlyInAPass123!") is not None
    assert authenticate(bundle_b, email="only-in-a@example.test", password="OnlyInAPass123!") is None  # isolated

    create_user(bundle_b, email="only-in-a@example.test", password="DifferentPass456!", display_name="Same Email Other DB", role="REVIEWER")
    user_from_b = authenticate(bundle_b, email="only-in-a@example.test", password="DifferentPass456!")
    assert user_from_b is not None
    assert user_from_b.role == "REVIEWER"  # a completely independent user, despite the same email


# --------------------------------------------------------------- K: no raw-exception/PII regression


def test_no_raw_exception_or_pii_regression_after_persistence_refactor(tmp_path, monkeypatch, caplog):
    """The job_runner.py sanitized-error-message fix from the prior
    milestone must survive this refactor unchanged: job.error_message
    stays a short, safe code even though job_runner now reaches
    processing_jobs through resolve_bundle() instead of a hardcoded
    PostgresProcessingJobRepository import."""
    import logging
    import time

    from rcs.review.excel_review_flow import write_fixture_workbook
    from rcs.review.excel_upload import save_uploaded_workbook
    from rcs.review.job_runner import start_or_resume_processing

    engine = _sqlite_engine(tmp_path)
    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")

    rows = [{
        "original_comment": "شكرًا", "original_title": None, "entry_id": None, "lemma": None,
        "visitor_name": "زائر", "visitor_email": "leak-check@example.test", "original_type": "تعليق",
        "comment_created_at": dt.datetime(2026, 1, 1), "platform": "الرياض",
    }]
    source = write_fixture_workbook(tmp_path / "src.xlsx", rows)
    saved = save_uploaded_workbook("src.xlsx", source.read_bytes())
    review_session = create_session(engine, original_filename="src.xlsx", source_file=saved.original_filename, upload_token=saved.token)

    secret_looking_text = "leak-check@example.test synthetic failure /secret/path"

    def _boom(*args, **kwargs):
        raise RuntimeError(secret_looking_text)

    monkeypatch.setattr("rcs.review.job_runner.process_uploaded_excel_batch", _boom)

    with caplog.at_level(logging.WARNING, logger="rcs.review.job_runner"):
        job_id = start_or_resume_processing(engine, review_session.session_id)
        deadline = time.time() + 30
        job = get_job(engine, job_id)
        while job.status == "running" and time.time() < deadline:
            time.sleep(0.1)
            job = get_job(engine, job_id)

    assert job.status == "failed"
    assert job.error_message == "processing_failed:RuntimeError"
    assert secret_looking_text not in job.error_message
    assert any(secret_looking_text not in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------- L: no organization DB/network access


def test_bundle_construction_never_touches_real_settings_values(tmp_path, monkeypatch):
    """Poisons settings.MONGODB_URI/MONGODB_DATABASE/DATABASE_URL with
    values that would fail loudly if anything actually tried to connect
    using them, then proves bundle_from_engine/bundle_from_mongo/
    resolve_bundle never read those settings at all -- only
    build_repository_bundle (never called in this test) does, and only
    when PERSISTENCE_BACKEND explicitly selects it."""
    monkeypatch.setattr(settings, "MONGODB_URI", "mongodb://poison.invalid:1/should-never-be-dialed")
    monkeypatch.setattr(settings, "MONGODB_DATABASE", "poison_should_never_be_opened")
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+psycopg://poison.invalid/should-never-connect")

    engine = _sqlite_engine(tmp_path)
    pg_bundle = bundle_from_engine(engine)  # must use the given tmp_path engine only
    assert type(pg_bundle.users).__name__ == "PostgresUserRepository"
    create_user(pg_bundle, email="isolated@example.test", password="IsolatedPass123!", display_name="Isolated", role="ADMIN")
    assert authenticate(pg_bundle, email="isolated@example.test", password="IsolatedPass123!") is not None

    mongo_bundle = bundle_from_mongo(_FakeDatabase())  # must use the given fake only
    assert type(mongo_bundle.users).__name__ == "MongoUserRepository"
    create_user(mongo_bundle, email="isolated2@example.test", password="IsolatedPass123!", display_name="Isolated 2", role="ADMIN")
    assert authenticate(mongo_bundle, email="isolated2@example.test", password="IsolatedPass123!") is not None

    assert resolve_bundle(engine) is not None  # normalization also never consults the poisoned settings
