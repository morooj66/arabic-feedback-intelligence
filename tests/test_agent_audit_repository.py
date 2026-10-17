"""
Phase M5E — concrete AgentAuditRepository implementations.

PostgreSQL: exercised against a real (SQLite-backed, via the same
`get_engine`/`init_db` used everywhere else in this test suite) engine
-- no live PostgreSQL required, matching this repo's own established
unit-test convention (SQLite standing in for PostgreSQL dialect-neutral
SQLAlchemy behavior; PostgreSQL-specific dialect quirks are out of
scope here, exactly as for every other repository test in this suite).

MongoDB: exercised against a small, purpose-built in-memory fake
(mirrors tests/test_mongo_repository_unit.py's own `_FakeCollection`
pattern) implementing exactly the pymongo surface
`MongoAgentAuditRepository` calls. NO live Mongo connection, NO
mongomock dependency added.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pymongo.errors import DuplicateKeyError

from rcs.response_intelligence.audit import build_audit_record
from rcs.review.repositories.mongo import MongoAgentAuditRepository
from rcs.review.repositories.postgres import PostgresAgentAuditRepository

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)


def _record(execution_id="exec-1", review_item_id="c-1", **overrides):
    base = dict(
        execution_id=execution_id,
        review_item_id=review_item_id,
        created_at=_SYNTHETIC_NOW,
        grounding_mode="IDENTIFYING_ONLY",
        official_retrieval_outcome="SUCCESS_EMPTY",
        official_result_count=0,
        official_grounding_available=False,
        style_retrieval_outcome="SUCCESS_EMPTY",
        style_result_count=0,
        logical_generation_count=1,
        refinement_attempted=False,
        initial_validation_passed=True,
        initial_failure_code=None,
        final_validation_passed=True,
        final_failure_code=None,
        generation_status="SUCCEEDED",
    )
    base.update(overrides)
    return build_audit_record(**base)


# --- PostgreSQL (SQLite-backed) ----------------------------------------------


@pytest.fixture
def pg_engine(tmp_path):
    from rcs.review.repository import get_engine, init_db

    engine = get_engine(tmp_path / "agent_audit.sqlite")
    init_db(engine)
    return engine


def test_postgres_record_and_get_by_execution_id(pg_engine):
    repo = PostgresAgentAuditRepository(pg_engine)
    record = _record()
    repo.record(record)
    fetched = repo.get_by_execution_id("exec-1")
    assert fetched == record


def test_postgres_list_for_review_item_returns_in_created_order(pg_engine):
    repo = PostgresAgentAuditRepository(pg_engine)
    early = _record(execution_id="exec-early", created_at=_SYNTHETIC_NOW)
    late = _record(execution_id="exec-late", created_at=_SYNTHETIC_NOW + dt.timedelta(minutes=5))
    repo.record(late)
    repo.record(early)
    traces = repo.list_for_review_item("c-1")
    assert [t.execution_id for t in traces] == ["exec-early", "exec-late"]


def test_postgres_record_is_idempotent_on_execution_id(pg_engine):
    repo = PostgresAgentAuditRepository(pg_engine)
    record = _record()
    repo.record(record)
    repo.record(record)  # simulated retry
    assert len(repo.list_for_review_item("c-1")) == 1


def test_postgres_two_genuine_executions_create_two_rows(pg_engine):
    repo = PostgresAgentAuditRepository(pg_engine)
    repo.record(_record(execution_id="exec-1"))
    repo.record(_record(execution_id="exec-2"))
    assert len(repo.list_for_review_item("c-1")) == 2


def test_postgres_get_by_execution_id_returns_none_when_absent(pg_engine):
    repo = PostgresAgentAuditRepository(pg_engine)
    assert repo.get_by_execution_id("does-not-exist") is None


def test_postgres_no_reviewer_row_dependency():
    """The audit repository never reads/writes review_items -- it is a
    fully independent table. Proven by recording a trace for a
    review_item_id that was never inserted into review_items at all."""
    from rcs.review.repository import get_engine, init_db

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        engine = get_engine(Path(d) / "no_review_item.sqlite")
        init_db(engine)
        repo = PostgresAgentAuditRepository(engine)
        repo.record(_record(review_item_id="never-inserted-anywhere"))
        assert len(repo.list_for_review_item("never-inserted-anywhere")) == 1


# --- MongoDB (fake collection, no live connection) --------------------------


class _FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key, direction=1):
        self._docs.sort(key=lambda d: d.get(key), reverse=(direction == -1))
        return self

    def __iter__(self):
        return iter(dict(d) for d in self._docs)


class _FakeCollection:
    def __init__(self):
        self._docs: list[dict] = []

    def insert_one(self, doc):
        if any(d["_id"] == doc["_id"] for d in self._docs):
            raise DuplicateKeyError(f"E11000 duplicate key error, _id: {doc['_id']!r}")
        self._docs.append(dict(doc))

    def find_one(self, filt):
        for d in self._docs:
            if all(d.get(k) == v for k, v in filt.items()):
                return dict(d)
        return None

    def find(self, filt):
        return _FakeCursor(d for d in self._docs if all(d.get(k) == v for k, v in filt.items()))


class _FakeDatabase:
    def __init__(self):
        self._collections: dict[str, _FakeCollection] = {}

    def __getitem__(self, name):
        return self._collections.setdefault(name, _FakeCollection())


@pytest.fixture
def mongo_db():
    return _FakeDatabase()


def test_mongo_record_and_get_by_execution_id(mongo_db):
    repo = MongoAgentAuditRepository(mongo_db)
    record = _record()
    repo.record(record)
    fetched = repo.get_by_execution_id("exec-1")
    assert fetched == record


def test_mongo_list_for_review_item_returns_in_created_order(mongo_db):
    repo = MongoAgentAuditRepository(mongo_db)
    early = _record(execution_id="exec-early", created_at=_SYNTHETIC_NOW)
    late = _record(execution_id="exec-late", created_at=_SYNTHETIC_NOW + dt.timedelta(minutes=5))
    repo.record(late)
    repo.record(early)
    traces = repo.list_for_review_item("c-1")
    assert [t.execution_id for t in traces] == ["exec-early", "exec-late"]


def test_mongo_record_is_idempotent_on_execution_id(mongo_db):
    repo = MongoAgentAuditRepository(mongo_db)
    record = _record()
    repo.record(record)
    repo.record(record)  # simulated retry -- must not raise, must not duplicate
    assert len(repo.list_for_review_item("c-1")) == 1


def test_mongo_two_genuine_executions_create_two_documents(mongo_db):
    repo = MongoAgentAuditRepository(mongo_db)
    repo.record(_record(execution_id="exec-1"))
    repo.record(_record(execution_id="exec-2"))
    assert len(repo.list_for_review_item("c-1")) == 2


def test_mongo_get_by_execution_id_returns_none_when_absent(mongo_db):
    repo = MongoAgentAuditRepository(mongo_db)
    assert repo.get_by_execution_id("does-not-exist") is None


def test_mongo_no_live_connection_required():
    """Structural proof: MongoAgentAuditRepository's constructor takes
    a plain injected `db` handle and never imports/constructs a real
    pymongo MongoClient itself."""
    import inspect

    from rcs.review.repositories import mongo as mongo_module

    source = inspect.getsource(mongo_module.MongoAgentAuditRepository)
    assert "MongoClient(" not in source
    assert "mongodb://" not in source
    assert "mongodb+srv://" not in source


# --- Backend-neutral contract proof ------------------------------------------


def test_both_backends_satisfy_the_same_protocol_structurally(pg_engine, mongo_db):
    """AgentAuditRepository is a plain (non-@runtime_checkable) Protocol
    -- matching this codebase's established convention (see
    SupportsGenerate) -- so this proves structural compatibility via
    the actual required method names/callability, not isinstance()."""
    from rcs.response_intelligence.audit import AgentAuditRepository

    required_methods = {"record", "list_for_review_item", "get_by_execution_id"}
    assert required_methods <= {name for name in dir(AgentAuditRepository) if not name.startswith("_")}

    pg_repo = PostgresAgentAuditRepository(pg_engine)
    mongo_repo = MongoAgentAuditRepository(mongo_db)
    for repo in (pg_repo, mongo_repo):
        for method in required_methods:
            assert callable(getattr(repo, method, None))


def test_both_backends_produce_equivalent_records_for_the_same_input(pg_engine, mongo_db):
    record = _record(execution_id="exec-parity")
    pg_repo = PostgresAgentAuditRepository(pg_engine)
    mongo_repo = MongoAgentAuditRepository(mongo_db)
    pg_repo.record(record)
    mongo_repo.record(record)
    assert pg_repo.get_by_execution_id("exec-parity") == mongo_repo.get_by_execution_id("exec-parity")
