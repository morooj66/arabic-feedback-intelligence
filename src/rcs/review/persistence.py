"""
Persistence composition root.

This is the ONE module in the served application that is allowed to know
both concrete repository backends (PostgreSQL, MongoDB) exist and decide
which one a running process uses. Every other runtime module — auth.py,
sessions.py, job_runner.py, service.py, excel_review_flow.py, http_api.py
— depends only on the repository Protocol contracts in
rcs.review.repositories.contracts, reached through a RepositoryBundle.

Backend selection is config-driven (config.settings.PERSISTENCE_BACKEND),
never inferred from the shape of an argument, and never silently falls
back from Mongo to PostgreSQL on missing/invalid Mongo configuration —
see build_repository_bundle's docstring.

Call sites that already hold a SQLAlchemy engine (every existing test and
script — this has not changed) pass it through resolve_bundle(engine),
which is a no-op for objects already a RepositoryBundle and otherwise
wraps the engine into the PostgreSQL bundle exactly as service.py/auth.py
/sessions.py/job_runner.py already did inline before this module existed.
This is what lets every existing caller keep working completely unchanged
while a new caller (the served application's entrypoint, when
PERSISTENCE_BACKEND=mongo) can instead build and pass a Mongo-backed
bundle — the business-logic modules never notice the difference.
"""

from __future__ import annotations

from dataclasses import dataclass

from rcs.response_intelligence.audit import AgentAuditRepository
from rcs.review.repositories.contracts import (
    AccountRequestRepository,
    AnalyticsRepository,
    DirectMessageRepository,
    PasswordResetTokenRepository,
    ProcessingJobRepository,
    ReviewItemRepository,
    ReviewSessionRepository,
    UserRepository,
    UserSessionRepository,
)


@dataclass(frozen=True)
class RepositoryBundle:
    """The complete set of repositories the served application needs,
    all from the SAME backend. There is no partial/mixed bundle — a
    bundle is always constructed by one of the two functions below, each
    of which populates every field from one concrete implementation."""

    review_items: ReviewItemRepository
    users: UserRepository
    user_sessions: UserSessionRepository
    review_sessions: ReviewSessionRepository
    processing_jobs: ProcessingJobRepository
    analytics: AnalyticsRepository
    audit: AgentAuditRepository
    # Optional so a hand-built bundle (e.g. a test double) that predates password reset keeps working; every real builder below populates it.
    password_reset_tokens: PasswordResetTokenRepository | None = None
    account_requests: AccountRequestRepository | None = None
    messages: DirectMessageRepository | None = None  # same optionality rule as password_reset_tokens


class UnsupportedPersistenceBackend(RuntimeError):
    """PERSISTENCE_BACKEND is set to something other than "postgres" or
    "mongo"."""


class MissingPersistenceConfig(RuntimeError):
    """The selected backend is missing configuration it requires (e.g.
    mongo without MONGODB_URI/MONGODB_DATABASE). Raised instead of
    falling back to another backend — a misconfigured deployment must
    fail loudly at startup, never silently run against the wrong
    database or the wrong engine."""


def bundle_from_engine(engine) -> RepositoryBundle:
    """Builds the PostgreSQL/SQLAlchemy bundle from an existing engine —
    the exact set of repository constructions that used to be written
    inline (and separately, per module) in auth.py/sessions.py/
    job_runner.py/service.py."""
    from rcs.review.repositories.postgres import (
        PostgresAccountRequestRepository,
        PostgresAgentAuditRepository,
        PostgresAnalyticsRepository,
        PostgresDirectMessageRepository,
        PostgresPasswordResetTokenRepository,
        PostgresProcessingJobRepository,
        PostgresReviewItemRepository,
        PostgresReviewSessionRepository,
        PostgresUserRepository,
        PostgresUserSessionRepository,
    )

    return RepositoryBundle(
        review_items=PostgresReviewItemRepository(engine),
        users=PostgresUserRepository(engine),
        user_sessions=PostgresUserSessionRepository(engine),
        review_sessions=PostgresReviewSessionRepository(engine),
        processing_jobs=PostgresProcessingJobRepository(engine),
        analytics=PostgresAnalyticsRepository(engine),
        audit=PostgresAgentAuditRepository(engine),
        password_reset_tokens=PostgresPasswordResetTokenRepository(engine),
        messages=PostgresDirectMessageRepository(engine),
        account_requests=PostgresAccountRequestRepository(engine),
    )


def bundle_from_mongo(db) -> RepositoryBundle:
    """Builds the MongoDB bundle from an existing pymongo Database
    handle. `db` already carries which deployment/database it points at
    (see rcs.review.repositories.mongo_client.get_mongo_database) — this
    function never hardcodes a database name or a connection string."""
    from rcs.review.repositories.mongo import (
        MongoAccountRequestRepository,
        MongoAgentAuditRepository,
        MongoAnalyticsRepository,
        MongoDirectMessageRepository,
        MongoPasswordResetTokenRepository,
        MongoProcessingJobRepository,
        MongoReviewItemRepository,
        MongoReviewSessionRepository,
        MongoUserRepository,
        MongoUserSessionRepository,
    )

    return RepositoryBundle(
        review_items=MongoReviewItemRepository(db),
        users=MongoUserRepository(db),
        user_sessions=MongoUserSessionRepository(db),
        review_sessions=MongoReviewSessionRepository(db),
        processing_jobs=MongoProcessingJobRepository(db),
        analytics=MongoAnalyticsRepository(db),
        audit=MongoAgentAuditRepository(db),
        password_reset_tokens=MongoPasswordResetTokenRepository(db),
        messages=MongoDirectMessageRepository(db),
        account_requests=MongoAccountRequestRepository(db),
    )


def resolve_bundle(engine_or_bundle) -> RepositoryBundle:
    """The single normalization point auth.py/sessions.py/job_runner.py/
    service.py call instead of importing a concrete Postgres/Mongo class
    themselves. Already a RepositoryBundle (the served application under
    PERSISTENCE_BACKEND=mongo, or any caller that built one directly) ->
    returned as-is. Anything else is treated as a SQLAlchemy engine and
    wrapped into the PostgreSQL bundle — today's exact behavior for
    every existing test and script, unchanged."""
    if isinstance(engine_or_bundle, RepositoryBundle):
        return engine_or_bundle
    return bundle_from_engine(engine_or_bundle)


def build_repository_bundle(engine_factory=None):
    """The served application's actual composition root
    (scripts/serve_reviewer_api.py calls this once at startup). Reads
    config.settings.PERSISTENCE_BACKEND:

      "postgres" (default) -- calls `engine_factory()` (the caller's
      existing get_engine(resolve_database_url(...)) construction, so
      the choice of --db / --seed / --production-db / DATABASE_URL is
      entirely unaffected) and wraps the result via bundle_from_engine.
      Returns (bundle, engine) so the caller can still run init_db(),
      resume_orphaned_jobs(), etc. exactly as today.

      "mongo" -- requires both settings.MONGODB_URI and
      settings.MONGODB_DATABASE. Connects and wraps the result via
      bundle_from_mongo. Returns (bundle, None) -- there is no
      SQLAlchemy engine in this path, and callers must not assume one
      exists (schema bootstrapping/init_db has no meaning for Mongo; see
      scripts/bootstrap_mongo_indexes.py for the equivalent, explicitly-
      invoked step).

      Anything else -- raises UnsupportedPersistenceBackend.

    Never falls back from mongo to postgres on missing Mongo config;
    raises MissingPersistenceConfig instead. This is the only function
    in the served application that reads PERSISTENCE_BACKEND."""
    from config import settings

    backend = (settings.PERSISTENCE_BACKEND or "postgres").strip().lower()

    if backend == "postgres":
        if engine_factory is None:
            raise MissingPersistenceConfig("PERSISTENCE_BACKEND=postgres requires an engine_factory")
        engine = engine_factory()
        return bundle_from_engine(engine), engine

    if backend == "mongo":
        if not settings.MONGODB_URI or not settings.MONGODB_DATABASE:
            raise MissingPersistenceConfig(
                "PERSISTENCE_BACKEND=mongo requires both MONGODB_URI and MONGODB_DATABASE to be set"
            )
        from rcs.review.repositories.mongo_client import get_mongo_database

        _client, db = get_mongo_database(settings.MONGODB_URI, settings.MONGODB_DATABASE)
        return bundle_from_mongo(db), None

    raise UnsupportedPersistenceBackend(f"unknown PERSISTENCE_BACKEND: {backend!r} (expected 'postgres' or 'mongo')")
