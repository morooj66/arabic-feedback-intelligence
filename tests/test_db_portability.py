"""Phase 5 — data-layer portability.

Covers:
  A. default SQLite startup still works exactly as before
  B. DATABASE_URL configuration boundary (config.settings.resolve_database_url)
  C. core SQLAlchemy schema compiles against the PostgreSQL dialect
  D. analytics trend bucketing is database-neutral (no func.strftime) and
     produces correct results
  E. get_engine() does not assume every caller passes a SQLite Path — a
     URL string is honored as-is (the only reason a real PostgreSQL URL
     doesn't connect here is the missing psycopg driver, not a hardcoded
     SQLite assumption in this codebase — proven explicitly below)

No real PostgreSQL server or driver is available in this environment
(psycopg/psycopg2 is not installed and none is required to run Phase 5).
Compatibility here is checked the strongest way possible without one:
compiling real DDL/queries against SQLAlchemy's PostgreSQL dialect.
"""

from __future__ import annotations

import datetime as dt
import inspect

from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from config import settings
from rcs.review import analytics
from rcs.review.models import ProcessingJob, ReviewItem, ReviewSession, User, UserSession
from rcs.review.repository import get_engine, init_db


# --- A. default SQLite startup -----------------------------------------


def test_default_sqlite_startup_unchanged(tmp_path):
    engine = get_engine(tmp_path / "phase5_a.sqlite")
    init_db(engine)
    assert engine.dialect.name == "sqlite"
    with Session(engine) as db:
        db.add(
            ReviewItem(
                comment_id="c1",
                source_file="f.xlsx",
                source_row_index=0,
                original_comment="تعليق حقيقي",
                review_status="DRAFT_READY",
            )
        )
        db.commit()
    with Session(engine) as db:
        assert db.get(ReviewItem, "c1") is not None


def test_get_engine_still_accepts_a_bare_path_like_every_existing_caller(tmp_path):
    """Every current call site (scripts, tests) passes a Path — this must
    keep producing sqlite:///<path> unchanged."""
    db_path = tmp_path / "phase5_a2.sqlite"
    engine = get_engine(db_path)
    assert str(engine.url) == f"sqlite:///{db_path}"


# --- B. DATABASE_URL boundary -------------------------------------------


def test_resolve_database_url_defaults_to_local_sqlite_when_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", None)
    default_path = tmp_path / "local.sqlite"
    assert settings.resolve_database_url(default_path) == f"sqlite:///{default_path}"


def test_resolve_database_url_prefers_env_when_set(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+psycopg://user:pass@host/dbname")
    default_path = tmp_path / "local.sqlite"
    assert settings.resolve_database_url(default_path) == "postgresql+psycopg://user:pass@host/dbname"


def test_dotenv_provided_database_url_resyncs_settings(tmp_path, monkeypatch):
    """A DATABASE_URL defined only in .env must actually take effect —
    settings.DATABASE_URL is captured once at import time (before .env is
    ever read), so load_dotenv_if_present() must re-sync it the same way
    it already does for GEMINI_API_KEY."""
    import os

    from rcs.review import demo_seed

    monkeypatch.setattr(settings, "DATABASE_URL", None)
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RCS_IGNORE_DOTENV", raising=False)        # this test exercises the loader itself, on a throw-away .env containing a dummy URL
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql+psycopg://user:pass@host/dbname\n")

    demo_seed.load_dotenv_if_present()

    assert os.environ.get("DATABASE_URL") == "postgresql+psycopg://user:pass@host/dbname"
    assert settings.DATABASE_URL == "postgresql+psycopg://user:pass@host/dbname"
    del os.environ["DATABASE_URL"]


def test_a_stray_database_url_never_affects_a_plain_get_engine_path_call(tmp_path, monkeypatch):
    """get_engine() itself must stay deterministic for Path callers (every
    test in this suite, every script) even if DATABASE_URL happens to be
    set in the environment — only resolve_database_url() call sites
    (scripts/serve_reviewer_api.py) opt into it."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+psycopg://user:pass@host/dbname")
    db_path = tmp_path / "unaffected.sqlite"
    engine = get_engine(db_path)
    assert str(engine.url) == f"sqlite:///{db_path}"


# --- C. PostgreSQL dialect compilation -----------------------------------


def test_all_models_compile_create_table_ddl_against_postgresql_dialect():
    dialect = postgresql.dialect()
    for model in (ReviewItem, ReviewSession, ProcessingJob, User, UserSession):
        ddl = str(CreateTable(model.__table__).compile(dialect=dialect))
        assert "CREATE TABLE" in ddl
        assert model.__tablename__ in ddl
        # SQLite type-affinity strings that would be a real problem if
        # they ever leaked into a PostgreSQL DDL string.
        assert "AUTOINCREMENT" not in ddl.upper()


def test_ensure_workflow_columns_is_a_strict_noop_on_postgresql(tmp_path):
    """The legacy SQLite-only ALTER TABLE patch-up must never run for a
    PostgreSQL engine. We can't open a real PostgreSQL connection here
    (no driver installed), so this checks the guard directly: any
    non-SQLite dialect name must short-circuit before touching SQL."""
    from rcs.review.repository import _ensure_workflow_columns

    class FakeDialect:
        name = "postgresql"

    class FakeEngine:
        dialect = FakeDialect()

        def begin(self):  # pragma: no cover - must never be called
            raise AssertionError("_ensure_workflow_columns must not open a connection for postgresql")

    _ensure_workflow_columns(FakeEngine())  # must return immediately, no error


# --- D. analytics is database-neutral ------------------------------------


def test_analytics_module_no_longer_uses_sqlite_only_strftime():
    source = inspect.getsource(analytics)
    assert "func.strftime" not in source


def test_trend_bucketing_produces_correct_day_grouped_counts(tmp_path):
    engine = get_engine(tmp_path / "phase5_d.sqlite")
    init_db(engine)
    rows = [
        ("t1", dt.datetime(2026, 8, 20, 9, 0), None),
        ("t2", dt.datetime(2026, 8, 20, 15, 0), dt.datetime(2026, 8, 21, 10, 0)),
        ("t3", dt.datetime(2026, 8, 21, 8, 0), None),
    ]
    with Session(engine) as db:
        for cid, ingested, reviewed in rows:
            db.add(
                ReviewItem(
                    comment_id=cid,
                    source_file="trend.xlsx",
                    source_row_index=0,
                    review_status="DRAFT_READY",
                    ingested_at=ingested,
                    reviewed_at=reviewed,
                )
            )
        db.commit()

    with Session(engine) as db:
        result = analytics.compute_operational_analytics(
            db, range_key="custom", now=dt.datetime(2026, 8, 22),
            custom_start=dt.datetime(2026, 8, 19), custom_end=dt.datetime(2026, 8, 22),
        )
    by_bucket = {p["bucket"]: p for p in result["trend"]}
    assert by_bucket["2026-08-20"]["incoming"] == 2
    assert by_bucket["2026-08-21"]["incoming"] == 1
    assert by_bucket["2026-08-21"]["reviewed"] == 1


# --- E. get_engine honors URL strings, not just SQLite Paths -------------


def test_get_engine_accepts_a_url_string_not_only_a_path(tmp_path):
    url = f"sqlite:///{tmp_path / 'phase5_e.sqlite'}"
    engine = get_engine(url)
    assert str(engine.url) == url


def test_postgresql_url_constructs_a_real_engine_via_the_installed_psycopg_driver(tmp_path, monkeypatch):
    """Phase 5.1: psycopg[binary] is now installed, so get_engine() must
    build a genuine PostgreSQL Engine object from a postgresql+psycopg
    URL — no SQLite assumption anywhere in the path (no Path coercion, no
    directory creation). Actually connecting requires a real reachable
    server and is intentionally NOT exercised here (no live credentials
    in this test) — that happens only via the explicit, credential-gated
    migration script."""
    from sqlalchemy.engine import Engine

    engine = get_engine("postgresql+psycopg://user:pass@localhost:5999/dbname")
    assert isinstance(engine, Engine)
    assert engine.dialect.name == "postgresql"
    assert engine.dialect.driver == "psycopg"
    # password is masked, never rendered raw
    assert "pass" not in str(engine.url)
    assert "***" in str(engine.url)
