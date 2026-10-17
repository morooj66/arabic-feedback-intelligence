"""Phase 5.1 — one-shot SQLite -> PostgreSQL data migration.

Narrowly scoped to exactly three tables: review_items, review_sessions,
processing_jobs. Every row is read via plain SQLAlchemy select()/ORM
(never string-built SQL) and written back via a single parameterized
bulk insert() per table, inside one PostgreSQL transaction that rolls
back completely on any error.

Safety, by construction:
  - the source SQLite file is opened read-only (SQLite's own ?mode=ro
    URI) — this process cannot write to it even if a bug tried to.
  - refuses if --target-url does not resolve to the postgresql dialect.
  - refuses if source and target resolve to the identical location.
  - refuses if the target already has rows in any of these three
    tables, unless --force-nonempty is passed explicitly (after the
    operator has audited what's there — this script does not decide
    that for you).
  - never issues DROP/TRUNCATE/DELETE — purely additive INSERTs.
  - migrates in a fixed, deterministic order (sessions, items, jobs).

Usage:
    python scripts/migrate_sqlite_to_postgres.py \\
        --source data/db/excel_review_demo.sqlite

  (--target-url defaults to DATABASE_URL via config.settings.
  resolve_database_url() — the same boundary the served app uses. Pass
  --target-url explicitly to override it.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, func, insert, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from config import settings  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402
from rcs.review.models import ProcessingJob, ReviewItem, ReviewSession  # noqa: E402

# Deterministic order. No SQL foreign keys are declared between these
# tables (review_sessions/processing_jobs reference review_items only by
# convention via source_file/session_id strings, see models.py), so this
# order is about readability/expectation, not constraint satisfaction.
TABLES_IN_ORDER = (ReviewSession, ReviewItem, ProcessingJob)


def _row_to_dict(row, model) -> dict:
    return {c.name: getattr(row, c.name) for c in model.__table__.columns}


def _readonly_sqlite_url(path: Path) -> str:
    """SQLite's own read-only connection mode — belt-and-suspenders on
    top of "this script simply never issues a write statement against
    the source engine": even a bug here cannot write to the file."""
    return f"sqlite:///file:{path.resolve()}?mode=ro&uri=true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Path to the source SQLite file (opened read-only).")
    parser.add_argument(
        "--target-url",
        default=None,
        help="Target PostgreSQL DATABASE_URL. Defaults to settings.resolve_database_url() (DATABASE_URL from .env/environment).",
    )
    parser.add_argument(
        "--force-nonempty",
        action="store_true",
        help="Allow migrating into a target that already has rows in review_items/review_sessions/processing_jobs. "
        "Off by default — audit what's there first.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"REFUSING: source SQLite file does not exist: {args.source}", file=sys.stderr)
        sys.exit(1)

    load_dotenv_if_present()
    target_url = args.target_url or settings.resolve_database_url(settings.EXCEL_OPERATIONAL_DB)

    if not target_url.startswith("postgresql"):
        print("REFUSING: target is not a postgresql:// URL — this script only migrates SQLite -> PostgreSQL.", file=sys.stderr)
        sys.exit(1)

    source_url = _readonly_sqlite_url(args.source)
    source_engine = create_engine(source_url, future=True)
    target_engine = create_engine(target_url, future=True)

    if target_engine.dialect.name != "postgresql":
        print(f"REFUSING: target dialect resolved to {target_engine.dialect.name!r}, expected postgresql.", file=sys.stderr)
        sys.exit(1)

    if str(source_engine.url) == str(target_engine.url):
        print("REFUSING: source and target resolve to the same location — ambiguous.", file=sys.stderr)
        sys.exit(1)

    # --- pre-flight: target must be empty for these three tables, unless explicitly overridden
    existing_counts: dict[str, int] = {}
    with Session(target_engine) as tgt:
        for model in TABLES_IN_ORDER:
            n = tgt.execute(select(func.count()).select_from(model)).scalar_one()
            if n:
                existing_counts[model.__tablename__] = n
    if existing_counts and not args.force_nonempty:
        print(f"REFUSING: target already has rows: {existing_counts}. Audit this, then pass --force-nonempty to proceed.", file=sys.stderr)
        sys.exit(1)

    # --- read source (SELECT-only; source engine is opened read-only)
    rows_by_table: dict[str, list[dict]] = {}
    with Session(source_engine) as src:
        for model in TABLES_IN_ORDER:
            rows = src.execute(select(model)).scalars().all()
            rows_by_table[model.__tablename__] = [_row_to_dict(r, model) for r in rows]
            print(f"read {len(rows_by_table[model.__tablename__])} rows from source.{model.__tablename__}")

    # --- write target inside one transaction; any error rolls back everything, source is never touched
    with target_engine.begin() as conn:
        for model in TABLES_IN_ORDER:
            payload = rows_by_table[model.__tablename__]
            if payload:
                conn.execute(insert(model.__table__), payload)
            print(f"wrote {len(payload)} rows to target.{model.__tablename__}")

    print("migration complete.")


if __name__ == "__main__":
    main()
