"""
Thin persistence layer around review_items.

Phase 2 needs `get_engine`, `init_db`, and `upsert_review_items`
(idempotent bulk upsert keyed by comment_id). Phase 3 adds
`load_for_prefilter` (read back the columns the prefilter rules need) and
`apply_prefilter_results` (write is_prefiltered_junk/prefilter_reason and,
for flagged rows only, the EXCLUDE routing fields) — both idempotent, both
UPDATE-only, never DELETE. Phase 4 adds `load_for_classification` (rows
that passed the prefilter and haven't been AI-classified yet) and
`apply_classification_results` (writes ai_* fields once, never again —
see that function's docstring for the exact overwrite-safety rules).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .models import Base, ReviewItem


def get_engine(db_path_or_url: Path | str):
    """Path (every current caller — scripts, tests) -> sqlite:///<path>,
    exactly today's behavior, unchanged. A string already containing
    "://" is treated as a complete SQLAlchemy URL (e.g. the result of
    config.settings.resolve_database_url(), which may point at
    PostgreSQL) and is used as-is. This is the one function the rest of
    the app goes through to get a database connection — nothing else
    constructs a URL or an Engine."""
    if isinstance(db_path_or_url, str) and "://" in db_path_or_url:
        url = db_path_or_url
    else:
        db_path = Path(db_path_or_url)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    # pool_pre_ping: a cheap liveness check before handing out a pooled
    # connection, so a connection gone stale between requests (the
    # Supabase session pooler's documented intermittent SSL drops) is
    # transparently replaced instead of surfacing as a request failure.
    # Harmless no-op overhead for SQLite.
    return create_engine(url, future=True, pool_pre_ping=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    _ensure_workflow_columns(engine)


_WORKFLOW_SQLITE_COLUMNS = (
    ("ai_sentiment", "VARCHAR"),
    ("sentiment", "VARCHAR"),
    ("generation_routing_decision", "VARCHAR"),
    ("generation_status", "VARCHAR"),
    ("generation_reason_code", "VARCHAR"),
    ("generation_provider", "VARCHAR"),
    ("generation_model_name", "VARCHAR"),
    ("generation_prompt_version", "VARCHAR"),
    ("generated_at", "DATETIME"),
    ("reviewer_display_name", "VARCHAR"),
    ("reviewer_started_at", "DATETIME"),
    ("no_response_reason", "TEXT"),
    ("outbound_status", "VARCHAR"),
    ("outbound_attempted_at", "DATETIME"),
    ("outbound_sent_at", "DATETIME"),
    ("outbound_external_id", "VARCHAR"),
    ("outbound_error_code", "VARCHAR"),
    ("outbound_error_message", "TEXT"),
    ("outbound_idempotency_key", "VARCHAR"),
    ("review_priority", "INTEGER"),
    ("priority_reason", "VARCHAR"),
    ("row_version", "INTEGER DEFAULT 1"),
    ("updated_at", "DATETIME"),
    ("source_reference", "VARCHAR"),
    ("content_dedupe_key", "VARCHAR"),
    ("dedupe_canonical_id", "VARCHAR"),
)


def _ensure_workflow_columns(engine) -> None:
    """Additive columns for pre-existing SQLite review.sqlite files that
    predate some of these columns (created before ORM-driven
    Base.metadata.create_all knew about them).

    Deliberately SQLite-only and dialect-guarded: a fresh PostgreSQL
    database created via init_db() already has every column the ORM
    models declare (create_all() always creates the full current
    schema), so this legacy patch-up never needs to run there — and its
    DDL type strings (e.g. "DATETIME") are SQLite type affinities, not
    valid PostgreSQL types, so it must never be attempted against
    PostgreSQL. Any future PostgreSQL schema change goes through Alembic
    (see migrations/), never this ad-hoc ALTER TABLE mechanism — that is
    exactly the gap Alembic exists to close (see Phase 5 report)."""
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "review_items" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("review_items")}
    with engine.begin() as conn:
        for name, ddl in _WORKFLOW_SQLITE_COLUMNS:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE review_items ADD COLUMN {name} {ddl}"))


def upsert_review_items(engine, rows: Iterable[dict]) -> dict:
    """
    Idempotent upsert by comment_id.

    - If a comment_id already exists, its ingest-owned fields are updated
      in place (no duplicate row is created).
    - If it doesn't exist, a new row is inserted.

    Returns counts: {"inserted": n, "updated": n}.
    """
    inserted = 0
    updated = 0
    with Session(engine) as session:
        for row in rows:
            existing = session.get(ReviewItem, row["comment_id"])
            if existing is None:
                session.add(ReviewItem(**row))
                inserted += 1
            else:
                for key, value in row.items():
                    setattr(existing, key, value)
                updated += 1
        session.commit()
    return {"inserted": inserted, "updated": updated}


INGEST_OWNED_FIELDS = (
    "source_file",
    "source_row_index",
    "original_comment",
    "original_title",
    "original_type",
    "entry_id",
    "lemma",
    "visitor_name",
    "visitor_email",
    "platform",
    "comment_created_at",
    "is_exact_duplicate_of",
    "ingested_at",
    "comment_age_days",
    "age_group",
    "source_reference",
)


def upsert_ingest_items(engine, rows: Iterable[dict]) -> dict:
    """
    Idempotent ingest write keyed by comment_id.

    New rows are inserted. Existing rows that already have generation or a
    reviewer decision are left unchanged. UNPROCESSED rows may refresh
    ingest-owned fields only — review/generation columns are never clobbered.

    Phase M4C.1: the actual persistence now lives in
    PostgresReviewItemRepository.upsert_ingest_batch (see
    rcs.review.repositories.postgres) -- this free function is kept as a
    thin, behavior-identical wrapper for any caller that still imports it
    directly (none does today; excel_review_flow.py now calls the
    repository). Import is local to avoid a module-level import cycle
    (repositories.postgres already imports INGEST_OWNED_FIELDS from this
    module).
    """
    from rcs.review.repositories.postgres import PostgresReviewItemRepository

    return PostgresReviewItemRepository(engine).upsert_ingest_batch(list(rows))


def count_rows(engine) -> int:
    with Session(engine) as session:
        return session.query(ReviewItem).count()


def load_for_prefilter(engine, comment_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Reads back exactly the columns the Phase 3 rules need, as a DataFrame
    indexed by comment_id. Does not read or expose any AI/review fields —
    those don't exist yet at this point in the pipeline.

    `comment_ids`, when given, restricts the read to that set (used by the
    Excel upload flow to (re)run the prefilter only for a specific batch of
    freshly-ingested rows, without touching every other row already in the
    table). Omitted (None) reproduces the original whole-table behavior
    used by scripts/run_pipeline.py.
    """
    from rcs.review.repositories.postgres import PostgresReviewItemRepository

    return PostgresReviewItemRepository(engine).load_for_prefilter(
        list(comment_ids) if comment_ids is not None else None
    )


def apply_prefilter_results(engine, results: pd.DataFrame) -> dict:
    """
    Idempotent UPDATE-only write of Phase 3 output.

    `results` must have one row per comment_id (same set already in the
    table) with columns: comment_id, is_prefiltered_junk, prefilter_reason.

    For flagged rows: sets primary_intent=SPAM, routing_decision=EXCLUDE,
    review_status=EXCLUDED, exclusion_reason=<prefilter_reason>. (Was
    category=TEST_SPAM_JUNK before the hierarchical taxonomy migration —
    TEST_SPAM_JUNK folded into primary_intent=SPAM per MIGRATION_PLAN.md
    Section 2. topic_subtype is left null: Phase 3's deterministic rules
    don't determine a topic, only that the row is junk.)
    For unflagged rows: clears those same fields back to their Phase-2
    defaults (UNPROCESSED / None) — safe in this MVP because no phase
    after Phase 3 exists yet to have moved a row further along.

    Note: this function is Phase 3's PERSISTENCE layer, not the prefilter
    RULES themselves (src/rcs/prefilter/spam_rules.py, untouched by this
    migration) — it only had to change here because it writes taxonomy
    field names/values that the hierarchical migration renamed/retired.

    No row is ever deleted. Returns {"excluded": n, "passed": n}.

    Phase M4C.2: the actual persistence now lives in
    PostgresReviewItemRepository.apply_prefilter_batch (see
    rcs.review.repositories.postgres) -- this free function is kept as a
    thin, behavior-identical wrapper for its existing direct callers
    (scripts/run_pipeline.py, local_demo_package.py, tests) that still
    pass a DataFrame; excel_review_flow.py now calls the repository
    directly with a list[dict]. Import is local to avoid a module-level
    import cycle (repositories.postgres already imports INGEST_OWNED_FIELDS
    from this module).
    """
    from rcs.review.repositories.postgres import PostgresReviewItemRepository

    return PostgresReviewItemRepository(engine).apply_prefilter_batch(results.to_dict("records"))


def load_for_classification(engine) -> pd.DataFrame:
    """
    Rows eligible for Gemini classification: passed the Phase 3 prefilter
    (is_prefiltered_junk == False) and not yet classified
    (review_status == "UNPROCESSED"). Both conditions are checked even
    though Phase 3 makes them equivalent today — defense in depth if that
    ever changes.

    Returns only the columns classify_row/build_prompt actually need.
    visitor_name/visitor_email are deliberately not included here, since
    Phase 4 must never send them to Gemini.

    Phase M4C.3: the actual persistence now lives in
    PostgresReviewItemRepository.load_for_classification (see
    rcs.review.repositories.postgres) -- this free function is kept as a
    thin, behavior-identical wrapper for its existing direct callers
    (scripts/run_classification.py, local_demo_package.py, tests);
    excel_review_flow.py now calls the repository directly.
    """
    from rcs.review.repositories.postgres import PostgresReviewItemRepository

    return PostgresReviewItemRepository(engine).load_for_classification()


def load_for_gold_sampling(engine) -> pd.DataFrame:
    """
    LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase B — the candidate pool for gold-
    dataset sample selection. Same eligibility filter as
    load_for_classification (passed Phase 3, not yet classified), but a
    DIFFERENT, purpose-built column set: adds `is_exact_duplicate_of`
    (needed by evaluation.sampling to collapse duplicate clusters to one
    representative before sampling) and `comment_created_at` (useful
    context for a labeler, and for later age-aware analysis), and does NOT
    return entry_id/lemma (irrelevant to sampling). Kept as a separate
    function rather than extending load_for_classification, which is
    Phase 4 infrastructure not touched by this phase.

    visitor_name/visitor_email are still deliberately excluded — the gold
    dataset is for human labeling of intent/topic, not for
    identity/contact review.
    """
    cols = [
        ReviewItem.comment_id,
        ReviewItem.original_comment,
        ReviewItem.original_title,
        ReviewItem.original_type,
        ReviewItem.age_group,
        ReviewItem.platform,
        ReviewItem.is_exact_duplicate_of,
        ReviewItem.comment_created_at,
    ]
    stmt = select(*cols).where(
        ReviewItem.review_status == "UNPROCESSED",
        ReviewItem.is_prefiltered_junk.is_(False),
    )
    with Session(engine) as session:
        rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=[c.key for c in cols])


def apply_classification_results(
    engine,
    results: list[dict],
    model_name: str,
    model_version: str | None,
    prompt_version: str,
    pipeline_version: str,
    inference_run_id: str,
) -> dict:
    """
    Idempotent, UPDATE-only write of Phase 4 output.

    `results` is a list of dicts, one per attempted comment_id, each
    shaped like a `ClassificationOutcome` (success flag + ai_* fields +
    routing_reason_code + error/attempts).

    Overwrite-safety rules, enforced here (not just at the query level in
    load_for_classification, so this function is safe even if called with
    stale/replayed input):

      - If the row's `ai_primary_intent` is already set, it is a previously
        SUCCESSFUL classification. This function skips it entirely —
        no field on that row is touched, successful or not. ai_* fields
        are immutable once written, per MIGRATION_PLAN.md Section 5.
      - If the row's `review_status` is no longer "UNPROCESSED" (e.g. a
        human reviewer has since acted on it, once Phase 7 exists), this
        function also skips it entirely, so a stale Phase 4 result can
        never clobber a review decision.
      - On success: writes ai_primary_intent/ai_topic_subtype/ai_subcategory/
        ai_secondary_intent/ai_routing_decision/ai_classification_reason/
        ai_confidence, initializes the matching *working* fields
        (primary_intent/topic_subtype/subcategory/secondary_intent/
        routing_decision) to the same values ("working value, initialized =
        ai_*"), sets exclusion_reason when routing is EXCLUDE, records
        model/prompt/pipeline/run metadata, and advances review_status to
        "CLASSIFIED". ai_immediate_reply_draft/ai_suggested_final_reply are
        NOT written here — Phase 4 is classification-only post-migration;
        those columns stay null until a future Phase 6 populates them.

        ai_classification_source/ai_needs_review (LOCAL_FIRST_ARCHITECTURE_
        PLAN.md Phase A) are read from `result` via `.get(...)`, defaulting
        to None if absent — the current Gemini-only classify_row/
        ClassificationOutcome path (unchanged in Phase A) does not populate
        these keys, so this stays backward-compatible with every existing
        Phase 4 caller. A future hybrid router can populate them once it
        exists, with no further change needed here.
      - On failure (schema validation failure or API/network error after
        retries): writes ai_routing_decision="REVIEW_ONLY" and
        ai_classification_reason describing the failure, records the
        attempt's metadata, but leaves ai_primary_intent unset and
        review_status as "UNPROCESSED" — so the row is picked up again by
        load_for_classification on the next pipeline run (retry across
        runs). reviewer_action/reviewer_edit/final_reply/ready_to_send/
        reviewed_at/reviewer_id are never touched by this function, on
        either path.

    Never deletes a row. Returns counts: {"classified": n, "failed": n,
    "skipped_already_classified": n}.

    Phase M4C.3: the actual persistence now lives in
    PostgresReviewItemRepository.apply_classification_batch (see
    rcs.review.repositories.postgres) -- this free function is kept as a
    thin, behavior-identical wrapper for its existing direct callers
    (scripts/run_classification.py, local_demo_package.py, tests);
    excel_review_flow.py now calls the repository directly.
    """
    from rcs.review.repositories.postgres import PostgresReviewItemRepository

    return PostgresReviewItemRepository(engine).apply_classification_batch(
        results,
        model_name=model_name,
        model_version=model_version,
        prompt_version=prompt_version,
        pipeline_version=pipeline_version,
        inference_run_id=inference_run_id,
    )
