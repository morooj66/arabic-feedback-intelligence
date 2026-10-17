#!/usr/bin/env python3
"""
CLI entrypoint for the Response Classification System pipeline.

Currently implements:
  Phase 2 — ingest (load the historical workbook) and cleaning (text
            normalization, exact-duplicate flagging, age computation),
            persisted with review_status="UNPROCESSED".
  Phase 3 — deterministic spam/junk prefilter (no network calls): flags
            rows per src/rcs/prefilter/spam_rules.py and routes them to
            EXCLUDE; everything else stays UNPROCESSED for Phase 4.

Phases 4-9 (Gemini classification, routing, drafting, review,
evaluation) will be appended to this script as they are implemented. No
network calls, no sending, are made anywhere in this script.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from rcs.ingest.cleaning import clean  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402
from rcs.prefilter.spam_rules import BurstConfig, apply_prefilter  # noqa: E402
from rcs.review.repository import (  # noqa: E402
    apply_prefilter_results,
    get_engine,
    init_db,
    load_for_prefilter,
    upsert_review_items,
)


def run_ingest_and_clean(as_of: dt.datetime | None = None) -> dict:
    as_of = as_of or dt.datetime.utcnow()

    source_path = settings.RAW_DIR / settings.SOURCE_FILE
    raw_df = load_raw(source_path, settings.SOURCE_COLUMNS)
    raw_row_count = len(raw_df)

    cleaned_df = clean(
        raw_df,
        as_of=as_of,
        recent_max_days=settings.AGE_RECENT_MAX_DAYS,
        delayed_max_days=settings.AGE_DELAYED_MAX_DAYS,
    )

    engine = get_engine(settings.DB_PATH)
    init_db(engine)

    columns_to_persist = [
        "comment_id", "source_file", "source_row_index",
        "original_comment", "original_title", "original_type",
        "entry_id", "lemma", "visitor_name", "visitor_email", "platform",
        "comment_created_at", "is_exact_duplicate_of", "ingested_at",
        "comment_age_days", "age_group", "review_status", "ready_to_send",
    ]
    records = cleaned_df[columns_to_persist].to_dict(orient="records")
    upsert_result = upsert_review_items(engine, records)

    duplicate_count = int(cleaned_df["is_exact_duplicate_of"].notna().sum())

    return {
        "raw_row_count": raw_row_count,
        "cleaned_row_count": len(cleaned_df),
        "duplicate_row_count": duplicate_count,
        "unique_row_count": raw_row_count - duplicate_count,
        "inserted": upsert_result["inserted"],
        "updated": upsert_result["updated"],
        "as_of": as_of.isoformat(),
        "db_path": str(settings.DB_PATH),
    }


def run_prefilter() -> dict:
    """
    Phase 3. Reads current review_items, applies the six deterministic
    rules, writes results back. Idempotent — safe to re-run any number of
    times; recomputes from stored fields rather than incremental state.
    """
    engine = get_engine(settings.DB_PATH)
    init_db(engine)

    df = load_for_prefilter(engine)
    total_rows = len(df)

    burst_config = BurstConfig(
        min_count=settings.BURST_MIN_COUNT,
        window_hours=settings.BURST_WINDOW_HOURS,
    )
    flagged_df = apply_prefilter(df, settings.KNOWN_JUNK_STRINGS, burst_config)

    write_result = apply_prefilter_results(
        engine, flagged_df[["comment_id", "is_prefiltered_junk", "prefilter_reason"]]
    )

    reason_counts = (
        flagged_df.loc[flagged_df["is_prefiltered_junk"], "prefilter_reason"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_rows": total_rows,
        "excluded_by_prefilter": write_result["excluded"],
        "remaining_for_classification": write_result["passed"],
        "reason_counts": reason_counts,
    }


if __name__ == "__main__":
    ingest_result = run_ingest_and_clean()
    for key, value in ingest_result.items():
        print(f"{key}: {value}")

    print()
    prefilter_result = run_prefilter()
    for key, value in prefilter_result.items():
        print(f"{key}: {value}")
