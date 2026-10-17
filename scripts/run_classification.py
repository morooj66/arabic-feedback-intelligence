#!/usr/bin/env python3
"""
CLI entrypoint for Phase 4 — Gemini classification.

Deliberately a SEPARATE script from scripts/run_pipeline.py. Running
`python scripts/run_pipeline.py` (Phases 2+3) never triggers a Gemini
call. This script has to be invoked explicitly, and will refuse to run
without a real GEMINI_API_KEY set, so a live bulk classification run can
only ever happen on purpose.

By default this processes every eligible row (review_status="UNPROCESSED",
is_prefiltered_junk=False) it finds. Use --limit to process a small batch
first — recommended before ever running it unbounded.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from rcs.classification.classifier import classify_row  # noqa: E402
from rcs.classification.gemini_client import GeminiClient  # noqa: E402
from rcs.review.repository import (  # noqa: E402
    apply_classification_results,
    get_engine,
    init_db,
    load_for_classification,
)


def count_eligible() -> int:
    """Read-only: how many rows are currently eligible for classification.
    Makes no Gemini call."""
    engine = get_engine(settings.DB_PATH)
    init_db(engine)
    return len(load_for_classification(engine))


def run_classification(limit: int | None = None) -> dict:
    engine = get_engine(settings.DB_PATH)
    init_db(engine)

    df = load_for_classification(engine)
    if limit is not None:
        df = df.head(limit)

    if df.empty:
        return {"eligible_rows": 0, "classified": 0, "failed": 0, "skipped_already_classified": 0}

    client = GeminiClient(api_key=settings.GEMINI_API_KEY, model_name=settings.GEMINI_MODEL_NAME)
    inference_run_id = str(uuid.uuid4())

    results = []
    for _, row in df.iterrows():
        outcome = classify_row(row.to_dict(), client, max_retries=settings.GEMINI_MAX_RETRIES)
        results.append({"comment_id": row["comment_id"], **outcome.__dict__})

    write_result = apply_classification_results(
        engine,
        results,
        model_name=settings.GEMINI_MODEL_NAME,
        model_version=None,
        prompt_version=settings.PROMPT_VERSION,
        pipeline_version=settings.PIPELINE_VERSION,
        inference_run_id=inference_run_id,
    )
    return {"eligible_rows": len(df), **write_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Gemini classification (live API calls).")
    parser.add_argument("--limit", type=int, default=None, help="Classify at most N rows (recommended for a first run).")
    parser.add_argument("--count-only", action="store_true", help="Only report how many rows are eligible; makes no Gemini call.")
    args = parser.parse_args()

    if args.count_only:
        print(f"eligible_rows: {count_eligible()}")
    else:
        result = run_classification(limit=args.limit)
        for key, value in result.items():
            print(f"{key}: {value}")
