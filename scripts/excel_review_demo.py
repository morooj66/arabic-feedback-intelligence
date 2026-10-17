#!/usr/bin/env python3
"""
Demo: read-only import from the historical visitors workbook, upsert ReviewItems,
leave them for the normal reviewer workflow, then export a NEW workbook.

Never overwrites the source Excel. Does not send to any external platform.

By default (no --live-gemini) generation is an offline canned stub — no
network, no GEMINI_API_KEY required. Pass --live-gemini for the real E2E
path: frozen classifiers (tfidf_logistic_intent / marbertv2_topic /
CAMeLBERT sentiment) -> existing eligibility -> real rg-v7 Gemini draft.
That path requires GEMINI_API_KEY and fails clearly without it.

  python scripts/excel_review_demo.py --limit 4
  python scripts/excel_review_demo.py --limit 4 --live-gemini
  python scripts/excel_review_demo.py --export-only --db data/db/excel_review_demo.sqlite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from rcs.review.excel_review_flow import (  # noqa: E402
    apply_offline_generation,
    export_reviewed_workbook,
    import_subset,
    process_uploaded_excel,
    source_digest,
)
from rcs.review.repository import get_engine, init_db  # noqa: E402
from rcs.review.service import ReviewWorkflowService  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "reviewed_responses_output.xlsx"
DEFAULT_DB = settings.DB_DIR / "excel_review_demo.sqlite"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Excel review import/export demo (source is read-only).")
    parser.add_argument("--source", type=Path, default=settings.RAW_DIR / settings.SOURCE_FILE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument(
        "--live-gemini",
        action="store_true",
        help=(
            "Use the real Excel E2E path: frozen classifiers -> eligibility -> "
            "real rg-v7 Gemini draft. Requires GEMINI_API_KEY. Without this "
            "flag, generation is an offline canned stub (no network)."
        ),
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"source workbook not found: {args.source}", file=sys.stderr)
        return 2

    if args.live_gemini:
        if os.environ.get("GEMINI_API_KEY") and not settings.GEMINI_API_KEY:
            settings.GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
        if not settings.GEMINI_API_KEY:
            print(
                "ERROR: --live-gemini requires GEMINI_API_KEY (set in the "
                "environment or .env). Refusing to fall back to canned replies.",
                file=sys.stderr,
            )
            return 2

    before = source_digest(args.source)
    engine = get_engine(args.db)
    init_db(engine)
    svc = ReviewWorkflowService(engine)

    imported = {"comment_ids": []}
    summary = None
    if not args.export_only:
        if args.live_gemini:
            summary = process_uploaded_excel(engine, args.source, limit=args.limit)
            imported = {"comment_ids": summary.comment_ids}
        else:
            imported = import_subset(engine, args.source, limit=args.limit)
            if not args.skip_generation:
                apply_offline_generation(svc, imported["comment_ids"])

    out = export_reviewed_workbook(engine, args.output, imported["comment_ids"] or None)
    after = source_digest(args.source)
    if after != before:
        print("ERROR: source workbook bytes changed", file=sys.stderr)
        return 1

    report = {
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "source_unchanged": True,
        "db": str(args.db),
        "output": str(out),
        "comment_ids": imported.get("comment_ids") or [],
        "note": "Reviewer UI/API can act on this isolated DB; re-run --export-only after actions.",
    }
    if summary is not None:
        report["processing_summary"] = {
            "total_rows": summary.total_rows,
            "inserted_rows": summary.inserted_rows,
            "duplicate_rows": summary.duplicate_rows,
            "excluded_rows": summary.excluded_rows,
            "classification_processed": summary.classification_processed,
            "generated_drafts": summary.generated_drafts,
            "no_response_recommended": summary.no_response_recommended,
            "review_only": summary.review_only,
            "failed_rows": summary.failed_rows,
        }
    else:
        report["import"] = {k: imported[k] for k in imported if k != "comment_ids"}
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
