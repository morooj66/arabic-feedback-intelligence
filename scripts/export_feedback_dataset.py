#!/usr/bin/env python3
"""
Phase 2 — export the internal human-feedback dataset (read-only over
review_items). One row per completed CANONICAL reviewer decision. No
visitor PII, no duplicate-source-row inflation. Never touches the
reviewer-facing Excel export, the source workbook, or review_items.

  python scripts/export_feedback_dataset.py --db data/db/excel_review_demo.sqlite
  python scripts/export_feedback_dataset.py --db data/db/excel_review_demo.sqlite --output /tmp/feedback.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from rcs.review.feedback import export_feedback_dataset, list_feedback_examples, summarize_feedback  # noqa: E402
from rcs.review.repository import get_engine, init_db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the internal Phase 2 feedback dataset (CSV).")
    parser.add_argument("--db", type=Path, default=settings.EXCEL_OPERATIONAL_DB)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "feedback_dataset.csv")
    parser.add_argument("--source-file", type=str, default=None, help="Restrict to one source_file, if given.")
    args = parser.parse_args()

    engine = get_engine(args.db)
    init_db(engine)

    examples = list_feedback_examples(engine, source_file=args.source_file)
    summary = summarize_feedback(examples)
    out = export_feedback_dataset(engine, args.output, source_file=args.source_file)

    report = {
        "db": str(args.db),
        "output": str(out),
        "rows_written": len(examples),
        "summary": {
            "completed_total": summary.completed_total,
            "send_as_is_count": summary.send_as_is_count,
            "edit_and_send_count": summary.edit_and_send_count,
            "no_response_needed_count": summary.no_response_needed_count,
            "acceptance_as_is_rate": summary.acceptance_as_is_rate,
            "edit_rate": summary.edit_rate,
            "no_response_rate": summary.no_response_rate,
            "average_edit_ratio": summary.average_edit_ratio,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
