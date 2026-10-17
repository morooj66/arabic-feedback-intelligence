"""Phase M-MANUAL — generate a TINY Extended JSON v2 sample package for
manual Atlas import verification. Scope, deliberately narrow:

  * exactly 2 review_items, 1 review_session, 1 processing_job
  * users and user_sessions are never read, never extracted, never
    exported by this script -- not "extracted then discarded", never
    touched at all (see extract.extract_tables' table_names scoping)
  * no MongoDB import anywhere in this script -- it never imports
    pymongo, rcs.migration.load, or rcs.migration.config
  * no write to PostgreSQL -- one REPEATABLE READ snapshot, read-only
  * output files go outside the git repository (refuses to run
    otherwise) and are chmod 0600
  * stdout/the manifest never contain field values -- only filenames,
    byte sizes, document counts, and SHA-256 hashes

Usage:
    .venv/bin/python scripts/migration_manual_sample_export.py --out-dir <path outside repo>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from rcs.migration import extract, tables  # noqa: E402
from rcs.migration.extjson import write_extended_json_array  # noqa: E402
from rcs.migration.logging_safe import log  # noqa: E402
from rcs.migration.transform import transform_table  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402
from rcs.review.repository import get_engine  # noqa: E402

SCOPE_TABLES = ("review_sessions", "processing_jobs", "review_items")


def _assert_outside_repo(out_dir: Path) -> None:
    real_out = out_dir.resolve()
    real_repo = ROOT.resolve()
    if real_out == real_repo or str(real_out).startswith(str(real_repo) + os.sep):
        raise RuntimeError(f"refusing to write export files inside the git repository: {real_out}")


def _select_two_review_items(rows: list[dict]) -> list[dict]:
    """Deterministic: sort by comment_id, take the first row that has at
    least one non-null datetime AND at least one null field (guarantees
    the Date + null criteria by construction), plus the next distinct
    row after it. Boolean (ready_to_send) and integer (row_version /
    source_row_index) criteria are NOT NULL columns -- present on every
    row -- so no special search is needed for those two."""
    ordered = sorted(rows, key=lambda r: r["comment_id"])
    dt_cols = tables.datetime_columns("review_items")
    nullable_cols = {c.name for c in tables.columns_for("review_items")} - tables.not_null_columns("review_items")

    def qualifies(row: dict) -> bool:
        has_date = any(row[c] is not None for c in dt_cols)
        has_null = any(row[c] is None for c in nullable_cols)
        return has_date and has_null

    idx_a = next((i for i, r in enumerate(ordered) if qualifies(r)), None)
    if idx_a is None:
        raise RuntimeError("no review_items row satisfies the required Date+null sample criteria")
    row_a = ordered[idx_a]
    row_b = ordered[idx_a + 1] if idx_a + 1 < len(ordered) else ordered[0 if idx_a != 0 else -1]
    return [row_a, row_b] if row_a["comment_id"] != row_b["comment_id"] else [row_a]


def _select_session_and_job(sessions: list[dict], jobs: list[dict]) -> tuple[dict, dict]:
    session = sorted(sessions, key=lambda r: r["session_id"])[0]
    matching_jobs = [j for j in jobs if j["session_id"] == session["session_id"]]
    job = sorted(matching_jobs, key=lambda r: r["job_id"])[0] if matching_jobs else sorted(jobs, key=lambda r: r["job_id"])[0]
    return session, job


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Output directory -- MUST be outside the git repository.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _assert_outside_repo(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("manual_sample.output_dir_confirmed_outside_repo", out_dir=str(out_dir))

    load_dotenv_if_present()
    engine = get_engine(settings.resolve_database_url(settings.EXCEL_OPERATIONAL_DB))

    log("manual_sample.extract.start", scope_tables=list(SCOPE_TABLES))
    extraction = extract.extract_tables(engine, SCOPE_TABLES)
    log("manual_sample.extract.done", counts=extraction.counts())

    review_items_rows = _select_two_review_items(extraction.rows_by_table["review_items"])
    session_row, job_row = _select_session_and_job(
        extraction.rows_by_table["review_sessions"], extraction.rows_by_table["processing_jobs"]
    )

    ri_result = transform_table("review_items", review_items_rows)
    rs_result = transform_table("review_sessions", [session_row])
    pj_result = transform_table("processing_jobs", [job_row])

    for name, result in (("review_items", ri_result), ("review_sessions", rs_result), ("processing_jobs", pj_result)):
        if result.high_severity_anomalies:
            raise RuntimeError(f"refusing to export -- {name} sample has High-severity anomalies")

    files_written = []
    for name, docs in (
        ("sample_review_items.json", list(ri_result.documents.values())),
        ("sample_review_session.json", list(rs_result.documents.values())),
        ("sample_processing_job.json", list(pj_result.documents.values())),
    ):
        path = out_dir / name
        write_extended_json_array(docs, path)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
        files_written.append((path, len(docs)))

    manifest = {
        "output_dir": str(out_dir),
        "note": "Extended JSON v2 (relaxed mode); users/user_sessions were never read by this run.",
        "files": [
            {
                "filename": path.name,
                "document_count": count,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_of(path),
            }
            for path, count in files_written
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
