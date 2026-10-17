"""Phase M-MANUAL-REAL — generate the verified real-data Atlas manual
staging import package for review_items / review_sessions /
processing_jobs / users.

Safety, by construction:
  * user_sessions is never read, never extracted, never in scope here.
  * No MongoDB connection anywhere in this script or anything it
    imports (does not import pymongo, rcs.migration.load, or
    rcs.migration.config).
  * One PostgreSQL REPEATABLE READ snapshot, one transaction, over
    exactly users/review_sessions/processing_jobs/review_items. If the
    connection/transaction is lost, extract_tables() raises and this
    script does not attempt a partial resume -- rerun it from scratch.
  * Re-confirms the extracted counts against the expected counts from
    the last verified audit before generating anything; on any
    mismatch, STOPS and reports the difference without writing a
    package.
  * Runs the full pre-write reconciliation (counts, ID-set parity,
    field preservation, null/type fidelity, determinism, dedupe
    grouping, status/source_file distribution, prefiltered-junk/
    EXCLUDED consistency, reviewer-assignment consistency, session/job
    relationship) and STOPS on any High or Medium anomaly -- stricter
    than earlier phases, per this run's explicit instruction.
  * Output must be outside the git repository (asserted); every
    real-data file is chmod 0600; nothing is printed to stdout except
    the final sanitized manifest (no PII, no secrets, no raw text).

Usage:
    .venv/bin/python scripts/migration_manual_real_package.py --out-dir <path outside repo>
"""

from __future__ import annotations

import argparse
import datetime as dt
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
from rcs.migration import extract, reconcile_pre, tables  # noqa: E402
from rcs.migration.extjson import write_extended_json_array  # noqa: E402
from rcs.migration.logging_safe import log  # noqa: E402
from rcs.migration.transform import transform_all  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402
from rcs.review.repository import get_engine  # noqa: E402

SCOPE_TABLES = ("users", "review_sessions", "processing_jobs", "review_items")
EXPECTED_COUNTS = {"review_items": 7443, "users": 4, "review_sessions": 2, "processing_jobs": 2}
BATCH_SIZE = 500  # matches extract.CHUNK_SIZE for consistency


def _assert_outside_repo(out_dir: Path) -> None:
    real_out = out_dir.resolve()
    real_repo = ROOT.resolve()
    if real_out == real_repo or str(real_out).startswith(str(real_repo) + os.sep):
        raise RuntimeError(f"refusing to write export files inside the git repository: {real_out}")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_of_sorted_ids(ids) -> str:
    joined = "\n".join(sorted(str(i) for i in ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _chmod_600(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Output directory -- MUST be outside the git repository.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _assert_outside_repo(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv_if_present()
    engine = get_engine(settings.resolve_database_url(settings.EXCEL_OPERATIONAL_DB))

    log("manual_real.extract.start", scope_tables=list(SCOPE_TABLES))
    extraction = extract.extract_tables(engine, SCOPE_TABLES)
    counts = extraction.counts()
    log("manual_real.extract.done", counts=counts)

    # --- Re-confirm counts BEFORE generating anything -----------------------
    diffs = {t: {"expected": EXPECTED_COUNTS[t], "actual": counts[t]} for t in EXPECTED_COUNTS if counts[t] != EXPECTED_COUNTS[t]}
    if diffs:
        print(json.dumps({"stopped_at": "count_reconfirmation", "diffs": diffs, "full_counts": counts}, indent=2))
        return 2

    # --- Transform + PRE-WRITE reconciliation -------------------------------
    transform_results = transform_all(extraction.rows_by_table)
    pre_report = reconcile_pre.run_pre_write_reconciliation(extraction.rows_by_table, transform_results)
    all_anomalies = pre_report.all_transform_anomalies(transform_results)
    checks_passed = all(c.passed for c in pre_report.checks)
    high = sum(1 for a in all_anomalies if a.severity == "high")
    medium = sum(1 for a in all_anomalies if a.severity == "medium")

    # This run's explicit rule: ANY anomaly (High or Medium) or failed check = STOP.
    if not checks_passed or high > 0 or medium > 0:
        print(json.dumps({
            "stopped_at": "pre_write_reconciliation",
            "checks_passed": checks_passed,
            "failed_checks": [c.name for c in pre_report.checks if not c.passed],
            "high_severity_anomaly_count": high,
            "medium_severity_anomaly_count": medium,
            "anomaly_sample": [
                {"table": a.table, "field": a.field_name, "kind": a.kind, "severity": a.severity}
                for a in all_anomalies[:20]
            ],
        }, indent=2))
        return 3

    # --- Package generation --------------------------------------------------
    snapshot_ts = extraction.started_at.isoformat() + "Z"
    files_meta = []

    def _write_collection_file(table_name: str, docs: list[dict], filename: str):
        path = out_dir / filename
        write_extended_json_array(docs, path)
        _chmod_600(path)
        ids_sorted = sorted(d["_id"] for d in docs)
        files_meta.append({
            "filename": filename,
            "table": table_name,
            "document_count": len(docs),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_of_file(path),
            "first_id": ids_sorted[0] if ids_sorted else None,
            "last_id": ids_sorted[-1] if ids_sorted else None,
        })

    # review_items: numbered batches, ordered by _id, BATCH_SIZE per file.
    review_item_docs = sorted(transform_results["review_items"].documents.values(), key=lambda d: d["_id"])
    review_items_batches = [review_item_docs[i:i + BATCH_SIZE] for i in range(0, len(review_item_docs), BATCH_SIZE)]
    for idx, batch in enumerate(review_items_batches, start=1):
        _write_collection_file("review_items", batch, f"review_items_{idx:03d}.json")

    _write_collection_file("review_sessions", list(transform_results["review_sessions"].documents.values()), "review_sessions.json")
    _write_collection_file("processing_jobs", list(transform_results["processing_jobs"].documents.values()), "processing_jobs.json")
    _write_collection_file("users", list(transform_results["users"].documents.values()), "users.json")

    overall_id_hashes = {
        table_name: _sha256_of_sorted_ids(transform_results[table_name].documents.keys())
        for table_name in SCOPE_TABLES
    }

    manifest = {
        "source_snapshot_timestamp_utc": snapshot_ts,
        "collections": list(SCOPE_TABLES),
        "expected_document_counts": EXPECTED_COUNTS,
        "actual_document_counts": counts,
        "sensitive_field_treatment": {
            "review_sessions.upload_token": "preserved (required for semantic staging parity: NOT NULL column); "
                                             "never printed/logged by this tooling; file is chmod 0600 and must "
                                             "not be committed or shared.",
            "users.password_hash": "preserved byte-for-byte (required for staging parity); never printed/logged "
                                    "by this tooling; file is chmod 0600 and must not be committed or shared.",
            "user_sessions": "excluded entirely -- not read, not extracted, not exported (session_token is a "
                              "live bearer credential).",
        },
        "pre_write_reconciliation": {
            "checks_passed": checks_passed,
            "check_count": len(pre_report.checks),
            "high_severity_anomaly_count": high,
            "medium_severity_anomaly_count": medium,
        },
        "batches": files_meta,
        "overall_sorted_id_sha256_per_collection": overall_id_hashes,
    }
    manifest_path = out_dir / "migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _chmod_600(manifest_path)

    print(json.dumps({
        "package_ready": True,
        "output_dir": str(out_dir),
        "manifest_file": "migration_manifest.json",
        "manifest_sha256": _sha256_of_file(manifest_path),
        "reconciliation": manifest["pre_write_reconciliation"],
        "files": files_meta,
        "overall_sorted_id_sha256_per_collection": overall_id_hashes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
