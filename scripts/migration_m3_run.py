"""Phase M3 — migration tooling driver: extraction -> transform ->
pre-write reconciliation -> Mongo connectivity check -> controlled
sample write -> sample verification -> idempotency re-run -> (only if
everything above passes) full staging load -> post-write reconciliation.

This is NOT a cutover tool. It never touches the served application,
never modifies PostgreSQL, and only ever writes to the fixed migration
staging database (rcs.migration.config.STAGING_DB_NAME) in MongoDB.

Safety gates (each one, if failed, stops the run before the next step):
  1. Extraction must complete as one PostgreSQL REPEATABLE READ snapshot.
  2. Pre-write reconciliation must report zero High-severity anomalies.
  3. MongoDB connectivity must be confirmed (sanitized diagnostics only).
  4. The controlled sample write must verify field-for-field.
  5. Re-loading the same sample must be a no-op (idempotency).
Only after all five does the full snapshot get loaded, followed by
post-write (data-layer) reconciliation.

Nothing in this script's stdout/stderr/report ever contains a raw
MONGODB_URI, DATABASE_URL, password_hash, session_token, upload_token,
visitor PII, or raw comment/reply text -- see rcs.migration.logging_safe
and rcs.migration.tables.NEVER_LOG_FIELDS.

Usage:
    .venv/bin/python scripts/migration_m3_run.py --report-out <path>
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rcs.migration import config, extract, load, reconcile_post, reconcile_pre, sample as sample_mod, tables  # noqa: E402
from rcs.migration.logging_safe import assert_no_secret_leak, log, short_ref  # noqa: E402
from rcs.migration.transform import transform_all  # noqa: E402
from rcs.review.repository import get_engine  # noqa: E402


def _jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", required=True, help="Path (outside the repo) to write the JSON report to.")
    args = parser.parse_args()

    report: dict = {"phase": "M3", "steps": {}}

    # --- Step: load environment (never printed) ---------------------------
    try:
        env = config.load_migration_env()
    except config.MissingEnvError as exc:
        report["steps"]["environment"] = {"ok": False, "error": str(exc)}
        _finish(report, args.report_out)
        return 2
    report["steps"]["environment"] = {"ok": True, "staging_db_name": env.staging_db_name}
    log("m3.environment_loaded", staging_db_name=env.staging_db_name)

    engine = get_engine(env.postgres_url)

    # --- Step 3: extraction (one snapshot) ---------------------------------
    log("m3.extract.start")
    extraction = extract.extract_all(engine)
    report["steps"]["extraction"] = {
        "ok": True,
        "counts": extraction.counts(),
        "started_at": extraction.started_at.isoformat(),
        "finished_at": extraction.finished_at.isoformat(),
        "drift_signals_informational": _jsonable(extraction.drift_signals),
    }
    log("m3.extract.done", counts=extraction.counts())

    # --- Step 4: transform --------------------------------------------------
    log("m3.transform.start")
    transform_results = transform_all(extraction.rows_by_table)
    report["steps"]["transform"] = {
        table_name: {
            "documents": len(r.documents),
            "blocked": len(r.blocked_ids),
            "anomalies_high": len(r.high_severity_anomalies),
            "anomalies_total": len(r.anomalies),
        }
        for table_name, r in transform_results.items()
    }
    log("m3.transform.done")

    # --- Step 5: pre-write reconciliation -----------------------------------
    log("m3.reconcile_pre.start")
    pre_report = reconcile_pre.run_pre_write_reconciliation(extraction.rows_by_table, transform_results)
    high_count = pre_report.high_severity_count(transform_results)
    all_anomalies = pre_report.all_transform_anomalies(transform_results)
    report["steps"]["pre_write_reconciliation"] = {
        "checks": [_jsonable(c) for c in pre_report.checks],
        "high_severity_anomaly_count": high_count,
        "medium_severity_anomaly_count": len(all_anomalies) - high_count,
        "anomalies_sample": [_jsonable(a) for a in all_anomalies[:50]],
    }
    checks_passed = all(c.passed for c in pre_report.checks)
    log("m3.reconcile_pre.done", checks_passed=checks_passed, high_severity_anomaly_count=high_count)

    if high_count > 0 or not checks_passed:
        report["stopped_at"] = "pre_write_reconciliation"
        report["reason"] = "High-severity anomaly or failed check present -- STOP before any MongoDB write."
        _finish(report, args.report_out)
        return 3

    # --- Step 6: MongoDB connectivity (sanitized) ---------------------------
    log("m3.connectivity.start")
    connectivity = load.check_connectivity(env.mongodb_uri)
    report["steps"]["connectivity"] = _jsonable(connectivity)
    log("m3.connectivity.done", ok=connectivity.ok, error_type=connectivity.error_type)

    if not connectivity.ok:
        report["stopped_at"] = "connectivity"
        report["reason"] = "MongoDB connectivity could not be confirmed -- STOP before any write."
        _finish(report, args.report_out)
        return 4

    report["steps"]["staging_database_name"] = env.staging_db_name  # reported per step 7 -- never the URI

    client, db = load.get_staging_db(env.mongodb_uri)
    try:
        # --- Step 7b: refuse to reuse/clear a pre-existing target -----------
        # Never assumed empty just because the name is new to us: list
        # what's actually there first, and STOP (no drop/clear/overwrite)
        # if any of our five target collection names already exist.
        conflict_check = load.check_no_pre_existing_targets(db)
        report["steps"]["pre_existing_target_check"] = _jsonable(conflict_check)
        log(
            "m3.pre_existing_target_check",
            safe_to_write=conflict_check.safe_to_write,
            existing_collections=conflict_check.existing_collections,
        )
        if not conflict_check.safe_to_write:
            report["stopped_at"] = "pre_existing_target_check"
            report["reason"] = (
                "One or more target collection names already exist in the staging database "
                f"({conflict_check.conflicting_target_names}) -- STOP. Refusing to reuse, clear, "
                "or overwrite a pre-existing database/collection."
            )
            _finish(report, args.report_out)
            return 7

        # --- Step 8: controlled sample selection + first write -------------
        chosen_sample = sample_mod.select_controlled_sample(extraction.rows_by_table)
        sample_ids_by_table = {
            "review_items": chosen_sample.review_item_ids,
            "users": chosen_sample.user_ids,
            "user_sessions": chosen_sample.user_session_tokens,
            "review_sessions": chosen_sample.review_session_ids,
            "processing_jobs": chosen_sample.processing_job_ids,
        }
        report["steps"]["controlled_sample"] = {
            "criteria_met": chosen_sample.criteria_met,
            "counts": {t: len(ids) for t, ids in sample_ids_by_table.items()},
        }
        log("m3.sample.selected", counts={t: len(ids) for t, ids in sample_ids_by_table.items()})

        sample_load_results = {}
        for table_name, ids in sample_ids_by_table.items():
            docs = [transform_results[table_name].documents[i] for i in ids if i in transform_results[table_name].documents]
            result = load.load_documents(db, table_name, docs)
            sample_load_results[table_name] = _jsonable(result)
        report["steps"]["sample_write"] = sample_load_results
        log("m3.sample.written")

        # --- Step 9: verify the sample immediately --------------------------
        verify_checks = reconcile_post.verify_sample_write(extraction.rows_by_table, db, sample_ids_by_table)
        report["steps"]["sample_verification"] = [_jsonable(c) for c in verify_checks]
        sample_verified = all(c.passed for c in verify_checks)
        log("m3.sample.verified", passed=sample_verified)

        if not sample_verified:
            report["stopped_at"] = "sample_verification"
            report["reason"] = "Controlled sample did not verify field-for-field -- STOP before idempotency/full load."
            _finish(report, args.report_out)
            return 5

        # --- Step 10: idempotency verification (re-run the same sample) -----
        idempotency_load_results = {}
        for table_name, ids in sample_ids_by_table.items():
            docs = [transform_results[table_name].documents[i] for i in ids if i in transform_results[table_name].documents]
            result = load.load_documents(db, table_name, docs)
            idempotency_load_results[table_name] = _jsonable(result)
        idempotency_verify_checks = reconcile_post.verify_sample_write(extraction.rows_by_table, db, sample_ids_by_table)
        idempotency_passed = all(c.passed for c in idempotency_verify_checks) and all(
            r["upserted"] == 0 for r in idempotency_load_results.values()
        )
        report["steps"]["idempotency_rerun"] = {
            "load_results": idempotency_load_results,
            "verify_checks": [_jsonable(c) for c in idempotency_verify_checks],
            "passed": idempotency_passed,
        }
        log("m3.idempotency.done", passed=idempotency_passed)

        if not idempotency_passed:
            report["stopped_at"] = "idempotency_rerun"
            report["reason"] = "Re-running the sample load was not a clean no-op -- STOP before full load."
            _finish(report, args.report_out)
            return 6

        # --- Step 11: full staging load --------------------------------------
        full_load_results = {}
        for table_name in tables.TABLE_ORDER:
            docs = list(transform_results[table_name].documents.values())
            result = load.load_documents(db, table_name, docs)
            full_load_results[table_name] = _jsonable(result)
        report["steps"]["full_staging_load"] = full_load_results
        log("m3.full_load.done", counts={t: r["attempted"] for t, r in full_load_results.items()})

        # --- Step 12: post-write (data-layer) reconciliation ------------------
        post_checks = reconcile_post.run_post_write_reconciliation(
            extraction.rows_by_table, db, sample_ids_by_table
        )
        report["steps"]["post_write_reconciliation"] = [_jsonable(c) for c in post_checks]
        post_passed = all(c.passed for c in post_checks)
        log("m3.post_write.done", passed=post_passed)
        report["post_write_passed"] = post_passed
    finally:
        client.close()

    report["completed"] = True
    _finish(report, args.report_out)
    return 0


def _finish(report: dict, report_out: str) -> None:
    text = json.dumps(_jsonable(report), ensure_ascii=False, indent=2, default=str)
    assert_no_secret_leak(text)
    Path(report_out).write_text(text, encoding="utf-8")
    log("m3.report_written", path=report_out)


if __name__ == "__main__":
    raise SystemExit(main())
