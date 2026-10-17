"""
POST-WRITE reconciliation (Phase M3, per the M2 amendment Section C):

    PostgreSQL snapshot (already extracted, in memory)
        <-> MongoDB staging (STAGING_DB_NAME)

Only the DATA-LAYER checks that are meaningful once a Mongo staging copy
exists are implemented here: collection/count parity, aggregation-shaped
parity, assignment/queue-set parity, session/job relationship parity,
status-aware session reconciliation, and sample-level field comparison.

This module deliberately does NOT attempt: Mongo-backed HTTP reviewer
isolation, list_review_queue()/compute_operational_analytics()-equivalent
application logic, or export semantic parity -- none of that has been
reimplemented against Mongo yet, and claiming to verify it here would be
claiming to test code that does not exist (Phase M2 Section J).
"""

from __future__ import annotations

from . import tables
from .logging_safe import short_ref
from .reconcile_pre import CheckResult
from .transform import transform_table

# review_sessions statuses that may legitimately have zero/partial
# matching review_items rows (Phase M1 amendment Correction 3 / M2
# Section F.16) -- job_runner.py sets "uploaded" on retry-after-failure
# and "processing" while a run is in flight, both before rows are
# necessarily attributable yet.
_SESSION_STATUSES_ALLOWING_ZERO_ROWS = {"uploaded", "processing", "failed"}


def _check_collection_counts(rows_by_table: dict, db) -> CheckResult:
    details = []
    all_ok = True
    for table_name in tables.TABLE_ORDER:
        pg_count = len(rows_by_table[table_name])
        mongo_count = db[table_name].count_documents({})
        ok = pg_count == mongo_count
        all_ok = all_ok and ok
        details.append(f"{table_name}: pg={pg_count} mongo={mongo_count} match={ok}")
    return CheckResult("collection_counts", all_ok, "; ".join(details))


def _check_id_set_parity(rows_by_table: dict, db) -> CheckResult:
    all_ok = True
    details = []
    for table_name in tables.TABLE_ORDER:
        id_field = tables.id_field(table_name)
        pg_ids = {r[id_field] for r in rows_by_table[table_name]}
        mongo_ids = {d["_id"] for d in db[table_name].find({}, {"_id": 1})}
        ok = pg_ids == mongo_ids
        all_ok = all_ok and ok
        details.append(f"{table_name}: match={ok} pg_only={len(pg_ids - mongo_ids)} mongo_only={len(mongo_ids - pg_ids)}")
    return CheckResult("id_set_parity_post_write", all_ok, "; ".join(details))


def _check_per_source_and_status_counts(rows_by_table: dict, db) -> CheckResult:
    items = rows_by_table["review_items"]
    pg_by_source: dict[str, int] = {}
    pg_by_status: dict[str, int] = {}
    for r in items:
        pg_by_source[r["source_file"]] = pg_by_source.get(r["source_file"], 0) + 1
        pg_by_status[r["review_status"]] = pg_by_status.get(r["review_status"], 0) + 1

    mongo_by_source = {
        d["_id"]: d["count"]
        for d in db["review_items"].aggregate([{"$group": {"_id": "$source_file", "count": {"$sum": 1}}}])
    }
    mongo_by_status = {
        d["_id"]: d["count"]
        for d in db["review_items"].aggregate([{"$group": {"_id": "$review_status", "count": {"$sum": 1}}}])
    }
    ok = pg_by_source == mongo_by_source and pg_by_status == mongo_by_status
    return CheckResult(
        "per_source_and_status_counts", ok,
        f"source_groups_pg={len(pg_by_source)} source_groups_mongo={len(mongo_by_source)} "
        f"status_groups_pg={len(pg_by_status)} status_groups_mongo={len(mongo_by_status)} match={ok}",
    )


def _check_assignment_parity(rows_by_table: dict, db) -> CheckResult:
    items = rows_by_table["review_items"]
    pg_assignment: dict[str, set] = {}
    for r in items:
        if r["reviewer_user_id"] is not None:
            pg_assignment.setdefault(r["reviewer_user_id"], set()).add(r["comment_id"])

    mongo_assignment: dict[str, set] = {}
    cursor = db["review_items"].find(
        {"reviewer_user_id": {"$ne": None}}, {"_id": 1, "reviewer_user_id": 1}
    )
    for d in cursor:
        mongo_assignment.setdefault(d["reviewer_user_id"], set()).add(d["_id"])

    ok = pg_assignment == mongo_assignment
    return CheckResult(
        "assignment_parity", ok,
        f"reviewers_pg={len(pg_assignment)} reviewers_mongo={len(mongo_assignment)} match={ok}",
    )


def _check_session_job_relationship_parity(rows_by_table: dict, db) -> CheckResult:
    pg_pairs = {(j["job_id"], j["session_id"]) for j in rows_by_table["processing_jobs"]}
    mongo_pairs = {(d["_id"], d["session_id"]) for d in db["processing_jobs"].find({}, {"_id": 1, "session_id": 1})}
    ok = pg_pairs == mongo_pairs
    return CheckResult("session_job_relationship_parity", ok, f"pg_pairs={len(pg_pairs)} mongo_pairs={len(mongo_pairs)} match={ok}")


def _check_status_aware_session_reconciliation(rows_by_table: dict, db) -> CheckResult:
    items_by_source: dict[str, int] = {}
    for r in rows_by_table["review_items"]:
        items_by_source[r["source_file"]] = items_by_source.get(r["source_file"], 0) + 1

    anomalous = []
    for s in rows_by_table["review_sessions"]:
        matching_rows = items_by_source.get(s["source_file"], 0)
        mongo_matching = db["review_items"].count_documents({"source_file": s["source_file"]})
        if s["status"] not in _SESSION_STATUSES_ALLOWING_ZERO_ROWS and matching_rows == 0:
            anomalous.append(f"session_id={s['session_id']} status={s['status']} pg_matching_rows=0")
        if matching_rows != mongo_matching:
            anomalous.append(f"session_id={s['session_id']} pg_rows={matching_rows} mongo_rows={mongo_matching}")

    ok = len(anomalous) == 0
    return CheckResult("status_aware_session_reconciliation", ok, f"anomalous_sessions={len(anomalous)}: {anomalous[:5]}")


def _check_dedupe_and_orphan_parity(rows_by_table: dict, db) -> CheckResult:
    """Confirms the same duplicate-grouping / orphan facts observed
    pre-write (reconcile_pre) still hold true of what actually landed in
    Mongo -- i.e. loading did not introduce or lose any grouping fact."""
    items = rows_by_table["review_items"]
    pg_groups: dict[str, set] = {}
    for r in items:
        if r["content_dedupe_key"] is not None:
            pg_groups.setdefault(r["content_dedupe_key"], set()).add(r["comment_id"])

    mongo_groups: dict[str, set] = {}
    cursor = db["review_items"].find({"content_dedupe_key": {"$ne": None}}, {"_id": 1, "content_dedupe_key": 1})
    for d in cursor:
        mongo_groups.setdefault(d["content_dedupe_key"], set()).add(d["_id"])

    ok = pg_groups == mongo_groups
    return CheckResult("dedupe_grouping_parity", ok, f"pg_groups={len(pg_groups)} mongo_groups={len(mongo_groups)} match={ok}")


def check_sample_field_comparison(rows_by_table: dict, db, table_name: str, sample_ids: list) -> CheckResult:
    id_field = tables.id_field(table_name)
    rows_by_id = {r[id_field]: r for r in rows_by_table[table_name]}
    never_log_id = (table_name, id_field) in tables.NEVER_LOG_FIELDS
    mismatches = []
    for sample_id in sample_ids:
        ref = short_ref(sample_id) if never_log_id else sample_id
        row = rows_by_id.get(sample_id)
        if row is None:
            mismatches.append(f"{ref}: not found in pg snapshot")
            continue
        expected = transform_table(table_name, [row]).documents.get(sample_id)
        actual = db[table_name].find_one({"_id": sample_id})
        if expected is None or actual is None or expected != actual:
            mismatches.append(f"{ref}: field mismatch between transformed doc and mongo actual")
    ok = len(mismatches) == 0
    return CheckResult(f"sample_field_comparison[{table_name}]", ok, f"sampled={len(sample_ids)} mismatches={len(mismatches)}")


def run_post_write_reconciliation(rows_by_table: dict, db, sample_ids_by_table: dict[str, list]) -> list[CheckResult]:
    """Full-dataset post-write reconciliation -- only meaningful after
    the FULL snapshot has been loaded (step 12), since collection_counts/
    id_set_parity compare against the whole PostgreSQL snapshot."""
    checks = [
        _check_collection_counts(rows_by_table, db),
        _check_id_set_parity(rows_by_table, db),
        _check_per_source_and_status_counts(rows_by_table, db),
        _check_assignment_parity(rows_by_table, db),
        _check_session_job_relationship_parity(rows_by_table, db),
        _check_status_aware_session_reconciliation(rows_by_table, db),
        _check_dedupe_and_orphan_parity(rows_by_table, db),
    ]
    for table_name, sample_ids in sample_ids_by_table.items():
        checks.append(check_sample_field_comparison(rows_by_table, db, table_name, sample_ids))
    return checks


def verify_sample_write(rows_by_table: dict, db, sample_ids_by_table: dict[str, list]) -> list[CheckResult]:
    """Scoped verification for step 9 (immediately after the controlled
    sample write) and step 10 (the idempotency re-run) -- deliberately
    does NOT run the full-collection checks above, since only the sample
    exists in staging at that point and a full count/id-set comparison
    against the whole PostgreSQL snapshot would legitimately fail for
    reasons that have nothing to do with a real defect."""
    return [
        check_sample_field_comparison(rows_by_table, db, table_name, sample_ids)
        for table_name, sample_ids in sample_ids_by_table.items()
        if sample_ids
    ]
