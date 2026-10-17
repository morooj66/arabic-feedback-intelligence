"""
PRE-WRITE reconciliation (Phase M3, per the M2 amendment Section C):

    PostgreSQL snapshot (extracted rows)
        <-> transformed intermediate documents

Everything in this module runs entirely on data already extracted into
memory -- it never opens a new PostgreSQL connection and never touches
MongoDB. This is deliberate: it is the set of checks that can be proven
correct before Mongo exists at all. Mongo/application-specific checks
(collection parity, aggregation parity, HTTP-level reviewer isolation,
export parity) belong to POST-WRITE reconciliation (reconcile_post.py)
and are not attempted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import tables
from .logging_safe import short_ref
from .transform import Anomaly, TableTransformResult, transform_table


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class PreWriteReport:
    checks: list[CheckResult] = field(default_factory=list)
    cross_table_anomalies: list[Anomaly] = field(default_factory=list)

    def all_transform_anomalies(self, transform_results: dict[str, TableTransformResult]) -> list[Anomaly]:
        out: list[Anomaly] = []
        for r in transform_results.values():
            out.extend(r.anomalies)
        out.extend(self.cross_table_anomalies)
        return out

    def high_severity_count(self, transform_results: dict[str, TableTransformResult]) -> int:
        return sum(1 for a in self.all_transform_anomalies(transform_results) if a.severity == "high")

    def passed_overall(self, transform_results: dict[str, TableTransformResult]) -> bool:
        return all(c.passed for c in self.checks) and self.high_severity_count(transform_results) == 0


def _check_counts(table_name: str, rows: list[dict], result: TableTransformResult) -> CheckResult:
    extracted = len(rows)
    accounted = len(result.documents) + len(result.blocked_ids)
    passed = extracted == accounted
    return CheckResult(
        f"counts[{table_name}]", passed,
        f"extracted={extracted} transformed={len(result.documents)} blocked={len(result.blocked_ids)} accounted={accounted}",
    )


def _check_id_set_parity(table_name: str, rows: list[dict], result: TableTransformResult) -> CheckResult:
    id_field = tables.id_field(table_name)
    source_ids = {row[id_field] for row in rows}
    covered_ids = set(result.documents.keys()) | result.blocked_ids
    passed = source_ids == covered_ids
    detail = f"source={len(source_ids)} covered={len(covered_ids)}"
    if not passed:
        missing = source_ids - covered_ids
        detail += f" missing_count={len(missing)}"
    return CheckResult(f"id_set_parity[{table_name}]", passed, detail)


def _check_field_preservation(table_name: str, rows: list[dict], result: TableTransformResult) -> CheckResult:
    id_field = tables.id_field(table_name)
    expected_fields = {c.name for c in tables.columns_for(table_name)} - {id_field}
    bad = 0
    for raw_id, doc in result.documents.items():
        doc_fields = set(doc.keys()) - {"_id"}
        if doc_fields != expected_fields:
            bad += 1
    passed = bad == 0
    return CheckResult(f"field_preservation[{table_name}]", passed, f"documents_with_field_mismatch={bad}")


def _wall_clock_tuple(value):
    return (value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond)


def _check_null_and_value_fidelity(table_name: str, rows: list[dict], result: TableTransformResult) -> CheckResult:
    id_field = tables.id_field(table_name)
    dt_cols = tables.datetime_columns(table_name)
    mismatches = 0
    for row in rows:
        raw_id = row[id_field]
        doc = result.documents.get(raw_id)
        if doc is None:
            continue  # blocked row -- covered by counts/id_set_parity checks instead
        for col in tables.columns_for(table_name):
            name = col.name
            if name == id_field:
                continue
            src_val = row[name]
            doc_val = doc[name]
            if src_val is None:
                if doc_val is not None:
                    mismatches += 1
                continue
            if doc_val is None:
                mismatches += 1
                continue
            if name in dt_cols:
                if _wall_clock_tuple(src_val) != _wall_clock_tuple(doc_val):
                    mismatches += 1
            else:
                if src_val != doc_val:
                    mismatches += 1
    passed = mismatches == 0
    return CheckResult(f"null_and_value_fidelity[{table_name}]", passed, f"mismatches={mismatches}")


def _check_determinism(table_name: str, rows: list[dict], result: TableTransformResult) -> CheckResult:
    rerun = transform_table(table_name, rows)
    same_docs = rerun.documents.keys() == result.documents.keys() and all(
        rerun.documents[k] == result.documents[k] for k in result.documents
    )
    same_blocked = rerun.blocked_ids == result.blocked_ids
    passed = same_docs and same_blocked
    return CheckResult(f"deterministic_transform[{table_name}]", passed, f"documents_match={same_docs} blocked_match={same_blocked}")


def _cross_table_orphans(rows_by_table: dict[str, list[dict]]) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    user_ids = {r["user_id"] for r in rows_by_table.get("users", [])}
    session_ids = {r["session_id"] for r in rows_by_table.get("review_sessions", [])}
    comment_ids = {r["comment_id"] for r in rows_by_table.get("review_items", [])}

    for r in rows_by_table.get("review_items", []):
        if r["reviewer_user_id"] is not None and r["reviewer_user_id"] not in user_ids:
            anomalies.append(Anomaly(
                "review_items", r["comment_id"], "reviewer_user_id",
                "orphan_reference", "medium",
                "reviewer_user_id does not resolve to a row in users (pre-existing -- no FK ever enforced this; migrate as-is)",
            ))
        if r["is_exact_duplicate_of"] is not None and r["is_exact_duplicate_of"] not in comment_ids:
            anomalies.append(Anomaly(
                "review_items", r["comment_id"], "is_exact_duplicate_of", "orphan_reference", "medium",
                "is_exact_duplicate_of does not resolve to a comment_id in this snapshot (migrate as-is)",
            ))
        if r["dedupe_canonical_id"] is not None and r["dedupe_canonical_id"] not in comment_ids:
            anomalies.append(Anomaly(
                "review_items", r["comment_id"], "dedupe_canonical_id", "orphan_reference", "medium",
                "dedupe_canonical_id does not resolve to a comment_id in this snapshot (migrate as-is)",
            ))

    for r in rows_by_table.get("user_sessions", []):
        if r["user_id"] not in user_ids:
            anomalies.append(Anomaly(
                "user_sessions", short_ref(r["session_token"]), "user_id", "orphan_reference", "medium",
                "user_sessions.user_id does not resolve to a row in users (migrate as-is)",
            ))

    for r in rows_by_table.get("processing_jobs", []):
        if r["session_id"] not in session_ids:
            anomalies.append(Anomaly(
                "processing_jobs", r["job_id"], "session_id", "orphan_reference", "medium",
                "processing_jobs.session_id does not resolve to a row in review_sessions (migrate as-is)",
            ))
    return anomalies


def _dedupe_grouping_check(rows_by_table: dict[str, list[dict]]) -> CheckResult:
    """content_dedupe_key groups must each have exactly one member whose
    dedupe_canonical_id points to a real, in-group comment_id (itself,
    for a group of one). Reports the fact -- never repairs it."""
    items = rows_by_table.get("review_items", [])
    groups: dict[str, list[dict]] = {}
    for r in items:
        key = r.get("content_dedupe_key")
        if key is not None:
            groups.setdefault(key, []).append(r)

    bad_groups = 0
    for key, members in groups.items():
        member_ids = {m["comment_id"] for m in members}
        canonical_ids = {m["dedupe_canonical_id"] for m in members if m["dedupe_canonical_id"] is not None}
        if not canonical_ids or not canonical_ids.issubset(member_ids):
            bad_groups += 1
    passed = bad_groups == 0
    return CheckResult("dedupe_grouping_consistency", passed, f"groups={len(groups)} inconsistent_groups={bad_groups}")


def _group_counts(rows: list[dict], field_name: str) -> dict:
    counts: dict = {}
    for r in rows:
        counts[r[field_name]] = counts.get(r[field_name], 0) + 1
    return counts


def _check_status_distribution(rows_by_table: dict, transform_results: dict) -> CheckResult:
    """The transform must not distort the review_status distribution --
    grouping the raw extracted rows and the transformed documents by
    review_status must yield identical counts."""
    rows = rows_by_table["review_items"]
    docs = transform_results["review_items"].documents
    raw_counts = _group_counts(rows, "review_status")
    doc_counts = _group_counts(list(docs.values()), "review_status")
    ok = raw_counts == doc_counts
    return CheckResult("status_distribution_preserved", ok, f"statuses_raw={len(raw_counts)} statuses_transformed={len(doc_counts)} match={ok}")


def _check_source_file_distribution(rows_by_table: dict, transform_results: dict) -> CheckResult:
    rows = rows_by_table["review_items"]
    docs = transform_results["review_items"].documents
    raw_counts = _group_counts(rows, "source_file")
    doc_counts = _group_counts(list(docs.values()), "source_file")
    ok = raw_counts == doc_counts
    return CheckResult("source_file_distribution_preserved", ok, f"files_raw={len(raw_counts)} files_transformed={len(doc_counts)} match={ok}")


def _check_prefiltered_junk_excluded_consistency(rows_by_table: dict) -> CheckResult:
    """Real, previously-confirmed invariant (Phase M3 provenance audit):
    every is_prefiltered_junk=True row is EXCLUDED, and vice versa. Not
    assumed here -- re-checked against this fresh snapshot."""
    rows = rows_by_table["review_items"]
    junk_true = sum(1 for r in rows if r["is_prefiltered_junk"] is True)
    excluded = sum(1 for r in rows if r["review_status"] == "EXCLUDED")
    mismatches = sum(1 for r in rows if (r["is_prefiltered_junk"] is True) != (r["review_status"] == "EXCLUDED"))
    ok = mismatches == 0
    return CheckResult(
        "is_prefiltered_junk_excluded_consistency", ok,
        f"junk_true={junk_true} excluded={excluded} row_level_mismatches={mismatches}",
    )


def _check_reviewer_assignment_consistency(rows_by_table: dict, transform_results: dict) -> CheckResult:
    rows = rows_by_table["review_items"]
    docs = transform_results["review_items"].documents
    raw_assignment: dict = {}
    for r in rows:
        if r["reviewer_user_id"] is not None:
            raw_assignment.setdefault(r["reviewer_user_id"], set()).add(r["comment_id"])
    doc_assignment: dict = {}
    for doc in docs.values():
        if doc["reviewer_user_id"] is not None:
            doc_assignment.setdefault(doc["reviewer_user_id"], set()).add(doc["_id"])
    ok = raw_assignment == doc_assignment
    return CheckResult(
        "reviewer_assignment_consistency", ok,
        f"reviewers_raw={len(raw_assignment)} reviewers_transformed={len(doc_assignment)} match={ok}",
    )


def _check_session_job_relationship(rows_by_table: dict) -> CheckResult:
    session_ids = {s["session_id"] for s in rows_by_table["review_sessions"]}
    orphan_jobs = [j["job_id"] for j in rows_by_table["processing_jobs"] if j["session_id"] not in session_ids]
    ok = len(orphan_jobs) == 0
    return CheckResult("session_job_relationship", ok, f"processing_jobs={len(rows_by_table['processing_jobs'])} orphans={len(orphan_jobs)}")


def run_pre_write_reconciliation(
    rows_by_table: dict[str, list[dict]],
    transform_results: dict[str, TableTransformResult],
) -> PreWriteReport:
    """Only checks the tables actually present in rows_by_table -- a
    caller scoped to a subset of tables (e.g. excluding user_sessions)
    gets checks for exactly that subset, never a KeyError for a table it
    deliberately never extracted."""
    report = PreWriteReport()
    present_tables = [t for t in tables.TABLE_ORDER if t in rows_by_table]
    for table_name in present_tables:
        rows = rows_by_table[table_name]
        result = transform_results[table_name]
        report.checks.append(_check_counts(table_name, rows, result))
        report.checks.append(_check_id_set_parity(table_name, rows, result))
        report.checks.append(_check_field_preservation(table_name, rows, result))
        report.checks.append(_check_null_and_value_fidelity(table_name, rows, result))
        report.checks.append(_check_determinism(table_name, rows, result))

    report.cross_table_anomalies.extend(_cross_table_orphans(rows_by_table))
    if "review_items" in rows_by_table:
        report.checks.append(_dedupe_grouping_check(rows_by_table))
        report.checks.append(_check_status_distribution(rows_by_table, transform_results))
        report.checks.append(_check_source_file_distribution(rows_by_table, transform_results))
        report.checks.append(_check_prefiltered_junk_excluded_consistency(rows_by_table))
        report.checks.append(_check_reviewer_assignment_consistency(rows_by_table, transform_results))
    if "processing_jobs" in rows_by_table and "review_sessions" in rows_by_table:
        report.checks.append(_check_session_job_relationship(rows_by_table))
    return report
