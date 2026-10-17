"""
Deterministic controlled-sample selection for the first Mongo staging
write (Phase M3, step 8). No randomness: rows are sorted by id and the
first row matching each criterion is taken, so re-running this against
the same extracted snapshot always yields the same sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_DECIDED_STATUSES = {"APPROVED", "EDITED_AND_APPROVED", "REJECTED", "NO_REPLY", "SENT"}


@dataclass
class ControlledSample:
    review_item_ids: list = field(default_factory=list)
    user_ids: list = field(default_factory=list)
    user_session_tokens: list = field(default_factory=list)
    review_session_ids: list = field(default_factory=list)
    processing_job_ids: list = field(default_factory=list)
    criteria_met: dict = field(default_factory=dict)  # criterion -> comment_id or None


def _first(rows, predicate):
    for row in sorted(rows, key=lambda r: r["comment_id"]):
        if predicate(row):
            return row
    return None


def select_controlled_sample(rows_by_table: dict[str, list[dict]]) -> ControlledSample:
    items = rows_by_table.get("review_items", [])
    sample = ControlledSample()
    picked: dict[str, dict] = {}

    criteria = {
        "assigned_item": lambda r: r["reviewer_user_id"] is not None,
        "unassigned_item": lambda r: r["reviewer_user_id"] is None,
        "reviewed_decided_item": lambda r: r["reviewer_action"] is not None or r["review_status"] in _DECIDED_STATUSES,
        "unprocessed_item_many_nulls": lambda r: r["review_status"] == "UNPROCESSED",
        "dedupe_grouped_item": lambda r: r["content_dedupe_key"] is not None and r["dedupe_canonical_id"] is not None,
        "row_version_gt_1": lambda r: r["row_version"] > 1,
        "row_version_eq_1": lambda r: r["row_version"] == 1,
        "outbound_idempotency_key_set": lambda r: r["outbound_idempotency_key"] is not None,
    }
    for name, predicate in criteria.items():
        row = _first(items, predicate)
        sample.criteria_met[name] = row["comment_id"] if row else None
        if row is not None:
            picked[row["comment_id"]] = row

    # Cover a spread of distinct review_status values (up to 6 distinct).
    seen_statuses: set[str] = set()
    for row in sorted(items, key=lambda r: r["comment_id"]):
        if row["review_status"] not in seen_statuses and len(seen_statuses) < 6:
            seen_statuses.add(row["review_status"])
            picked[row["comment_id"]] = row

    sample.review_item_ids = sorted(picked.keys())

    # One review_session (+ its processing_job, if any) whose source_file
    # matches a picked item, when available -- otherwise just the first
    # of each, deterministically.
    sessions = sorted(rows_by_table.get("review_sessions", []), key=lambda r: r["session_id"])
    picked_source_files = {r["source_file"] for r in picked.values()}
    session = _first(sessions, lambda s: s["source_file"] in picked_source_files) or (sessions[0] if sessions else None)
    if session is not None:
        sample.review_session_ids = [session["session_id"]]
        jobs = sorted(rows_by_table.get("processing_jobs", []), key=lambda j: j["job_id"])
        job = _first(jobs, lambda j: j["session_id"] == session["session_id"])
        if job is not None:
            sample.processing_job_ids = [job["job_id"]]

    # One user (+ one of their user_sessions, if any) -- prefer a user who
    # appears as a reviewer_user_id among the picked items.
    users = sorted(rows_by_table.get("users", []), key=lambda r: r["user_id"])
    assigned_reviewer_ids = {r["reviewer_user_id"] for r in picked.values() if r["reviewer_user_id"]}
    user = _first(users, lambda u: u["user_id"] in assigned_reviewer_ids) or (users[0] if users else None)
    if user is not None:
        sample.user_ids = [user["user_id"]]
        user_sessions = sorted(rows_by_table.get("user_sessions", []), key=lambda s: s["session_token"])
        us = _first(user_sessions, lambda s: s["user_id"] == user["user_id"])
        if us is not None:
            sample.user_session_tokens = [us["session_token"]]

    return sample


def slice_rows_by_table(rows_by_table: dict[str, list[dict]], sample: ControlledSample) -> dict[str, list[dict]]:
    id_fields = {
        "review_items": ("review_item_ids", "comment_id"),
        "users": ("user_ids", "user_id"),
        "user_sessions": ("user_session_tokens", "session_token"),
        "review_sessions": ("review_session_ids", "session_id"),
        "processing_jobs": ("processing_job_ids", "job_id"),
    }
    out: dict[str, list[dict]] = {}
    for table_name, (attr, id_field) in id_fields.items():
        wanted = set(getattr(sample, attr))
        out[table_name] = [r for r in rows_by_table.get(table_name, []) if r[id_field] in wanted]
    return out
