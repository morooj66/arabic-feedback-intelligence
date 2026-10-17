"""
Phase M4A — focused regression tests for the extracted PostgreSQL
repository contracts (rcs.review.repositories.contracts/postgres).

These exercise OBSERVABLE behavior through ReviewWorkflowService / the
HTTP API / export, the same way the rest of this test suite does — not
PostgresReviewItemRepository's internals — so they prove the refactor
preserved behavior rather than merely re-describing the new code.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

import pandas as pd
import pytest

from rcs.review.auth import REVIEWER as REVIEWER_ROLE
from rcs.review.auth import create_user
from rcs.review.excel_review_flow import (
    _EXPORT_SHEET1_HEADERS,
    _EXPORT_SHEET1_TITLE,
    _EXPORT_SHEET2_HEADERS,
    _EXPORT_SHEET2_TITLE,
    export_reviewed_workbook,
)
from rcs.review.http_api import serve_in_thread
from rcs.review.identity import ReviewerIdentity
from rcs.review.repositories import postgres as postgres_repo
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.review.sessions import compute_session_counts
from rcs.review.workflow import VersionConflict


def _engine(tmp_path, name="contract.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _item(comment_id: str, *, source_row_index: int = 0, **overrides) -> dict:
    row = dict(
        comment_id=comment_id,
        source_file="contract.xlsx",
        source_row_index=source_row_index,
        original_comment=f"تعليق {comment_id}",
        review_status="DRAFT_READY",
        suggested_final_reply="رد مقترح",
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0),
        ready_to_send=False,
        row_version=1,
    )
    row.update(overrides)
    return row


def _json(url: str, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), err.headers


def _login(base, email, password):
    _, _, headers = _json(f"{base}/api/login", method="POST", data={"email": email, "password": password})
    return headers.get("Set-Cookie").split(";")[0]


# 1-4, 6-7: reviewer isolation / admin access / idempotent assignment /
# decided-item buckets are already covered thoroughly by
# tests/test_reviewer_isolation.py (unchanged, still passing post-
# extraction) — these two are a light, self-contained confirmation that
# the extracted repository still enforces the same isolation gate.
def test_reviewer_cannot_access_another_reviewers_item(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@contract.test", password="realpassword123", display_name="A", role=REVIEWER_ROLE)
    b = create_user(engine, email="b@contract.test", password="realpassword123", display_name="B", role=REVIEWER_ROLE)
    upsert_review_items(engine, [_item("for-b", reviewer_user_id=b.user_id)])
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@contract.test", "realpassword123")
        status, payload, _ = _json(f"{base}/api/items/for-b", cookie=cookie_a)
        assert status == 403
        assert payload["error"] == "forbidden"
    finally:
        server.shutdown()


def test_reviewer_queue_stays_scoped_to_own_assignments(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@contract.test", password="realpassword123", display_name="A", role=REVIEWER_ROLE)
    b = create_user(engine, email="b@contract.test", password="realpassword123", display_name="B", role=REVIEWER_ROLE)
    upsert_review_items(engine, [_item("for-a", reviewer_user_id=a.user_id), _item("for-b", reviewer_user_id=b.user_id)])
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@contract.test", "realpassword123")
        status, page, _ = _json(f"{base}/api/queue?filter=all", cookie=cookie_a)
        assert status == 200
        assert {i["comment_id"] for i in page["items"]} == {"for-a"}
    finally:
        server.shutdown()


# 5: assignment rollback on injected mid-batch failure -- the sharpest
# proof of the "one transaction, whole batch" contract this extraction
# was required to preserve exactly (service.py:assign_reviewer ->
# PostgresReviewItemRepository.assign_batch).
def test_assignment_rolls_back_whole_batch_on_injected_mid_batch_failure(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    reviewer = create_user(engine, email="r@contract.test", password="realpassword123", display_name="R", role=REVIEWER_ROLE)
    upsert_review_items(engine, [_item("c1"), _item("c2"), _item("c3")])
    svc = ReviewWorkflowService(engine)

    call_count = {"n": 0}
    real_add_event = postgres_repo._add_event

    def _add_event_then_fail_on_second(*args, **kwargs):          # the audit event is staged right after each row's compare-and-set UPDATE
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-batch failure")
        real_add_event(*args, **kwargs)

    monkeypatch.setattr("rcs.review.repositories.postgres._add_event", _add_event_then_fail_on_second)

    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        svc.assign_reviewer(["c1", "c2", "c3"], reviewer.user_id)

    # Whole batch rolled back -- including c1, which "succeeded" (its
    # UPDATE and event ran) before the injected failure on c2's _bump.
    for cid in ("c1", "c2", "c3"):
        item = svc.repo.get(cid)
        assert item.reviewer_user_id is None, f"{cid} should not have been persisted after rollback"
        assert item.row_version == 1


def test_assignment_batch_reports_correct_buckets_and_is_idempotent(tmp_path):
    engine = _engine(tmp_path)
    reviewer = create_user(engine, email="r2@contract.test", password="realpassword123", display_name="R2", role=REVIEWER_ROLE)
    upsert_review_items(
        engine,
        [
            _item("eligible-1"),
            _item("already-decided", reviewer_action="SEND_AS_IS", review_status="APPROVED"),
            _item("not-reviewable", review_status="EXCLUDED"),
        ],
    )
    svc = ReviewWorkflowService(engine)
    result = svc.assign_reviewer(["eligible-1", "already-decided", "not-reviewable", "missing"], reviewer.user_id)
    assert result.assigned == ("eligible-1",)
    assert result.skipped_already_decided == ("already-decided",)
    assert result.skipped_not_reviewable == ("not-reviewable",)
    assert result.not_found == ("missing",)

    # Idempotent re-run: same reviewer, same eligible item -> no row_version bump.
    before = svc.repo.get("eligible-1").row_version
    result2 = svc.assign_reviewer(["eligible-1"], reviewer.user_id)
    after = svc.repo.get("eligible-1").row_version
    assert result2.assigned == ("eligible-1",)
    assert before == after


# 8: single-item row_version conflict behavior.
def test_row_version_conflict_on_stale_expected_version(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(engine, [_item("v1")])
    svc = ReviewWorkflowService(engine)
    reviewer = ReviewerIdentity("rev-1", "مراجع")
    svc.claim_review("v1", reviewer, expected_version=1)  # row_version now 2
    with pytest.raises(VersionConflict):
        svc.claim_review("v1", ReviewerIdentity("rev-2", "مراجع آخر"), expected_version=1)


# 9: export ordering/canonical dedupe behavior remains identical.
def test_export_orders_by_source_row_index_and_shows_canonical_for_duplicates(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(
        engine,
        [
            _item("row-high", source_row_index=8, primary_intent="COMPLAINT"),
            _item("row-low", source_row_index=2, primary_intent="THANKS_PRAISE"),
            _item(
                "row-dup",
                source_row_index=5,
                review_status="DUPLICATE_CONTENT",
                dedupe_canonical_id="row-low",
                primary_intent=None,  # never independently classified
            ),
        ],
    )
    out = tmp_path / "export.xlsx"
    export_reviewed_workbook(engine, out)
    sheet1 = pd.read_excel(out, sheet_name=_EXPORT_SHEET1_TITLE, dtype=str)
    sheet2 = pd.read_excel(out, sheet_name=_EXPORT_SHEET2_TITLE, dtype=str)
    assert list(sheet1.columns) == _EXPORT_SHEET1_HEADERS
    assert list(sheet2.columns) == _EXPORT_SHEET2_HEADERS

    # Ordered by source_row_index ascending: row-low(2), row-dup(5), row-high(8).
    assert list(sheet2["comment_id"]) == ["row-low", "row-dup", "row-high"]
    # The duplicate row displays the CANONICAL's classification (primary
    # intent column, sheet1 index 4), not its own (null) value.
    assert sheet1.iloc[1, 4] == "THANKS_PRAISE"
    assert sheet2.iloc[1]["dedupe_canonical_id"] == "row-low"


# 10: session count semantics remain identical (single-round-trip
# conditional-aggregate contract, moved into PostgresReviewSessionRepository).
def test_session_counts_semantics_unchanged(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(
        engine,
        [
            _item("s1", source_file="s.xlsx", review_status="EXCLUDED"),
            _item("s2", source_file="s.xlsx", review_status="DUPLICATE_CONTENT"),
            _item("s3", source_file="s.xlsx", review_status="DRAFT_READY"),
            _item("s4", source_file="s.xlsx", review_status="APPROVED", reviewer_action="SEND_AS_IS"),
            _item("s5", source_file="s.xlsx", review_status="NO_RESPONSE_RECOMMENDED"),
        ],
    )
    counts = compute_session_counts(engine, "s.xlsx")
    assert counts.total == 5
    assert counts.excluded == 1
    assert counts.duplicate_content == 1
    assert counts.canonical_total == 3  # 5 - excluded(1) - duplicate(1)
    assert counts.completed == 1  # reviewer-decided s4 only; s5 is a recommendation, still pending
    assert counts.remaining == 2  # s3 + s5
