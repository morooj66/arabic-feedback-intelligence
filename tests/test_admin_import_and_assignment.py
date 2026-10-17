"""Phase 8 follow-up — Admin-owned Excel import (gated behind
_require_admin) and the minimal admin assignment-browsing read API
(GET /api/admin/assignable-items, GET /api/admin/source-summary).
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

from rcs.review.auth import ADMIN, REVIEWER, create_user
from rcs.review.excel_review_flow import write_fixture_workbook
from rcs.review.http_api import serve_in_thread
from rcs.review.repository import get_engine, init_db, upsert_review_items


def _json(url: str, *, method="GET", data=None, raw=None, headers=None, cookie=None):
    body = raw if raw is not None else (None if data is None else json.dumps(data).encode("utf-8"))
    hdrs = dict(headers or {})
    if raw is None:
        hdrs.setdefault("Content-Type", "application/json")
    if cookie:
        hdrs["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), err.headers


def _engine(tmp_path, name="admin_import.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _login(base, email, password):
    _, _, headers = _json(f"{base}/api/login", method="POST", data={"email": email, "password": password})
    return headers.get("Set-Cookie").split(";")[0]


def _seed_item(comment_id: str, **overrides) -> dict:
    row = dict(
        comment_id=comment_id,
        source_file="batch.xlsx",
        source_row_index=0,
        original_comment=f"تعليق {comment_id}",
        review_status="DRAFT_READY",
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0),
        ready_to_send=False,
        row_version=1,
    )
    row.update(overrides)
    return row


# --- Part 1: Excel import is ADMIN-owned -----------------------------------


def test_reviewer_gets_403_on_all_excel_import_and_admin_state_endpoints(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="r@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    server, base = serve_in_thread(engine)
    try:
        cookie = _login(base, "r@x.com", "realpassword123")
        assert (
            _json(
                f"{base}/api/excel/upload",
                method="POST",
                raw=b"not-a-real-xlsx",
                headers={"X-Filename": "a.xlsx", "Content-Type": "application/octet-stream"},
                cookie=cookie,
            )[0]
            == 403
        )
        assert _json(f"{base}/api/excel/process", method="POST", data={"upload_token": "x"}, cookie=cookie)[0] == 403
        assert (
            _json(f"{base}/api/excel/sessions", method="POST", data={"upload_token": "x"}, cookie=cookie)[0] == 403
        )
        assert _json(f"{base}/api/excel/sessions/nope/process", method="POST", data={}, cookie=cookie)[0] == 403
        assert _json(f"{base}/api/excel/sessions/nope/resume", method="POST", data={}, cookie=cookie)[0] == 403
        assert _json(f"{base}/api/excel/export", cookie=cookie)[0] == 403
        assert _json(f"{base}/api/excel/sessions/active", cookie=cookie)[0] == 403
        assert _json(f"{base}/api/excel/sessions/nope", cookie=cookie)[0] == 403
        assert _json(f"{base}/api/excel/jobs/nope", cookie=cookie)[0] == 403
    finally:
        server.shutdown()


def test_unauthenticated_still_gets_401_not_403_on_excel_endpoints(tmp_path):
    engine = _engine(tmp_path)
    server, base = serve_in_thread(engine)
    try:
        assert _json(f"{base}/api/excel/sessions/active")[0] == 401
        assert _json(f"{base}/api/excel/upload", method="POST", raw=b"x", headers={"X-Filename": "a.xlsx", "Content-Type": "application/octet-stream"})[0] == 401
    finally:
        server.shutdown()


def test_admin_retains_excel_upload_and_process_access(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    server, base = serve_in_thread(engine)
    try:
        cookie = _login(base, "admin@x.com", "realpassword123")
        fixture = write_fixture_workbook(
            tmp_path / "f.xlsx",
            [
                {
                    "original_comment": "شكرًا جزيلاً على هذا المعجم",
                    "original_title": None,
                    "entry_id": None,
                    "lemma": None,
                    "visitor_name": "زائر",
                    "visitor_email": "v@example.com",
                    "original_type": "رسالة تواصل",
                    "comment_created_at": dt.datetime(2026, 8, 1, 9, 0, 0),
                    "platform": "الرياض",
                }
            ],
        )
        raw = fixture.read_bytes()
        status, upload, _ = _json(
            f"{base}/api/excel/upload",
            method="POST",
            raw=raw,
            headers={"X-Filename": "f.xlsx", "Content-Type": "application/octet-stream"},
            cookie=cookie,
        )
        assert status == 200
        status, session_resp, _ = _json(
            f"{base}/api/excel/sessions",
            method="POST",
            data={"original_filename": "f.xlsx", "upload_token": upload["upload_token"]},
            cookie=cookie,
        )
        assert status == 200
        assert "job_id" in session_resp
        assert _json(f"{base}/api/excel/sessions/active", cookie=cookie)[0] == 200
    finally:
        server.shutdown()


# --- Part 2: admin assignment-browsing read API -----------------------------


def test_reviewer_gets_403_on_assignable_items_and_source_summary(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="r@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    server, base = serve_in_thread(engine)
    try:
        cookie = _login(base, "r@x.com", "realpassword123")
        assert _json(f"{base}/api/admin/assignable-items", cookie=cookie)[0] == 403
        assert _json(f"{base}/api/admin/source-summary", cookie=cookie)[0] == 403
    finally:
        server.shutdown()


def test_admin_sees_only_eligible_undecided_items_and_excludes_decided(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    reviewer = create_user(engine, email="r@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            _seed_item("eligible-1", review_status="DRAFT_READY"),
            _seed_item("eligible-2", review_status="REVIEW_ONLY", reviewer_user_id=reviewer.user_id),
            _seed_item(
                "decided-1",
                review_status="APPROVED",
                reviewer_action="SEND_AS_IS",
                reviewer_user_id=reviewer.user_id,
            ),
            _seed_item("excluded-1", review_status="EXCLUDED"),
            _seed_item("dup-1", review_status="DUPLICATE_CONTENT"),
            _seed_item("unprocessed-1", review_status="UNPROCESSED"),
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        cookie = _login(base, "admin@x.com", "realpassword123")
        status, payload, _ = _json(f"{base}/api/admin/assignable-items?limit=50", cookie=cookie)
        assert status == 200
        ids = {i["comment_id"] for i in payload["items"]}
        assert ids == {"eligible-1", "eligible-2"}
        item = next(i for i in payload["items"] if i["comment_id"] == "eligible-2")
        assert item["reviewer_user_id"] == reviewer.user_id
        assert item["source_file"] == "batch.xlsx"
        assert item["review_status"] == "REVIEW_ONLY"
        assert "visitor_email" not in item
        assert "visitor_name" not in item
        assert "original_comment" not in item  # only the short redacted preview is exposed
    finally:
        server.shutdown()


def test_assignable_items_filters_and_pagination(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    reviewer = create_user(engine, email="r@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            _seed_item("u1", source_file="a.xlsx"),
            _seed_item("u2", source_file="a.xlsx"),
            _seed_item("assigned-1", source_file="a.xlsx", reviewer_user_id=reviewer.user_id),
            _seed_item("other-file-1", source_file="b.xlsx"),
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        cookie = _login(base, "admin@x.com", "realpassword123")

        status, payload, _ = _json(f"{base}/api/admin/assignable-items?source_file=a.xlsx", cookie=cookie)
        assert status == 200
        assert {i["comment_id"] for i in payload["items"]} == {"u1", "u2", "assigned-1"}

        status, payload, _ = _json(f"{base}/api/admin/assignable-items?unassigned_only=true", cookie=cookie)
        assert status == 200
        ids = {i["comment_id"] for i in payload["items"]}
        assert ids == {"u1", "u2", "other-file-1"}
        assert "assigned-1" not in ids

        status, page1, _ = _json(f"{base}/api/admin/assignable-items?limit=2&offset=0", cookie=cookie)
        status2, page2, _ = _json(f"{base}/api/admin/assignable-items?limit=2&offset=2", cookie=cookie)
        assert status == 200 and status2 == 200
        assert page1["total"] == 4
        assert len(page1["items"]) == 2
        assert len(page2["items"]) == 2
        first_ids = {i["comment_id"] for i in page1["items"]}
        second_ids = {i["comment_id"] for i in page2["items"]}
        assert first_ids.isdisjoint(second_ids)
    finally:
        server.shutdown()


def test_source_summary_reflects_assignment_and_completion_state(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    reviewer = create_user(engine, email="r@x.com", password="realpassword123", display_name="R", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            _seed_item("u1", source_file="a.xlsx"),
            _seed_item("assigned-1", source_file="a.xlsx", reviewer_user_id=reviewer.user_id),
            _seed_item(
                "decided-1",
                source_file="a.xlsx",
                review_status="APPROVED",
                reviewer_action="SEND_AS_IS",
                reviewer_user_id=reviewer.user_id,
            ),
            _seed_item("excluded-1", source_file="a.xlsx", review_status="EXCLUDED"),
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        cookie = _login(base, "admin@x.com", "realpassword123")
        status, payload, _ = _json(f"{base}/api/admin/source-summary", cookie=cookie)
        assert status == 200
        row = next(s for s in payload["sources"] if s["source_file"] == "a.xlsx")
        assert row["total_reviewable"] == 3  # excludes excluded-1
        assert row["unassigned"] == 1
        assert row["assigned"] == 2
        assert row["completed"] == 1
    finally:
        server.shutdown()
