"""
UI reconciliation milestone — classification filtering (primary_intent /
topic_subtype, and reviewer_action for Completed Reviews) is a real,
end-to-end capability: repository contract -> both backend
implementations -> service -> HTTP API -> (frontend, exercised
separately). This file proves the backend half at every layer, plus the
one small ADMIN-only "view a specific reviewer's items" scope override
and the ui_filters.py "completed" filter key added alongside it.

Repository behavioral parity for Mongo is proven against the existing
in-memory fake (see tests/test_mongo_repository_unit.py), never a real
MongoDB connection.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

from rcs.review.auth import create_user, start_session
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem as _ReviewItemModel
from rcs.review.persistence import bundle_from_engine, bundle_from_mongo
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.review.ui_filters import statuses_for_filter
from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake

_NOW = dt.datetime(2026, 1, 15, 12, 0, 0)
_ALL_REVIEW_ITEM_COLUMNS = [c.name for c in _ReviewItemModel.__table__.columns]

_ROWS = [
    {
        "comment_id": "cf-1", "source_file": "f.xlsx", "source_row_index": 0,
        "original_comment": "أقترح إضافة قسم جديد", "review_status": "DRAFT_READY",
        "primary_intent": "SUGGESTION", "topic_subtype": "DICTIONARY_SUGGESTION",
    },
    {
        "comment_id": "cf-2", "source_file": "f.xlsx", "source_row_index": 1,
        "original_comment": "لدي شكوى بخصوص الموقع", "review_status": "DRAFT_READY",
        "primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE",
    },
    {
        "comment_id": "cf-3", "source_file": "f.xlsx", "source_row_index": 2,
        "original_comment": "شكرًا جزيلاً على المعجم", "review_status": "APPROVED",
        "primary_intent": "THANKS_PRAISE", "topic_subtype": "GENERAL",
        "reviewer_action": "SEND_AS_IS", "reviewer_user_id": None,
    },
    {
        "comment_id": "cf-4", "source_file": "f.xlsx", "source_row_index": 3,
        "original_comment": "اقترح تحديث الكلمة", "review_status": "DRAFT_READY",
        "primary_intent": "SUGGESTION", "topic_subtype": "WORD_SUBMISSION",
    },
]


def _seed_postgres(engine, rows=_ROWS):
    for row in rows:
        full = {"ready_to_send": False, "row_version": 1, "ingested_at": _NOW, **row}
        upsert_review_items(engine, [full])


def _seed_mongo(bundle, rows=_ROWS):
    for row in rows:
        doc = {col: None for col in _ALL_REVIEW_ITEM_COLUMNS}
        doc.update({"ready_to_send": False, "row_version": 1, "ingested_at": _NOW})
        doc.update(row)
        doc["_id"] = doc.pop("comment_id")
        bundle.review_items.collection.insert_one(doc)


# --------------------------------------------------------------- service-level (Postgres)


def test_list_review_queue_filters_by_primary_intent(tmp_path):
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    _seed_postgres(engine)
    svc = ReviewWorkflowService(engine)
    page = svc.list_review_queue(statuses=("DRAFT_READY",), primary_intents=("SUGGESTION",))
    assert {i.comment_id for i in page.items} == {"cf-1", "cf-4"}


def test_list_review_queue_filters_by_topic_subtype(tmp_path):
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    _seed_postgres(engine)
    svc = ReviewWorkflowService(engine)
    page = svc.list_review_queue(statuses=("DRAFT_READY",), topic_subtypes=("DICTIONARY_SUGGESTION",))
    assert {i.comment_id for i in page.items} == {"cf-1"}
    # DICTIONARY_SUGGESTION (topic_subtype) must never be conflated with
    # SUGGESTION (primary_intent) -- cf-4 has primary_intent SUGGESTION
    # but a different topic_subtype and must not match this filter.
    assert "cf-4" not in {i.comment_id for i in page.items}


def test_list_review_queue_filters_by_reviewer_action(tmp_path):
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    _seed_postgres(engine)
    svc = ReviewWorkflowService(engine)
    completed_statuses = statuses_for_filter("completed")
    page = svc.list_review_queue(statuses=completed_statuses, reviewer_actions=("SEND_AS_IS",))
    assert {i.comment_id for i in page.items} == {"cf-3"}


def test_list_review_queue_classification_filter_composes_with_search_and_status(tmp_path):
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    _seed_postgres(engine)
    svc = ReviewWorkflowService(engine)
    page = svc.list_review_queue(statuses=("DRAFT_READY",), primary_intents=("SUGGESTION",), search="تحديث")
    assert {i.comment_id for i in page.items} == {"cf-4"}


def test_list_review_queue_unknown_classification_value_matches_nothing_not_an_error(tmp_path):
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    _seed_postgres(engine)
    svc = ReviewWorkflowService(engine)
    page = svc.list_review_queue(statuses=("DRAFT_READY",), primary_intents=("NOT_A_REAL_INTENT",))
    assert page.total == 0
    assert page.items == ()


def test_list_assignable_items_filters_by_classification(tmp_path):
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    _seed_postgres(engine)
    svc = ReviewWorkflowService(engine)
    page = svc.list_assignable_items(primary_intents=("COMPLAINT",))
    assert {i.comment_id for i in page.items} == {"cf-2"}


def test_list_assignable_items_search_matches_comment_text_and_id(tmp_path):
    """Visual reconciliation milestone — the Assignment toolbar's search
    box needs a real filter behind it; reuses list_queue's exact
    substring-match semantics rather than inventing a second one."""
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    _seed_postgres(engine)
    svc = ReviewWorkflowService(engine)
    by_text = svc.list_assignable_items(search="شكوى")
    assert {i.comment_id for i in by_text.items} == {"cf-2"}
    by_id = svc.list_assignable_items(search="cf-4")
    assert {i.comment_id for i in by_id.items} == {"cf-4"}
    no_match = svc.list_assignable_items(search="لا يوجد نص مطابق")
    assert no_match.items == ()


def test_reviewer_isolation_preserved_with_classification_filter(tmp_path):
    """A REVIEWER-scoped queue call composes the classification filter
    with reviewer_user_id -- never bypassing isolation to reach another
    reviewer's (or unassigned) items."""
    engine = get_engine(tmp_path / "cf.sqlite")
    init_db(engine)
    rows = [dict(r) for r in _ROWS]
    rows[0]["reviewer_user_id"] = "rev-a"
    rows[3]["reviewer_user_id"] = "rev-b"  # same primary_intent, different reviewer
    _seed_postgres(engine, rows)
    svc = ReviewWorkflowService(engine)
    page = svc.list_review_queue(statuses=("DRAFT_READY",), primary_intents=("SUGGESTION",), reviewer_user_id="rev-a")
    assert {i.comment_id for i in page.items} == {"cf-1"}  # not cf-4, despite matching intent


# --------------------------------------------------------------- Postgres/Mongo parity


def test_classification_filter_parity_between_postgres_and_mongo(tmp_path):
    pg_engine = get_engine(tmp_path / "cf_pg.sqlite")
    init_db(pg_engine)
    _seed_postgres(pg_engine)
    pg_bundle = bundle_from_engine(pg_engine)

    mongo_bundle = bundle_from_mongo(_FakeDatabase())
    _seed_mongo(mongo_bundle)

    for bundle in (pg_bundle, mongo_bundle):
        svc = ReviewWorkflowService(bundle)
        queue_page = svc.list_review_queue(statuses=("DRAFT_READY",), primary_intents=("SUGGESTION",))
        assert {i.comment_id for i in queue_page.items} == {"cf-1", "cf-4"}

        topic_page = svc.list_review_queue(statuses=("DRAFT_READY",), topic_subtypes=("DICTIONARY_SUGGESTION",))
        assert {i.comment_id for i in topic_page.items} == {"cf-1"}

        assignable_page = svc.list_assignable_items(primary_intents=("COMPLAINT",))
        assert {i.comment_id for i in assignable_page.items} == {"cf-2"}

        completed_page = svc.list_review_queue(statuses=statuses_for_filter("completed"), reviewer_actions=("SEND_AS_IS",))
        assert {i.comment_id for i in completed_page.items} == {"cf-3"}


# --------------------------------------------------------------- ui_filters.py


def test_completed_filter_key_unions_terminal_statuses():
    completed = set(statuses_for_filter("completed"))
    assert completed == {
        "APPROVED", "EDITED_AND_APPROVED", "SENDING", "SENT", "NO_REPLY", "SEND_FAILED",
    }
    # A model recommendation is pending human review, never "completed".
    assert "NO_RESPONSE_RECOMMENDED" not in completed
    assert "DRAFT_READY" not in completed
    assert "IN_REVIEW" not in completed


# --------------------------------------------------------------- HTTP-level (real server, real roles)


def _json_req(url: str, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), err.headers


def _login_cookie(base, email, password):
    _, _, headers = _json_req(f"{base}/api/login", method="POST", data={"email": email, "password": password})
    return headers.get("Set-Cookie").split(";")[0]


def test_http_queue_classification_filter_and_admin_reviewer_override(tmp_path):
    engine = get_engine(tmp_path / "cf_http.sqlite")
    init_db(engine)
    admin = create_user(engine, email="admin@example.test", password="AdminPass123!", display_name="مسؤول", role="ADMIN")
    reviewer_a = create_user(engine, email="rev-a@example.test", password="RevAPass123!", display_name="مراجع أ", role="REVIEWER")
    reviewer_b = create_user(engine, email="rev-b@example.test", password="RevBPass123!", display_name="مراجع ب", role="REVIEWER")

    rows = [dict(r) for r in _ROWS]
    rows[0]["reviewer_user_id"] = reviewer_a.user_id
    rows[3]["reviewer_user_id"] = reviewer_b.user_id
    _seed_postgres(engine, rows)

    server, base = serve_in_thread(engine)
    try:
        admin_cookie = _login_cookie(base, "admin@example.test", "AdminPass123!")
        rev_a_cookie = _login_cookie(base, "rev-a@example.test", "RevAPass123!")

        # ADMIN: classification filter alone, global scope.
        status, page, _ = _json_req(f"{base}/api/queue?filter=pending&primary_intent=SUGGESTION", cookie=admin_cookie)
        assert status == 200
        assert {i["comment_id"] for i in page["items"]} == {"cf-1", "cf-4"}

        # ADMIN: explicit reviewer_user_id override narrows to one reviewer,
        # composed with the classification filter.
        status, page, _ = _json_req(
            f"{base}/api/queue?filter=pending&primary_intent=SUGGESTION&reviewer_user_id={reviewer_a.user_id}",
            cookie=admin_cookie,
        )
        assert status == 200
        assert {i["comment_id"] for i in page["items"]} == {"cf-1"}

        # REVIEWER: the SAME reviewer_user_id query param is silently
        # ignored -- their scope is always their own id, never overridable.
        status, page, _ = _json_req(
            f"{base}/api/queue?filter=pending&primary_intent=SUGGESTION&reviewer_user_id={reviewer_b.user_id}",
            cookie=rev_a_cookie,
        )
        assert status == 200
        assert {i["comment_id"] for i in page["items"]} == {"cf-1"}  # still only their own item, not reviewer_b's

        # Multi-select (comma-separated).
        status, page, _ = _json_req(f"{base}/api/queue?filter=pending&primary_intent=SUGGESTION,COMPLAINT", cookie=admin_cookie)
        assert status == 200
        assert {i["comment_id"] for i in page["items"]} == {"cf-1", "cf-2", "cf-4"}

        # Admin assignable-items endpoint, same filter shape.
        status, page, _ = _json_req(f"{base}/api/admin/assignable-items?primary_intent=COMPLAINT", cookie=admin_cookie)
        assert status == 200
        assert {i["comment_id"] for i in page["items"]} == {"cf-2"}

        # Completed Reviews destination: filter=completed + reviewer_action.
        status, page, _ = _json_req(f"{base}/api/queue?filter=completed&reviewer_action=SEND_AS_IS", cookie=admin_cookie)
        assert status == 200
        assert {i["comment_id"] for i in page["items"]} == {"cf-3"}
    finally:
        server.shutdown()


def test_http_export_accepts_explicit_comment_ids_and_caps_size(tmp_path):
    engine = get_engine(tmp_path / "cf_export.sqlite")
    init_db(engine)
    create_user(engine, email="admin2@example.test", password="AdminPass123!", display_name="مسؤول", role="ADMIN")
    _seed_postgres(engine)
    server, base = serve_in_thread(engine)
    try:
        cookie = _login_cookie(base, "admin2@example.test", "AdminPass123!")
        req = urllib.request.Request(f"{base}/api/excel/export?comment_ids=cf-1,cf-2", headers={"Cookie": cookie})
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            assert len(resp.read()) > 0

        too_many = ",".join(f"id-{i}" for i in range(201))
        status, payload, _ = _json_req(f"{base}/api/excel/export?comment_ids={too_many}", cookie=cookie)
        assert status == 400
        assert payload["error"] == "too_many_ids"
    finally:
        server.shutdown()
