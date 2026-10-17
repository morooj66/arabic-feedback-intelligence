"""Phase 8 — reviewer isolation and admin assignment (backend foundation).

Reuses ReviewItem.reviewer_user_id (already migrated in Phase 6) as the
single ownership/assignment column — no new table, no new migration. See
http_api.py::_require_item_access and service.py::assign_reviewer /
reviewer_summaries.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

from sqlalchemy.orm import Session

from rcs.review.auth import ADMIN, REVIEWER, create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db, upsert_review_items


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


def _engine(tmp_path, name="isolation.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _login(base, email, password):
    _, _, headers = _json(f"{base}/api/login", method="POST", data={"email": email, "password": password})
    return headers.get("Set-Cookie").split(";")[0]


def _item(comment_id: str, *, reviewer_user_id: str | None = None, **overrides) -> dict:
    row = dict(
        comment_id=comment_id,
        source_file="isolation.xlsx",
        source_row_index=0,
        original_comment=f"تعليق {comment_id}",
        review_status="DRAFT_READY",
        suggested_final_reply="رد مقترح",
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0),
        ready_to_send=False,
        row_version=1,
        reviewer_user_id=reviewer_user_id,
    )
    row.update(overrides)
    return row


def test_reviewer_a_cannot_see_reviewer_b_queue_items(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(engine, [_item("for-a", reviewer_user_id=a.user_id), _item("for-b", reviewer_user_id=b.user_id)])
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@x.com", "realpassword123")
        status, page, _ = _json(f"{base}/api/queue?filter=all", cookie=cookie_a)
        assert status == 200
        ids = {i["comment_id"] for i in page["items"]}
        assert ids == {"for-a"}
    finally:
        server.shutdown()


def test_reviewer_a_cannot_fetch_reviewer_b_item_detail(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(engine, [_item("for-b", reviewer_user_id=b.user_id)])
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@x.com", "realpassword123")
        status, payload, _ = _json(f"{base}/api/items/for-b", cookie=cookie_a)
        assert status == 403
        assert payload["error"] == "forbidden"
    finally:
        server.shutdown()


def test_reviewer_a_cannot_mutate_reviewer_b_item(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(engine, [_item("for-b", reviewer_user_id=b.user_id)])
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@x.com", "realpassword123")
        for action in ("claim", "send-as-is"):
            status, payload, _ = _json(
                f"{base}/api/items/for-b/{action}", method="POST", data={"expected_version": 1}, cookie=cookie_a
            )
            assert status == 403, action
            assert payload["error"] == "forbidden"
        status, payload, _ = _json(
            f"{base}/api/items/for-b/edit-and-send",
            method="POST",
            data={"expected_version": 1, "edited_response": "محاولة تعديل"},
            cookie=cookie_a,
        )
        assert status == 403
        status, payload, _ = _json(
            f"{base}/api/items/for-b/no-response",
            method="POST",
            data={"expected_version": 1, "reason": "x"},
            cookie=cookie_a,
        )
        assert status == 403
    finally:
        server.shutdown()

    # And the item itself was never touched by A's rejected attempts.
    with Session(engine) as db:
        row = db.get(ReviewItem, "for-b")
        assert row.reviewer_user_id == b.user_id
        assert row.review_status == "DRAFT_READY"
        assert row.reviewer_action is None


def test_unassigned_items_are_invisible_to_reviewers(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    upsert_review_items(engine, [_item("unassigned-1"), _item("assigned-to-a", reviewer_user_id=a.user_id)])
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@x.com", "realpassword123")
        status, page, _ = _json(f"{base}/api/queue?filter=all", cookie=cookie_a)
        assert status == 200
        ids = {i["comment_id"] for i in page["items"]}
        assert ids == {"assigned-to-a"}
        assert "unassigned-1" not in ids
        # Direct enumeration by ID is rejected identically to "someone
        # else's item" — a REVIEWER can't distinguish the two cases.
        status, payload, _ = _json(f"{base}/api/items/unassigned-1", cookie=cookie_a)
        assert status == 403
        assert payload["error"] == "forbidden"
    finally:
        server.shutdown()


def test_admin_sees_global_items_regardless_of_assignment(tmp_path):
    engine = _engine(tmp_path)
    admin = create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    upsert_review_items(engine, [_item("unassigned-1"), _item("assigned-to-a", reviewer_user_id=a.user_id)])
    server, base = serve_in_thread(engine)
    try:
        cookie_admin = _login(base, "admin@x.com", "realpassword123")
        status, page, _ = _json(f"{base}/api/queue?filter=all", cookie=cookie_admin)
        assert status == 200
        ids = {i["comment_id"] for i in page["items"]}
        assert ids == {"unassigned-1", "assigned-to-a"}
        status, detail, _ = _json(f"{base}/api/items/unassigned-1", cookie=cookie_admin)
        assert status == 200
        status, detail, _ = _json(f"{base}/api/items/assigned-to-a", cookie=cookie_admin)
        assert status == 200
    finally:
        server.shutdown()


def test_admin_can_assign_unassigned_eligible_items(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    upsert_review_items(engine, [_item("u1"), _item("u2")])
    server, base = serve_in_thread(engine)
    try:
        cookie_admin = _login(base, "admin@x.com", "realpassword123")
        status, payload, _ = _json(
            f"{base}/api/admin/assign",
            method="POST",
            data={"comment_ids": ["u1", "u2"], "reviewer_user_id": a.user_id},
            cookie=cookie_admin,
        )
        assert status == 200
        assert set(payload["assigned"]) == {"u1", "u2"}
        assert payload["skipped_already_decided"] == []
        assert payload["not_found"] == []

        cookie_a = _login(base, "a@x.com", "realpassword123")
        status, page, _ = _json(f"{base}/api/queue?filter=all", cookie=cookie_a)
        assert {i["comment_id"] for i in page["items"]} == {"u1", "u2"}

        status, summary, _ = _json(f"{base}/api/admin/reviewers", cookie=cookie_admin)
        assert status == 200
        row = next(r for r in summary["reviewers"] if r["user_id"] == a.user_id)
        assert row["assigned_total"] == 2
        assert row["pending"] == 2
        assert row["completed"] == 0
    finally:
        server.shutdown()


def test_repeated_same_assignment_is_idempotent(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    upsert_review_items(engine, [_item("u1")])
    server, base = serve_in_thread(engine)
    try:
        cookie_admin = _login(base, "admin@x.com", "realpassword123")
        first = _json(
            f"{base}/api/admin/assign",
            method="POST",
            data={"comment_ids": ["u1"], "reviewer_user_id": a.user_id},
            cookie=cookie_admin,
        )[1]
        assert first["assigned"] == ["u1"]
        with Session(engine) as db:
            version_after_first = db.get(ReviewItem, "u1").row_version

        second = _json(
            f"{base}/api/admin/assign",
            method="POST",
            data={"comment_ids": ["u1"], "reviewer_user_id": a.user_id},
            cookie=cookie_admin,
        )[1]
        assert second["assigned"] == ["u1"]  # still reported as a successful (no-op) assignment
        with Session(engine) as db:
            row = db.get(ReviewItem, "u1")
            assert row.reviewer_user_id == a.user_id
            assert row.row_version == version_after_first  # unchanged — genuinely a no-op, not a silent rewrite
    finally:
        server.shutdown()


def test_decided_items_cannot_be_silently_reassigned(tmp_path):
    engine = _engine(tmp_path)
    create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            _item(
                "decided-1",
                reviewer_user_id=a.user_id,
                review_status="APPROVED",
                reviewer_action="SEND_AS_IS",
                final_reply="رد نهائي",
            )
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        cookie_admin = _login(base, "admin@x.com", "realpassword123")
        status, payload, _ = _json(
            f"{base}/api/admin/assign",
            method="POST",
            data={"comment_ids": ["decided-1"], "reviewer_user_id": b.user_id},
            cookie=cookie_admin,
        )
        assert status == 200
        assert payload["assigned"] == []
        assert payload["skipped_already_decided"] == ["decided-1"]
    finally:
        server.shutdown()

    with Session(engine) as db:
        row = db.get(ReviewItem, "decided-1")
        assert row.reviewer_user_id == a.user_id  # ownership of the decided item was never overwritten


# --- supporting coverage: the new admin endpoints are actually admin-only ---


def test_admin_assign_endpoint_rejects_reviewer_role(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(engine, [_item("u1")])
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@x.com", "realpassword123")
        status, payload, _ = _json(
            f"{base}/api/admin/assign",
            method="POST",
            data={"comment_ids": ["u1"], "reviewer_user_id": b.user_id},
            cookie=cookie_a,
        )
        assert status == 403
        assert payload["error"] == "forbidden"
    finally:
        server.shutdown()

    with Session(engine) as db:
        assert db.get(ReviewItem, "u1").reviewer_user_id is None


def test_admin_assign_rejects_non_reviewer_target(tmp_path):
    engine = _engine(tmp_path)
    admin = create_user(engine, email="admin@x.com", password="realpassword123", display_name="Admin", role=ADMIN)
    other_admin = create_user(engine, email="admin2@x.com", password="realpassword123", display_name="Admin2", role=ADMIN)
    upsert_review_items(engine, [_item("u1")])
    server, base = serve_in_thread(engine)
    try:
        cookie_admin = _login(base, "admin@x.com", "realpassword123")
        status, payload, _ = _json(
            f"{base}/api/admin/assign",
            method="POST",
            data={"comment_ids": ["u1"], "reviewer_user_id": other_admin.user_id},
            cookie=cookie_admin,
        )
        assert status == 400
        assert payload["error"] == "invalid_reviewer"
        status, payload, _ = _json(
            f"{base}/api/admin/assign",
            method="POST",
            data={"comment_ids": ["u1"], "reviewer_user_id": "does-not-exist"},
            cookie=cookie_admin,
        )
        assert status == 400
        assert payload["error"] == "invalid_reviewer"
    finally:
        server.shutdown()


def test_reviewer_dashboard_metrics_are_scoped_to_own_assignments(tmp_path):
    engine = _engine(tmp_path)
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            _item("for-a-1", reviewer_user_id=a.user_id),
            _item("for-a-2", reviewer_user_id=a.user_id),
            _item("for-b-1", reviewer_user_id=b.user_id),
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "a@x.com", "realpassword123")
        status, counts, _ = _json(f"{base}/api/counts", cookie=cookie_a)
        assert status == 200
        assert counts["pending"] == 2  # only A's own two items, not B's

        status, analytics, _ = _json(
            f"{base}/api/analytics?range=week&source_file=isolation.xlsx", cookie=cookie_a
        )
        assert status == 200
        assert analytics["kpis"]["pending"] == 2
    finally:
        server.shutdown()


def test_assigned_no_response_recommended_item_is_scoped_correctly_across_queue_filters(tmp_path):
    """Regression for a reported "items assigned by Admin are not
    appearing in the reviewer's Review Queue" investigation: live tracing
    proved every assigned item is served correctly, and the only reason a
    NO_RESPONSE_RECOMMENDED-status item doesn't show under the default
    "بانتظار المراجعة" (pending) tab is that it's a different, deliberate
    status bucket (see service.py's QUEUE_PENDING_STATUSES / ui_filters.py)
    — it correctly appears under "لا يحتاج إلى رد" (no_reply) and "الكل"
    (all) instead. No existing test exercised a reviewer-scoped queue with
    a NO_RESPONSE_RECOMMENDED row before this one — every prior isolation
    test used DRAFT_READY. Locks in the current, correct behavior so a
    future change can't silently regress it either way (hiding it
    everywhere, or leaking it to another reviewer)."""
    engine = _engine(tmp_path)
    a = create_user(engine, email="nrr-a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="nrr-b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            _item("nrr-for-a", reviewer_user_id=a.user_id, review_status="NO_RESPONSE_RECOMMENDED"),
            _item("pending-for-a", reviewer_user_id=a.user_id, review_status="DRAFT_READY"),
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        cookie_a = _login(base, "nrr-a@x.com", "realpassword123")
        cookie_b = _login(base, "nrr-b@x.com", "realpassword123")

        # /api/counts: a recommendation is pending human work, not a decision.
        status, counts, _ = _json(f"{base}/api/counts", cookie=cookie_a)
        assert status == 200
        assert counts["all"] == 2
        assert counts["pending"] == 2
        assert counts["no_reply"] == 0

        # Default (pending) tab: both the DRAFT_READY and the NRR item.
        status, page, _ = _json(f"{base}/api/queue?filter=pending", cookie=cookie_a)
        assert status == 200
        ids = {i["comment_id"] for i in page["items"]}
        assert ids == {"pending-for-a", "nrr-for-a"}

        # "الكل" shows it too; the final-decision "لا يحتاج إلى رد" tab does not.
        status, page, _ = _json(f"{base}/api/queue?filter=all", cookie=cookie_a)
        assert "nrr-for-a" in {i["comment_id"] for i in page["items"]}
        status, page, _ = _json(f"{base}/api/queue?filter=no_reply", cookie=cookie_a)
        assert page["items"] == []

        # Override: reviewer B (not assigned) is forbidden; A may reply.
        status, _, _ = _json(f"{base}/api/items/nrr-for-a/edit-and-send", cookie=cookie_b, method="POST",
                             data={"edited_response": "x", "expected_version": 1})
        assert status == 403
        status, detail, _ = _json(f"{base}/api/items/nrr-for-a/edit-and-send", cookie=cookie_a, method="POST",
                                  data={"edited_response": "رد المراجع", "expected_version": 1})
        assert status == 200
        assert detail["review_status"] == "EDITED_AND_APPROVED"

        # Isolation: reviewer B never sees A's items under any tab.
        for filt in ("pending", "no_reply", "all"):
            status, page, _ = _json(f"{base}/api/queue?filter={filt}", cookie=cookie_b)
            assert status == 200
            assert page["items"] == []
    finally:
        server.shutdown()
