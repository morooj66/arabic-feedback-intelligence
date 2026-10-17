"""HTTP adapter tests. Uses ReviewWorkflowService; no mock workflow backend."""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from urllib.parse import quote

from config import settings
from rcs.review.auth import create_user
from rcs.review.demo_seed import seed_demo_db
from rcs.review.http_api import serve_in_thread
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.response_generation.schema import GenerationResult


def _json(url: str, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload, resp.headers
    except urllib.error.HTTPError as err:
        payload = json.loads(err.read().decode("utf-8"))
        return err.code, payload, err.headers


def test_real_login_does_not_echo_password_and_rejects_wrong_credentials(tmp_path):
    engine = get_engine(tmp_path / "r.sqlite")
    init_db(engine)
    create_user(engine, email="reviewer@example.org", password="a-real-password-123", display_name="مراجع تجريبي", role="REVIEWER")
    server, base = serve_in_thread(engine)
    try:
        # wrong password
        status, payload, _ = _json(
            f"{base}/api/login", method="POST", data={"email": "reviewer@example.org", "password": "wrong"}
        )
        assert status == 401
        assert payload["error"] == "invalid_credentials"

        # unknown email — identical generic response, never reveals which reason
        status, payload_unknown, _ = _json(
            f"{base}/api/login", method="POST", data={"email": "nobody@example.org", "password": "whatever"}
        )
        assert status == 401
        assert payload_unknown["error"] == "invalid_credentials"

        # correct login
        status, payload, headers = _json(
            f"{base}/api/login",
            method="POST",
            data={"email": "reviewer@example.org", "password": "a-real-password-123"},
        )
        assert status == 200
        assert payload["auth_mode"] == "session"
        assert payload["role"] == "REVIEWER"
        assert "password" not in json.dumps(payload)
        assert "password_hash" not in json.dumps(payload)
        cookie = headers.get("Set-Cookie")
        assert "rcs_session=" in cookie
        status, session, _ = _json(f"{base}/api/session", cookie=cookie.split(";")[0])
        assert status == 200
        assert session["email"] == "reviewer@example.org"
        assert session["role"] == "REVIEWER"
    finally:
        server.shutdown()


def test_queue_and_review_actions_via_http(tmp_path):
    db = seed_demo_db(tmp_path / "demo.sqlite")
    engine = get_engine(db)
    # ADMIN here (not REVIEWER): this test exercises the full unscoped
    # /api/analytics view too, which is an ADMIN-only capability as of
    # Phase 6 — see tests/test_auth.py for the dedicated REVIEWER-is-
    # blocked-from-it security test.
    create_user(engine, email="reviewer@example.org", password="a-real-password-123", display_name="مراجع تجريبي", role="ADMIN")
    server, base = serve_in_thread(engine)
    try:
        _, login, headers = _json(
            f"{base}/api/login",
            method="POST",
            data={"email": "reviewer@example.org", "password": "a-real-password-123"},
        )
        cookie = headers.get("Set-Cookie").split(";")[0]
        status, counts, _ = _json(f"{base}/api/counts", cookie=cookie)
        assert status == 200
        assert counts["pending"] >= 1
        assert counts["approved"] >= 1
        status, page, _ = _json(f"{base}/api/queue?filter=pending&sort=priority&limit=10", cookie=cookie)
        assert status == 200
        assert page["total"] >= 1
        newest = _json(f"{base}/api/queue?filter=pending&sort=newest", cookie=cookie)[1]
        oldest = _json(f"{base}/api/queue?filter=pending&sort=oldest", cookie=cookie)[1]
        assert newest["items"][0]["comment_id"] == "8742"
        assert oldest["items"][0]["comment_id"] == "8746"
        complaint = next(i for i in page["items"] if i["comment_id"] == "8743")
        assert complaint["review_priority"] == 1
        status, detail, _ = _json(f"{base}/api/items/8742", cookie=cookie)
        assert status == 200
        assert "شكرًا" in (detail["original_comment"] or "")
        original = detail["suggested_response"]
        assert original
        assert original != "شكرًا لتواصلكم معنا، نسعد بثقتكم بالمعجم."
        status, claimed, _ = _json(
            f"{base}/api/items/8742/claim",
            method="POST",
            data={"expected_version": detail["row_version"]},
            cookie=cookie,
        )
        assert status == 200
        assert claimed["review_status"] == "IN_REVIEW"
        status, sent, _ = _json(
            f"{base}/api/items/8742/send-as-is",
            method="POST",
            data={"expected_version": claimed["row_version"]},
            cookie=cookie,
        )
        assert status == 200
        assert sent["review_status"] == "APPROVED"
        assert sent["outbound_status"] == "PENDING"
        assert sent["suggested_response"] == original
        assert sent["final_response"] == original

        status, inq, _ = _json(f"{base}/api/items/8744", cookie=cookie)
        claimed2 = _json(
            f"{base}/api/items/8744/claim",
            method="POST",
            data={"expected_version": inq["row_version"]},
            cookie=cookie,
        )[1]
        edited = claimed2["suggested_response"] + " مع تعديل المراجع."
        status, edited_out, _ = _json(
            f"{base}/api/items/8744/edit-and-send",
            method="POST",
            data={"expected_version": claimed2["row_version"], "edited_response": edited},
            cookie=cookie,
        )
        assert status == 200
        assert edited_out["review_status"] == "EDITED_AND_APPROVED"
        assert edited_out["reviewer_action"] == "EDIT_AND_SEND"
        assert edited_out["suggested_response"] == claimed2["suggested_response"]
        assert edited_out["final_response"] == edited

        nrr = _json(f"{base}/api/items/8801", cookie=cookie)[1]
        status, closed, _ = _json(
            f"{base}/api/items/8801/no-response",
            method="POST",
            data={"expected_version": nrr["row_version"], "reason": "محتوى غير ذي صلة"},
            cookie=cookie,
        )
        assert status == 200
        assert closed["review_status"] == "NO_REPLY"
        assert closed["no_response_reason"] == "محتوى غير ذي صلة"

        approved_q = _json(f"{base}/api/queue?filter=approved", cookie=cookie)[1]
        assert any(i["comment_id"] == "8742" for i in approved_q["items"])
        search_pending = _json(f"{base}/api/queue?filter=all&search=" + quote("النسخة الورقية"), cookie=cookie)[1]
        assert any(i["comment_id"] == "8746" for i in search_pending["items"])
        page1 = _json(f"{base}/api/queue?filter=all&limit=2&offset=0", cookie=cookie)[1]
        page2 = _json(f"{base}/api/queue?filter=all&limit=2&offset=2", cookie=cookie)[1]
        assert page1["items"][0]["comment_id"] != page2["items"][0]["comment_id"]
        status, analytics, _ = _json(f"{base}/api/analytics?range=week", cookie=cookie)
        assert status == 200
        assert analytics["kpis"]["pending"] >= 1
        assert "trend" in analytics
        assert analytics["outcomes"]["SEND_AS_IS"] >= 1
        assert analytics["outcomes"]["EDIT_AND_SEND"] >= 1
        assert analytics["outcomes"]["NO_RESPONSE_NEEDED"] >= 1
        assert [row["key"] for row in analytics["classification"]["topic"]] == list(settings.TOPIC_SUBTYPES)
        assert "WORD_OF_YEAR" not in [row["key"] for row in analytics["classification"]["topic"]]
        assert login["display_name"] == "مراجع تجريبي"
    finally:
        server.shutdown()


def test_http_concurrency_and_unauthenticated(tmp_path):
    engine = get_engine(tmp_path / "c.sqlite")
    init_db(engine)
    now = dt.datetime(2026, 1, 1)
    user_a = create_user(engine, email="a@example.org", password="passwordA123", display_name="A", role="REVIEWER")
    create_user(engine, email="b@example.org", password="passwordB123", display_name="B", role="REVIEWER")
    upsert_review_items(
        engine,
        [
            {
                "comment_id": "c-conf",
                "source_file": "t",
                "source_row_index": 1,
                "original_comment": "شكرًا",
                "ingested_at": now,
                "review_status": "CLASSIFIED",
                "ready_to_send": False,
                "row_version": 1,
                # Phase 8: pre-assigned to A — a REVIEWER can only act on
                # an item already assigned to them (see
                # tests/test_reviewer_isolation.py for the dedicated
                # cross-reviewer authorization tests this scenario used
                # to conflate with plain optimistic-concurrency conflicts).
                "reviewer_user_id": user_a.user_id,
            }
        ],
    )
    svc = ReviewWorkflowService(engine)
    svc.apply_generation_result(
        GenerationResult(
            comment_id="c-conf",
            routing_decision="GENERATE_DRAFT",
            generation_status="SUCCEEDED",
            suggested_response="شكرًا لتواصلكم معنا، نسعد بثقتكم بالمعجم.",
            review_required=True,
            reason_code="thanks_praise_eligible",
            provider="gemini",
            model_name="gemini-3.6-flash",
            prompt_version="rg-v5",
            generated_at="2026-08-25T12:00:00+00:00",
        )
    )
    server, base = serve_in_thread(engine)
    try:
        assert _json(f"{base}/api/queue")[0] == 401
        cookie = _json(f"{base}/api/login", method="POST", data={"email": "a@example.org", "password": "passwordA123"})[2].get(
            "Set-Cookie"
        ).split(";")[0]
        other = _json(f"{base}/api/login", method="POST", data={"email": "b@example.org", "password": "passwordB123"})[2].get(
            "Set-Cookie"
        ).split(";")[0]
        detail = _json(f"{base}/api/items/c-conf", cookie=cookie)[1]
        claimed = _json(
            f"{base}/api/items/c-conf/claim",
            method="POST",
            data={"expected_version": detail["row_version"]},
            cookie=cookie,
        )[1]
        status, conflict, _ = _json(
            f"{base}/api/items/c-conf/send-as-is",
            method="POST",
            data={"expected_version": 1},
            cookie=cookie,
        )
        assert status == 409
        assert conflict["error"] == "version_conflict"
        # B has no assignment on this item at all — rejected up front as a
        # cross-reviewer access attempt, before version/claim logic is
        # ever reached.
        status, forbidden, _ = _json(
            f"{base}/api/items/c-conf/claim",
            method="POST",
            data={"expected_version": claimed["row_version"]},
            cookie=other,
        )
        assert status == 403
        assert forbidden["error"] == "forbidden"
        missing = _json(f"{base}/api/items/does-not-exist", cookie=cookie)
        assert missing[0] == 404
        bad_edit = _json(
            f"{base}/api/items/c-conf/edit-and-send",
            method="POST",
            data={"expected_version": claimed["row_version"], "edited_response": "   "},
            cookie=cookie,
        )
        assert bad_edit[0] == 409
    finally:
        server.shutdown()
