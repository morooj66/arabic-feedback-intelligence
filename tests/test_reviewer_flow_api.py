"""Reviewer workflow over the real HTTP server (synthetic data): assigned-only work, filters, the three response decisions, persistence, counters,
completed vs pending, conflicts, reassignment mid-review, and the rule that reviewers can only decide on generated RESPONSES -- never on classification."""
from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

import pytest
from sqlalchemy.orm import Session

from rcs.review.auth import create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem, ReviewItemEvent
from rcs.review.repository import get_engine, init_db

T0 = dt.datetime(2026, 8, 1, 9, 0, 0)
DRAFT = "نشكركم على ملاحظتكم القيّمة."


def _json(raw):
    try:
        return json.loads(raw.decode() or "{}")
    except ValueError:
        return {"raw": raw[:80].decode(errors="ignore")}


def req(base, path, *, method="GET", data=None, cookie=None):
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    r = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(), headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, _json(resp.read()), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, _json(err.read()), err.headers


class Env:
    pass


@pytest.fixture()
def env(tmp_path):
    e = get_engine(tmp_path / "f.sqlite")
    init_db(e)
    users = {k: create_user(e, email=f"{k}@t.local", password="testpassword123", display_name=k, role=r) for k, r in (("admin", "ADMIN"), ("a", "REVIEWER"), ("b", "REVIEWER"))}
    intents, topics = ["SUGGESTION", "INQUIRY", "COMPLAINT", "THANKS_PRAISE"], ["WORD_SUBMISSION", "GENERAL", "TECHNICAL_ISSUE", "GENERAL"]
    with Session(e) as s:
        def add(prefix, i, owner, **kw):
            k = i % 4
            s.add(ReviewItem(comment_id=f"{prefix}{i:02d}", source_file="f.xlsx", source_row_index=i, original_comment=f"تعليق اختباري رقم {prefix}{i:02d}", review_status=kw.pop("status", "DRAFT_READY"),
                             ai_primary_intent=intents[k], primary_intent=intents[k], ai_topic_subtype=topics[k], topic_subtype=topics[k], sentiment="NEUTRAL", ai_sentiment="NEUTRAL",
                             suggested_final_reply=DRAFT, ai_suggested_final_reply=DRAFT, reviewer_user_id=owner, review_priority=(i % 3) + 1,
                             ingested_at=T0 + dt.timedelta(minutes=i), comment_created_at=T0 + dt.timedelta(minutes=i), row_version=1, **kw))
        for i in range(20):
            add("a", i, users["a"].user_id)
        for i in range(2):
            add("d", i, users["a"].user_id, status="APPROVED", reviewer_action="SEND_AS_IS", final_reply=DRAFT)
        for i in range(6):
            add("b", i, users["b"].user_id)
        add("u", 0, None)
        s.commit()
    server, base = serve_in_thread(e)
    env_ = Env()
    env_.engine, env_.base, env_.users = e, base, users
    env_.cookie = {}
    for k in users:
        _, _, h = req(base, "/api/login", method="POST", data={"email": f"{k}@t.local", "password": "testpassword123"})
        env_.cookie[k] = h.get("Set-Cookie").split(";")[0]
    env_.get = lambda who, path: req(base, path, cookie=env_.cookie[who])
    env_.post = lambda who, path, data=None: req(base, path, method="POST", data=data if data is not None else {}, cookie=env_.cookie[who])
    env_.row = lambda cid: _row(e, cid)
    yield env_
    server.shutdown()


def _row(engine, cid):
    with Session(engine) as s:
        r = s.get(ReviewItem, cid)
        s.expunge(r)
        return r


# ------------------------------------------------------------------------------------------------------------------------ visibility, filters, counters
def test_reviewer_sees_only_their_assigned_work_and_counters_match(env):
    s, q, _ = env.get("a", "/api/queue?limit=100&filter=pending")
    assert s == 200 and q["total"] == 20 and all(i["comment_id"].startswith("a") for i in q["items"])
    s, c, _ = env.get("a", "/api/counts")
    assert (c["all"], c["pending"], c["approved"]) == (22, 20, 2)
    _, qb, _ = env.get("b", "/api/queue?limit=100")
    assert qb["total"] == 6 and not any(i["comment_id"][0] in "adu" for i in qb["items"])
    assert env.get("b", "/api/items/a00")[0] in (403, 404) and env.get("a", "/api/items/u00")[0] in (403, 404)


def test_filters_work_on_the_reviewers_own_queue(env):
    _, q, _ = env.get("a", "/api/queue?limit=100&primary_intent=COMPLAINT")
    assert q["total"] == 5 and all(i["primary_intent"] == "COMPLAINT" for i in q["items"])
    _, q, _ = env.get("a", "/api/queue?limit=100&topic_subtype=GENERAL&primary_intent=INQUIRY,THANKS_PRAISE")
    assert q["total"] == 10
    _, q, _ = env.get("a", "/api/queue?limit=100&search=a07")
    assert [i["comment_id"] for i in q["items"]] == ["a07"]
    _, q, _ = env.get("a", "/api/queue?limit=5&offset=15&sort=oldest")
    assert len(q["items"]) == 5 and q["items"][0]["comment_id"] == "a15"                 # stable paging across the whole working set
    _, done, _ = env.get("a", "/api/queue?limit=100&filter=completed")
    assert done["total"] == 2 and {i["comment_id"] for i in done["items"]} == {"d00", "d01"}
    _, other, _ = env.get("a", "/api/queue?limit=100&source_file=f.xlsx&reviewer_user_id=" + env.users["b"].user_id)
    assert other["total"] == 20                                                          # a reviewer-supplied reviewer filter is ignored, never an override


# ------------------------------------------------------------------------------------------------------------------------ decisions and persistence
def test_three_decisions_persist_and_update_progress(env):
    s, d, _ = env.post("a", "/api/items/a00/claim")
    assert s == 200 and d["review_status"] == "IN_REVIEW"
    ver = d["row_version"]
    s, d, _ = env.post("a", "/api/items/a00/send-as-is", {"expected_version": ver})
    assert s == 200 and d["review_status"] == "APPROVED" and env.row("a00").final_reply == DRAFT
    s, d, _ = env.post("a", "/api/items/a01/edit-and-send", {"edited_response": "نشكركم على تواصلكم، وسيُنظر في ملاحظتكم."})
    assert s == 200 and d["review_status"] == "EDITED_AND_APPROVED"
    r = env.row("a01")
    assert r.reviewer_edit == "نشكركم على تواصلكم، وسيُنظر في ملاحظتكم." and r.final_reply == r.reviewer_edit and r.suggested_final_reply == DRAFT
    s, d, _ = env.post("a", "/api/items/a02/no-response", {"reason": "لا يتطلب ردًا"})
    assert s == 200 and d["review_status"] == "NO_REPLY" and env.row("a02").no_response_reason == "لا يتطلب ردًا" and env.row("a02").final_reply is None
    _, c, _ = env.get("a", "/api/counts")
    assert (c["all"], c["pending"], c["approved"], c["no_reply"]) == (22, 17, 4, 1)
    _, pend, _ = env.get("a", "/api/queue?limit=100&filter=pending")
    _, done, _ = env.get("a", "/api/queue?limit=100&filter=completed")
    pend_ids, done_ids = {i["comment_id"] for i in pend["items"]}, {i["comment_id"] for i in done["items"]}
    assert not pend_ids & done_ids and len(pend_ids) == 17 and {"a00", "a01", "a02", "d00", "d01"} == done_ids
    # a "new session" (fresh login after the browser was closed) sees the same persisted state
    _, _, h = req(env.base, "/api/login", method="POST", data={"email": "a@t.local", "password": "testpassword123"})
    again = h.get("Set-Cookie").split(";")[0]
    assert req(env.base, "/api/counts", cookie=again)[1]["pending"] == 17
    with Session(env.engine) as s_:
        types = sorted(e.event_type for e in s_.query(ReviewItemEvent).filter(ReviewItemEvent.review_item_id.in_(["a00", "a01", "a02"])).all())
    assert types.count("SEND_AS_IS") == 1 and types.count("EDIT_AND_SEND") == 1 and types.count("NO_RESPONSE_NEEDED") == 1


def test_a_decided_comment_cannot_be_changed_by_a_second_decision(env):
    assert env.post("a", "/api/items/a00/send-as-is")[0] == 200
    for path, body in (("/api/items/a00/edit-and-send", {"edited_response": "رد آخر مختلف تماما"}), ("/api/items/a00/no-response", {"reason": "x"})):
        s, _, _ = env.post("a", path, body)
        assert s in (400, 409, 422), (path, s)
    assert env.post("a", "/api/items/a00/send-as-is")[0] == 200                              # repeating the SAME decision is an idempotent no-op
    r = env.row("a00")
    assert r.review_status == "APPROVED" and r.final_reply == DRAFT and r.reviewer_edit is None and r.no_response_reason is None
    with Session(env.engine) as s_:
        assert s_.query(ReviewItemEvent).filter(ReviewItemEvent.review_item_id == "a00", ReviewItemEvent.event_type == "SEND_AS_IS").count() == 1


# ------------------------------------------------------------------------------------------------------------------------ classification is not reviewer-editable
@pytest.mark.parametrize("path,body", [
    ("/api/items/a03/send-as-is", {}), ("/api/items/a03/edit-and-send", {"edited_response": "نشكركم على ملاحظتكم الكريمة."}), ("/api/items/a03/no-response", {"reason": "x"}),
])
def test_classification_fields_in_a_decision_request_are_ignored(env, path, body):
    before = env.row("a03")
    payload = {**body, "primary_intent": "SPAM", "topic_subtype": "OFF_TOPIC_PERSONAL", "sentiment": "NEGATIVE", "ai_primary_intent": "SPAM", "routing_decision": "EXCLUDE", "reviewer_user_id": env.users["b"].user_id}
    s, _, _ = env.post("a", path, payload)
    assert s == 200
    after = env.row("a03")
    assert (after.primary_intent, after.topic_subtype, after.sentiment, after.ai_primary_intent, after.ai_topic_subtype) == (before.primary_intent, before.topic_subtype, before.sentiment, before.ai_primary_intent, before.ai_topic_subtype)
    assert after.reviewer_user_id == before.reviewer_user_id and after.routing_decision == before.routing_decision


@pytest.mark.parametrize("method,path", [("PUT", "/api/items/a04"), ("PATCH", "/api/items/a04"), ("POST", "/api/items/a04/classification"), ("POST", "/api/items/a04/reclassify"),
                                          ("POST", "/api/items/a04/change-classification"), ("POST", "/api/items/a04/set-intent")])
def test_there_is_no_endpoint_that_changes_classification(env, method, path):
    before = env.row("a04")
    s, _, _ = req(env.base, path, method=method, data={"primary_intent": "SPAM"}, cookie=env.cookie["a"])
    assert s in (404, 405, 501)
    assert env.row("a04").primary_intent == before.primary_intent


def test_the_item_detail_marks_only_response_actions_as_allowed(env):
    _, d, _ = env.get("a", "/api/items/a05")
    assert d["allowed_actions"] and set(d["allowed_actions"]) <= {"SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"}
    assert not any("classif" in a.lower() or "intent" in a.lower() for a in d["allowed_actions"])


# ------------------------------------------------------------------------------------------------------------------------ concurrency and ownership changes
def test_stale_version_cannot_overwrite_a_decision_made_in_another_tab(env):
    _, d, _ = env.post("a", "/api/items/a06/claim")
    stale = d["row_version"]
    assert env.post("a", "/api/items/a06/edit-and-send", {"edited_response": "نشكركم على ملاحظتكم الأولى.", "expected_version": stale})[0] == 200
    s, out, _ = env.post("a", "/api/items/a06/edit-and-send", {"edited_response": "رد آخر من تبويب قديم.", "expected_version": stale})
    assert s == 409 and out["error"] in ("version_conflict", "invalid_transition", "conflict")
    assert env.row("a06").reviewer_edit == "نشكركم على ملاحظتكم الأولى."
    with Session(env.engine) as s_:
        assert s_.query(ReviewItemEvent).filter(ReviewItemEvent.review_item_id == "a06", ReviewItemEvent.event_type == "EDIT_AND_SEND").count() == 1


def test_reassigned_away_comment_can_no_longer_be_decided_by_the_former_reviewer(env):
    assert env.post("admin", "/api/admin/assign", {"comment_ids": ["a07"], "reviewer_user_id": env.users["b"].user_id})[0] == 200
    for path in ("/api/items/a07", "/api/items/a07/claim", "/api/items/a07/send-as-is", "/api/items/a07/no-response"):
        s, _, _ = (env.get("a", path) if path.endswith("a07") else env.post("a", path, {"reason": "x"}))
        assert s in (403, 404), (path, s)
    assert env.row("a07").reviewer_action is None
    assert env.post("b", "/api/items/a07/send-as-is")[0] == 200                              # the new owner can


def test_unauthorized_and_cross_role_access(env):
    anon = lambda p, m="GET": req(env.base, p, method=m, data={} if m == "POST" else None)
    assert anon("/api/queue")[0] == 401 and anon("/api/items/a00")[0] == 401 and anon("/api/items/a00/send-as-is", "POST")[0] == 401
    for who in ("b",):
        for path in ("/api/items/a00/claim", "/api/items/a00/send-as-is", "/api/items/a00/no-response", "/api/items/a00/edit-and-send"):
            assert env.post(who, path, {"reason": "x", "edited_response": "نص"})[0] in (403, 404)
    assert env.get("a", "/api/admin/reviewers")[0] == 403 and env.get("a", "/api/excel/export")[0] == 403
    assert env.post("a", "/api/admin/assign", {"comment_ids": ["u00"], "reviewer_user_id": env.users["a"].user_id})[0] == 403
    assert env.row("u00").reviewer_user_id is None and env.row("a00").review_status == "DRAFT_READY"
