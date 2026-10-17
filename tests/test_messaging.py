"""Internal admin <-> reviewer messaging: service rules on BOTH repository backends (SQL + the Mongo test double) and the real HTTP surface. Synthetic users/messages only."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

import test_mongo_repository_unit as mongo_fake
from rcs.review import messaging
from rcs.review.auth import ADMIN, REVIEWER, AuthenticatedUser, create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.persistence import bundle_from_engine, bundle_from_mongo
from rcs.review.repository import get_engine, init_db

PW = "testpassword123"


def _auth(u):
    return AuthenticatedUser(u.user_id, u.email, u.display_name, u.role)


@pytest.fixture(params=["sql", "mongo"])
def world(request, tmp_path):
    if request.param == "sql":
        engine = get_engine(tmp_path / "m.sqlite"); init_db(engine); bundle = bundle_from_engine(engine)
    else:
        bundle = bundle_from_mongo(mongo_fake._FakeDatabase())
    mk = lambda k, role: create_user(bundle, email=f"{k}@t.local", password=PW, display_name=k, role=role)
    w = type("W", (), {})()
    w.bundle, w.kind = bundle, request.param
    w.admin, w.admin2, w.r1, w.r2 = (_auth(mk("admin", ADMIN)), _auth(mk("admin2", ADMIN)), _auth(mk("r1", REVIEWER)), _auth(mk("r2", REVIEWER)))
    return w


def send(w, actor, body="مرحبا", rid=None, cid="client-id-0001"):
    return messaging.send_message(w.bundle, actor, reviewer_user_id=rid, body=body, client_message_id=cid)


def test_admin_sends_reviewer_replies_history_has_sender_identity_order_and_timestamps(world):
    w = world
    a = send(w, w.admin, "رسالة من الإدارة", rid=w.r1.user_id, cid="admin-msg-0001")
    r = send(w, w.r1, "رد المراجع", cid="rev-msg-00001")
    assert a["sender_role"] == "ADMIN" and a["sender_user_id"] == w.admin.user_id and a["sender_name"] == "admin" and a["created_at"].endswith("Z") and a["read_at"] is None
    assert r["sender_role"] == "REVIEWER" and r["reviewer_user_id"] == w.r1.user_id
    for actor, rid in ((w.admin, w.r1.user_id), (w.r1, None), (w.admin2, w.r1.user_id)):                      # every admin and the reviewer see the same history
        conv = messaging.get_conversation(w.bundle, actor, reviewer_user_id=rid)
        assert [m["body"] for m in conv["messages"]] == ["رسالة من الإدارة", "رد المراجع"] and conv["reviewer_name"] == "r1"


def test_unread_counts_and_marking_read_only_affect_the_other_sides_messages(world):
    w = world
    send(w, w.admin, "1", rid=w.r1.user_id, cid="admin-msg-0001"); send(w, w.admin, "2", rid=w.r1.user_id, cid="admin-msg-0002")
    send(w, w.r1, "3", cid="rev-msg-00001"); send(w, w.r2, "4", cid="rev2-msg-0001")
    assert messaging.unread_total(w.bundle, w.r1) == 2 and messaging.unread_total(w.bundle, w.r2) == 0
    assert messaging.unread_total(w.bundle, w.admin) == 2                                                      # one unread from each reviewer
    by = {c["reviewer_user_id"]: c for c in messaging.list_conversations(w.bundle, w.admin)}
    assert by[w.r1.user_id]["unread"] == 1 and by[w.r2.user_id]["unread"] == 1 and by[w.r1.user_id]["last_message"]["body"] == "3"
    assert messaging.mark_conversation_read(w.bundle, w.r1, reviewer_user_id=None)["marked"] == 2
    assert messaging.unread_total(w.bundle, w.r1) == 0 and messaging.unread_total(w.bundle, w.admin) == 2    # the reviewer reading does not mark admin-bound messages
    assert messaging.mark_conversation_read(w.bundle, w.admin, reviewer_user_id=w.r1.user_id)["marked"] == 1
    assert messaging.unread_total(w.bundle, w.admin) == 1 and messaging.mark_conversation_read(w.bundle, w.admin2, reviewer_user_id=w.r1.user_id)["marked"] == 0
    conv = messaging.get_conversation(w.bundle, w.r1, reviewer_user_id=None)
    assert [m["read_at"] is not None for m in conv["messages"]] == [True, True, True]


def test_conversations_list_is_admin_only_and_includes_reviewers_without_messages(world):
    w = world
    send(w, w.r2, "x", cid="rev2-msg-0001")
    names = [c["reviewer_name"] for c in messaging.list_conversations(w.bundle, w.admin)]
    assert names == ["r2", "r1"]                                                                               # newest conversation first, then reviewers with none
    with pytest.raises(messaging.MessagingError) as e:
        messaging.list_conversations(w.bundle, w.r1)
    assert e.value.status == 403


def test_reviewer_isolation_in_the_service(world):
    w = world
    send(w, w.admin, "سري لـ r1", rid=w.r1.user_id, cid="admin-msg-0001")
    for fn in (lambda: messaging.get_conversation(w.bundle, w.r2, reviewer_user_id=w.r1.user_id),
               lambda: messaging.mark_conversation_read(w.bundle, w.r2, reviewer_user_id=w.r1.user_id),
               lambda: send(w, w.r2, "x", rid=w.r1.user_id, cid="rev2-msg-0001")):
        with pytest.raises(messaging.MessagingError) as e:
            fn()
        assert e.value.status == 403
    assert messaging.get_conversation(w.bundle, w.r2, reviewer_user_id=None)["messages"] == []
    assert len(messaging.get_conversation(w.bundle, w.r1, reviewer_user_id=w.r1.user_id)["messages"]) == 1       # naming oneself is fine


def test_admin_must_name_an_existing_reviewer(world):
    w = world
    for rid in (None, "no-such-user", w.admin2.user_id):                                                         # missing, unknown, and an admin id
        with pytest.raises(messaging.MessagingError) as e:
            send(w, w.admin, "x", rid=rid)
        assert e.value.status == 404
    with pytest.raises(messaging.MessagingError) as e:
        messaging.get_conversation(w.bundle, w.admin, reviewer_user_id=w.admin2.user_id)
    assert e.value.status == 404


@pytest.mark.parametrize("body,code", [("", "empty_message"), ("   \n\t ", "empty_message"), (None, "invalid_message"), (123, "invalid_message"), (["a"], "invalid_message"),
                                       ("x" * 2001, "message_too_long"), ("hi\x00there", "invalid_message")])
def test_empty_and_invalid_messages_are_rejected_and_store_nothing(world, body, code):
    w = world
    with pytest.raises(messaging.MessagingError) as e:
        send(w, w.r1, body)
    assert (e.value.status, e.value.code) == (400, code)
    assert messaging.get_conversation(w.bundle, w.r1, reviewer_user_id=None)["messages"] == []


@pytest.mark.parametrize("cid", [None, "", "short", "has space in it!", "x" * 65, 12345678])
def test_a_valid_client_message_id_is_required(world, cid):
    with pytest.raises(messaging.MessagingError) as e:
        send(world, world.r1, "نص", cid=cid)
    assert e.value.code == "invalid_client_message_id"


def test_message_text_is_stored_exactly_trimmed_with_line_breaks_and_arabic_intact(world):
    w = world
    out = send(w, w.r1, "  السلام عليكم\nشكرًا <b>لكم</b> 😊  ", cid="rev-msg-00001")
    assert out["body"] == "السلام عليكم\nشكرًا <b>لكم</b> 😊"                                                    # no HTML interpretation server-side; the UI renders it as text
    assert send(w, w.r1, "a" * 2000, cid="rev-msg-00002")["body"] == "a" * 2000


def test_a_retried_send_is_idempotent_and_the_same_client_id_of_another_sender_is_not_a_collision(world):
    w = world
    first = send(w, w.r1, "مرة واحدة", cid="retry-id-0001")
    again = send(w, w.r1, "مرة واحدة", cid="retry-id-0001")
    changed = send(w, w.r1, "نص مختلف", cid="retry-id-0001")                                                     # same id -> still the original message
    assert first["duplicate"] is False and again["duplicate"] is True and changed["duplicate"] is True
    assert again["message_id"] == first["message_id"] == changed["message_id"] and changed["body"] == "مرة واحدة"
    other = send(w, w.r2, "رسالة r2", cid="retry-id-0001")
    assert other["duplicate"] is False
    assert len(messaging.get_conversation(w.bundle, w.r1, reviewer_user_id=None)["messages"]) == 1


def test_concurrent_identical_sends_create_exactly_one_message(world):
    w = world
    if w.kind == "mongo":
        pytest.skip("the in-memory Mongo double is not thread-safe; its idempotency key is the primary key, covered by the sequential test")
    results, errors = [], []

    def go():
        try:
            results.append(send(w, w.r1, "تزامن", cid="race-id-00001"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors and len({r["message_id"] for r in results}) == 1 and sum(1 for r in results if not r["duplicate"]) == 1
    assert len(messaging.get_conversation(w.bundle, w.r1, reviewer_user_id=None)["messages"]) == 1


def test_concurrent_distinct_sends_are_all_kept(world):
    w = world
    if w.kind == "mongo":
        pytest.skip("in-memory Mongo double is not thread-safe")
    errs = []

    def go(i):
        try:
            send(w, w.admin if i % 2 else w.r1, f"m{i}", rid=w.r1.user_id if i % 2 else None, cid=f"par-id-{i:05d}")
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=go, args=(i,)) for i in range(12)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs and len(messaging.get_conversation(w.bundle, w.r1, reviewer_user_id=None)["messages"]) == 12


def test_history_is_capped_to_the_most_recent_messages_oldest_first(world):
    w = world
    for i in range(messaging.HISTORY_LIMIT + 5):
        send(w, w.r1, f"n{i}", cid=f"cap-id-{i:05d}")
    msgs = messaging.get_conversation(w.bundle, w.r1, reviewer_user_id=None)["messages"]
    assert len(msgs) == messaging.HISTORY_LIMIT and msgs[-1]["body"] == f"n{messaging.HISTORY_LIMIT + 4}" and msgs[0]["body"] == "n5"


def test_inactive_reviewer_cannot_be_messaged_but_history_stays_readable(world):
    if world.kind == "mongo":
        world.bundle.users.collection.update_one({"_id": world.r1.user_id}, {"$set": {"is_active": False}})
    else:
        from sqlalchemy.orm import Session
        from rcs.review.models import User
        with Session(world.bundle.users.engine) as s:
            s.get(User, world.r1.user_id).is_active = False; s.commit()
    with pytest.raises(messaging.MessagingError) as e:
        send(world, world.admin, "x", rid=world.r1.user_id)
    assert e.value.status == 409
    assert messaging.get_conversation(world.bundle, world.admin, reviewer_user_id=world.r1.user_id)["messages"] == []


# ------------------------------------------------------------------ the real HTTP surface
@pytest.fixture(scope="module")
def http(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("msg")
    e = get_engine(tmp / "h.sqlite"); init_db(e)
    users = {k: create_user(e, email=f"{k}@t.local", password=PW, display_name=k, role=r) for k, r in (("admin", "ADMIN"), ("r1", "REVIEWER"), ("r2", "REVIEWER"))}
    server, base = serve_in_thread(e)
    h = type("H", (), {})()
    h.base, h.users, h.cookie = base, users, {}

    def raw(method, path, who=None, body=None):
        headers = {"Content-Type": "application/json"}
        if who:
            headers["Cookie"] = h.cookie[who]
        r = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}"), resp.headers
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode() or "{}"), err.headers

    h.call = raw
    for k in users:
        _, _, hd = raw("POST", "/api/login", body={"email": f"{k}@t.local", "password": PW})
        h.cookie[k] = hd.get("Set-Cookie").split(";")[0]
    yield h
    server.shutdown()


def test_http_unauthenticated_requests_are_rejected_everywhere(http):
    for method, path, body in (("GET", "/api/messages/conversations", None), ("GET", "/api/messages/unread", None), ("GET", "/api/messages/conversation", None),
                               ("POST", "/api/messages/send", {"body": "x", "client_message_id": "anon-id-0001"}), ("POST", "/api/messages/read", {})):
        assert http.call(method, path, None, body)[0] == 401, path


def test_http_full_exchange_refresh_persistence_and_isolation(http):
    r1, r2, adm = http.users["r1"].user_id, http.users["r2"].user_id, http.users["admin"].user_id
    s, out, _ = http.call("POST", "/api/messages/send", "admin", {"reviewer_user_id": r1, "body": "تعليمات للمراجع الأول", "client_message_id": "http-admin-001"})
    assert s == 200 and out["sender_user_id"] == adm and out["sender_role"] == "ADMIN"
    assert http.call("GET", "/api/messages/unread", "r1")[1] == {"unread": 1} and http.call("GET", "/api/messages/unread", "r2")[1] == {"unread": 0}
    s, conv, _ = http.call("GET", "/api/messages/conversation", "r1")
    assert s == 200 and [m["body"] for m in conv["messages"]] == ["تعليمات للمراجع الأول"] and conv["unread"] == 1
    assert http.call("POST", "/api/messages/read", "r1", {})[1]["marked"] == 1
    assert http.call("GET", "/api/messages/unread", "r1")[1] == {"unread": 0}
    s, rep, _ = http.call("POST", "/api/messages/send", "r1", {"body": "تم", "client_message_id": "http-rev1-001"})
    assert s == 200 and rep["sender_role"] == "REVIEWER" and rep["sender_user_id"] == r1 and rep["reviewer_user_id"] == r1
    # a new login (fresh session) still sees the stored history
    _, _, hd = http.call("POST", "/api/login", None, {"email": "r1@t.local", "password": PW})
    http.cookie["r1b"] = hd.get("Set-Cookie").split(";")[0]
    assert [m["body"] for m in http.call("GET", "/api/messages/conversation", "r1b")[1]["messages"]] == ["تعليمات للمراجع الأول", "تم"]
    s, lst, _ = http.call("GET", "/api/messages/conversations", "admin")
    row = {c["reviewer_user_id"]: c for c in lst["conversations"]}
    assert s == 200 and row[r1]["unread"] == 1 and row[r1]["last_message"]["body"] == "تم" and row[r2]["last_message"] is None
    # isolation + role enforcement
    assert http.call("GET", f"/api/messages/conversation?reviewer_user_id={r1}", "r2")[0] == 403
    assert http.call("POST", "/api/messages/send", "r2", {"reviewer_user_id": r1, "body": "x", "client_message_id": "http-rev2-001"})[0] == 403
    assert http.call("POST", "/api/messages/read", "r2", {"reviewer_user_id": r1})[0] == 403
    assert http.call("GET", "/api/messages/conversations", "r1")[0] == 403
    assert http.call("GET", "/api/messages/conversation", "r2")[1]["messages"] == []
    assert http.call("POST", "/api/messages/send", "admin", {"body": "بلا مراجع", "client_message_id": "http-admin-002"})[0] == 404
    assert http.call("GET", f"/api/messages/conversation?reviewer_user_id={adm}", "admin")[0] == 404


def test_http_sender_identity_cannot_be_forged_and_invalid_input_is_a_clean_400(http):
    r1, r2 = http.users["r1"].user_id, http.users["r2"].user_id
    s, out, _ = http.call("POST", "/api/messages/send", "r2", {"body": "من r2", "client_message_id": "http-forge-001", "sender_user_id": http.users["admin"].user_id, "sender_role": "ADMIN", "created_at": "2001-01-01T00:00:00Z"})
    assert s == 200 and out["sender_user_id"] == r2 and out["sender_role"] == "REVIEWER" and out["created_at"].startswith("20") and not out["created_at"].startswith("2001")
    for body in ({"body": "", "client_message_id": "http-bad-0001"}, {"body": "x", "client_message_id": "!"}, {"client_message_id": "http-bad-0002"}, {"body": {"a": 1}, "client_message_id": "http-bad-0003"}):
        assert http.call("POST", "/api/messages/send", "r1", body)[0] == 400
    assert http.call("POST", "/api/messages/send", "r1", {"body": "dup", "client_message_id": "http-dup-0001"})[1]["duplicate"] is False
    assert http.call("POST", "/api/messages/send", "r1", {"body": "dup", "client_message_id": "http-dup-0001"})[1]["duplicate"] is True
    assert sum(1 for m in http.call("GET", "/api/messages/conversation", "r1")[1]["messages"] if m["body"] == "dup") == 1
    assert http.call("GET", f"/api/messages/conversation?reviewer_user_id={r2}", "admin")[1]["messages"][0]["body"] == "من r2"


def test_http_concurrent_retries_of_one_send_store_one_message(http):
    res = []

    def go():
        res.append(http.call("POST", "/api/messages/send", "r1", {"body": "سباق", "client_message_id": "http-race-0001"}))

    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(s == 200 for s, _, _ in res) and len({b["message_id"] for _, b, _ in res}) == 1
    assert sum(1 for m in http.call("GET", "/api/messages/conversation", "r1")[1]["messages"] if m["body"] == "سباق") == 1
