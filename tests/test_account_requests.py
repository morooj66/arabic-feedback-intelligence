"""Self-service account requests with admin approval: service rules on BOTH repository backends (SQL + the Mongo test double) and the real HTTP surface.
Synthetic users only; no e-mail is sent anywhere."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

import test_mongo_repository_unit as mongo_fake
from rcs.review import account_requests as ar
from rcs.review.auth import ADMIN, REVIEWER, AuthenticatedUser, authenticate, create_user, verify_password
from rcs.review.http_api import serve_in_thread
from rcs.review.persistence import bundle_from_engine, bundle_from_mongo
from rcs.review.repository import get_engine, init_db

PW = "testpassword123"
ME = "new.person@example.invalid"


@pytest.fixture(autouse=True)
def _clean_limits():
    ar.reset_limits_for_tests()
    yield
    ar.reset_limits_for_tests()


def _auth(u):
    return AuthenticatedUser(u.user_id, u.email, u.display_name, u.role)


@pytest.fixture(params=["sql", "mongo"])
def world(request, tmp_path):
    if request.param == "sql":
        engine = get_engine(tmp_path / "a.sqlite"); init_db(engine); bundle = bundle_from_engine(engine)
    else:
        bundle = bundle_from_mongo(mongo_fake._FakeDatabase())
    w = type("W", (), {})()
    w.bundle, w.kind = bundle, request.param
    w.admin = _auth(create_user(bundle, email="admin@example.invalid", password=PW, display_name="admin", role=ADMIN))
    w.rev = _auth(create_user(bundle, email="rev@example.invalid", password=PW, display_name="rev", role=REVIEWER))
    return w


def submit(w, email=ME, name="مستخدم جديد", password="a-good-password", addr="10.0.0.1"):
    return ar.submit_request(w.bundle, email=email, display_name=name, password=password, client_address=addr)


def test_a_request_is_not_a_login_until_an_admin_approves_it(world):
    w = world
    assert submit(w) is None
    assert authenticate(w.bundle, email=ME, password="a-good-password") is None                 # no account exists yet
    assert w.bundle.users.get_by_email(ME) is None
    pending = ar.list_pending(w.bundle, w.admin)
    assert [(p["email"], p["display_name"]) for p in pending] == [(ME, "مستخدم جديد")] and "password" not in json.dumps(pending)
    out = ar.approve(w.bundle, w.admin, pending[0]["request_id"])
    assert out["role"] == "REVIEWER"
    u = authenticate(w.bundle, email=ME, password="a-good-password")                            # the password chosen at request time now works
    assert u is not None and u.role == "REVIEWER" and u.is_active and u.display_name == "مستخدم جديد"
    assert ar.list_pending(w.bundle, w.admin) == []


def test_the_password_is_stored_only_as_a_bcrypt_hash(world):
    w = world
    submit(w, password="plain-secret-123")
    rec = w.bundle.account_requests.list_pending(10)[0]
    assert rec.password_hash != "plain-secret-123" and rec.password_hash.startswith("$2") and verify_password("plain-secret-123", rec.password_hash)


def test_the_role_can_never_be_chosen_or_elevated_by_the_requester(world):
    w = world
    with pytest.raises(TypeError):
        ar.submit_request(w.bundle, email=ME, display_name="x", password="a-good-password", client_address="1", role="ADMIN")   # no such parameter exists
    submit(w)
    ar.approve(w.bundle, w.admin, ar.list_pending(w.bundle, w.admin)[0]["request_id"])
    assert w.bundle.users.get_by_email(ME).role == "REVIEWER"                                    # approval has no role input either: always REVIEWER


def test_only_admins_can_list_approve_or_reject(world):
    w = world
    submit(w)
    rid = ar.list_pending(w.bundle, w.admin)[0]["request_id"]
    for fn in (lambda: ar.list_pending(w.bundle, w.rev), lambda: ar.approve(w.bundle, w.rev, rid), lambda: ar.reject(w.bundle, w.rev, rid),
               lambda: ar.approve(w.bundle, None, rid)):
        with pytest.raises(ar.AccountRequestError) as e:
            fn()
        assert e.value.status == 403
    assert w.bundle.users.get_by_email(ME) is None and len(ar.list_pending(w.bundle, w.admin)) == 1


def test_no_account_enumeration_known_pending_and_new_addresses_look_identical(world):
    w = world
    assert submit(w, email="rev@example.invalid") is None                                       # already an account
    assert submit(w, email=ME) is None
    assert submit(w, email=ME, name="مختلف", password="another-password-1") is None             # already pending
    assert [p["email"] for p in ar.list_pending(w.bundle, w.admin)] == [ME]                     # nothing created for the known address, no duplicate for the pending one
    assert w.bundle.users.get_by_email("rev@example.invalid").display_name == "rev"             # the existing account is untouched
    assert authenticate(w.bundle, email="rev@example.invalid", password="a-good-password") is None


def test_a_rejected_requester_cannot_log_in_and_cannot_silently_re_enter_the_queue(world):
    w = world
    submit(w); rid = ar.list_pending(w.bundle, w.admin)[0]["request_id"]
    assert ar.reject(w.bundle, w.admin, rid)["status"] == "REJECTED"
    assert authenticate(w.bundle, email=ME, password="a-good-password") is None
    assert submit(w) is None and ar.list_pending(w.bundle, w.admin) == []                       # same generic answer, nothing queued
    with pytest.raises(ar.AccountRequestError) as e:
        ar.approve(w.bundle, w.admin, rid)
    assert e.value.status == 409 and w.bundle.users.get_by_email(ME) is None


def test_a_request_can_be_decided_exactly_once(world):
    w = world
    submit(w); rid = ar.list_pending(w.bundle, w.admin)[0]["request_id"]
    ar.approve(w.bundle, w.admin, rid)
    for fn in (ar.approve, ar.reject):
        with pytest.raises(ar.AccountRequestError) as e:
            fn(w.bundle, w.admin, rid)
        assert e.value.status == 409
    assert w.bundle.users.get_by_email(ME).role == "REVIEWER"
    with pytest.raises(ar.AccountRequestError) as e:
        ar.approve(w.bundle, w.admin, "no-such-id")
    assert e.value.status == 404


def test_approving_when_the_account_appeared_meanwhile_closes_the_request_without_touching_the_account(world):
    w = world
    submit(w); rid = ar.list_pending(w.bundle, w.admin)[0]["request_id"]
    create_user(w.bundle, email=ME, password="someone-elses-password", display_name="existing", role=REVIEWER)
    with pytest.raises(ar.AccountRequestError) as e:
        ar.approve(w.bundle, w.admin, rid)
    assert e.value.code == "account_exists"
    assert authenticate(w.bundle, email=ME, password="someone-elses-password") is not None and authenticate(w.bundle, email=ME, password="a-good-password") is None
    assert ar.list_pending(w.bundle, w.admin) == []


@pytest.mark.parametrize("kwargs,code", [
    ({"email": "not-an-email"}, "invalid_email"), ({"email": ""}, "invalid_email"), ({"email": "a b@example.invalid"}, "invalid_email"), ({"email": "a@@b.c"}, "invalid_email"),
    ({"email": "x" * 250 + "@e.co"}, "invalid_email"), ({"name": ""}, "invalid_name"), ({"name": "   "}, "invalid_name"), ({"name": "n" * 101}, "invalid_name"),
    ({"password": "short"}, "invalid_password"), ({"password": "é" * 40}, "invalid_password"), ({"password": ""}, "invalid_password"),
    ({"email": None}, "invalid_request"), ({"password": 12345678}, "invalid_request"), ({"name": ["x"]}, "invalid_request")])
def test_invalid_input_is_a_clean_400_and_stores_nothing(world, kwargs, code):
    w = world
    with pytest.raises(ar.AccountRequestError) as e:
        submit(w, **kwargs)
    assert (e.value.status, e.value.code) == (400, code)
    assert w.bundle.account_requests.count_pending() == 0


def test_rate_limits_per_address_and_per_email(world):
    w = world
    for i in range(ar.REQUEST_LIMIT_PER_ADDRESS):
        submit(w, email=f"p{i}@example.invalid", addr="203.0.113.9")
    with pytest.raises(ar.AccountRequestError) as e:
        submit(w, email="p-extra@example.invalid", addr="203.0.113.9")
    assert e.value.status == 429
    submit(w, email="other@example.invalid", addr="203.0.113.10")                                 # another address is unaffected
    for _ in range(ar.REQUEST_LIMIT_PER_EMAIL):
        submit(w, email="same@example.invalid", addr=f"198.51.100.{_}")
    with pytest.raises(ar.AccountRequestError) as e:
        submit(w, email="same@example.invalid", addr="198.51.100.77")
    assert e.value.status == 429


def test_the_pending_queue_is_capped_without_revealing_it(world, monkeypatch):
    w = world
    monkeypatch.setattr(ar, "PENDING_CAP", 2)
    for i in range(4):
        assert submit(w, email=f"c{i}@example.invalid", addr=f"192.0.2.{i}") is None
    assert w.bundle.account_requests.count_pending() == 2


def test_names_are_whitespace_normalised_and_arabic_is_preserved(world):
    w = world
    submit(w, name="  أحمد    بن  علي \n")
    assert ar.list_pending(w.bundle, w.admin)[0]["display_name"] == "أحمد بن علي"


def test_concurrent_identical_requests_store_one(world):
    w = world
    if w.kind == "mongo":
        pytest.skip("the in-memory Mongo double is not thread-safe; uniqueness there is the primary-key sentinel, covered sequentially")
    ts = [threading.Thread(target=lambda i=i: submit(w, addr=f"10.1.0.{i}")) for i in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert w.bundle.account_requests.count_pending() == 1


def test_concurrent_approvals_create_one_account(world):
    w = world
    if w.kind == "mongo":
        pytest.skip("the in-memory Mongo double is not thread-safe")
    submit(w); rid = ar.list_pending(w.bundle, w.admin)[0]["request_id"]
    outcomes = []

    def go():
        try:
            ar.approve(w.bundle, w.admin, rid); outcomes.append("ok")
        except ar.AccountRequestError as exc:
            outcomes.append(exc.code)
    ts = [threading.Thread(target=go) for _ in range(5)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert outcomes.count("ok") == 1 and all(o in ("ok", "already_decided", "account_exists") for o in outcomes)
    assert w.bundle.account_requests.get(rid).status == "APPROVED"                                  # a losing approver can never flip the winner's decision
    assert len([u for u in w.bundle.users.list_all() if u.email == ME]) == 1


# ------------------------------------------------------------------------------------------------------ the real HTTP surface
@pytest.fixture(scope="module")
def http(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arq")
    e = get_engine(tmp / "h.sqlite"); init_db(e)
    create_user(e, email="admin@example.invalid", password=PW, display_name="admin", role="ADMIN")
    create_user(e, email="rev@example.invalid", password=PW, display_name="rev", role="REVIEWER")
    server, base = serve_in_thread(e)
    h = type("H", (), {})(); h.cookie = {}; h.engine = e

    def call(method, path, who=None, body=None):
        headers = {"Content-Type": "application/json"}
        if who:
            headers["Cookie"] = h.cookie[who]
        r = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}"), resp.headers
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode() or "{}"), err.headers

    h.call = call
    for k in ("admin", "rev"):
        h.cookie[k] = call("POST", "/api/login", None, {"email": f"{k}@example.invalid", "password": PW})[2].get("Set-Cookie").split(";")[0]
    yield h
    server.shutdown()


def test_http_health_advertises_account_requests(http):
    assert http.call("GET", "/api/health")[1]["account_requests"] is True


def test_http_full_flow_request_pending_login_denied_approve_login_works(http):
    ar.reset_limits_for_tests()
    s, out, _ = http.call("POST", "/api/account-requests", None, {"email": "Flow.User@Example.Invalid", "display_name": "مستخدم", "password": "flow-password-1", "role": "ADMIN", "is_active": True})
    assert (s, out) == (200, {"ok": True})
    assert http.call("POST", "/api/login", None, {"email": "flow.user@example.invalid", "password": "flow-password-1"})[0] == 401      # still inactive
    s, lst, _ = http.call("GET", "/api/admin/account-requests", "admin")
    assert s == 200 and [r["email"] for r in lst["requests"]] == ["flow.user@example.invalid"] and "password" not in json.dumps(lst)
    rid = lst["requests"][0]["request_id"]
    assert http.call("GET", "/api/admin/account-requests", "rev")[0] == 403
    assert http.call("GET", "/api/admin/account-requests", None)[0] == 401
    assert http.call("POST", "/api/admin/account-requests/approve", "rev", {"request_id": rid})[0] == 403
    assert http.call("POST", "/api/admin/account-requests/approve", None, {"request_id": rid})[0] == 401
    s, ok, _ = http.call("POST", "/api/admin/account-requests/approve", "admin", {"request_id": rid, "role": "ADMIN"})
    assert s == 200 and ok["role"] == "REVIEWER"                                                                                 # a role in the body is ignored
    s, sess, hd = http.call("POST", "/api/login", None, {"email": "flow.user@example.invalid", "password": "flow-password-1"})
    assert s == 200 and sess.get("role") == "REVIEWER"
    http.cookie["new"] = hd.get("Set-Cookie").split(";")[0]
    assert http.call("GET", "/api/admin/account-requests", "new")[0] == 403                                                      # the new reviewer has no admin power
    assert http.call("POST", "/api/admin/account-requests/approve", "admin", {"request_id": rid})[0] == 409


def test_http_invalid_and_rate_limited_requests(http):
    ar.reset_limits_for_tests()
    assert http.call("POST", "/api/account-requests", None, {"email": "bad", "display_name": "x", "password": "long-enough-1"})[0] == 400
    assert http.call("POST", "/api/account-requests", None, {"email": "ok@example.invalid", "display_name": "x", "password": "short"})[0] == 400
    assert http.call("POST", "/api/account-requests", None, {"display_name": "x"})[0] == 400
    codes = [http.call("POST", "/api/account-requests", None, {"email": f"u{i}@example.invalid", "display_name": "x", "password": "long-enough-1"})[0] for i in range(ar.REQUEST_LIMIT_PER_ADDRESS + 2)]
    assert codes[:ar.REQUEST_LIMIT_PER_ADDRESS] == [200] * ar.REQUEST_LIMIT_PER_ADDRESS and codes[-1] == 429
    ar.reset_limits_for_tests()


def test_http_response_is_identical_for_known_and_unknown_addresses(http):
    ar.reset_limits_for_tests()
    a = http.call("POST", "/api/account-requests", None, {"email": "rev@example.invalid", "display_name": "x", "password": "long-enough-1"})[:2]
    b = http.call("POST", "/api/account-requests", None, {"email": "brand.new@example.invalid", "display_name": "x", "password": "long-enough-1"})[:2]
    assert a == b == (200, {"ok": True})
    ar.reset_limits_for_tests()
