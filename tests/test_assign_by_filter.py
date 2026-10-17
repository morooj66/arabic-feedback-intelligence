"""Server-side count/filter assignment (POST /api/admin/assign-by-filter, ReviewItemRepository.assign_by_filter).

Real SQLite repository + the real HTTP server; no mocked helpers. Mongo is exercised with the in-memory fake only (UNVERIFIED against a live MongoDB),
and PostgreSQL row-locking (FOR UPDATE SKIP LOCKED) cannot be exercised here: SQLite serialises writers, the compare-and-set UPDATE is what is verified.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import urllib.error
import urllib.request

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rcs.review.auth import create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem, ReviewItemEvent
from rcs.review.persistence import resolve_bundle
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService

T0 = dt.datetime(2026, 8, 1, 9, 0, 0)


def seed(engine, n, *, prefix="c", source_file="s.xlsx", status="DRAFT_READY", intent="SUGGESTION", topic="GENERAL", assignee=None, action=None, start=0):
    with Session(engine) as s:
        for i in range(start, start + n):
            s.add(ReviewItem(comment_id=f"{prefix}{i:04d}", source_file=source_file, source_row_index=i, original_comment=f"تعليق تجريبي رقم {prefix}{i} للاختبار",
                             review_status=status, ai_primary_intent=intent, primary_intent=intent, ai_topic_subtype=topic, topic_subtype=topic,
                             ingested_at=T0 + dt.timedelta(minutes=i), reviewer_user_id=assignee, reviewer_action=action, row_version=1))
        s.commit()


def rows(engine):
    with Session(engine) as s:
        return {r.comment_id: r for r in s.query(ReviewItem).all()}


def events(engine, event_type="ASSIGNED"):
    with Session(engine) as s:
        return s.query(ReviewItemEvent).filter(ReviewItemEvent.event_type == event_type).all()


@pytest.fixture()
def engine(tmp_path):
    e = get_engine(tmp_path / "a.sqlite")
    init_db(e)
    return e


def repo(engine):
    return resolve_bundle(engine).review_items


# ------------------------------------------------------------------------------------------------------------------------ counts
@pytest.mark.parametrize("n", [1, 5, 20, 150])
def test_assigns_exactly_the_requested_number_in_deterministic_order(engine, n):
    seed(engine, 200)
    res = repo(engine).assign_by_filter("rev-1", count=n, actor_user_id="adm", actor_role="ADMIN")
    assert (res.requested, res.assigned, res.shortfall, res.matched_before, res.remaining_eligible) == (n, n, 0, 200, 200 - n)
    r = rows(engine)
    mine = sorted(k for k, v in r.items() if v.reviewer_user_id == "rev-1")
    assert mine == [f"c{i:04d}" for i in range(n)]                                      # oldest first (ingested_at, then comment id)
    assert all(v.reviewer_user_id is None for k, v in r.items() if k not in set(mine))
    assert all(r[k].row_version == 2 for k in mine) and all(r[k].row_version == 1 for k in r if k not in set(mine))
    ev = events(engine)
    assert sorted(e.review_item_id for e in ev) == mine and {e.actor_user_id for e in ev} == {"adm"} and {e.actor_role for e in ev} == {"ADMIN"}
    assert all(e.to_assignee_user_id == "rev-1" and e.from_assignee_user_id is None for e in ev)


def test_assign_all_eligible(engine):
    seed(engine, 137)
    res = repo(engine).assign_by_filter("rev-1", count=None)
    assert (res.requested, res.assigned, res.shortfall, res.remaining_eligible) == (None, 137, 0, 0)
    assert all(v.reviewer_user_id == "rev-1" for v in rows(engine).values()) and len(events(engine)) == 137


def test_count_above_available_assigns_everything_and_reports_the_shortfall(engine):
    seed(engine, 30)
    res = repo(engine).assign_by_filter("rev-1", count=50)
    assert (res.matched_before, res.assigned, res.shortfall, res.remaining_eligible) == (30, 30, 20, 0)


def test_empty_eligible_result_assigns_nothing(engine):
    res = repo(engine).assign_by_filter("rev-1", count=10)
    assert (res.matched_before, res.assigned, res.shortfall) == (0, 0, 10) and events(engine) == []
    seed(engine, 5, intent="SPAM", topic="OFF_TOPIC_PERSONAL")
    res = repo(engine).assign_by_filter("rev-1", count=3, primary_intents=("COMPLAINT",))      # filter matches nothing
    assert (res.assigned, res.shortfall) == (0, 3) and all(v.reviewer_user_id is None for v in rows(engine).values())


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5"])
def test_invalid_counts_are_rejected(engine, bad):
    with pytest.raises(ValueError):
        repo(engine).assign_by_filter("rev-1", count=bad)


# ------------------------------------------------------------------------------------------------------------------------ filters, pagination, eligibility
def test_filtered_assignment_spans_pages_and_matches_the_assignable_listing(engine):
    seed(engine, 70, prefix="a", source_file="one.xlsx", intent="SUGGESTION")
    seed(engine, 70, prefix="b", source_file="two.xlsx", intent="SUGGESTION")
    seed(engine, 40, prefix="x", source_file="one.xlsx", intent="INQUIRY", start=100)
    r = repo(engine)
    listed, total = r.list_assignable(unassigned_only=True, source_file="one.xlsx", offset=0, limit=20, primary_intents=("SUGGESTION",))
    assert total == 70 and len(listed) == 20                                                   # one page shows 20, the match is 70
    res = r.assign_by_filter("rev-1", count=45, source_file="one.xlsx", primary_intents=("SUGGESTION",))
    assert res.matched_before == total and res.assigned == 45 and res.remaining_eligible == 25
    got = {k for k, v in rows(engine).items() if v.reviewer_user_id == "rev-1"}
    assert got == {f"a{i:04d}" for i in range(45)}                                             # only the filter's rows, deterministic prefix
    res2 = r.assign_by_filter("rev-2", count=None, source_file="one.xlsx", primary_intents=("SUGGESTION",))
    assert res2.assigned == 25 and not ({k for k, v in rows(engine).items() if v.reviewer_user_id == "rev-2"} & got)
    assert all(v.reviewer_user_id is None for k, v in rows(engine).items() if k.startswith(("b", "x")))


def test_existing_assignments_decisions_and_ineligible_rows_are_never_touched(engine):
    seed(engine, 10, prefix="p", assignee="rev-1")                                              # already assigned (pending)
    seed(engine, 5, prefix="d", assignee="rev-1", action="SEND_AS_IS", status="APPROVED")      # decided
    seed(engine, 3, prefix="e", status="EXCLUDED")
    seed(engine, 3, prefix="u", status="DUPLICATE_CONTENT")
    seed(engine, 12, prefix="n")                                                               # the only eligible rows
    before = {k: (v.reviewer_user_id, v.reviewer_action, v.review_status, v.row_version) for k, v in rows(engine).items()}
    res = repo(engine).assign_by_filter("rev-2", count=None)
    assert (res.matched_before, res.assigned) == (12, 12)
    after = {k: (v.reviewer_user_id, v.reviewer_action, v.review_status, v.row_version) for k, v in rows(engine).items()}
    for k in before:
        if k.startswith("n"):
            assert after[k] == ("rev-2", None, "DRAFT_READY", 2)
        else:
            assert after[k] == before[k], k                                                      # untouched, version included
    assert sorted(e.review_item_id for e in events(engine)) == sorted(k for k in before if k.startswith("n"))
    again = repo(engine).assign_by_filter("rev-3", count=5)
    assert again.assigned == 0 and again.shortfall == 5                                          # nothing left to take; rev-2's rows are not reassigned


def test_selected_id_assignment_still_works_alongside(engine):
    seed(engine, 6)
    svc = ReviewWorkflowService(engine)
    out = svc.assign_reviewer(["c0001", "c0003", "nope"], "rev-1", actor_user_id="adm", actor_role="ADMIN")
    assert out.assigned == ("c0001", "c0003") and out.not_found == ("nope",)
    res = repo(engine).assign_by_filter("rev-2", count=10)
    assert res.assigned == 4 and rows(engine)["c0001"].reviewer_user_id == "rev-1"               # count assignment skips what is already assigned
    out2 = svc.assign_reviewer(["c0001"], "rev-3", actor_user_id="adm", actor_role="ADMIN")        # explicit reassignment of a pending item is unchanged
    assert out2.assigned == ("c0001",) and rows(engine)["c0001"].reviewer_user_id == "rev-3"


# ------------------------------------------------------------------------------------------------------------------------ concurrency
@pytest.mark.parametrize("rounds", range(4))
def test_two_admins_assigning_concurrently_never_double_assign_or_lose_rows(tmp_path, rounds):
    path = tmp_path / "c.sqlite"
    e0 = get_engine(path)
    init_db(e0)
    seed(e0, 150)
    results, errors = {}, []
    barrier = threading.Barrier(2)

    def worker(name, reviewer, count):
        try:
            eng = get_engine(path)                                                               # its own engine/connection, like a separate request
            barrier.wait()
            results[name] = resolve_bundle(eng).review_items.assign_by_filter(reviewer, count=count, actor_user_id=name, actor_role="ADMIN")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    ts = [threading.Thread(target=worker, args=("admin-a", "rev-A", 100)), threading.Thread(target=worker, args=("admin-b", "rev-B", 100))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
    assert results["admin-a"].assigned + results["admin-b"].assigned == 150                      # every eligible row claimed exactly once
    r = rows(e0)
    by = {"rev-A": [k for k, v in r.items() if v.reviewer_user_id == "rev-A"], "rev-B": [k for k, v in r.items() if v.reviewer_user_id == "rev-B"]}
    assert len(by["rev-A"]) == results["admin-a"].assigned and len(by["rev-B"]) == results["admin-b"].assigned
    assert not set(by["rev-A"]) & set(by["rev-B"]) and all(v.reviewer_user_id for v in r.values())
    assert all(v.row_version == 2 for v in r.values())                                             # claimed once, not twice
    ev = events(e0)
    assert len(ev) == 150 and len({e.review_item_id for e in ev}) == 150                           # exactly one audit event per row
    assert {(e.review_item_id, e.actor_user_id) for e in ev} == {(k, "admin-a" if v == "rev-A" else "admin-b") for v, ks in by.items() for k in ks}
    short = [x for x in results.values() if x.shortfall]
    assert len(short) == 1 and short[0].shortfall == 50                                             # 200 requested, 150 available


def test_concurrent_count_assignment_and_selected_id_assignment_do_not_corrupt_each_other(tmp_path):
    path = tmp_path / "m.sqlite"
    e0 = get_engine(path)
    init_db(e0)
    seed(e0, 80)
    barrier = threading.Barrier(2)
    errors = []

    def by_count():
        try:
            eng = get_engine(path)
            barrier.wait()
            resolve_bundle(eng).review_items.assign_by_filter("rev-A", count=None)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    def by_ids():
        try:
            eng = get_engine(path)
            barrier.wait()
            ReviewWorkflowService(eng).assign_reviewer([f"c{i:04d}" for i in range(0, 80, 2)], "rev-B", actor_user_id="adm", actor_role="ADMIN")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    ts = [threading.Thread(target=by_count), threading.Thread(target=by_ids)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
    r = rows(e0)
    assert all(v.reviewer_user_id in ("rev-A", "rev-B") for v in r.values())
    ev = events(e0, "ASSIGNED") + events(e0, "REASSIGNED")
    final_assignee = {k: v.reviewer_user_id for k, v in r.items()}
    for cid in r:                                                                                 # the last audit event of every row agrees with its final assignee
        last = sorted((e for e in ev if e.review_item_id == cid), key=lambda e: (e.created_at, e.event_id))[-1]
        assert last.to_assignee_user_id == final_assignee[cid]


# ------------------------------------------------------------------------------------------------------------------------ HTTP: permissions, validation, reviewer isolation
class Api:
    def __init__(self, base):
        self.base = base

    def req(self, path, *, method="GET", data=None, cookie=None):
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        r = urllib.request.Request(self.base + path, data=None if data is None else json.dumps(data).encode(), headers=headers, method=method)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}"), resp.headers
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode() or "{}"), err.headers

    def login(self, email):
        _, _, h = self.req("/api/login", method="POST", data={"email": email, "password": "testpassword123"})
        return h.get("Set-Cookie").split(";")[0]


@pytest.fixture()
def http(tmp_path):
    e = get_engine(tmp_path / "h.sqlite")
    init_db(e)
    users = {}
    for key, role in (("admin", "ADMIN"), ("r1", "REVIEWER"), ("r2", "REVIEWER"), ("r3", "REVIEWER")):
        users[key] = create_user(e, email=f"{key}@test.local", password="testpassword123", display_name=key, role=role)
    seed(e, 60, prefix="a", source_file="one.xlsx", intent="SUGGESTION")
    seed(e, 40, prefix="b", source_file="two.xlsx", intent="INQUIRY")
    server, base = serve_in_thread(e)
    api = Api(base)
    ctx = type("Ctx", (), {})()
    ctx.engine, ctx.api, ctx.users = e, api, users
    ctx.cookie = {k: api.login(f"{k}@test.local") for k in users}
    yield ctx
    server.shutdown()


def test_http_count_assignment_reports_actual_and_shortfall(http):
    a, c = http.api, http.cookie
    s, out, _ = a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"],
                      data={"reviewer_user_id": http.users["r1"].user_id, "count": 25, "filters": {"source_file": "one.xlsx", "primary_intent": ["SUGGESTION"]}})
    assert s == 200 and (out["requested"], out["assigned"], out["shortfall"], out["matched_before"], out["remaining_eligible"]) == (25, 25, 0, 60, 35)
    s, out, _ = a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"],
                      data={"reviewer_user_id": http.users["r2"].user_id, "count": 500, "filters": {"source_file": "one.xlsx"}})
    assert s == 200 and (out["assigned"], out["shortfall"], out["remaining_eligible"]) == (35, 465, 0)
    s, out, _ = a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"], data={"reviewer_user_id": http.users["r3"].user_id, "all": True})
    assert s == 200 and (out["requested"], out["assigned"], out["shortfall"]) == (None, 40, 0)                # the rest (batch two)
    s, out, _ = a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"], data={"reviewer_user_id": http.users["r3"].user_id, "all": True})
    assert s == 200 and out["assigned"] == 0 and out["matched_before"] == 0


def test_http_reviewer_isolation_after_count_assignment(http):
    a, c = http.api, http.cookie
    a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"], data={"reviewer_user_id": http.users["r1"].user_id, "count": 12, "filters": {"source_file": "one.xlsx"}})
    a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"], data={"reviewer_user_id": http.users["r2"].user_id, "count": 7, "filters": {"source_file": "two.xlsx"}})
    s, q1, _ = a.req("/api/queue?limit=200", cookie=c["r1"])
    s2, q2, _ = a.req("/api/queue?limit=200", cookie=c["r2"])
    s3, q3, _ = a.req("/api/queue?limit=200", cookie=c["r3"])
    ids1, ids2 = {i["comment_id"] for i in q1["items"]}, {i["comment_id"] for i in q2["items"]}
    assert (len(ids1), len(ids2), q3["total"]) == (12, 7, 0) and not ids1 & ids2
    other = sorted(ids2)[0]
    assert a.req(f"/api/items/{other}", cookie=c["r1"])[0] in (403, 404)
    assert a.req(f"/api/items/{other}/claim", method="POST", data={}, cookie=c["r1"])[0] in (403, 404)
    unassigned = next(k for k in rows(http.engine) if rows(http.engine)[k].reviewer_user_id is None)
    assert a.req(f"/api/items/{unassigned}", cookie=c["r1"])[0] in (403, 404)


def test_http_permissions_and_validation(http):
    a, c = http.api, http.cookie
    r1 = http.users["r1"].user_id
    body = {"reviewer_user_id": r1, "count": 3}
    assert a.req("/api/admin/assign-by-filter", method="POST", data=body)[0] == 401
    assert a.req("/api/admin/assign-by-filter", method="POST", cookie=c["r2"], data=body)[0] == 403
    assert all(v.reviewer_user_id is None for v in rows(http.engine).values())
    bad = [
        {"reviewer_user_id": "nobody", "count": 3}, {"reviewer_user_id": http.users["admin"].user_id, "count": 3},         # unknown / not a reviewer
        {"reviewer_user_id": r1}, {"reviewer_user_id": r1, "count": 3, "all": True},                                         # neither / both
        {"reviewer_user_id": r1, "count": 0}, {"reviewer_user_id": r1, "count": -2}, {"reviewer_user_id": r1, "count": "5"},
        {"reviewer_user_id": r1, "count": True}, {"reviewer_user_id": r1, "count": 2.5}, {"reviewer_user_id": r1, "all": "yes"},
        {"reviewer_user_id": r1, "count": 3, "filters": {"reviewer_user_id": "x"}}, {"reviewer_user_id": r1, "count": 3, "filters": "x"},
    ]
    for payload in bad:
        s, out, _ = a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"], data=payload)
        assert s == 400, (payload, s, out)
    s, out, _ = a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"], data={"reviewer_user_id": r1, "count": 3, "filters": {"primary_intent": "NOT_A_CLASS"}})
    assert s == 400 and out["error"] == "invalid_filter"
    assert all(v.reviewer_user_id is None for v in rows(http.engine).values())                                             # nothing changed by any rejected call


def test_http_inactive_reviewer_is_rejected_and_existing_id_endpoint_is_unchanged(http):
    a, c = http.api, http.cookie
    s, out, _ = a.req("/api/admin/assign", method="POST", cookie=c["admin"], data={"comment_ids": ["a0000", "a0001"], "reviewer_user_id": http.users["r1"].user_id})
    assert s == 200 and out["assigned"] == ["a0000", "a0001"]
    s, out, _ = a.req("/api/admin/assign-by-filter", method="POST", cookie=c["admin"], data={"reviewer_user_id": http.users["r2"].user_id, "count": 5, "filters": {"source_file": "one.xlsx"}})
    assert out["assigned"] == 5 and rows(http.engine)["a0000"].reviewer_user_id == http.users["r1"].user_id


def test_http_concurrent_admins_through_the_server(http):
    a, c = http.api, http.cookie
    admin2 = a.login("admin@test.local")
    out = {}
    barrier = threading.Barrier(2)

    def go(name, cookie, reviewer, count):
        barrier.wait()
        out[name] = a.req("/api/admin/assign-by-filter", method="POST", cookie=cookie, data={"reviewer_user_id": http.users[reviewer].user_id, "count": count})

    ts = [threading.Thread(target=go, args=("x", c["admin"], "r1", 70)), threading.Thread(target=go, args=("y", admin2, "r2", 70))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out["x"][0] == 200 and out["y"][0] == 200
    assert out["x"][1]["assigned"] + out["y"][1]["assigned"] == 100
    r = rows(http.engine)
    assert sum(1 for v in r.values() if v.reviewer_user_id == http.users["r1"].user_id) == out["x"][1]["assigned"]
    assert sum(1 for v in r.values() if v.reviewer_user_id == http.users["r2"].user_id) == out["y"][1]["assigned"]
    assert len(events(http.engine)) == 100


# ------------------------------------------------------------------------------------------------------------------------ MongoDB (in-memory fake only)
def test_mongo_repository_contract_with_the_in_memory_fake():
    from test_mongo_repository_unit import MongoReviewItemRepository, _FakeDatabase, _item_doc

    db = _FakeDatabase()
    coll = db["review_items"]
    for i in range(30):
        coll.insert_one(_item_doc(f"m{i:03d}", source_row_index=i, ingested_at=T0 + dt.timedelta(minutes=i)))
    coll.insert_one(_item_doc("assigned", reviewer_user_id="rev-0"))
    coll.insert_one(_item_doc("decided", reviewer_action="SEND_AS_IS", review_status="APPROVED"))
    coll.insert_one(_item_doc("excluded", review_status="EXCLUDED"))
    repo_ = MongoReviewItemRepository(db)
    res = repo_.assign_by_filter("rev-1", count=12, actor_user_id="adm", actor_role="ADMIN")
    assert (res.matched_before, res.assigned, res.shortfall, res.remaining_eligible) == (30, 12, 0, 18)
    mine = sorted(d["_id"] for d in coll.find({"reviewer_user_id": "rev-1"}))
    assert mine == [f"m{i:03d}" for i in range(12)]
    assert coll.find_one({"_id": "assigned"})["reviewer_user_id"] == "rev-0" and coll.find_one({"_id": "decided"})["reviewer_user_id"] is None
    res2 = repo_.assign_by_filter("rev-2", count=None)
    assert res2.assigned == 18 and res2.remaining_eligible == 0
    res3 = repo_.assign_by_filter("rev-3", count=5)
    assert (res3.assigned, res3.shortfall) == (0, 5)
    assert len(list(db["review_item_events"].find({}))) == 30
