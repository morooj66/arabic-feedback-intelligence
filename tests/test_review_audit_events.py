"""Append-only review_item_events audit log (Phase 2).

Covers: exactly-once events per successful action, no event on failed /
stale / unauthorized / idempotent repeats, atomic rollback when the audit
write fails (PostgreSQL and the Mongo transactional path), the admin-only
history endpoint, the no-PII guarantee, no automatic dispatch, and the
Alembic migration (applied only to an isolated temp SQLite database).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

import rcs.review.repositories.mongo as mongo_mod
import rcs.review.repositories.postgres as pg_mod
from rcs.review.auth import ADMIN, REVIEWER, create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.identity import ReviewerIdentity
from rcs.review.models import Base, ReviewItem, ReviewItemEvent
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.repositories.mongo import MongoReviewItemRepository
from rcs.review.service import ReviewWorkflowService
from rcs.review.workflow import InvalidTransition, ItemNotFound, VersionConflict
from test_mongo_repository_unit import _FakeDatabase, _item_doc

ROOT = Path(__file__).resolve().parents[1]
COMMENT_TEXT = "نص تعليق الزائر الحساس"
REPLY_TEXT = "رد المراجع الكامل"
REASON_TEXT = "سبب حر حساس"


def _engine(tmp_path):
    engine = get_engine(tmp_path / "audit.sqlite")
    init_db(engine)
    return engine


def _row(comment_id, **overrides):
    row = dict(
        comment_id=comment_id,
        source_file="audit.xlsx",
        source_row_index=0,
        original_comment=COMMENT_TEXT,
        review_status="DRAFT_READY",
        suggested_final_reply="رد مقترح",
        ingested_at=dt.datetime(2026, 8, 1, 9, 0, 0),
        ready_to_send=False,
        row_version=1,
    )
    row.update(overrides)
    return row


def _events(engine, comment_id=None):
    with Session(engine) as s:
        stmt = select(ReviewItemEvent).order_by(ReviewItemEvent.created_at, ReviewItemEvent.event_id)
        if comment_id:
            stmt = stmt.where(ReviewItemEvent.review_item_id == comment_id)
        return list(s.execute(stmt).scalars())


def _ident(user_id="u-r1", role="REVIEWER"):
    return ReviewerIdentity(reviewer_id=user_id, display_name="R", user_id=user_id, role=role)


def _item(engine, comment_id):
    with Session(engine) as s:
        it = s.get(ReviewItem, comment_id)
        s.expunge(it)
        return it


# ------------------------------------------------------------- PostgreSQL path


def test_each_successful_action_writes_exactly_one_event(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(engine, [_row("c1"), _row("c2"), _row("c3"), _row("c4")])
    svc = ReviewWorkflowService(engine)
    r = _ident()

    svc.claim_review("c1", r)
    svc.claim_review("c1", r)  # idempotent repeat: no second event
    assert [e.event_type for e in _events(engine, "c1")] == ["CLAIMED"]

    svc.send_as_is("c2", r)
    svc.send_as_is("c2", r)  # idempotent repeat
    svc.edit_and_send("c3", r, REPLY_TEXT)
    svc.edit_and_send("c3", r, REPLY_TEXT)  # same body repeat
    svc.mark_no_response_needed("c4", r, reason=REASON_TEXT)
    svc.mark_no_response_needed("c4", r, reason=REASON_TEXT)

    for cid, etype, to_status in (
        ("c2", "SEND_AS_IS", "APPROVED"),
        ("c3", "EDIT_AND_SEND", "EDITED_AND_APPROVED"),
        ("c4", "NO_RESPONSE_NEEDED", "NO_REPLY"),
    ):
        evs = _events(engine, cid)
        assert len(evs) == 1, cid
        e = evs[0]
        assert (e.event_type, e.from_status, e.to_status) == (etype, "DRAFT_READY", to_status)
        assert e.actor_user_id == "u-r1" and e.actor_role == "REVIEWER"
        assert e.to_assignee_user_id == "u-r1" and e.from_assignee_user_id is None
        assert e.overrode_recommendation is False
        assert e.created_at is not None


def test_override_of_recommendation_is_flagged(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(
        engine,
        [
            _row("n1", review_status="NO_RESPONSE_RECOMMENDED", suggested_final_reply=None, reviewer_user_id="u-r1"),
            _row("n2", review_status="NO_RESPONSE_RECOMMENDED", suggested_final_reply=None, reviewer_user_id="u-r1"),
        ],
    )
    svc = ReviewWorkflowService(engine)
    svc.edit_and_send("n1", _ident(), REPLY_TEXT)
    svc.mark_no_response_needed("n2", _ident(), reason=REASON_TEXT)
    (e1,) = _events(engine, "n1")
    assert (e1.from_status, e1.to_status, e1.overrode_recommendation) == (
        "NO_RESPONSE_RECOMMENDED", "EDITED_AND_APPROVED", True)
    (e2,) = _events(engine, "n2")
    assert (e2.from_status, e2.to_status, e2.overrode_recommendation) == (
        "NO_RESPONSE_RECOMMENDED", "NO_REPLY", False)


def test_assignment_and_reassignment_events(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(engine, [_row("a1"), _row("a2", reviewer_action="SEND_AS_IS", review_status="APPROVED")])
    svc = ReviewWorkflowService(engine)
    svc.assign_reviewer(["a1", "a2", "missing"], "u-A", actor_user_id="admin-1", actor_role="ADMIN")
    svc.assign_reviewer(["a1"], "u-A", actor_user_id="admin-1", actor_role="ADMIN")  # no change -> no event
    svc.assign_reviewer(["a1"], "u-B", actor_user_id="admin-1", actor_role="ADMIN")
    evs = _events(engine, "a1")
    assert [(e.event_type, e.from_assignee_user_id, e.to_assignee_user_id) for e in evs] == [
        ("ASSIGNED", None, "u-A"),
        ("REASSIGNED", "u-A", "u-B"),
    ]
    assert all(e.actor_user_id == "admin-1" and e.actor_role == "ADMIN" for e in evs)
    assert _events(engine, "a2") == []  # decided item skipped
    assert len(_events(engine)) == 2


def test_assignment_without_actor_still_records_event(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(engine, [_row("s1")])
    ReviewWorkflowService(engine).assign_reviewer(["s1"], "u-A")
    (e,) = _events(engine, "s1")
    assert e.actor_user_id == "system"


def test_failed_stale_and_forbidden_actions_write_no_event(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(
        engine,
        [_row("f1"), _row("f2", suggested_final_reply=None), _row("f3", review_status="IN_REVIEW", reviewer_id="other")],
    )
    svc = ReviewWorkflowService(engine)
    r = _ident()
    with pytest.raises(VersionConflict):
        svc.claim_review("f1", r, expected_version=99)
    with pytest.raises(VersionConflict):
        svc.edit_and_send("f1", r, REPLY_TEXT, expected_version=99)
    with pytest.raises(InvalidTransition):
        svc.send_as_is("f2", r)  # no draft
    with pytest.raises(InvalidTransition):
        svc.edit_and_send("f3", r, REPLY_TEXT)  # claimed by someone else
    with pytest.raises(InvalidTransition):
        svc.edit_and_send("f1", r, "   ")
    with pytest.raises(ItemNotFound):
        svc.claim_review("nope", r)
    assert _events(engine) == []
    assert _item(engine, "f1").review_status == "DRAFT_READY"


def test_audit_write_failure_rolls_back_the_action(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    upsert_review_items(engine, [_row("r1"), _row("r2"), _row("r3"), _row("r4")])
    svc = ReviewWorkflowService(engine)

    def boom(*a, **k):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(pg_mod, "_add_event", boom)
    r = _ident()
    for call in (
        lambda: svc.claim_review("r1", r),
        lambda: svc.send_as_is("r1", r),
        lambda: svc.edit_and_send("r2", r, REPLY_TEXT),
        lambda: svc.mark_no_response_needed("r3", r, reason=REASON_TEXT),
        lambda: svc.assign_reviewer(["r4"], "u-A", actor_user_id="admin-1", actor_role="ADMIN"),
    ):
        with pytest.raises(RuntimeError):
            call()
    monkeypatch.undo()
    assert _events(engine) == []
    for cid in ("r1", "r2", "r3", "r4"):
        it = _item(engine, cid)
        assert it.review_status == "DRAFT_READY" and it.reviewer_action is None
        assert it.row_version == 1 and it.reviewer_user_id is None and it.final_reply is None


def test_batch_assignment_is_all_or_nothing_when_audit_fails_midway(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    upsert_review_items(engine, [_row("b1"), _row("b2")])
    svc = ReviewWorkflowService(engine)
    real = pg_mod._add_event
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second audit write fails")
        return real(*a, **k)

    monkeypatch.setattr(pg_mod, "_add_event", flaky)
    with pytest.raises(RuntimeError):
        svc.assign_reviewer(["b1", "b2"], "u-A", actor_user_id="admin-1", actor_role="ADMIN")
    monkeypatch.undo()
    assert _events(engine) == []
    assert _item(engine, "b1").reviewer_user_id is None


def test_events_contain_no_visitor_text_reply_text_or_reason(tmp_path):
    engine = _engine(tmp_path)
    upsert_review_items(engine, [_row("p1"), _row("p2")])
    svc = ReviewWorkflowService(engine)
    svc.edit_and_send("p1", _ident(), REPLY_TEXT)
    svc.mark_no_response_needed("p2", _ident(), reason=REASON_TEXT)
    blob = json.dumps(
        [{c.name: str(getattr(e, c.name)) for c in ReviewItemEvent.__table__.columns} for e in _events(engine)],
        ensure_ascii=False,
    )
    for secret in (COMMENT_TEXT, REPLY_TEXT, REASON_TEXT, "رد مقترح"):
        assert secret not in blob
    assert {c.name for c in ReviewItemEvent.__table__.columns} == {
        "event_id", "review_item_id", "event_type", "actor_user_id", "actor_role", "from_status",
        "to_status", "from_assignee_user_id", "to_assignee_user_id", "overrode_recommendation", "created_at",
    }


def test_repository_exposes_no_update_or_delete_for_events():
    for repo_cls in (pg_mod.PostgresReviewItemRepository, MongoReviewItemRepository):
        public = {n for n in dir(repo_cls) if "event" in n.lower() and not n.startswith("__")}
        assert not any(n.startswith(("update", "delete", "remove", "clear")) for n in public)


# ------------------------------------------------------------------------ HTTP


def _json(url, *, method="GET", data=None, cookie=None):
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


def test_http_history_is_admin_only_and_actions_are_audited(tmp_path):
    engine = _engine(tmp_path)
    admin = create_user(engine, email="adm@x.com", password="realpassword123", display_name="Adm", role=ADMIN)
    a = create_user(engine, email="ra@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="rb@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    upsert_review_items(
        engine,
        [
            _row("h-nrr", review_status="NO_RESPONSE_RECOMMENDED", suggested_final_reply=None),
            _row("h-other"),
        ],
    )
    server, base = serve_in_thread(engine)
    try:
        ca = _login(base, "ra@x.com", "realpassword123")
        cb = _login(base, "rb@x.com", "realpassword123")
        cadm = _login(base, "adm@x.com", "realpassword123")

        # Admin assigns -> ASSIGNED event with the admin as actor.
        st, out, _ = _json(f"{base}/api/admin/assign", method="POST", cookie=cadm,
                           data={"comment_ids": ["h-nrr", "h-other"], "reviewer_user_id": a.user_id})
        assert st == 200 and sorted(out["assigned"]) == ["h-nrr", "h-other"]

        # Unassigned reviewer B: forbidden, no event, nothing changes.
        before = len(_events(engine))
        st, _, _ = _json(f"{base}/api/items/h-nrr/edit-and-send", method="POST", cookie=cb,
                         data={"edited_response": REPLY_TEXT, "expected_version": 2})
        assert st == 403
        assert len(_events(engine)) == before

        # Stale version by the assigned reviewer: 409, no event.
        st, _, _ = _json(f"{base}/api/items/h-nrr/edit-and-send", method="POST", cookie=ca,
                         data={"edited_response": REPLY_TEXT, "expected_version": 1})
        assert st == 409
        assert len(_events(engine)) == before

        # Assigned reviewer overrides the recommendation.
        st, detail, _ = _json(f"{base}/api/items/h-nrr/edit-and-send", method="POST", cookie=ca,
                              data={"edited_response": REPLY_TEXT, "expected_version": 2})
        assert st == 200 and detail["review_status"] == "EDITED_AND_APPROVED"

        # No automatic dispatch: queued as PENDING, nothing sent.
        row = _item(engine, "h-nrr")
        assert row.outbound_status == "PENDING"
        assert row.outbound_sent_at is None and row.outbound_external_id is None

        # History: admin only.
        st, hist, _ = _json(f"{base}/api/admin/items/h-nrr/history", cookie=cadm)
        assert st == 200
        assert [(e["event_type"], e["actor_role"], e["overrode_recommendation"]) for e in hist["events"]] == [
            ("ASSIGNED", "ADMIN", False),
            ("EDIT_AND_SEND", "REVIEWER", True),
        ]
        assert hist["events"][1]["actor_user_id"] == a.user_id
        assert REPLY_TEXT not in json.dumps(hist, ensure_ascii=False)
        assert COMMENT_TEXT not in json.dumps(hist, ensure_ascii=False)
        for cookie in (ca, cb):
            st, _, _ = _json(f"{base}/api/admin/items/h-nrr/history", cookie=cookie)
            assert st == 403
        st, _, _ = _json(f"{base}/api/admin/items/h-nrr/history")
        assert st == 401
        st, _, _ = _json(f"{base}/api/admin/items/does-not-exist/history", cookie=cadm)
        assert st == 404
    finally:
        server.shutdown()


# ------------------------------------------------------------------- MongoDB


def _mongo():
    db = _FakeDatabase()
    return db, MongoReviewItemRepository(db)


def _mongo_events(db, comment_id):
    return [d for d in db["review_item_events"]._docs if d["review_item_id"] == comment_id]


def test_mongo_actions_write_one_event_in_the_transaction():
    db, repo = _mongo()
    for cid in ("m1", "m2", "m3", "m4"):
        db["review_items"].insert_one(_item_doc(cid))
    db["review_items"].insert_one(
        _item_doc("m5", review_status="NO_RESPONSE_RECOMMENDED", suggested_final_reply=None, reviewer_user_id="u1")
    )
    kw = dict(reviewer_id="u1", display_name="R", user_id="u1", expected_version=None, actor_role="REVIEWER")
    repo.claim("m1", **kw)
    repo.claim("m1", **kw)  # idempotent
    repo.send_as_is("m2", **kw)
    repo.send_as_is("m2", **kw)
    repo.mark_no_response_needed("m3", reason=REASON_TEXT, **kw)
    repo.edit_and_send("m5", edited_response=REPLY_TEXT, **kw)
    for cid, etype in (("m1", "CLAIMED"), ("m2", "SEND_AS_IS"), ("m3", "NO_RESPONSE_NEEDED"), ("m5", "EDIT_AND_SEND")):
        evs = _mongo_events(db, cid)
        assert [e["event_type"] for e in evs] == [etype], cid
        assert evs[0]["actor_user_id"] == "u1" and evs[0]["actor_role"] == "REVIEWER"
    assert _mongo_events(db, "m5")[0]["overrode_recommendation"] is True
    assert [e.event_type for e in repo.list_events("m5")] == ["EDIT_AND_SEND"]
    blob = json.dumps(db["review_item_events"]._docs, ensure_ascii=False, default=str)
    assert REPLY_TEXT not in blob and REASON_TEXT not in blob


def test_mongo_failed_stale_actions_write_no_event():
    db, repo = _mongo()
    db["review_items"].insert_one(_item_doc("x1"))
    db["review_items"].insert_one(_item_doc("x2", suggested_final_reply=None))
    kw = dict(reviewer_id="u1", display_name="R", user_id="u1")
    with pytest.raises(VersionConflict):
        repo.claim("x1", expected_version=9, **kw)
    with pytest.raises(InvalidTransition):
        repo.send_as_is("x2", expected_version=None, **kw)
    with pytest.raises(ItemNotFound):
        repo.claim("nope", expected_version=None, **kw)
    assert db["review_item_events"]._docs == []


def test_mongo_audit_failure_aborts_the_transaction(monkeypatch):
    db, repo = _mongo()
    db["review_items"].insert_one(_item_doc("y1"))

    def boom(**_):
        raise RuntimeError("audit insert failed")

    monkeypatch.setattr(mongo_mod, "build_event_fields", boom)
    with pytest.raises(RuntimeError):
        repo.edit_and_send("y1", reviewer_id="u1", display_name="R", user_id="u1",
                           edited_response=REPLY_TEXT, expected_version=None)
    monkeypatch.undo()
    doc = db["review_items"].find_one({"_id": "y1"})
    assert doc["review_status"] == "DRAFT_READY" and doc["row_version"] == 1 and doc["final_reply"] is None
    assert db["review_item_events"]._docs == []


def test_mongo_assign_batch_events_and_atomic_rollback(monkeypatch):
    db, repo = _mongo()
    for cid in ("z1", "z2"):
        db["review_items"].insert_one(_item_doc(cid))
    repo.assign_batch(["z1"], "u-A", actor_user_id="admin-1", actor_role="ADMIN")
    repo.assign_batch(["z1"], "u-A", actor_user_id="admin-1", actor_role="ADMIN")
    repo.assign_batch(["z1"], "u-B", actor_user_id="admin-1", actor_role="ADMIN")
    assert [(e["event_type"], e["from_assignee_user_id"], e["to_assignee_user_id"])
            for e in _mongo_events(db, "z1")] == [("ASSIGNED", None, "u-A"), ("REASSIGNED", "u-A", "u-B")]

    real = mongo_mod.build_event_fields
    n = {"c": 0}

    def flaky(**kw):
        n["c"] += 1
        if n["c"] == 2:
            raise RuntimeError("fail on second")
        return real(**kw)

    monkeypatch.setattr(mongo_mod, "build_event_fields", flaky)
    db["review_items"].insert_one(_item_doc("z3"))
    with pytest.raises(RuntimeError):
        repo.assign_batch(["z2", "z3"], "u-C", actor_user_id="admin-1", actor_role="ADMIN")
    monkeypatch.undo()
    assert db["review_items"].find_one({"_id": "z2"})["reviewer_user_id"] is None
    assert _mongo_events(db, "z2") == []


# ------------------------------------------------------------------ migration


def _load_migration():
    path = next((ROOT / "migrations" / "versions").glob("a7c1e4b92d10_*.py"))
    spec = importlib.util.spec_from_file_location("mig_a7c1e4b92d10", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_is_single_head_on_top_of_verified_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    script = ScriptDirectory.from_config(cfg)
    assert script.get_heads() == ["d5f1b3c8a2e6"]                                            # account_requests on direct_messages on password_reset_tokens on the audit-log revision
    assert script.get_revision("d5f1b3c8a2e6").down_revision == "c4e8a2d17f35"
    assert script.get_revision("c4e8a2d17f35").down_revision == "b3d9f0a1c7e2"
    assert script.get_revision("b3d9f0a1c7e2").down_revision == "a7c1e4b92d10"
    assert script.get_revision("a7c1e4b92d10").down_revision == "5bc30c06b065"


def test_migration_upgrade_and_downgrade_on_isolated_sqlite(tmp_path):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    mig = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'mig.sqlite'}")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mig.upgrade()
    insp = inspect(engine)
    assert "review_item_events" in insp.get_table_names()
    cols = {c["name"]: c for c in insp.get_columns("review_item_events")}
    assert set(cols) == {c.name for c in ReviewItemEvent.__table__.columns}
    assert {n for n, c in cols.items() if not c["nullable"]} == {
        "event_id", "review_item_id", "event_type", "actor_user_id", "overrode_recommendation", "created_at"}
    assert {i["name"] for i in insp.get_indexes("review_item_events")} == {
        "ix_review_item_events_item_created", "ix_review_item_events_actor_created"}
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mig.downgrade()
    assert "review_item_events" not in inspect(engine).get_table_names()
