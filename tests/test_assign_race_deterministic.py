"""Deterministic regression tests for the select-by-id assignment race (assign_batch).

A competing write is injected exactly between assign_batch's read of a row and its write, which is the window the old code lost updates in. No threads, no
timing: the interleaving is forced, so these tests fail on the old read-modify-write code and pass on the compare-and-set implementation."""
from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

import rcs.review.repositories.postgres as pg
from rcs.review.models import ReviewItem, ReviewItemEvent
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService

T0 = dt.datetime(2026, 8, 1, 9, 0, 0)


@pytest.fixture
def env(tmp_path):
    path = tmp_path / "race.sqlite"
    eng = get_engine(path)
    init_db(eng)
    with Session(eng) as s:
        for i in range(3):
            s.add(ReviewItem(comment_id=f"c{i}", source_file="s.xlsx", source_row_index=i, original_comment=f"تعليق تجريبي رقم {i}", review_status="DRAFT_READY",
                             ai_primary_intent="SUGGESTION", primary_intent="SUGGESTION", ai_topic_subtype="GENERAL", topic_subtype="GENERAL", ingested_at=T0, row_version=1))
        s.commit()
    return eng, path


def inject_after_first_read(monkeypatch, engine_factory, action):
    """After assign_batch reads `c0` for the first time, run `action(session_on_another_connection)` once, then continue."""
    state = {"fired": False}
    real_session = pg.Session

    class HookedSession(real_session):
        def get(self, entity, ident, *a, **kw):
            row = super().get(entity, ident, *a, **kw)
            if not state["fired"] and ident == "c0":
                state["fired"] = True
                with real_session(engine_factory()) as other:
                    action(other)
                    other.commit()
            return row

    monkeypatch.setattr(pg, "Session", HookedSession)
    return state


def assign(engine, ids, reviewer):
    return ReviewWorkflowService(engine).assign_reviewer(ids, reviewer, actor_user_id="adm", actor_role="ADMIN")


def item(engine, cid):
    with Session(engine) as s:
        r = s.get(ReviewItem, cid)
        s.expunge(r)
        return r


def evs(engine, cid):
    with Session(engine) as s:
        out = s.query(ReviewItemEvent).filter(ReviewItemEvent.review_item_id == cid).order_by(ReviewItemEvent.created_at, ReviewItemEvent.event_id).all()
        for e in out:
            s.expunge(e)
        return out


def competitor_assigns(to):
    def run(other):
        other.execute(update(ReviewItem).where(ReviewItem.comment_id == "c0").values(reviewer_user_id=to, row_version=ReviewItem.row_version + 1))
        other.add(ReviewItemEvent(event_id=f"evt-competitor-{to}", review_item_id="c0", event_type="ASSIGNED", actor_user_id="other-admin", actor_role="ADMIN", from_status="DRAFT_READY",
                                  to_status="DRAFT_READY", from_assignee_user_id=None, to_assignee_user_id=to, overrode_recommendation=False, created_at=dt.datetime.now()))
    return run


def test_a_concurrent_assignment_between_read_and_write_is_not_overwritten_from_a_stale_read(env, monkeypatch):
    eng, _ = env
    inject_after_first_read(monkeypatch, lambda: eng, competitor_assigns("rev-A"))
    res = assign(eng, ["c0", "c1"], "rev-B")
    assert set(res.assigned) == {"c0", "c1"}
    r = item(eng, "c0")
    assert r.reviewer_user_id == "rev-B"                                       # the batch re-read, saw rev-A, and reassigned deliberately
    assert r.row_version == 3                                                  # 1 -> 2 (competitor) -> 3 (this batch): no lost version bump
    chain = [(e.event_type, e.from_assignee_user_id, e.to_assignee_user_id) for e in evs(eng, "c0")]
    assert ("REASSIGNED", "rev-A", "rev-B") in chain and chain[-1] == ("REASSIGNED", "rev-A", "rev-B") or chain[-1][2] == "rev-B"
    mine = [e for e in evs(eng, "c0") if e.actor_user_id == "adm"]
    assert len(mine) == 1 and mine[0].from_assignee_user_id == "rev-A" and mine[0].to_assignee_user_id == "rev-B"   # not the stale "from None"
    last = sorted(evs(eng, "c0"), key=lambda e: (e.created_at, e.event_id))[-1]
    assert last.to_assignee_user_id == r.reviewer_user_id                      # the newest audit event agrees with the final state
    assert item(eng, "c1").row_version == 2 and item(eng, "c1").reviewer_user_id == "rev-B"


def test_a_concurrent_assignment_to_the_same_reviewer_is_a_no_op_not_a_duplicate_event(env, monkeypatch):
    eng, _ = env
    inject_after_first_read(monkeypatch, lambda: eng, competitor_assigns("rev-B"))
    res = assign(eng, ["c0"], "rev-B")
    assert res.assigned == ("c0",)
    r = item(eng, "c0")
    assert r.reviewer_user_id == "rev-B" and r.row_version == 2                # only the competitor's bump
    assert [e.actor_user_id for e in evs(eng, "c0")] == ["other-admin"]        # no second event for a change that did not happen


def test_a_decision_made_between_read_and_write_is_never_overwritten(env, monkeypatch):
    eng, _ = env

    def decide(other):
        other.execute(update(ReviewItem).where(ReviewItem.comment_id == "c0").values(reviewer_user_id="rev-A", reviewer_action="SEND_AS_IS", review_status="APPROVED",
                                                                                     row_version=ReviewItem.row_version + 1))

    inject_after_first_read(monkeypatch, lambda: eng, decide)
    res = assign(eng, ["c0", "c1"], "rev-B")
    assert res.assigned == ("c1",) and res.skipped_already_decided == ("c0",)
    r = item(eng, "c0")
    assert r.reviewer_user_id == "rev-A" and r.reviewer_action == "SEND_AS_IS" and r.row_version == 2
    assert evs(eng, "c0") == []                                                # nothing recorded for an assignment that did not happen


def test_unraced_behaviour_is_unchanged(env):
    eng, _ = env
    res = assign(eng, ["c0", "c1", "nope"], "rev-A")
    assert res.assigned == ("c0", "c1") and res.not_found == ("nope",)
    assert [item(eng, c).row_version for c in ("c0", "c1", "c2")] == [2, 2, 1]
    res = assign(eng, ["c0"], "rev-A")                                         # same reviewer again: idempotent, no event, no bump
    assert res.assigned == ("c0",) and item(eng, "c0").row_version == 2 and len(evs(eng, "c0")) == 1
    assign(eng, ["c0"], "rev-B")
    assert [(e.event_type, e.from_assignee_user_id, e.to_assignee_user_id) for e in evs(eng, "c0")] == [("ASSIGNED", None, "rev-A"), ("REASSIGNED", "rev-A", "rev-B")]
    assert item(eng, "c0").row_version == 3
