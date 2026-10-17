"""Filters, deterministic ordering and pagination for the queue / assignment lists.

Correctness is checked against a plain-Python oracle over the seeded rows
(not just HTTP 200), on PostgreSQL (SQLite engine) and the Mongo repository
(in-memory fake), which must return identical ordered ids.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

import pytest
from sqlalchemy.orm import Session

from rcs.review.auth import ADMIN, REVIEWER, create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem
from rcs.review.repositories.mongo import MongoReviewItemRepository
from rcs.review.repositories.postgres import PostgresReviewItemRepository
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from test_mongo_repository_unit import _FakeDatabase, _item_doc

T0 = dt.datetime(2026, 5, 1, 9, 0, 0)  # shared by many rows: forces ties
T1 = dt.datetime(2026, 5, 2, 9, 0, 0)


def D(y, m, d):
    return dt.datetime(y, m, d, 12, 0, 0)


# id, intent, topic, sentiment, priority, comment_created_at, ingested_at, status, reviewer, reviewer_action
ROWS = [
    ("r01", "COMPLAINT", "TECHNICAL_ISSUE", "NEGATIVE", 1, D(2026, 1, 5), T0, "DRAFT_READY", "A", None),
    ("r02", "COMPLAINT", "GENERAL", "NEGATIVE", 1, D(2026, 1, 10), T0, "DRAFT_READY", "A", None),
    ("r03", "SUGGESTION", "DICTIONARY_SUGGESTION", "NEUTRAL", 3, D(2026, 2, 1), T0, "DRAFT_READY", "A", None),
    ("r04", "SUGGESTION", "WORD_SUBMISSION", "POSITIVE", 3, D(2026, 2, 15), T0, "REVIEW_ONLY", "A", None),
    ("r05", "INQUIRY", "GENERAL", "NEUTRAL", 2, D(2026, 3, 1), T0, "IN_REVIEW", "B", None),
    ("r06", "INQUIRY", "ENTRY_CORRECTION", "NEUTRAL", 2, D(2026, 3, 20), T0, "DRAFT_READY", "B", None),
    ("r07", "THANKS_PRAISE", "GENERAL", "POSITIVE", 3, D(2026, 4, 2), T1, "NO_RESPONSE_RECOMMENDED", "A", None),
    ("r08", "THANKS_PRAISE", "GENERAL", "POSITIVE", 3, None, T1, "DRAFT_READY", "A", None),
    ("r09", "SPAM", "OFF_TOPIC_PERSONAL", "NEUTRAL", 3, D(2026, 4, 10), T1, "APPROVED", "B", "SEND_AS_IS"),
    ("r10", "COMPLAINT", "TECHNICAL_ISSUE", "NEGATIVE", 1, D(2026, 4, 12), T1, "DRAFT_READY", None, None),
    ("r11", "SUGGESTION", "DICTIONARY_SUGGESTION", None, 3, D(2026, 4, 12), T1, "DRAFT_READY", None, None),
    ("r12", "INQUIRY", "TECHNICAL_ISSUE", "NEGATIVE", 1, D(2026, 1, 10), T1, "DRAFT_READY", "A", None),
]
PENDING = ("DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW", "NO_RESPONSE_RECOMMENDED")
ALL_SORTS = ("priority_oldest", "priority_newest", "newest", "oldest", "comment_newest", "comment_oldest")


def _users(engine):
    a = create_user(engine, email="a@x.com", password="realpassword123", display_name="A", role=REVIEWER)
    b = create_user(engine, email="b@x.com", password="realpassword123", display_name="B", role=REVIEWER)
    create_user(engine, email="adm@x.com", password="realpassword123", display_name="Adm", role=ADMIN)
    return {"A": a.user_id, "B": b.user_id, None: None}


def _pg_dicts(uid):
    return [
        dict(
            comment_id=i, source_file="f.xlsx", source_row_index=n, original_comment=f"c {i}",
            primary_intent=it, topic_subtype=tp, sentiment=se, review_priority=pr,
            comment_created_at=cc, ingested_at=ing, review_status=st,
            reviewer_user_id=uid[rv], reviewer_action=ra, suggested_final_reply="x", row_version=1,
        )
        for n, (i, it, tp, se, pr, cc, ing, st, rv, ra) in enumerate(ROWS)
    ]


def _mongo_docs(uid):
    return [
        _item_doc(
            i, source_row_index=n, primary_intent=it, topic_subtype=tp, sentiment=se, review_priority=pr,
            comment_created_at=cc, ingested_at=ing, review_status=st, reviewer_user_id=uid[rv], reviewer_action=ra,
        )
        for n, (i, it, tp, se, pr, cc, ing, st, rv, ra) in enumerate(ROWS)
    ]


@pytest.fixture
def env(tmp_path):
    engine = get_engine(tmp_path / "qf.sqlite")
    init_db(engine)
    uid = _users(engine)
    upsert_review_items(engine, _pg_dicts(uid))
    db = _FakeDatabase()
    for d in _mongo_docs(uid):
        db["review_items"].insert_one(d)
    return dict(engine=engine, uid=uid, pg=PostgresReviewItemRepository(engine), mongo=MongoReviewItemRepository(db))


def _oracle(uid, sort, *, statuses=PENDING, reviewer=None, intents=None, topics=None, sentiments=None,
            priorities=None, created_from=None, created_to=None):
    rows = []
    for (i, it, tp, se, pr, cc, ing, st, rv, ra) in ROWS:
        if st not in statuses:
            continue
        if reviewer is not None and uid[rv] != reviewer:
            continue
        if intents and it not in intents:
            continue
        if topics and tp not in topics:
            continue
        if sentiments and se not in sentiments:
            continue
        if priorities and pr not in priorities:
            continue
        if created_from and (cc is None or cc < created_from):
            continue
        if created_to and (cc is None or cc > created_to):
            continue
        rows.append((i, pr, cc, ing))

    def key(r):
        i, pr, cc, ing = r
        ts = ing.timestamp()
        if sort == "newest":
            return (-ts, i)
        if sort == "oldest":
            return (ts, i)
        if sort == "comment_newest":
            return (cc is None, -cc.timestamp() if cc else 0, -ts, i)
        if sort == "comment_oldest":
            return (cc is None, cc.timestamp() if cc else 0, ts, i)
        if sort == "priority_newest":
            return (pr, -ts, i)
        return (pr, ts, i)

    return [r[0] for r in sorted(rows, key=key)]


def _ids(repo, sort, **kw):
    kw = dict(kw)
    reviewer = kw.pop("reviewer", None)
    statuses = kw.pop("statuses", PENDING)
    page, total = repo.list_queue(
        statuses=statuses, source_file=None, reviewer_user_id=reviewer, search=None, sort=sort,
        offset=0, limit=100, primary_intents=kw.get("intents"), topic_subtypes=kw.get("topics"),
        sentiments=kw.get("sentiments"), priorities=kw.get("priorities"),
        created_from=kw.get("created_from"), created_to=kw.get("created_to"),
    )
    assert total == len(page)
    return [p.comment_id for p in page]


COMBOS = {
    "none": {},
    "intent+topic": dict(intents=("COMPLAINT", "INQUIRY"), topics=("TECHNICAL_ISSUE",)),
    "topic+priority": dict(topics=("GENERAL", "TECHNICAL_ISSUE"), priorities=(1,)),
    "sentiment+date": dict(sentiments=("NEGATIVE", "POSITIVE"), created_from=D(2026, 1, 8), created_to=D(2026, 4, 10)),
    "reviewer+status": dict(reviewer="A", statuses=("DRAFT_READY", "NO_RESPONSE_RECOMMENDED")),
    "date only": dict(created_from=D(2026, 3, 1)),
    "all": dict(reviewer="A", statuses=("DRAFT_READY", "REVIEW_ONLY"), intents=("COMPLAINT", "SUGGESTION", "INQUIRY"),
                topics=("TECHNICAL_ISSUE", "GENERAL", "WORD_SUBMISSION"), sentiments=("NEGATIVE", "POSITIVE"),
                priorities=(1, 3), created_from=D(2026, 1, 1), created_to=D(2026, 3, 1)),
}


@pytest.mark.parametrize("name", list(COMBOS))
@pytest.mark.parametrize("sort", ALL_SORTS)
def test_combined_filters_match_oracle_on_both_backends(env, name, sort):
    uid = env["uid"]
    kw = dict(COMBOS[name])
    if "reviewer" in kw:
        kw["reviewer"] = uid[kw["reviewer"]]
    expected = _oracle(uid, sort, **kw)
    assert _ids(env["pg"], sort, **kw) == expected
    assert _ids(env["mongo"], sort, **kw) == expected  # parity, same order


def test_filter_combinations_are_not_trivially_empty(env):
    uid = env["uid"]
    for name, kw in COMBOS.items():
        kw = dict(kw)
        if "reviewer" in kw:
            kw["reviewer"] = uid[kw["reviewer"]]
        assert _oracle(uid, "oldest", **kw), name


@pytest.mark.parametrize("backend", ["pg", "mongo"])
@pytest.mark.parametrize("sort", ALL_SORTS)
def test_pagination_is_stable_with_ties_and_nulls(env, backend, sort):
    repo = env[backend]
    full = _ids(repo, sort)
    paged = []
    for off in range(0, len(full) + 3, 3):
        page, total = repo.list_queue(
            statuses=PENDING, source_file=None, reviewer_user_id=None, search=None, sort=sort, offset=off, limit=3)
        assert total == len(full)
        paged += [p.comment_id for p in page]
    assert paged == full
    assert len(set(paged)) == len(paged)


def test_comment_date_sort_uses_comment_created_at_not_ingestion(env):
    asc = _ids(env["pg"], "comment_oldest")
    assert asc[0] == "r01" and asc[-1] == "r08"  # earliest comment first, NULL date last
    desc = _ids(env["pg"], "comment_newest")
    assert desc[0] in ("r10", "r11") and desc[-1] == "r08"  # NULL still last descending
    assert asc != _ids(env["pg"], "oldest")  # distinct from ingestion order
    assert asc.index("r02") < asc.index("r12")  # same comment date: ingestion then id


def test_assignable_filters_sort_and_assignee(env):
    uid = env["uid"]
    for backend in ("pg", "mongo"):
        repo = env[backend]
        page, _ = repo.list_assignable(
            unassigned_only=False, source_file=None, offset=0, limit=100, sentiments=("NEGATIVE",),
            priorities=(1,), created_from=D(2026, 1, 1), created_to=D(2026, 1, 31), reviewer_user_id=uid["A"],
            sort="comment_oldest")
        assert [p.comment_id for p in page] == ["r01", "r02", "r12"], backend
        page, _ = repo.list_assignable(unassigned_only=True, source_file=None, offset=0, limit=100,
                                       statuses=("DRAFT_READY",), sort="newest")
        assert [p.comment_id for p in page] == ["r10", "r11"], backend
        page, _ = repo.list_assignable(unassigned_only=True, source_file=None, offset=0, limit=100,
                                       reviewer_user_id=uid["A"])
        assert page == [], backend  # conflicting predicates match nothing on both backends


def test_service_rejects_unknown_sort_and_keeps_empty_status_intersection_empty(env):
    svc = ReviewWorkflowService(env["engine"])
    with pytest.raises(ValueError):
        svc.list_review_queue(sort="sideways")
    assert svc.list_review_queue(statuses=()).total == 0  # not silently the pending default
    a = svc.list_review_queue(sort="ingested_newest")
    b = svc.list_review_queue(sort="newest")
    assert [i.comment_id for i in a.items] == [i.comment_id for i in b.items]


# ------------------------------------------------------------------------ HTTP


def _get(url, cookie=None):
    req = urllib.request.Request(url, headers={"Cookie": cookie} if cookie else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _login(base, email):
    req = urllib.request.Request(
        f"{base}/api/login", data=json.dumps({"email": email, "password": "realpassword123"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers.get("Set-Cookie").split(";")[0]


def test_http_filters_validation_pagination_and_isolation(env):
    engine, uid = env["engine"], env["uid"]
    server, base = serve_in_thread(engine)
    try:
        adm, ca, cb = _login(base, "adm@x.com"), _login(base, "a@x.com"), _login(base, "b@x.com")

        # Correct records, not just 200: admin, every filter combined.
        url = (f"{base}/api/queue?filter=pending&primary_intent=COMPLAINT,SUGGESTION,INQUIRY"
               "&topic_subtype=TECHNICAL_ISSUE,GENERAL,WORD_SUBMISSION&sentiment=NEGATIVE,POSITIVE&priority=1,3"
               "&created_from=2026-01-01&created_to=2026-03-01&sort=comment_oldest"
               f"&reviewer_user_id={uid['A']}&review_status=DRAFT_READY,REVIEW_ONLY")
        st, body = _get(url, adm)
        exp = _oracle(uid, "comment_oldest", reviewer=uid["A"], statuses=("DRAFT_READY", "REVIEW_ONLY"),
                      intents=("COMPLAINT", "SUGGESTION", "INQUIRY"),
                      topics=("TECHNICAL_ISSUE", "GENERAL", "WORD_SUBMISSION"), sentiments=("NEGATIVE", "POSITIVE"),
                      priorities=(1, 3), created_from=D(2026, 1, 1), created_to=dt.datetime(2026, 3, 1, 23, 59, 59))
        assert exp, "scenario must select something"
        assert st == 200 and [i["comment_id"] for i in body["items"]] == exp and body["total"] == len(exp)

        # Pagination through HTTP is stable.
        seen = []
        for off in range(0, 12, 5):
            st, body = _get(f"{base}/api/queue?filter=all&sort=comment_newest&limit=5&offset={off}", adm)
            assert st == 200
            seen += [i["comment_id"] for i in body["items"]]
        assert len(seen) == len(set(seen)) == body["total"]

        # Reviewer isolation: A sees only A's items under every filter; B's id is ignored.
        st, body = _get(f"{base}/api/queue?filter=pending&sentiment=NEGATIVE&reviewer_user_id={uid['B']}", ca)
        assert st == 200
        assert {i["comment_id"] for i in body["items"]} == {"r01", "r02", "r12"}
        st, body = _get(f"{base}/api/queue?filter=pending&priority=2", ca)
        assert body["items"] == []  # priority-2 rows belong to B
        st, body = _get(f"{base}/api/queue?filter=pending&priority=2", cb)
        assert {i["comment_id"] for i in body["items"]} == {"r05", "r06"}

        # Unknown values fail clearly (400 + param) on both endpoints.
        bad = [
            "sort=sideways", "primary_intent=NOPE", "topic_subtype=NOPE", "sentiment=HAPPY",
            "priority=9", "priority=x", "review_status=BOGUS", "reviewer_action=nope", "created_from=garbage",
            "created_to=2026-13-40", "created_from=2026-05-01&created_to=2026-04-01", "limit=0", "limit=abc",
            "offset=-1", "primary_intent=COMPLAINT,NOPE",
        ]
        for q in bad:
            st, body = _get(f"{base}/api/queue?{q}", adm)
            assert st == 400 and body["error"] == "invalid_filter" and body["param"], q
            st, body = _get(f"{base}/api/admin/assignable-items?{q}", adm)
            assert st == 400 and body["error"] == "invalid_filter" and body["param"], q
        st, body = _get(f"{base}/api/queue?filter=bogus", adm)
        assert st == 400 and body["param"] == "filter"
        st, body = _get(f"{base}/api/queue?reviewer_user_id=does-not-exist", adm)
        assert st == 400 and body["param"] == "reviewer_user_id"
        st, body = _get(f"{base}/api/admin/assignable-items?unassigned_only=maybe", adm)
        assert st == 400

        # Assignment endpoint: correct rows, admin only.
        st, body = _get(f"{base}/api/admin/assignable-items?sentiment=NEGATIVE&priority=1&sort=comment_oldest"
                        f"&reviewer_user_id={uid['A']}&review_status=DRAFT_READY", adm)
        assert st == 200 and [i["comment_id"] for i in body["items"]] == ["r01", "r02", "r12"]
        st, _ = _get(f"{base}/api/admin/assignable-items?sentiment=NEGATIVE", ca)
        assert st == 403
        st, body = _get(f"{base}/api/admin/assignable-items?unassigned_only=true&reviewer_user_id={uid['A']}", adm)
        assert st == 400

        # Existing behaviour preserved: default tab is still the pending set (incl. NRR).
        st, body = _get(f"{base}/api/queue", adm)
        assert st == 200 and "r07" in {i["comment_id"] for i in body["items"]}
    finally:
        server.shutdown()


# --------------------------------------------------- sentiment at classification


def _cls_result(sentiment):
    return dict(
        comment_id="u1", success=True, ai_primary_intent="SUGGESTION", ai_topic_subtype="GENERAL",
        ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="ACKNOWLEDGE_THEN_REVIEW",
        ai_classification_reason="r", ai_confidence=0.9, ai_sentiment=sentiment,
        ai_classification_source="s", ai_needs_review=False, routing_reason_code="x")


def test_sentiment_is_persisted_by_classification_before_generation_postgres(tmp_path):
    engine = get_engine(tmp_path / "s.sqlite")
    init_db(engine)
    upsert_review_items(engine, [dict(
        comment_id="u1", source_file="g.xlsx", source_row_index=0, original_comment="c",
        review_status="UNPROCESSED", ingested_at=T0, ready_to_send=False, row_version=1)])
    out = PostgresReviewItemRepository(engine).apply_classification_batch(
        [_cls_result("NEGATIVE")], model_name="m", model_version=None, prompt_version="p",
        pipeline_version="v", inference_run_id="run")
    assert out["classified"] == 1
    with Session(engine) as s:
        it = s.get(ReviewItem, "u1")
        assert (it.sentiment, it.ai_sentiment, it.review_status) == ("NEGATIVE", "NEGATIVE", "CLASSIFIED")
        assert it.generation_routing_decision is None and it.generated_at is None  # no generation yet


def test_sentiment_is_persisted_by_classification_before_generation_mongo():
    db = _FakeDatabase()
    db["review_items"].insert_one(_item_doc("u1", review_status="UNPROCESSED", ai_primary_intent=None,
                                            sentiment=None, suggested_final_reply=None))
    out = MongoReviewItemRepository(db).apply_classification_batch(
        [_cls_result("POSITIVE")], model_name="m", model_version=None, prompt_version="p",
        pipeline_version="v", inference_run_id="run")
    assert out["classified"] == 1
    doc = db["review_items"].find_one({"_id": "u1"})
    assert (doc["sentiment"], doc["ai_sentiment"], doc["review_status"]) == ("POSITIVE", "POSITIVE", "CLASSIFIED")
