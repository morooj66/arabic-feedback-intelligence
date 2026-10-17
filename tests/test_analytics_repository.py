"""
Phase M4C.5 — analytics repository contract + parity tests.

Postgres and Mongo scenarios are placed side by side per behavior, same
convention as test_ingest_repository.py / test_prefilter_repository.py /
test_classification_repository.py / test_generation_repository.py. The
Mongo tests run against the same in-memory fake used in
test_mongo_repository_unit.py -- they prove THIS CODE's logic, not
MongoDB's own server-side aggregation-engine behavior. See
test_mongo_repository_live.py for what remains BLOCKED pending a working
Atlas connection.

Read-only extraction: no write path exists in either
PostgresAnalyticsRepository or MongoAnalyticsRepository (confirmed by
the M4C.5 audit itself; both classes call only
select()/count_documents()/aggregate()/find()/find_one(), never a
commit/update_one/insert_one).

Datetime-convention note, load-bearing for several tests below:
`ingested_at` is a "utc"-convention column while the analytics
`now`/`start`/`end` window boundaries are naive-Riyadh wall-clock values
-- the ORIGINAL (pre-M4C.5) Postgres code already compared them with
zero conversion, an embedded ~3-hour mismatch that predates this
migration (see the M4C.5 report, item 9). These tests use identical raw
datetime literals for `ingested_at` in both backends' fixtures --
because neither backend converts a "utc"-convention value, this
reproduces the exact same (preserved, not fixed) comparison on both
sides. `reviewed_at`/`reviewer_started_at` ARE "riyadh"-convention: the
Postgres fixture stores the naive-Riyadh business value directly (no
conversion, matching how Postgres stores it in production); the Mongo
fixture must store the *raw BSON-decoded equivalent* of that same value
via `_riyadh_raw()` below, matching what a real ingested document's
field would actually decode to.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.review import clock
from rcs.review.models import ReviewItem
from rcs.review.repositories import contracts
from rcs.review.repositories.mongo import MongoAnalyticsRepository
from rcs.review.repositories.postgres import PostgresAnalyticsRepository
from rcs.review.repository import get_engine, init_db, upsert_review_items

from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake

_ALL_COLUMNS = tuple(c.name for c in ReviewItem.__table__.columns)


def _riyadh_raw(value: dt.datetime) -> dt.datetime:
    """The raw, naive-UTC value a real MongoDB document's "riyadh"-
    convention field would decode to for the given naive-Riyadh business
    value -- i.e. what `_to_bson_value(value, "riyadh")` followed by a
    real BSON round-trip actually produces. The in-memory fake does not
    perform real BSON encoding (see test_generation_repository.py's own
    note on this), so fixtures must pre-compute this value themselves to
    exercise MongoAnalyticsRepository's query-boundary conversion
    (_query_bound) correctly."""
    return clock.as_app(value).astimezone(dt.timezone.utc).replace(tzinfo=None)


def _engine(tmp_path, name="analytics.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _seed_pg(engine, comment_id: str, **overrides) -> None:
    row = dict(
        comment_id=comment_id,
        source_file="f.xlsx",
        source_row_index=0,
        review_status="DRAFT_READY",
        review_priority=3,
        ingested_at=None,
        reviewed_at=None,
        reviewer_started_at=None,
        reviewer_action=None,
        reviewer_id=None,
        reviewer_user_id=None,
        primary_intent=None,
        topic_subtype=None,
        sentiment=None,
        ready_to_send=False,
        row_version=1,
    )
    row.update(overrides)
    upsert_review_items(engine, [row])


def _mongo_doc(comment_id: str, **overrides) -> dict:
    """Every declared ReviewItem column, defaulted to None -- same
    completeness discipline established in test_classification_
    repository.py after the SimpleNamespace-vs-incomplete-fixture bug
    found there. `reviewed_at`/`reviewer_started_at` overrides must
    already be passed through `_riyadh_raw()` by the caller; `ingested_at`
    is passed as the identical raw literal used in the Postgres fixture
    (see module docstring)."""
    doc = {name: None for name in _ALL_COLUMNS if name != "comment_id"}
    doc.update(
        _id=comment_id,
        source_file="f.xlsx",
        source_row_index=0,
        review_status="DRAFT_READY",
        review_priority=3,
        ready_to_send=False,
        row_version=1,
    )
    doc.update(overrides)
    return doc


def _insert_many(db, docs):
    coll = db["review_items"]
    for doc in docs:
        coll.insert_one(doc)


NOW = dt.datetime(2026, 6, 15, 12, 0, 0)  # naive-Riyadh "now" used throughout


# A. Contract/signature ---------------------------------------------------


def test_contract_declares_compute_operational_analytics():
    assert hasattr(contracts.AnalyticsRepository, "compute_operational_analytics")


def test_both_backends_implement_compute_operational_analytics():
    assert hasattr(PostgresAnalyticsRepository, "compute_operational_analytics")
    assert hasattr(MongoAnalyticsRepository, "compute_operational_analytics")


# B. Empty dataset ----------------------------------------------------------


def test_empty_dataset_postgres(tmp_path):
    engine = _engine(tmp_path)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["incoming"] == 0
    assert result["kpis"]["pending"] == 0
    assert result["kpis"]["avg_review_seconds"] is None
    assert result["quality"]["send_as_is_rate"] is None
    assert result["queue_health"]["oldest_pending_age_seconds"] is None
    assert result["availability"]["avg_review_time"] == "insufficient_timestamps"
    assert result["availability"]["outcomes"] == "no_completed_actions"
    assert result["availability"]["classification"] == "no_labels"
    assert all(p["incoming"] == 0 and p["reviewed"] == 0 for p in result["trend"])


def test_empty_dataset_mongo():
    db = _FakeDatabase()
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["incoming"] == 0
    assert result["kpis"]["pending"] == 0
    assert result["kpis"]["avg_review_seconds"] is None
    assert result["quality"]["send_as_is_rate"] is None
    assert result["queue_health"]["oldest_pending_age_seconds"] is None
    assert result["availability"]["avg_review_time"] == "insufficient_timestamps"
    assert all(p["incoming"] == 0 and p["reviewed"] == 0 for p in result["trend"])


# C. Single-row dataset (full happy path) ------------------------------------


def _happy_path_kwargs():
    ingested = dt.datetime(2026, 6, 12, 8, 0, 0)
    started = dt.datetime(2026, 6, 12, 9, 0, 0)
    reviewed = dt.datetime(2026, 6, 12, 9, 10, 0)
    return ingested, started, reviewed


def test_single_row_happy_path_postgres(tmp_path):
    engine = _engine(tmp_path)
    ingested, started, reviewed = _happy_path_kwargs()
    _seed_pg(
        engine, "c1", review_status="APPROVED", ingested_at=ingested,
        reviewer_started_at=started, reviewed_at=reviewed, reviewer_action="SEND_AS_IS",
        reviewer_id="r1",
    )
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, reviewer_id="r1")
    assert result["kpis"]["incoming"] == 1
    assert result["kpis"]["reviewed"] == 1
    assert result["kpis"]["avg_review_seconds"] == 600.0
    assert result["outcomes"]["SEND_AS_IS"] == 1
    assert result["quality"]["send_as_is_rate"] == 1.0


def test_single_row_happy_path_mongo():
    db = _FakeDatabase()
    ingested, started, reviewed = _happy_path_kwargs()
    _insert_many(db, [
        _mongo_doc(
            "c1", review_status="APPROVED", ingested_at=ingested,
            reviewer_started_at=_riyadh_raw(started), reviewed_at=_riyadh_raw(reviewed),
            reviewer_action="SEND_AS_IS", reviewer_id="r1",
        )
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, reviewer_id="r1")
    assert result["kpis"]["incoming"] == 1
    assert result["kpis"]["reviewed"] == 1
    assert result["kpis"]["avg_review_seconds"] == 600.0
    assert result["outcomes"]["SEND_AS_IS"] == 1
    assert result["quality"]["send_as_is_rate"] == 1.0


# D/E. Mixed status dataset; NOT scoped to "reviewable" rows -----------------


def test_incoming_counts_all_rows_regardless_of_status_postgres(tmp_path):
    """Audited finding: `incoming`/`reviewed` are NOT filtered by
    review_status at all -- only by a non-null timestamp in range. An
    EXCLUDED row still counts toward `incoming` if it has ingested_at."""
    engine = _engine(tmp_path)
    ingested = dt.datetime(2026, 6, 12, 8, 0, 0)
    _seed_pg(engine, "c1", review_status="EXCLUDED", ingested_at=ingested)
    _seed_pg(engine, "c2", review_status="DUPLICATE_CONTENT", ingested_at=ingested)
    _seed_pg(engine, "c3", review_status="UNPROCESSED", ingested_at=ingested)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["incoming"] == 3
    assert result["kpis"]["pending"] == 0  # none of these statuses are QUEUE_PENDING_STATUSES


def test_incoming_counts_all_rows_regardless_of_status_mongo():
    db = _FakeDatabase()
    ingested = dt.datetime(2026, 6, 12, 8, 0, 0)
    _insert_many(db, [
        _mongo_doc("c1", review_status="EXCLUDED", ingested_at=ingested),
        _mongo_doc("c2", review_status="DUPLICATE_CONTENT", ingested_at=ingested),
        _mongo_doc("c3", review_status="UNPROCESSED", ingested_at=ingested),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["incoming"] == 3
    assert result["kpis"]["pending"] == 0


def test_mixed_pending_statuses_postgres(tmp_path):
    engine = _engine(tmp_path)
    ingested = NOW - dt.timedelta(days=1)
    _seed_pg(engine, "p1", review_status="DRAFT_READY", ingested_at=ingested, review_priority=1)
    _seed_pg(engine, "p2", review_status="REVIEW_ONLY", ingested_at=ingested, review_priority=2)
    _seed_pg(engine, "p3", review_status="IN_REVIEW", ingested_at=ingested, review_priority=3)
    _seed_pg(engine, "done", review_status="SENT", ingested_at=ingested)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["pending"] == 3
    assert result["kpis"]["urgent_pending"] == 1


def test_mixed_pending_statuses_mongo():
    db = _FakeDatabase()
    ingested = NOW - dt.timedelta(days=1)
    _insert_many(db, [
        _mongo_doc("p1", review_status="DRAFT_READY", ingested_at=ingested, review_priority=1),
        _mongo_doc("p2", review_status="REVIEW_ONLY", ingested_at=ingested, review_priority=2),
        _mongo_doc("p3", review_status="IN_REVIEW", ingested_at=ingested, review_priority=3),
        _mongo_doc("done", review_status="SENT", ingested_at=ingested),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["pending"] == 3
    assert result["kpis"]["urgent_pending"] == 1


# F. Date-range boundaries (inclusive <= / >=), including the audited
#    ingested_at (utc) vs naive-Riyadh-boundary mismatch, reproduced
#    identically on both backends ------------------------------------------


def test_custom_range_boundary_is_inclusive_postgres(tmp_path):
    engine = _engine(tmp_path)
    start = dt.datetime(2026, 6, 1, 0, 0, 0)
    end = dt.datetime(2026, 6, 10, 0, 0, 0)
    _seed_pg(engine, "on_start", ingested_at=start)
    _seed_pg(engine, "on_end", ingested_at=end)
    _seed_pg(engine, "before", ingested_at=start - dt.timedelta(seconds=1))
    _seed_pg(engine, "after", ingested_at=end + dt.timedelta(seconds=1))
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(
        range_key="custom", now=NOW, custom_start=start, custom_end=end
    )
    assert result["kpis"]["incoming"] == 2  # on_start and on_end only


def test_custom_range_boundary_is_inclusive_mongo():
    db = _FakeDatabase()
    start = dt.datetime(2026, 6, 1, 0, 0, 0)
    end = dt.datetime(2026, 6, 10, 0, 0, 0)
    _insert_many(db, [
        _mongo_doc("on_start", ingested_at=start),
        _mongo_doc("on_end", ingested_at=end),
        _mongo_doc("before", ingested_at=start - dt.timedelta(seconds=1)),
        _mongo_doc("after", ingested_at=end + dt.timedelta(seconds=1)),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(
        range_key="custom", now=NOW, custom_start=start, custom_end=end
    )
    assert result["kpis"]["incoming"] == 2


def test_ingested_at_utc_vs_riyadh_boundary_mismatch_is_reproduced_identically():
    """The audited pre-existing quirk (M4C.5 report item 9): `ingested_at`
    is stored naive-UTC, but the analytics window boundary is naive-
    Riyadh "now" with zero conversion in the ORIGINAL Postgres code. A
    row ingested at 23:00 on the boundary day, when "now" is exactly
    midnight Riyadh the next day, falls OUTSIDE a naive literal same-day
    comparison window in a timezone-correct system, but this codebase's
    literal (unconverted) comparison must behave identically on both
    backends -- this test proves that identical (not "fixed") behavior,
    it does not assert what the "correct" timezone-aware answer would be.
    """
    ingested = dt.datetime(2026, 6, 15, 0, 0, 0)  # exactly "today_start" per NOW
    now = NOW  # 2026-06-15 12:00:00, naive-Riyadh

    engine_row = dict(ingested_at=ingested)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        engine = get_engine(pathlib.Path(d) / "boundary.sqlite")
        init_db(engine)
        _seed_pg(engine, "today", **engine_row)
        pg_result = PostgresAnalyticsRepository(engine).compute_operational_analytics(
            range_key="today", now=now
        )

    db = _FakeDatabase()
    _insert_many(db, [_mongo_doc("today", ingested_at=ingested)])
    mongo_result = MongoAnalyticsRepository(db).compute_operational_analytics(range_key="today", now=now)

    assert pg_result["kpis"]["incoming"] == mongo_result["kpis"]["incoming"]


# G. Null datetime handling ---------------------------------------------------


def test_null_reviewer_started_at_excluded_from_average_postgres(tmp_path):
    engine = _engine(tmp_path)
    reviewed = dt.datetime(2026, 6, 12, 9, 0, 0)
    _seed_pg(engine, "no_start", ingested_at=reviewed, reviewed_at=reviewed, reviewer_started_at=None)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["avg_review_seconds"] is None
    assert result["kpis"]["avg_review_sample"] == 0
    assert result["kpis"]["reviewed"] == 1  # still counted as "reviewed"


def test_null_reviewer_started_at_excluded_from_average_mongo():
    db = _FakeDatabase()
    reviewed = dt.datetime(2026, 6, 12, 9, 0, 0)
    _insert_many(db, [
        _mongo_doc("no_start", ingested_at=reviewed, reviewed_at=_riyadh_raw(reviewed), reviewer_started_at=None)
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["avg_review_seconds"] is None
    assert result["kpis"]["avg_review_sample"] == 0
    assert result["kpis"]["reviewed"] == 1


def test_reviewed_before_started_anomaly_silently_excluded_postgres(tmp_path):
    """Preserved (not "fixed") anomaly-guard: reviewed_at < reviewer_
    started_at is silently dropped from the average, not raised/clamped."""
    engine = _engine(tmp_path)
    started = dt.datetime(2026, 6, 12, 9, 0, 0)
    reviewed = started - dt.timedelta(minutes=5)
    _seed_pg(engine, "backwards", ingested_at=started, reviewer_started_at=started, reviewed_at=reviewed)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["avg_review_seconds"] is None
    assert result["kpis"]["avg_review_sample"] == 0


def test_reviewed_before_started_anomaly_silently_excluded_mongo():
    db = _FakeDatabase()
    started = dt.datetime(2026, 6, 12, 9, 0, 0)
    reviewed = started - dt.timedelta(minutes=5)
    _insert_many(db, [
        _mongo_doc(
            "backwards", ingested_at=started,
            reviewer_started_at=_riyadh_raw(started), reviewed_at=_riyadh_raw(reviewed),
        )
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["avg_review_seconds"] is None
    assert result["kpis"]["avg_review_sample"] == 0


# H. Source filter -------------------------------------------------------


def test_source_file_scoping_postgres(tmp_path):
    engine = _engine(tmp_path)
    ingested = NOW - dt.timedelta(days=1)
    _seed_pg(engine, "a1", source_file="a.xlsx", ingested_at=ingested)
    _seed_pg(engine, "b1", source_file="b.xlsx", ingested_at=ingested)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, source_file="a.xlsx")
    assert result["kpis"]["incoming"] == 1


def test_source_file_scoping_mongo():
    db = _FakeDatabase()
    ingested = NOW - dt.timedelta(days=1)
    _insert_many(db, [
        _mongo_doc("a1", source_file="a.xlsx", ingested_at=ingested),
        _mongo_doc("b1", source_file="b.xlsx", ingested_at=ingested),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, source_file="a.xlsx")
    assert result["kpis"]["incoming"] == 1


# I. Reviewer filters (reviewer_user_id scopes everything; reviewer_id
#    only scopes "reviewed_by_me") -------------------------------------------


def test_reviewer_user_id_scopes_entire_result_postgres(tmp_path):
    engine = _engine(tmp_path)
    ingested = NOW - dt.timedelta(days=1)
    _seed_pg(engine, "mine", reviewer_user_id="u1", ingested_at=ingested)
    _seed_pg(engine, "theirs", reviewer_user_id="u2", ingested_at=ingested)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, reviewer_user_id="u1")
    assert result["kpis"]["incoming"] == 1


def test_reviewer_user_id_scopes_entire_result_mongo():
    db = _FakeDatabase()
    ingested = NOW - dt.timedelta(days=1)
    _insert_many(db, [
        _mongo_doc("mine", reviewer_user_id="u1", ingested_at=ingested),
        _mongo_doc("theirs", reviewer_user_id="u2", ingested_at=ingested),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, reviewer_user_id="u1")
    assert result["kpis"]["incoming"] == 1


def test_reviewer_id_only_scopes_reviewed_by_me_postgres(tmp_path):
    engine = _engine(tmp_path)
    today = NOW.replace(hour=1, minute=0, second=0, microsecond=0)
    _seed_pg(engine, "mine", reviewer_id="legacy-r1", ingested_at=today, reviewed_at=today)
    _seed_pg(engine, "theirs", reviewer_id="legacy-r2", ingested_at=today, reviewed_at=today)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, reviewer_id="legacy-r1")
    assert result["today"]["reviewed_by_me"] == 1
    assert result["today"]["reviewed_all"] == 2  # not reviewer-scoped


def test_reviewer_id_only_scopes_reviewed_by_me_mongo():
    db = _FakeDatabase()
    today = NOW.replace(hour=1, minute=0, second=0, microsecond=0)
    _insert_many(db, [
        _mongo_doc("mine", reviewer_id="legacy-r1", ingested_at=today, reviewed_at=_riyadh_raw(today)),
        _mongo_doc("theirs", reviewer_id="legacy-r2", ingested_at=today, reviewed_at=_riyadh_raw(today)),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW, reviewer_id="legacy-r1")
    assert result["today"]["reviewed_by_me"] == 1
    assert result["today"]["reviewed_all"] == 2


# J. Intent/topic/sentiment grouping (canonical vs unmapped) -----------------


def test_classification_grouping_canonical_and_unmapped_postgres(tmp_path):
    engine = _engine(tmp_path)
    ingested = NOW - dt.timedelta(days=1)
    _seed_pg(engine, "c1", ingested_at=ingested, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE")
    _seed_pg(engine, "c2", ingested_at=ingested, primary_intent="LEGACY_UNKNOWN_LABEL")
    _seed_pg(engine, "c3", ingested_at=ingested, primary_intent=None)
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    intent_by_key = {row["key"]: row["count"] for row in result["classification"]["intent"]}
    assert intent_by_key["THANKS_PRAISE"] == 1
    assert result["data_quality"]["unmapped_classification"]["intent"] == [
        {"key": "LEGACY_UNKNOWN_LABEL", "count": 1}
    ]
    assert result["data_quality"]["note"] == "legacy_unmapped_labels_excluded_from_canonical_taxonomy"


def test_classification_grouping_canonical_and_unmapped_mongo():
    db = _FakeDatabase()
    ingested = NOW - dt.timedelta(days=1)
    _insert_many(db, [
        _mongo_doc("c1", ingested_at=ingested, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE"),
        _mongo_doc("c2", ingested_at=ingested, primary_intent="LEGACY_UNKNOWN_LABEL"),
        _mongo_doc("c3", ingested_at=ingested, primary_intent=None),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    intent_by_key = {row["key"]: row["count"] for row in result["classification"]["intent"]}
    assert intent_by_key["THANKS_PRAISE"] == 1
    assert result["data_quality"]["unmapped_classification"]["intent"] == [
        {"key": "LEGACY_UNKNOWN_LABEL", "count": 1}
    ]
    assert result["data_quality"]["note"] == "legacy_unmapped_labels_excluded_from_canonical_taxonomy"


# K. Response-quality rates ---------------------------------------------


def test_quality_rates_formula_postgres(tmp_path):
    engine = _engine(tmp_path)
    reviewed = NOW - dt.timedelta(hours=1)
    _seed_pg(engine, "s1", ingested_at=reviewed, reviewed_at=reviewed, reviewer_action="SEND_AS_IS")
    _seed_pg(engine, "s2", ingested_at=reviewed, reviewed_at=reviewed, reviewer_action="SEND_AS_IS")
    _seed_pg(engine, "s3", ingested_at=reviewed, reviewed_at=reviewed, reviewer_action="EDIT_AND_SEND")
    _seed_pg(engine, "s4", ingested_at=reviewed, reviewed_at=reviewed, reviewer_action="NO_RESPONSE_NEEDED")
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["quality"]["sample"] == 4
    assert result["quality"]["send_as_is_rate"] == 0.5
    assert result["quality"]["edited_rate"] == 0.25
    assert result["quality"]["no_reply_rate"] == 0.25


def test_quality_rates_formula_mongo():
    db = _FakeDatabase()
    reviewed = NOW - dt.timedelta(hours=1)
    _insert_many(db, [
        _mongo_doc("s1", ingested_at=reviewed, reviewed_at=_riyadh_raw(reviewed), reviewer_action="SEND_AS_IS"),
        _mongo_doc("s2", ingested_at=reviewed, reviewed_at=_riyadh_raw(reviewed), reviewer_action="SEND_AS_IS"),
        _mongo_doc("s3", ingested_at=reviewed, reviewed_at=_riyadh_raw(reviewed), reviewer_action="EDIT_AND_SEND"),
        _mongo_doc("s4", ingested_at=reviewed, reviewed_at=_riyadh_raw(reviewed), reviewer_action="NO_RESPONSE_NEEDED"),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["quality"]["sample"] == 4
    assert result["quality"]["send_as_is_rate"] == 0.5
    assert result["quality"]["edited_rate"] == 0.25
    assert result["quality"]["no_reply_rate"] == 0.25


# L/O. Average-time calculation + numeric precision --------------------------


def test_average_review_seconds_rounding_postgres(tmp_path):
    engine = _engine(tmp_path)
    started = dt.datetime(2026, 6, 12, 9, 0, 0)
    _seed_pg(engine, "a", ingested_at=started, reviewer_started_at=started, reviewed_at=started + dt.timedelta(seconds=100))
    _seed_pg(engine, "b", ingested_at=started, reviewer_started_at=started, reviewed_at=started + dt.timedelta(seconds=101))
    _seed_pg(engine, "c", ingested_at=started, reviewer_started_at=started, reviewed_at=started + dt.timedelta(seconds=101))
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["avg_review_seconds"] == round((100 + 101 + 101) / 3, 1)
    assert result["kpis"]["avg_review_sample"] == 3


def test_average_review_seconds_rounding_mongo():
    db = _FakeDatabase()
    started = dt.datetime(2026, 6, 12, 9, 0, 0)
    _insert_many(db, [
        _mongo_doc("a", ingested_at=started, reviewer_started_at=_riyadh_raw(started), reviewed_at=_riyadh_raw(started + dt.timedelta(seconds=100))),
        _mongo_doc("b", ingested_at=started, reviewer_started_at=_riyadh_raw(started), reviewed_at=_riyadh_raw(started + dt.timedelta(seconds=101))),
        _mongo_doc("c", ingested_at=started, reviewer_started_at=_riyadh_raw(started), reviewed_at=_riyadh_raw(started + dt.timedelta(seconds=101))),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["kpis"]["avg_review_seconds"] == round((100 + 101 + 101) / 3, 1)
    assert result["kpis"]["avg_review_sample"] == 3


# M. Zero-denominator behavior (see also test B) -----------------------------


def test_zero_denominator_quality_rates_are_none_not_zero_postgres(tmp_path):
    engine = _engine(tmp_path)
    _seed_pg(engine, "unreviewed", ingested_at=NOW - dt.timedelta(days=1))
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["quality"]["sample"] == 0
    assert result["quality"]["send_as_is_rate"] is None
    assert result["quality"]["edited_rate"] is None
    assert result["quality"]["no_reply_rate"] is None


def test_zero_denominator_quality_rates_are_none_not_zero_mongo():
    db = _FakeDatabase()
    _insert_many(db, [_mongo_doc("unreviewed", ingested_at=NOW - dt.timedelta(days=1))])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    assert result["quality"]["sample"] == 0
    assert result["quality"]["send_as_is_rate"] is None


# N. Ordering parity (trend bucket keys ascending, contiguous, gap-filled) ---


def test_trend_buckets_are_contiguous_and_gap_filled_postgres(tmp_path):
    engine = _engine(tmp_path)
    start = dt.datetime(2026, 6, 1, 0, 0, 0)
    end = dt.datetime(2026, 6, 5, 0, 0, 0)
    _seed_pg(engine, "c1", ingested_at=dt.datetime(2026, 6, 1, 9, 0, 0))
    _seed_pg(engine, "c2", ingested_at=dt.datetime(2026, 6, 4, 9, 0, 0))
    repo = PostgresAnalyticsRepository(engine)
    result = repo.compute_operational_analytics(range_key="custom", now=NOW, custom_start=start, custom_end=end)
    buckets = [p["bucket"] for p in result["trend"]]
    assert buckets == sorted(buckets)
    assert buckets[0] == "2026-06-01"
    assert buckets[-1] == "2026-06-05"
    assert len(buckets) == 5
    by_bucket = {p["bucket"]: p["incoming"] for p in result["trend"]}
    assert by_bucket["2026-06-01"] == 1
    assert by_bucket["2026-06-02"] == 0
    assert by_bucket["2026-06-04"] == 1


def test_trend_buckets_are_contiguous_and_gap_filled_mongo():
    db = _FakeDatabase()
    start = dt.datetime(2026, 6, 1, 0, 0, 0)
    end = dt.datetime(2026, 6, 5, 0, 0, 0)
    _insert_many(db, [
        _mongo_doc("c1", ingested_at=dt.datetime(2026, 6, 1, 9, 0, 0)),
        _mongo_doc("c2", ingested_at=dt.datetime(2026, 6, 4, 9, 0, 0)),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="custom", now=NOW, custom_start=start, custom_end=end)
    buckets = [p["bucket"] for p in result["trend"]]
    assert buckets == sorted(buckets)
    assert len(buckets) == 5
    by_bucket = {p["bucket"]: p["incoming"] for p in result["trend"]}
    assert by_bucket["2026-06-01"] == 1
    assert by_bucket["2026-06-02"] == 0
    assert by_bucket["2026-06-04"] == 1


# P. Mongo null/missing parity for is_not(None)-translated predicates -------


def test_primary_intent_null_excluded_from_grouping_mongo():
    """Mongo `{"$ne": None}` must exclude explicit-null exactly like
    Postgres's `.is_not(None)` -- not a broad predicate that would
    accidentally include it."""
    db = _FakeDatabase()
    ingested = NOW - dt.timedelta(days=1)
    _insert_many(db, [
        _mongo_doc("has_intent", ingested_at=ingested, primary_intent="THANKS_PRAISE"),
        _mongo_doc("null_intent", ingested_at=ingested, primary_intent=None),
    ])
    repo = MongoAnalyticsRepository(db)
    result = repo.compute_operational_analytics(range_key="week", now=NOW)
    total_intent_count = sum(row["count"] for row in result["classification"]["intent"])
    assert total_intent_count == 1


# Q. Read retry behavior: analytics reads ARE allowed to use the retry layer -


def test_analytics_reads_use_transient_retry_mongo(monkeypatch):
    """Unlike apply_generation_result's mutation (which must NEVER use
    the retry layer, see test_generation_repository.py), analytics is
    entirely read-only, so with_transient_retry IS expected to wrap
    every query here -- this test proves it is actually invoked, not
    merely imported."""
    import rcs.review.repositories.mongo as mongo_module

    calls = {"n": 0}
    real = mongo_module.with_transient_retry

    def _counting(fn):
        calls["n"] += 1
        return real(fn)

    monkeypatch.setattr(mongo_module, "with_transient_retry", _counting)
    db = _FakeDatabase()
    repo = MongoAnalyticsRepository(db)
    repo.compute_operational_analytics(range_key="week", now=NOW)
    assert calls["n"] > 0
