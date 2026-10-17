"""
Mongo runtime boundary milestone — behavioral parity for the
operations newly extracted from direct SQLAlchemy usage:

  - excel_review_flow._ids_with_status         (status ID lookup)
  - excel_review_flow._next_batch_row_indices   (next-batch selection)
  - excel_review_flow._source_file_aggregate    (source-file aggregation)
  - ReviewWorkflowService.dispatch_outbound     (begin/complete_outbound_
    dispatch: CAS/version-conflict behavior and outbound persistence)

Every Mongo-shaped bundle in this file is the existing in-memory fake
(_FakeDatabase, reused from tests/test_mongo_repository_unit.py, the
established cross-file pattern also used by
tests/test_analytics_repository.py and tests/test_persistence_bundle.py).
This proves REPOSITORY BEHAVIORAL PARITY between the two backends
against safe, disposable test infrastructure only — it is not, and
does not claim to be, real MongoDB integration validation. No live
Mongo connection is used or required anywhere in this file.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.review.excel_review_flow import (
    _ids_with_status,
    _next_batch_row_indices,
    _source_file_aggregate,
    write_fixture_workbook,
)
from rcs.review.models import ReviewItem as _ReviewItemModel
from rcs.review.outbound import OutboundSendResult, OutboundSender, UnconfiguredOutboundSender
from rcs.review.persistence import bundle_from_engine, bundle_from_mongo
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.review.workflow import VersionConflict
from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_ALL_REVIEW_ITEM_COLUMNS = [c.name for c in _ReviewItemModel.__table__.columns]


def _sqlite_bundle(tmp_path, name="boundary.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return bundle_from_engine(engine)


def _mongo_bundle():
    return bundle_from_mongo(_FakeDatabase())


def _both_backends(tmp_path):
    return [_sqlite_bundle(tmp_path), _mongo_bundle()]


def _seed(bundle, rows: list[dict]) -> None:
    """Insert rows through whichever backend `bundle` wraps. Every
    ReviewItem column is given an explicit default (None, or the sane
    default below) before a row's own fields override it -- required
    for Mongo parity, since the fake stores exactly the keys given it
    (SimpleNamespace(**doc) has no attribute for an omitted key), unlike
    the SQLAlchemy ORM column, which always exists. Mirrors the "never
    missing, always explicit null" convention real Mongo writes
    (upsert_ingest_batch) already follow."""
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    for row in rows:
        full = {col: None for col in _ALL_REVIEW_ITEM_COLUMNS}
        full.update(
            {
                "review_status": "UNPROCESSED",
                "ready_to_send": False,
                "row_version": 1,
                "ingested_at": now,
            }
        )
        full.update(row)
        if hasattr(bundle.review_items, "engine"):
            upsert_review_items(bundle.review_items.engine, [full])
        else:
            doc = dict(full)
            doc["_id"] = doc.pop("comment_id")
            bundle.review_items.collection.insert_one(doc)


# ------------------------------------------------------------- status ID lookup


@pytest.mark.parametrize("backend_index", [0, 1])
def test_ids_with_status_mixed_statuses(tmp_path, backend_index):
    bundle = _both_backends(tmp_path)[backend_index]
    _seed(
        bundle,
        [
            {"comment_id": "s-1", "source_file": "f", "source_row_index": 0, "review_status": "CLASSIFIED"},
            {"comment_id": "s-2", "source_file": "f", "source_row_index": 1, "review_status": "UNPROCESSED"},
            {"comment_id": "s-3", "source_file": "f", "source_row_index": 2, "review_status": "CLASSIFIED"},
            {"comment_id": "s-4", "source_file": "f", "source_row_index": 3, "review_status": "EXCLUDED"},
        ],
    )
    result = _ids_with_status(bundle, ["s-1", "s-2", "s-3", "s-4"], "CLASSIFIED")
    assert result == ["s-1", "s-3"]  # same relative order as the input list, deterministic


@pytest.mark.parametrize("backend_index", [0, 1])
def test_ids_with_status_empty_input(tmp_path, backend_index):
    bundle = _both_backends(tmp_path)[backend_index]
    assert _ids_with_status(bundle, [], "CLASSIFIED") == []


@pytest.mark.parametrize("backend_index", [0, 1])
def test_ids_with_status_unknown_ids_excluded_not_errored(tmp_path, backend_index):
    bundle = _both_backends(tmp_path)[backend_index]
    _seed(bundle, [{"comment_id": "s-known", "source_file": "f", "source_row_index": 0, "review_status": "CLASSIFIED"}])
    result = _ids_with_status(bundle, ["s-known", "s-does-not-exist"], "CLASSIFIED")
    assert result == ["s-known"]


def test_ids_with_status_parity_between_backends(tmp_path):
    pg, mongo = _both_backends(tmp_path)
    rows = [
        {"comment_id": "p-1", "source_file": "f", "source_row_index": 0, "review_status": "CLASSIFIED"},
        {"comment_id": "p-2", "source_file": "f", "source_row_index": 1, "review_status": "UNPROCESSED"},
        {"comment_id": "p-3", "source_file": "f", "source_row_index": 2, "review_status": "CLASSIFIED"},
    ]
    _seed(pg, rows)
    _seed(mongo, rows)
    ids = [r["comment_id"] for r in rows]
    assert _ids_with_status(pg, ids, "CLASSIFIED") == _ids_with_status(mongo, ids, "CLASSIFIED") == ["p-1", "p-3"]
    assert _ids_with_status(pg, ids, "UNPROCESSED") == _ids_with_status(mongo, ids, "UNPROCESSED") == ["p-2"]
    assert _ids_with_status(pg, ids, "SENT") == _ids_with_status(mongo, ids, "SENT") == []


# ------------------------------------------------------------- next-batch selection


def test_next_batch_row_indices_parity_no_prior_rows(tmp_path):
    source = write_fixture_workbook(
        tmp_path / "batch.xlsx",
        [{"original_comment": f"comment number {i} is meaningful"} for i in range(5)],
    )
    pg, mongo = _both_backends(tmp_path)
    for bundle in (pg, mongo):
        pending, total_rows, meaningful_count = _next_batch_row_indices(bundle, source, "batch.xlsx", batch_size=3)
        assert pending == [0, 1, 2]  # first 3 of 5 meaningful rows, deterministic order
        assert total_rows == 5
        assert meaningful_count == 5


def test_next_batch_row_indices_parity_with_already_ingested_rows(tmp_path):
    """Resumability: rows already persisted for this source_file (by
    their stable comment_id) are excluded from the next batch, on both
    backends identically."""
    from rcs.ingest.loader import _comment_id

    source = write_fixture_workbook(
        tmp_path / "resume.xlsx",
        [{"original_comment": f"comment number {i} is meaningful"} for i in range(5)],
    )
    pg, mongo = _both_backends(tmp_path)
    already_ingested_indices = [0, 1, 3]
    for bundle in (pg, mongo):
        _seed(
            bundle,
            [
                {
                    "comment_id": _comment_id("resume.xlsx", i),
                    "source_file": "resume.xlsx",
                    "source_row_index": i,
                    "review_status": "UNPROCESSED",
                }
                for i in already_ingested_indices
            ],
        )
        pending, total_rows, meaningful_count = _next_batch_row_indices(bundle, source, "resume.xlsx", batch_size=10)
        assert pending == [2, 4]
        assert total_rows == 5
        assert meaningful_count == 5


def test_next_batch_row_indices_boundary_batch_sizes(tmp_path):
    source = write_fixture_workbook(
        tmp_path / "bounds.xlsx",
        [{"original_comment": f"comment number {i} is meaningful"} for i in range(4)],
    )
    for bundle in _both_backends(tmp_path):
        pending_zero, _, _ = _next_batch_row_indices(bundle, source, "bounds.xlsx", batch_size=0)
        assert pending_zero == []
        pending_all, _, meaningful_count = _next_batch_row_indices(bundle, source, "bounds.xlsx", batch_size=1000)
        assert pending_all == [0, 1, 2, 3]
        assert meaningful_count == 4


def test_next_batch_row_indices_empty_workbook(tmp_path):
    source = write_fixture_workbook(tmp_path / "empty.xlsx", [])
    for bundle in _both_backends(tmp_path):
        pending, total_rows, meaningful_count = _next_batch_row_indices(bundle, source, "empty.xlsx", batch_size=10)
        assert pending == []
        assert total_rows == 0
        assert meaningful_count == 0


# ------------------------------------------------------------- source-file aggregation


_AGGREGATE_ROWS = [
    {"comment_id": "a-1", "source_file": "agg.xlsx", "source_row_index": 0, "review_status": "EXCLUDED"},
    {
        "comment_id": "a-2",
        "source_file": "agg.xlsx",
        "source_row_index": 1,
        "review_status": "DRAFT_READY",
        "generation_routing_decision": "GENERATE_DRAFT",
        "generation_status": "SUCCEEDED",
    },
    {"comment_id": "a-3", "source_file": "agg.xlsx", "source_row_index": 2, "review_status": "NO_RESPONSE_RECOMMENDED"},
    {"comment_id": "a-4", "source_file": "agg.xlsx", "source_row_index": 3, "review_status": "REVIEW_ONLY"},
    {"comment_id": "a-5", "source_file": "agg.xlsx", "source_row_index": 4, "review_status": "DUPLICATE_CONTENT"},
    # a different source_file -- must never be counted into agg.xlsx's totals
    {"comment_id": "a-6", "source_file": "other.xlsx", "source_row_index": 0, "review_status": "EXCLUDED"},
]


def test_source_file_aggregate_parity_mixed_statuses(tmp_path):
    pg, mongo = _both_backends(tmp_path)
    _seed(pg, _AGGREGATE_ROWS)
    _seed(mongo, _AGGREGATE_ROWS)
    expected = {
        "processed_total": 5,
        "excluded_total": 1,
        "generated_drafts_total": 1,
        "no_response_recommended_total": 1,
        # a-2 (DRAFT_READY) AND a-4 (REVIEW_ONLY) both count -- the
        # buckets are independent, not mutually exclusive; a-2 counts
        # as both a generated draft and a needs-review row at once,
        # exactly like the original inline-SQL aggregation.
        "needs_review_total": 2,
        "duplicate_content_total": 1,
    }
    assert _source_file_aggregate(pg, "agg.xlsx") == expected
    assert _source_file_aggregate(mongo, "agg.xlsx") == expected


def test_source_file_aggregate_empty_dataset(tmp_path):
    for bundle in _both_backends(tmp_path):
        assert _source_file_aggregate(bundle, "never-uploaded.xlsx") == {
            "processed_total": 0,
            "excluded_total": 0,
            "generated_drafts_total": 0,
            "no_response_recommended_total": 0,
            "needs_review_total": 0,
            "duplicate_content_total": 0,
        }


def test_source_file_aggregate_generation_success_requires_both_fields(tmp_path):
    """A row routed to GENERATE_DRAFT that hasn't SUCCEEDED yet must not
    count as a generated draft on either backend -- the bucket is a
    conjunction, not just routing_decision alone."""
    rows = [
        {
            "comment_id": "g-1",
            "source_file": "gen.xlsx",
            "source_row_index": 0,
            "review_status": "DRAFT_READY",
            "generation_routing_decision": "GENERATE_DRAFT",
            "generation_status": "FAILED",
        }
    ]
    for bundle in _both_backends(tmp_path):
        _seed(bundle, rows)
        agg = _source_file_aggregate(bundle, "gen.xlsx")
        assert agg["generated_drafts_total"] == 0
        assert agg["processed_total"] == 1


# ------------------------------------------------------------- outbound dispatch: CAS + persistence


def _seed_approved_for_dispatch(bundle, comment_id: str = "d-1") -> None:
    _seed(
        bundle,
        [
            {
                "comment_id": comment_id,
                "source_file": "dispatch.xlsx",
                "source_row_index": 0,
                "review_status": "APPROVED",
                "reviewer_action": "SEND_AS_IS",
                "final_reply": "شكرًا لتواصلكم معنا.",
                "source_reference": "dispatch.xlsx",
                "ready_to_send": True,
                "row_version": 3,
            }
        ],
    )


class _ScriptedSender(OutboundSender):
    def __init__(self, ok: bool, *, external_id: str | None = None, error_code: str | None = None, error_message: str | None = None):
        self.ok = ok
        self.external_id = external_id
        self.error_code = error_code
        self.error_message = error_message
        self.calls: list[dict] = []

    def send_response(self, *, comment_id, source_reference, final_response, idempotency_key):
        self.calls.append(
            {
                "comment_id": comment_id,
                "source_reference": source_reference,
                "final_response": final_response,
                "idempotency_key": idempotency_key,
            }
        )
        return OutboundSendResult(
            ok=self.ok,
            comment_id=comment_id,
            idempotency_key=idempotency_key,
            external_id=self.external_id,
            error_code=self.error_code,
            error_message=self.error_message,
        )


@pytest.mark.parametrize("backend_index", [0, 1])
def test_begin_outbound_dispatch_version_conflict_no_mutation(tmp_path, backend_index):
    """A version mismatch raises immediately, before any field is
    touched -- never silently retried, on either backend."""
    bundle = _both_backends(tmp_path)[backend_index]
    _seed_approved_for_dispatch(bundle)
    before = bundle.review_items.get("d-1")
    with pytest.raises(VersionConflict):
        bundle.review_items.begin_outbound_dispatch("d-1", expected_version=999)
    after = bundle.review_items.get("d-1")
    assert after.review_status == before.review_status == "APPROVED"
    assert after.outbound_status == before.outbound_status
    assert after.row_version == before.row_version == 3


def test_outbound_dispatch_full_success_parity_between_backends(tmp_path):
    pg, mongo = _both_backends(tmp_path)
    for bundle in (pg, mongo):
        _seed_approved_for_dispatch(bundle)
        sender = _ScriptedSender(ok=True, external_id="ext-123")
        svc = ReviewWorkflowService(bundle)
        detail, result = svc.dispatch_outbound("d-1", sender, expected_version=3)

        assert result.ok is True
        assert result.external_id == "ext-123"
        assert len(sender.calls) == 1
        assert sender.calls[0]["final_response"] == "شكرًا لتواصلكم معنا."
        assert sender.calls[0]["idempotency_key"]  # deterministic, non-empty

        item = bundle.review_items.get("d-1")
        assert item.review_status == "SENT"
        assert item.outbound_status == "SENT"
        assert item.outbound_external_id == "ext-123"
        assert item.outbound_error_code is None
        assert item.outbound_error_message is None
        assert item.final_reply == "شكرًا لتواصلكم معنا."  # preserved, not cleared
        # row_version starts at 3; begin_outbound_dispatch and
        # complete_outbound_dispatch each bump it once (matching the
        # original single-function implementation's two _bump() calls),
        # so a full successful dispatch ends at 5, not 4.
        assert item.row_version == 5


def test_outbound_dispatch_failure_parity_between_backends(tmp_path):
    pg, mongo = _both_backends(tmp_path)
    for bundle in (pg, mongo):
        _seed_approved_for_dispatch(bundle)
        sender = _ScriptedSender(ok=False, error_code="TIMEOUT", error_message="upstream timed out")
        svc = ReviewWorkflowService(bundle)
        detail, result = svc.dispatch_outbound("d-1", sender)

        assert result.ok is False
        item = bundle.review_items.get("d-1")
        assert item.review_status == "SEND_FAILED"
        assert item.outbound_status == "FAILED"
        assert item.outbound_error_code == "TIMEOUT"
        assert item.outbound_error_message == "upstream timed out"
        assert item.ready_to_send is True  # SEND_FAILED is explicitly ready-to-retry


def test_outbound_dispatch_idempotent_short_circuit_never_calls_sender_again(tmp_path):
    """Once an item is fully SENT with an idempotency key recorded, a
    second dispatch_outbound call must not invoke the sender at all --
    proven on both backends."""
    pg, mongo = _both_backends(tmp_path)
    for bundle in (pg, mongo):
        _seed_approved_for_dispatch(bundle)
        first_sender = _ScriptedSender(ok=True, external_id="ext-first")
        svc = ReviewWorkflowService(bundle)
        svc.dispatch_outbound("d-1", first_sender)

        second_sender = _ScriptedSender(ok=True, external_id="ext-should-never-be-used")
        detail, result = svc.dispatch_outbound("d-1", second_sender)

        assert second_sender.calls == []  # never invoked
        assert result.ok is True
        assert result.external_id == "ext-first"  # the ORIGINAL send's outcome, unchanged


def test_outbound_dispatch_never_activates_unconfigured_sender(tmp_path):
    """UnconfiguredOutboundSender must remain non-live: it is the
    default adapter (sender=None) and, when used, always reports
    CONNECTOR_NOT_CONFIGURED rather than pretending to have sent
    anything -- dispatch_outbound never masks that as a success."""
    bundle = _sqlite_bundle(tmp_path)
    _seed_approved_for_dispatch(bundle)
    svc = ReviewWorkflowService(bundle)
    detail, result = svc.dispatch_outbound("d-1", UnconfiguredOutboundSender())
    assert result.ok is False
    assert result.error_code == "CONNECTOR_NOT_CONFIGURED"
    item = bundle.review_items.get("d-1")
    assert item.review_status == "SEND_FAILED"
    assert item.outbound_status == "FAILED"
    assert item.final_reply == "شكرًا لتواصلكم معنا."  # approved text never dropped
