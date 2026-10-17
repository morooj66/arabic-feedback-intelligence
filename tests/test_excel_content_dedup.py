"""Exact-content deduplication for the Excel operational flow.

Fixture (per the agreed acceptance scenario):
  - 4 exact-duplicate valid comments (should dedupe to 1 canonical)
  - 2 exact-duplicate comments likely to resolve to no-response
  - 2 exact-duplicate KNOWN junk/bot-flood rows (must be excluded by the
    deterministic prefilter before dedup/classification ever sees them)
  - 3 unique comments (must be completely unaffected by dedup)
"""

from __future__ import annotations

import datetime as dt
import json

from rcs.review.excel_review_flow import (
    export_reviewed_workbook,
    process_uploaded_excel,
    run_content_deduplication,
    source_digest,
    write_fixture_workbook,
)
from rcs.review.identity import ReviewerIdentity
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from rcs.review.service import ReviewWorkflowService
from sqlalchemy.orm import Session

VALID_DUP_TEXT = "أقترح إضافة كلمة رائعة إلى المعجم لفائدة الباحثين والدارسين"
NO_RESPONSE_DUP_TEXT = "اشترك الآن في قناتنا على يوتيوب وشاهد أحدث الفيديوهات على الرابط"
JUNK_DUP_TEXT = "jhnjkn njnun Phone Number: 111111111111j"  # settings.KNOWN_JUNK_STRINGS[0]
UNIQUE_TEXTS = [
    "هل يمكن توضيح معنى كلمة استبانة من فضلكم؟",
    "شكرًا جزيلاً على هذا المعجم الرائع والمفيد",
    "لاحظت خطأ إملائيًا في أحد مدخلات المعجم",
]


class _RecordingClient:
    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.calls.append(prompt)
        return json.dumps(
            {"suggested_response": f"ردٌ رقم {len(self.calls)} مخصص لهذا التعليق."}, ensure_ascii=False
        )

    def calls_mentioning(self, text: str) -> int:
        return sum(1 for p in self.calls if text in p)


def _fixture(path):
    rows = []
    t0 = dt.datetime(2026, 8, 1, 9, 0, 0)

    def _row(text, minutes, email):
        return {
            "original_comment": text,
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "زائر",
            "visitor_email": email,
            "original_type": "رسالة تواصل",
            "comment_created_at": t0 + dt.timedelta(minutes=minutes),
            "platform": "الرياض",
        }

    for i in range(4):
        rows.append(_row(VALID_DUP_TEXT, i, f"valid{i}@example.com"))
    for i in range(2):
        rows.append(_row(NO_RESPONSE_DUP_TEXT, 10 + i, f"promo{i}@example.com"))
    for i in range(2):
        rows.append(_row(JUNK_DUP_TEXT, 20 + i, f"junk{i}@example.com"))
    for i, text in enumerate(UNIQUE_TEXTS):
        rows.append(_row(text, 30 + i, f"unique{i}@example.com"))

    return write_fixture_workbook(path, rows)


def _engine(tmp_path):
    engine = get_engine(tmp_path / "dedup_flow.sqlite")
    init_db(engine)
    return engine


def _by_text(engine, text: str) -> list[ReviewItem]:
    with Session(engine) as session:
        items = session.query(ReviewItem).filter(ReviewItem.original_comment == text).all()
        session.expunge_all()
        return items


def test_dedup_full_scenario(tmp_path):
    source = _fixture(tmp_path / "dedup_src.xlsx")
    engine = _engine(tmp_path)
    before = source_digest(source)
    client = _RecordingClient()

    summary = process_uploaded_excel(engine, source, limit=11, client=client)

    assert summary.total_rows == 11
    assert summary.inserted_rows == 11
    assert source_digest(source) == before  # never touched

    # --- 4 exact-duplicate valid comments -> exactly 1 canonical ----------
    valid_group = _by_text(engine, VALID_DUP_TEXT)
    assert len(valid_group) == 4
    keys = {i.content_dedupe_key for i in valid_group}
    assert len(keys) == 1 and next(iter(keys)) is not None
    canon_ids = {i.dedupe_canonical_id for i in valid_group}
    assert len(canon_ids) == 1
    canonical_id = next(iter(canon_ids))
    canonicals = [i for i in valid_group if i.comment_id == canonical_id]
    duplicates = [i for i in valid_group if i.comment_id != canonical_id]
    assert len(canonicals) == 1
    assert len(duplicates) == 3
    for d in duplicates:
        assert d.review_status == "DUPLICATE_CONTENT"
        assert d.ai_primary_intent is None  # never independently classified
        assert d.generation_status is None  # never independently generated
    # Only the canonical was ever classified/routed anywhere.
    assert canonicals[0].review_status in ("DRAFT_READY", "REVIEW_ONLY", "NO_RESPONSE_RECOMMENDED")
    # At most one Gemini call for this exact text, however eligibility resolved it.
    assert client.calls_mentioning(VALID_DUP_TEXT) <= 1

    # --- 2 exact-duplicate no-response-shaped comments --------------------
    nr_group = _by_text(engine, NO_RESPONSE_DUP_TEXT)
    assert len(nr_group) == 2
    nr_canon_ids = {i.dedupe_canonical_id for i in nr_group}
    assert len(nr_canon_ids) == 1  # same canonical for both
    nr_dupes = [i for i in nr_group if i.review_status == "DUPLICATE_CONTENT"]
    assert len(nr_dupes) == 1  # exactly one of the two is the non-canonical duplicate
    assert client.calls_mentioning(NO_RESPONSE_DUP_TEXT) <= 1

    # --- 2 exact-duplicate KNOWN junk rows: excluded, never reach dedup/Gemini
    junk_group = _by_text(engine, JUNK_DUP_TEXT)
    assert len(junk_group) == 2
    for j in junk_group:
        assert j.is_prefiltered_junk is True
        assert j.review_status == "EXCLUDED"
        assert j.dedupe_canonical_id is None  # excluded rows never get a canonical
        assert j.content_dedupe_key is None
    assert client.calls_mentioning(JUNK_DUP_TEXT) == 0

    # --- unique comments: each is its own singleton canonical --------------
    for text in UNIQUE_TEXTS:
        rows = _by_text(engine, text)
        assert len(rows) == 1
        assert rows[0].dedupe_canonical_id == rows[0].comment_id

    # --- reviewer queue shows the duplicate group once, not four times ----
    svc = ReviewWorkflowService(engine)
    page = svc.list_review_queue(offset=0, limit=50)
    queue_ids = {item.comment_id for item in page.items}
    for d in duplicates:
        assert d.comment_id not in queue_ids
    for j in junk_group:
        assert j.comment_id not in queue_ids
    if canonicals[0].review_status in ("DRAFT_READY", "REVIEW_ONLY"):
        assert canonicals[0].comment_id in queue_ids


def test_dedup_reviewer_action_applies_to_canonical_duplicates_preserved(tmp_path):
    source = _fixture(tmp_path / "dedup_src.xlsx")
    engine = _engine(tmp_path)
    client = _RecordingClient()
    process_uploaded_excel(engine, source, limit=11, client=client)

    valid_group = _by_text(engine, VALID_DUP_TEXT)
    canon_id = valid_group[0].dedupe_canonical_id
    canonical = next(i for i in valid_group if i.comment_id == canon_id)
    duplicates = [i for i in valid_group if i.comment_id != canon_id]

    svc = ReviewWorkflowService(engine)
    reviewer = ReviewerIdentity("rev-dedup", "مراجع التكرار")

    if canonical.review_status == "DRAFT_READY":
        detail = svc.send_as_is(canon_id, reviewer)
        assert detail.review_status == "APPROVED"
        assert detail.reviewer_action == "SEND_AS_IS"
    elif canonical.review_status == "REVIEW_ONLY":
        detail = svc.edit_and_send(canon_id, reviewer, "رد يدوي من المراجع.")
    else:
        detail = svc.mark_no_response_needed(canon_id, reviewer, reason="لا يستدعي ردًا")

    # Duplicates were never a valid workflow target and remain DUPLICATE_CONTENT.
    with Session(engine) as session:
        for d in duplicates:
            item = session.get(ReviewItem, d.comment_id)
            assert item.review_status == "DUPLICATE_CONTENT"
            assert item.reviewer_action is None

    # --- export re-expands to all 4 original rows, all sharing the canonical's decision
    from rcs.review.excel_review_flow import _EXPORT_SHEET1_TITLE, _EXPORT_SHEET2_TITLE

    out = export_reviewed_workbook(engine, tmp_path / "out.xlsx")
    import pandas as pd

    # Phase 4: two-sheet export (business-facing Arabic + technical audit),
    # written in identical row order — concat to reconstruct a combined view.
    sheet1 = pd.read_excel(out, sheet_name=_EXPORT_SHEET1_TITLE, dtype=str)
    sheet2 = pd.read_excel(out, sheet_name=_EXPORT_SHEET2_TITLE, dtype=str)
    frame = pd.concat([sheet1, sheet2], axis=1)
    dup_ids = {d.comment_id for d in duplicates}
    exported_dup_rows = frame[frame["comment_id"].isin(dup_ids)]
    assert len(exported_dup_rows) == 3  # every original duplicate row still exported

    def _norm(v):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    for _, row in exported_dup_rows.iterrows():
        assert row["إجراء المراجع"] == detail.reviewer_action
        assert row["حالة المراجعة"] == detail.review_status
        assert _norm(row["الرد النهائي"]) == _norm(detail.final_response)
        assert row["تعليق الزائر"] == VALID_DUP_TEXT  # each row keeps its OWN text/provenance
    canonical_row = frame[frame["comment_id"] == canon_id].iloc[0]
    assert canonical_row["إجراء المراجع"] == detail.reviewer_action
    # total export row count == total imported rows (no row collapsed away)
    assert len(frame) == 11


def test_dedup_reimport_is_idempotent(tmp_path):
    source = _fixture(tmp_path / "dedup_src.xlsx")
    engine = _engine(tmp_path)
    before = source_digest(source)
    client = _RecordingClient()

    process_uploaded_excel(engine, source, limit=11, client=client)
    calls_after_first = len(client.calls)

    with Session(engine) as session:
        row_count_after_first = session.query(ReviewItem).count()

    second = process_uploaded_excel(engine, source, limit=11, client=client)

    assert source_digest(source) == before
    assert second.inserted_rows == 0
    assert len(client.calls) == calls_after_first  # zero new Gemini calls on re-import
    with Session(engine) as session:
        assert session.query(ReviewItem).count() == row_count_after_first  # no duplicate rows created


def test_dedup_backfill_never_converts_a_row_a_reviewer_already_engaged(tmp_path):
    """Regression test for the real edge case found before applying dedup
    to the live operational DB: a duplicate group may contain rows that
    already went through generation/review BEFORE dedup existed — one
    already APPROVED, one currently IN_REVIEW (claimed by a reviewer, no
    decision yet). Running the dedup backfill over such a group must
    never overwrite either row's status/decision, even when the claimed
    (IN_REVIEW) row has an earlier source_row_index than the approved one.
    """
    engine = _engine(tmp_path)
    shared_text = VALID_DUP_TEXT
    now = dt.datetime(2026, 8, 27, 14, 55, 0)

    # Simulate the pre-dedup production state directly: 4 rows sharing
    # content, already at various post-generation/review states, as if
    # process_uploaded_excel had run against them before this feature
    # existed (no content_dedupe_key/dedupe_canonical_id set yet). Row
    # order deliberately puts the IN_REVIEW (claimed, undecided) row at
    # an EARLIER source_row_index than the APPROVED row, to prove
    # canonical selection prefers a real decision over row order.
    rows = [
        ReviewItem(
            comment_id="claimed-earlier-row",
            source_file="prod.xlsx",
            source_row_index=10,
            original_comment=shared_text,
            review_status="IN_REVIEW",
            reviewer_id="demo:reviewer",
            reviewer_display_name="مروج الراجحي",
            reviewer_started_at=now,
            suggested_final_reply="مسودة قديمة لهذا الصف المطالب به.",
        ),
        ReviewItem(
            comment_id="approved-later-row",
            source_file="prod.xlsx",
            source_row_index=11,
            original_comment=shared_text,
            review_status="APPROVED",
            reviewer_id="demo:reviewer",
            reviewer_display_name="مروج الراجحي",
            reviewer_action="SEND_AS_IS",
            final_reply="الرد المعتمد فعليًا لهذا التعليق.",
            suggested_final_reply="الرد المعتمد فعليًا لهذا التعليق.",
            outbound_status="PENDING",
        ),
        ReviewItem(
            comment_id="untouched-draft-1",
            source_file="prod.xlsx",
            source_row_index=12,
            original_comment=shared_text,
            review_status="DRAFT_READY",
            suggested_final_reply="مسودة أخرى لم يطّلع عليها أحد بعد.",
        ),
        ReviewItem(
            comment_id="untouched-draft-2",
            source_file="prod.xlsx",
            source_row_index=13,
            original_comment=shared_text,
            review_status="DRAFT_READY",
            suggested_final_reply="ومسودة ثالثة أيضًا لم يطّلع عليها أحد.",
        ),
    ]
    all_ids = [r.comment_id for r in rows]
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()

    result = run_content_deduplication(engine, all_ids)

    with Session(engine) as session:
        claimed = session.get(ReviewItem, "claimed-earlier-row")
        approved = session.get(ReviewItem, "approved-later-row")
        draft1 = session.get(ReviewItem, "untouched-draft-1")
        draft2 = session.get(ReviewItem, "untouched-draft-2")

        # Neither reviewer-engaged row was converted or altered in any way.
        assert claimed.review_status == "IN_REVIEW"
        assert claimed.reviewer_started_at == now
        assert claimed.dedupe_canonical_id is None  # left completely untouched
        assert approved.review_status == "APPROVED"
        assert approved.reviewer_action == "SEND_AS_IS"
        assert approved.final_reply == "الرد المعتمد فعليًا لهذا التعليق."

        # The APPROVED row becomes canonical (protected + real decision),
        # even though the claimed row has an earlier row_index.
        assert approved.dedupe_canonical_id == "approved-later-row"

        # The two plain, never-touched drafts are correctly redirected.
        assert draft1.review_status == "DUPLICATE_CONTENT"
        assert draft1.dedupe_canonical_id == "approved-later-row"
        assert draft2.review_status == "DUPLICATE_CONTENT"
        assert draft2.dedupe_canonical_id == "approved-later-row"

    assert result["duplicate_content"] == 2  # only the two plain drafts converted
    assert result["canonical"] == 1

