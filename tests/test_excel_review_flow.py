"""Excel review-flow import/export. Source workbook is never overwritten."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from rcs.ingest.loader import _comment_id
from rcs.response_generation.schema import GenerationResult
from rcs.review.excel_review_flow import (
    _EXPORT_SHEET1_HEADERS,
    _EXPORT_SHEET1_TITLE,
    _EXPORT_SHEET2_HEADERS,
    _EXPORT_SHEET2_TITLE,
    apply_offline_generation,
    export_reviewed_workbook,
    import_subset,
    source_digest,
    write_fixture_workbook,
)


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def _read_export(out):
    """Phase 4: export is two sheets (business-facing Arabic + technical
    audit). Both are written in identical row order (one row per source
    row, never reordered between the two), so a plain column concat
    reconstructs a single combined view for test assertions."""
    sheet1 = pd.read_excel(out, sheet_name=_EXPORT_SHEET1_TITLE, dtype=str)
    sheet2 = pd.read_excel(out, sheet_name=_EXPORT_SHEET2_TITLE, dtype=str)
    assert list(sheet1.columns) == _EXPORT_SHEET1_HEADERS
    assert list(sheet2.columns) == _EXPORT_SHEET2_HEADERS
    return pd.concat([sheet1, sheet2], axis=1)
from rcs.review.excel_review_flow import apply_live_generation
from rcs.review.identity import ReviewerIdentity
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService
from rcs.review.workflow import InvalidTransition
from sqlalchemy.orm import Session

REVIEWER = ReviewerIdentity("rev-excel", "مراجع التصدير")


def _engine(tmp_path: Path):
    engine = get_engine(tmp_path / "excel_flow.sqlite")
    init_db(engine)
    return engine


def _fixture(path: Path) -> Path:
    return write_fixture_workbook(
        path,
        [
            {
                "original_comment": "شكرًا لكم على هذا المعجم الرائع.",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "زائر",
                "visitor_email": "visitor@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": dt.datetime(2026, 8, 1, 10, 0, 0),
                "platform": "الرياض",
            },
            {
                "original_comment": "كيف أضيف كلمة جديدة في الموقع؟",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "زائر",
                "visitor_email": "visitor2@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": dt.datetime(2026, 8, 2, 10, 0, 0),
                "platform": "سوار",
            },
            {
                "original_comment": "إعلان تجاري غير متعلق بالمعجم",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "زائر",
                "visitor_email": "spam@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": dt.datetime(2026, 8, 3, 10, 0, 0),
                "platform": "الرياض",
            },
        ],
    )


def _gen(comment_id: str, **overrides) -> GenerationResult:
    base = dict(
        comment_id=comment_id,
        routing_decision="GENERATE_DRAFT",
        generation_status="SUCCEEDED",
        suggested_response="شكرًا لتواصلكم معنا، نسعد بثقتكم بالمعجم.",
        review_required=True,
        reason_code="thanks_praise_eligible",
        provider="gemini",
        model_name="gemini-3.6-flash",
        prompt_version="rg-v7",
        generated_at="2026-08-26T09:00:00+00:00",
    )
    base.update(overrides)
    return GenerationResult(**base)


def test_source_bytes_unchanged_after_import_and_export(tmp_path):
    source = _fixture(tmp_path / "visitors-reports-fixture.xlsx")
    before = source_digest(source)
    engine = _engine(tmp_path)
    imported = import_subset(engine, source, limit=10)
    svc = ReviewWorkflowService(engine)
    cid = imported["comment_ids"][0]
    svc.apply_generation_result(_gen(cid))
    svc.send_as_is(cid, REVIEWER)
    export_reviewed_workbook(engine, tmp_path / "reviewed_responses_output.xlsx", imported["comment_ids"])
    assert source_digest(source) == before
    assert imported["inserted"] == 3


def test_duplicate_import_is_idempotent_and_preserves_review(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    first = import_subset(engine, source, limit=10)
    svc = ReviewWorkflowService(engine)
    cid = first["comment_ids"][0]
    suggestion = _gen(cid).suggested_response
    svc.apply_generation_result(_gen(cid))
    svc.send_as_is(cid, REVIEWER)
    second = import_subset(engine, source, limit=10)
    assert second["inserted"] == 0
    assert first["comment_ids"] == second["comment_ids"]
    with Session(engine) as session:
        item = session.get(ReviewItem, cid)
        assert item.reviewer_action == "SEND_AS_IS"
        assert item.final_reply == suggestion
        assert item.suggested_final_reply == suggestion
        assert session.query(ReviewItem).count() == 3


def test_export_matches_reviewer_actions(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    imported = import_subset(engine, source, limit=10)
    ids = imported["comment_ids"]
    fname = source.name
    assert ids[0] == _comment_id(fname, 0)
    svc = ReviewWorkflowService(engine)
    gemini = "شكرًا لتواصلكم معنا، نسعد بثقتكم بالمعجم."
    edited = "يمكنكم استخدام نموذج إضافة كلمة في الموقع."

    svc.apply_generation_result(_gen(ids[0], suggested_response=gemini), intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE")
    send = svc.send_as_is(ids[0], REVIEWER)
    assert send.outbound_status == "PENDING"
    assert send.review_status == "APPROVED"

    svc.apply_generation_result(
        _gen(ids[1], routing_decision="REVIEW_ONLY", generation_status="SKIPPED", suggested_response=None, reason_code="inquiry_without_trusted_grounding", provider=None, model_name=None),
        intent="INQUIRY",
    )
    with pytest.raises(InvalidTransition):
        svc.send_as_is(ids[1], REVIEWER)
    with pytest.raises(InvalidTransition):
        svc.edit_and_send(ids[1], REVIEWER, "   ")
    edit = svc.edit_and_send(ids[1], REVIEWER, edited)
    assert edit.suggested_response is None
    assert edit.edited_response == edited
    assert edit.final_response == edited
    assert edit.outbound_status == "PENDING"

    svc.apply_generation_result(_gen(ids[2], suggested_response=gemini), intent="THANKS_PRAISE")
    none = svc.mark_no_response_needed(ids[2], REVIEWER, reason="غير متعلق")
    assert none.final_response is None
    assert none.outbound_status == "NOT_APPLICABLE"

    out = tmp_path / "reviewed_responses_output.xlsx"
    export_reviewed_workbook(engine, out, ids)
    frame = _read_export(out)
    by_id = frame.set_index("comment_id")

    row0 = by_id.loc[ids[0]]
    assert row0["إجراء المراجع"] == "SEND_AS_IS"
    assert row0["الرد المقترح"] == gemini
    assert row0["الرد النهائي"] == gemini
    assert row0["اسم المراجع"] == REVIEWER.display_name
    assert pd.notna(row0["تاريخ المراجعة"])
    assert row0["outbound_status"] == "PENDING"
    assert row0["حالة المراجعة"] == "APPROVED"

    row1 = by_id.loc[ids[1]]
    assert row1["إجراء المراجع"] == "EDIT_AND_SEND"
    assert (row1["الرد المقترح"] in ("", "nan") or pd.isna(row1["الرد المقترح"]))
    assert row1["الرد بعد التعديل"] == edited
    assert row1["الرد النهائي"] == edited
    assert row1["outbound_status"] == "PENDING"

    row2 = by_id.loc[ids[2]]
    assert row2["إجراء المراجع"] == "NO_RESPONSE_NEEDED"
    assert row2["الرد المقترح"] == gemini
    assert row2["الرد النهائي"] in ("", "nan") or pd.isna(row2["الرد النهائي"])
    assert row2["اسم المراجع"] == REVIEWER.display_name

    assert "visitor_email" not in frame.columns


def test_edit_preserves_gemini_suggestion_in_export(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    cid = import_subset(engine, source, limit=1)["comment_ids"][0]
    svc = ReviewWorkflowService(engine)
    original = "مسودة جيميني الأصلية للتعليق."
    svc.apply_generation_result(_gen(cid, suggested_response=original))
    edited = "رد المراجع بعد التعديل على المسودة."
    detail = svc.edit_and_send(cid, REVIEWER, edited)
    assert detail.suggested_response == original
    assert detail.final_response == edited
    out = export_reviewed_workbook(engine, tmp_path / "out.xlsx", [cid])
    frame = _read_export(out)
    assert frame.loc[0, "الرد المقترح"] == original
    assert frame.loc[0, "الرد بعد التعديل"] == edited
    assert frame.loc[0, "الرد النهائي"] == edited
    with Session(engine) as session:
        item = session.get(ReviewItem, cid)
        assert item.ai_suggested_final_reply == original
        assert item.suggested_final_reply == original


def test_offline_generation_then_export(tmp_path):
    source = _fixture(tmp_path / "src.xlsx")
    engine = _engine(tmp_path)
    imported = import_subset(engine, source, limit=10)
    svc = ReviewWorkflowService(engine)
    apply_offline_generation(svc, imported["comment_ids"])
    dest = tmp_path / "out.xlsx"
    export_reviewed_workbook(engine, dest, imported["comment_ids"])
    frame = _read_export(dest)
    assert len(frame) == 3
    assert list(frame["comment_id"]) == imported["comment_ids"]


# --- Export privacy + source date (Phase — export privacy fix) -------------


def test_export_redacts_phone_includes_source_date_and_title_keeps_reviewed_at_distinct(tmp_path):
    """B) exported comment text is redacted (no raw phone numbers).
    C) comment_created_at (visitor's original date) is present and
       distinct from تاريخ المراجعة (reviewed_at, when the reviewer decided).
    D) entry_id and original_title are preserved in the export.
    Also: no visitor_name/visitor_email anywhere in either sheet.
    """
    source = write_fixture_workbook(
        tmp_path / "src_privacy.xlsx",
        [
            {
                "original_comment": "خطأ في هذا المدخل، تواصلوا معي على 0501234567 من فضلكم",
                "original_title": "حَسَبَ",
                "entry_id": "entry-9001",
                "lemma": None,
                "visitor_name": "زائر حقيقي",
                "visitor_email": "real.visitor@example.com",
                "original_type": "التعليق على المدخل",
                "comment_created_at": dt.datetime(2026, 6, 30, 12, 30, 0),
                "platform": "الرياض",
            },
        ],
    )
    engine = _engine(tmp_path)
    imported = import_subset(engine, source, limit=10)
    svc = ReviewWorkflowService(engine)
    apply_offline_generation(svc, imported["comment_ids"])
    cid = imported["comment_ids"][0]
    # A real reviewer decision so reviewed_at is set to a timestamp that
    # must differ from the visitor's original comment_created_at.
    svc.send_as_is(cid, REVIEWER)

    dest = tmp_path / "out_privacy.xlsx"
    export_reviewed_workbook(engine, dest, imported["comment_ids"])

    import pandas as pd

    sheet1 = pd.read_excel(dest, sheet_name=_EXPORT_SHEET1_TITLE, dtype=str)
    sheet2 = pd.read_excel(dest, sheet_name=_EXPORT_SHEET2_TITLE, dtype=str)
    row = sheet1.iloc[0]

    # B) redacted — no raw phone number, placeholder present instead.
    assert "0501234567" not in row["تعليق الزائر"]
    assert "[phone number removed]" in row["تعليق الزائر"]

    # C) source date present and distinct from the reviewer-decision date.
    assert "تاريخ التعليق" in sheet1.columns
    assert row["تاريخ التعليق"] and str(row["تاريخ التعليق"]) != "nan"
    assert "2026-06-30" in str(row["تاريخ التعليق"])
    assert row["تاريخ المراجعة"] and str(row["تاريخ المراجعة"]) != "nan"
    assert str(row["تاريخ التعليق"]) != str(row["تاريخ المراجعة"])

    # D) entry_id and original_title preserved.
    assert row["معرف المدخل"] == "entry-9001"
    assert "العنوان" in sheet1.columns
    assert row["العنوان"] == "حَسَبَ"

    # No visitor PII anywhere in either sheet.
    for col in sheet1.columns:
        assert col not in ("visitor_name", "visitor_email")
    combined_text = sheet1.to_csv() + sheet2.to_csv()
    assert "real.visitor@example.com" not in combined_text
    assert "زائر حقيقي" not in combined_text
    assert "0501234567" not in combined_text


def test_export_source_bytes_unchanged_with_new_export_columns(tmp_path):
    """A) exporting with the new العنوان/تاريخ التعليق columns must never
    touch the original uploaded source workbook."""
    source = write_fixture_workbook(
        tmp_path / "src_untouched.xlsx",
        [
            {
                "original_comment": "شكرًا لكم",
                "original_title": "كلمة",
                "entry_id": "e-1",
                "lemma": None,
                "visitor_name": "ز",
                "visitor_email": "z@example.com",
                "original_type": "رسالة تواصل",
                "comment_created_at": dt.datetime(2026, 8, 1, 10, 0, 0),
                "platform": "الرياض",
            },
        ],
    )
    before = source.read_bytes()
    engine = _engine(tmp_path)
    imported = import_subset(engine, source, limit=10)
    svc = ReviewWorkflowService(engine)
    apply_offline_generation(svc, imported["comment_ids"])
    export_reviewed_workbook(engine, tmp_path / "out_untouched.xlsx", imported["comment_ids"])
    assert source.read_bytes() == before


# --- Response-generation entry-context (grounding) — apply_live_generation ---


class _FakeGeminiClient:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.prompts_seen: list[str] = []

    def generate(self, prompt: str, response_schema: dict) -> str:
        import json

        self.prompts_seen.append(prompt)
        return json.dumps({"suggested_response": self._response_text}, ensure_ascii=False)


def _seed_classified_inquiry(engine, comment_id: str, **overrides):
    row = {
        "comment_id": comment_id,
        "source_file": "grounding.xlsx",
        "source_row_index": 0,
        "original_comment": "فيه خطأ في هذا المدخل، هل يمكن التأكد منه؟",
        "review_status": "CLASSIFIED",
        "primary_intent": "INQUIRY",
        "topic_subtype": "GENERAL",
        "ai_confidence": 0.85,
        "ready_to_send": False,
        "row_version": 1,
    }
    row.update(overrides)
    upsert_review_items(engine, [row])


def test_generation_grounding_a_vague_reference_with_entry_context(tmp_path):
    """A) A vague "هذا المدخل" comment WITH original_title/entry_id present
    must still reach Gemini with identifying grounding text in the
    prompt (title/entry_id are still useful for "which entry"). Phase
    M5D.1 correction: the eligibility ROUTE label no longer becomes
    "inquiry_with_trusted_grounding" for this case -- identifying
    context alone is not factual authority and must not grant the
    grounded-inquiry route by itself (see
    rcs.response_intelligence.evidence's module docstring and
    tests/test_m5d1_grounding_closure.py). Only real Official Knowledge
    (not wired into excel_review_flow's request construction, and with
    no configured corpus today regardless) can do that, so this row
    correctly routes as inquiry_without_trusted_grounding even though
    identifying grounding text is present in the prompt.

    Phase M5D.2 correction: identifying context now appears under its
    own "IDENTIFYING CONTEXT" prompt section, never under "TRUSTED
    GROUNDING"/"OFFICIAL KNOWLEDGE" -- that label is reserved exclusively
    for real Official Knowledge evidence, which this row has none of, so
    the OFFICIAL KNOWLEDGE section here correctly shows the explicit
    "(none — do not invent facts)" placeholder instead.

    Phase M6.2.1 correction: `entry_id` is internal metadata only and must
    never be rendered into the prompt -- only `original_title` (the
    human-readable identifying reference) is. This assertion previously
    required "entry-4471" in the prompt, which was the exact
    trust-boundary violation M6.2.1 fixed."""
    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-grounded", original_title="حَسَبَ", entry_id="entry-4471")
    svc = ReviewWorkflowService(engine)
    client = _FakeGeminiClient("بخصوص المدخل الذي أشرتم إليه، وصلتنا ملاحظتكم.")
    apply_live_generation(svc, ["c-grounded"], client)

    detail = svc.get_review_item("c-grounded")
    assert detail.generation_status == "SUCCEEDED"
    assert detail.generation_reason_code == "inquiry_without_trusted_grounding"
    assert len(client.prompts_seen) == 1
    prompt = client.prompts_seen[0]
    assert "IDENTIFYING CONTEXT" in prompt
    assert "OFFICIAL KNOWLEDGE" in prompt
    assert "(none — do not invent facts)" in prompt
    assert "حَسَبَ" in prompt
    assert "entry-4471" not in prompt
    assert "identifying metadata" in prompt


def test_generation_grounding_b_vague_reference_without_entry_context(tmp_path):
    """B) Same kind of vague comment, but no title/entry_id on the row —
    grounding must stay off (the "(none — do not invent facts)"
    fallback), exactly like before this fix, not a fabricated one."""
    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-ungrounded", original_title=None, entry_id=None)
    svc = ReviewWorkflowService(engine)
    client = _FakeGeminiClient("وصلتنا ملاحظتكم بخصوص الموضوع الذي أشرتم إليه.")
    apply_live_generation(svc, ["c-ungrounded"], client)

    detail = svc.get_review_item("c-ungrounded")
    assert detail.generation_status == "SUCCEEDED"
    assert detail.generation_reason_code == "inquiry_without_trusted_grounding"
    prompt = client.prompts_seen[0]
    assert "(none — do not invent facts)" in prompt


def test_generation_grounding_c_visitor_pii_never_enters_prompt(tmp_path):
    """C) visitor_name/visitor_email must never reach GenerationRequest
    or the built Gemini prompt, even when present on the same row as
    title/entry_id grounding."""
    engine = _engine(tmp_path)
    _seed_classified_inquiry(
        engine,
        "c-pii-check",
        original_title="حَسَبَ",
        entry_id="entry-4471",
        visitor_name="زائر حقيقي جدًا",
        visitor_email="do.not.leak@example.com",
    )
    svc = ReviewWorkflowService(engine)
    client = _FakeGeminiClient("بخصوص المدخل الذي أشرتم إليه، شكرًا لملاحظتكم.")
    apply_live_generation(svc, ["c-pii-check"], client)

    prompt = client.prompts_seen[0]
    assert "زائر حقيقي جدًا" not in prompt
    assert "do.not.leak@example.com" not in prompt
    assert "example.com" not in prompt


def test_generation_grounding_d_no_unsupported_facts_are_invented(tmp_path):
    """D) build_identifying_grounding only ever echoes the literal
    title/entry_id it was given — it must never contain definition-like
    or fact-like content beyond that identity, and must explicitly warn
    the model not to treat it as a definition/correction."""
    from rcs.response_generation.prompt import build_identifying_grounding

    text = build_identifying_grounding("حَسَبَ", "entry-4471")
    assert text is not None
    assert "حَسَبَ" in text
    assert "entry-4471" in text
    assert "identifying metadata" in text
    assert "not a definition" in text
    # Nothing beyond the two literal inputs is present — no invented
    # dictionary content, spelling correction, or factual claim.
    for forbidden in ("يعني", "تعريف", "الصواب", "الصحيح"):
        assert forbidden not in text

    assert build_identifying_grounding(None, None) is None
    only_title = build_identifying_grounding("حَسَبَ", None)
    assert only_title is not None
    assert "حَسَبَ" in only_title


# ============================================================================
# Phase M5E.1 — live audit wiring: the REAL production path
# (apply_live_generation -> generate_response_with_orchestrator ->
# run_agent_pipeline -> svc.audit_repo -> an actual persisted
# AgentExecutionTrace row) — not a direct repository call, which was
# already covered in M5E's own tests/test_agent_audit_repository.py.
# ============================================================================


class _ScriptedGeminiClient:
    """Like the file's own `_FakeGeminiClient`, but supports a list of
    responses (needed to exercise the one-bounded-refinement path) and
    optionally raising."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts_seen: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        import json

        self.prompts_seen.append(prompt)
        self.call_count += 1
        item = self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return json.dumps({"suggested_response": item}, ensure_ascii=False)


class _FailingAuditRepo:
    """Satisfies AgentAuditRepository structurally; always fails on
    record() to prove the documented M5E atomicity policy through the
    REAL call path, not just at the orchestrator-unit level."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.record_call_count = 0

    def record(self, record) -> None:
        self.record_call_count += 1
        raise self._exc

    def list_for_review_item(self, review_item_id):
        return ()

    def get_by_execution_id(self, execution_id):
        return None


def _query_traces_for(engine, review_item_id: str):
    from rcs.review.models import AgentExecutionTrace

    with Session(engine) as session:
        rows = (
            session.query(AgentExecutionTrace)
            .filter(AgentExecutionTrace.review_item_id == review_item_id)
            .order_by(AgentExecutionTrace.created_at)
            .all()
        )
        return [
            {c.name: getattr(row, c.name) for c in AgentExecutionTrace.__table__.columns}
            for row in rows
        ]


def test_m5e1_live_path_persists_exactly_one_audit_trace(tmp_path):
    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-live-1", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL")
    svc = ReviewWorkflowService(engine)
    client = _ScriptedGeminiClient(["شكرًا لملاحظتكم القيّمة حول هذا الأمر."])
    apply_live_generation(svc, ["c-live-1"], client)

    traces = _query_traces_for(engine, "c-live-1")
    assert len(traces) == 1  # test 1
    assert traces[0]["review_item_id"] == "c-live-1"  # test 2
    assert traces[0]["execution_id"]  # test 3
    assert traces[0]["logical_generation_count"] == 1  # test 4


def test_m5e1_live_path_refined_execution_persists_logical_generation_count_two(tmp_path):
    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-live-2", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL")
    svc = ReviewWorkflowService(engine)
    # First draft too short -> refinable failure -> exactly one refinement.
    client = _ScriptedGeminiClient(["شكرًا", "شكرًا لملاحظتكم القيّمة حول هذا الأمر."])
    apply_live_generation(svc, ["c-live-2"], client)

    traces = _query_traces_for(engine, "c-live-2")
    assert len(traces) == 1  # test 5 (one execution, refined internally)
    assert traces[0]["logical_generation_count"] == 2
    assert traces[0]["refinement_attempted"] is True
    assert client.call_count == 2  # test 27: max logical Gemini calls <= 2


def test_m5e1_live_path_retrieval_and_grounding_reach_trace(tmp_path):
    engine = _engine(tmp_path)
    _seed_classified_inquiry(
        engine, "c-live-3", primary_intent="INQUIRY", topic_subtype="GENERAL",
        original_title="حَسَبَ", entry_id="entry-4471",
    )
    svc = ReviewWorkflowService(engine)
    client = _ScriptedGeminiClient(["بخصوص المدخل الذي أشرتم إليه، شكرًا لملاحظتكم."])
    apply_live_generation(svc, ["c-live-3"], client)

    traces = _query_traces_for(engine, "c-live-3")
    trace = traces[0]
    # test 6: no Official Knowledge source is configured today -- SUCCESS_EMPTY reaches the trace.
    assert trace["official_retrieval_outcome"] == "SUCCESS_EMPTY"
    assert trace["style_retrieval_outcome"] == "SUCCESS_EMPTY"
    # test 7: identifying-context-only (no Official Knowledge) -> IDENTIFYING_ONLY mode.
    assert trace["grounding_mode"] == "IDENTIFYING_ONLY"
    assert trace["official_grounding_available"] is False
    # test 8 / 28: M5D.1-correct -- identifying context alone does not grant the grounded route.
    assert trace["final_validation_passed"] is True
    assert trace["generation_status"] == "SUCCEEDED"
    detail = svc.get_review_item("c-live-3")
    assert detail.generation_reason_code == "inquiry_without_trusted_grounding"
    # test 29: M5D.2-correct prompt trust labels reached the real prompt.
    prompt = client.prompts_seen[0]
    assert "IDENTIFYING CONTEXT" in prompt
    assert "OFFICIAL KNOWLEDGE" in prompt
    assert "حَسَبَ" in prompt


def test_m5e1_reviewer_approved_row_remains_protected_with_live_audit_wired(tmp_path):
    engine = _engine(tmp_path)
    _seed_classified_inquiry(
        engine, "c-live-4", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL",
        review_status="APPROVED", reviewer_action="SEND_AS_IS", final_reply="رد المراجع المعتمد.",
        ready_to_send=True,
    )
    svc = ReviewWorkflowService(engine)
    client = _ScriptedGeminiClient(["رد جديد لن يُستخدم أبدًا لأن الصف محمي مسبقًا."])
    # apply_live_generation only targets CLASSIFIED rows -- an already-
    # APPROVED row is untouched by construction; this proves the audit
    # wiring introduces no NEW path around that protection.
    apply_live_generation(svc, ["c-live-4"], client)

    detail = svc.get_review_item("c-live-4")
    assert detail.review_status == "APPROVED"  # test 9
    assert detail.final_response == "رد المراجع المعتمد."
    assert client.call_count == 0


def test_m5e1_audit_failure_does_not_break_live_generation_or_mutate_data(tmp_path):
    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-live-5", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL")
    failing_repo = _FailingAuditRepo(RuntimeError("synthetic DB outage with /secret/conn leak"))
    svc = ReviewWorkflowService(engine, audit_repo=failing_repo)
    client = _ScriptedGeminiClient(["شكرًا لملاحظتكم القيّمة حول هذا الأمر."])
    apply_live_generation(svc, ["c-live-5"], client)  # test 13: must not raise

    detail = svc.get_review_item("c-live-5")
    assert detail.generation_status == "SUCCEEDED"  # business result unaffected
    assert detail.suggested_response == "شكرًا لملاحظتكم القيّمة حول هذا الأمر."
    assert failing_repo.record_call_count == 1  # test 14: audit was attempted, not silently skipped


def test_m5e1_audit_failure_logs_sanitized_message_only(tmp_path, caplog):
    import logging

    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-live-6", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL")
    failing_repo = _FailingAuditRepo(RuntimeError("synthetic DB outage with /secret/conn leak"))
    svc = ReviewWorkflowService(engine, audit_repo=failing_repo)
    client = _ScriptedGeminiClient(["شكرًا لملاحظتكم القيّمة حول هذا الأمر."])
    with caplog.at_level(logging.WARNING, logger="rcs.response_intelligence.orchestrator"):
        apply_live_generation(svc, ["c-live-6"], client)

    assert len(caplog.records) == 1  # test 15
    message = caplog.records[0].getMessage()
    assert "RuntimeError" in message
    assert "/secret/conn" not in message
    assert "synthetic DB outage" not in message


def test_m5e1_no_duplicate_trace_from_one_successful_execution(tmp_path):
    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-live-7", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL")
    svc = ReviewWorkflowService(engine)
    client = _ScriptedGeminiClient(["شكرًا لملاحظتكم القيّمة حول هذا الأمر."])
    apply_live_generation(svc, ["c-live-7"], client)
    assert len(_query_traces_for(engine, "c-live-7")) == 1  # test 16


def test_m5e1_current_runtime_uses_postgres_audit_repository(tmp_path):
    from rcs.review.repositories.postgres import PostgresAgentAuditRepository

    engine = _engine(tmp_path)
    svc = ReviewWorkflowService(engine)
    assert isinstance(svc.audit_repo, PostgresAgentAuditRepository)  # test 17


def test_m5e1_no_mongo_or_backend_selector_in_service_wiring():
    """tests 18-20: MongoAgentAuditRepository is never referenced by
    ReviewWorkflowService/excel_review_flow.py, no backend-selection
    logic (env var / if-dialect branch) was added, and no live Mongo
    connection is required anywhere in this test module."""
    import inspect

    from rcs.review import excel_review_flow as excel_review_flow_module
    from rcs.review import service as service_module

    for module in (service_module, excel_review_flow_module):
        source = inspect.getsource(module)
        assert "MongoAgentAuditRepository" not in source
        assert "mongodb://" not in source
        assert "mongodb+srv://" not in source


def test_m5e1_no_forbidden_content_in_real_persisted_trace(tmp_path):
    """tests 21-26: exercise the REAL call path with adversarial content
    and inspect the ACTUAL persisted database row (not a mock/fake)."""
    engine = _engine(tmp_path)
    raw_comment = "تعليق حقيقي جدًا برقم 0501234567 وبريد leak@example.com"
    _seed_classified_inquiry(
        engine, "c-live-8", primary_intent="INQUIRY", topic_subtype="GENERAL",
        original_comment=raw_comment, original_title="عنوان سري للغاية", entry_id="ENTRY-SECRET",
    )
    svc = ReviewWorkflowService(engine)
    client = _ScriptedGeminiClient(["بخصوص المدخل الذي أشرتم إليه، شكرًا لملاحظتكم."])
    apply_live_generation(svc, ["c-live-8"], client)

    trace = _query_traces_for(engine, "c-live-8")[0]
    trace_repr = repr(trace)
    assert raw_comment not in trace_repr  # 22: no raw visitor comment
    assert "عنوان سري للغاية" not in trace_repr  # 23: no identifying-context text
    assert "ENTRY-SECRET" not in trace_repr
    assert "0501234567" not in trace_repr
    assert "leak@example.com" not in trace_repr
    assert client.prompts_seen[0] not in trace_repr  # 21/25: no prompt persisted
    # 24/26: field-level structural proof (no content-shaped column exists at all).
    for banned in ("prompt", "reasoning", "chain", "thought", "knowledge_text", "style_text"):
        assert not any(banned in key.lower() for key in trace.keys())


def test_m5e1_generation_result_schema_unchanged_through_live_path(tmp_path):
    from dataclasses import fields

    engine = _engine(tmp_path)
    _seed_classified_inquiry(engine, "c-live-9", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL")
    svc = ReviewWorkflowService(engine)
    client = _ScriptedGeminiClient(["شكرًا لملاحظتكم القيّمة حول هذا الأمر."])
    apply_live_generation(svc, ["c-live-9"], client)

    field_names = {f.name for f in fields(GenerationResult)}
    assert field_names == {
        "comment_id", "routing_decision", "generation_status", "suggested_response",
        "review_required", "reason_code", "provider", "model_name", "prompt_version",
        "generated_at", "future_reviewer_actions", "eligibility_rule_id", "error",
    }
