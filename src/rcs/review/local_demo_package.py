"""Closed local end-to-end demo: ingest → classify persist → generate → review → export.

Isolated SQLite only. Never writes the production DB, gold datasets, or any external platform.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from config import settings
from rcs.prefilter.spam_rules import BurstConfig, apply_prefilter
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest, generate_suggested_response
from rcs.review.demo_seed import DEMO_DB, _ScriptedGenerationClient, _review_only_skip
from rcs.review.excel_review_flow import (
    export_reviewed_workbook,
    import_subset,
    write_fixture_workbook,
)
from rcs.review.identity import demo_microsoft_identity
from rcs.review.models import ReviewItem
from rcs.review.repository import (
    apply_classification_results,
    apply_prefilter_results,
    get_engine,
    init_db,
    load_for_prefilter,
)
from rcs.review.service import ReviewWorkflowService

PACKAGE_DIR = settings.BASE_DIR / "demo_package"
PACKAGE_DB = settings.DB_DIR / "local_demo_package.sqlite"
PACKAGE_EXPORT = PACKAGE_DIR / "reviewed_responses_output.xlsx"

THANKS_TEXT = "شكرًا لكم على هذا المعجم الرائع، استفدت كثيرًا من شروحه."
COMPLAINT_TEXT = "الموقع لا يعمل عند البحث عن بعض الكلمات، أرجو المعالجة."
INCOMPLETE_TEXT = "محتوى غير مكتمل يحتاج مراجعة بشرية قبل صياغة رد."
SPAM_TEXT = "إعلان تجاري غير ذي صلة بالمعجم."

_EDIT_SUFFIX = " إضافة من المراجع للعرض."
_MANUAL_REPLY = "يمكنكم استخدام نموذج إضافة كلمة في الموقع."


def _fixture_rows() -> list[dict]:
    now = dt.datetime(2026, 8, 26, 10, 0, 0)
    return [
        {
            "original_comment": THANKS_TEXT,
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "زائر",
            "visitor_email": "thanks@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": now,
            "platform": "المعجم",
        },
        {
            "original_comment": COMPLAINT_TEXT,
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "زائر",
            "visitor_email": "complaint@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": now - dt.timedelta(hours=2),
            "platform": "المعجم",
        },
        {
            "original_comment": INCOMPLETE_TEXT,
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "زائر",
            "visitor_email": "incomplete@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": now - dt.timedelta(hours=3),
            "platform": "المعجم",
        },
        {
            "original_comment": SPAM_TEXT,
            "original_title": None,
            "entry_id": None,
            "lemma": None,
            "visitor_name": "زائر",
            "visitor_email": "ads@example.com",
            "original_type": "رسالة تواصل",
            "comment_created_at": now - dt.timedelta(days=1),
            "platform": "المعجم",
        },
    ]


def _role_for_text(text: str | None) -> str:
    body = text or ""
    if body == THANKS_TEXT:
        return "thanks"
    if body == COMPLAINT_TEXT:
        return "complaint"
    if body == INCOMPLETE_TEXT:
        return "incomplete"
    if body == SPAM_TEXT:
        return "spam"
    return "other"


def _classification_payload(comment_id: str, role: str) -> dict:
    if role == "thanks":
        return {
            "comment_id": comment_id,
            "success": True,
            "ai_primary_intent": "THANKS_PRAISE",
            "ai_topic_subtype": "GENERAL",
            "ai_subcategory": None,
            "ai_secondary_intent": None,
            "ai_routing_decision": "REVIEW_ONLY",
            "ai_classification_reason": "local_demo_closed_taxonomy",
            "ai_confidence": 0.92,
            "routing_reason_code": None,
            "ai_classification_source": "local_demo_package",
        }
    if role == "complaint":
        return {
            "comment_id": comment_id,
            "success": True,
            "ai_primary_intent": "COMPLAINT",
            "ai_topic_subtype": "TECHNICAL_ISSUE",
            "ai_subcategory": None,
            "ai_secondary_intent": None,
            "ai_routing_decision": "REVIEW_ONLY",
            "ai_classification_reason": "local_demo_closed_taxonomy",
            "ai_confidence": 0.81,
            "routing_reason_code": None,
            "ai_classification_source": "local_demo_package",
            "ai_needs_review": True,
        }
    if role == "incomplete":
        return {
            "comment_id": comment_id,
            "success": True,
            "ai_primary_intent": "INQUIRY",
            "ai_topic_subtype": "GENERAL",
            "ai_subcategory": None,
            "ai_secondary_intent": None,
            "ai_routing_decision": "REVIEW_ONLY",
            "ai_classification_reason": "incomplete_needs_human_draft",
            "ai_confidence": 0.33,
            "routing_reason_code": None,
            "ai_classification_source": "local_demo_package",
            "ai_needs_review": True,
        }
    return {
        "comment_id": comment_id,
        "success": True,
        "ai_primary_intent": "SPAM",
        "ai_topic_subtype": "OFF_TOPIC_PERSONAL",
        "ai_subcategory": None,
        "ai_secondary_intent": None,
        "ai_routing_decision": "EXCLUDE",
        "ai_classification_reason": "off_topic_commercial",
        "ai_confidence": 0.9,
        "routing_reason_code": "spam",
        "ai_classification_source": "local_demo_package",
    }


def _load_live_drafts(demo_db: Path) -> dict[str, tuple[str, str | None, str | None]]:
    """Map original_comment → (suggested_response, prompt_version, model_name)."""
    if not demo_db.exists():
        return {}
    engine = get_engine(demo_db)
    out: dict[str, tuple[str, str | None, str | None]] = {}
    with Session(engine) as session:
        for item in session.query(ReviewItem).all():
            text = item.original_comment or ""
            draft = item.suggested_final_reply
            if text and draft:
                out[text] = (draft, item.generation_prompt_version, item.generation_model_name)
    return out


def _generation_for(
    item: ReviewItem,
    *,
    client,
    live_drafts: dict[str, tuple[str, str | None, str | None]],
) -> GenerationResult:
    role = _role_for_text(item.original_comment)
    if role == "incomplete":
        return _review_only_skip(item.comment_id)
    cached = live_drafts.get(item.original_comment or "")
    if cached and role in {"thanks", "complaint"}:
        draft, prompt_version, model_name = cached
        return GenerationResult(
            comment_id=item.comment_id,
            routing_decision="GENERATE_DRAFT",
            generation_status="SUCCEEDED",
            suggested_response=draft,
            review_required=True,
            reason_code="local_demo_current_pipeline",
            provider=settings.RESPONSE_GENERATION_PROVIDER,
            model_name=model_name or settings.RESPONSE_GENERATION_MODEL_NAME,
            prompt_version=prompt_version or settings.RESPONSE_GENERATION_PROMPT_VERSION,
            generated_at=dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        )
    return generate_suggested_response(
        GenerationRequest(
            comment_id=item.comment_id,
            comment_text=item.original_comment or "",
            primary_intent=item.primary_intent,
            topic_subtype=item.topic_subtype,
            sentiment=item.sentiment,
            confidence=item.ai_confidence,
            needs_review=bool(item.needs_review),
            is_content_meaningful=True,
            insufficient_context=role == "incomplete",
            trusted_grounding_available=False,
            classification_routing_decision=item.routing_decision,
        ),
        client=client,
    )


def _assert_not_sent(detail) -> None:
    assert detail.review_status != "SENT"
    assert detail.outbound_status != "SENT"


def run_closed_local_demo(
    *,
    db_path: Path | None = None,
    export_path: Path | None = None,
    source_workbook: Path | None = None,
    generation_client=None,
    live_drafts_db: Path | None = None,
) -> dict[str, Any]:
    db_path = db_path or PACKAGE_DB
    export_path = export_path or PACKAGE_EXPORT
    if db_path.exists():
        db_path.unlink()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if source_workbook is None:
        source_workbook = PACKAGE_DIR / "_local_demo_source.xlsx"
        write_fixture_workbook(source_workbook, _fixture_rows())

    engine = get_engine(db_path)
    init_db(engine)
    imported = import_subset(engine, source_workbook, limit=10)
    ids = list(imported["comment_ids"])
    assert len(ids) == 4

    df = load_for_prefilter(engine)
    flagged = apply_prefilter(
        df,
        settings.KNOWN_JUNK_STRINGS,
        BurstConfig(min_count=settings.BURST_MIN_COUNT, window_hours=settings.BURST_WINDOW_HOURS),
    )
    apply_prefilter_results(engine, flagged[["comment_id", "is_prefiltered_junk", "prefilter_reason"]])

    with Session(engine) as session:
        items = [session.get(ReviewItem, cid) for cid in ids]
        class_rows = [_classification_payload(it.comment_id, _role_for_text(it.original_comment)) for it in items]
    classified = apply_classification_results(
        engine,
        class_rows,
        model_name="local_demo_closed_taxonomy",
        model_version=None,
        prompt_version="local-demo-e2e",
        pipeline_version=settings.PIPELINE_VERSION,
        inference_run_id="local-demo-package",
    )

    svc = ReviewWorkflowService(engine)
    client = generation_client or _ScriptedGenerationClient()
    live_drafts = _load_live_drafts(live_drafts_db or DEMO_DB)
    roles: dict[str, str] = {}
    pending_generation: list[tuple[GenerationResult, ReviewItem]] = []
    with Session(engine) as session:
        for cid in ids:
            item = session.get(ReviewItem, cid)
            roles[cid] = _role_for_text(item.original_comment)
            result = _generation_for(item, client=client, live_drafts=live_drafts)
            pending_generation.append(
                (result, item.primary_intent, item.topic_subtype, item.sentiment, item.ai_confidence, item.needs_review)
            )
    for result, intent, topic, sentiment, confidence, needs_review in pending_generation:
        svc.apply_generation_result(
            result,
            intent=intent,
            topic=topic,
            sentiment=sentiment,
            confidence=confidence,
            needs_review=needs_review,
        )

    reviewer = demo_microsoft_identity()
    by_role = {role: cid for cid, role in roles.items()}
    actions: list[dict[str, Any]] = []

    def snapshot(label: str) -> dict[str, Any]:
        counts = svc.workflow_counts()
        analytics = svc.operational_analytics(range_key="week")
        pending_ids = [i.comment_id for i in svc.list_review_queue(limit=50).items]
        return {
            "label": label,
            "pending": counts["pending"],
            "approved": counts["approved"],
            "no_reply": counts["no_reply"],
            "outcomes": analytics["outcomes"],
            "pending_ids": pending_ids,
        }

    thanks_id = by_role["thanks"]
    before = snapshot("start")
    claimed = svc.claim_review(thanks_id, reviewer)
    suggestion = claimed.suggested_response
    assert suggestion
    sent = svc.send_as_is(thanks_id, reviewer, expected_version=claimed.row_version)
    _assert_not_sent(sent)
    assert sent.reviewer_action == "SEND_AS_IS"
    assert sent.final_response == suggestion
    assert sent.suggested_response == suggestion
    assert sent.outbound_status == "PENDING"
    after_send = snapshot("after_SEND_AS_IS")
    assert thanks_id not in after_send["pending_ids"]
    assert after_send["outcomes"]["SEND_AS_IS"] == before["outcomes"]["SEND_AS_IS"] + 1
    actions.append(
        {
            "action": "SEND_AS_IS",
            "comment_id": thanks_id,
            "suggested_response": suggestion,
            "final_response": sent.final_response,
            "outbound_status": sent.outbound_status,
            "review_status": sent.review_status,
        }
    )

    complaint_id = by_role["complaint"]
    claimed = svc.claim_review(complaint_id, reviewer)
    original = claimed.suggested_response
    assert original
    edited = original.rstrip() + _EDIT_SUFFIX
    edited_out = svc.edit_and_send(complaint_id, reviewer, edited, expected_version=claimed.row_version)
    _assert_not_sent(edited_out)
    assert edited_out.reviewer_action == "EDIT_AND_SEND"
    assert edited_out.suggested_response == original
    assert edited_out.final_response == edited
    assert edited_out.outbound_status == "PENDING"
    after_edit = snapshot("after_EDIT_AND_SEND")
    assert complaint_id not in after_edit["pending_ids"]
    assert after_edit["outcomes"]["EDIT_AND_SEND"] == after_send["outcomes"]["EDIT_AND_SEND"] + 1
    actions.append(
        {
            "action": "EDIT_AND_SEND",
            "comment_id": complaint_id,
            "suggested_response": original,
            "final_response": edited_out.final_response,
            "outbound_status": edited_out.outbound_status,
            "review_status": edited_out.review_status,
        }
    )

    incomplete_id = by_role["incomplete"]
    claimed = svc.claim_review(incomplete_id, reviewer)
    assert not (claimed.suggested_response or "").strip()
    manual = svc.edit_and_send(
        incomplete_id, reviewer, _MANUAL_REPLY, expected_version=claimed.row_version
    )
    _assert_not_sent(manual)
    assert manual.reviewer_action == "EDIT_AND_SEND"
    assert not (manual.suggested_response or "").strip()
    assert manual.final_response == _MANUAL_REPLY
    assert manual.outbound_status == "PENDING"
    after_manual = snapshot("after_MANUAL_EDIT_AND_SEND")
    assert incomplete_id not in after_manual["pending_ids"]
    assert after_manual["outcomes"]["EDIT_AND_SEND"] == after_edit["outcomes"]["EDIT_AND_SEND"] + 1
    actions.append(
        {
            "action": "EDIT_AND_SEND_MANUAL",
            "comment_id": incomplete_id,
            "suggested_response": manual.suggested_response,
            "final_response": manual.final_response,
            "outbound_status": manual.outbound_status,
            "review_status": manual.review_status,
        }
    )

    spam_id = by_role["spam"]
    detail = svc.get_review_item(spam_id)
    closed = svc.mark_no_response_needed(
        spam_id, reviewer, reason="محتوى غير ذي صلة", expected_version=detail.row_version
    )
    _assert_not_sent(closed)
    assert closed.reviewer_action == "NO_RESPONSE_NEEDED"
    assert closed.final_response in (None, "")
    assert closed.outbound_status == "NOT_APPLICABLE"
    after_none = snapshot("after_NO_RESPONSE_NEEDED")
    assert spam_id not in after_none["pending_ids"]
    assert after_none["outcomes"]["NO_RESPONSE_NEEDED"] == after_manual["outcomes"]["NO_RESPONSE_NEEDED"] + 1
    actions.append(
        {
            "action": "NO_RESPONSE_NEEDED",
            "comment_id": spam_id,
            "suggested_response": closed.suggested_response,
            "final_response": closed.final_response,
            "outbound_status": closed.outbound_status,
            "review_status": closed.review_status,
        }
    )

    with Session(engine) as session:
        sent_n = session.query(ReviewItem).filter(ReviewItem.review_status == "SENT").count()
        outbound_sent = session.query(ReviewItem).filter(ReviewItem.outbound_status == "SENT").count()
    assert sent_n == 0
    assert outbound_sent == 0

    export_reviewed_workbook(engine, export_path, ids)
    with Session(engine) as session:
        thanks_row = session.get(ReviewItem, thanks_id)
        prompt_version = thanks_row.generation_prompt_version or settings.RESPONSE_GENERATION_PROMPT_VERSION
        model_name = thanks_row.generation_model_name or settings.RESPONSE_GENERATION_MODEL_NAME
    return {
        "db_path": str(db_path),
        "export_path": str(export_path),
        "source_workbook": str(source_workbook),
        "comment_ids": ids,
        "roles": roles,
        "import": {k: imported[k] for k in imported if k != "comment_ids"},
        "classified": classified,
        "prompt_version": prompt_version,
        "model_name": model_name,
        "used_live_ui_drafts": THANKS_TEXT in live_drafts,
        "actions": actions,
        "final_counts": after_none,
        "external_send": False,
    }
