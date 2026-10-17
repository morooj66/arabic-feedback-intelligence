"""Isolated demo SQLite seed for the reviewer UI. Does not touch gold datasets.

Suggested replies come from `generate_suggested_response` (current prompt
version). Live Gemini is opt-in for demo CLI; tests inject a scripted client.
Never writes canned rg-v4/rg-v5 placeholders when a generation client exists.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from config import settings
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest, build_live_gemini_client, generate_suggested_response
from rcs.review.identity import demo_microsoft_identity
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import ReviewWorkflowService

DEMO_DB = settings.DB_DIR / "reviewer_ui_demo.sqlite"

# Distinct test-double drafts (not the old shared rg-v5 canned line). Used only
# when live Gemini is not requested. Each GENERATE_DRAFT item gets its own text.
_TEST_PIPELINE_DRAFTS = {
    "8742": "سعدنا بثقتكم، وملاحظاتكم محل تقدير.",
    "8743": "نعتذر عمّا واجهتموه في البحث، ووصلت ملاحظتكم عن أداة البحث.",
    "8744": "شكرًا لاقتراح إضافة معنى «غرارة»، ووصلت ملاحظتكم.",
    "8745": "وصل استفساركم عن مواعيد إضافة الكلمات الجديدة.",
    "8746": "وصل سؤالكم عن النسخة الورقية من المعجم.",
    "9100": "بارك الله فيكم، ويسعد هذا الثناء.",
}

_STALE_CANNED = "شكرًا لتواصلكم معنا، نسعد بثقتكم بالمعجم."


class _ScriptedGenerationClient:
    """Deterministic GENERATE_DRAFT client for tests. Not a visitor-facing source."""

    def generate(self, prompt: str, response_schema: dict) -> str:
        for comment_id, text in _TEST_PIPELINE_DRAFTS.items():
            if comment_id in prompt or f"#{comment_id}" in prompt:
                return json.dumps({"suggested_response": text}, ensure_ascii=False)
        # Prompt includes the visitor comment, not the id — match by comment snippets.
        if "المعجم الرائع" in prompt:
            return json.dumps({"suggested_response": _TEST_PIPELINE_DRAFTS["8742"]}, ensure_ascii=False)
        if "البحث عن بعض الكلمات" in prompt:
            return json.dumps({"suggested_response": _TEST_PIPELINE_DRAFTS["8743"]}, ensure_ascii=False)
        if "غرارة" in prompt:
            return json.dumps({"suggested_response": _TEST_PIPELINE_DRAFTS["8744"]}, ensure_ascii=False)
        if "الكلمات الجديدة" in prompt:
            return json.dumps({"suggested_response": _TEST_PIPELINE_DRAFTS["8745"]}, ensure_ascii=False)
        if "النسخة الورقية" in prompt:
            return json.dumps({"suggested_response": _TEST_PIPELINE_DRAFTS["8746"]}, ensure_ascii=False)
        if "بارك الله" in prompt:
            return json.dumps({"suggested_response": _TEST_PIPELINE_DRAFTS["9100"]}, ensure_ascii=False)
        raise AssertionError("scripted demo client has no draft for this prompt")


def load_dotenv_if_present() -> None:
    """Load .env into os.environ and the settings read before startup.

    Public and side-effect-free to call more than once (idempotent — never
    overwrites an already-set os.environ key). Used by seed_demo_db() below
    AND, independently, by scripts/serve_reviewer_api.py's normal (non-seed)
    startup path — Gemini availability must not depend on demo seeding.
    """
    if os.environ.get("RCS_IGNORE_DOTENV") == "1":        # set by the test suite (and usable by any caller) so real credentials are never loaded
        return
    env_path = settings.BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    if os.environ.get("GEMINI_API_KEY") and not settings.GEMINI_API_KEY:
        settings.GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    # config.settings is imported before this loader by the server entrypoint.
    # Keep the experimental classification switches in sync for the same
    # reason GEMINI_API_KEY is synchronized above. Values are never logged.
    settings.CLASSIFICATION_MODE = os.environ.get(
        "RCS_CLASSIFICATION_MODE", settings.CLASSIFICATION_MODE
    ).strip().lower()
    settings.TOPIC_BOUNDARY_LAYER_ENABLED = os.environ.get("RCS_TOPIC_BOUNDARY_LAYER", "0").strip().lower() in ("1", "true", "yes")
    if os.environ.get("OPENROUTER_API_KEY") and not settings.OPENROUTER_API_KEY:
        settings.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    # Same re-sync, same reason: settings.DATABASE_URL is captured once at
    # config.settings import time (before this function ever runs), so a
    # DATABASE_URL defined only in .env — the natural way to configure a
    # future PostgreSQL URL without exporting it in the shell — would
    # otherwise silently never take effect.
    if os.environ.get("DATABASE_URL") and not settings.DATABASE_URL:
        settings.DATABASE_URL = os.environ["DATABASE_URL"]


def _want_live(live: bool | None) -> bool:
    if live is True:
        return True
    if live is False:
        return False
    return os.environ.get("RCS_DEMO_LIVE_GEMINI", "").strip().lower() in {"1", "true", "yes"}


def _demo_rows() -> list[dict]:
    now = dt.datetime(2026, 8, 25, 18, 0, 0)
    return [
        {
            "comment_id": "8742",
            "source_file": "demo-source",
            "source_row_index": 1,
            "source_reference": "المعجم / نموذج",
            "original_comment": "شكرًا لكم على هذا المعجم الرائع، استفدت كثيرًا من شروحه.",
            "ingested_at": now - dt.timedelta(minutes=15),
            "comment_created_at": now - dt.timedelta(minutes=15),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "THANKS_PRAISE",
            "topic_subtype": "GENERAL",
            "ai_primary_intent": "THANKS_PRAISE",
            "ai_topic_subtype": "GENERAL",
            "ai_confidence": 0.92,
            "sentiment": "POSITIVE",
            "row_version": 1,
        },
        {
            "comment_id": "8743",
            "source_file": "demo-source",
            "source_row_index": 2,
            "source_reference": "المعجم / نموذج",
            "original_comment": "الموقع لا يعمل عند البحث عن بعض الكلمات، أرجو المعالجة.",
            "ingested_at": now - dt.timedelta(hours=2),
            "comment_created_at": now - dt.timedelta(hours=2),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "COMPLAINT",
            "topic_subtype": "TECHNICAL_ISSUE",
            "ai_primary_intent": "COMPLAINT",
            "ai_topic_subtype": "TECHNICAL_ISSUE",
            "ai_confidence": 0.81,
            "sentiment": "NEGATIVE",
            "needs_review": True,
            "row_version": 1,
        },
        {
            "comment_id": "8744",
            "source_file": "demo-source",
            "source_row_index": 3,
            "source_reference": "المعجم / نموذج",
            "original_comment": "هل يمكن إضافة معنى كلمة غرارة في الإصدار القادم؟",
            "ingested_at": now - dt.timedelta(days=1),
            "comment_created_at": now - dt.timedelta(days=1),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "SUGGESTION",
            "topic_subtype": "WORD_OF_YEAR_NOMINATION",
            "ai_primary_intent": "SUGGESTION",
            "ai_topic_subtype": "WORD_OF_YEAR_NOMINATION",
            "ai_confidence": 0.7,
            "sentiment": "NEUTRAL",
            "row_version": 1,
        },
        {
            "comment_id": "8745",
            "source_file": "demo-source",
            "source_row_index": 4,
            "source_reference": "المعجم / نموذج",
            "original_comment": "متى يُحدَّث المعجم بالكلمات الجديدة؟",
            "ingested_at": now - dt.timedelta(days=4),
            "comment_created_at": now - dt.timedelta(days=4),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "INQUIRY",
            "topic_subtype": "GENERAL",
            "ai_primary_intent": "INQUIRY",
            "ai_topic_subtype": "GENERAL",
            "ai_confidence": 0.4,
            "needs_review": True,
            "sentiment": "NEUTRAL",
            "row_version": 1,
        },
        {
            "comment_id": "8746",
            "source_file": "demo-source",
            "source_row_index": 5,
            "source_reference": "المعجم / نموذج",
            "original_comment": "كيف أصل إلى النسخة الورقية؟",
            "ingested_at": now - dt.timedelta(weeks=1),
            "comment_created_at": now - dt.timedelta(weeks=1),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "INQUIRY",
            "topic_subtype": "GENERAL",
            "ai_primary_intent": "INQUIRY",
            "ai_topic_subtype": "GENERAL",
            "ai_confidence": 0.55,
            "sentiment": "NEUTRAL",
            "row_version": 1,
        },
        {
            "comment_id": "8801",
            "source_file": "demo-source",
            "source_row_index": 6,
            "source_reference": "المعجم / نموذج",
            "original_comment": "محتوى غير مكتمل يحتاج مراجعة بشرية قبل صياغة رد.",
            "ingested_at": now - dt.timedelta(hours=5),
            "comment_created_at": now - dt.timedelta(hours=5),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "INQUIRY",
            "topic_subtype": "GENERAL",
            "ai_primary_intent": "INQUIRY",
            "ai_confidence": 0.33,
            "needs_review": True,
            "row_version": 1,
        },
        {
            "comment_id": "9001",
            "source_file": "demo-source",
            "source_row_index": 7,
            "source_reference": "المعجم / نموذج",
            "original_comment": "إعلان تجاري غير ذي صلة بالمعجم.",
            "ingested_at": now - dt.timedelta(days=2),
            "comment_created_at": now - dt.timedelta(days=2),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "SPAM",
            "topic_subtype": "OFF_TOPIC_PERSONAL",
            "ai_primary_intent": "SPAM",
            "row_version": 1,
        },
        {
            "comment_id": "9100",
            "source_file": "demo-source",
            "source_row_index": 8,
            "source_reference": "المعجم / نموذج",
            "original_comment": "عمل مبارك، بارك الله فيكم.",
            "ingested_at": now - dt.timedelta(days=8),
            "comment_created_at": now - dt.timedelta(days=8),
            "review_status": "CLASSIFIED",
            "ready_to_send": False,
            "primary_intent": "THANKS_PRAISE",
            "topic_subtype": "GENERAL",
            "ai_primary_intent": "THANKS_PRAISE",
            "ai_confidence": 0.95,
            "sentiment": "POSITIVE",
            "row_version": 1,
        },
    ]


def _generation_request(row: dict) -> GenerationRequest:
    return GenerationRequest(
        comment_id=row["comment_id"],
        comment_text=row["original_comment"],
        primary_intent=row.get("primary_intent"),
        topic_subtype=row.get("topic_subtype"),
        sentiment=row.get("sentiment"),
        confidence=row.get("ai_confidence"),
        needs_review=bool(row.get("needs_review")),
        is_content_meaningful=True,
        insufficient_context=row["comment_id"] == "8801",
        trusted_grounding_available=False,
    )


def _review_only_skip(comment_id: str) -> GenerationResult:
    return GenerationResult(
        comment_id=comment_id,
        routing_decision="REVIEW_ONLY",
        generation_status="SKIPPED",
        suggested_response=None,
        review_required=True,
        reason_code="incomplete_comment_needs_human_draft",
        provider=None,
        model_name=None,
        prompt_version=settings.RESPONSE_GENERATION_PROMPT_VERSION,
        generated_at="2026-08-26T09:00:00+00:00",
    )


def seed_demo_db(
    db_path: Path | None = None,
    *,
    live: bool | None = None,
    generation_client=None,
) -> Path:
    path = db_path or DEMO_DB
    if path.exists():
        path.unlink()
    engine = get_engine(path)
    init_db(engine)
    rows = _demo_rows()
    upsert_review_items(engine, rows)
    reviewer = demo_microsoft_identity()
    svc = ReviewWorkflowService(engine)

    load_dotenv_if_present()
    client = generation_client
    used_live = False
    if client is None and _want_live(live):
        if not (settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")):
            raise RuntimeError("RCS_DEMO_LIVE_GEMINI set but GEMINI_API_KEY is missing")
        client = build_live_gemini_client()
        used_live = True
    if client is None:
        client = _ScriptedGenerationClient()

    refreshed: list[str] = []
    for row in rows:
        cid = row["comment_id"]
        if cid == "8801":
            result = _review_only_skip(cid)
        else:
            result = generate_suggested_response(_generation_request(row), client=client)
        if result.suggested_response == _STALE_CANNED:
            raise RuntimeError(f"refusing stale canned draft for {cid}")
        if result.routing_decision == "GENERATE_DRAFT" and result.suggested_response:
            refreshed.append(cid)
        svc.apply_generation_result(
            result,
            intent=row.get("primary_intent"),
            topic=row.get("topic_subtype"),
            sentiment=row.get("sentiment"),
            confidence=row.get("ai_confidence"),
            needs_review=row.get("needs_review"),
        )

    svc.claim_review("9100", reviewer)
    svc.send_as_is("9100", reviewer)
    d8745 = svc.get_review_item("8745")
    if d8745.suggested_response:
        svc.claim_review("8745", reviewer)
        svc.edit_and_send(
            "8745",
            reviewer,
            f"{d8745.suggested_response.rstrip()} نقدر حرصكم على متابعة الإصدارات.",
        )
    svc.mark_no_response_needed("9001", reviewer, reason="محتوى غير ذي صلة")
    seed_demo_db.last_used_live = used_live  # type: ignore[attr-defined]
    seed_demo_db.last_prompt_version = settings.RESPONSE_GENERATION_PROMPT_VERSION  # type: ignore[attr-defined]
    seed_demo_db.last_model = settings.RESPONSE_GENERATION_MODEL_NAME  # type: ignore[attr-defined]
    seed_demo_db.last_refreshed_comment_ids = refreshed  # type: ignore[attr-defined]
    return path
