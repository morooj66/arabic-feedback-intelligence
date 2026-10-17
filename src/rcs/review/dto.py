"""Reviewer-facing DTOs. Internal ML/prompt fields stay off the default page contract."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from rcs.classification.redact import redact_phone_numbers
from rcs.review.models import ReviewItem


@dataclass(frozen=True)
class ReviewerQueueItem:
    comment_id: str
    review_status: str
    generation_routing_decision: Optional[str]
    suggested_response: Optional[str]
    comment_preview: Optional[str]
    primary_intent: Optional[str]
    topic_subtype: Optional[str]
    sentiment: Optional[str]
    review_priority: Optional[int]
    priority_reason: Optional[str]
    claimed_by: Optional[str]
    comment_created_at: Optional[datetime]
    ingested_at: Optional[datetime]
    # Additive (UI reconciliation milestone) — lets an ADMIN-facing list
    # (Completed Reviews) show the final decision/reviewer/date without
    # a separate per-row detail fetch. Always present (None when not yet
    # decided); a REVIEWER's own pending queue simply has these as None.
    reviewer_action: Optional[str]
    reviewer_display_name: Optional[str]
    reviewed_at: Optional[datetime]


@dataclass(frozen=True)
class ReviewerItemDetail:
    comment_id: str
    source_reference: Optional[str]
    original_comment: Optional[str]
    comment_created_at: Optional[datetime]
    # Non-PII entry-identifying context (Phase — reviewer entry context):
    # helps a reviewer understand vague references like "هذا المدخل" /
    # "هذه الكلمة". Never visitor_name/visitor_email — those stay off
    # every reviewer-facing DTO, as above.
    original_title: Optional[str]
    entry_id: Optional[str]
    platform: Optional[str]
    original_type: Optional[str]
    primary_intent: Optional[str]
    topic_subtype: Optional[str]
    sentiment: Optional[str]
    confidence: Optional[float]
    needs_review: Optional[bool]
    classification_routing_decision: Optional[str]
    generation_routing_decision: Optional[str]
    generation_status: Optional[str]
    generation_reason_code: Optional[str]
    suggested_response: Optional[str]
    review_status: str
    reviewer_id: Optional[str]
    reviewer_display_name: Optional[str]
    reviewer_started_at: Optional[datetime]
    reviewed_at: Optional[datetime]
    reviewer_action: Optional[str]
    edited_response: Optional[str]
    final_response: Optional[str]
    no_response_reason: Optional[str]
    reviewer_notes: Optional[str]
    ready_to_send: bool
    outbound_status: Optional[str]
    outbound_error_code: Optional[str]
    review_priority: Optional[int]
    priority_reason: Optional[str]
    row_version: int
    allowed_actions: tuple[str, ...] = field(default_factory=tuple)
    # Contextual-signal conflicts already stored with the classification (rcs.classification.context_signals), as "dimension:code" strings. Empty when the layer
    # was off or found nothing. Display-only: never a confidence score and never an input to routing.
    context_conflicts: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AssignableItemSummary:
    """Phase 8 follow-up — minimal, PII-conscious shape for the admin
    assignment-browsing endpoint. Deliberately excludes visitor_name/
    visitor_email and the full comment text (a redacted preview only,
    same helper the reviewer queue already uses)."""

    comment_id: str
    comment_preview: Optional[str]
    source_file: str
    review_status: str
    routing_decision: Optional[str]
    reviewer_user_id: Optional[str]
    # Additive (UI reconciliation milestone) — the Assignment workspace
    # shows each row's classification so a multi-select classification
    # filter is legible against what's actually being assigned.
    primary_intent: Optional[str]
    topic_subtype: Optional[str]
    # Additive (visual reconciliation milestone) — the approved
    # reference table shows priority and date-added; both are already
    # persisted on every row, just not previously surfaced here.
    review_priority: Optional[int]
    ingested_at: Optional[datetime]


@dataclass(frozen=True)
class InternalReviewSnapshot:
    """Persistence/audit view including generation provider metadata."""

    comment_id: str
    generation_provider: Optional[str]
    generation_model_name: Optional[str]
    generation_prompt_version: Optional[str]
    generated_at: Optional[datetime]
    outbound_idempotency_key: Optional[str]
    outbound_external_id: Optional[str]
    outbound_error_message: Optional[str]


def _preview(text: str | None, limit: int = 120) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"


def _reviewer_facing_text(text: str | None) -> str | None:
    """Redacted view of stored comment text for reviewer-facing DTOs only.

    The persisted ReviewItem.original_comment is never modified — this
    redacts a copy at read time, the same rcs.classification.redact logic
    already used for outbound Gemini prompts (never a second, divergent
    implementation). Export/audit paths read ReviewItem columns directly
    and are unaffected.
    """
    return redact_phone_numbers(text)


def allowed_actions_for(status: str, has_suggestion: bool) -> tuple[str, ...]:
    if status in ("DRAFT_READY", "IN_REVIEW"):
        if has_suggestion:
            return ("SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED")
        return ("EDIT_AND_SEND", "NO_RESPONSE_NEEDED")
    if status == "REVIEW_ONLY":
        return ("EDIT_AND_SEND", "NO_RESPONSE_NEEDED")
    if status == "NO_RESPONSE_RECOMMENDED":
        # No SEND_AS_IS: a recommendation carries no draft to send. The
        # reviewer either confirms "no reply" or writes one (override).
        return ("EDIT_AND_SEND", "NO_RESPONSE_NEEDED")
    return ()


def to_queue_item(item: ReviewItem) -> ReviewerQueueItem:
    return ReviewerQueueItem(
        comment_id=item.comment_id,
        review_status=item.review_status,
        generation_routing_decision=item.generation_routing_decision,
        suggested_response=item.suggested_final_reply,
        comment_preview=_preview(_reviewer_facing_text(item.original_comment)),
        primary_intent=item.primary_intent,
        topic_subtype=item.topic_subtype,
        sentiment=item.sentiment,
        review_priority=item.review_priority,
        priority_reason=item.priority_reason,
        claimed_by=item.reviewer_id if item.review_status == "IN_REVIEW" else None,
        comment_created_at=item.comment_created_at,
        ingested_at=item.ingested_at,
        reviewer_action=item.reviewer_action,
        reviewer_display_name=item.reviewer_display_name,
        reviewed_at=item.reviewed_at,
    )


_CONFLICT_TOKEN = re.compile(r"^(intent|topic|sentiment):[a-z_]{3,80}$")


def parse_context_conflicts(reason: Optional[str]) -> tuple[str, ...]:
    """Extract the conflict codes the classification step appended to `ai_classification_reason` ("...; context_conflicts=a:b,c:d"). Only well-formed
    "dimension:code" tokens are returned, so nothing else stored in that text can reach a reviewer."""
    if not reason or "context_conflicts=" not in reason:
        return ()
    tail = reason.split("context_conflicts=", 1)[1].split(";", 1)[0]
    return tuple(tok for tok in (t.strip() for t in tail.split(",")) if _CONFLICT_TOKEN.match(tok))


def to_detail(item: ReviewItem) -> ReviewerItemDetail:
    suggestion = item.suggested_final_reply
    return ReviewerItemDetail(
        comment_id=item.comment_id,
        source_reference=item.source_reference or item.source_file,
        original_comment=_reviewer_facing_text(item.original_comment),
        comment_created_at=item.comment_created_at,
        original_title=item.original_title,
        entry_id=item.entry_id,
        platform=item.platform,
        original_type=item.original_type,
        primary_intent=item.primary_intent,
        topic_subtype=item.topic_subtype,
        sentiment=item.sentiment,
        confidence=item.ai_confidence,
        needs_review=item.needs_review,
        classification_routing_decision=item.routing_decision,
        generation_routing_decision=item.generation_routing_decision,
        generation_status=item.generation_status,
        generation_reason_code=item.generation_reason_code,
        suggested_response=suggestion,
        review_status=item.review_status,
        reviewer_id=item.reviewer_id,
        reviewer_display_name=item.reviewer_display_name,
        reviewer_started_at=item.reviewer_started_at,
        reviewed_at=item.reviewed_at,
        reviewer_action=item.reviewer_action,
        edited_response=item.reviewer_edit,
        final_response=item.final_reply,
        no_response_reason=item.no_response_reason,
        reviewer_notes=item.notes,
        ready_to_send=bool(item.ready_to_send),
        outbound_status=item.outbound_status,
        outbound_error_code=item.outbound_error_code,
        review_priority=item.review_priority,
        priority_reason=item.priority_reason,
        row_version=int(item.row_version or 1),
        allowed_actions=allowed_actions_for(item.review_status, bool(suggestion)),
        context_conflicts=parse_context_conflicts(getattr(item, "ai_classification_reason", None)),
    )


def to_assignable_item_summary(item: ReviewItem) -> AssignableItemSummary:
    return AssignableItemSummary(
        comment_id=item.comment_id,
        comment_preview=_preview(_reviewer_facing_text(item.original_comment)),
        source_file=item.source_file,
        review_status=item.review_status,
        routing_decision=item.generation_routing_decision,
        reviewer_user_id=item.reviewer_user_id,
        primary_intent=item.primary_intent,
        topic_subtype=item.topic_subtype,
        review_priority=item.review_priority,
        ingested_at=item.ingested_at,
    )


def to_internal(item: ReviewItem) -> InternalReviewSnapshot:
    return InternalReviewSnapshot(
        comment_id=item.comment_id,
        generation_provider=item.generation_provider,
        generation_model_name=item.generation_model_name,
        generation_prompt_version=item.generation_prompt_version,
        generated_at=item.generated_at,
        outbound_idempotency_key=item.outbound_idempotency_key,
        outbound_external_id=item.outbound_external_id,
        outbound_error_message=item.outbound_error_message,
    )


def _iso(value: Any) -> Any:
    from rcs.review.clock import iso_app

    if hasattr(value, "isoformat"):
        return iso_app(value)
    return value


def queue_item_to_dict(item: ReviewerQueueItem) -> dict[str, Any]:
    payload = item.__dict__.copy()
    payload["comment_created_at"] = _iso(item.comment_created_at)
    payload["ingested_at"] = _iso(item.ingested_at)
    payload["reviewed_at"] = _iso(item.reviewed_at)
    return payload


def assignable_item_to_dict(item: AssignableItemSummary) -> dict[str, Any]:
    payload = item.__dict__.copy()
    payload["ingested_at"] = _iso(item.ingested_at)
    return payload


def detail_to_dict(detail: ReviewerItemDetail) -> dict[str, Any]:
    payload = detail.__dict__.copy()
    payload["allowed_actions"] = list(detail.allowed_actions)
    payload["context_conflicts"] = list(detail.context_conflicts)
    for key in (
        "comment_created_at",
        "reviewer_started_at",
        "reviewed_at",
    ):
        payload[key] = _iso(payload[key])
    return payload
