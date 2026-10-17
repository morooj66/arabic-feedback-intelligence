"""Deterministic, auditable queue priority. Sentiment is never used alone."""

from __future__ import annotations

from rcs.review.models import ReviewItem


def compute_priority(
    *,
    primary_intent: str | None,
    topic_subtype: str | None,
    generation_routing: str | None,
    needs_review: bool | None,
    confidence: float | None,
    low_confidence_threshold: float = 0.45,
) -> tuple[int, str]:
    """Lower number = higher priority (1 urgent, 3 routine)."""
    if topic_subtype == "TECHNICAL_ISSUE" or primary_intent == "COMPLAINT":
        priority, reason = 1, "complaint_or_technical"
    elif primary_intent == "INQUIRY" or generation_routing == "REVIEW_ONLY":
        priority, reason = 2, "inquiry_or_review_only"
    else:
        priority, reason = 3, "routine"

    uncertain = bool(needs_review) or (
        confidence is not None and confidence < low_confidence_threshold
    )
    if uncertain and priority > 1:
        priority -= 1
        reason = f"{reason}+uncertain_classification"
    return priority, reason


def priority_for_item(item: ReviewItem) -> tuple[int, str]:
    return compute_priority(
        primary_intent=item.primary_intent or item.ai_primary_intent,
        topic_subtype=item.topic_subtype or item.ai_topic_subtype,
        generation_routing=item.generation_routing_decision,
        needs_review=item.needs_review if item.needs_review is not None else item.ai_needs_review,
        confidence=item.ai_confidence,
    )
