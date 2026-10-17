"""
Phase 4 orchestration: build prompt -> call Gemini (with bounded retry) ->
validate structured output -> compute routing (via the separate routing
engine) -> return a single outcome record ready for persistence.

Three concepts, kept deliberately distinct (per product requirement):
  1. AI classification  — what `classify_row` gets back from Gemini and
                           validates (primary_intent, topic_subtype,
                           subcategory, secondary_intent, ...).
  2. Routing decision    — computed by src/rcs/routing/engine.py, called
                           from here but implemented independently.
  3. Response generation — NOT done here, NOT part of this call at all.
                           Gemini's structured response is classification
                           ONLY (no immediate_reply_draft, no
                           suggested_final_reply — removed from the schema
                           per MIGRATION_PLAN.md). Immediate-ACK generation
                           and final-response drafting are separate,
                           deterministic/human-reviewed downstream concerns,
                           not part of this phase.

Never invents an intent or topic. Any failure — schema validation,
unexpected API error, or an API response that doesn't even parse as JSON —
produces a `success=False` outcome that deterministically routes to
REVIEW_ONLY and leaves ai_primary_intent unset, per MIGRATION_PLAN.md
Section 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Protocol

from pydantic import ValidationError

from .prompt import build_prompt
from .schema import JSON_SCHEMA, GeminiClassificationResponse
from ..routing.engine import determine_routing


class SupportsGenerate(Protocol):
    """Structural type for anything classify_row can call — the real
    GeminiClient, or a test double. No inheritance required."""

    def generate(self, prompt: str, response_schema: dict) -> str: ...


@dataclass(frozen=True)
class ClassificationOutcome:
    success: bool

    ai_primary_intent: Optional[str] = None
    ai_topic_subtype: Optional[str] = None
    ai_subcategory: Optional[str] = None
    ai_secondary_intent: Optional[str] = None
    ai_routing_decision: Optional[str] = None
    ai_classification_reason: Optional[str] = None
    ai_confidence: Optional[float] = None

    # Kept separate from ai_classification_reason on purpose: the latter is
    # the MODEL's own stated reasoning (faithful, unmodified, for audit);
    # this is the ROUTING ENGINE's reason code, used by the repository to
    # populate exclusion_reason without having to parse model prose.
    routing_reason_code: Optional[str] = None

    attempts: int = 0
    error: Optional[str] = None


def _call_with_retry(client: SupportsGenerate, prompt: str, max_retries: int) -> tuple[str | None, int, str | None]:
    """Returns (raw_text_or_None, attempts_made, last_error_or_None)."""
    last_error: str | None = None
    for attempt in range(1, max_retries + 2):  # max_retries=2 -> up to 3 attempts
        try:
            return client.generate(prompt, JSON_SCHEMA), attempt, None
        except Exception as exc:  # noqa: BLE001 - any transport/API failure is retried the same way
            last_error = f"{type(exc).__name__}: {exc}"
    return None, max_retries + 1, last_error


def classify_row(
    row: dict,
    client: SupportsGenerate,
    max_retries: int,
) -> ClassificationOutcome:
    prompt = build_prompt(row)

    raw_text, attempts, api_error = _call_with_retry(client, prompt, max_retries)

    if raw_text is None:
        return ClassificationOutcome(
            success=False,
            ai_routing_decision="REVIEW_ONLY",
            ai_classification_reason=f"api_error: {api_error}",
            attempts=attempts,
            error=api_error,
        )

    try:
        parsed = GeminiClassificationResponse.model_validate_json(raw_text)
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        return ClassificationOutcome(
            success=False,
            ai_routing_decision="REVIEW_ONLY",
            ai_classification_reason=f"classification_failed: {exc}",
            attempts=attempts,
            error=str(exc),
        )

    routing = determine_routing(
        primary_intent=parsed.primary_intent,
        topic_subtype=parsed.topic_subtype,
        subcategory=parsed.subcategory,
        secondary_intent=parsed.secondary_intent,
        is_content_meaningful=parsed.is_content_meaningful,
    )

    return ClassificationOutcome(
        success=True,
        ai_primary_intent=parsed.primary_intent,
        ai_topic_subtype=parsed.topic_subtype,
        ai_subcategory=parsed.subcategory,
        ai_secondary_intent=parsed.secondary_intent,
        ai_routing_decision=routing.routing_decision,
        ai_classification_reason=parsed.classification_reason,
        ai_confidence=parsed.confidence,
        routing_reason_code=routing.reason_code,
        attempts=attempts,
    )
