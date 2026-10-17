"""
Phase B.2 orchestration: build prompt (with the fixed few-shot block) ->
call Gemini (bounded retry, same pattern as src/rcs/classification/
classifier.py) -> validate structured output against AISuggestionResponse
-> return a single outcome record.

No routing decision is computed here (Phase B.2 suggests gold LABELS, not
a live routing outcome). Any failure — API error, JSON parse failure, or a
schema/business-rule validation failure — produces success=False and the
caller must leave that row's ai_suggested_* columns blank; it must never
guess or partially fill a suggestion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Protocol

from pydantic import ValidationError

from .ai_suggestion_schema import JSON_SCHEMA, AISuggestionResponse
from .suggestion_prompt import build_suggestion_prompt


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, response_schema: dict) -> str: ...


@dataclass(frozen=True)
class SuggestionOutcome:
    success: bool

    primary_intent: Optional[str] = None
    topic_subtype: Optional[str] = None
    secondary_intent: Optional[str] = None
    subcategory: Optional[str] = None
    sentiment: Optional[str] = None
    confidence: Optional[float] = None

    attempts: int = 0
    error: Optional[str] = None


def _call_with_retry(client: SupportsGenerate, prompt: str, max_retries: int) -> tuple[str | None, int, str | None]:
    last_error: str | None = None
    for attempt in range(1, max_retries + 2):
        try:
            return client.generate(prompt, JSON_SCHEMA), attempt, None
        except Exception as exc:  # noqa: BLE001 - any transport/API failure is retried the same way
            last_error = f"{type(exc).__name__}: {exc}"
    return None, max_retries + 1, last_error


def suggest_row(
    row: dict,
    few_shot_examples: list[dict],
    client: SupportsGenerate,
    max_retries: int,
) -> SuggestionOutcome:
    prompt = build_suggestion_prompt(row, few_shot_examples)

    raw_text, attempts, api_error = _call_with_retry(client, prompt, max_retries)

    if raw_text is None:
        return SuggestionOutcome(success=False, attempts=attempts, error=f"api_error: {api_error}")

    try:
        parsed = AISuggestionResponse.model_validate_json(raw_text)
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        return SuggestionOutcome(success=False, attempts=attempts, error=f"validation_failed: {exc}")

    return SuggestionOutcome(
        success=True,
        primary_intent=parsed.primary_intent,
        topic_subtype=parsed.topic_subtype,
        secondary_intent=parsed.secondary_intent,
        subcategory=parsed.subcategory,
        sentiment=parsed.sentiment,
        confidence=parsed.confidence,
        attempts=attempts,
    )
