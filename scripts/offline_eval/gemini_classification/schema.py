"""Structured-output schema and STRICT validation. The API enforces the schema only on a best-effort basis, so every response is re-validated here."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import taxonomy as tx

REASON_MAX_CHARS = 200
_REQUIRED = ("intent", "topic", "sentiment", "ambiguous", "confidence")
_ALLOWED = _REQUIRED + ("reason",)

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(tx.INTENTS)},
        "topic": {"type": "string", "enum": list(tx.TOPICS)},
        "sentiment": {"type": "string", "enum": list(tx.SENTIMENTS)},
        "ambiguous": {"type": "boolean"},
        "confidence": {"type": "string", "enum": list(tx.CONFIDENCE_CATEGORIES)},
        "reason": {"type": "string", "description": "At most one short sentence naming the decisive cue. Never quote personal data."},
    },
    "required": list(_REQUIRED),
    "additionalProperties": False,
}


class SchemaError(ValueError):
    """The model output violated the contract. `code` is a stable, aggregatable failure reason (no text from the comment)."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Classification:
    intent: str
    topic: str
    sentiment: str
    ambiguous: bool
    confidence: str
    reason: str = ""


_IDENT = re.compile(r"(@|https?://|www\.|\d{7,})", re.I)


def parse_and_validate(raw: str | None) -> Classification:
    if not isinstance(raw, str) or not raw.strip():
        raise SchemaError("empty_response")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        raise SchemaError("invalid_json") from None
    if not isinstance(data, dict):
        raise SchemaError("not_an_object")
    keys = set(data)
    if not set(_REQUIRED) <= keys:
        raise SchemaError("missing_field")
    if keys - set(_ALLOWED):
        raise SchemaError("unexpected_field")
    for task, field in (("intent", "intent"), ("topic", "topic"), ("sentiment", "sentiment")):
        value = data[field]
        if not isinstance(value, str):
            raise SchemaError(f"{task}_not_string")
        if value not in tx.LABELS[task]:
            raise SchemaError(f"{task}_out_of_taxonomy")
    if not isinstance(data["ambiguous"], bool):
        raise SchemaError("ambiguous_not_boolean")
    if data["confidence"] not in tx.CONFIDENCE_CATEGORIES:
        raise SchemaError("confidence_out_of_range")
    reason = data.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise SchemaError("reason_not_string")
    reason = reason.strip()
    if len(reason) > REASON_MAX_CHARS:
        raise SchemaError("reason_too_long")
    if _IDENT.search(reason):
        raise SchemaError("reason_contains_identifier")
    return Classification(data["intent"], data["topic"], data["sentiment"], data["ambiguous"], data["confidence"], reason)
