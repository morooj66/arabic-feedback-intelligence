"""
Phase B.2 — the AI-assisted-suggestion structured-output contract.

Deliberately a SEPARATE schema from src/rcs/classification/schema.py's
GeminiClassificationResponse (Phase 4's live-classification contract).
Phase B.2 suggests labels for `ai_assisted_train` rows of the GOLD DATASET
only — it never touches review_items, never computes a routing_decision,
and adds a `sentiment` field the Phase 4 schema doesn't have
(REVISED_ARCHITECTURE_PLAN.md v2 Section 3: sentiment is a separate signal
from primary_intent). Keeping the two schemas distinct keeps the "AI
classification (live, on real visitor comments) vs. AI-assisted gold-label
suggestion (on historical rows, always human-reviewed before becoming
gold)" boundary explicit in code, not just in the docs.

Two representations, kept in lockstep on purpose (same pattern as
schema.py): JSON_SCHEMA (sent to Gemini as response_schema) and
AISuggestionResponse (a Pydantic model that independently validates
whatever text actually comes back).

Business rules are enforced as HARD VALIDATION FAILURES here (unlike
Phase 4's GeminiClassificationResponse, which leaves secondary==primary to
the routing engine) because there is no downstream routing layer in Phase
B.2 to catch an inconsistent suggestion — an invalid combination must
simply not be written to ai_suggested_* at all.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from config import settings

# DUA_BLESSING removed 2026-08-24 (merged into THANKS_PRAISE, see config/settings.py)
PrimaryIntent = Literal[
    "THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM",
]
TopicSubtype = Literal[
    "WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION",
    "TECHNICAL_ISSUE", "WORD_OF_YEAR_NOMINATION", "OFF_TOPIC_PERSONAL", "GENERAL",
]
Subcategory = Literal["MeaningFault", "TechFault", "typo", "GrammerFault", "Other"]
Sentiment = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]

assert set(PrimaryIntent.__args__) == set(settings.PRIMARY_INTENTS)
assert set(TopicSubtype.__args__) == set(settings.TOPIC_SUBTYPES)
assert set(Subcategory.__args__) == set(settings.SUBCATEGORIES)
assert set(Sentiment.__args__) == set(settings.SENTIMENTS)

_TOPICS_REQUIRING_SUBCATEGORY = set(settings.TOPICS_REQUIRING_SUBCATEGORY)


class AISuggestionResponse(BaseModel):
    model_config = {"extra": "forbid"}

    primary_intent: PrimaryIntent
    topic_subtype: TopicSubtype
    secondary_intent: Optional[PrimaryIntent] = None
    subcategory: Optional[Subcategory] = None
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _business_rules(self) -> "AISuggestionResponse":
        if self.secondary_intent is not None and self.secondary_intent == self.primary_intent:
            raise ValueError(
                f"secondary_intent must differ from primary_intent (both were {self.primary_intent!r})"
            )
        requires_subcat = self.topic_subtype in _TOPICS_REQUIRING_SUBCATEGORY
        if requires_subcat and self.subcategory is None:
            raise ValueError(
                f"subcategory is required when topic_subtype={self.topic_subtype!r}"
            )
        if not requires_subcat and self.subcategory is not None:
            raise ValueError(
                f"subcategory must be null when topic_subtype={self.topic_subtype!r} "
                f"(only ENTRY_CORRECTION/TECHNICAL_ISSUE take a subcategory)"
            )
        return self



# JSON_SCHEMA below targets the CURRENT google-genai SDK's structured-output
# path (client.interactions.create(response_format={..., "schema": ...}),
# see src/rcs/evaluation/genai_client.py), whose documented schema subset
# (ai.google.dev/gemini-api/docs/structured-output, checked live 2026-08)
# explicitly supports "type": ["string", "null"] for nullable fields and
# "minimum"/"maximum" for numbers — so this uses plain JSON Schema, no
# workarounds needed. (An earlier version of this file worked around the
# OLD, now-abandoned `google-generativeai` SDK's stricter/older Schema type,
# which rejected both of those — that workaround no longer applies now that
# genai_client.py replaced that SDK entirely for Phase B.2.) Validity is
# still independently re-checked by AISuggestionResponse below regardless
# of what the API's own schema enforcement does.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_intent": {
            "type": "string",
            "enum": list(settings.PRIMARY_INTENTS),
            "description": "The visitor's communicative intent. Classify from the actual text, not from topic.",
        },
        "topic_subtype": {
            "type": "string",
            "enum": list(settings.TOPIC_SUBTYPES),
            "description": "What the message is about. Independent of primary_intent.",
        },
        "secondary_intent": {
            "type": ["string", "null"],
            "enum": list(settings.PRIMARY_INTENTS) + [None],
            "description": "Only for a genuine second, distinct intent. Must not equal primary_intent. Null otherwise.",
        },
        "subcategory": {
            "type": ["string", "null"],
            "enum": list(settings.SUBCATEGORIES) + [None],
            "description": "Only when topic_subtype is ENTRY_CORRECTION or TECHNICAL_ISSUE; null otherwise.",
        },
        "sentiment": {
            "type": "string",
            "enum": list(settings.SENTIMENTS),
            "description": "The comment's overall TONE — independent of primary_intent and topic_subtype.",
        },
        "confidence": {
            "type": "number", "minimum": 0, "maximum": 1,
            "description": "Self-reported confidence in this suggestion (0.0-1.0), for human-reviewer triage only.",
        },
    },
    "required": ["primary_intent", "topic_subtype", "sentiment", "confidence"],
}
