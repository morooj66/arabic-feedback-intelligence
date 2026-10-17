"""
Phase 4 — the Gemini structured-output contract.

This is the ONLY place the shape of a Gemini classification response is
defined. Two representations, kept in lockstep on purpose:

  - JSON_SCHEMA:      a plain dict, passed to Gemini as `response_schema`
                       so the model is constrained to produce conforming
                       JSON in the first place.
  - GeminiClassificationResponse: a Pydantic model used to VALIDATE
                       whatever text actually comes back, before any of it
                       is trusted. Gemini's schema constraint is a strong
                       hint to the model, not a guarantee about what a
                       given API response contains — so the app still
                       validates independently ("structured, validated
                       output rather than parsing free-form prose").

Both mirror MIGRATION_PLAN.md's hierarchical taxonomy exactly. If the
taxonomy ever changes, change config/settings.py first, then update both
of these together — do not let them drift apart.

Responsibility boundary (post-migration): this schema is classification
ONLY. Gemini no longer drafts replies as part of this call — no
immediate_reply_draft, no suggested_final_reply. extra="forbid" means a
stray reply-draft-shaped field in a Gemini response is rejected at
validation, not silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from config import settings

PrimaryIntent = Literal[
    "THANKS_PRAISE", "DUA_BLESSING", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM",
]
TopicSubtype = Literal[
    "WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION",
    "TECHNICAL_ISSUE", "WORD_OF_YEAR_NOMINATION", "OFF_TOPIC_PERSONAL", "GENERAL",
]
Subcategory = Literal["MeaningFault", "TechFault", "typo", "GrammerFault", "Other"]

assert set(PrimaryIntent.__args__) - {"DUA_BLESSING"} == set(settings.PRIMARY_INTENTS), (
    "schema.py PrimaryIntent literal has drifted from config.settings.PRIMARY_INTENTS"
)
assert set(TopicSubtype.__args__) == set(settings.TOPIC_SUBTYPES), (
    "schema.py TopicSubtype literal has drifted from config.settings.TOPIC_SUBTYPES"
)
assert set(Subcategory.__args__) == set(settings.SUBCATEGORIES), (
    "schema.py Subcategory literal has drifted from config.settings.SUBCATEGORIES"
)


class GeminiClassificationResponse(BaseModel):
    model_config = {"extra": "forbid"}

    primary_intent: PrimaryIntent
    topic_subtype: TopicSubtype
    subcategory: Optional[Subcategory] = None
    secondary_intent: Optional[PrimaryIntent] = None
    is_content_meaningful: bool
    confidence: float = Field(ge=0.0, le=1.0)
    classification_reason: str

    @model_validator(mode="after")
    def _secondary_must_differ_from_primary(self) -> "GeminiClassificationResponse":
        # Not rejected here — routing engine Section treats
        # secondary_intent == primary_intent as a deterministic ROUTING
        # ambiguity (-> REVIEW_ONLY), not a schema validation failure.
        # Validation only enforces shape/types; the routing engine
        # (src/rcs/routing/engine.py) enforces this rule so the two
        # concepts (classification vs. routing) stay separate, as required.
        return self


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_intent": {
            "type": "string",
            "enum": list(settings.PRIMARY_INTENTS),
            "description": "The visitor's communicative intent — what kind of message this is. Classify from the actual text, not from topic/metadata.",
        },
        "topic_subtype": {
            "type": "string",
            "enum": list(settings.TOPIC_SUBTYPES),
            "description": "What the message is about. Independent of primary_intent — do not infer one from the other.",
        },
        "subcategory": {
            "type": ["string", "null"],
            "enum": list(settings.SUBCATEGORIES) + [None],
            "description": "Only populate when topic_subtype is ENTRY_CORRECTION or TECHNICAL_ISSUE; null otherwise.",
        },
        "secondary_intent": {
            "type": ["string", "null"],
            "enum": list(settings.PRIMARY_INTENTS) + [None],
            "description": "Populate only for a genuine second, distinct intent. Must not equal primary_intent. Leave null for single-intent comments.",
        },
        "is_content_meaningful": {
            "type": "boolean",
            "description": "False if the text is empty, gibberish, or a placeholder even though it wasn't caught by string-based prefiltering.",
        },
        "confidence": {
            "type": "number", "minimum": 0, "maximum": 1,
            "description": "Self-reported, recorded for analytics only. Does not affect routing.",
        },
        "classification_reason": {
            "type": "string",
            "description": "One or two sentences, for a human reviewer, explaining the intent/topic choice.",
        },
    },
    "required": [
        "primary_intent", "topic_subtype", "is_content_meaningful", "confidence",
        "classification_reason",
    ],
}


# ---------------------------------------------------------------------------
# Unified classifier contract — LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase A.
#
# NOT a replacement for GeminiClassificationResponse above, which still
# validates exactly what Gemini's structured-output API returns. This is
# the common shape every classification TIER (rule-based, TF-IDF,
# embedding-similarity, a local transformer, or Gemini) is wrapped into
# AFTER its own tier-specific output is produced/validated, so
# hybrid_router.py (a later phase, not implemented yet) and downstream
# persistence never need to know which engine actually answered.
#
# `category` here is deliberately a single string, not the two-dimensional
# primary_intent/topic_subtype pair — early local tiers (rule-based,
# TF-IDF) are expected to target one dimension at a time as they're built
# and evaluated; how a ClassificationResult maps onto
# primary_intent/topic_subtype/ai_classification_source/ai_needs_review on
# a review_items row is a later-phase wiring decision, not decided here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationResult:
    category: Optional[str]  # None if this tier declines to answer
    confidence: float  # 0-1; meaning is tier-specific, comparable in RANK within a tier, not necessarily across tiers
    classification_source: str  # e.g. "RULE_BASED" | "TFIDF_LOGREG" | "TFIDF_SVM" | "TRANSFORMER_AR" | "GEMINI_FALLBACK"
    needs_review: bool  # this tier's own opinion on whether a human/next tier should look at it
