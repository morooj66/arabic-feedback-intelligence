"""
Deterministic routing engine — MIGRATION_PLAN.md Section 4 (hierarchical
taxonomy: PRIMARY_INTENT x TOPIC_SUBTYPE).

Deliberately a separate module from src/rcs/classification/, called by it
but never the other way around. This is the architectural boundary
requested for Phase 4: "AI classification" (what did the visitor write)
and "routing decision" (what should the system do about it) are two
distinct, independently-testable steps. Gemini never decides routing —
this module takes Gemini's *already-validated* structured fields as plain
Python values and applies fixed rules over them. No network calls, no
model access, nothing probabilistic.

`response_generation` (Phase 6 — what to actually SAY to the visitor) is
a third, later concept and is not touched here at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import settings


@dataclass(frozen=True)
class RoutingResult:
    routing_decision: str
    reason_code: Optional[str]  # None for the "ordinary" lookup path


def determine_routing(
    primary_intent: str,
    topic_subtype: Optional[str],
    subcategory: Optional[str],
    secondary_intent: Optional[str],
    is_content_meaningful: bool,
) -> RoutingResult:
    """
    Pure function over already-validated classification fields.
    Mirrors MIGRATION_PLAN.md Section 4, in order:

      Step 1 — safety override: meaningless content that wasn't already
               going to EXCLUDE anyway (SPAM intent, or OFF_TOPIC_PERSONAL
               topic) is excluded outright.
      Step 2 — ambiguity fallbacks: routes to REVIEW_ONLY rather than
               guessing when the model's own output is self-inconsistent.
      Step 3 — primary_intent lookup table (config.settings.INTENT_ROUTING),
               then topic-based overrides (config.settings.
               TOPIC_ROUTING_OVERRIDES) applied on top, regardless of intent.
    """
    # --- Step 1: safety override -----------------------------------------
    if not is_content_meaningful and primary_intent != "SPAM" and topic_subtype != "OFF_TOPIC_PERSONAL":
        return RoutingResult("EXCLUDE", "model_detected_meaningless")

    # --- Step 2: ambiguity fallbacks --------------------------------------
    if secondary_intent is not None and secondary_intent == primary_intent:
        return RoutingResult("REVIEW_ONLY", "invalid_duplicate_intent")

    if topic_subtype in settings.TOPICS_REQUIRING_SUBCATEGORY and not subcategory:
        return RoutingResult("REVIEW_ONLY", "missing_required_subcategory")

    # --- Step 3: intent lookup, then topic override -----------------------
    effective_decision = settings.INTENT_ROUTING.get(primary_intent)
    if effective_decision is None:
        # Defensive-only: with Pydantic's Literal-constrained schema this
        # should be unreachable in practice, but MIGRATION_PLAN.md explicitly
        # calls for this fallback in case an out-of-enum value ever reaches
        # this layer.
        return RoutingResult("REVIEW_ONLY", "unsupported_output")

    # Topic overrides apply AFTER the intent lookup, regardless of what the
    # intent produced — WORD_OF_YEAR_NOMINATION always forces REVIEW_ONLY,
    # OFF_TOPIC_PERSONAL always forces EXCLUDE (decided, MIGRATION_PLAN.md
    # Open Question B). Every other topic leaves the intent decision as-is.
    topic_override = settings.TOPIC_ROUTING_OVERRIDES.get(topic_subtype)
    if topic_override is not None:
        effective_decision = topic_override

    if effective_decision == "EXCLUDE":
        # Always a concrete, non-None reason code so exclusion_reason is
        # always populated whenever routing_decision == EXCLUDE, regardless
        # of which path produced it (intent-driven vs. topic-override-driven).
        if topic_override == "EXCLUDE":
            reason = f"ai_topic_{topic_subtype.lower()}"
        else:
            reason = f"ai_intent_{primary_intent.lower()}"
        return RoutingResult("EXCLUDE", reason)

    return RoutingResult(effective_decision, None)
