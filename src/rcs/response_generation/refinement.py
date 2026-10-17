"""
Phase M5D — bounded, deterministic refinement eligibility and prompt
context. Lives in `rcs.response_generation` (not `rcs.response_intelligence`)
on purpose: `rcs.response_intelligence.orchestrator` already imports FROM
`rcs.response_generation`, so this stays a pure, dependency-free module
that package can use internally without creating an import cycle.

--- Audit this allowlist is based on (evidence, not a guess) --------------

`rcs.response_generation.validate.run_validation_gates` produces exactly
these reason codes, in this order, first-failure-wins:
  content_shape gate : "empty_output", "too_short", "excessive_length", "missing_arabic"
  internal_leak gate : "internal_leak"
  pii gate           : "pii_in_draft"
  boilerplate gate   : "generic_boilerplate"
  unsupported_claim  : "unsupported_operational_claim"

Separately, `rcs.response_generation.service.generate_suggested_response`
produces THREE more reason codes for failures that occur BEFORE any
draft is validated (no draft exists to refine at all):
  "gemini_client_missing", "gemini_api_error", "malformed_gemini_output"

Per M5D instruction ("use only reason codes actually proven in current
code... do not guess names"), `REFINEMENT_ELIGIBLE_REASON_CODES` below
contains ONLY validation-gate codes explicitly named as recoverable
categories in the M5D spec (unsupported operational claim, boilerplate,
content-shape). `internal_leak` and `pii_in_draft` are deliberately
EXCLUDED — the spec explicitly says leakage/PII should fail closed
rather than resend potentially unsafe content through another
generation call. The three pre-validation codes are excluded simply
because they are not in this set at all (no membership check needed);
see `is_refinement_eligible`'s docstring.

Ambiguous/Insufficient-Context Response Handling milestone:
`unsupported_visitor_intent_claim` (validate.py's
`_visitor_intent_attribution_gate`) added to the allowlist below for the
same reason as `unsupported_operational_claim` — it is a wrong-content
failure (the draft attributed an action to the visitor their comment
never established), not a safety violation, and rewriting once while
preserving the visitor's actual, textually-established point is exactly
what one bounded refinement pass is for.
"""

from __future__ import annotations

from dataclasses import dataclass

REFINEMENT_ELIGIBLE_REASON_CODES = frozenset(
    {
        "unsupported_operational_claim",
        "generic_boilerplate",
        "empty_output",
        "too_short",
        "excessive_length",
        "missing_arabic",
        # rg-v9 (2026-09): same category as generic_boilerplate -- a
        # wrong-register opening, not a safety violation. Rewriting the
        # opening while preserving the same substance is exactly what
        # one bounded refinement pass is for. See validate.py's
        # `_evaluator_voice_gate`.
        "evaluator_voice_opening",
        "unsupported_visitor_intent_claim",
        "wrong_pronoun_construction",
        # Reply-safety gates (rcs.response_generation.claim_gates): content failures that one bounded rewrite can repair. The rewrite is
        # validated by the SAME gates, so refinement can only turn a fail-closed "no draft" into a draft that passes every check; it cannot
        # relax anything. Leakage and personal data stay non-refinable (see NON_REFINABLE_VALIDATION_REASON_CODES).
        "unsupported_assertion",
        "unsupported_commitment",
        "unsupported_policy_claim",
        "unsupported_definition",
        "unsupported_correctness_claim",
        "unsupported_number",
        "unapproved_url",
    }
)

# One-sentence, code-specific rewrite guidance shown in the refinement block. Codes without an entry keep the original generic instruction
# (the block is byte-identical for them).
REFINEMENT_GUIDANCE = {
    "wrong_pronoun_construction": (
        "Thank the visitor for THEIR act with the second-person possessive (على مشاركتكم / على اقتراحكم / على ملاحظتكم); never use «على مشاركتنا» or another «-نا» possessive for something the visitor did."
    ),
    "unsupported_visitor_intent_claim": (
        "Do not state or imply that the visitor proposed, suggested, requested, asked or intended anything unless their comment says so. If they only shared a word or text, "
        "thank them for sharing it or acknowledge receipt without describing their purpose, do not ask what they meant, and do not describe, repeat as fact or endorse the "
        "meaning they supplied."
    ),
    "unsupported_assertion": (
        "With no official evidence available, the reply may only: thank the visitor, acknowledge receipt of their message, apologise briefly when they "
        "report a problem, ask them for specific details, or say that the information is not available to us. Remove any statement about the "
        "dictionary, the platform, a word, or the quality of the visitor's contribution."
    ),
    "unsupported_commitment": "Remove every promise, completed or planned action, hand-off to a team, and time reference. Do not say anything will be done.",
    "unsupported_policy_claim": "Remove any statement of a policy, procedure, criterion, duration, channel or resource that is not in the official evidence.",
    "unsupported_definition": "Remove every definition, meaning, etymology or grammatical statement about a word; none is in the official evidence.",
    "unsupported_correctness_claim": (
        "Do not say whether an entry, a correction or the visitor's claim is right or wrong, and do not apologise for or admit an error in an entry or a technical failure; "
        "if an apology is natural, apologise only for the visitor's inconvenience or experience."
    ),
    "unsupported_number": "Remove any number that appears neither in the visitor's comment nor in the official evidence.",
    "unapproved_url": "Remove every link and web address; none is approved for this reply.",
}

# Explicitly non-refinable, named here only for documentation/tests --
# NOT consulted by is_refinement_eligible (absence from the allowlist
# above is already sufficient; this tuple exists so a test can assert
# the two lists are disjoint and each code lands in exactly one).
NON_REFINABLE_VALIDATION_REASON_CODES = frozenset({"internal_leak", "pii_in_draft"})
NON_REFINABLE_PRE_VALIDATION_REASON_CODES = frozenset(
    {"gemini_client_missing", "gemini_api_error", "malformed_gemini_output"}
)


def is_refinement_eligible(generation_status: str, reason_code: str | None) -> bool:
    """True only for a genuine validation-gate failure (generation_status
    == "FAILED_REVIEW_ONLY") whose reason_code is in the explicit
    allowlist above. An unknown/unrecognized reason_code is never
    eligible -- membership must be explicit, not the absence of a
    denial."""
    if generation_status != "FAILED_REVIEW_ONLY":
        return False
    return reason_code in REFINEMENT_ELIGIBLE_REASON_CODES


@dataclass(frozen=True)
class RefinementContext:
    """Everything `build_generation_prompt`'s refinement block needs.
    No chain-of-thought, no request to "explain" or "think step by
    step" -- just the previous draft and the deterministic failure
    code, so the correction instruction can be as small and specific as
    possible."""

    previous_draft: str
    failure_reason_code: str

    def __post_init__(self) -> None:
        if not self.previous_draft or not self.previous_draft.strip():
            raise ValueError("previous_draft is required and cannot be blank")
        if not self.failure_reason_code or not self.failure_reason_code.strip():
            raise ValueError("failure_reason_code is required and cannot be blank")
