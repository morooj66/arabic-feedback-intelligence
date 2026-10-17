"""
Typed Response Strategy contract, VERSION 2 -- EVALUATION ONLY, not wired
into production. Does NOT overwrite or import strategy.py (V1); both
exist side by side so the V1 A/B artifacts remain reproducible.

--- Why V2 exists ---------------------------------------------------------

The first blinded A/B (8 cases, strategy.py / experiment.py) showed the
baseline (rg-v10) beating Strategy V1 5/8 to 3/8. Root cause identified
by inspection of V1's derive_strategy(): it lets `topic_subtype`
override `primary_intent` (`if topic_subtype == "WORD_SUBMISSION": goal
= RECEIVE_WORD_SUBMISSION`, checked BEFORE intent). That silently
recasts a THANKS_PRAISE, COMPLAINT, or INQUIRY comment as if it were
merely a word submission, because production commonly attaches
WORD_SUBMISSION as topic regardless of what the visitor is actually
doing.

V2's one structural change: intent decides the communicative goal
FIRST, unconditionally. Topic never overrides it -- topic only
refines HOW the goal is pursued (`response_mode`) and WHAT the reply
should point at (`key_focus`), never WHETHER a different goal applies.

--- Field-by-field rationale ----------------------------------------------

communicative_goal (closed, intent-first):
    ACKNOWLEDGE_PRAISE      <- THANKS_PRAISE / DUA_BLESSING, ANY topic
    ADDRESS_COMPLAINT       <- COMPLAINT, ANY topic
    ANSWER_INQUIRY          <- INQUIRY, ANY topic
    ENGAGE_SUGGESTION       <- SUGGESTION, ANY topic
    GENERAL_ACKNOWLEDGMENT  <- fallback (intents with no dedicated goal,
                               e.g. TECHNICAL_ISSUE); SPAM/ineligible
                               comments never reach this module at all,
                               since eligibility is decided upstream by
                               the unmodified production routing engine.
    There is deliberately NO "word submission" goal -- what V1 modeled
    as a goal is, in V2, WORD_SUBMISSION's contribution to
    `response_mode` and `key_focus` under whatever goal intent selected.

response_mode (closed, HOW the goal is pursued -- informed by topic and
    depth, but never able to change communicative_goal):
    ACKNOWLEDGMENT_ONLY   -- a direct, brief acknowledgment is enough
    ENGAGE_WITH_CONTENT   -- substantively engage the one central point
    ANSWER_IF_POSSIBLE    -- answer if Official Knowledge supports it,
                             otherwise acknowledge safely
    ACKNOWLEDGE_CONCERN   -- acknowledge the problem/correction raised
                             without promising resolution or action

response_depth: BRIEF / SUBSTANTIVE -- same deterministic character-
    length threshold V1 used (kept, since it was never the flawed part).

factual_answer_allowed: True only when Official Knowledge grounding is
    actually available for this case (identical rule to V1) -- Approved
    Responses and Identifying Context are NEVER a factual authority.

acknowledge_claim_without_confirming: True whenever the reply would
    otherwise be tempted to confirm/validate a visitor's linguistic or
    factual claim (ENGAGE_SUGGESTION, ANSWER_INQUIRY) and no Official
    Knowledge backs it. This is the field that most directly encodes
    the Hard Trust Rules from the task brief -- it is a boolean gate,
    never a place to store reasoning about WHY.

key_focus: a single bounded string -- either a literal, short quoted
    term extracted from the comment (the dataset's own «...» quoting
    convention) or, when no such term exists, ONE closed fallback label
    describing the topic category. Never free-text summarization, never
    a generated phrase -- this is deterministic extraction or a fixed
    vocabulary, nothing else.

evidence_requirement: closed enum stating the evidentiary standard for
    any factual content in the reply. Exists to make the Hard Trust
    Rule renderable as one explicit, unambiguous line in the prompt
    block rather than scattered prose.

No field stores raw model reasoning, chain-of-thought, or generated
reply text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COMMUNICATIVE_GOALS_V2 = (
    "ACKNOWLEDGE_PRAISE",
    "ADDRESS_COMPLAINT",
    "ANSWER_INQUIRY",
    "ENGAGE_SUGGESTION",
    "GENERAL_ACKNOWLEDGMENT",
)

RESPONSE_MODES_V2 = (
    "ACKNOWLEDGMENT_ONLY",
    "ENGAGE_WITH_CONTENT",
    "ANSWER_IF_POSSIBLE",
    "ACKNOWLEDGE_CONCERN",
)

RESPONSE_DEPTHS_V2 = ("BRIEF", "SUBSTANTIVE")

EVIDENCE_REQUIREMENTS_V2 = ("OFFICIAL_KNOWLEDGE_ONLY", "NONE_AVAILABLE_ACKNOWLEDGE_ONLY")

# Same threshold V1 used -- not tuned per case, not part of what broke V1.
_SUBSTANTIVE_CHAR_THRESHOLD = 400

_QUOTED_TERM_RE = re.compile(r"«([^»]{1,60})»")

_KEY_FOCUS_FALLBACK_BY_TOPIC = {
    "WORD_SUBMISSION": "SUBMITTED_WORD",
    "ENTRY_CORRECTION": "CORRECTION_TARGET",
    "DICTIONARY_SUGGESTION": "GENERAL_SUGGESTION",
}
_KEY_FOCUS_DEFAULT_FALLBACK = "GENERAL_COMMENT"


@dataclass(frozen=True)
class ResponseStrategyV2:
    """Planning metadata only -- NEVER contains generated text, a
    template, or reasoning. See render_strategy_v2_block for how this is
    turned into a short, labeled prompt section."""

    communicative_goal: str
    response_mode: str
    response_depth: str
    factual_answer_allowed: bool
    acknowledge_claim_without_confirming: bool
    key_focus: str
    evidence_requirement: str

    def __post_init__(self) -> None:
        if self.communicative_goal not in COMMUNICATIVE_GOALS_V2:
            raise ValueError(f"unknown communicative_goal: {self.communicative_goal!r}")
        if self.response_mode not in RESPONSE_MODES_V2:
            raise ValueError(f"unknown response_mode: {self.response_mode!r}")
        if self.response_depth not in RESPONSE_DEPTHS_V2:
            raise ValueError(f"unknown response_depth: {self.response_depth!r}")
        if self.evidence_requirement not in EVIDENCE_REQUIREMENTS_V2:
            raise ValueError(f"unknown evidence_requirement: {self.evidence_requirement!r}")


def _extract_key_focus(comment_text: str, topic_subtype: str | None) -> str:
    """Deterministic, bounded, never hallucinated: prefer a literal
    quoted term (this dataset's own convention for a proposed/discussed
    term); otherwise fall back to one of a small closed set of category
    labels derived from topic_subtype -- never free-text analysis."""
    quoted = _QUOTED_TERM_RE.findall(comment_text)
    if quoted:
        return quoted[0]
    return _KEY_FOCUS_FALLBACK_BY_TOPIC.get(topic_subtype or "", _KEY_FOCUS_DEFAULT_FALLBACK)


def derive_strategy_v2(
    *,
    comment_text: str,
    primary_intent: str | None,
    topic_subtype: str | None,
    official_grounding_available: bool,
) -> ResponseStrategyV2:
    """Pure, deterministic derivation from signals the pipeline already
    computes today -- no additional Gemini/LLM call, no randomness.
    `primary_intent` decides `communicative_goal` UNCONDITIONALLY;
    `topic_subtype` never overrides it (this is the fix for V1's known
    issue). Topic only shapes `response_mode` (via `response_depth`) and
    `key_focus`."""
    if primary_intent in ("THANKS_PRAISE", "DUA_BLESSING"):
        goal = "ACKNOWLEDGE_PRAISE"
    elif primary_intent == "COMPLAINT":
        goal = "ADDRESS_COMPLAINT"
    elif primary_intent == "INQUIRY":
        goal = "ANSWER_INQUIRY"
    elif primary_intent == "SUGGESTION":
        goal = "ENGAGE_SUGGESTION"
    else:
        goal = "GENERAL_ACKNOWLEDGMENT"

    depth = "SUBSTANTIVE" if len(comment_text) >= _SUBSTANTIVE_CHAR_THRESHOLD else "BRIEF"

    if goal == "ACKNOWLEDGE_PRAISE":
        mode = "ACKNOWLEDGMENT_ONLY"
    elif goal == "ADDRESS_COMPLAINT":
        mode = "ACKNOWLEDGE_CONCERN"
    elif goal == "ANSWER_INQUIRY":
        mode = "ANSWER_IF_POSSIBLE"
    elif goal == "ENGAGE_SUGGESTION":
        mode = "ENGAGE_WITH_CONTENT" if depth == "SUBSTANTIVE" else "ACKNOWLEDGMENT_ONLY"
    else:
        mode = "ACKNOWLEDGMENT_ONLY"

    factual_answer_allowed = bool(official_grounding_available)
    acknowledge_claim_without_confirming = goal in ("ENGAGE_SUGGESTION", "ANSWER_INQUIRY") and not factual_answer_allowed
    evidence_requirement = "OFFICIAL_KNOWLEDGE_ONLY" if factual_answer_allowed else "NONE_AVAILABLE_ACKNOWLEDGE_ONLY"

    return ResponseStrategyV2(
        communicative_goal=goal,
        response_mode=mode,
        response_depth=depth,
        factual_answer_allowed=factual_answer_allowed,
        acknowledge_claim_without_confirming=acknowledge_claim_without_confirming,
        key_focus=_extract_key_focus(comment_text, topic_subtype),
        evidence_requirement=evidence_requirement,
    )


def render_strategy_v2_block(strategy: ResponseStrategyV2) -> str:
    """Renders the V2 strategy as ONE additional, clearly labeled prompt
    section -- describing what the reply needs to accomplish and which
    institutional-employee pitfalls to avoid, never providing wording.
    Never includes chain-of-thought. Appended via safe string-insert into
    the SAME unmodified rg-v10 prompt -- never a rewrite of prompt.py."""
    lines = [
        "--- RESPONSE STRATEGY V2 (internal planning aid; NOT visible to the visitor; "
        "never mention or explain this section) ---",
        f"communicative_goal: {strategy.communicative_goal} "
        "(this reflects what the visitor is trying to accomplish -- respond to THIS, "
        "not to the topic category alone)",
        f"key_focus: {strategy.key_focus} (the one central point to react to directly)",
        f"response_mode: {strategy.response_mode}",
        f"response_depth: {strategy.response_depth} "
        "(BRIEF = one focused, proportional reaction, nothing more; "
        "SUBSTANTIVE = you may add one supporting sentence engaging the reasoning, "
        "but never a summary of the whole comment)",
        f"evidence_requirement: {strategy.evidence_requirement} "
        "(only OFFICIAL KNOWLEDGE above may ever satisfy this -- never your own knowledge, "
        "never the visitor's own claim, never Approved Responses style memory, "
        "never Identifying Context)",
    ]
    if strategy.acknowledge_claim_without_confirming:
        lines.append(
            "acknowledge_claim_without_confirming: true -- recognize the visitor's specific "
            "point/citation/correction as received and considered, but do NOT confirm it is "
            "correct, accurate, historically attested, accepted, or that it will be "
            "implemented/added/updated. Acknowledging is not the same as validating."
        )
    lines.append(
        "Write as a skilled institutional employee would, not an evaluator. Specifically avoid: "
        "a topic-preamble + restatement + generic-thanks template; phrases like "
        "\"إن التمييز الذي أوضحتموه...\" or \"هذا الطرح يبين...\"; summarizing the whole comment "
        "back to the visitor; exaggerated praise; and any promise that a suggestion will be "
        "added, accepted, or implemented."
    )
    lines.append(
        "This strategy describes WHAT the reply needs to accomplish. It is not a template, "
        "not wording, and not a required sentence structure."
    )
    lines.append("--- END RESPONSE STRATEGY V2 ---")
    return "\n".join(lines)
