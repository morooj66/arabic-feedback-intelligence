"""
Typed Response Strategy contract -- EVALUATION ONLY, not wired into
production. Describes WHAT a reply needs to accomplish (a structured
planning aid for the SAME single Response Agent/Gemini call), never HOW
to phrase it and never a canned template. No chain-of-thought: every
field is a short, closed-vocabulary value or a short extracted phrase --
never free-text reasoning, never the model's "thinking".

--- Why this exact set of goals ------------------------------------------

Chosen from what the EXISTING classification taxonomy already
distinguishes (config.settings.PRIMARY_INTENTS/TOPIC_SUBTYPES) plus one
fallback -- not invented independently:
  ACKNOWLEDGE_PRAISE       <- THANKS_PRAISE / DUA_BLESSING
  ADDRESS_COMPLAINT        <- COMPLAINT
  ANSWER_QUESTION          <- INQUIRY
  RECEIVE_WORD_SUBMISSION  <- topic_subtype == WORD_SUBMISSION
  ACKNOWLEDGE_SUGGESTION   <- SUGGESTION (any other topic)
  GENERAL_ACKNOWLEDGMENT   <- fallback for anything else (e.g. TECHNICAL_ISSUE)
REQUEST_CLARIFICATION (suggested as an example in the task) was
considered and DROPPED: no existing signal (intent/topic/is_content_
meaningful) deterministically indicates "this needs clarification" --
adding it would require guessing, which this module's derivation
function must not do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COMMUNICATIVE_GOALS = (
    "ACKNOWLEDGE_PRAISE",
    "ADDRESS_COMPLAINT",
    "ANSWER_QUESTION",
    "RECEIVE_WORD_SUBMISSION",
    "ACKNOWLEDGE_SUGGESTION",
    "GENERAL_ACKNOWLEDGMENT",
)

RESPONSE_DEPTHS = ("BRIEF", "SUBSTANTIVE")

# A single, documented length threshold -- not tuned per case. Comments
# under this length get BRIEF; longer ones (e.g. the dense linguistic
# essays in comments 2/3/4/5) get SUBSTANTIVE.
_SUBSTANTIVE_CHAR_THRESHOLD = 400

_QUOTED_TERM_RE = re.compile(r"«([^»]{1,60})»")
_MAX_QUOTED_TERMS = 2


@dataclass(frozen=True)
class ResponseStrategy:
    """Planning metadata only -- NEVER contains generated text, a
    template, or reasoning. See render_strategy_block for how this is
    turned into a short, labeled prompt section."""

    communicative_goal: str
    quoted_terms: tuple[str, ...]
    response_depth: str
    factual_answer_allowed: bool
    acknowledgment_only: bool

    def __post_init__(self) -> None:
        if self.communicative_goal not in COMMUNICATIVE_GOALS:
            raise ValueError(f"unknown communicative_goal: {self.communicative_goal!r}")
        if self.response_depth not in RESPONSE_DEPTHS:
            raise ValueError(f"unknown response_depth: {self.response_depth!r}")


def _extract_quoted_terms(text: str) -> tuple[str, ...]:
    """Deterministic, no model call: Arabic guillemets «...» are this
    dataset's own convention for quoting a proposed/discussed term or
    correction (confirmed by direct inspection of real comments, e.g.
    «اعتبار», «التراكيب المتلازمة», «قروء»). A plain regex extraction of
    the first two such spans -- never free-text summarization."""
    return tuple(_QUOTED_TERM_RE.findall(text)[:_MAX_QUOTED_TERMS])


def derive_strategy(
    *,
    comment_text: str,
    primary_intent: str | None,
    topic_subtype: str | None,
    official_grounding_available: bool,
) -> ResponseStrategy:
    """Pure, deterministic derivation from signals the pipeline already
    computes today (classification + grounding availability) -- no
    additional Gemini/LLM call, no randomness."""
    if topic_subtype == "WORD_SUBMISSION":
        goal = "RECEIVE_WORD_SUBMISSION"
    elif primary_intent in ("THANKS_PRAISE", "DUA_BLESSING"):
        goal = "ACKNOWLEDGE_PRAISE"
    elif primary_intent == "COMPLAINT":
        goal = "ADDRESS_COMPLAINT"
    elif primary_intent == "INQUIRY":
        goal = "ANSWER_QUESTION"
    elif primary_intent == "SUGGESTION":
        goal = "ACKNOWLEDGE_SUGGESTION"
    else:
        goal = "GENERAL_ACKNOWLEDGMENT"

    depth = "SUBSTANTIVE" if len(comment_text) >= _SUBSTANTIVE_CHAR_THRESHOLD else "BRIEF"
    acknowledgment_only = goal == "ACKNOWLEDGE_PRAISE" or (goal == "RECEIVE_WORD_SUBMISSION" and depth == "BRIEF")

    return ResponseStrategy(
        communicative_goal=goal,
        quoted_terms=_extract_quoted_terms(comment_text),
        response_depth=depth,
        factual_answer_allowed=official_grounding_available,
        acknowledgment_only=acknowledgment_only,
    )


def render_strategy_block(strategy: ResponseStrategy) -> str:
    """Renders the strategy as ONE additional, clearly labeled prompt
    section -- describing what the reply needs to accomplish, never
    providing wording. Never includes chain-of-thought."""
    lines = [
        "--- RESPONSE STRATEGY (internal planning aid; NOT visible to the visitor; "
        "never mention or explain this section) ---",
        f"communicative_goal: {strategy.communicative_goal}",
    ]
    if strategy.quoted_terms:
        lines.append(f"central term(s) to react to directly: {', '.join(strategy.quoted_terms)}")
    lines.append(
        f"response_depth: {strategy.response_depth} "
        "(BRIEF = one focused reaction to the central point, nothing more; "
        "SUBSTANTIVE = you may add one supporting sentence engaging the reasoning, "
        "but still never a summary of the whole comment)"
    )
    lines.append(
        f"factual_answer_allowed: {strategy.factual_answer_allowed} "
        "(only OFFICIAL KNOWLEDGE above may ever make this true -- this field never grants "
        "permission to answer from your own knowledge)"
    )
    if strategy.acknowledgment_only:
        lines.append(
            "acknowledgment_only: true -- a direct, brief acknowledgment is fully sufficient; "
            "do not manufacture additional substance or analysis."
        )
    lines.append(
        "This strategy describes WHAT the reply needs to accomplish. It is not a template, "
        "not wording, and not a required sentence structure."
    )
    lines.append("--- END RESPONSE STRATEGY ---")
    return "\n".join(lines)
