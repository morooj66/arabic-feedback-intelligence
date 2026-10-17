"""
Response Strategy V3 -- EVALUATION ONLY, not wired into production.

V3 does NOT redesign the strategy contract. It reuses V2's deterministic
`derive_strategy_v2` / `render_strategy_v2_block` (imported, never
copied or reimplemented) and adds exactly one new thing: a deterministic
GATE deciding whether the strategy block is supplied to Gemini at all.

--- Why a gate ------------------------------------------------------------

The V2 error analysis (results/strategy_v2_error_analysis.md, 25-case
blinded A/B) found V2 helped on substantive content-engagement cases
(SUBSTANTIVE/ENGAGE_WITH_CONTENT: V2 won 4/6) but lost badly on brief,
low-content cases (ACKNOWLEDGMENT_ONLY: V2 won only 2/11) -- injecting a
strategy block onto a comment with little real content gave Gemini
instructions to follow with nothing to apply them to, and in one traced
case (row 112, "ابرايل الله") contributed to a misleading reply. The
analysis's smallest recommended gate:

    primary_intent == "SUGGESTION" AND response_depth == "SUBSTANTIVE"

V3 implements exactly that gate, nothing more. `response_depth` is
computed by the SAME deterministic character-length rule
`derive_strategy_v2` already uses (>= 400 chars => SUBSTANTIVE) -- V3
does not introduce a second threshold.

--- What "gate closed" means ----------------------------------------------

When the gate is closed, `strategy_enabled` is False and NO strategy
block is appended to the prompt at all -- generation input for the
"V3 variant" becomes byte-for-byte the same rg-v10 prompt the baseline
uses (see experiment_v3.py's `_generate_reply_v3`). This is deliberate:
"baseline-equivalent" means literally the unmodified baseline prompt,
not a strategy block with different content.

--- Known independent technical debt (NOT addressed by this task) --------

The "ابرايل الله" issue traced in the V2 error analysis originates in
`identifying_context` (production's `build_identifying_grounding`,
`src/rcs/response_generation/prompt.py`), not in the strategy contract
or the gate. It is recorded here as a known, separate technical-debt
item and is explicitly OUT OF SCOPE for V3. V3 does not touch
`identifying_context` construction or usage in any way, and continues
to preserve the existing rule: Official Knowledge is the only factual
authority; visitor text, identifying context, and Approved Responses
are never treated as factual authority by anything in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from strategy_v2 import ResponseStrategyV2, derive_strategy_v2, render_strategy_v2_block

GATE_INTENT = "SUGGESTION"
GATE_DEPTH = "SUBSTANTIVE"


@dataclass(frozen=True)
class StrategyV3Decision:
    """Whether the strategy block should be USED for this case, and the
    underlying V2 strategy -- always computed (for transparency/analysis,
    e.g. to record what response_depth a gate-negative case had), even
    when `strategy_enabled` is False and the block is therefore never
    rendered into any prompt. Never contains generated text."""

    strategy_enabled: bool
    strategy: ResponseStrategyV2


def gate_is_open(*, primary_intent: str | None, response_depth: str | None) -> bool:
    """Pure, deterministic, two-field gate. No other signal (topic,
    sentiment, confidence, key_focus) participates -- this is
    deliberately the SMALLEST gate the V2 error analysis could justify."""
    return primary_intent == GATE_INTENT and response_depth == GATE_DEPTH


def derive_strategy_v3(
    *,
    comment_text: str,
    primary_intent: str | None,
    topic_subtype: str | None,
    official_grounding_available: bool,
) -> StrategyV3Decision:
    """Computes the V2 strategy (reused, unchanged) purely to read its
    deterministic `response_depth`, then applies the gate. No extra
    Gemini/LLM call -- this is the same single deterministic computation
    V2 already does, with a gate layered on top."""
    strategy = derive_strategy_v2(
        comment_text=comment_text,
        primary_intent=primary_intent,
        topic_subtype=topic_subtype,
        official_grounding_available=official_grounding_available,
    )
    enabled = gate_is_open(primary_intent=primary_intent, response_depth=strategy.response_depth)
    return StrategyV3Decision(strategy_enabled=enabled, strategy=strategy)


def render_strategy_v3_block(decision: StrategyV3Decision) -> str | None:
    """Returns the SAME V2-rendered block (reused, not reimplemented)
    when the gate is open; None when closed -- callers must append
    nothing to the prompt in that case, keeping it baseline-equivalent."""
    if not decision.strategy_enabled:
        return None
    return render_strategy_v2_block(decision.strategy)
