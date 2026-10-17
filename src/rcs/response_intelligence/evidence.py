"""
Phase M5C — pure evidence assembly for the deterministic response agent
layer. No I/O, no retrieval, no Gemini, no database: this module only
combines retrieval results that were already computed elsewhere into
what the existing prompt-building/eligibility seam needs.

Phase M5D — clearer, unambiguous internal naming. `GroundingAssembly`
exposes two DISTINCT concepts:

  - `identifying_context`: the pre-existing `build_identifying_grounding`
    text (title/entry_id) — metadata about WHICH entry a comment
    concerns, NEVER a fact about that entry.
  - `official_grounding_text` / `official_grounding_available`: set
    ONLY from a real `OfficialKnowledgeRetrievalResult` at
    SUCCESS_WITH_RESULTS. This is the ONLY thing in this module that
    may ever represent factual authority.

Phase M5D.1 — `trusted_grounding_available` (the caller-facing
eligibility flag) is set by the orchestrator directly to
`official_grounding_available`, nothing else. Identifying context can
never flip it, alone or combined with Approved Response Memory.

Phase M5D.2 — removed the combined `prompt_text` field entirely.
Earlier, `assemble_grounding` concatenated `identifying_context` and
`official_grounding_text` into one string for
`rcs.response_generation.prompt.build_generation_prompt`'s single
`trusted_grounding_text` parameter, labeled "TRUSTED GROUNDING" in the
actual prompt sent to Gemini. That label was semantically wrong
whenever the block held only identifying context — it taught the model
that identity metadata was trustworthy factual evidence, exactly the
confusion M5D.1 fixed at the eligibility layer but left standing in the
PROMPT TEXT itself. `build_generation_prompt` now takes
`identifying_context` and `trusted_grounding_text` (official-only) as
TWO SEPARATE parameters and renders them under two separate, honestly
labeled sections ("IDENTIFYING CONTEXT" and "OFFICIAL KNOWLEDGE") — see
that module. This module no longer needs to produce a combined string
at all; `official_grounding_text` IS what
`rcs.response_intelligence.orchestrator` now passes as
`GenerationRequest.trusted_grounding_text`, and `identifying_context` is
passed as the new `GenerationRequest.identifying_context` field,
unmerged.

--- Grounding Mode ----------------------------------------------------------

- `GROUNDING_MODE_IDENTIFYING_ONLY`: no Official Knowledge results.
  `official_grounding_text` is None, `official_grounding_available` is
  False. `identifying_context` (if any) is unaffected — it is a
  completely separate concept from grounding mode.
- `GROUNDING_MODE_FACTUAL`: at least one Official Knowledge result was
  retrieved. `official_grounding_text` is a clearly labeled factual
  block built ONLY from `OfficialKnowledgeResult.content`/
  `source_title` — `identifying_context` is never concatenated into it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
)

GROUNDING_MODE_IDENTIFYING_ONLY = "IDENTIFYING_ONLY"
GROUNDING_MODE_FACTUAL = "FACTUAL"

_FACTUAL_PREFACE = (
    "The following are APPROVED FACTUAL statements retrieved from Official "
    "Knowledge. You may state these as fact. Do not add, infer, or combine "
    "them with anything beyond what is written here."
)


@dataclass(frozen=True)
class GroundingAssembly:
    """See module docstring for what each field means and why they are
    kept separate. `official_grounding_available` is the ONLY field in
    this module that may ever be treated as "we have real facts" —
    `identifying_context`'s mere presence must never be read that way."""

    identifying_context: str | None
    official_grounding_text: str | None
    official_grounding_available: bool
    mode: str


def assemble_grounding(
    identifying_context: str | None,
    official_result: OfficialKnowledgeRetrievalResult,
) -> GroundingAssembly:
    if official_result.outcome != SUCCESS_WITH_RESULTS:
        return GroundingAssembly(
            identifying_context=identifying_context,
            official_grounding_text=None,
            official_grounding_available=False,
            mode=GROUNDING_MODE_IDENTIFYING_ONLY,
        )

    facts = "\n".join(f"- ({r.source_title}) {r.content}" for r in official_result.results)
    official_grounding_text = f"{_FACTUAL_PREFACE}\n{facts}"
    return GroundingAssembly(
        identifying_context=identifying_context,
        official_grounding_text=official_grounding_text,
        official_grounding_available=True,
        mode=GROUNDING_MODE_FACTUAL,
    )


def assemble_style_reference_texts(
    approved_result: ApprovedResponseMemoryRetrievalResult,
) -> tuple[str, ...]:
    """Style/pattern examples only. Returns `()` for every outcome other
    than SUCCESS_WITH_RESULTS — including today's actual SUCCESS_EMPTY,
    which is why the prompt gains no new section by default (see
    `rcs.response_generation.prompt.build_generation_prompt`).

    Never contributes to `GroundingAssembly` in any way — there is no
    code path from this function's output to `official_grounding_text`
    or `official_grounding_available`."""
    if approved_result.outcome != SUCCESS_WITH_RESULTS:
        return ()
    return tuple(r.response_text for r in approved_result.results)
