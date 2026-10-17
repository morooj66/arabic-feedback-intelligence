"""
Identifying Context Mechanism experiment -- EVALUATION ONLY.

Follow-up to the 8-case Identifying Context Safety Check
(scripts/offline_eval/identifying_context_safety/), which found exactly
one confirmed context-hijacking case (source_row_index 6281, "لا اعلم")
and no existing deterministic signal (letter count, is_content_meaningful,
insufficient_context, intent/topic confidence) that separates it from
three other equally short/vacuous comments that did NOT hijack. This
experiment does not implement a suppression rule -- it isolates WHICH
side of the pair (comment vacuousness vs. identifying-context
specificity) actually drives the risk, by holding one side fixed while
varying the other, in a small controlled factorial.

Every variant calls the SAME real, unmodified
`rcs.response_intelligence.orchestrator.run_agent_pipeline` (rg-v10).
There is no manually-built prompt anywhere in this module, and no
Strategy V1/V2/V3 code is imported. The ONLY thing that varies within a
base case's condition set is `identifying_context`; comment text,
classification, client, and retrievers are held identical.

--- The two independent variables -----------------------------------------

COMMENT CLASS (an independent property of the base comment, not
manipulated by this experiment):
  VERY_SHORT_NONCOMMITTAL -- "لا اعلم", "الله اعلم", "تتت", "أنشطة لا صفية"
  SHORT_MEANINGFUL        -- a specific short praise / a specific short complaint
  SUBSTANTIVE             -- a long linguistic suggestion / a factual inquiry

IDENTIFYING-CONTEXT SPECIFICITY CONDITION (the manipulated variable).
Phase M6.2 update: every condition now supplies STRUCTURED
`IdentifyingMetadata` (never pre-rendered text) via
`GenerationRequest.identifying_metadata` -- whether it actually reaches
Gemini is decided once, inside the real orchestrator, by
`identifying_context_policy.evaluate_identifying_context`. This
experiment no longer decides that itself; it only supplies the raw
metadata for each condition, exactly like every other canonical caller:
  NONE     -- identifying_metadata=None
  GENERIC  -- IdentifyingMetadata("إثراء المعجم", None) -- a real,
             observed, purely administrative form-section title (it is
             the actual original_title on several real rows, e.g. 6236,
             6278, 6281 -- genuinely non-specific, not invented for this
             experiment)
  SPECIFIC -- IdentifyingMetadata("قناطر", None) -- a real,
             specific Arabic dictionary word: the EXACT entry_id that
             caused the confirmed hijacking in row 6281, deliberately
             reused as a FIXED specific-condition stimulus so all
             VERY_SHORT_NONCOMMITTAL base comments are tested against
             the identical specific stimulus (isolates comment-side
             variance, holds context-side "specificity" constant)
  OWN_REAL -- IdentifyingMetadata(row.original_title, row.entry_id)
             -- the base comment's own, real, naturally-occurring
             identifying context exactly as production would supply it.
             Used only for the SHORT_MEANINGFUL/SUBSTANTIVE base cases,
             where the question is "does identifying context (as
             production actually supplies it) destabilize an
             already-meaningful comment", not the full specificity
             gradient (already tested via the VERY_SHORT_NONCOMMITTAL
             cases).

--- 8 base cases, 18 generated variants total (within the requested
    12-18 range) ---

  لا اعلم (row 6281)              NONE, GENERIC, SPECIFIC   (3) -- the confirmed hijacking case
  الله اعلم (row 112)             NONE, GENERIC, SPECIFIC   (3) -- the original discovery case
  تتت (row 6278)                  NONE, SPECIFIC            (2)
  أنشطة لا صفية (row 6236)        NONE, SPECIFIC            (2)
  short praise (row 15)           NONE, OWN_REAL            (2)
  short complaint (row 211)       NONE, OWN_REAL            (2)
  linguistic suggestion (row 1)   NONE, OWN_REAL            (2)
  factual inquiry (row 10)        NONE, OWN_REAL            (2)
                                                            ----
                                                             18

Writes only to results/identifying_context_mechanism_cases.jsonl. No
database write of any kind. No source Excel mutation (read-only via
`rcs.ingest.loader.load_raw`). Does not import prompt_candidates.py or
any strategy_v*.py module.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402
from rcs.response_generation.prompt import build_identifying_grounding, redact_for_generation  # noqa: E402
from rcs.response_generation.service import GenerationRequest, SupportsGenerate, build_live_gemini_client  # noqa: E402
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata  # noqa: E402
from rcs.response_intelligence.inspector import inspect_classification, inspect_eligibility  # noqa: E402
from rcs.response_intelligence.orchestrator import run_agent_pipeline  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402

SOURCE_FILE = _ROOT / "data" / "raw" / "historical-export.xlsx"
RESULTS_PATH = _HERE / "results" / "identifying_context_mechanism_cases.jsonl"

GENERIC_TITLE = "إثراء المعجم"
SPECIFIC_TITLE = "قناطر"

CONDITIONS = ("NONE", "GENERIC", "SPECIFIC", "OWN_REAL")


@dataclass(frozen=True)
class BaseCase:
    base_case_id: str
    source_row_index: int
    role: str
    comment_class: str
    conditions: tuple[str, ...]


SELECTED_BASE_CASES: tuple[BaseCase, ...] = (
    BaseCase("laa_aelam", 6281, "VACUOUS", "VERY_SHORT_NONCOMMITTAL", ("NONE", "GENERIC", "SPECIFIC")),
    BaseCase("allah_aelam", 112, "VACUOUS", "VERY_SHORT_NONCOMMITTAL", ("NONE", "GENERIC", "SPECIFIC")),
    BaseCase("ttt_fragment", 6278, "VACUOUS", "VERY_SHORT_NONCOMMITTAL", ("NONE", "SPECIFIC")),
    BaseCase("anshita_la_safiya", 6236, "VACUOUS", "VERY_SHORT_NONCOMMITTAL", ("NONE", "SPECIFIC")),
    BaseCase("short_praise", 15, "PRAISE", "SHORT_MEANINGFUL", ("NONE", "OWN_REAL")),
    BaseCase("short_complaint", 211, "COMPLAINT", "SHORT_MEANINGFUL", ("NONE", "OWN_REAL")),
    BaseCase("linguistic_suggestion", 1, "LINGUISTIC_SUGGESTION", "SUBSTANTIVE", ("NONE", "OWN_REAL")),
    BaseCase("factual_inquiry", 10, "FACTUAL_INQUIRY", "SUBSTANTIVE", ("NONE", "OWN_REAL")),
)


@dataclass(frozen=True)
class MechanismCaseResult:
    eval_id: str
    base_case_id: str
    source_row_index: int
    role: str
    comment_class: str
    context_specificity: str
    comment_preview: str
    identifying_context_preview: str | None
    primary_intent: str
    topic_subtype: str
    sentiment: str
    routing_decision: str
    official_grounding_available: bool
    generated_reply: str | None
    generation_status: str
    reason_code: str | None


def _resolve_identifying_metadata(condition: str, row) -> IdentifyingMetadata | None:
    """Phase M6.2: returns STRUCTURED metadata, never pre-rendered text.
    Whether this reaches Gemini at all is decided once, inside
    run_agent_pipeline, by identifying_context_policy.
    evaluate_identifying_context -- not by this experiment script."""
    if condition == "NONE":
        return None
    if condition == "GENERIC":
        return IdentifyingMetadata(GENERIC_TITLE, None)
    if condition == "SPECIFIC":
        return IdentifyingMetadata(SPECIFIC_TITLE, None)
    if condition == "OWN_REAL":
        title = str(row["original_title"]) if row["original_title"] not in (None, "") else None
        entry_id = str(row["entry_id"]) if row["entry_id"] not in (None, "") else None
        return IdentifyingMetadata(title, entry_id)
    raise ValueError(f"unknown condition: {condition!r}")


def _load_rows():
    df = load_raw(SOURCE_FILE, settings.SOURCE_COLUMNS)
    row_indices = {bc.source_row_index for bc in SELECTED_BASE_CASES}
    df = df[df["source_row_index"].isin(row_indices)]
    by_index = {int(r["source_row_index"]): r for _, r in df.iterrows()}
    missing = [i for i in row_indices if i not in by_index]
    if missing:
        raise ValueError(f"selected source_row_index values not found in workbook: {missing}")
    return by_index


def run_experiment() -> Path:
    load_dotenv_if_present()
    client = build_live_gemini_client()
    rows_by_index = _load_rows()

    cases: list[MechanismCaseResult] = []
    for base_case in SELECTED_BASE_CASES:
        row = rows_by_index[base_case.source_row_index]
        text = str(row["original_comment"])
        classification = inspect_classification(text)
        eligibility = inspect_eligibility(
            comment_text=text,
            primary_intent=classification.primary_intent,
            topic_subtype=classification.topic_subtype,
            sentiment=classification.sentiment,
            confidence=classification.primary_intent_confidence,
        )
        comment_preview = redact_for_generation(text)[:100]

        for condition in base_case.conditions:
            identifying_metadata = _resolve_identifying_metadata(condition, row)
            # Preview-only rendering (for human review display) -- NOT
            # what reaches Gemini. The actual GenerationRequest below
            # carries only structured `identifying_metadata`; whether it
            # is rendered/used at all is decided by the real orchestrator
            # policy, independent of this preview.
            identifying_context_preview = None
            if identifying_metadata is not None:
                rendered_for_preview = build_identifying_grounding(
                    identifying_metadata.original_title, identifying_metadata.entry_id
                )
                identifying_context_preview = (
                    redact_for_generation(rendered_for_preview)[:200] if rendered_for_preview else None
                )
            eval_id = f"{base_case.base_case_id}-{condition}"

            if eligibility.routing_decision != "GENERATE_DRAFT":
                cases.append(
                    MechanismCaseResult(
                        eval_id=eval_id, base_case_id=base_case.base_case_id,
                        source_row_index=base_case.source_row_index, role=base_case.role,
                        comment_class=base_case.comment_class, context_specificity=condition,
                        comment_preview=comment_preview, identifying_context_preview=identifying_context_preview,
                        primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                        sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                        official_grounding_available=False, generated_reply=None,
                        generation_status="NOT_ELIGIBLE", reason_code=eligibility.reason_code,
                    )
                )
                continue

            request = GenerationRequest(
                comment_id=f"idctx-mechanism-{eval_id}",
                comment_text=text,
                primary_intent=classification.primary_intent,
                topic_subtype=classification.topic_subtype,
                sentiment=classification.sentiment,
                confidence=classification.primary_intent_confidence,
                identifying_metadata=identifying_metadata,
            )
            exec_result = run_agent_pipeline(request, client=client)

            cases.append(
                MechanismCaseResult(
                    eval_id=eval_id, base_case_id=base_case.base_case_id,
                    source_row_index=base_case.source_row_index, role=base_case.role,
                    comment_class=base_case.comment_class, context_specificity=condition,
                    comment_preview=comment_preview, identifying_context_preview=identifying_context_preview,
                    primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                    sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                    official_grounding_available=exec_result.metadata.official_grounding_available,
                    generated_reply=exec_result.generation_result.suggested_response,
                    generation_status=exec_result.generation_result.generation_status,
                    reason_code=exec_result.generation_result.reason_code,
                )
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    return RESULTS_PATH


if __name__ == "__main__":
    path = run_experiment()
    print(f"wrote {path}")
