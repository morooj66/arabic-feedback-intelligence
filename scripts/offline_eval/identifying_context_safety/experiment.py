"""
Identifying Context Safety Check -- EVALUATION ONLY.

Answers ONE narrow question: now that `ResponseAgentState.from_request`
correctly resolves `identifying_context` (Phase M6.1), does rg-v10 (the
real, unmodified production generation path) ever misuse dictionary-
entry identifying metadata (title/entry_id) as though it were the
subject of an unrelated or ambiguous visitor comment?

This is NOT a Reply Strategy experiment. No V4. No re-comparison of
V1/V2/V3. There is no manually-built alternative prompt anywhere in
this module: BOTH variants call the SAME real, unmodified
`rcs.response_intelligence.orchestrator.run_agent_pipeline` (rg-v10).
The ONLY experimental variable is whether `identifying_context` is set
on the `GenerationRequest`:

  CONTROL  -- identifying_context=None
  CONTEXT  -- identifying_context=build_identifying_grounding(original_title, entry_id)

Everything else (comment text, classification, topic, sentiment,
retriever instances, client instance) is identical between the two
calls for a given case, so any difference in Official Knowledge
availability or reply content can only be attributed to that one field.

Writes only to results/identifying_context_safety_cases.jsonl. No
database write of any kind. No source Excel mutation (read-only via
`rcs.ingest.loader.load_raw`). Does not touch prompt.py, orchestrator.py,
eligibility.py, or validate.py -- all imported and reused unchanged.

--- Case selection (fixed, documented, 9 rows) -----------------------------

Deliberately reuses a mix of ALREADY-known rows (from prior audits/
evaluations) plus fresh ones -- this check is about identifying-context
SAFETY, not blinded reply-quality comparison, so there is no confound
in reusing a row already seen in V1/V2/V3.

  A. Ambiguous / very short comments:
     row 112  -- "الله اعلم" (THANKS_PRAISE), title "ابرايل الله"
                 (the exact case that originally surfaced this risk)
     row 6281 -- "لا اعلم" (COMPLAINT), entry_id "قناطر" (a real,
                 unrelated word)
     row 6278 -- "تتت" (unparseable fragment), entry_id "خ"
     row 6236 -- "أنشطة لا صفية" (ambiguous short phrase), generic
                 title "إثراء المعجم", no entry_id

  B. Subject clearly different from the entry title:
     row 991  -- COMPLAINT genuinely about the word "زان", but the
                 title is the generic administrative label "تصحيح"
                 ("correction"), not a specific word at all
     (rows 112 and 6281 above are ALSO B cases -- title/entry_id is a
     real but unrelated word/phrase)

  C. Positive controls -- comment genuinely about the entry:
     row 1    -- SUGGESTION about «اعتبار», title "اِعْتِبَار" (exact match)
     row 10   -- INQUIRY about "اشتد", title "اِشْتَدَّ" (exact match)
     row 6285 -- INQUIRY about "عفريه", entry_id "عفريه" (exact match)
     row 15   -- THANKS_PRAISE "أحسنتم", title "إِطَاعَة" (present but
                 the comment itself is too short to reference it either
                 way -- a mild version of the A/C boundary)

  D. Factual inquiry, identifying context present, no Official
     Knowledge (guaranteed by today's empty corpus -- see
     rcs.response_intelligence.retrievers):
     row 10, row 6285 (already listed above; both are also C)

Excluded finding, not part of the sample: source_row_index 25's
`original_title` field is itself a raw email address
("sample.person1@example.test") -- a pre-existing source-data-quality/PII
concern in the workbook, unrelated to this check. Deliberately excluded
here to avoid any risk of persisting raw PII in evaluation output;
reported separately, not investigated further in this milestone.
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
from rcs.response_intelligence.retrieval.contracts import (
    ApprovedResponseMemoryRetriever,
    OfficialKnowledgeRetriever,
)
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402

SOURCE_FILE = _ROOT / "data" / "raw" / "historical-export.xlsx"
RESULTS_PATH = _HERE / "results" / "identifying_context_safety_cases.jsonl"

# source_row_index -> tuple of category tags (see module docstring)
SELECTED_ROWS: dict[int, tuple[str, ...]] = {
    112: ("A", "B"),
    6281: ("A", "B"),
    6278: ("A",),
    6236: ("A",),
    991: ("B",),
    1: ("C",),
    10: ("C", "D"),
    6285: ("C", "D"),
    15: ("A", "C"),
}


@dataclass(frozen=True)
class IdentifyingContextPairCase:
    eval_id: str
    source_row_index: int
    categories: tuple[str, ...]
    comment_preview: str
    identifying_context_preview: str | None
    primary_intent: str
    topic_subtype: str
    sentiment: str
    routing_decision: str
    official_grounding_available_control: bool
    official_grounding_available_context: bool
    control_reply: str | None
    control_status: str
    context_reply: str | None
    context_status: str


def _load_selected_rows():
    df = load_raw(SOURCE_FILE, settings.SOURCE_COLUMNS)
    df = df[df["source_row_index"].isin(SELECTED_ROWS.keys())]
    by_index = {int(r["source_row_index"]): r for _, r in df.iterrows()}
    missing = [i for i in SELECTED_ROWS if i not in by_index]
    if missing:
        raise ValueError(f"selected source_row_index values not found in workbook: {missing}")
    return [by_index[i] for i in SELECTED_ROWS]


def run_pair(
    *,
    comment_id_prefix: str,
    comment_text: str,
    primary_intent: str | None,
    topic_subtype: str | None,
    sentiment: str | None,
    confidence: float | None,
    identifying_metadata: IdentifyingMetadata | None,
    client: SupportsGenerate | None,
    official_knowledge_retriever: OfficialKnowledgeRetriever | None = None,
    approved_response_retriever: ApprovedResponseMemoryRetriever | None = None,
):
    """Runs the CONTROL (identifying_metadata=None) and CONTEXT
    (identifying_metadata=<value>) variants through the SAME real,
    unmodified `run_agent_pipeline` -- no manually-built prompt anywhere.
    Both calls share the same client/retriever instances so any
    difference in behavior traces to exactly one field. Phase M6.2:
    structured metadata only -- whether the CONTEXT variant's metadata
    is actually used is now decided by the real orchestrator policy,
    not by this script."""
    control_request = GenerationRequest(
        comment_id=f"{comment_id_prefix}-control",
        comment_text=comment_text,
        primary_intent=primary_intent,
        topic_subtype=topic_subtype,
        sentiment=sentiment,
        confidence=confidence,
        identifying_metadata=None,
    )
    context_request = GenerationRequest(
        comment_id=f"{comment_id_prefix}-context",
        comment_text=comment_text,
        primary_intent=primary_intent,
        topic_subtype=topic_subtype,
        sentiment=sentiment,
        confidence=confidence,
        identifying_metadata=identifying_metadata,
    )

    control_result = run_agent_pipeline(
        control_request, client=client,
        official_knowledge_retriever=official_knowledge_retriever,
        approved_response_retriever=approved_response_retriever,
    )
    context_result = run_agent_pipeline(
        context_request, client=client,
        official_knowledge_retriever=official_knowledge_retriever,
        approved_response_retriever=approved_response_retriever,
    )
    return control_result, context_result


def run_experiment() -> Path:
    load_dotenv_if_present()
    client = build_live_gemini_client()
    rows = _load_selected_rows()

    cases: list[IdentifyingContextPairCase] = []
    for row in rows:
        row_index = int(row["source_row_index"])
        text = str(row["original_comment"])
        classification = inspect_classification(text)
        eligibility = inspect_eligibility(
            comment_text=text,
            primary_intent=classification.primary_intent,
            topic_subtype=classification.topic_subtype,
            sentiment=classification.sentiment,
            confidence=classification.primary_intent_confidence,
        )

        title = str(row["original_title"]) if row["original_title"] not in (None, "") else None
        entry_id = str(row["entry_id"]) if row["entry_id"] not in (None, "") else None
        identifying_metadata = IdentifyingMetadata(title, entry_id)
        comment_preview = redact_for_generation(text)[:100]
        # Preview-only rendering (human review display) -- NOT what
        # necessarily reaches Gemini; the real orchestrator policy
        # decides that independently for the CONTEXT variant below.
        rendered_for_preview = build_identifying_grounding(title, entry_id)
        identifying_context_preview = (
            redact_for_generation(rendered_for_preview)[:200] if rendered_for_preview else None
        )

        if eligibility.routing_decision != "GENERATE_DRAFT":
            cases.append(
                IdentifyingContextPairCase(
                    eval_id=f"row{row_index}", source_row_index=row_index, categories=SELECTED_ROWS[row_index],
                    comment_preview=comment_preview, identifying_context_preview=identifying_context_preview,
                    primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                    sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                    official_grounding_available_control=False, official_grounding_available_context=False,
                    control_reply=None, control_status="NOT_ELIGIBLE",
                    context_reply=None, context_status="NOT_ELIGIBLE",
                )
            )
            continue

        control_result, context_result = run_pair(
            comment_id_prefix=f"idctx-safety-row{row_index}",
            comment_text=text,
            primary_intent=classification.primary_intent,
            topic_subtype=classification.topic_subtype,
            sentiment=classification.sentiment,
            confidence=classification.primary_intent_confidence,
            identifying_metadata=identifying_metadata,
            client=client,
        )

        cases.append(
            IdentifyingContextPairCase(
                eval_id=f"row{row_index}", source_row_index=row_index, categories=SELECTED_ROWS[row_index],
                comment_preview=comment_preview, identifying_context_preview=identifying_context_preview,
                primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                official_grounding_available_control=control_result.metadata.official_grounding_available,
                official_grounding_available_context=context_result.metadata.official_grounding_available,
                control_reply=control_result.generation_result.suggested_response,
                control_status=control_result.generation_result.generation_status,
                context_reply=context_result.generation_result.suggested_response,
                context_status=context_result.generation_result.generation_status,
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
