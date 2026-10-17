"""
Response-Strategy V2 A/B experiment -- EVALUATION ONLY.

Does NOT overwrite or read V1's result files (results/strategy_ab_*).
Writes to results/strategy_v2_ab_cases.jsonl only.

For each selected real, eligible comment:
  Reply A = the REAL, unmodified production orchestrator
            (rcs.response_intelligence.orchestrator.run_agent_pipeline),
            exactly as production calls it -- real retrieval, real
            grounding enforcement, real validation, real bounded
            refinement (up to one). Identical code path to V1's Reply A.
  Reply B = the SAME rg-v10 SYSTEM_INSTRUCTIONS (imported, never
            duplicated) plus ONE additional "RESPONSE STRATEGY V2" block
            (see strategy_v2.py), through a single real Gemini call and
            the SAME real validate_draft -- WITHOUT the bounded-
            refinement pass (same deliberate scope reduction as V1, so
            the comparison stays at exactly one Gemini call per variant
            per case).

Phase M6.2 note: Reply A's GenerationRequest now supplies structured
`identifying_metadata`; the real orchestrator's Context/Trust Policies
step decides whether it is used. Reply B deliberately bypasses the
orchestrator (the whole point of this experiment) and renders
identifying context directly via the unchanged, unconditional
`build_identifying_grounding`, explicitly out of scope for the new
policy -- same reasoning as V1.

Neither variant writes to any database. Neither variant modifies
production code -- prompt.py, orchestrator.py, service.py, and
validate.py are IMPORTED and reused unchanged, never edited.

--- Case selection (fixed, documented, derived BEFORE seeing any
    generated replies -- never re-picked after seeing output) ---

Source: scanned source_row_index range [0, 1500) of the same real
workbook (data/raw/historical-export.xlsx), batch-
classified with the real frozen classifiers (predict_intent/
predict_topic/predict_sentiment) and filtered to
inspect_eligibility(...).routing_decision == GENERATE_DRAFT.

Exact-normalized-comment duplicates were collapsed to at most one
representative (the routing-safety evaluation separately found one
10-row exact-duplicate cluster at source_row_index 1002-1011 -- "مِضْرَادٌ";
NONE of those rows are included here, to avoid one duplicate cluster
dominating the A/B sample). Rows 1190/1191 are a near-duplicate pair
(not byte-identical -- 1190 = 1191 + an appended phone-number fragment);
only 1191 (the version without the appended contact fragment) is
included, per the same "at most one representative" rule applied to
near-duplicates found during manual inspection.

Selected buckets (25 rows total, within the requested 20-30 range):
  SUGGESTION + ENTRY_CORRECTION (5): 1, 2, 63, 146, 987
      -- rows 1 ("اعتبار") and 2 ("المتلازمات اللفظية") are the two
      difficult linguistic cases the V1 evaluation explicitly wanted to
      improve on (reused deliberately, unlike every other row here).
  SUGGESTION + WORD_SUBMISSION (6): 9, 16, 18, 22, 25, 92
  INQUIRY, mixed topics (4): 10, 19, 67, 84
  COMPLAINT, mixed topics (4): 211, 603, 845, 991
  THANKS_PRAISE, mixed topics (4): 15, 112, 180, 362
  DICTIONARY_SUGGESTION, dedup representative (1): 1191
  Additional long/substantive diversity (1): 1074

Rows 1, 2, 9, 10, 15, 22, 63, 211 were also used in the V1 experiment;
all other rows are new to this evaluation, chosen to cover intent x
topic combinations V1's design flaw would have mishandled (e.g.
INQUIRY+WORD_SUBMISSION at rows 10/19, COMPLAINT+WORD_SUBMISSION at
rows 211/603, THANKS_PRAISE+WORD_SUBMISSION at rows 15/112).
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
sys.path.insert(0, str(_HERE))

from config import settings  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402
from rcs.response_generation.prompt import build_generation_prompt, build_identifying_grounding  # noqa: E402
from rcs.response_generation.schema import JSON_SCHEMA  # noqa: E402
from rcs.response_generation.service import GenerationRequest, build_live_gemini_client  # noqa: E402
from rcs.response_generation.validate import validate_draft  # noqa: E402
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata  # noqa: E402
from rcs.response_intelligence.inspector import inspect_classification, inspect_eligibility  # noqa: E402
from rcs.response_intelligence.orchestrator import run_agent_pipeline  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402

from strategy_v2 import derive_strategy_v2, render_strategy_v2_block  # noqa: E402

SOURCE_FILE = _ROOT / "data" / "raw" / "historical-export.xlsx"
RESULTS_PATH = _HERE / "results" / "strategy_v2_ab_cases.jsonl"

SELECTED_SOURCE_ROW_INDICES_V2 = (
    1, 2, 9, 10, 15, 16, 18, 19, 22, 25,
    63, 67, 84, 92, 112, 146, 180, 211, 362, 603,
    845, 987, 991, 1074, 1191,
)


@dataclass(frozen=True)
class ABCaseV2:
    eval_id: str
    source_row_index: int
    comment_preview: str
    primary_intent: str
    topic_subtype: str
    sentiment: str
    routing_decision: str
    reply_a: str | None
    reply_a_status: str
    reply_b: str | None
    reply_b_status: str
    strategy_goal: str
    strategy_mode: str
    strategy_depth: str
    strategy_key_focus: str


def _load_selected_rows():
    df = load_raw(SOURCE_FILE, settings.SOURCE_COLUMNS)
    df = df[df["source_row_index"].isin(SELECTED_SOURCE_ROW_INDICES_V2)]
    by_index = {int(r["source_row_index"]): r for _, r in df.iterrows()}
    missing = [i for i in SELECTED_SOURCE_ROW_INDICES_V2 if i not in by_index]
    if missing:
        raise ValueError(f"selected source_row_index values not found in workbook: {missing}")
    return [by_index[i] for i in SELECTED_SOURCE_ROW_INDICES_V2]


def _generate_reply_b(
    request: GenerationRequest, client, reason_code: str, identifying_context: str | None
) -> tuple[str | None, str]:
    """Single-shot: eligibility already confirmed GENERATE_DRAFT by the
    caller. Reuses build_generation_prompt (rg-v10, unmodified) and
    inserts the V2 strategy block via a safe string append -- the SAME
    function production uses, never a reimplementation of prompt
    assembly. No extra Gemini call is made to derive the strategy.

    Phase M6.2: `identifying_context` is passed in explicitly (rendered
    by the caller), not read off `request.identifying_context` -- this
    manual path deliberately bypasses the orchestrator's Context/Trust
    Policies step and is explicitly out of scope for the new policy;
    see module docstring."""
    strategy = derive_strategy_v2(
        comment_text=request.comment_text,
        primary_intent=request.primary_intent,
        topic_subtype=request.topic_subtype,
        official_grounding_available=False,  # no Official Knowledge source is configured today (see M5B.4)
    )
    base_prompt = build_generation_prompt(
        comment_text=request.comment_text,
        primary_intent=request.primary_intent,
        topic_subtype=request.topic_subtype,
        sentiment=request.sentiment,
        reason_code=reason_code,
        identifying_context=identifying_context,
        trusted_grounding_text=None,
    )
    strategy_block = render_strategy_v2_block(strategy)
    marker = "\n\nRespond with JSON only."
    assert base_prompt.endswith(marker)
    prompt_b = base_prompt[: -len(marker)] + "\n\n" + strategy_block + marker

    try:
        raw = client.generate(prompt_b, JSON_SCHEMA)
    except Exception as exc:  # noqa: BLE001 -- evaluation harness, report and continue
        return None, f"transport_error:{type(exc).__name__}"

    try:
        parsed = json.loads(raw)
        draft = parsed["suggested_response"]
    except Exception:  # noqa: BLE001
        return None, "malformed_output"

    ok, why = validate_draft(draft, trusted_grounding_text=None)
    if not ok:
        return draft, f"failed_validation:{why}"
    return draft.strip(), "SUCCEEDED"


def run_experiment() -> Path:
    load_dotenv_if_present()
    client = build_live_gemini_client()
    rows = _load_selected_rows()

    cases: list[ABCaseV2] = []
    for row in rows:
        text = str(row["original_comment"])
        cid = str(row["comment_id"])
        classification = inspect_classification(text)
        eligibility = inspect_eligibility(
            comment_text=text,
            primary_intent=classification.primary_intent,
            topic_subtype=classification.topic_subtype,
            sentiment=classification.sentiment,
            confidence=classification.primary_intent_confidence,
        )

        from rcs.response_generation.prompt import redact_for_generation

        preview = redact_for_generation(text)[:100]

        if eligibility.routing_decision != "GENERATE_DRAFT":
            cases.append(
                ABCaseV2(
                    eval_id=cid, source_row_index=int(row["source_row_index"]), comment_preview=preview,
                    primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                    sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                    reply_a=None, reply_a_status="NOT_ELIGIBLE", reply_b=None, reply_b_status="NOT_ELIGIBLE",
                    strategy_goal="", strategy_mode="", strategy_depth="", strategy_key_focus="",
                )
            )
            continue

        title = str(row["original_title"]) if row["original_title"] not in (None, "") else None
        entry_id = str(row["entry_id"]) if row["entry_id"] not in (None, "") else None
        request = GenerationRequest(
            comment_id=f"strategy-v2-eval-row{row['source_row_index']}",
            comment_text=text,
            primary_intent=classification.primary_intent,
            topic_subtype=classification.topic_subtype,
            sentiment=classification.sentiment,
            confidence=classification.primary_intent_confidence,
            identifying_metadata=IdentifyingMetadata(title, entry_id),
        )

        exec_result = run_agent_pipeline(request, client=client, max_retries=None)
        reply_a = exec_result.generation_result.suggested_response
        reply_a_status = exec_result.generation_result.generation_status

        # Reply B: manual path, deliberately outside the orchestrator/policy -- see module docstring.
        identifying_context_for_manual_path = build_identifying_grounding(title, entry_id)
        reply_b, reply_b_status = _generate_reply_b(
            request, client, eligibility.reason_code, identifying_context_for_manual_path
        )
        strategy = derive_strategy_v2(
            comment_text=text, primary_intent=classification.primary_intent,
            topic_subtype=classification.topic_subtype, official_grounding_available=False,
        )

        cases.append(
            ABCaseV2(
                eval_id=cid, source_row_index=int(row["source_row_index"]), comment_preview=preview,
                primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                reply_a=reply_a, reply_a_status=reply_a_status, reply_b=reply_b, reply_b_status=reply_b_status,
                strategy_goal=strategy.communicative_goal, strategy_mode=strategy.response_mode,
                strategy_depth=strategy.response_depth, strategy_key_focus=strategy.key_focus,
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
