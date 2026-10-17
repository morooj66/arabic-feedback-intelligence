"""
Response-Strategy V3 A/B experiment -- EVALUATION ONLY, selective gate.

Does NOT overwrite or read V1/V2 result files (results/strategy_ab_*,
results/strategy_v2_ab_*). Writes only to results/strategy_v3_ab_cases.jsonl.

For each selected real, eligible comment:
  Reply A = the REAL, unmodified production orchestrator
            (rcs.response_intelligence.orchestrator.run_agent_pipeline),
            exactly as production calls it. Identical code path to
            V1/V2's Reply A.
  Reply V3 = gated:
    - gate OPEN  (primary_intent == SUGGESTION and response_depth ==
      SUBSTANTIVE): SAME rg-v10 SYSTEM_INSTRUCTIONS plus ONE additional
      "RESPONSE STRATEGY V2" block (strategy content reused unchanged
      from V2 -- V3 only adds the gate, not a new strategy design),
      through a single real Gemini call, same real validate_draft, no
      refinement pass (same deliberate scope reduction as V1/V2).
    - gate CLOSED: the prompt is BYTE-FOR-BYTE the same rg-v10 prompt
      Reply A's own prompt construction would produce for this input --
      no strategy block appended at all. This case is expected to
      behave observably the same as baseline (see
      test_gate_closed_case_prompt_is_byte_identical_to_baseline_prompt
      in the test suite), and is included in the sample specifically to
      verify that, not to "win" anything.

Neither variant writes to any database. Neither variant modifies
production code -- prompt.py, orchestrator.py, service.py, and
validate.py are IMPORTED and reused unchanged, never edited.

Phase M6.2 note: Reply A's GenerationRequest now supplies structured
`identifying_metadata`; the real orchestrator's Context/Trust Policies
step decides whether it is used (this is, in fact, the production
resolution of the "ابرايل الله" issue this file's earlier docstring
recorded as known, separate technical debt). Reply V3's manual path
deliberately bypasses the orchestrator and renders identifying context
directly via the unchanged, unconditional `build_identifying_grounding`,
explicitly out of scope for the new policy -- same reasoning as V1/V2.

--- Case selection (fixed, documented, derived BEFORE seeing any
    generated replies) ---

Source: scanned source_row_index range [1500, 8413) of the same real
workbook -- deliberately NOT the [0, 1500) range V1/V2 scanned, and
every row already used by V1 or V2 (25 rows, a superset of V1's 8) is
explicitly excluded, along with the known exact-duplicate cluster
(1002-1011, outside this range anyway) and one manually-identified
near-duplicate pair found during scanning (rows 1581/1582 and 1550/1551
each pair being two near-identical resubmissions of the same point;
only one representative per pair -- or neither -- is included).
Exact-normalized-text duplicates were deduplicated automatically during
the scan (first occurrence kept, later duplicates skipped).

Selected buckets (26 rows total, within the requested 24-30 range):
  GATE-POSITIVE: SUGGESTION + SUBSTANTIVE (>=400 chars) (10):
      1595, 1636, 1724, 1777, 1824, 1851, 1868, 2088, 2097, 6240
  GATE-NEGATIVE control -- short SUGGESTION (BRIEF) (4):
      1544, 1546, 1567, 1600
  GATE-NEGATIVE control -- THANKS_PRAISE (4):
      1560, 1612, 1625, 1714
  GATE-NEGATIVE control -- INQUIRY (4):
      1550, 1572, 1601, 1642
  GATE-NEGATIVE control -- COMPLAINT (4):
      1739, 1823, 6236, 6281

None of these 26 rows were used in the V1 (8 rows) or V2 (25 rows)
evaluations.
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

from strategy_v3 import derive_strategy_v3, render_strategy_v3_block  # noqa: E402

SOURCE_FILE = _ROOT / "data" / "raw" / "historical-export.xlsx"
RESULTS_PATH = _HERE / "results" / "strategy_v3_ab_cases.jsonl"

# Rows already used by V1 (8) and V2 (25, a superset of V1's rows) --
# excluded from selection by construction (asserted in run_experiment).
V1_ROWS = (1, 2, 9, 10, 15, 22, 63, 211)
V2_ROWS = (
    1, 2, 9, 10, 15, 16, 18, 19, 22, 25, 63, 67, 84, 92, 112, 146, 180,
    211, 362, 603, 845, 987, 991, 1074, 1191,
)
PREVIOUSLY_EVALUATED_ROWS = frozenset(V1_ROWS) | frozenset(V2_ROWS)

SELECTED_SOURCE_ROW_INDICES_V3 = (
    # gate-positive: SUGGESTION + SUBSTANTIVE
    1595, 1636, 1724, 1777, 1824, 1851, 1868, 2088, 2097, 6240,
    # gate-negative: short SUGGESTION (BRIEF)
    1544, 1546, 1567, 1600,
    # gate-negative: THANKS_PRAISE
    1560, 1612, 1625, 1714,
    # gate-negative: INQUIRY
    1550, 1572, 1601, 1642,
    # gate-negative: COMPLAINT
    1739, 1823, 6236, 6281,
)

assert not (set(SELECTED_SOURCE_ROW_INDICES_V3) & PREVIOUSLY_EVALUATED_ROWS), (
    "V3 sample must not reuse any V1/V2 evaluated row"
)


@dataclass(frozen=True)
class ABCaseV3:
    eval_id: str
    source_row_index: int
    comment_preview: str
    primary_intent: str
    topic_subtype: str
    sentiment: str
    routing_decision: str
    reply_a: str | None
    reply_a_status: str
    reply_v3: str | None
    reply_v3_status: str
    strategy_enabled: bool
    strategy_goal: str
    strategy_mode: str
    strategy_depth: str
    strategy_key_focus: str


def _load_selected_rows():
    df = load_raw(SOURCE_FILE, settings.SOURCE_COLUMNS)
    df = df[df["source_row_index"].isin(SELECTED_SOURCE_ROW_INDICES_V3)]
    by_index = {int(r["source_row_index"]): r for _, r in df.iterrows()}
    missing = [i for i in SELECTED_SOURCE_ROW_INDICES_V3 if i not in by_index]
    if missing:
        raise ValueError(f"selected source_row_index values not found in workbook: {missing}")
    return [by_index[i] for i in SELECTED_SOURCE_ROW_INDICES_V3]


def _generate_reply_v3(
    request: GenerationRequest, client, reason_code: str, identifying_context: str | None
):
    """Single-shot: eligibility already confirmed GENERATE_DRAFT by the
    caller. Builds the SAME rg-v10 base prompt production uses
    (build_generation_prompt, unmodified) and, ONLY if the gate is open,
    appends the V2 strategy block via safe string-append -- exactly like
    V2's own Reply B. When the gate is closed, the prompt sent to Gemini
    is the unmodified base prompt, byte-for-byte -- no strategy text, no
    marker, nothing added. No extra Gemini call is made to derive the
    strategy or the gate decision (both are pure Python).

    Phase M6.2: `identifying_context` is passed in explicitly (rendered
    by the caller), not read off `request.identifying_context` -- this
    manual path deliberately bypasses the orchestrator's Context/Trust
    Policies step, explicitly out of scope for the new policy."""
    decision = derive_strategy_v3(
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
    strategy_block = render_strategy_v3_block(decision)
    if strategy_block is None:
        prompt = base_prompt  # baseline-equivalent: nothing appended
    else:
        marker = "\n\nRespond with JSON only."
        assert base_prompt.endswith(marker)
        prompt = base_prompt[: -len(marker)] + "\n\n" + strategy_block + marker

    try:
        raw = client.generate(prompt, JSON_SCHEMA)
    except Exception as exc:  # noqa: BLE001 -- evaluation harness, report and continue
        return None, f"transport_error:{type(exc).__name__}", decision

    try:
        parsed = json.loads(raw)
        draft = parsed["suggested_response"]
    except Exception:  # noqa: BLE001
        return None, "malformed_output", decision

    ok, why = validate_draft(draft, trusted_grounding_text=None)
    if not ok:
        return draft, f"failed_validation:{why}", decision
    return draft.strip(), "SUCCEEDED", decision


def run_experiment() -> Path:
    load_dotenv_if_present()
    client = build_live_gemini_client()
    rows = _load_selected_rows()

    cases: list[ABCaseV3] = []
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
                ABCaseV3(
                    eval_id=cid, source_row_index=int(row["source_row_index"]), comment_preview=preview,
                    primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                    sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                    reply_a=None, reply_a_status="NOT_ELIGIBLE", reply_v3=None, reply_v3_status="NOT_ELIGIBLE",
                    strategy_enabled=False, strategy_goal="", strategy_mode="", strategy_depth="",
                    strategy_key_focus="",
                )
            )
            continue

        title = str(row["original_title"]) if row["original_title"] not in (None, "") else None
        entry_id = str(row["entry_id"]) if row["entry_id"] not in (None, "") else None
        request = GenerationRequest(
            comment_id=f"strategy-v3-eval-row{row['source_row_index']}",
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

        # Reply V3: manual path, deliberately outside the orchestrator/policy -- see module docstring.
        identifying_context_for_manual_path = build_identifying_grounding(title, entry_id)
        reply_v3, reply_v3_status, decision = _generate_reply_v3(
            request, client, eligibility.reason_code, identifying_context_for_manual_path
        )
        strategy = decision.strategy  # always populated (see StrategyV3Decision) -- recorded for analysis
        # even on gate-negative cases, though never rendered into the prompt when disabled.

        cases.append(
            ABCaseV3(
                eval_id=cid, source_row_index=int(row["source_row_index"]), comment_preview=preview,
                primary_intent=classification.primary_intent, topic_subtype=classification.topic_subtype,
                sentiment=classification.sentiment, routing_decision=eligibility.routing_decision,
                reply_a=reply_a, reply_a_status=reply_a_status, reply_v3=reply_v3, reply_v3_status=reply_v3_status,
                strategy_enabled=decision.strategy_enabled,
                strategy_goal=strategy.communicative_goal,
                strategy_mode=strategy.response_mode,
                strategy_depth=strategy.response_depth,
                strategy_key_focus=strategy.key_focus,
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
