"""
Developer CLI — human review of the REAL Response Agent output, one
visitor comment at a time, against a real source Excel workbook.

EVALUATION ONLY. Reuses existing production functions end to end and
introduces NO new decision logic:

  1. rcs.ingest.loader.load_raw()                          -- read Excel (pure, never writes back)
  2. rcs.response_intelligence.inspector.inspect_classification() -- the configured classification runtime
  3. rcs.response_intelligence.inspector.inspect_eligibility()    -- the real, unmodified eligibility rules
  4. rcs.response_intelligence.orchestrator.run_agent_pipeline()  -- the real Agent Layer
     (retrieval, evidence assembly, grounding enforcement, Gemini draft,
     validation gates, bounded refinement) -- called directly, not
     re-implemented, and with NO audit_repository (so zero audit
     persistence is even possible; see orchestrator.py's own
     `audit_repository: AgentAuditRepository | None = None` default).

Safety, by construction (see the module docstring items this file was
built against):
  - This script imports NOTHING from rcs.review.repositories,
    rcs.review.repository, or rcs.review.service -- there is no name in
    this file's import graph capable of writing to PostgreSQL, Supabase,
    or MongoDB, claiming an assignment, or mutating review_items. This
    is a structural guarantee, not a runtime check (see
    tests/test_review_agent_replies.py's static import-graph test).
  - The source workbook is opened only via pandas.read_excel (inside
    load_raw) and never written back to.
  - comment_id values used for generation are namespaced "cli-eval-row{N}"
    so they can never collide with, or be mistaken for, a real
    review_items.comment_id.
  - visitor_name / visitor_email are dropped from the working DataFrame
    immediately after loading and are never read, printed, or stored
    again anywhere in this file.
  - Comment text shown on screen and written to the JSONL is passed
    through rcs.response_generation.prompt.redact_for_generation (the
    project's existing phone/email redaction) -- an ADDITIONAL layer on
    top of (not instead of) the orchestrator's own internal redaction of
    prompt/query text.
  - The only file this script ever writes is one new line per reviewed
    row, appended to a fresh JSONL file under
    data/processed/agent_reply_review/.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Callable, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402
from rcs.response_generation.eligibility import GENERATE_DRAFT  # noqa: E402
from rcs.response_generation.prompt import redact_for_generation  # noqa: E402
from rcs.response_generation.service import GenerationRequest, SupportsGenerate, build_live_gemini_client  # noqa: E402
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata  # noqa: E402
from rcs.response_intelligence.inspector import (  # noqa: E402
    ClassificationInspection,
    inspect_classification,
    inspect_eligibility,
)
from rcs.response_intelligence.orchestrator import AgentExecutionResult, run_agent_pipeline  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402

OUTPUT_DIR = _ROOT / "data" / "processed" / "agent_reply_review"

_COMMENT_ID_PREFIX = "cli-eval-row"

# Columns this script will ever read from the loaded DataFrame. Anything
# else (visitor_name, visitor_email) is dropped immediately in
# `_load_rows` and never referenced again below.
_KEEP_COLUMNS = [
    "comment_id",
    "source_row_index",
    "original_comment",
    "original_title",
    "entry_id",
]


@dataclasses.dataclass(frozen=True)
class SourceRow:
    comment_id: str
    source_row_index: int
    original_comment: str
    original_title: Optional[str]
    entry_id: Optional[str]


def _load_rows(path: Path, start_row: int, limit: Optional[int], rows_arg: Optional[list[int]] = None) -> list[SourceRow]:
    """Read-only. `load_raw` (rcs.ingest.loader) is the exact production
    Excel loader -- never reimplemented here. visitor_name/visitor_email
    are dropped immediately; nothing downstream of this function can see
    them.

    `rows_arg`, when given, selects exactly those 0-based row positions
    (in the given order is not preserved -- callers wanting a specific
    order should sort `rows_arg` themselves) instead of the contiguous
    `start_row`/`limit` slice -- added so a deterministic, category-
    diverse sample can be selected across the whole file (a single
    contiguous slice cannot do this for a large, heterogeneous source
    file). `start_row`/`limit` are ignored when `rows_arg` is given."""
    df = load_raw(path, settings.SOURCE_COLUMNS)
    df = df[_KEEP_COLUMNS]
    if rows_arg is not None:
        df = df.iloc[rows_arg]
    else:
        df = df.iloc[start_row:]
        if limit is not None:
            df = df.iloc[:limit]
    rows = []
    for record in df.to_dict("records"):
        comment_text = record["original_comment"]
        if comment_text is None or (isinstance(comment_text, float)):
            comment_text = ""
        rows.append(
            SourceRow(
                comment_id=str(record["comment_id"]),
                source_row_index=int(record["source_row_index"]),
                original_comment=str(comment_text),
                original_title=(str(record["original_title"]) if record["original_title"] not in (None, "") and not isinstance(record["original_title"], float) else None),
                entry_id=(str(record["entry_id"]) if record["entry_id"] not in (None, "") and not isinstance(record["entry_id"], float) else None),
            )
        )
    return rows


@dataclasses.dataclass(frozen=True)
class ClassificationAndRouting:
    classification: ClassificationInspection
    routing_decision: str
    reason_code: str
    rule_id: str


ClassifierFn = Callable[[str], ClassificationInspection]


def classify_and_route(comment_text: str, classifier: ClassifierFn = inspect_classification) -> ClassificationAndRouting:
    """Steps 2-3. `classifier` is injectable for tests only (default is
    the real `inspect_classification`, i.e. the configured runtime with
    ``legacy_rcs`` remaining the default) --
    `inspect_eligibility` below is always the real, unmodified
    `determine_response_eligibility`, never faked."""
    classification = classifier(comment_text)
    eligibility = inspect_eligibility(
        comment_text=comment_text,
        primary_intent=classification.primary_intent,
        topic_subtype=classification.topic_subtype,
        sentiment=classification.sentiment,
        confidence=classification.primary_intent_confidence,
    )
    return ClassificationAndRouting(
        classification=classification,
        routing_decision=eligibility.routing_decision,
        reason_code=eligibility.reason_code,
        rule_id=eligibility.rule_id,
    )


def generate_reply(row: SourceRow, classification: ClassificationInspection, client: SupportsGenerate) -> AgentExecutionResult:
    """Step 4. Calls the REAL, unmodified `run_agent_pipeline` directly
    -- the real default retrievers, real evidence assembly/grounding
    enforcement, real validation gates, real bounded refinement. No
    `audit_repository` is ever passed, so no audit trace is ever
    persisted anywhere by this call.

    Phase M6.2: supplies structured `identifying_metadata`; whether it
    is actually used is decided by the orchestrator's Context/Trust
    Policies step, identical to production."""
    request = GenerationRequest(
        comment_id=f"{_COMMENT_ID_PREFIX}{row.source_row_index}",
        comment_text=row.original_comment,
        primary_intent=classification.primary_intent,
        topic_subtype=classification.topic_subtype,
        sentiment=classification.sentiment,
        confidence=classification.primary_intent_confidence,
        identifying_metadata=IdentifyingMetadata(row.original_title, row.entry_id),
    )
    return run_agent_pipeline(request, client=client, max_retries=None)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def append_evaluation_row(out_path: Path, row: dict) -> None:
    """The ONLY file write this script ever performs. Plain JSONL
    append -- one new line, never a rewrite, never a delete."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_evaluation_row(
    row: SourceRow,
    routing: ClassificationAndRouting,
    exec_result: Optional[AgentExecutionResult],
    human_judgment: str,
    edited_response: Optional[str] = None,
) -> dict:
    """Only the fields explicitly allowed. No visitor identity, no raw
    prompt, no raw Gemini response, no chain-of-thought, no secrets.

    Production login/DB audit follow-up (Response Agent verification
    hardening): every provenance field below is read directly off
    values `run_agent_pipeline` already computed for THIS execution
    (GenerationResult.prompt_version/model_name/provider,
    AgentExecutionMetadata.official_grounding_available/style_result_count/
    refinement_attempted) -- nothing here is invented or hardcoded.
    `generation_prompt_version`/`generation_model_name` in particular
    come from `config.settings.RESPONSE_GENERATION_PROMPT_VERSION`/
    `RESPONSE_GENERATION_MODEL_NAME` exactly as service.py's real
    `generate_once_with_draft` already reads them for production -- this
    script never sets or overrides either.

    `edited_response` (only when the human judgment is EDIT) is stored
    ALONGSIDE `generated_reply`, never in place of it -- the original
    Agent output is always preserved verbatim."""
    sanitized_comment = redact_for_generation(row.original_comment)
    payload = {
        "comment_id": row.comment_id,
        "source_row_index": row.source_row_index,
        "sanitized_comment": sanitized_comment,
        "intent": routing.classification.primary_intent,
        "topic": routing.classification.topic_subtype,
        "sentiment": routing.classification.sentiment,
        "routing_decision": routing.routing_decision,
        "routing_reason_code": routing.reason_code,
        "generated_reply": None,
        "generation_status": None,
        "validation_reason_code": None,
        "grounding_mode": None,
        "official_knowledge_used": None,
        "approved_response_memory_used": None,
        "refinement_attempted": None,
        "generation_provider": None,
        "generation_model_name": None,
        "generation_prompt_version": None,
        "generated_at": None,
        "human_judgment": human_judgment,
        "edited_response": redact_for_generation(edited_response) if edited_response else None,
        "reviewed_at": _now_iso(),
    }
    if exec_result is not None:
        gr = exec_result.generation_result
        meta = exec_result.metadata
        payload["generated_reply"] = redact_for_generation(gr.suggested_response) if gr.suggested_response else None
        payload["generation_status"] = gr.generation_status
        payload["validation_reason_code"] = gr.reason_code
        payload["grounding_mode"] = meta.grounding_mode
        # Official Knowledge: the metadata flag the orchestrator itself
        # uses to decide whether official evidence actually entered the
        # prompt (see orchestrator.py's grounding-enforcement docstring)
        # -- NOT just "retrieval returned something", which could still
        # be excluded from grounding.
        payload["official_knowledge_used"] = meta.official_grounding_available
        # Approved Response Memory: only ever assembled into
        # style_reference_texts, never into official grounding (see
        # orchestrator.py) -- non-empty style_result_count is the exact
        # condition under which those texts were actually built.
        payload["approved_response_memory_used"] = meta.style_result_count > 0
        payload["refinement_attempted"] = meta.refinement_attempted
        payload["generation_provider"] = gr.provider
        payload["generation_model_name"] = gr.model_name
        payload["generation_prompt_version"] = gr.prompt_version
        payload["generated_at"] = gr.generated_at
    return payload


def _print_header(index: int, total: int, row: SourceRow) -> None:
    print(f"\nComment {index} / {total}")
    print("Visitor comment:")
    print(redact_for_generation(row.original_comment) or "(فارغ)")


def _print_classification(routing: ClassificationAndRouting) -> None:
    c = routing.classification
    print(f"Intent: {c.primary_intent}")
    print(f"Topic: {c.topic_subtype}")
    print(f"Sentiment: {c.sentiment}")
    print(f"Route: {routing.routing_decision} ({routing.reason_code})")


def _print_generation(exec_result: AgentExecutionResult) -> None:
    gr = exec_result.generation_result
    print("Agent reply:")
    print(gr.suggested_response or "(لا يوجد رد)")
    print(f"Validation: {'PASS' if gr.generation_status == 'SUCCEEDED' else 'FAIL'}")
    print(f"Grounding: {exec_result.metadata.grounding_mode}")
    print(f"Refinement: {'YES' if exec_result.metadata.refinement_attempted else 'NO'}")


def _prompt_decision(prompt_fn: Callable[[str], str]) -> str:
    while True:
        raw = prompt_fn("[A] Approve as-is   [E] Edit   [R] Reject   [S] Skip   [Q] Quit\n> ").strip().upper()
        if raw in ("A", "E", "R", "S", "Q"):
            return raw
        print("الرجاء اختيار A أو E أو R أو S أو Q.")


_DECISION_LABEL = {"A": "APPROVE_AS_IS", "E": "EDIT", "R": "REJECT", "S": "SKIP"}


def process_row(
    row: SourceRow,
    index: int,
    total: int,
    client: SupportsGenerate,
    out_path: Path,
    classifier: ClassifierFn = inspect_classification,
    prompt_fn: Optional[Callable[[str], str]] = input,
) -> Optional[str]:
    """Processes exactly one row end to end. Returns the raw decision
    ("A"/"E"/"R"/"S"/"Q") the human chose. Only "A"/"E"/"R"/"S" ever
    cause an append to `out_path`; "Q" never does. "E" additionally
    prompts for the human-edited response text, which is stored
    alongside (never in place of) the original Agent output.

    `prompt_fn=None` runs this row NON-interactively: classification,
    eligibility, and (when eligible) the real Gemini generation still
    run exactly as above, but no decision is prompted for and no
    "A"/"E"/"R"/"S" value is invented -- the appended row's
    `human_judgment` is left `None` (blank), for a human to fill in
    later via their own review of the artifact. Returns `None`."""
    _print_header(index, total, row)
    routing = classify_and_route(row.original_comment, classifier=classifier)
    _print_classification(routing)

    exec_result: Optional[AgentExecutionResult] = None
    if routing.routing_decision == GENERATE_DRAFT:
        exec_result = generate_reply(row, routing.classification, client)
        _print_generation(exec_result)
    else:
        print("غير مؤهل للرد — لن يتم استدعاء Gemini.")

    if prompt_fn is None:
        evaluation_row = build_evaluation_row(row, routing, exec_result, None, None)
        append_evaluation_row(out_path, evaluation_row)
        return None

    decision = _prompt_decision(prompt_fn)
    if decision == "Q":
        return decision

    edited_response = None
    if decision == "E":
        edited_response = prompt_fn("أدخل الرد بعد التعديل:\n> ").strip()

    evaluation_row = build_evaluation_row(row, routing, exec_result, _DECISION_LABEL[decision], edited_response)
    append_evaluation_row(out_path, evaluation_row)
    return decision


def _default_output_path() -> Path:
    """Dated (not timestamped-to-the-second) and, crucially, versioned
    from the ACTUAL current `config.settings.RESPONSE_GENERATION_
    PROMPT_VERSION` -- never a hardcoded "rg-v13" string, so the
    filename never claims a prompt version that wasn't genuinely active
    for this run. A run-to-run rename risk (same day, same prompt
    version, invoked twice) is deliberately left to the caller passing
    an explicit `--out` -- this default optimizes for "one clearly
    dated, clearly versioned file per verification session", not for
    silently never colliding."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d")
    version_slug = (settings.RESPONSE_GENERATION_PROMPT_VERSION or "unknown-prompt-version").strip().replace("-", "_")
    return OUTPUT_DIR / f"{version_slug}_human_verification_{stamp}.jsonl"


def run_cli(
    file: Path,
    limit: Optional[int],
    start_row: int,
    client: Optional[SupportsGenerate] = None,
    out_path: Optional[Path] = None,
    classifier: ClassifierFn = inspect_classification,
    prompt_fn: Callable[[str], str] = input,
    rows: Optional[list[int]] = None,
    interactive: bool = True,
) -> Path:
    """`interactive=False` (CLI: `--auto`) generates and logs every
    selected row without ever calling `prompt_fn` -- no decision is
    prompted for or invented; `human_judgment` is left blank in every
    appended row. Use this to produce a fresh, dated generation record
    for later human inspection (e.g. via the resulting workbook) without
    requiring a live person to sit through the whole batch. Interactive
    per-row review (A/E/R/S/Q) remains available and unchanged when
    `interactive=True` (the default)."""
    load_dotenv_if_present()
    resolved_client = client if client is not None else build_live_gemini_client()
    resolved_out = out_path if out_path is not None else _default_output_path()

    loaded_rows = _load_rows(file, start_row, limit, rows_arg=rows)
    total = len(loaded_rows)
    print(f"Loaded {total} row(s) from {file.name} (read-only; nothing will be written back to it).")
    print(f"Evaluation output: {resolved_out}")

    for i, row in enumerate(loaded_rows, start=1):
        decision = process_row(
            row, i, total, resolved_client, resolved_out,
            classifier=classifier, prompt_fn=prompt_fn if interactive else None,
        )
        if decision == "Q":
            print("تم إنهاء الجلسة.")
            break

    return resolved_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Human review of the real Response Agent output (evaluation only).")
    parser.add_argument("--file", type=Path, required=True, help="Path to the source visitor-comments Excel workbook.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to review.")
    parser.add_argument("--start-row", type=int, default=0, help="0-based row offset to start from.")
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help="Comma-separated 0-based row indices to load explicitly (overrides --start-row/--limit). "
        "Use this for a deterministic, non-contiguous sample instead of a single slice.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path. Default: data/processed/agent_reply_review/"
        "<current-prompt-version>_human_verification_<date>.jsonl",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive: generate and log every row without prompting for a decision. "
        "human_judgment is left blank in every row for later manual review.",
    )
    args = parser.parse_args()
    rows = [int(x.strip()) for x in args.rows.split(",")] if args.rows else None
    run_cli(args.file, args.limit, args.start_row, out_path=args.out, rows=rows, interactive=not args.auto)


if __name__ == "__main__":
    main()
