"""
Response Agent — single-comment demonstration wrapper.

Thin CLI for showing the CURRENT Response Agent's proposed response for
ONE typed-in comment during a live/technical demonstration. Introduces
NO new Agent logic of its own: it imports and calls the exact same
functions scripts/review_agent_replies.py already uses --
`classify_and_route` (real frozen classification + real, unmodified
eligibility routing) and `generate_reply` (the real
`run_agent_pipeline`, unchanged) -- so this is the SAME execution path
as the multi-comment CLI and as production, never a second Gemini call
made directly or a separate prompt.

This is not a chatbot: it takes exactly one comment, runs it through
the pipeline once, prints the result, and exits. There is no
conversation state, no follow-up turn, no memory between invocations.

Prints only: the (redacted) comment text, classification, eligibility,
the proposed response, and non-secret trace fields (prompt version,
model name, whether Official Knowledge / Approved Response Memory /
refinement were used, and the validation outcome). Never prints an API
key, a raw prompt, a connection string, or anything from `.env`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from rcs.response_generation.prompt import redact_for_generation  # noqa: E402
from rcs.response_generation.service import build_live_gemini_client  # noqa: E402
from rcs.response_intelligence.orchestrator import AgentExecutionResult  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402
from scripts.review_agent_replies import (  # noqa: E402
    ClassificationAndRouting,
    SourceRow,
    classify_and_route,
    generate_reply,
)

_DEMO_COMMENT_ID = "single-comment-demo"


def run_single_comment(comment_text: str) -> tuple[ClassificationAndRouting, AgentExecutionResult | None]:
    """The exact same two real calls scripts/review_agent_replies.py
    makes per row -- reused, not reimplemented. Returns (routing,
    exec_result); exec_result is None when the comment was not routed
    to GENERATE_DRAFT (Gemini is never called in that case, identical
    to the multi-comment CLI and to production)."""
    from rcs.response_generation.eligibility import GENERATE_DRAFT

    routing = classify_and_route(comment_text)
    exec_result: AgentExecutionResult | None = None
    if routing.routing_decision == GENERATE_DRAFT:
        row = SourceRow(
            comment_id=_DEMO_COMMENT_ID,
            source_row_index=0,
            original_comment=comment_text,
            original_title=None,
            entry_id=None,
        )
        client = build_live_gemini_client()
        exec_result = generate_reply(row, routing.classification, client)
    return routing, exec_result


def _print_report(comment_text: str, routing: ClassificationAndRouting, exec_result: AgentExecutionResult | None) -> None:
    c = routing.classification
    print("Response Agent")
    print("--------------")
    print("Comment:")
    print(redact_for_generation(comment_text) or "(فارغ)")
    print()
    print("Classification:")
    print(f"Intent: {c.primary_intent}")
    print(f"Topic: {c.topic_subtype}")
    print(f"Sentiment: {c.sentiment}")
    print()
    print("Eligibility:")
    print(f"{routing.routing_decision} ({routing.reason_code})")
    print()
    print("Proposed response:")
    if exec_result is None:
        print("(لا يوجد رد — التعليق غير مؤهل لإنشاء رد، ولم يتم استدعاء Gemini)")
    else:
        gr = exec_result.generation_result
        print(gr.suggested_response or "(لم يُنتَج رد صالح)")
    print()
    print("Trace:")
    if exec_result is None:
        print(f"Prompt version: {settings.RESPONSE_GENERATION_PROMPT_VERSION} (not used for this comment)")
        print("Knowledge used: N/A (not eligible)")
        print("Refinement used: N/A (not eligible)")
        print("Validation: N/A (not eligible)")
    else:
        gr = exec_result.generation_result
        meta = exec_result.metadata
        print(f"Prompt version: {gr.prompt_version}")
        print(f"Model: {gr.model_name}")
        knowledge_bits = []
        if meta.official_grounding_available:
            knowledge_bits.append("Official Knowledge")
        if meta.style_result_count > 0:
            knowledge_bits.append("Approved Response Memory")
        print(f"Knowledge used: {', '.join(knowledge_bits) if knowledge_bits else 'none'}")
        print(f"Refinement used: {'yes' if meta.refinement_attempted else 'no'}")
        print(f"Validation: {'PASS' if gr.generation_status == 'SUCCEEDED' else f'FAIL ({gr.reason_code})'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the current Response Agent's proposed reply for one comment (demo/inspection only)."
    )
    parser.add_argument("--comment", type=str, required=True, help="The visitor comment text to run through the Agent.")
    args = parser.parse_args()

    load_dotenv_if_present()
    routing, exec_result = run_single_comment(args.comment)
    _print_report(args.comment, routing, exec_result)


if __name__ == "__main__":
    main()
