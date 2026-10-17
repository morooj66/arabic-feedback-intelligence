"""
Phase M5F — OPTIONAL, opt-in live-Gemini evaluation harness.

OFFLINE EVALUATION — NOT PRODUCTION DATA. Still fully isolated from
review_items/review_sessions/audit tables: this script only calls the
REAL Gemini API through the same `SupportsGenerate` interface, driven
by the same deterministic case dataset and fake retrievers as
run_evaluation.py. Nothing here writes to any database.

Policy (per M5F instruction):
  - disabled by default -- requires RUN_LIVE_GEMINI_EVAL=1
  - no secrets committed -- reads settings.GEMINI_API_KEY the same way
    rcs.response_generation.service.build_live_gemini_client() already
    does; never hardcodes a key
  - case count is bounded (LIVE_EVAL_CASE_LIMIT, default 5) to bound cost
  - output is explicitly labeled non-deterministic (real Gemini output
    is not guaranteed reproducible even with fixed inputs)
  - never part of pytest collection: this file's name matches neither
    `test_*.py` nor `*_test.py` (the same isolation mechanism already
    used for tests/e5_smoke_check.py in M5D.2/M5B.3)

Run:
    RUN_LIVE_GEMINI_EVAL=1 .venv/bin/python scripts/offline_eval/response_agent/run_live_optional.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if os.environ.get("RUN_LIVE_GEMINI_EVAL") != "1":
        print(
            "SKIPPED by default: set RUN_LIVE_GEMINI_EVAL=1 to run a bounded, opt-in "
            "evaluation against the real Gemini API. Output would be NON-DETERMINISTIC "
            "and is never used for the primary M5F deterministic evaluation."
        )
        return

    from config import settings
    from rcs.response_generation.service import build_live_gemini_client
    from scripts.offline_eval.response_agent.dataset import CASES
    from scripts.offline_eval.response_agent.fakes import FakeApprovedResponseMemoryRetriever, FakeOfficialKnowledgeRetriever
    from rcs.response_intelligence.orchestrator import run_agent_pipeline
    from scripts.offline_eval.response_agent.runner import _build_request

    if not settings.GEMINI_API_KEY:
        print("GEMINI_API_KEY is not configured -- cannot run the live evaluation.", file=sys.stderr)
        return

    limit = int(os.environ.get("LIVE_EVAL_CASE_LIMIT", "5"))
    # Only cases with real scripted-for-fake responses make sense to
    # skip here -- the live client generates its OWN text, so only
    # cases that don't depend on a SPECIFIC scripted draft are useful;
    # this script deliberately runs only the eligible-for-draft cases,
    # bounded to `limit`.
    bounded_cases = [c for c in CASES if c.gemini_responses][:limit]
    client = build_live_gemini_client()

    print(f"OFFLINE EVALUATION -- NOT PRODUCTION DATA. Live Gemini, {len(bounded_cases)} case(s), NON-DETERMINISTIC output.")
    for case in bounded_cases:
        request = _build_request(case)
        exec_result = run_agent_pipeline(
            request, client, 0,
            official_knowledge_retriever=FakeOfficialKnowledgeRetriever(case.official_result),
            approved_response_retriever=FakeApprovedResponseMemoryRetriever(case.approved_result),
        )
        print(f"  {case.case_id}: status={exec_result.generation_result.generation_status} "
              f"route={exec_result.generation_result.eligibility_rule_id}")


if __name__ == "__main__":
    main()
