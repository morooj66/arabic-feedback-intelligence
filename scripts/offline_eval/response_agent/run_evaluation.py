"""
Phase M5F — offline evaluation entrypoint.

OFFLINE EVALUATION — NOT PRODUCTION DATA.

Run:
    .venv/bin/python scripts/offline_eval/response_agent/run_evaluation.py

Uses only deterministic fake Gemini clients/retrievers (see fakes.py) —
zero network, zero database, zero mutation of any production table.
Writes three artifacts under scripts/offline_eval/response_agent/results/:
  - m5f_results.jsonl   (one sanitized CaseEvaluationResult per line)
  - m5f_summary.json    (aggregate report + run metadata)
  - m5f_report.md       (human-readable summary)
"""

from __future__ import annotations

from scripts.offline_eval.response_agent.aggregate import build_aggregate_report
from scripts.offline_eval.response_agent.dataset import CASES
from scripts.offline_eval.response_agent.fakes import InMemoryAuditRepository
from scripts.offline_eval.response_agent.report_writer import (
    RESULTS_DIR,
    write_jsonl_results,
    write_markdown_report,
    write_summary_json,
)
from scripts.offline_eval.response_agent.reproducibility import build_run_metadata
from scripts.offline_eval.response_agent.retrieval_fixtures import (
    ARM_QUERIES,
    OK_QUERIES,
    rank_approved_response_memory_fixture,
    rank_official_knowledge_fixture,
)
from scripts.offline_eval.response_agent.retrieval_metrics import compute_aggregate_metrics
from scripts.offline_eval.response_agent.runner import run_ablation_suite, run_dataset


def _official_knowledge_ablation_template():
    return next(c for c in CASES if c.case_id == "EC05_INQUIRY_WITH_OFFICIAL_GROUNDING")


def main() -> None:
    print("Phase M5F offline evaluation -- OFFLINE EVALUATION, NOT PRODUCTION DATA")

    audit_repository = InMemoryAuditRepository()
    results = run_dataset(CASES, audit_repository=audit_repository)
    aggregate_report = build_aggregate_report(results)

    print(f"\n{len(CASES)} cases evaluated.")
    print(f"Deterministic pass rate: {aggregate_report.deterministic_pass_rate:.3f}")
    print(f"Routing accuracy: {aggregate_report.routing_accuracy:.3f}")
    print(f"Grounding accuracy: {aggregate_report.grounding_accuracy:.3f}")

    print("\n--- Ablation suite (EC05 template) ---")
    ablation_results = run_ablation_suite(_official_knowledge_ablation_template())
    for label, result in ablation_results.items():
        print(f"  {label}: route={result.actual_route} grounding={result.actual_grounding_mode} "
              f"validation_passed={result.validation_passed}")

    print("\n--- Retrieval metrics (controlled fixture corpora, NOT production data) ---")
    ok_rankings = [rank_official_knowledge_fixture(q) for _, q, _ in OK_QUERIES]
    ok_relevant = [rel for _, _, rel in OK_QUERIES]
    ok_metrics = compute_aggregate_metrics(ok_rankings, ok_relevant)
    print(f"  Official Knowledge (fixture): Recall@1={ok_metrics.recall_at_1} MRR={ok_metrics.mrr}")

    arm_rankings = [rank_approved_response_memory_fixture(q) for _, q, _ in ARM_QUERIES]
    arm_relevant = [rel for _, _, rel in ARM_QUERIES]
    arm_metrics = compute_aggregate_metrics(arm_rankings, arm_relevant)
    print(f"  Approved Response Memory (fixture): Recall@1={arm_metrics.recall_at_1} MRR={arm_metrics.mrr}")

    # Trace correlation proof (M5E schema, in-memory only -- never production trace rows).
    example_case_id = CASES[0].case_id
    traces = audit_repository.list_for_review_item(example_case_id)
    print(f"\nTrace correlation check: {len(traces)} in-memory trace(s) for {example_case_id!r} "
          f"(grounding_mode={traces[0].grounding_mode if traces else 'n/a'})")

    run_metadata = build_run_metadata()
    write_jsonl_results(results, RESULTS_DIR / "m5f_results.jsonl")
    write_summary_json(aggregate_report, run_metadata, RESULTS_DIR / "m5f_summary.json")
    write_markdown_report(aggregate_report, run_metadata, RESULTS_DIR / "m5f_report.md")
    print(f"\nWrote artifacts to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
