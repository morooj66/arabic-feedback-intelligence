"""
Phase M5F — writes the three offline evaluation artifacts. Every
artifact is explicitly labeled "OFFLINE EVALUATION — NOT PRODUCTION
DATA" so it can never be mistaken for a production export.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

_LABEL = "OFFLINE EVALUATION — NOT PRODUCTION DATA"

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _result_to_dict(result) -> dict:
    return asdict(result)


def write_jsonl_results(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_label": _LABEL}, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps(_result_to_dict(r), ensure_ascii=False) + "\n")


def write_summary_json(aggregate_report, run_metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_label": _LABEL,
        "run_metadata": run_metadata,
        "aggregate": asdict(aggregate_report),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def write_markdown_report(aggregate_report, run_metadata: dict, path: Path) -> None:
    lines = [
        f"# {_LABEL}",
        "",
        "## M5F Offline Response Agent Evaluation",
        "",
        f"- Dataset version: `{run_metadata['dataset_version']}`",
        f"- Run timestamp (UTC): `{run_metadata['run_timestamp_utc']}`",
        f"- Model: `{run_metadata['model_name']}` (provider `{run_metadata['provider']}`)",
        f"- Prompt version: `{run_metadata['prompt_version']}`",
        f"- Official Knowledge source configured: `{run_metadata['retrieval_configuration']['official_knowledge_source_configured']}`",
        f"- Approved Response allowlist size: `{run_metadata['retrieval_configuration']['approved_response_allowlist_size']}`",
        "",
        "## Aggregate results",
        "",
        f"- Total cases: **{aggregate_report.total_cases}**",
        f"- Deterministic pass rate: **{_pct(aggregate_report.deterministic_pass_rate)}**",
        f"- Routing accuracy: {_pct(aggregate_report.routing_accuracy)}",
        f"- Grounding accuracy: {_pct(aggregate_report.grounding_accuracy)}",
        f"- Unsupported-claim rate: {_pct(aggregate_report.unsupported_claim_rate)}",
        f"- PII leakage rate: {_pct(aggregate_report.pii_leak_rate)}",
        f"- Internal-leak rate: {_pct(aggregate_report.internal_leak_rate)}",
        f"- Validation pass rate: {_pct(aggregate_report.validation_pass_rate)}",
        f"- Refinement trigger rate: {_pct(aggregate_report.refinement_trigger_rate)}",
        f"- Refinement success rate (of triggered): {_pct(aggregate_report.refinement_success_rate)}",
        f"- Refinement regression count: {aggregate_report.refinement_regression_count} "
        f"(rate of triggered: {_pct(aggregate_report.refinement_regression_rate)})",
        f"- No-response accuracy: {_pct(aggregate_report.no_response_accuracy)} "
        f"(n={aggregate_report.no_response_expected_count})",
        "",
        "## Failure buckets",
        "",
    ]
    if aggregate_report.failure_bucket_counts:
        for tag, count in sorted(aggregate_report.failure_bucket_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{tag}`: {count}")
    else:
        lines.append("- (none)")
    lines += ["", "## Per-scenario breakdown", ""]
    lines.append("| Scenario | Cases | Deterministic pass rate | Failure tags |")
    lines.append("|---|---|---|---|")
    for sb in aggregate_report.per_scenario:
        tags = ", ".join(f"`{t}`" for t in sb.failure_tags) or "-"
        lines.append(f"| {sb.scenario_type} | {sb.case_count} | {_pct(sb.deterministic_pass_rate)} | {tags} |")
    lines += [
        "",
        "## Known limitations",
        "",
        "- Quality rubric dimensions (relevance, clarity, professionalism, Arabic "
        "naturalness, helpfulness) are NOT scored in this automated run — no LLM judge "
        "was used (per instruction) and no human review has occurred yet (M5G). Only "
        "`factual_restraint` is derived deterministically from the existing validation gates.",
        "- Official Knowledge / Approved Response Memory retrieval metrics (Recall@K, MRR) "
        "are computed against a small controlled FIXTURE corpus, not a real production "
        "corpus — no Official Knowledge source or Approved Response allowlist is "
        "configured today.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
