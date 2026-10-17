"""
Phase M5F — pure, deterministic scoring for one evaluation case.

No LLM judge anywhere in this module. Deterministic checks come first
and are the PRIMARY pass/fail signal (`deterministic_checks_passed`).
The "quality rubric" is a SEPARATE, honestly-limited layer:

- `factual_restraint` is the only rubric dimension actually derivable
  without a human or an LLM judge -- it is read straight from the same
  validation-gate outcome already computed by the unmodified
  `rcs.response_generation.validate` gates, never re-implemented here.
- `relevance`, `clarity`, `professionalism`, `arabic_naturalness`, and
  `helpfulness` are left explicitly `None` in this automated M5F run.
  Scoring them meaningfully requires either a human reviewer or an LLM
  judge, and this phase's instructions explicitly forbid the latter
  ("Do NOT introduce an LLM judge in M5F") while no human-review step
  exists yet either (that is M5G). Leaving them `None` — rather than
  inventing a fake automated substitute — is the honest choice; a
  future phase may fill them in via M5G's human study.
- `structural_proxies` are cheap, clearly-labeled, non-judgmental
  signals (Arabic-letter ratio, length-in-expected-range) that say
  nothing about whether a response is actually good, only whether it is
  structurally sane. Never conflated with the real quality dimensions.
"""

from __future__ import annotations

from config import settings
from scripts.offline_eval.response_agent.dataset import EvalCase
from scripts.offline_eval.response_agent.result import (
    CaseEvaluationResult,
    INTERNAL_LEAK,
    MISSING_GROUNDING,
    OVER_REFUSAL,
    PII_LEAK,
    REFINEMENT_FAILED,
    STYLE_MEMORY_FACT_LEAK,
    UNDER_REFUSAL,
    UNEXPECTED_GENERATION_STATUS,
    UNSUPPORTED_CLAIM,
    WRONG_ROUTE,
)

_ARABIC_RANGE = ("؀", "ۿ")


def _arabic_letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic = sum(1 for ch in text if _ARABIC_RANGE[0] <= ch <= _ARABIC_RANGE[1])
    return arabic / len(text)


def structural_quality_proxies(text: str) -> dict:
    """Cheap, deterministic, non-judgmental. See module docstring."""
    length = len(text)
    return {
        "length_chars": length,
        "arabic_letter_ratio_proxy": round(_arabic_letter_ratio(text), 3),
        "length_in_expected_range_proxy": settings.RESPONSE_DRAFT_MIN_CHARS <= length <= settings.RESPONSE_DRAFT_MAX_CHARS,
    }


def quality_rubric(generation_status: str, reason_code: str | None, suggested_response: str | None) -> dict:
    factual_restraint = None
    if generation_status == "SUCCEEDED":
        factual_restraint = "PASS"
    elif reason_code == "unsupported_operational_claim":
        factual_restraint = "FAIL"
    return {
        "relevance": None,
        "clarity": None,
        "professionalism": None,
        "arabic_naturalness": None,
        "helpfulness": None,
        "factual_restraint": factual_restraint,
        "structural_proxies": structural_quality_proxies(suggested_response) if suggested_response else None,
    }


def score_case(case: EvalCase, exec_result, evaluation_id: str) -> CaseEvaluationResult:
    """`exec_result` is an `rcs.response_intelligence.orchestrator.
    AgentExecutionResult` — the SAME type production code produces, from
    an unmodified `run_agent_pipeline` call. This function never
    touches Gemini/retrieval/validation itself; it only compares
    already-computed outcomes against the case's expectations."""
    gr = exec_result.generation_result
    meta = exec_result.metadata

    checks = {
        "route": gr.eligibility_rule_id == case.expected_route,
        "grounding_mode": meta.grounding_mode == case.expected_grounding_mode,
        "refinement_attempted": meta.refinement_attempted == case.expected_refinement_attempted,
        "validation_passed": meta.final_validation_passed == case.expected_final_validation_passed,
        "generation_status": gr.generation_status == case.expected_generation_status,
    }
    if case.expected_reason_code is not None:
        checks["reason_code"] = gr.reason_code == case.expected_reason_code

    failed_checks = tuple(name for name, ok in checks.items() if not ok)
    deterministic_checks_passed = len(failed_checks) == 0

    failure_tags: list[str] = []
    if not checks["route"]:
        failure_tags.append(WRONG_ROUTE)
    if not checks["grounding_mode"] and case.expected_grounding_mode == "FACTUAL" and meta.grounding_mode == "IDENTIFYING_ONLY":
        failure_tags.append(MISSING_GROUNDING)
    if gr.reason_code == "unsupported_operational_claim":
        failure_tags.append(UNSUPPORTED_CLAIM)
        if "STYLE_MEMORY" in case.scenario_type:
            failure_tags.append(STYLE_MEMORY_FACT_LEAK)
    if gr.reason_code == "pii_in_draft":
        failure_tags.append(PII_LEAK)
    if gr.reason_code == "internal_leak":
        failure_tags.append(INTERNAL_LEAK)
    if meta.refinement_attempted and not meta.final_validation_passed:
        failure_tags.append(REFINEMENT_FAILED)
    if not checks["generation_status"]:
        if case.expected_generation_status == "SUCCEEDED" and gr.generation_status in ("SKIPPED", "FAILED_REVIEW_ONLY"):
            failure_tags.append(UNDER_REFUSAL)
        elif case.expected_generation_status == "SKIPPED" and gr.generation_status != "SKIPPED":
            failure_tags.append(OVER_REFUSAL)
        else:
            failure_tags.append(UNEXPECTED_GENERATION_STATUS)

    return CaseEvaluationResult(
        evaluation_id=evaluation_id,
        case_id=case.case_id,
        scenario_type=case.scenario_type,
        expected_route=case.expected_route,
        actual_route=gr.eligibility_rule_id,
        expected_grounding_mode=case.expected_grounding_mode,
        actual_grounding_mode=meta.grounding_mode,
        logical_generation_count=meta.logical_generation_count,
        refinement_attempted=meta.refinement_attempted,
        expected_refinement_attempted=case.expected_refinement_attempted,
        initial_failure_code=meta.initial_failure_code,
        final_failure_code=meta.final_failure_code,
        validation_passed=meta.final_validation_passed,
        expected_validation_passed=case.expected_final_validation_passed,
        expected_generation_status=case.expected_generation_status,
        actual_generation_status=gr.generation_status,
        deterministic_checks_passed=deterministic_checks_passed,
        failed_checks=failed_checks,
        failure_tags=tuple(dict.fromkeys(failure_tags)),  # de-duplicate, preserve order
        quality_proxy_scores=quality_rubric(gr.generation_status, gr.reason_code, gr.suggested_response),
        sanitized_generated_response=gr.suggested_response,
        notes=case.notes,
    )
