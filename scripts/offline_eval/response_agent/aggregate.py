"""
Phase M5F — aggregate metrics over a list of `CaseEvaluationResult`.

Every rate is zero-division-safe: computed only over the subset of
cases where the denominator concept actually applies (e.g.
refinement_success_rate is only over cases where refinement was
actually triggered), and is `None` — never a fabricated 0.0 or 1.0 --
when that subset is empty. An empty overall result list returns an
all-`None`/zero `AggregateReport`, never raises.

Per-scenario breakdown is always included so a weak subgroup can never
hide behind one blended aggregate average.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


@dataclass(frozen=True)
class ScenarioBreakdown:
    scenario_type: str
    case_count: int
    deterministic_pass_rate: float | None
    failure_tags: tuple[str, ...]


@dataclass(frozen=True)
class AggregateReport:
    total_cases: int
    deterministic_pass_rate: float | None
    routing_accuracy: float | None
    grounding_accuracy: float | None
    unsupported_claim_rate: float | None
    pii_leak_rate: float | None
    internal_leak_rate: float | None
    validation_pass_rate: float | None
    refinement_trigger_rate: float | None
    refinement_success_rate: float | None
    refinement_regression_count: int
    refinement_regression_rate: float | None
    no_response_expected_count: int
    no_response_accuracy: float | None
    per_scenario: tuple[ScenarioBreakdown, ...]
    failure_bucket_counts: dict = field(default_factory=dict)


def build_aggregate_report(results) -> AggregateReport:
    results = list(results)
    n = len(results)
    if n == 0:
        return AggregateReport(
            total_cases=0,
            deterministic_pass_rate=None,
            routing_accuracy=None,
            grounding_accuracy=None,
            unsupported_claim_rate=None,
            pii_leak_rate=None,
            internal_leak_rate=None,
            validation_pass_rate=None,
            refinement_trigger_rate=None,
            refinement_success_rate=None,
            refinement_regression_count=0,
            refinement_regression_rate=None,
            no_response_expected_count=0,
            no_response_accuracy=None,
            per_scenario=(),
            failure_bucket_counts={},
        )

    deterministic_passed = sum(1 for r in results if r.deterministic_checks_passed)
    routing_correct = sum(1 for r in results if r.actual_route == r.expected_route)
    grounding_correct = sum(1 for r in results if r.actual_grounding_mode == r.expected_grounding_mode)
    unsupported_claim_hits = sum(1 for r in results if r.final_failure_code == "unsupported_operational_claim")
    pii_hits = sum(1 for r in results if r.final_failure_code == "pii_in_draft")
    internal_leak_hits = sum(1 for r in results if r.final_failure_code == "internal_leak")
    validation_passed = sum(1 for r in results if r.validation_passed)

    refinement_triggered = [r for r in results if r.refinement_attempted]
    refinement_succeeded = [r for r in refinement_triggered if r.validation_passed]
    refinement_regressed = [
        r for r in refinement_triggered
        if not r.validation_passed and r.final_failure_code is not None and r.final_failure_code != r.initial_failure_code
    ]

    no_response_expected = [r for r in results if r.expected_generation_status == "SKIPPED"]
    no_response_correct = [r for r in no_response_expected if r.actual_generation_status == "SKIPPED"]

    failure_bucket_counts = Counter(tag for r in results for tag in r.failure_tags)

    by_scenario: dict[str, list] = defaultdict(list)
    for r in results:
        by_scenario[r.scenario_type].append(r)
    per_scenario = tuple(
        ScenarioBreakdown(
            scenario_type=scenario,
            case_count=len(group),
            deterministic_pass_rate=_safe_rate(sum(1 for r in group if r.deterministic_checks_passed), len(group)),
            failure_tags=tuple(sorted({tag for r in group for tag in r.failure_tags})),
        )
        for scenario, group in sorted(by_scenario.items())
    )

    return AggregateReport(
        total_cases=n,
        deterministic_pass_rate=_safe_rate(deterministic_passed, n),
        routing_accuracy=_safe_rate(routing_correct, n),
        grounding_accuracy=_safe_rate(grounding_correct, n),
        unsupported_claim_rate=_safe_rate(unsupported_claim_hits, n),
        pii_leak_rate=_safe_rate(pii_hits, n),
        internal_leak_rate=_safe_rate(internal_leak_hits, n),
        validation_pass_rate=_safe_rate(validation_passed, n),
        refinement_trigger_rate=_safe_rate(len(refinement_triggered), n),
        refinement_success_rate=_safe_rate(len(refinement_succeeded), len(refinement_triggered)),
        refinement_regression_count=len(refinement_regressed),
        refinement_regression_rate=_safe_rate(len(refinement_regressed), len(refinement_triggered)),
        no_response_expected_count=len(no_response_expected),
        no_response_accuracy=_safe_rate(len(no_response_correct), len(no_response_expected)),
        per_scenario=per_scenario,
        failure_bucket_counts=dict(failure_bucket_counts),
    )
