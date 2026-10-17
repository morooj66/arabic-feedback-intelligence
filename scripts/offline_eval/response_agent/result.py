"""
Phase M5F — the per-case offline evaluation result entity.

OFFLINE EVALUATION — NOT PRODUCTION DATA. `sanitized_generated_response`
is the ONLY content-shaped field here, and it holds nothing but the
already-validated draft text itself (never a prompt, never raw
identifying context/Official Knowledge/style text, never a reasoning
trace) — kept for human review of THIS offline artifact only, never
written to any production table (production's own `AgentExecutionTrace`
carries no content field at all — see rcs.response_intelligence.audit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Stable, machine-readable failure-bucket tags.
WRONG_ROUTE = "WRONG_ROUTE"
MISSING_GROUNDING = "MISSING_GROUNDING"
UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
OVER_REFUSAL = "OVER_REFUSAL"
UNDER_REFUSAL = "UNDER_REFUSAL"
PII_LEAK = "PII_LEAK"
INTERNAL_LEAK = "INTERNAL_LEAK"
POOR_RELEVANCE = "POOR_RELEVANCE"
STYLE_MEMORY_FACT_LEAK = "STYLE_MEMORY_FACT_LEAK"
REFINEMENT_FAILED = "REFINEMENT_FAILED"
RETRIEVAL_MISS = "RETRIEVAL_MISS"
UNEXPECTED_GENERATION_STATUS = "UNEXPECTED_GENERATION_STATUS"


@dataclass(frozen=True)
class CaseEvaluationResult:
    evaluation_id: str
    case_id: str
    scenario_type: str

    expected_route: str
    actual_route: str | None

    expected_grounding_mode: str
    actual_grounding_mode: str

    logical_generation_count: int
    refinement_attempted: bool
    expected_refinement_attempted: bool

    initial_failure_code: str | None
    final_failure_code: str | None

    validation_passed: bool
    expected_validation_passed: bool

    expected_generation_status: str
    actual_generation_status: str

    deterministic_checks_passed: bool
    failed_checks: tuple[str, ...] = field(default_factory=tuple)
    failure_tags: tuple[str, ...] = field(default_factory=tuple)

    quality_proxy_scores: dict = field(default_factory=dict)
    sanitized_generated_response: str | None = None

    notes: str = ""
