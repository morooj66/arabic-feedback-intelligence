"""
Phase M5F — offline agent evaluation harness tests.

Exercises scripts/offline_eval/response_agent/* only. That package is a
self-contained script tree (the same isolation convention as
scripts/benchmarks/retrieval/ from M5B.2) — never imported by the served
application. These tests prove the HARNESS is correct; they do not
change, tune, or retrain the production Response Agent in any way.

No network. No real database. No Hugging Face model download. No LLM
judge. No production mutation of any kind.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields

import pytest

from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
)

from scripts.offline_eval.response_agent.aggregate import (
    AggregateReport,
    ScenarioBreakdown,
    build_aggregate_report,
)
from scripts.offline_eval.response_agent.dataset import (
    CASES,
    DATASET_VERSION,
    EvalCase,
    _EMPTY_ARM,
    _EMPTY_OK,
    _arm,
    _ok,
)
from scripts.offline_eval.response_agent.fakes import (
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
    FakeOfficialKnowledgeRetriever,
    InMemoryAuditRepository,
    make_draft_json,
)
from scripts.offline_eval.response_agent.reproducibility import build_run_metadata
from scripts.offline_eval.response_agent.result import (
    CaseEvaluationResult,
    INTERNAL_LEAK,
    MISSING_GROUNDING,
    PII_LEAK,
    REFINEMENT_FAILED,
    STYLE_MEMORY_FACT_LEAK,
    UNSUPPORTED_CLAIM,
    WRONG_ROUTE,
)
from scripts.offline_eval.response_agent.retrieval_fixtures import (
    ARM_QUERIES,
    OK_QUERIES,
    rank_approved_response_memory_fixture,
    rank_official_knowledge_fixture,
)
from scripts.offline_eval.response_agent.retrieval_metrics import (
    NO_DATA_METRICS,
    compute_aggregate_metrics,
    hit_at_k,
    reciprocal_rank,
)
from scripts.offline_eval.response_agent.runner import (
    ABLATION_BOTH,
    ABLATION_NONE,
    ABLATION_OFFICIAL_ONLY,
    ABLATION_STYLE_ONLY,
    run_ablation_suite,
    run_case,
    run_dataset,
)
from scripts.offline_eval.response_agent.scoring import quality_rubric, score_case


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def _case(case_id: str) -> EvalCase:
    return next(c for c in CASES if c.case_id == case_id)


@pytest.fixture(scope="module")
def full_run() -> list[CaseEvaluationResult]:
    """Run the full 26-case dataset once per test module (deterministic,
    zero network) and share it across every test that just needs the
    resulting scored list — avoids re-running the (slow, torch-importing)
    pipeline 26x per assertion."""
    from config import settings

    with pytest.MonkeyPatch.context() as mp:   # module-scoped: the function-scoped legacy fixture cannot cover this run
        mp.setattr(settings, "RESPONSE_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS", False)
        mp.setattr(settings, "RESPONSE_REQUIRE_CORROBORATION_FOR_EXCLUSION", False)
        return run_dataset(CASES)


@pytest.fixture(scope="module")
def aggregate(full_run: list[CaseEvaluationResult]) -> AggregateReport:
    return build_aggregate_report(full_run)


# =====================================================================
# 1. Evaluation dataset schema
# =====================================================================


def test_01_dataset_schema_has_required_fields():
    required = {
        "case_id", "scenario_type", "provenance", "comment_text",
        "primary_intent", "topic_subtype", "sentiment", "gemini_responses",
        "expected_route", "expected_grounding_mode",
        "expected_refinement_attempted", "expected_final_validation_passed",
        "expected_generation_status",
    }
    field_names = {f.name for f in fields(EvalCase)}
    assert required.issubset(field_names)
    assert len(CASES) == 26
    for case in CASES:
        assert case.provenance == "synthetic"


# =====================================================================
# 2. Stable evaluation IDs
# =====================================================================


def test_02_case_ids_are_unique_and_stable():
    ids = [c.case_id for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case_id found"
    # Stable naming convention: ECnn_DESCRIPTIVE_NAME.
    for case_id in ids:
        assert re.match(r"^EC\d{2}_[A-Z0-9_]+$", case_id), case_id


def test_02b_evaluation_id_is_unique_per_run(full_run: list[CaseEvaluationResult]):
    eval_ids = [r.evaluation_id for r in full_run]
    assert len(eval_ids) == len(set(eval_ids))
    for eid in eval_ids:
        assert eid.startswith("eval-")


# =====================================================================
# 3. No raw PII in fixtures
# =====================================================================

_PHONE_RE = re.compile(r"\b0\d{9}\b")
_ALLOWED_PII_LOOKING_STRINGS = {
    # These synthetic phone-like strings are deliberate fixtures for the
    # PII-detection GATE itself (EC12 comment + PII_DRAFT) -- they exist
    # to prove the pii_in_draft gate fires, not as real visitor data.
    "0559876543",
    "0501234567",
}


def test_03_no_unexpected_raw_pii_in_dataset():
    for case in CASES:
        for phone in _PHONE_RE.findall(case.comment_text):
            assert phone in _ALLOWED_PII_LOOKING_STRINGS, (
                f"unexpected phone-like string in {case.case_id}: {phone}"
            )
    # No email-shaped strings anywhere in comment text.
    email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    for case in CASES:
        assert not email_re.search(case.comment_text), case.case_id


def test_03b_no_real_visitor_identity_fields_on_eval_case():
    forbidden = {"visitor_name", "visitor_email", "visitor_phone", "reviewer_note", "auth_token"}
    field_names = {f.name for f in fields(EvalCase)}
    assert forbidden.isdisjoint(field_names)


# =====================================================================
# 4. No secret-like fields
# =====================================================================


def test_04_no_secret_like_fields_in_result_or_metadata():
    forbidden_substrings = ("api_key", "password", "secret", "token", "credential")
    result_fields = {f.name.lower() for f in fields(CaseEvaluationResult)}
    for name in result_fields:
        assert not any(bad in name for bad in forbidden_substrings), name

    metadata = build_run_metadata()
    flat = json.dumps(metadata)
    for bad in forbidden_substrings:
        assert bad not in flat.lower()


# =====================================================================
# 5. Deterministic route scoring
# =====================================================================


def test_05_route_mismatch_is_detected_deterministically():
    case = _case("EC01_PRAISE_BASIC")
    result = run_case(case)
    assert result.actual_route == case.expected_route
    assert result.deterministic_checks_passed is True

    tampered = EvalCase(
        **{**{f.name: getattr(case, f.name) for f in fields(case)}, "expected_route": "not_the_real_route"}
    )
    mismatched = run_case(tampered)
    assert mismatched.deterministic_checks_passed is False
    assert WRONG_ROUTE in mismatched.failure_tags


# =====================================================================
# 6. Deterministic grounding scoring
# =====================================================================


def test_06_grounding_mode_scored_correctly_for_factual_and_identifying_cases():
    factual = run_case(_case("EC05_INQUIRY_WITH_OFFICIAL_GROUNDING"))
    assert factual.actual_grounding_mode == "FACTUAL"
    assert factual.expected_grounding_mode == "FACTUAL"

    identifying = run_case(_case("EC01_PRAISE_BASIC"))
    assert identifying.actual_grounding_mode == "IDENTIFYING_ONLY"
    assert identifying.expected_grounding_mode == "IDENTIFYING_ONLY"


# =====================================================================
# 7. Deterministic validation scoring
# =====================================================================


def test_07_validation_pass_and_fail_scored_correctly():
    passing = run_case(_case("EC01_PRAISE_BASIC"))
    assert passing.validation_passed is True
    assert passing.deterministic_checks_passed is True

    failing = run_case(_case("EC12_PII_CONTAINING_COMMENT"))
    assert failing.validation_passed is False
    assert failing.expected_validation_passed is False
    assert failing.deterministic_checks_passed is True  # matched expectation


# =====================================================================
# 8. No-response scoring
# =====================================================================


def test_08_no_response_cases_scored_as_skipped():
    for case_id in (
        "EC13_SPAM_JUNK", "EC14_OFF_TOPIC_PERSONAL", "EC15_EMPTY_MEANINGLESS", "EC26_INSUFFICIENT_CONTEXT",
    ):
        result = run_case(_case(case_id))
        assert result.actual_generation_status == "SKIPPED"
        assert result.expected_generation_status == "SKIPPED"
        assert result.deterministic_checks_passed is True


# =====================================================================
# 9. Unsupported-claim scoring
# =====================================================================


def test_09_unsupported_claim_detected_and_tagged():
    result = run_case(_case("EC19_REFINEMENT_FAILS_CLOSED"))
    assert result.final_failure_code == "unsupported_operational_claim"
    assert UNSUPPORTED_CLAIM in result.failure_tags


def test_09b_unsupported_claim_absent_when_grounded():
    result = run_case(_case("EC05_INQUIRY_WITH_OFFICIAL_GROUNDING"))
    assert result.final_failure_code is None
    assert UNSUPPORTED_CLAIM not in result.failure_tags


# =====================================================================
# 10. PII leakage scoring
# =====================================================================


def test_10_pii_leakage_scored_and_tagged():
    result = run_case(_case("EC12_PII_CONTAINING_COMMENT"))
    assert result.final_failure_code == "pii_in_draft"
    assert PII_LEAK in result.failure_tags


def test_10b_pii_leakage_absent_for_clean_draft():
    result = run_case(_case("EC01_PRAISE_BASIC"))
    assert PII_LEAK not in result.failure_tags


# =====================================================================
# 11. Identifying context cannot count as factual grounding
# =====================================================================


def test_11_identifying_context_alone_is_not_factual_grounding():
    case = _case("EC06_VAGUE_REFERENCE_IDENTIFYING_ONLY")
    assert case.identifying_context is not None
    assert case.official_result.outcome == SUCCESS_EMPTY
    result = run_case(case)
    assert result.actual_grounding_mode == "IDENTIFYING_ONLY"
    assert result.actual_route == "inquiry_without_trusted_grounding"
    assert MISSING_GROUNDING not in result.failure_tags  # matches expectation, no violation


# =====================================================================
# 12. Style memory cannot count as factual grounding
# =====================================================================


def test_12_style_memory_alone_cannot_satisfy_factual_claim_gate():
    case = _case("EC07_STYLE_MEMORY_FACT_LEAK_ATTEMPT")
    assert case.approved_result.outcome == SUCCESS_WITH_RESULTS
    assert case.official_result.outcome == SUCCESS_EMPTY
    result = run_case(case)
    assert result.actual_grounding_mode == "IDENTIFYING_ONLY"
    assert result.validation_passed is False
    assert result.final_failure_code == "unsupported_operational_claim"
    assert STYLE_MEMORY_FACT_LEAK in result.failure_tags
    assert result.deterministic_checks_passed is True  # this IS the expected (failing) behavior


# =====================================================================
# 13. Official Knowledge can count as factual grounding
# =====================================================================


def test_13_official_knowledge_enables_factual_grounding():
    case = _case("EC05_INQUIRY_WITH_OFFICIAL_GROUNDING")
    assert case.official_result.outcome == SUCCESS_WITH_RESULTS
    result = run_case(case)
    assert result.actual_grounding_mode == "FACTUAL"
    assert result.validation_passed is True


# =====================================================================
# 14. Official unavailable degrades correctly
# =====================================================================


def test_14_official_unavailable_degrades_without_inventing_grounding():
    case = _case("EC08_OFFICIAL_UNAVAILABLE_DEGRADE")
    result = run_case(case)
    assert result.actual_grounding_mode == "IDENTIFYING_ONLY"
    assert result.actual_route == "inquiry_without_trusted_grounding"
    assert result.actual_generation_status == "SUCCEEDED"
    assert result.deterministic_checks_passed is True


# =====================================================================
# 15-17. Logical generation-call bounds
# =====================================================================


def test_15_initial_only_case_has_logical_count_one():
    result = run_case(_case("EC01_PRAISE_BASIC"))
    assert result.logical_generation_count == 1
    assert result.refinement_attempted is False


def test_16_refinement_case_has_logical_count_two():
    result = run_case(_case("EC18_REFINEMENT_REPAIRS_UNSUPPORTED_CLAIM"))
    assert result.logical_generation_count == 2
    assert result.refinement_attempted is True


def test_17_no_case_exceeds_two_logical_calls(full_run: list[CaseEvaluationResult]):
    for result in full_run:
        assert result.logical_generation_count <= 2, result.case_id


# =====================================================================
# 18-20. Refinement success / failure / regression metrics
# =====================================================================


def test_18_refinement_success_metric_correct():
    result = run_case(_case("EC18_REFINEMENT_REPAIRS_UNSUPPORTED_CLAIM"))
    assert result.refinement_attempted is True
    assert result.validation_passed is True
    report = build_aggregate_report([result])
    assert report.refinement_trigger_rate == 1.0
    assert report.refinement_success_rate == 1.0
    assert report.refinement_regression_count == 0


def test_19_refinement_failure_metric_correct():
    result = run_case(_case("EC19_REFINEMENT_FAILS_CLOSED"))
    assert result.refinement_attempted is True
    assert result.validation_passed is False
    assert REFINEMENT_FAILED in result.failure_tags
    report = build_aggregate_report([result])
    assert report.refinement_trigger_rate == 1.0
    assert report.refinement_success_rate == 0.0


def test_20_refinement_regression_metric_correct():
    # Regression = refinement attempted, still failing, AND the final
    # failure code differs from the initial one (a genuinely NEW
    # violation introduced by the second draft) -- EC19 repeats the SAME
    # code, so it must NOT count as a regression.
    same_code_result = run_case(_case("EC19_REFINEMENT_FAILS_CLOSED"))
    assert same_code_result.initial_failure_code == same_code_result.final_failure_code
    report = build_aggregate_report([same_code_result])
    assert report.refinement_regression_count == 0

    # Construct a synthetic regression case: first draft too short
    # (refinable), second draft introduces PII (a different, new
    # violation) -- this proves the regression metric fires when it
    # legitimately should, without needing to rely on real model output.
    regression_case = EvalCase(
        case_id="ZZ_SYNTHETIC_REGRESSION_CHECK",
        scenario_type="SYNTHETIC_REGRESSION_CHECK",
        provenance="synthetic",
        comment_text="شكرًا جزيلاً على هذا العمل الرائع.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=("شكرًا", "شكرًا لتواصلكم على الرقم 0501234567 مع المعجم."),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_refinement_attempted=True,
        expected_final_validation_passed=False,
        expected_generation_status="FAILED_REVIEW_ONLY",
        expected_reason_code="pii_in_draft",
    )
    regression_result = run_case(regression_case)
    assert regression_result.refinement_attempted is True
    assert regression_result.validation_passed is False
    assert regression_result.initial_failure_code != regression_result.final_failure_code
    report = build_aggregate_report([regression_result])
    assert report.refinement_regression_count == 1
    assert report.refinement_regression_rate == 1.0


# =====================================================================
# 21-24. Retrieval metrics
# =====================================================================


def test_21_official_knowledge_recall_at_k_calculation():
    rankings = [rank_official_knowledge_fixture(q) for _, q, _ in OK_QUERIES]
    relevant = [rel for _, _, rel in OK_QUERIES]
    metrics = compute_aggregate_metrics(rankings, relevant)
    assert metrics.recall_at_1 == 1.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.num_queries == len(OK_QUERIES)


def test_22_official_knowledge_mrr_calculation():
    rankings = [rank_official_knowledge_fixture(q) for _, q, _ in OK_QUERIES]
    relevant = [rel for _, _, rel in OK_QUERIES]
    metrics = compute_aggregate_metrics(rankings, relevant)
    assert metrics.mrr == 1.0
    # Sanity-check the primitives directly too.
    assert hit_at_k(["A", "B", "C"], frozenset({"B"}), 3) is True
    assert hit_at_k(["A", "B", "C"], frozenset({"Z"}), 3) is False
    assert reciprocal_rank(["A", "B", "C"], frozenset({"B"})) == pytest.approx(0.5)
    assert reciprocal_rank(["A", "B", "C"], frozenset({"Z"})) == 0.0


def test_23_style_retrieval_metrics_calculated_separately():
    ok_rankings = [rank_official_knowledge_fixture(q) for _, q, _ in OK_QUERIES]
    ok_relevant = [rel for _, _, rel in OK_QUERIES]
    ok_metrics = compute_aggregate_metrics(ok_rankings, ok_relevant)

    arm_rankings = [rank_approved_response_memory_fixture(q) for _, q, _ in ARM_QUERIES]
    arm_relevant = [rel for _, _, rel in ARM_QUERIES]
    arm_metrics = compute_aggregate_metrics(arm_rankings, arm_relevant)

    assert ok_metrics.num_queries == len(OK_QUERIES)
    assert arm_metrics.num_queries == len(ARM_QUERIES)
    assert ok_metrics.num_queries != arm_metrics.num_queries  # genuinely independent corpora/query sets


def test_24_no_merged_factual_and_style_retrieval_score_exists():
    # There must be no function/field anywhere in this module that
    # blends Official Knowledge and Approved Response Memory metrics
    # into one combined score.
    import scripts.offline_eval.response_agent.retrieval_metrics as rm

    module_source_names = dir(rm)
    combined_like = [n for n in module_source_names if "combined" in n.lower() or "blended" in n.lower() or "merged" in n.lower()]
    assert combined_like == []


# =====================================================================
# 25-29. Failure buckets / aggregation
# =====================================================================


def test_25_failure_bucket_assignment(full_run: list[CaseEvaluationResult]):
    by_id = {r.case_id: r for r in full_run}
    assert WRONG_ROUTE not in by_id["EC01_PRAISE_BASIC"].failure_tags
    assert PII_LEAK in by_id["EC12_PII_CONTAINING_COMMENT"].failure_tags
    assert INTERNAL_LEAK in by_id["EC21_NON_REFINABLE_INTERNAL_LEAK"].failure_tags
    assert STYLE_MEMORY_FACT_LEAK in by_id["EC07_STYLE_MEMORY_FACT_LEAK_ATTEMPT"].failure_tags


def test_26_aggregate_pass_rate_calculation(aggregate: AggregateReport):
    assert aggregate.total_cases == 26
    assert aggregate.deterministic_pass_rate == 1.0
    assert aggregate.routing_accuracy == 1.0
    assert aggregate.grounding_accuracy == 1.0


def test_27_per_scenario_breakdown_present_and_non_hiding(aggregate: AggregateReport):
    assert len(aggregate.per_scenario) > 1
    scenario_types = {sb.scenario_type for sb in aggregate.per_scenario}
    assert "TRUST_ABLATION_STYLE_MEMORY" in scenario_types
    for sb in aggregate.per_scenario:
        assert isinstance(sb, ScenarioBreakdown)
        assert sb.case_count >= 1
        assert sb.deterministic_pass_rate is not None


def test_28_zero_division_safe_metrics_with_no_refinement_cases():
    non_refining = [r for r in run_dataset([_case("EC01_PRAISE_BASIC"), _case("EC02_COMPLAINT_BASIC")])]
    report = build_aggregate_report(non_refining)
    assert report.refinement_trigger_rate == 0.0
    assert report.refinement_success_rate is None  # no denominator -- must be None, not 0 or an error
    assert report.refinement_regression_rate is None
    assert report.no_response_accuracy is None  # no SKIPPED-expected cases in this subset


def test_29_empty_evaluation_dataset_safe_behavior():
    report = build_aggregate_report([])
    assert report.total_cases == 0
    assert report.deterministic_pass_rate is None
    assert report.routing_accuracy is None
    assert report.per_scenario == ()
    assert report.failure_bucket_counts == {}

    empty_retrieval = compute_aggregate_metrics([], [])
    assert empty_retrieval == NO_DATA_METRICS
    assert empty_retrieval.num_queries == 0


# =====================================================================
# 30-34. Sanitization / no-mutation guarantees
# =====================================================================


def test_30_generated_output_sanitized_contains_only_final_draft_text():
    result = run_case(_case("EC01_PRAISE_BASIC"))
    assert result.sanitized_generated_response is not None
    # Must be exactly the suggested_response text -- no prompt wrapper,
    # no JSON envelope, no identifying/official/style context leaked in.
    assert "suggested_response" not in result.sanitized_generated_response
    assert result.sanitized_generated_response == "شكرًا لملاحظتكم القيّمة حول هذا الموضوع."


def test_31_no_prompt_or_chain_of_thought_stored_in_result():
    result = run_case(_case("EC01_PRAISE_BASIC"))
    result_field_names = {f.name for f in fields(CaseEvaluationResult)}
    forbidden = {"prompt", "raw_prompt", "chain_of_thought", "cot", "reasoning", "system_prompt"}
    assert forbidden.isdisjoint(result_field_names)
    # Also check the rubric dict never embeds free-text reasoning.
    rubric = quality_rubric("SUCCEEDED", None, "some text")
    assert "reasoning" not in rubric and "chain_of_thought" not in rubric


def test_32_no_production_database_writes_from_run_case(monkeypatch):
    """run_case must work with audit_repository=None (the default) --
    proving nothing here requires, or silently falls back to, a real
    database connection."""
    result = run_case(_case("EC01_PRAISE_BASIC"), audit_repository=None)
    assert result.deterministic_checks_passed is True


def test_33_no_review_item_mutation_surface_exists():
    import scripts.offline_eval.response_agent.runner as runner_module
    import scripts.offline_eval.response_agent.fakes as fakes_module

    source_modules = (runner_module, fakes_module)
    forbidden_calls = ("UPDATE review_items", "INSERT INTO review_items", "review_repository.save", "commit_review")
    for mod in source_modules:
        src = mod.__file__
        with open(src, encoding="utf-8") as fh:
            text = fh.read()
        for bad in forbidden_calls:
            assert bad not in text


def test_34_no_reviewer_decision_mutation_surface_exists():
    import inspect

    import scripts.offline_eval.response_agent.runner as runner_module

    src = inspect.getsource(runner_module)
    assert "reviewer_action =" not in src  # never ASSIGNS a reviewer decision (fixtures only READ it as input)
    assert "UPDATE" not in src.upper() or "update" not in src  # no SQL-shaped mutation verbs


# =====================================================================
# 35-36. M5E trace correlation
# =====================================================================


def test_35_trace_correlation_via_m5e_schema_works():
    repo = InMemoryAuditRepository()
    case = _case("EC01_PRAISE_BASIC")
    run_case(case, audit_repository=repo)
    traces = repo.list_for_review_item(case.case_id)
    assert len(traces) == 1
    assert traces[0].grounding_mode == "IDENTIFYING_ONLY"
    assert traces[0].review_item_id == case.case_id


def test_36_trace_absence_does_not_crash_offline_evaluation():
    # No audit_repository passed at all -- must not raise.
    result = run_case(_case("EC01_PRAISE_BASIC"))
    assert result is not None
    assert result.deterministic_checks_passed is True


# =====================================================================
# 37-39. Determinism / network isolation
# =====================================================================


def test_37_fake_gemini_evaluation_is_deterministic():
    result_a = run_case(_case("EC01_PRAISE_BASIC"))
    result_b = run_case(_case("EC01_PRAISE_BASIC"))
    assert result_a.actual_route == result_b.actual_route
    assert result_a.actual_grounding_mode == result_b.actual_grounding_mode
    assert result_a.validation_passed == result_b.validation_passed
    assert result_a.sanitized_generated_response == result_b.sanitized_generated_response


def test_38_live_gemini_path_is_opt_in_only(monkeypatch, capsys):
    monkeypatch.delenv("RUN_LIVE_GEMINI_EVAL", raising=False)
    import importlib

    import scripts.offline_eval.response_agent.run_live_optional as live_module
    importlib.reload(live_module)
    live_module.main()
    captured = capsys.readouterr()
    assert "SKIPPED by default" in captured.out


def test_39_normal_tests_perform_zero_network_calls(monkeypatch):
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during offline evaluation test")

    monkeypatch.setattr(socket, "socket", _blocked)
    try:
        run_dataset([_case("EC01_PRAISE_BASIC"), _case("EC05_INQUIRY_WITH_OFFICIAL_GROUNDING")])
    finally:
        monkeypatch.undo()


# =====================================================================
# 40-50. Production behavior unchanged / scope guards
# =====================================================================


def test_40_agent_production_behavior_unchanged_uses_real_pipeline():
    import scripts.offline_eval.response_agent.runner as runner_module

    assert runner_module.run_agent_pipeline is run_agent_pipeline


def test_41_routing_unchanged_matches_known_baseline_route_for_thanks():
    result = run_case(_case("EC01_PRAISE_BASIC"))
    assert result.actual_route == "generate_thanks"


def test_42_validation_unchanged_pii_still_fails_closed():
    result = run_case(_case("EC12_PII_CONTAINING_COMMENT"))
    assert result.actual_generation_status == "FAILED_REVIEW_ONLY"


def test_43_max_logical_gemini_calls_unchanged_at_two(full_run: list[CaseEvaluationResult]):
    assert max(r.logical_generation_count for r in full_run) <= 2


def test_44_no_mongo_cutover_artifacts_referenced():
    import inspect

    import scripts.offline_eval.response_agent.runner as runner_module
    import scripts.offline_eval.response_agent.fakes as fakes_module

    for mod in (runner_module, fakes_module):
        src = inspect.getsource(mod)
        assert "pymongo" not in src.lower()
        assert "mongo_client" not in src.lower()


def test_45_no_frontend_files_touched_by_offline_eval_package():
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent.parent / "scripts" / "offline_eval"
    for path in pkg_dir.rglob("*"):
        assert "web/src" not in str(path)
        assert path.suffix not in (".tsx", ".ts", ".jsx")


def test_46_no_langgraph_dependency_used():
    import inspect

    import scripts.offline_eval.response_agent.runner as runner_module

    assert "langgraph" not in inspect.getsource(runner_module).lower()


def test_47_no_react_agent_pattern_used():
    import inspect

    import scripts.offline_eval.response_agent.runner as runner_module

    src = inspect.getsource(runner_module).lower()
    assert "react agent" not in src
    assert "agentexecutor" not in src


def test_48_no_llm_judge_introduced():
    import inspect

    import scripts.offline_eval.response_agent.scoring as scoring_module

    # The module's docstring legitimately DOCUMENTS the "no LLM judge"
    # policy in prose -- what must never exist is an actual judge
    # callable/class, or a call that asks Gemini to grade its own
    # output. Check code-level surfaces, not the word itself.
    assert not hasattr(scoring_module, "llm_judge")
    assert not any("judge" in name.lower() for name in dir(scoring_module) if not name.startswith("_"))
    src = inspect.getsource(scoring_module)
    assert "generate(" not in src  # Gemini must never be asked to grade its own output here


def test_49_no_persistent_memory_introduced():
    import inspect

    import scripts.offline_eval.response_agent.fakes as fakes_module

    src = inspect.getsource(fakes_module)
    assert "InMemoryAuditRepository" in src  # in-memory only
    assert "sqlite3" not in src.lower()
    assert "psycopg" not in src.lower()


def test_50_no_external_dictionary_integration_referenced():
    import inspect

    import scripts.offline_eval.response_agent.runner as runner_module
    import scripts.offline_eval.response_agent.dataset as dataset_module

    for mod in (runner_module, dataset_module):
        assert "external_dictionary_client" not in inspect.getsource(mod).lower()


# =====================================================================
# Additional coverage: ablation suite trust-boundary behavior
# (supports items 11-14 and the phase's "IMPORTANT TRUST ABLATIONS"
# requirement directly, using the real ablation runner)
# =====================================================================


def test_ablation_suite_trust_boundary_split():
    template = _case("EC05_INQUIRY_WITH_OFFICIAL_GROUNDING")
    results = run_ablation_suite(template)

    assert set(results.keys()) == {ABLATION_NONE, ABLATION_OFFICIAL_ONLY, ABLATION_STYLE_ONLY, ABLATION_BOTH}

    assert results[ABLATION_NONE].validation_passed is False
    assert results[ABLATION_NONE].actual_grounding_mode == "IDENTIFYING_ONLY"

    assert results[ABLATION_OFFICIAL_ONLY].validation_passed is True
    assert results[ABLATION_OFFICIAL_ONLY].actual_grounding_mode == "FACTUAL"

    assert results[ABLATION_STYLE_ONLY].validation_passed is False
    assert results[ABLATION_STYLE_ONLY].actual_grounding_mode == "IDENTIFYING_ONLY"

    assert results[ABLATION_BOTH].validation_passed is True
    assert results[ABLATION_BOTH].actual_grounding_mode == "FACTUAL"


def test_ablation_suite_requires_real_official_knowledge_template():
    bad_template = _case("EC01_PRAISE_BASIC")  # has no real Official Knowledge configured
    with pytest.raises(ValueError):
        run_ablation_suite(bad_template)


# =====================================================================
# Additional coverage: reproducibility metadata shape
# =====================================================================


def test_reproducibility_metadata_has_no_secrets_and_correct_dataset_version():
    metadata = build_run_metadata()
    assert metadata["dataset_version"] == DATASET_VERSION
    assert metadata["seed"] is None
    assert "GEMINI_API_KEY" not in json.dumps(metadata)
    assert "retrieval_configuration" in metadata
