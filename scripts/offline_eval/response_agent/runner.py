"""
Phase M5F — the offline evaluation runner.

Calls the REAL, unmodified `rcs.response_intelligence.orchestrator.
run_agent_pipeline` for every case — no agent redesign, no new
retrieval/refinement behavior, exactly the production code path,
driven entirely by dependency-injected fakes (retrievers, Gemini
client, audit repository). Zero network. Zero database.
"""

from __future__ import annotations

import uuid

from rcs.response_generation.service import GenerationRequest
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
)
from scripts.offline_eval.response_agent.dataset import EvalCase, _EMPTY_ARM, _EMPTY_OK
from scripts.offline_eval.response_agent.fakes import (
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
    FakeOfficialKnowledgeRetriever,
)
from scripts.offline_eval.response_agent.result import CaseEvaluationResult
from scripts.offline_eval.response_agent.scoring import score_case


def _build_request(case: EvalCase) -> GenerationRequest:
    # Phase M6.2: structured metadata (case.identifying_metadata) is what
    # actually reaches the orchestrator now -- case.identifying_context
    # (the old pre-rendered field) is left untouched on EvalCase for
    # whatever else references it, but is no longer read here.
    return GenerationRequest(
        comment_id=case.case_id,
        comment_text=case.comment_text,
        primary_intent=case.primary_intent,
        topic_subtype=case.topic_subtype,
        sentiment=case.sentiment,
        insufficient_context=case.insufficient_context,
        identifying_metadata=case.identifying_metadata,
    )


def run_case(case: EvalCase, *, audit_repository=None) -> CaseEvaluationResult:
    request = _build_request(case)
    client = FakeGeminiClient(case.gemini_responses) if case.gemini_responses else None
    exec_result = run_agent_pipeline(
        request,
        client,
        0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(case.official_result),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(case.approved_result),
        audit_repository=audit_repository,
    )
    evaluation_id = f"eval-{uuid.uuid4().hex}"
    return score_case(case, exec_result, evaluation_id)


def run_dataset(cases, *, audit_repository=None) -> list[CaseEvaluationResult]:
    return [run_case(case, audit_repository=audit_repository) for case in cases]


# --- Ablation suite ----------------------------------------------------------

ABLATION_NONE = "no_retrieval_context"
ABLATION_OFFICIAL_ONLY = "official_knowledge_only"
ABLATION_STYLE_ONLY = "style_memory_only"
ABLATION_BOTH = "official_and_style"


def run_ablation_suite(template_case: EvalCase) -> dict[str, CaseEvaluationResult]:
    """Runs the SAME comment/scripted-response template under four
    controlled retrieval-context ablations, using dependency injection
    only — production code is never modified to produce these modes."""
    official_present = template_case.official_result
    approved_present = template_case.approved_result
    if official_present.outcome != SUCCESS_WITH_RESULTS:
        raise ValueError("ablation template case must have a real Official Knowledge result configured")

    variants = {
        ABLATION_NONE: (_EMPTY_OK, _EMPTY_ARM),
        ABLATION_OFFICIAL_ONLY: (official_present, _EMPTY_ARM),
        ABLATION_STYLE_ONLY: (_EMPTY_OK, approved_present if approved_present.outcome == SUCCESS_WITH_RESULTS else _EMPTY_ARM),
        ABLATION_BOTH: (official_present, approved_present if approved_present.outcome == SUCCESS_WITH_RESULTS else _EMPTY_ARM),
    }

    results = {}
    for label, (ok_result, arm_result) in variants.items():
        variant_case = EvalCase(
            case_id=f"{template_case.case_id}__{label}",
            scenario_type=f"ABLATION_{label.upper()}",
            provenance=template_case.provenance,
            comment_text=template_case.comment_text,
            primary_intent=template_case.primary_intent,
            topic_subtype=template_case.topic_subtype,
            sentiment=template_case.sentiment,
            identifying_context=template_case.identifying_context,
            identifying_metadata=template_case.identifying_metadata,
            official_result=ok_result,
            approved_result=arm_result,
            gemini_responses=template_case.gemini_responses,
            expected_route=(
                template_case.expected_route if ok_result.outcome == SUCCESS_WITH_RESULTS
                else "inquiry_without_trusted_grounding"
            ),
            expected_grounding_mode="FACTUAL" if ok_result.outcome == SUCCESS_WITH_RESULTS else "IDENTIFYING_ONLY",
            # Without real Official Knowledge, the scripted claim is
            # unsupported -- a refinement-eligible failure code, so the
            # orchestrator WILL attempt exactly one refinement (which
            # repeats the same claim here, since only one response is
            # scripted, and therefore still fails).
            expected_refinement_attempted=(ok_result.outcome != SUCCESS_WITH_RESULTS),
            expected_final_validation_passed=(ok_result.outcome == SUCCESS_WITH_RESULTS),
            expected_generation_status="SUCCEEDED" if ok_result.outcome == SUCCESS_WITH_RESULTS else "FAILED_REVIEW_ONLY",
            expected_reason_code=None if ok_result.outcome == SUCCESS_WITH_RESULTS else "unsupported_operational_claim",
            notes=f"Ablation variant '{label}' of {template_case.case_id}.",
        )
        results[label] = run_case(variant_case)
    return results
