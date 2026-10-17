"""
Phase DEV-INSPECT — HTTP-shape adapter for the admin-only Agent
Inspector. Mirrors the existing `rcs.review.dto` separation: this
module only translates between plain JSON-able dicts and
`rcs.response_intelligence.inspector`'s typed functions — it contains
no business logic and no database access of its own. Every ADMIN-role
check happens in `http_api.py` before any function here is called,
exactly like every other admin-only route in that file.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from config import settings
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import build_live_gemini_client
from rcs.response_intelligence import inspector
from rcs.response_intelligence.orchestrator import AgentExecutionResult


def _generation_result_to_dict(result: GenerationResult) -> dict[str, Any]:
    payload = dataclasses.asdict(result)
    return payload


def inspector_architecture_payload() -> dict[str, Any]:
    """Static, read-only explainability content. Every name below is an
    exact identifier that exists in this repository today (see
    rcs.response_intelligence.orchestrator, .evidence, .state, .audit,
    rcs.response_generation.validate/refinement) — nothing here is
    invented."""
    return {
        "brain": {"name": "Gemini", "provider": settings.RESPONSE_GENERATION_PROVIDER, "model": settings.RESPONSE_GENERATION_MODEL_NAME},
        "controller": "rcs.response_intelligence.orchestrator.run_agent_pipeline (Response Agent Orchestrator)",
        "tools": [
            "Official Knowledge Retrieval (rcs.response_intelligence.retrieval.contracts.OfficialKnowledgeRetriever)",
            "Approved Response Memory Retrieval (rcs.response_intelligence.retrieval.contracts.ApprovedResponseMemoryRetriever)",
            "Validation Gates (rcs.response_generation.validate.run_validation_gates)",
            "Evidence/Grounding Assembly (rcs.response_intelligence.evidence.assemble_grounding / assemble_style_reference_texts)",
        ],
        "memory": {
            "kind": "Approved Response organizational/style memory only",
            "note": "No persistent conversational/user memory. Style texts never contribute to factual grounding (see evidence.py).",
        },
        "knowledge": {
            "factual_authority": "Official Knowledge (OfficialKnowledgeRetrievalResult at SUCCESS_WITH_RESULTS) is the ONLY factual authority.",
        },
        "state": "rcs.response_intelligence.state.ResponseAgentState (transient, per-call snapshot; never persisted)",
        "safety": [
            "PII redaction (rcs.response_generation.prompt.redact_for_generation)",
            "Validation gates (content_shape, internal_leak, pii, boilerplate, unsupported_claim)",
            "Factual grounding boundary (only Official Knowledge may satisfy the unsupported-claim gate)",
        ],
        "self_correction": {
            "max_refinement_attempts": 1,
            "max_logical_gemini_generations": 2,
            "note": "rcs.response_generation.refinement.is_refinement_eligible gates a single bounded refinement pass.",
        },
        "human_in_the_loop": "Final response remains under human reviewer control (SEND_AS_IS / EDIT_AND_SEND / NO_RESPONSE_NEEDED) — Gemini never sends.",
    }


def inspector_config_payload() -> dict[str, Any]:
    return {
        "provider": settings.RESPONSE_GENERATION_PROVIDER,
        "model_name": settings.RESPONSE_GENERATION_MODEL_NAME,
        "prompt_version": settings.RESPONSE_GENERATION_PROMPT_VERSION,
        "enabled_retrieval_tools": ["official_knowledge", "approved_response_memory"],
        "official_knowledge_max_results_per_query": settings.OFFICIAL_KNOWLEDGE_MAX_RESULTS_PER_QUERY,
        "approved_response_memory_max_results_per_query": settings.APPROVED_RESPONSE_MEMORY_MAX_RESULTS_PER_QUERY,
        "official_knowledge_source_configured": bool(settings.OFFICIAL_KNOWLEDGE_SOURCE_PATH),
        "approved_response_allowlist_size": len(settings.APPROVED_RESPONSE_SOURCE_ALLOWLIST),
        "memory_mode": "approved_response_style_only",
        "max_logical_generations": 2,
    }


def handle_inspector_classification(body: dict[str, Any]) -> dict[str, Any]:
    comment_text = str(body.get("comment_text") or "").strip()
    if not comment_text:
        raise ValueError("comment_text is required")
    result = inspector.inspect_classification(comment_text)
    return dataclasses.asdict(result)


def handle_inspector_eligibility(body: dict[str, Any]) -> dict[str, Any]:
    comment_text = str(body.get("comment_text") or "").strip()
    if not comment_text:
        raise ValueError("comment_text is required")
    result = inspector.inspect_eligibility(
        comment_text=comment_text,
        primary_intent=body.get("primary_intent"),
        topic_subtype=body.get("topic_subtype"),
        sentiment=body.get("sentiment"),
        confidence=body.get("confidence"),
        needs_review=bool(body.get("needs_review", False)),
        insufficient_context=bool(body.get("insufficient_context", False)),
        trusted_grounding_available=bool(body.get("trusted_grounding_available", False)),
    )
    return dataclasses.asdict(result)


def handle_inspector_retrieval(body: dict[str, Any]) -> dict[str, Any]:
    comment_text = str(body.get("comment_text") or "").strip()
    if not comment_text:
        raise ValueError("comment_text is required")
    result = inspector.inspect_retrieval(
        comment_text=comment_text,
        primary_intent=body.get("primary_intent"),
        topic_subtype=body.get("topic_subtype"),
        reveal_content=bool(body.get("reveal_content", False)),
    )
    return dataclasses.asdict(result)


def handle_inspector_grounding(body: dict[str, Any]) -> dict[str, Any]:
    import datetime as dt

    from rcs.response_intelligence.retrieval.contracts import SUCCESS_EMPTY, SUCCESS_WITH_RESULTS, OfficialKnowledgeRetrievalResult, OfficialKnowledgeResult

    if bool(body.get("official_knowledge_available", False)):
        official_result = OfficialKnowledgeRetrievalResult(
            outcome=SUCCESS_WITH_RESULTS,
            results=(
                OfficialKnowledgeResult(
                    source_id="dev-inspect",
                    source_title=str(body.get("official_source_title") or "test source"),
                    source_type="dictionary_entry",
                    content=str(body.get("official_content") or ""),
                    version="v1",
                    effective_from=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
                    authoritative=True,
                    retrieval_score=1.0,
                ),
            ),
        )
    else:
        official_result = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
    result = inspector.inspect_grounding(
        identifying_context=body.get("identifying_context"),
        official_result=official_result,
    )
    return dataclasses.asdict(result)


def handle_inspector_validation(body: dict[str, Any]) -> dict[str, Any]:
    draft_text = str(body.get("draft_text") or "")
    summary = inspector.inspect_validation(
        draft_text=draft_text,
        trusted_grounding_text=body.get("trusted_grounding_text"),
    )
    return {
        "passed": summary.passed,
        "first_failure": dataclasses.asdict(summary.first_failure) if summary.first_failure else None,
        "evaluated_gates": [dataclasses.asdict(g) for g in summary.evaluated_gates],
    }


def _agent_execution_result_to_dict(result: AgentExecutionResult) -> dict[str, Any]:
    trace = result.trace
    return {
        "generation_result": _generation_result_to_dict(result.generation_result),
        "metadata": dataclasses.asdict(result.metadata),
        "stages": {
            "identifying_context": trace.identifying_context if trace else None,
            "identifying_context_decision_reason": trace.identifying_context_decision_reason if trace else None,
            "official_knowledge_retrieval": dataclasses.asdict(trace.official_knowledge_result) if trace else None,
            "approved_response_memory_retrieval": {
                **dataclasses.asdict(trace.approved_response_result),
                "label": "STYLE ONLY -- never factual grounding",
            }
            if trace
            else None,
            "official_grounding_text": trace.official_grounding_text if trace else None,
            "style_reference_texts": list(trace.style_reference_texts) if trace else None,
            "draft_1": trace.initial_draft if trace else None,
            "draft_1_validation": {
                "passed": result.metadata.initial_validation_passed,
                "failure_code": result.metadata.initial_failure_code,
            },
            "refinement": {
                "attempted": result.metadata.refinement_attempted,
                "reason_code": result.metadata.initial_failure_code if result.metadata.refinement_attempted else None,
            },
            "draft_2": trace.refinement_draft if trace else None,
            "draft_2_validation": (
                {
                    "passed": result.metadata.final_validation_passed,
                    "failure_code": result.metadata.final_failure_code,
                }
                if result.metadata.refinement_attempted
                else None
            ),
        },
    }


def handle_inspector_full_run(body: dict[str, Any]) -> dict[str, Any]:
    comment_text = str(body.get("comment_text") or "").strip()
    if not comment_text:
        raise ValueError("comment_text is required")
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured -- the Inspector's full run calls the real Gemini API")
    client = build_live_gemini_client()
    result = inspector.run_full_inspection(
        comment_text=comment_text,
        primary_intent=body.get("primary_intent"),
        topic_subtype=body.get("topic_subtype"),
        sentiment=body.get("sentiment"),
        confidence=body.get("confidence"),
        insufficient_context=bool(body.get("insufficient_context", False)),
        # Phase M6.2: structured metadata only -- the admin supplies the
        # title/entry_id an admin-created test case would carry, and the
        # Response Agent's own Context/Trust Policies step decides
        # whether it is used. The body no longer accepts a pre-rendered
        # `identifying_context` string; there is no way to bypass the
        # policy via this endpoint.
        original_title=body.get("original_title"),
        entry_id=body.get("entry_id"),
        client=client,
    )
    return _agent_execution_result_to_dict(result)
