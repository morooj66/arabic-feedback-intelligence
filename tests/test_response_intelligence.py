"""
Phase M5A.1 — behavioral-parity tests for the response orchestration
foundation (rcs.response_intelligence).

This is a structural boundary, not an agent: generate_response_with_
orchestrator must produce byte-for-byte identical GenerationResult
fields (generated_at excepted, since it is a fresh timestamp per call —
see test_strong_parity_* below) to calling generate_suggested_response
directly, for every scenario. No RAG, no tools, no memory, no
refinement exist yet; these tests prove parity, not new capability.

Reuses the exact FakeGeminiClient/_draft_json fixtures already
established in test_response_generation.py rather than duplicating
them, per this session's own convention of importing shared test
doubles across files.
"""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from rcs.response_generation.service import GenerationRequest, generate_suggested_response
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from rcs.response_intelligence.orchestrator import generate_response_with_orchestrator
from rcs.response_intelligence.state import ResponseAgentState

from test_response_generation import FakeGeminiClient, _draft_json  # noqa: E402 -- reuse the existing fakes

THANKS_DRAFT = "شكرًا لكلماتك الطيبة عن العمل على المعجم."
INQUIRY_DRAFT = "يمكنكم اقتراح كلمة جديدة عبر نموذج الإضافة في الموقع."


def _req(**overrides) -> GenerationRequest:
    base = dict(
        comment_id="c1",
        comment_text="شكرًا لكم على هذا العمل الرائع",
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        sentiment="POSITIVE",
        confidence=0.9,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def _result_fields_excluding_timestamp(result) -> dict:
    d = result.to_dict()
    d.pop("generated_at")
    return d


# --- state model -----------------------------------------------------------


def test_state_from_request_maps_fields_directly():
    request = _req(identifying_metadata=IdentifyingMetadata("عنوان", None))
    state = ResponseAgentState.from_request(request)
    assert state.comment_id == request.comment_id
    assert state.comment_text == request.comment_text
    assert state.primary_intent == request.primary_intent
    assert state.topic_subtype == request.topic_subtype
    assert state.sentiment == request.sentiment
    assert state.confidence == request.confidence
    assert state.needs_review == request.needs_review
    assert state.identifying_metadata == request.identifying_metadata


def test_state_is_frozen():
    state = ResponseAgentState.from_request(_req())
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        state.comment_id = "different"


# --- strong parity: same request through both paths, same fake client -----


_PARITY_SCENARIOS = [
    pytest.param(
        "generate_draft_success",
        {},
        [_draft_json(THANKS_DRAFT)],
        {},
        id="generate_draft_success",
    ),
    pytest.param(
        "review_only_short_circuit",
        {"primary_intent": None, "topic_subtype": "OFF_TOPIC_PERSONAL"},
        [],
        {},
        id="review_only_short_circuit",
    ),
    pytest.param(
        "no_response_recommended_short_circuit",
        {"comment_text": "   ", "primary_intent": None, "topic_subtype": None, "sentiment": None},
        [],
        {},
        id="no_response_recommended_short_circuit",
    ),
    pytest.param(
        "malformed_json",
        {},
        ["not-json"],
        {"max_retries": 0},
        id="malformed_json",
    ),
    pytest.param(
        "validation_failure_pii",
        {},
        [_draft_json("شكرًا لتواصلكم على 0501234567 مع المعجم.")],
        {"max_retries": 0},
        id="validation_failure_pii",
    ),
    pytest.param(
        "retry_exhaustion",
        {},
        [httpx.ReadTimeout("timed out")] * 5,
        {"max_retries": 2},
        id="retry_exhaustion",
    ),
]


@pytest.mark.parametrize("_label,req_overrides,scripted,call_kwargs", _PARITY_SCENARIOS)
def test_strong_parity_orchestrator_matches_direct_call(monkeypatch, _label, req_overrides, scripted, call_kwargs):
    import rcs.response_generation.service as service_module

    monkeypatch.setattr(service_module, "_sleep", lambda seconds: None)  # no real waits (retry_exhaustion scenario)
    request = _req(**req_overrides)
    client_a = FakeGeminiClient(list(scripted))
    client_b = FakeGeminiClient(list(scripted))

    result_direct = generate_suggested_response(request, client_a, **call_kwargs)
    result_orchestrated = generate_response_with_orchestrator(request, client_b, **call_kwargs)

    assert _result_fields_excluding_timestamp(result_direct) == _result_fields_excluding_timestamp(result_orchestrated)
    # generated_at is a fresh timestamp per call (service.py's _now(), truncated
    # to the second) -- not asserted equal, but both must be present and
    # well-formed, proving the orchestrator did not skip result construction.
    assert result_direct.generated_at
    assert result_orchestrated.generated_at

    # Call-count parity: the orchestrator must invoke the fake exactly as
    # many times as the direct call does for the identical scenario.
    assert client_a.call_count == client_b.call_count


# --- M5D.1: identifying-context-alone no longer satisfies factual grounding -
#
# Removed scenario "trusted_grounding_preserved" asserted the OLD (M5C/M5D)
# behavior: a caller claiming trusted_grounding_available=True with
# identifying-context-shaped text saw byte-identical eligibility routing
# through direct and orchestrated calls. That premise -- "identifying
# context alone is trusted factual grounding" -- was exactly what M5D's
# own trust rule forbids and is now corrected (see
# rcs.response_intelligence.evidence's module docstring). The direct call
# still honors whatever the caller claims (generate_suggested_response is
# unmodified and trusts its input); the orchestrated call now enforces the
# boundary and INTENTIONALLY diverges. This test asserts the corrected
# behavior directly instead of parity.


def test_identifying_context_alone_no_longer_grants_grounded_eligibility_through_orchestrator(legacy_eligibility_policy):
    """Phase M6.2: the orchestrated path now requires structured
    `identifying_metadata` (never the legacy pre-rendered
    `trusted_grounding_text`), gated by the Identifying Context Policy --
    the comment below corroborates the title so the policy allows it."""
    request = _req(
        primary_intent="INQUIRY",
        topic_subtype="WORD_SUBMISSION",
        comment_text="كيف أضيف كلمة عبر «نموذج اقتراح كلمة» في الموقع؟",
        trusted_grounding_available=True,  # caller's claim -- must not be blindly trusted
        trusted_grounding_text="الزائر يضيف كلمة عبر نموذج اقتراح كلمة في الموقع.",
    )
    client_direct = FakeGeminiClient([_draft_json(INQUIRY_DRAFT)])
    client_orchestrated = FakeGeminiClient([_draft_json(INQUIRY_DRAFT)])

    direct = generate_suggested_response(request, client_direct)

    orchestrated_request = dataclasses.replace(
        request, identifying_metadata=IdentifyingMetadata("نموذج اقتراح كلمة", None)
    )
    orchestrated = generate_response_with_orchestrator(orchestrated_request, client_orchestrated)

    # Direct call trusts the caller's claim (unmodified, pre-existing
    # behavior -- generate_suggested_response never goes through the
    # orchestrator/policy at all).
    assert direct.eligibility_rule_id == "generate_inquiry_with_grounding"
    # Orchestrated call does NOT -- no real Official Knowledge exists, so
    # identifying context alone cannot grant the grounded route.
    assert orchestrated.eligibility_rule_id == "inquiry_without_trusted_grounding"
    # The identifying-context TEXT itself still reaches the prompt (the
    # comment corroborates it, so the policy allows it) -- it is still
    # useful for "which entry", just not factual authority. The legacy
    # trusted_grounding_text sentence set above is inert for the
    # orchestrated path and must NOT appear.
    assert "نموذج اقتراح كلمة" in client_orchestrated.prompts_seen[0]
    assert "IDENTIFYING CONTEXT" in client_orchestrated.prompts_seen[0]
    assert "الزائر يضيف كلمة عبر نموذج اقتراح كلمة في الموقع." not in client_orchestrated.prompts_seen[0]


# --- classification signals preserved (dedicated, not only via parity) ----


def test_classification_signals_preserved_through_orchestrator():
    client = FakeGeminiClient([_draft_json(THANKS_DRAFT)])
    request = _req(primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE", confidence=0.77)
    result = generate_response_with_orchestrator(request, client)
    assert result.routing_decision == "GENERATE_DRAFT"
    assert result.reason_code == "thanks_praise_eligible"


# --- exact metadata parity --------------------------------------------------


def test_provider_model_prompt_version_metadata_parity():
    client = FakeGeminiClient([_draft_json(THANKS_DRAFT)])
    result = generate_response_with_orchestrator(_req(), client)
    direct = generate_suggested_response(_req(), FakeGeminiClient([_draft_json(THANKS_DRAFT)]))
    assert result.provider == direct.provider
    assert result.model_name == direct.model_name
    assert result.prompt_version == direct.prompt_version


# --- downstream GenerationResult compatibility ------------------------------


def test_orchestrator_result_is_accepted_by_apply_generation_result(tmp_path):
    """The orchestrator's output must be indistinguishable, to the
    persistence layer, from generate_suggested_response's own output --
    proven by actually running it through apply_generation_result."""
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    engine = get_engine(tmp_path / "orchestrator_parity.sqlite")
    init_db(engine)
    upsert_review_items(engine, [dict(
        comment_id="c1", source_file="f", source_row_index=0,
        review_status="CLASSIFIED", is_prefiltered_junk=False, ready_to_send=False, row_version=1,
        ai_primary_intent="THANKS_PRAISE",
    )])
    client = FakeGeminiClient([_draft_json(THANKS_DRAFT)])
    result = generate_response_with_orchestrator(_req(), client)
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(result, intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE")
    assert detail.review_status == "DRAFT_READY"
    assert detail.suggested_response == THANKS_DRAFT


# --- call-count / side-effect proofs ----------------------------------------


def test_non_generate_draft_case_makes_zero_gemini_calls():
    client = FakeGeminiClient([])  # would raise AssertionError if ever called
    request = _req(primary_intent=None, topic_subtype="OFF_TOPIC_PERSONAL")
    result = generate_response_with_orchestrator(request, client)
    assert result.routing_decision == "REVIEW_ONLY"      # OFF_TOPIC_PERSONAL alone is uncorroborated -> human review, still no model call
    assert result.reason_code == "uncertain_exclusion_requires_review"
    assert client.call_count == 0


def test_successful_generation_makes_exactly_one_gemini_call():
    client = FakeGeminiClient([_draft_json(THANKS_DRAFT)])
    generate_response_with_orchestrator(_req(), client)
    assert client.call_count == 1


def test_retry_exhaustion_makes_exact_same_attempt_count_as_direct_call(monkeypatch):
    """Uses a genuinely retryable failure shape (httpx.ReadTimeout) so
    this actually exercises retry exhaustion (3 attempts), not a fast
    non-retryable fail-fast path -- see Phase M5A.2's retry taxonomy in
    rcs.evaluation.genai_client. Backoff is disabled here (this test's
    concern is attempt-count parity, not timing -- see
    test_generation_reliability.py for backoff-specific coverage)."""
    import rcs.response_generation.service as service_module

    monkeypatch.setattr(service_module, "_sleep", lambda seconds: None)
    scripted = [httpx.ReadTimeout("timed out")] * 5
    client_direct = FakeGeminiClient(list(scripted))
    client_orchestrated = FakeGeminiClient(list(scripted))
    generate_suggested_response(_req(), client_direct, max_retries=2)
    generate_response_with_orchestrator(_req(), client_orchestrated, max_retries=2)
    assert client_direct.call_count == client_orchestrated.call_count == 3  # max_retries=2 -> 3 attempts


def test_no_autonomous_agent_framework_or_out_of_scope_integrations():
    """Static guard, updated for M5C then M5D: M5A.1's version of this
    test banned "retriev"/"memory" outright because NEITHER existed
    yet. M5C intentionally added real (bounded, deterministic)
    retrieval and Approved Response MEMORY, so those two substrings
    were un-banned. M5D intentionally adds exactly ONE bounded
    refinement attempt, so "refinement" is un-banned here too, for the
    same reason -- it now names a legitimate, required, hard-capped
    M5D concept (see rcs.response_generation.refinement), not an
    accidental scope leak. Boundedness itself (never a third draft, no
    loop) is proven by call-count tests in tests/test_m5d_refinement.py,
    not by banning the word here. What's still checked: no autonomous
    agent framework, no tool-calling/tool-selection, no persistent
    cross-request memory, and none of the explicitly out-of-scope
    integrations (any external dictionary service, any vector database)."""
    import ast
    import inspect

    import rcs.response_intelligence.orchestrator as orchestrator_module

    tree = ast.parse(inspect.getsource(orchestrator_module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    banned_substrings = (
        "langgraph",
        "toolcall",
        "tool_call",
        "react_agent",
        "checkpoint",
        "conversation_memory",
        "session_memory",
        "external_dictionary_client",
        "pinecone",
        "chromadb",
        "weaviate",
        "qdrant",
        "pgvector",
        "elasticsearch",
        "faiss",
    )
    for name in imported_names:
        lowered = name.lower()
        assert not any(b in lowered for b in banned_substrings), f"unexpected out-of-scope import: {name!r}"

    defined_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for name in defined_names:
        lowered = name.lower()
        assert not any(b in lowered for b in banned_substrings), f"unexpected out-of-scope definition: {name!r}"
