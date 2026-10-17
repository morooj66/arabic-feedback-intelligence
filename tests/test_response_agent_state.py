"""
Phase M6.2 — focused regression tests for `ResponseAgentState.from_request`
(`rcs.response_intelligence.state`).

Supersedes Phase M6.1's dual-field (`trusted_grounding_text` vs
`identifying_context`) compatibility rule, which is retired: this state
layer now carries a single, structured `identifying_metadata` field,
copied directly from `GenerationRequest.identifying_metadata` with no
resolution, no fallback, no reverse-parsing. Whether that metadata is
actually USED is decided later, exactly once, by
`rcs.response_intelligence.identifying_context_policy.
evaluate_identifying_context` inside the orchestrator -- not here. See
tests/test_identifying_context_policy.py for that decision logic.
"""

from __future__ import annotations

from rcs.response_generation.service import GenerationRequest
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from rcs.response_intelligence.state import ResponseAgentState


def _req(**overrides) -> GenerationRequest:
    base = dict(comment_id="c1", comment_text="نص تجريبي")
    base.update(overrides)
    return GenerationRequest(**base)


def test_identifying_metadata_copied_directly_from_request():
    metadata = IdentifyingMetadata("كلمة", "entry-1")
    request = _req(identifying_metadata=metadata)
    state = ResponseAgentState.from_request(request)
    assert state.identifying_metadata is metadata


def test_no_identifying_metadata_resolves_to_none():
    request = _req(identifying_metadata=None)
    state = ResponseAgentState.from_request(request)
    assert state.identifying_metadata is None


def test_legacy_fields_are_not_read_into_state_at_all():
    """The retired M6.1 fields (trusted_grounding_text/identifying_context
    as pre-rendered text) must have zero effect on state -- only
    identifying_metadata is copied."""
    request = _req(
        trusted_grounding_text='submission title: "LEGACY".',
        identifying_context='submission title: "ALSO_LEGACY".',
        identifying_metadata=None,
    )
    state = ResponseAgentState.from_request(request)
    assert state.identifying_metadata is None


def test_production_excel_caller_shape():
    """Mirrors exactly how rcs.review.excel_review_flow.apply_live_generation
    builds its GenerationRequest today (Phase M6.2)."""
    request = GenerationRequest(
        comment_id="row-1",
        comment_text="تعليق تجريبي",
        primary_intent="SUGGESTION",
        topic_subtype="WORD_SUBMISSION",
        sentiment="NEUTRAL",
        confidence=0.9,
        needs_review=False,
        identifying_metadata=IdentifyingMetadata("كلمة", "entry-1"),
    )
    state = ResponseAgentState.from_request(request)
    assert state.identifying_metadata == IdentifyingMetadata("كلمة", "entry-1")


def test_response_agent_state_has_no_trusted_grounding_text_field():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(ResponseAgentState)}
    assert "trusted_grounding_text" not in fields


def test_from_request_still_copies_every_other_field_unchanged():
    request = _req(
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL", confidence=0.5, needs_review=True,
    )
    state = ResponseAgentState.from_request(request)
    assert state.comment_id == "c1"
    assert state.comment_text == "نص تجريبي"
    assert state.primary_intent == "INQUIRY"
    assert state.topic_subtype == "GENERAL"
    assert state.sentiment == "NEUTRAL"
    assert state.confidence == 0.5
    assert state.needs_review is True
    assert state.official_knowledge_result is None
    assert state.approved_response_result is None
