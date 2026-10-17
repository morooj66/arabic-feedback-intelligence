"""
Phase DEV-INSPECT — admin-only Agent Inspector / Test Console backend.

This module is NOT a new agent and introduces NO new decision logic.
Every stage function here calls the exact same production function the
real pipeline calls (`rcs.classification.runtime.*`,
`rcs.response_generation.eligibility.determine_response_eligibility`,
the default retrievers from `rcs.response_intelligence.retrievers`,
`rcs.response_intelligence.evidence.*`,
`rcs.response_generation.validate.run_validation_gates`, and
`rcs.response_intelligence.orchestrator.run_agent_pipeline` itself for
the full end-to-end run) — it only packages already-computed return
values into a structured, developer-facing shape. Nothing here is
reachable from any reviewer-facing code path.

TEST / DRY-RUN GUARANTEE: no function in this module ever touches
`review_items`, `review_sessions`, reviewer assignment, or the
production `AgentAuditRepository`. The full-run path uses
`InspectorAuditRepository`, an isolated in-memory trace store scoped to
this process only — test-console executions must never be
indistinguishable from real reviewer-triggered executions in the real
M5E audit trail, so they are never written there.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass

from rcs.classification import runtime as classification_runtime
from rcs.response_generation.eligibility import EligibilityInput, EligibilityResult, determine_response_eligibility
from rcs.response_generation.service import GenerationRequest, SupportsGenerate
from rcs.response_generation.validate import ValidationSummary, run_validation_gates
from rcs.response_intelligence import evidence
from rcs.response_intelligence.audit import AgentAuditRecord, AgentAuditRepository, build_audit_record
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from rcs.response_intelligence.orchestrator import AgentExecutionResult, run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeQuery,
    OfficialKnowledgeRetrievalResult,
)
from rcs.response_intelligence.retrievers import (
    default_approved_response_memory_retriever,
    default_official_knowledge_retriever,
)


class InspectorAuditRepository:
    """Isolated, in-memory `AgentAuditRepository` for the Inspector only.
    Never the real production repository -- see module docstring. Data
    lives only as long as this process and is never read by anything
    reviewer-facing; it exists purely so a dev session can see 'this
    inspector run produced trace X' without contaminating real M5E
    audit history with test-console executions."""

    def __init__(self) -> None:
        self._by_id: dict[str, AgentAuditRecord] = {}

    def record(self, record: AgentAuditRecord) -> None:
        self._by_id[record.execution_id] = record

    def list_for_review_item(self, review_item_id: str) -> tuple[AgentAuditRecord, ...]:
        return tuple(r for r in self._by_id.values() if r.review_item_id == review_item_id)

    def get_by_execution_id(self, execution_id: str) -> AgentAuditRecord | None:
        return self._by_id.get(execution_id)

    def clear(self) -> None:
        self._by_id.clear()


# One process-lifetime instance -- deliberately never the same object as
# any production audit_repository wiring (there is none passed here by
# any caller other than this module).
INSPECTOR_AUDIT_REPOSITORY = InspectorAuditRepository()

# DEV-INSPECT comment_ids are namespaced so they can never collide with,
# or be mistaken for, a real review_items.comment_id.
_INSPECTOR_COMMENT_ID_PREFIX = "dev-inspect-"


def _inspector_comment_id() -> str:
    return f"{_INSPECTOR_COMMENT_ID_PREFIX}{uuid.uuid4().hex}"


@dataclass(frozen=True)
class ClassificationInspection:
    primary_intent: str | None
    primary_intent_confidence: float | None
    topic_subtype: str | None
    topic_subtype_confidence: float | None
    sentiment: str | None
    sentiment_confidence: float | None


def inspect_classification(comment_text: str) -> ClassificationInspection:
    """Calls the SAME configured classification runtime used in production.

    ``legacy_rcs`` remains the default frozen local composition.  When the
    explicitly experimental ``hybrid_jev_topic`` mode is configured, only
    Topic crosses the existing redacted Jev boundary; Intent and Sentiment
    remain the frozen local models.  Runtime errors propagate fail-closed --
    the Inspector never substitutes a different classifier.
    """
    cid = _inspector_comment_id()
    intent, intent_conf = classification_runtime.predict_intent([comment_text])[0]
    topic, topic_conf = classification_runtime.predict_topic([cid], [comment_text])[cid]
    sentiment, sentiment_conf = classification_runtime.predict_sentiment([cid], [comment_text])[cid]
    return ClassificationInspection(
        primary_intent=intent,
        primary_intent_confidence=intent_conf,
        topic_subtype=topic,
        topic_subtype_confidence=topic_conf,
        sentiment=sentiment,
        sentiment_confidence=sentiment_conf,
    )


def inspect_eligibility(
    *,
    comment_text: str,
    primary_intent: str | None = None,
    topic_subtype: str | None = None,
    sentiment: str | None = None,
    confidence: float | None = None,
    needs_review: bool = False,
    insufficient_context: bool = False,
    trusted_grounding_available: bool = False,
) -> EligibilityResult:
    """Pure passthrough to the real, unmodified
    `determine_response_eligibility` -- no network, no Gemini, safe to
    call standalone for a 'Retrieval Test' preview or the 'Eligibility /
    Routing' inspector stage."""
    return determine_response_eligibility(
        EligibilityInput(
            comment_text=comment_text,
            primary_intent=primary_intent,
            topic_subtype=topic_subtype,
            sentiment=sentiment,
            confidence=confidence,
            needs_review=needs_review,
            insufficient_context=insufficient_context,
            trusted_grounding_available=trusted_grounding_available,
        )
    )


@dataclass(frozen=True)
class RetrievalInspection:
    official_outcome: str
    official_result_count: int
    official_contents: tuple[str, ...] | None  # None unless reveal_content=True
    approved_outcome: str
    approved_result_count: int
    approved_contents: tuple[str, ...] | None  # None unless reveal_content=True -- STYLE ONLY, never factual


def inspect_retrieval(
    *,
    comment_text: str,
    primary_intent: str | None = None,
    topic_subtype: str | None = None,
    reveal_content: bool = False,
) -> RetrievalInspection:
    """Calls the REAL default retrievers
    (`rcs.response_intelligence.retrievers.default_official_knowledge_retriever`/
    `default_approved_response_memory_retriever`) -- the exact same
    process-lifetime cached instances production uses. `reveal_content`
    defaults to False: by default only outcome/count are returned, never
    raw source text (per DEV-INSPECT's exclusion list)."""
    official = default_official_knowledge_retriever().retrieve(
        OfficialKnowledgeQuery(query_text=comment_text, max_results=3, topic=topic_subtype, intent=primary_intent)
    )
    approved = default_approved_response_memory_retriever().retrieve(
        ApprovedResponseMemoryQuery(query_text=comment_text, max_results=2, intent=primary_intent, topic=topic_subtype)
    )
    return RetrievalInspection(
        official_outcome=official.outcome,
        official_result_count=len(official.results),
        official_contents=tuple(r.content for r in official.results) if reveal_content else None,
        approved_outcome=approved.outcome,
        approved_result_count=len(approved.results),
        approved_contents=tuple(r.response_text for r in approved.results) if reveal_content else None,
    )


@dataclass(frozen=True)
class GroundingInspection:
    grounding_mode: str
    official_grounding_available: bool
    identifying_context_present: bool


def inspect_grounding(
    *,
    identifying_context: str | None,
    official_result: OfficialKnowledgeRetrievalResult,
) -> GroundingInspection:
    """Pure passthrough to the real, unmodified
    `evidence.assemble_grounding` -- see that module for the trust-
    boundary rules this reuses unchanged. `identifying_context` is
    title/entry_id metadata (e.g. from `build_identifying_grounding`),
    never a fact -- only a real `OfficialKnowledgeRetrievalResult` at
    SUCCESS_WITH_RESULTS can ever flip `official_grounding_available`."""
    grounding = evidence.assemble_grounding(identifying_context, official_result)
    return GroundingInspection(
        grounding_mode=grounding.mode,
        official_grounding_available=grounding.official_grounding_available,
        identifying_context_present=grounding.identifying_context is not None,
    )


def inspect_validation(*, draft_text: str, trusted_grounding_text: str | None = None) -> ValidationSummary:
    """Pure passthrough to the real, unmodified `run_validation_gates` —
    no network, no Gemini. Lets a developer paste an arbitrary draft and
    see exactly which gate would pass/fail it, independent of a live
    generation call."""
    return run_validation_gates(draft_text, trusted_grounding_text=trusted_grounding_text)


def run_full_inspection(
    *,
    comment_text: str,
    primary_intent: str | None,
    topic_subtype: str | None,
    sentiment: str | None,
    confidence: float | None,
    insufficient_context: bool,
    original_title: str | None = None,
    entry_id: str | None = None,
    client: SupportsGenerate | None,
) -> AgentExecutionResult:
    """The 'Full Agent E2E Test'. Calls the REAL, unmodified
    `run_agent_pipeline` with `capture_trace=True` so every stage's
    intermediate value is available, and `audit_repository=
    INSPECTOR_AUDIT_REPOSITORY` (isolated, never the production audit
    store — see module docstring). `comment_id` is a fresh, namespaced
    `dev-inspect-*` id: this execution is NOT tied to, and never
    mutates, any real `review_items` row.

    Phase M6.2: takes structured `original_title`/`entry_id` (not a
    pre-rendered `identifying_context` string) -- exactly like every
    other canonical caller of the Response Agent, this one supplies
    metadata only and lets `run_agent_pipeline`'s Context/Trust Policies
    step (identifying_context_policy.evaluate_identifying_context)
    decide whether it is ever used. There is no way to inject
    already-rendered identifying-context prose that bypasses that
    policy through this function."""
    request = GenerationRequest(
        comment_id=_inspector_comment_id(),
        comment_text=comment_text,
        primary_intent=primary_intent,
        topic_subtype=topic_subtype,
        sentiment=sentiment,
        confidence=confidence,
        insufficient_context=insufficient_context,
        identifying_metadata=IdentifyingMetadata(original_title, entry_id),
    )
    return run_agent_pipeline(
        request,
        client,
        None,
        audit_repository=INSPECTOR_AUDIT_REPOSITORY,
        capture_trace=True,
    )
