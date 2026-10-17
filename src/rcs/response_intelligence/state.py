"""
Phase M5A.1 — typed state for the response orchestration boundary.
Phase M5C — extended with the two retrieval-result slots the
deterministic agent layer fills in after `from_request` (still no
retrieval logic here — see `rcs.response_intelligence.orchestrator`).

`from_request` remains the exact M5A.1 pure conversion: it never touches
retrieval and never reads a retriever. Retrieval results are attached
afterward via `with_retrieval`, which returns a NEW frozen instance (no
mutation) — this state is a transient, per-call snapshot, never
persisted, never logged, and carries no reasoning trace or refinement
count (those remain out of scope, per instruction).

--- Phase M6.2 — structured identifying metadata, policy-gated ------------

`ResponseAgentState.identifying_metadata` is a direct, unresolved
passthrough of `GenerationRequest.identifying_metadata` (a structured
`IdentifyingMetadata(original_title, entry_id)` — see
`rcs.response_intelligence.identifying_context_policy`). No resolution,
no fallback, no reverse-parsing happens here: `from_request` only copies
the field. `rcs.response_intelligence.orchestrator` is the sole caller
that evaluates it (via `evaluate_identifying_context`) and decides
whether identifying context reaches the prompt at all.

This supersedes Phase M6.1's `_resolve_identifying_context` (removed),
which read the LEGACY `trusted_grounding_text`/`identifying_context`
pre-rendered-text fields and picked one via a text-level precedence
rule. That legacy dual-field convention is retired for orchestrator
purposes — `ResponseAgentState` no longer carries a
`trusted_grounding_text` field at all (nothing else read it; see
`rcs.response_intelligence.orchestrator`, its only consumer). Those two
`GenerationRequest` fields remain valid for direct
(non-orchestrator) callers of `rcs.response_generation.service.
generate_once_with_draft` — a pre-existing, separate calling
convention this state layer was never part of.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from rcs.response_generation.service import GenerationRequest
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from rcs.response_intelligence.retrieval.contracts import (
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
)


@dataclass(frozen=True)
class ResponseAgentState:
    """A read-only snapshot of the inputs the orchestrator hands to the
    existing generation machinery, plus (once `with_retrieval` has been
    called) the two retrieval outcomes for this one request. Field names
    for the M5A.1 portion mirror `GenerationRequest` deliberately, for
    the same reason the rest of this codebase keeps one name per concept
    across layers (see `ReviewItem`/`GenerationRequest` both using
    `primary_intent`/`topic_subtype`) — a reviewer should never need to
    mentally remap a field name when moving between this state and the
    request it was built from."""

    comment_id: str
    comment_text: str
    primary_intent: Optional[str]
    topic_subtype: Optional[str]
    sentiment: Optional[str]
    confidence: Optional[float]
    needs_review: bool
    identifying_metadata: Optional[IdentifyingMetadata]
    official_knowledge_result: Optional[OfficialKnowledgeRetrievalResult] = None
    approved_response_result: Optional[ApprovedResponseMemoryRetrievalResult] = None

    @staticmethod
    def from_request(request: GenerationRequest) -> "ResponseAgentState":
        return ResponseAgentState(
            comment_id=request.comment_id,
            comment_text=request.comment_text,
            primary_intent=request.primary_intent,
            topic_subtype=request.topic_subtype,
            sentiment=request.sentiment,
            confidence=request.confidence,
            needs_review=request.needs_review,
            identifying_metadata=request.identifying_metadata,
        )

    def with_retrieval(
        self,
        official_knowledge_result: OfficialKnowledgeRetrievalResult,
        approved_response_result: ApprovedResponseMemoryRetrievalResult,
    ) -> "ResponseAgentState":
        return dataclasses.replace(
            self,
            official_knowledge_result=official_knowledge_result,
            approved_response_result=approved_response_result,
        )
