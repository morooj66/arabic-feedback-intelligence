"""
Phase M5F — deterministic test doubles for the offline evaluation
harness. No network, no database, no Hugging Face model. Mirrors the
exact fake shapes already established and tested across
tests/test_m5c_agent_layer.py, tests/test_m5d_refinement.py, and
tests/test_m5e_agent_audit.py — nothing new is invented here.
"""

from __future__ import annotations

from rcs.response_intelligence.audit import AgentAuditRecord
from rcs.response_intelligence.retrieval.contracts import (
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeQuery,
    OfficialKnowledgeRetrievalResult,
)


class FakeOfficialKnowledgeRetriever:
    """Always returns the one fixed result it was built with —
    sufficient for this harness, since each evaluation case defines its
    OWN Official Knowledge outcome as part of its fixture, not a live
    corpus lookup."""

    def __init__(self, result: OfficialKnowledgeRetrievalResult):
        self.result = result
        self.calls: list[OfficialKnowledgeQuery] = []

    def retrieve(self, query: OfficialKnowledgeQuery) -> OfficialKnowledgeRetrievalResult:
        self.calls.append(query)
        return self.result


class FakeApprovedResponseMemoryRetriever:
    def __init__(self, result: ApprovedResponseMemoryRetrievalResult):
        self.result = result
        self.calls: list[ApprovedResponseMemoryQuery] = []

    def retrieve(self, query: ApprovedResponseMemoryQuery) -> ApprovedResponseMemoryRetrievalResult:
        self.calls.append(query)
        return self.result


class FakeGeminiClient:
    """Scripted, deterministic. `responses` items are either a JSON
    string to return or an Exception instance to raise on that call —
    exactly the same double used throughout the M5D/M5E test suites."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        item = self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


class InMemoryAuditRepository:
    """Satisfies rcs.response_intelligence.audit.AgentAuditRepository.
    Used to prove M5E trace correlation works from the offline harness
    WITHOUT ever touching a real database — an entirely separate,
    in-memory evaluation-run trace store, never mixed with production
    audit history."""

    def __init__(self):
        self._by_id: dict[str, AgentAuditRecord] = {}

    def record(self, record: AgentAuditRecord) -> None:
        if record.execution_id in self._by_id:
            return
        self._by_id[record.execution_id] = record

    def list_for_review_item(self, review_item_id: str) -> tuple[AgentAuditRecord, ...]:
        return tuple(
            sorted(
                (r for r in self._by_id.values() if r.review_item_id == review_item_id),
                key=lambda r: r.created_at,
            )
        )

    def get_by_execution_id(self, execution_id: str) -> AgentAuditRecord | None:
        return self._by_id.get(execution_id)


def make_draft_json(text: str) -> str:
    import json

    return json.dumps({"suggested_response": text}, ensure_ascii=False)
