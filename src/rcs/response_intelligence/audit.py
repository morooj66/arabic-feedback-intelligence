"""
Phase M5E — sanitized agent execution audit trace: pure types + a
storage-agnostic Protocol. No SQLAlchemy, no pymongo, no database
access, no Gemini, no retrieval logic. Concrete implementations
(`PostgresAgentAuditRepository`, `MongoAgentAuditRepository`) live in
`rcs.review.repositories.{postgres,mongo}` — this module never imports
them, matching the same storage-agnostic-core pattern already
established for the M5B retrieval Protocols
(`OfficialSourceProvider`/`ApprovedResponseCandidateSource`) and kept
for the same reason: `rcs.response_intelligence.orchestrator` must stay
free of any direct database dependency.

--- What this record is, and is not -------------------------------------

`AgentAuditRecord` answers "what happened during this agent execution"
for observability/auditability — NOT "what did the model see or think."
Every field is a short outcome code, a count, or a boolean. There is no
field here that could hold a prompt, a comment, retrieved content, or a
raw exception. `FORBIDDEN_FIELD_SUBSTRINGS` below is the enforced list
of substrings that must never appear as a field name here or on the ORM
`AgentExecutionTrace` model (`rcs.review.models`) that mirrors this
shape.

Phase M5E.1: this policy is enforced by
tests/test_m5e_agent_audit.py's `test_no_forbidden_field_names_*` tests
ONLY — there is deliberately no import-time assertion/side effect in
this module. Importing a plain domain module must never itself execute
schema-policy enforcement; a future violation of this policy should
fail CI/tests, not unexpectedly break application startup for every
caller that happens to import this module.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

# Field/column names must never contain any of these substrings.
# Imported directly by tests/test_m5e_agent_audit.py (single source of
# truth -- not duplicated into the test file) and checked there against
# both `AgentAuditRecord`'s dataclass fields and the ORM
# `AgentExecutionTrace` model's columns.
FORBIDDEN_FIELD_SUBSTRINGS = (
    "prompt",
    "reasoning",
    "chain",
    "thought",
    "raw_response",
    "raw_request",
    "comment_text",
    "email",
    "phone",
    "token",
    "secret",
    "api_key",
    "knowledge_text",
    "style_text",
)


def assert_no_forbidden_field_names(field_names) -> None:
    """Pure, side-effect-free (never called at import time — see module
    docstring). Callers (tests) pass field names from wherever they
    need to check: `[f.name for f in dataclasses.fields(AgentAuditRecord)]`
    or `[c.name for c in sqlalchemy.inspect(AgentExecutionTrace).columns]`."""
    for name in field_names:
        lowered = name.lower()
        for banned in FORBIDDEN_FIELD_SUBSTRINGS:
            assert banned not in lowered, f"forbidden substring {banned!r} found in field name {name!r}"


@dataclass(frozen=True)
class AgentAuditRecord:
    """One row: one agent execution. Append-only by convention (see
    the repository Protocol below) -- never updated or overwritten
    after `record()` succeeds.

    `execution_id` is this record's stable, unique identity (a fresh
    uuid4 hex minted once per `rcs.response_intelligence.orchestrator.
    run_agent_pipeline` call -- never derived from or reused as
    `review_item_id`, since one review item may be generated more than
    once). `review_item_id` is a plain reference column, not an
    enforced foreign key (matching this codebase's existing convention
    for `ProcessingJob.session_id`)."""

    execution_id: str
    review_item_id: str
    created_at: dt.datetime

    grounding_mode: str

    official_retrieval_outcome: str
    official_result_count: int
    official_grounding_available: bool

    style_retrieval_outcome: str
    style_result_count: int

    logical_generation_count: int
    refinement_attempted: bool

    initial_validation_passed: bool
    initial_failure_code: str | None
    final_validation_passed: bool
    final_failure_code: str | None

    generation_status: str


class AgentAuditRepository(Protocol):
    """Storage-agnostic. `record()` must be idempotent on
    `execution_id` -- inserting the same `execution_id` twice (e.g. a
    persistence-layer retry of the same write) must not create a
    second row; a genuinely new execution always has a fresh
    `execution_id` and always produces a new row. No `update`/`delete`
    method exists on this Protocol on purpose: traces are append-only
    from the application's perspective."""

    def record(self, record: AgentAuditRecord) -> None: ...

    def list_for_review_item(self, review_item_id: str) -> tuple[AgentAuditRecord, ...]: ...

    def get_by_execution_id(self, execution_id: str) -> AgentAuditRecord | None: ...


def build_audit_record(
    *,
    execution_id: str,
    review_item_id: str,
    created_at: dt.datetime,
    grounding_mode: str,
    official_retrieval_outcome: str,
    official_result_count: int,
    official_grounding_available: bool,
    style_retrieval_outcome: str,
    style_result_count: int,
    logical_generation_count: int,
    refinement_attempted: bool,
    initial_validation_passed: bool,
    initial_failure_code: str | None,
    final_validation_passed: bool,
    final_failure_code: str | None,
    generation_status: str,
) -> AgentAuditRecord:
    """Thin, explicit constructor -- kept as a function (not just the
    dataclass itself) so `rcs.response_intelligence.orchestrator` has
    one obvious call site to read when checking exactly what gets
    persisted, without needing to also read `AgentExecutionMetadata`'s
    definition side by side."""
    return AgentAuditRecord(
        execution_id=execution_id,
        review_item_id=review_item_id,
        created_at=created_at,
        grounding_mode=grounding_mode,
        official_retrieval_outcome=official_retrieval_outcome,
        official_result_count=official_result_count,
        official_grounding_available=official_grounding_available,
        style_retrieval_outcome=style_retrieval_outcome,
        style_result_count=style_result_count,
        logical_generation_count=logical_generation_count,
        refinement_attempted=refinement_attempted,
        initial_validation_passed=initial_validation_passed,
        initial_failure_code=initial_failure_code,
        final_validation_passed=final_validation_passed,
        final_failure_code=final_failure_code,
        generation_status=generation_status,
    )


