"""
Phase M5A.1 — response orchestration foundation.
Phase M5C — the deterministic response agent layer: retrieval + evidence
assembly wired into the existing generation seam.
Phase M5D — grounding enforcement + one bounded refinement attempt.

Pipeline (fixed, non-branching, no loop, at most TWO logical Gemini calls):

  Frozen Classification (upstream, already done before this is called)
  -> Existing Eligibility/Routing (inside generate_once_with_draft, unmodified)
  -> Agent State (ResponseAgentState.from_request)
  -> Context/Trust Policies (Phase M6.2 — identifying_context_policy.
     evaluate_identifying_context; see its own module docstring)
  -> Official Knowledge Retrieval
  -> Approved Response Memory Retrieval
  -> Evidence Assembly (rcs.response_intelligence.evidence)
  -> Grounding Mode / Grounding Enforcement (evidence.GroundingAssembly)
  -> Gemini Initial Draft (generate_once_with_draft, unmodified logic)
  -> Existing Validation Gates (unmodified, inside generate_once_with_draft)
  -> IF AND ONLY IF the initial failure's reason_code is in
     rcs.response_generation.refinement.REFINEMENT_ELIGIBLE_REASON_CODES:
       -> exactly ONE refinement Gemini call (same unmodified
          generate_once_with_draft, called again with a refinement-context
          request)
       -> validation again (same unmodified gates)
  -> GenerationResult (unmodified public schema)
  -> Existing Human Review Flow (untouched by this phase)

This is NOT an autonomous agent: no LangGraph, no ReAct, no branching on
model output beyond the single deterministic refinement-eligibility
check below, no tool selection, no third draft, no recursion, no
while-loop, no persisted trace or chain-of-thought. The hard invariant
is `logical_generation_count <= 2` — see `AgentExecutionMetadata`.

--- Grounding enforcement (M5D, corrected in M5D.1) ------------------------

Only `OfficialKnowledgeRetrievalResult` at SUCCESS_WITH_RESULTS may ever
set `GroundingAssembly.official_grounding_available = True` (see
evidence.py). Approved Response Memory results are assembled into
`style_reference_texts` ONLY — there is no code path from an
`ApprovedResponseMemoryResult` to `official_grounding_text` anywhere in
this module. `identifying_context` (the pre-existing title/entry_id
text) is also structurally excluded from `official_grounding_text` and
from `official_grounding_available`.

As of M5D.1, `trusted_grounding_available` (the ELIGIBILITY-facing flag
passed to `generate_once_with_draft`) is set to `grounding.
official_grounding_available` DIRECTLY — nothing else. M5D had instead
computed it as `identifying_context OR official_grounding_available`,
preserving a pre-M5A behavior where identifying context alone could
route a comment onto the grounded-inquiry path; that was inconsistent
with M5D's own trust rule and has been removed (see evidence.py's
module docstring for the full history).

As of M5D.2, this separation is enforced all the way into the PROMPT
TEXT itself, not just the eligibility flag: `enriched_request.
trusted_grounding_text` is set to `grounding.official_grounding_text`
(official evidence ONLY, `None` when absent), and a NEW, separate
`enriched_request.identifying_context` field carries the pre-existing
title/entry_id text. `rcs.response_generation.prompt.
build_generation_prompt` renders these under two distinct, honestly
labeled sections ("OFFICIAL KNOWLEDGE" and "IDENTIFYING CONTEXT") — it
never concatenates them. Before M5D.2, both were merged into one string
under a single "TRUSTED GROUNDING" prompt block, which was misleading
whenever that block held only identifying context (it read as if
identity metadata were trustworthy factual evidence — the exact
confusion M5D.1 had already fixed at the eligibility layer but left
standing in the text actually shown to Gemini).

--- Phase M6.2 — Identifying Context Policy ---------------------------------

Before M6.2, `identifying_context` (if any) reached `assemble_grounding`
unconditionally whenever a caller supplied it, regardless of whether the
visitor's comment gave any reason to believe it was relevant. As of
M6.2, `state.identifying_metadata` (structured `IdentifyingMetadata`,
never pre-rendered text) is evaluated exactly once, here, by
`rcs.response_intelligence.identifying_context_policy.
evaluate_identifying_context`, BEFORE `build_identifying_grounding` is
ever called. `build_identifying_grounding` is invoked only when the
policy allows it; otherwise `resolved_identifying_context` stays `None`
and `assemble_grounding` sees exactly what it already saw for any
comment with no identifying metadata at all — no new code path, no new
special case downstream of this point. This is the ONLY place the
policy runs; no other module may re-implement it.

The unsupported-operational-claim validation gate (unmodified) is the
other, independent layer that actually gates factual claims in the
DRAFT text itself: identifying-context text can never satisfy it
because `build_identifying_grounding`'s fixed template never contains
any of `validate._UNSUPPORTED_ACTION_PATTERNS` — see
tests/test_m5d_refinement.py's structural proof of this.

Retrieval runs UNCONDITIONALLY before eligibility is evaluated (which
happens inside `generate_once_with_draft`, a black box to this module
by design). Both retrievers return SUCCESS_EMPTY without ever touching
the embedding model when their corpora are empty (true for every
retriever `rcs.response_intelligence.retrievers` builds today).

Dependency injection: `official_knowledge_retriever`/
`approved_response_retriever` may be passed explicitly. When neither is
given, `rcs.response_intelligence.retrievers`'s process-lifetime cached
defaults are used — built once, never per request.

--- DEV-INSPECT: verbose trace capture (opt-in, additive) ------------------

`capture_trace=True` returns an `AgentExecutionTrace` alongside the
normal result (see its docstring above). Built for the admin-only
Agent Inspector (`rcs.response_intelligence.inspector`) so a developer
can see Draft 1, Draft 2 (if any), and the assembled grounding/style
text without re-deriving them outside this function. Off by default;
zero effect on any existing caller.

--- Phase M5E: audit trace persistence (opt-in, additive) ------------------

`run_agent_pipeline` accepts an OPTIONAL `audit_repository`
(`rcs.response_intelligence.audit.AgentAuditRepository`). When given, a
sanitized `AgentAuditRecord` (outcome codes, counts, booleans only — see
audit.py's own forbidden-field guard) is recorded AFTER the execution
state is fully known, mirroring exactly what M5B.3's retrievers already
established: significant external-resource capabilities are ALWAYS
opt-in via an explicit parameter, never auto-defaulted (compare
`client: SupportsGenerate | None = None` — no live Gemini client is
auto-constructed either). When `audit_repository` is `None` (the
default, and every existing caller today), NO audit persistence is
attempted at all — zero behavior change for any pre-M5E caller.

Atomicity: audit persistence is DELIBERATELY NOT coupled to
`GenerationResult` persistence. The two were already non-atomic before
M5E — `run_agent_pipeline` has never touched a database itself; only
the caller's LATER, separate call to `ReviewWorkflowService.
apply_generation_result` does, in its own transaction. Making the trace
write atomic with that later call would be a NEW coupling this phase
does not introduce. A failure to record the audit trace is therefore
caught here, never allowed to raise out of `run_agent_pipeline` (the
business result — the actual `GenerationResult` a human reviewer needs
— must never be lost or blocked because an observability store is
down), and surfaced only through a sanitized `logger.warning` call
(execution_id/review_item_id/exception TYPE name only — never the raw
exception message, which could in principle echo a connection string).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import uuid

from config import settings
from rcs.response_generation.prompt import build_identifying_grounding, redact_for_generation
from rcs.response_generation.refinement import RefinementContext, is_refinement_eligible
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import (
    GenerationRequest,
    SupportsGenerate,
    generate_once_with_draft,
)
from rcs.response_intelligence import evidence
from rcs.response_intelligence.audit import AgentAuditRepository, build_audit_record
from rcs.response_intelligence.identifying_context_policy import evaluate_identifying_context
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryRetrievalResult,
    ApprovedResponseMemoryRetriever,
    OfficialKnowledgeQuery,
    OfficialKnowledgeRetrievalResult,
    OfficialKnowledgeRetriever,
)
from rcs.response_intelligence.retrievers import (
    default_approved_response_memory_retriever,
    default_official_knowledge_retriever,
)
from rcs.response_intelligence.state import ResponseAgentState

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AgentExecutionMetadata:
    """Structured, per-request execution metadata — NEVER logged in
    full, discarded when the caller is done with it unless explicitly
    persisted via `audit_repository` (Phase M5E). Carries no prompt
    text, no chain-of-thought, no reasoning: only an id, counts, codes,
    and booleans. This is deliberately NOT part of `GenerationResult`
    (see schema.py, unchanged)."""

    execution_id: str
    logical_generation_count: int
    refinement_attempted: bool
    initial_validation_passed: bool
    initial_failure_code: str | None
    final_validation_passed: bool
    final_failure_code: str | None
    grounding_mode: str
    official_grounding_available: bool
    official_retrieval_outcome: str
    official_result_count: int
    style_retrieval_outcome: str
    style_result_count: int
    generation_status: str
    # Phase M6.2 — Identifying Context Policy outcome. In-memory only for
    # now: NOT persisted to AgentAuditRecord/agent_execution_traces (see
    # module docstring's M6.2 section) -- available here for any caller
    # that wants to inspect/log it, and for the DEV-INSPECT trace below.
    identifying_context_allowed: bool
    identifying_context_decision_reason: str
    identifying_context_matched_reference: str | None


@dataclasses.dataclass(frozen=True)
class AgentExecutionTrace:
    """OPTIONAL, opt-in verbose capture of intermediate pipeline state —
    added for the DEV-INSPECT Agent Inspector (an admin-only, read-only
    introspection surface; see rcs.response_intelligence.inspector).

    Never populated unless `capture_trace=True` is explicitly passed to
    `run_agent_pipeline` -- every existing caller (production code,
    tests, M5F offline eval) leaves `capture_trace` at its default
    `False` and gets `AgentExecutionResult.trace = None`, identical to
    pre-DEV-INSPECT behavior. No new decision logic lives here: every
    field is a value the pipeline already computes and would otherwise
    discard when this function returns. No prompt text, no
    chain-of-thought, no raw exception bodies -- only the same kind of
    structured outcome/text-artifact values `GenerationResult` and
    `AgentExecutionMetadata` already expose elsewhere."""

    identifying_context: str | None
    identifying_context_decision_reason: str
    official_knowledge_result: OfficialKnowledgeRetrievalResult
    approved_response_result: ApprovedResponseMemoryRetrievalResult
    official_grounding_text: str | None
    style_reference_texts: tuple[str, ...]
    initial_draft: str | None
    initial_generation_result: GenerationResult
    refinement_draft: str | None
    refinement_generation_result: GenerationResult | None


@dataclasses.dataclass(frozen=True)
class AgentExecutionResult:
    generation_result: GenerationResult
    metadata: AgentExecutionMetadata
    trace: AgentExecutionTrace | None = None


def _retrieve_evidence(
    state: ResponseAgentState,
    ok_retriever: OfficialKnowledgeRetriever,
    arm_retriever: ApprovedResponseMemoryRetriever,
) -> tuple[OfficialKnowledgeRetrievalResult, ApprovedResponseMemoryRetrievalResult]:
    # Per contracts.py's own "query text vs corpus content" note: query_text
    # is redacted by the CALLER the same way generation-prompt input already
    # is -- this is unrelated to, and not restricted by, the corpus's own
    # storage policy.
    redacted_query_text = redact_for_generation(state.comment_text)

    if not redacted_query_text:
        # A blank/whitespace-only (or fully redacted-to-nothing) comment
        # has nothing to query with -- both M5B.1 query contracts reject
        # a blank query_text outright. Eligibility will route this to
        # NO_RESPONSE_RECOMMENDED anyway (empty_or_meaningless); skipping
        # retrieval here is correct, not just defensive.
        return (
            OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY),
            ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY),
        )

    official_result = ok_retriever.retrieve(
        OfficialKnowledgeQuery(
            query_text=redacted_query_text,
            max_results=settings.OFFICIAL_KNOWLEDGE_MAX_RESULTS_PER_QUERY,
            topic=state.topic_subtype,
            intent=state.primary_intent,
        )
    )
    approved_result = arm_retriever.retrieve(
        ApprovedResponseMemoryQuery(
            query_text=redacted_query_text,
            max_results=settings.APPROVED_RESPONSE_MEMORY_MAX_RESULTS_PER_QUERY,
            intent=state.primary_intent,
            topic=state.topic_subtype,
        )
    )
    return official_result, approved_result


def run_agent_pipeline(
    request: GenerationRequest,
    client: SupportsGenerate | None = None,
    max_retries: int | None = None,
    *,
    official_knowledge_retriever: OfficialKnowledgeRetriever | None = None,
    approved_response_retriever: ApprovedResponseMemoryRetriever | None = None,
    audit_repository: AgentAuditRepository | None = None,
    capture_trace: bool = False,
) -> AgentExecutionResult:
    """The full M5D/M5E pipeline. `generate_response_with_orchestrator`
    below is a thin wrapper that returns only `.generation_result`,
    preserving its exact pre-M5D public signature/return type for every
    existing caller (production code and tests alike).

    `execution_id` (a fresh uuid4 hex) identifies THIS ONE call —
    minted unconditionally, whether or not `audit_repository` is given,
    so `AgentExecutionMetadata.execution_id` is always populated and
    structured logs can always correlate to it even when persistence is
    off.

    `capture_trace` (default `False`, DEV-INSPECT): when `True`, returns
    an additional `AgentExecutionTrace` with intermediate values this
    function already computes internally. Purely additive -- changes
    nothing about eligibility, retrieval, grounding, validation, or
    refinement decisions; it only decides whether already-computed
    local values are also returned instead of discarded."""
    execution_id = uuid.uuid4().hex
    state = ResponseAgentState.from_request(request)

    # Context/Trust Policies (Phase M6.2, tightened M6.2.1) -- runs
    # exactly once, before retrieval, on the unresolved structured
    # metadata only. See rcs.response_intelligence.identifying_context_policy
    # for the full rule set. `build_identifying_grounding` (unchanged,
    # unmodified) is called ONLY when the policy allows it -- a denied
    # decision never produces rendered text at all.
    #
    # M6.2.1: `entry_id` is passed as `None` here UNCONDITIONALLY,
    # regardless of what `state.identifying_metadata.entry_id` holds
    # (a UUID, an Arabic word, or anything else). `entry_id` is internal
    # metadata only -- it never participates in the policy decision (see
    # identifying_context_policy's own rule) and must never become
    # prompt-visible text either. Only `original_title` -- the
    # human-readable identifying reference the policy actually
    # corroborated -- is ever rendered into the prompt. `entry_id`
    # remains on `IdentifyingMetadata` for any future internal
    # trace/lookup use; it simply never reaches `build_identifying_grounding`
    # from this, the canonical Response Agent path.
    identifying_context_decision = evaluate_identifying_context(
        comment_text=state.comment_text,
        identifying_metadata=state.identifying_metadata,
    )
    resolved_identifying_context: str | None = None
    if identifying_context_decision.allowed and state.identifying_metadata is not None:
        resolved_identifying_context = build_identifying_grounding(
            state.identifying_metadata.original_title,
            None,  # entry_id is internal-only -- never rendered into the prompt
        )

    ok_retriever = (
        official_knowledge_retriever
        if official_knowledge_retriever is not None
        else default_official_knowledge_retriever()
    )
    arm_retriever = (
        approved_response_retriever
        if approved_response_retriever is not None
        else default_approved_response_memory_retriever()
    )

    official_result, approved_result = _retrieve_evidence(state, ok_retriever, arm_retriever)

    # Transient, per-call snapshot only -- never persisted, never logged,
    # discarded when this function returns.
    state = state.with_retrieval(official_result, approved_result)

    grounding = evidence.assemble_grounding(resolved_identifying_context, official_result)
    style_reference_texts = evidence.assemble_style_reference_texts(approved_result)

    enriched_request = dataclasses.replace(
        request,
        # M5D.2: official-only, never merged with identifying context --
        # see prompt.py's separate "OFFICIAL KNOWLEDGE" / "IDENTIFYING
        # CONTEXT" sections.
        trusted_grounding_text=grounding.official_grounding_text,
        identifying_context=grounding.identifying_context,
        # M5D.1: strictly official-knowledge-only -- identifying context
        # and Approved Response Memory must never flip this, alone or
        # combined (see evidence.py's module docstring).
        trusted_grounding_available=grounding.official_grounding_available,
        style_reference_texts=style_reference_texts,
    )

    initial_result, initial_draft = generate_once_with_draft(enriched_request, client=client, max_retries=max_retries)
    initial_passed = initial_result.generation_status == "SUCCEEDED"

    final_result = initial_result
    refinement_attempted = False
    refinement_draft: str | None = None
    refinement_generation_result: GenerationResult | None = None

    if not initial_passed and is_refinement_eligible(initial_result.generation_status, initial_result.reason_code):
        # `is_refinement_eligible` only returns True for a genuine
        # validation-gate failure, which only happens after a draft was
        # successfully parsed -- `initial_draft` is guaranteed non-None here.
        refinement_attempted = True
        refinement_request = dataclasses.replace(
            enriched_request,
            refinement_context=RefinementContext(
                previous_draft=initial_draft,
                failure_reason_code=initial_result.reason_code,
            ),
        )
        final_result, refinement_draft = generate_once_with_draft(
            refinement_request, client=client, max_retries=max_retries
        )
        refinement_generation_result = final_result
        # Hard invariant: no third attempt, no recursion, no loop --
        # whatever `final_result` is now (pass or fail) IS the answer.

    final_passed = final_result.generation_status == "SUCCEEDED"
    metadata = AgentExecutionMetadata(
        execution_id=execution_id,
        logical_generation_count=2 if refinement_attempted else 1,
        refinement_attempted=refinement_attempted,
        initial_validation_passed=initial_passed,
        initial_failure_code=None if initial_passed else initial_result.reason_code,
        final_validation_passed=final_passed,
        final_failure_code=None if final_passed else final_result.reason_code,
        grounding_mode=grounding.mode,
        official_grounding_available=grounding.official_grounding_available,
        official_retrieval_outcome=official_result.outcome,
        official_result_count=len(official_result.results),
        style_retrieval_outcome=approved_result.outcome,
        style_result_count=len(approved_result.results),
        generation_status=final_result.generation_status,
        identifying_context_allowed=identifying_context_decision.allowed,
        identifying_context_decision_reason=identifying_context_decision.reason,
        identifying_context_matched_reference=identifying_context_decision.matched_reference,
    )

    if audit_repository is not None:
        _persist_audit_trace(audit_repository, execution_id, request.comment_id, metadata)

    trace: AgentExecutionTrace | None = None
    if capture_trace:
        trace = AgentExecutionTrace(
            identifying_context=resolved_identifying_context,
            identifying_context_decision_reason=identifying_context_decision.reason,
            official_knowledge_result=official_result,
            approved_response_result=approved_result,
            official_grounding_text=grounding.official_grounding_text,
            style_reference_texts=style_reference_texts,
            initial_draft=initial_draft,
            initial_generation_result=initial_result,
            refinement_draft=refinement_draft,
            refinement_generation_result=refinement_generation_result,
        )

    return AgentExecutionResult(generation_result=final_result, metadata=metadata, trace=trace)


def _persist_audit_trace(
    audit_repository: AgentAuditRepository,
    execution_id: str,
    review_item_id: str,
    metadata: AgentExecutionMetadata,
) -> None:
    record = build_audit_record(
        execution_id=execution_id,
        review_item_id=review_item_id,
        created_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
        grounding_mode=metadata.grounding_mode,
        official_retrieval_outcome=metadata.official_retrieval_outcome,
        official_result_count=metadata.official_result_count,
        official_grounding_available=metadata.official_grounding_available,
        style_retrieval_outcome=metadata.style_retrieval_outcome,
        style_result_count=metadata.style_result_count,
        logical_generation_count=metadata.logical_generation_count,
        refinement_attempted=metadata.refinement_attempted,
        initial_validation_passed=metadata.initial_validation_passed,
        initial_failure_code=metadata.initial_failure_code,
        final_validation_passed=metadata.final_validation_passed,
        final_failure_code=metadata.final_failure_code,
        generation_status=metadata.generation_status,
    )
    try:
        audit_repository.record(record)
    except Exception as exc:  # noqa: BLE001 -- observability-store failure must never break generation
        logger.warning(
            "agent audit trace persistence failed: execution_id=%s review_item_id=%s error_type=%s",
            execution_id,
            review_item_id,
            type(exc).__name__,
        )


def generate_response_with_orchestrator(
    request: GenerationRequest,
    client: SupportsGenerate | None = None,
    max_retries: int | None = None,
    *,
    official_knowledge_retriever: OfficialKnowledgeRetriever | None = None,
    approved_response_retriever: ApprovedResponseMemoryRetriever | None = None,
    audit_repository: AgentAuditRepository | None = None,
) -> GenerationResult:
    return run_agent_pipeline(
        request,
        client,
        max_retries,
        official_knowledge_retriever=official_knowledge_retriever,
        approved_response_retriever=approved_response_retriever,
        audit_repository=audit_repository,
    ).generation_result
