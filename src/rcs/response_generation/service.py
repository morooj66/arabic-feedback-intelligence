"""
Response-generation orchestration: eligibility → (optional) Gemini → validate.

Never sends to the visitor. Non-actionable comments skip Gemini.
API/validation failures become REVIEW_ONLY with no invented draft.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Protocol

from pydantic import ValidationError

from config import settings
from rcs.evaluation.genai_client import classify_gemini_error, is_retryable_category
from rcs.response_generation.eligibility import (
    EligibilityInput,
    GENERATE_DRAFT,
    NO_RESPONSE_RECOMMENDED,
    REVIEW_ONLY,
    determine_response_eligibility,
)
from rcs.response_generation.outbound_privacy import OutboundPrivacyError, guard_prompt
from rcs.response_generation.prompt import build_generation_prompt
from rcs.response_generation.refinement import RefinementContext
from rcs.response_generation.schema import JSON_SCHEMA, GeminiDraftResponse, GenerationResult
from rcs.response_generation.validate import validate_draft

if TYPE_CHECKING:
    # Type-only: rcs.response_intelligence depends on this module (via
    # the orchestrator), so a real top-level import here would be
    # circular. `from __future__ import annotations` (above) means this
    # annotation is never evaluated at runtime -- see identifying_metadata
    # field below.
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, response_schema: dict) -> str: ...


@dataclass(frozen=True)
class GenerationRequest:
    comment_id: str
    comment_text: str
    primary_intent: Optional[str] = None
    topic_subtype: Optional[str] = None
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    needs_review: bool = False
    is_content_meaningful: bool = True
    insufficient_context: bool = False
    trusted_grounding_available: bool = False
    trusted_grounding_text: Optional[str] = None
    # Phase M5D.2 — identifying context (title/entry_id/lemma/case
    # identity) ONLY: which entry/case is being discussed, never a fact
    # about it. Kept separate from trusted_grounding_text (which, as of
    # M5D.2, carries ONLY official factual evidence) so the prompt can
    # label the two honestly instead of concatenating them under one
    # "grounding" block. Defaults to None: every existing direct caller
    # that never sets this is unaffected.
    #
    # Phase M6.2 — LEGACY as of the Identifying Context Policy: this
    # field (and trusted_grounding_text, when used for the same purpose)
    # is no longer read by rcs.response_intelligence.orchestrator for
    # identifying-context decisions at all. It remains here ONLY because
    # rcs.response_generation.service.generate_once_with_draft (called
    # directly, bypassing the orchestrator) still renders whatever text
    # is here as-is — a pre-existing, legitimate calling convention for
    # tests/tools that intentionally bypass the Response Agent. Any
    # caller that goes through run_agent_pipeline must use
    # `identifying_metadata` below instead; this field is never consulted
    # by that path and cannot be used to bypass the new policy.
    identifying_context: Optional[str] = None
    # Phase M6.2 — the canonical, structured input to
    # rcs.response_intelligence.identifying_context_policy.
    # evaluate_identifying_context, consumed ONLY by
    # rcs.response_intelligence.orchestrator.run_agent_pipeline. Defaults
    # to None: every existing direct (non-orchestrator) caller is
    # unaffected.
    identifying_metadata: Optional["IdentifyingMetadata"] = None
    classification_routing_decision: Optional[str] = None
    # Phase M5C — Approved Response Memory style/pattern examples ONLY
    # (never factual grounding). Defaults to `()`, which makes
    # `build_generation_prompt`'s output byte-identical to pre-M5C —
    # every existing caller that never sets this is completely unaffected.
    style_reference_texts: tuple[str, ...] = ()
    # Phase M5D — set ONLY by the orchestrator's bounded refinement
    # pass (see rcs.response_intelligence.orchestrator). Defaults to
    # None, which makes build_generation_prompt's output byte-identical
    # to pre-M5D for every existing caller and every initial draft.
    refinement_context: Optional[RefinementContext] = None


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _actions_for(decision: str, has_draft: bool) -> tuple[str, ...]:
    if has_draft:
        return settings.FUTURE_REVIEWER_RESPONSE_ACTIONS
    if decision == NO_RESPONSE_RECOMMENDED:
        return ("NO_RESPONSE_NEEDED",)
    return ("NO_RESPONSE_NEEDED",)


def _skipped(
    req: GenerationRequest,
    decision: str,
    reason: str,
    rule_id: str,
) -> GenerationResult:
    return GenerationResult(
        comment_id=req.comment_id,
        routing_decision=decision,
        generation_status="SKIPPED",
        suggested_response=None,
        review_required=True,
        reason_code=reason,
        provider=None,
        model_name=None,
        prompt_version=None,
        generated_at=_now(),
        future_reviewer_actions=_actions_for(decision, False),
        eligibility_rule_id=rule_id,
    )


def _failed(req: GenerationRequest, reason: str, error: str | None, rule_id: str | None) -> GenerationResult:
    return GenerationResult(
        comment_id=req.comment_id,
        routing_decision=REVIEW_ONLY,
        generation_status="FAILED_REVIEW_ONLY",
        suggested_response=None,
        review_required=True,
        reason_code=reason,
        provider=settings.RESPONSE_GENERATION_PROVIDER,
        model_name=settings.RESPONSE_GENERATION_MODEL_NAME,
        prompt_version=settings.RESPONSE_GENERATION_PROMPT_VERSION,
        generated_at=_now(),
        future_reviewer_actions=_actions_for(REVIEW_ONLY, False),
        eligibility_rule_id=rule_id,
        error=error,
    )


def _sleep(seconds: float) -> None:
    """The one seam between _call_with_retry's backoff and the real
    clock — tests monkeypatch this module-level function directly
    (`monkeypatch.setattr(service, "_sleep", ...)`) rather than patching
    the global `time.sleep`, so nothing outside this module's own retry
    loop is ever affected."""
    time.sleep(seconds)


def _backoff_seconds(attempt_index: int) -> float:
    """Bounded exponential backoff, no jitter (see settings.py's
    GEMINI_RETRY_BACKOFF_* comment for why jitter is skipped at this call
    volume). attempt_index is 0-based: 0 -> BASE, 1 -> BASE*2, ... capped
    at MAX."""
    base = settings.GEMINI_RETRY_BACKOFF_BASE_SECONDS
    cap = settings.GEMINI_RETRY_BACKOFF_MAX_SECONDS
    return min(base * (2**attempt_index), cap)


def _call_with_retry(client: SupportsGenerate, prompt: str, max_retries: int) -> tuple[str | None, str | None]:
    """Same attempt-count contract as before Phase M5A.2: at most
    max_retries + 1 total attempts. What changed: an attempt is only
    retried when classify_gemini_error says the failure is transient
    (TIMEOUT/RATE_LIMITED/TRANSIENT_NETWORK/SERVICE_UNAVAILABLE) --
    AUTHENTICATION/PERMISSION/INVALID_REQUEST/UNKNOWN stop immediately,
    since retrying a permanent or unrecognized failure three times just
    repeats a guaranteed failure. Backoff (_backoff_seconds) only runs
    between two retryable attempts -- never before the first attempt,
    never after the loop's final attempt either way."""
    prompt = guard_prompt(prompt)      # FINAL egress boundary for the initial request, every retry and the refinement request: raises OutboundPrivacyError, no request is made
    last_error: str | None = None
    attempts = max_retries + 1
    for attempt_index in range(attempts):
        try:
            return client.generate(prompt, JSON_SCHEMA), None
        except Exception as exc:  # noqa: BLE001 — caught broadly so classify_gemini_error can inspect any shape; the classification below decides retry, not this except clause
            last_error = f"{type(exc).__name__}: {exc}"
            is_last_attempt = attempt_index == attempts - 1
            if is_last_attempt or not is_retryable_category(classify_gemini_error(exc)):
                break
            _sleep(_backoff_seconds(attempt_index))
    return None, last_error


def build_live_gemini_client():
    """Opt-in live client. Unit tests must inject a fake instead."""
    from rcs.evaluation.genai_client import GenAIClient

    return GenAIClient(
        settings.GEMINI_API_KEY,
        settings.RESPONSE_GENERATION_MODEL_NAME,
        timeout_seconds=settings.GEMINI_REQUEST_TIMEOUT_SECONDS,
    )


def generate_once_with_draft(
    request: GenerationRequest,
    client: SupportsGenerate | None = None,
    max_retries: int | None = None,
) -> tuple[GenerationResult, Optional[str]]:
    """Phase M5D: the exact body `generate_suggested_response` has always
    had, factored out ONLY so the orchestrator's bounded refinement pass
    can see the raw draft text even when it fails validation (needed to
    build the refinement prompt's "previous draft" block) — `GenerationResult`
    itself never carries a rejected draft (see schema.py, unchanged).
    `generate_suggested_response` below is unchanged in behavior: same
    logic, same order, same return value for its one output.

    Second tuple element is the parsed, stripped draft text whenever
    Gemini's output was successfully parsed as valid JSON — regardless
    of whether it then passed validation — and `None` for every case
    where no draft was ever parsed (skipped, no client, transport
    failure, malformed JSON)."""
    retries = settings.GEMINI_MAX_RETRIES if max_retries is None else max_retries
    eligibility = determine_response_eligibility(
        EligibilityInput(
            comment_text=request.comment_text,
            primary_intent=request.primary_intent,
            topic_subtype=request.topic_subtype,
            sentiment=request.sentiment,
            confidence=request.confidence,
            needs_review=request.needs_review,
            is_content_meaningful=request.is_content_meaningful,
            insufficient_context=request.insufficient_context,
            trusted_grounding_available=request.trusted_grounding_available,
            classification_routing_decision=request.classification_routing_decision,
        )
    )

    if eligibility.routing_decision != GENERATE_DRAFT:
        return (
            _skipped(
                request,
                eligibility.routing_decision,
                eligibility.reason_code,
                eligibility.rule_id,
            ),
            None,
        )

    if client is None:
        return (
            _failed(
                request,
                "gemini_client_missing",
                "No Gemini client provided; live construction is opt-in at the caller.",
                eligibility.rule_id,
            ),
            None,
        )

    prompt = build_generation_prompt(
        comment_text=request.comment_text,
        primary_intent=request.primary_intent,
        topic_subtype=request.topic_subtype,
        sentiment=request.sentiment,
        reason_code=eligibility.reason_code,
        trusted_grounding_text=request.trusted_grounding_text,
        identifying_context=request.identifying_context,
        style_reference_texts=request.style_reference_texts,
        refinement_context=request.refinement_context,
    )
    try:
        raw, api_error = _call_with_retry(client, prompt, retries)
    except OutboundPrivacyError as exc:       # fail closed: nothing left the machine, no retry, not refinable
        return _failed(request, "outbound_privacy_blocked", exc.reason, eligibility.rule_id), None
    if raw is None:
        return _failed(request, "gemini_api_error", api_error, eligibility.rule_id), None

    try:
        parsed = GeminiDraftResponse.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        return _failed(request, "malformed_gemini_output", str(exc), eligibility.rule_id), None

    draft = parsed.suggested_response.strip()

    ok, why = validate_draft(
        parsed.suggested_response,
        trusted_grounding_text=request.trusted_grounding_text,
        comment_text=request.comment_text,
    )
    if not ok:
        return _failed(request, why or "draft_validation_failed", why, eligibility.rule_id), draft

    return (
        GenerationResult(
            comment_id=request.comment_id,
            routing_decision=GENERATE_DRAFT,
            generation_status="SUCCEEDED",
            suggested_response=draft,
            review_required=True,
            reason_code=eligibility.reason_code,
            provider=settings.RESPONSE_GENERATION_PROVIDER,
            model_name=settings.RESPONSE_GENERATION_MODEL_NAME,
            prompt_version=settings.RESPONSE_GENERATION_PROMPT_VERSION,
            generated_at=_now(),
            future_reviewer_actions=_actions_for(GENERATE_DRAFT, True),
            eligibility_rule_id=eligibility.rule_id,
        ),
        draft,
    )


def generate_suggested_response(
    request: GenerationRequest,
    client: SupportsGenerate | None = None,
    max_retries: int | None = None,
) -> GenerationResult:
    result, _draft = generate_once_with_draft(request, client, max_retries)
    return result
