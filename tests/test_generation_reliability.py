"""
Phase M5A.2 — Gemini generation reliability tests: explicit timeout,
retry taxonomy, and bounded backoff.

Uses REAL httpx / google.genai.errors exception types (constructed
directly, no live network call) to prove the classifier against actual
SDK shapes rather than guessed exception names — the classifier itself
lives in rcs.evaluation.genai_client (see that module for why).

No prompt, routing, validation, or persistence behavior is exercised
differently here than in tests/test_response_generation.py; those
suites remain the source of truth for that behavior and are re-run
unchanged as part of this phase's regression baseline.
"""

from __future__ import annotations

import json

import httpx
import pytest
from google.genai import errors as genai_errors

from config import settings
from rcs.evaluation.genai_client import (
    AUTHENTICATION,
    INVALID_REQUEST,
    PERMISSION,
    RATE_LIMITED,
    SERVICE_UNAVAILABLE,
    TIMEOUT,
    TRANSIENT_NETWORK,
    UNKNOWN,
    GenAIClient,
    classify_gemini_error,
    is_retryable_category,
)
from rcs.response_generation import service as service_module
from rcs.response_generation.service import GenerationRequest, generate_suggested_response

from test_response_generation import FakeGeminiClient, _draft_json  # noqa: E402 -- reuse the existing fake

THANKS_DRAFT = "شكرًا لكلماتك الطيبة عن العمل على المعجم."


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


@pytest.fixture
def no_real_sleep(monkeypatch):
    """The one seam: replace the module-level _sleep, never the global
    time.sleep. Records every requested duration so backoff timing can
    be asserted without a single real second elapsing."""
    calls: list[float] = []
    monkeypatch.setattr(service_module, "_sleep", lambda seconds: calls.append(seconds))
    return calls


# --- classifier: real SDK exception shapes ---------------------------------


def test_classify_timeout():
    assert classify_gemini_error(httpx.ReadTimeout("timed out")) == TIMEOUT
    assert classify_gemini_error(httpx.ConnectTimeout("timed out")) == TIMEOUT


def test_classify_transient_network():
    assert classify_gemini_error(httpx.ConnectError("connection refused")) == TRANSIENT_NETWORK


def test_classify_rate_limited():
    assert classify_gemini_error(genai_errors.ClientError(429, {"message": "quota"})) == RATE_LIMITED


def test_classify_service_unavailable():
    assert classify_gemini_error(genai_errors.ServerError(503, {"message": "unavailable"})) == SERVICE_UNAVAILABLE
    assert classify_gemini_error(genai_errors.ServerError(500, {"message": "internal"})) == SERVICE_UNAVAILABLE


def test_classify_authentication():
    assert classify_gemini_error(genai_errors.ClientError(401, {"message": "bad key"})) == AUTHENTICATION


def test_classify_permission():
    assert classify_gemini_error(genai_errors.ClientError(403, {"message": "forbidden"})) == PERMISSION


def test_classify_invalid_request():
    assert classify_gemini_error(genai_errors.ClientError(400, {"message": "bad request"})) == INVALID_REQUEST
    assert classify_gemini_error(genai_errors.ClientError(404, {"message": "not found"})) == INVALID_REQUEST


def test_classify_unknown_for_unrecognized_exception():
    assert classify_gemini_error(ValueError("some programming error")) == UNKNOWN
    assert classify_gemini_error(RuntimeError("generic")) == UNKNOWN
    assert classify_gemini_error(KeyError("oops")) == UNKNOWN


def test_retryable_categories():
    for category in (TIMEOUT, RATE_LIMITED, TRANSIENT_NETWORK, SERVICE_UNAVAILABLE):
        assert is_retryable_category(category) is True
    for category in (AUTHENTICATION, PERMISSION, INVALID_REQUEST, UNKNOWN):
        assert is_retryable_category(category) is False


# --- 1. success on first attempt --------------------------------------------


def test_success_on_first_attempt_no_backoff(no_real_sleep):
    client = FakeGeminiClient([_draft_json(THANKS_DRAFT)])
    result = generate_suggested_response(_req(), client, max_retries=2)
    assert result.generation_status == "SUCCEEDED"
    assert client.call_count == 1
    assert no_real_sleep == []


# --- 2/3/4. retryable failure then success ----------------------------------


@pytest.mark.parametrize(
    "first_failure",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("connection refused"),
        genai_errors.ServerError(503, {"message": "unavailable"}),
        genai_errors.ClientError(429, {"message": "quota"}),
    ],
    ids=["timeout", "transient_network", "service_unavailable", "rate_limited"],
)
def test_retryable_failure_then_success(no_real_sleep, first_failure):
    client = FakeGeminiClient([first_failure, _draft_json(THANKS_DRAFT)])
    result = generate_suggested_response(_req(), client, max_retries=2)
    assert result.generation_status == "SUCCEEDED"
    assert result.suggested_response == THANKS_DRAFT
    assert client.call_count == 2
    assert len(no_real_sleep) == 1  # exactly one backoff wait, between the two attempts


# --- 5. exhausted retryable failures -----------------------------------------


def test_exhausted_retryable_failures_fails_closed(no_real_sleep):
    client = FakeGeminiClient([genai_errors.ServerError(503, {"message": "unavailable"})] * 3)
    result = generate_suggested_response(_req(), client, max_retries=2)
    assert result.generation_status == "FAILED_REVIEW_ONLY"
    assert result.routing_decision == "REVIEW_ONLY"
    assert result.reason_code == "gemini_api_error"  # unchanged external reason_code, per instruction
    assert result.suggested_response is None
    assert client.call_count == 3  # max_retries=2 -> 3 total attempts, unchanged contract
    assert len(no_real_sleep) == 2  # backoff between attempts 1->2 and 2->3, never after the 3rd


# --- 6/7/8. non-retryable failures are not retried ---------------------------


@pytest.mark.parametrize(
    "failure,label",
    [
        (genai_errors.ClientError(401, {"message": "bad key"}), "authentication"),
        (genai_errors.ClientError(403, {"message": "forbidden"}), "permission"),
        (genai_errors.ClientError(400, {"message": "bad request"}), "invalid_request"),
    ],
)
def test_non_retryable_failure_is_not_retried(no_real_sleep, failure, label):
    client = FakeGeminiClient([failure])  # would raise AssertionError if called a second time
    result = generate_suggested_response(_req(), client, max_retries=2)
    assert result.generation_status == "FAILED_REVIEW_ONLY"
    assert result.reason_code == "gemini_api_error"
    assert client.call_count == 1
    assert no_real_sleep == []


# --- 9. unknown/programming exception is not blindly retried ----------------


def test_unknown_exception_is_not_blindly_retried(no_real_sleep):
    client = FakeGeminiClient([RuntimeError("some unrecognized failure")])
    result = generate_suggested_response(_req(), client, max_retries=2)
    assert result.generation_status == "FAILED_REVIEW_ONLY"
    assert client.call_count == 1
    assert no_real_sleep == []


# --- 10/11/12. backoff placement ---------------------------------------------


def test_no_backoff_before_first_attempt(no_real_sleep):
    client = FakeGeminiClient([_draft_json(THANKS_DRAFT)])
    generate_suggested_response(_req(), client, max_retries=2)
    assert no_real_sleep == []


def test_backoff_only_between_retryable_attempts(no_real_sleep):
    client = FakeGeminiClient([
        genai_errors.ServerError(503, {"message": "unavailable"}),
        genai_errors.ServerError(503, {"message": "unavailable"}),
        _draft_json(THANKS_DRAFT),
    ])
    result = generate_suggested_response(_req(), client, max_retries=2)
    assert result.generation_status == "SUCCEEDED"
    assert client.call_count == 3
    assert len(no_real_sleep) == 2  # one wait after attempt 1, one after attempt 2 -- none after attempt 3 (success)


def test_no_backoff_after_final_attempt(no_real_sleep):
    client = FakeGeminiClient([genai_errors.ServerError(503, {"message": "unavailable"})] * 3)
    generate_suggested_response(_req(), client, max_retries=2)
    # 3 attempts, all fail -> exactly 2 backoff waits (between 1->2 and 2->3), never a 3rd after the loop ends
    assert len(no_real_sleep) == 2


def test_backoff_values_are_bounded_exponential(no_real_sleep):
    client = FakeGeminiClient([genai_errors.ServerError(503, {"message": "x"})] * 3)
    generate_suggested_response(_req(), client, max_retries=2)
    base = settings.GEMINI_RETRY_BACKOFF_BASE_SECONDS
    cap = settings.GEMINI_RETRY_BACKOFF_MAX_SECONDS
    assert no_real_sleep[0] == min(base, cap)
    assert no_real_sleep[1] == min(base * 2, cap)
    assert all(w <= cap for w in no_real_sleep)


# --- 13. exact attempt count: max_retries=2 -> at most 3 attempts -----------


@pytest.mark.parametrize("max_retries,failures,expected_attempts", [
    (0, 1, 1),
    (2, 3, 3),
])
def test_exact_attempt_count_bound(no_real_sleep, max_retries, failures, expected_attempts):
    client = FakeGeminiClient([genai_errors.ServerError(503, {"message": "x"})] * failures)
    generate_suggested_response(_req(), client, max_retries=max_retries)
    assert client.call_count == expected_attempts


# --- 14. configured timeout reaches the SDK/client boundary -----------------


def test_configured_timeout_reaches_the_sdk_call(monkeypatch):
    """GenAIClient must forward timeout_seconds to interactions.create's
    own `timeout` parameter -- proven without any real network call by
    stubbing the client construction and asserting the exact kwarg."""
    captured = {}

    class _FakeInteraction:
        output_text = _draft_json(THANKS_DRAFT)

    class _FakeInteractions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeInteraction()

    class _FakeSdkClient:
        interactions = _FakeInteractions()

    client = GenAIClient(api_key="fake-key-not-a-real-secret", model_name="gemini-3.6-flash", timeout_seconds=7.5)
    client._client = _FakeSdkClient()  # bypass genai.Client(...) construction entirely
    result = client.generate("prompt text", {"type": "object"})
    assert result == _draft_json(THANKS_DRAFT)
    assert captured["timeout"] == 7.5


def test_no_timeout_seconds_means_no_timeout_kwarg_sent():
    """ai_suggester.py's existing construction (no timeout_seconds) must
    keep sending nothing extra to the SDK -- unaffected by this phase."""
    captured = {}

    class _FakeInteraction:
        output_text = _draft_json(THANKS_DRAFT)

    class _FakeInteractions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeInteraction()

    class _FakeSdkClient:
        interactions = _FakeInteractions()

    client = GenAIClient(api_key="fake-key-not-a-real-secret", model_name="gemini-3.6-flash")
    client._client = _FakeSdkClient()
    client.generate("prompt text", {"type": "object"})
    assert "timeout" not in captured


def test_build_live_gemini_client_passes_configured_timeout(monkeypatch):
    from rcs.response_generation.service import build_live_gemini_client

    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key-not-a-real-secret")
    client = build_live_gemini_client()
    assert client._timeout_seconds == settings.GEMINI_REQUEST_TIMEOUT_SECONDS


# --- 15. malformed JSON behavior unchanged -----------------------------------


def test_malformed_json_behavior_unchanged(no_real_sleep):
    client = FakeGeminiClient(["not-json"])
    result = generate_suggested_response(_req(), client, max_retries=0)
    assert result.reason_code == "malformed_gemini_output"
    assert result.suggested_response is None
    assert client.call_count == 1  # a successful HTTP call with a bad body -- never transport-retried
    assert no_real_sleep == []


# --- 16. validate_draft failure behavior unchanged ---------------------------


def test_validate_draft_failure_behavior_unchanged(no_real_sleep):
    client = FakeGeminiClient([_draft_json("شكرًا لتواصلكم على 0501234567 مع المعجم.")])
    result = generate_suggested_response(_req(), client, max_retries=0)
    assert result.reason_code == "pii_in_draft"
    assert result.suggested_response is None
    assert client.call_count == 1  # a valid schema response that fails validation is not retried either
    assert no_real_sleep == []


# --- 17. REVIEW_ONLY fail-closed behavior unchanged --------------------------


def test_review_only_fail_closed_behavior_unchanged():
    client = FakeGeminiClient([])  # would raise AssertionError if ever called
    result = generate_suggested_response(_req(primary_intent=None, topic_subtype="OFF_TOPIC_PERSONAL"), client)
    assert result.routing_decision == "REVIEW_ONLY"
    assert client.call_count == 0


# --- 18. orchestrator parity still holds -------------------------------------


def test_orchestrator_parity_still_holds_for_retryable_then_success(no_real_sleep):
    from rcs.response_intelligence.orchestrator import generate_response_with_orchestrator

    scripted = [genai_errors.ServerError(503, {"message": "unavailable"}), _draft_json(THANKS_DRAFT)]
    client_direct = FakeGeminiClient(list(scripted))
    client_orchestrated = FakeGeminiClient(list(scripted))

    result_direct = generate_suggested_response(_req(), client_direct, max_retries=2)
    result_orchestrated = generate_response_with_orchestrator(_req(), client_orchestrated, max_retries=2)

    direct_dict = result_direct.to_dict()
    orchestrated_dict = result_orchestrated.to_dict()
    direct_dict.pop("generated_at")
    orchestrated_dict.pop("generated_at")
    assert direct_dict == orchestrated_dict
    assert client_direct.call_count == client_orchestrated.call_count == 2
