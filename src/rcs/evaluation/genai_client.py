"""
Phase B.2 — thin wrapper around the CURRENT Google GenAI SDK (`google-genai`
PyPI package, `from google import genai`), used ONLY by Phase B.2's
AI-assisted suggestion orchestration (ai_suggester.py).

Deliberately a SEPARATE file from src/rcs/classification/gemini_client.py,
which wraps the deprecated `google.generativeai` SDK for Phase 4's live
classification. Phase 4 has never made a live call and updating it is out
of scope for this change (explicit instruction: Phase B.2 only) — it stays
on its old SDK/model until it is itself revisited.

Why the switch: a live local smoke test against gemini-2.5-flash returned
404 (that model is no longer available to new API keys). Per
ai.google.dev/gemini-api/docs (checked live, 2026-08): gemini-2.5-flash
shuts down Oct 2026; gemini-3.6-flash is GA and is the current recommended
Flash model. The old `google.generativeai` SDK is itself deprecated in
favor of `google-genai`, whose current documented structured-output path
is `client.interactions.create(...)`, not the old
`model.generate_content(..., generation_config={...})` — this wrapper uses
that current path.

Same lazy-import discipline as gemini_client.py: `google.genai` is
imported inside the method that needs it, not at module import time, so
importing this module never requires the package installed or an API key
set, and constructing a GenAIClient without calling `.generate()` is
always safe (e.g. for wiring/config checks, or the mocked tests in
tests/test_ai_suggestion.py, which inject a fake client and never
construct this class at all).

Implements the same `generate(prompt, response_schema) -> str` shape as
GeminiClient, so it's a drop-in for ai_suggester.py's SupportsGenerate
protocol — no other Phase B.2 file needed to change for this migration.

Phase M5A.2 — two additions, both opt-in via constructor arguments so
ai_suggester.py's existing construction (api_key=, model_name= only) is
completely unaffected:

  1. `timeout_seconds`: when given, passed straight to google-genai's
     own `client.interactions.create(..., timeout=...)` parameter
     (confirmed present on the installed SDK by inspecting
     `Interactions.create`'s signature — a plain float, in seconds,
     understood by the underlying httpx client). Per-attempt, not a
     total budget: response_generation/service.py's retry loop calls
     `generate()` again for each attempt, and each call gets this same
     full timeout again, since a retry is a fresh HTTP request with no
     partial progress to preserve.
  2. `classify_gemini_error`/`is_retryable_category`: a small, explicit
     boundary for "was this failure worth retrying" that knows about
     google-genai/httpx exception shapes so no other file has to. Based
     on inspecting the installed SDK (google-genai 2.18.1): HTTP error
     responses raise `google.genai.errors.ClientError` (4xx) or
     `ServerError` (5xx), both exposing a `.code` int matching the HTTP
     status; anything below the HTTP layer (timeout, connection refused,
     DNS failure) raises a plain `httpx` exception instead, since no
     response was ever received to wrap. Both imports stay lazy, for
     the same reason `_ensure_client` does — importing this module must
     never require the packages installed.
"""

from __future__ import annotations

# --- failure classification (Phase M5A.2) ----------------------------------
#
# One of these four for a failure worth retrying at the transport level;
# anything else is not retried automatically, including UNKNOWN -- an
# unrecognized exception is far more likely a programming/configuration
# error than a transient one, and retrying it would just repeat a
# guaranteed failure three times instead of once. See classify_gemini_error.
TIMEOUT = "TIMEOUT"
RATE_LIMITED = "RATE_LIMITED"
TRANSIENT_NETWORK = "TRANSIENT_NETWORK"
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
AUTHENTICATION = "AUTHENTICATION"
PERMISSION = "PERMISSION"
INVALID_REQUEST = "INVALID_REQUEST"
UNKNOWN = "UNKNOWN"

_RETRYABLE_CATEGORIES = frozenset({TIMEOUT, RATE_LIMITED, TRANSIENT_NETWORK, SERVICE_UNAVAILABLE})


def classify_gemini_error(exc: BaseException) -> str:
    """Maps a raised exception to one of the category constants above.
    Never raises; an exception type this function doesn't recognize
    (including from a test double that doesn't use the real SDK at all)
    always falls through to UNKNOWN, never to a guess."""
    try:
        import httpx
    except ImportError:
        httpx = None  # pragma: no cover -- httpx ships with google-genai

    if httpx is not None:
        if isinstance(exc, httpx.TimeoutException):
            return TIMEOUT
        if isinstance(exc, httpx.TransportError):
            return TRANSIENT_NETWORK

    try:
        from google.genai import errors as genai_errors
    except ImportError:
        genai_errors = None  # pragma: no cover -- declared dependency

    if genai_errors is not None:
        if isinstance(exc, genai_errors.ClientError):
            code = getattr(exc, "code", None)
            if code == 401:
                return AUTHENTICATION
            if code == 403:
                return PERMISSION
            if code == 429:
                return RATE_LIMITED
            return INVALID_REQUEST
        if isinstance(exc, genai_errors.ServerError):
            return SERVICE_UNAVAILABLE

    return UNKNOWN


def is_retryable_category(category: str) -> bool:
    return category in _RETRYABLE_CATEGORIES


class GenAIClient:
    def __init__(self, api_key: str | None, model_name: str, timeout_seconds: float | None = None):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Set it in the environment (see .env / "
                ".env.example) before constructing a live GenAIClient."
            )
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            from google import genai  # lazy import, see module docstring

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, response_schema: dict) -> str:
        """
        Sends one structured-output request via the current Interactions
        API (client.interactions.create), the documented path for
        structured JSON output in the current google-genai SDK. Returns
        the response's raw JSON text (interaction.output_text). Raises on
        any network/API-level failure — callers are responsible for
        catching and falling back safely (see ai_suggester.py and
        rcs.response_generation.service._call_with_retry).
        """
        client = self._ensure_client()
        kwargs: dict = dict(
            model=self._model_name,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": response_schema,
            },
        )
        if self._timeout_seconds is not None:
            kwargs["timeout"] = self._timeout_seconds
        interaction = client.interactions.create(**kwargs)
        return interaction.output_text
