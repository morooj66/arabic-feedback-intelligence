"""Gemini clients behind one tiny interface. Only MockGeminiClient is used in this phase. LiveGeminiClient exists so the live stage is a configuration
change, but it refuses to be constructed without a written LiveApproval and has NEVER been executed (UNVERIFIED against the live API)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import GenerationSettings
from .pricing import Usage, estimate_tokens
from .privacy import LiveApproval, RealDataNotApproved
from .schema import RESPONSE_SCHEMA


class ClientError(Exception):
    def __init__(self, code: str, retryable: bool = False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ClientResponse:
    text: str
    usage: Usage


class ClassifierClient(Protocol):
    def classify(self, prompt: str, schema: dict, settings: GenerationSettings) -> ClientResponse: ...


# ------------------------------------------------------------------------------------------------------------------------ mock
_BLOCK = re.compile(r"--- UNTRUSTED VISITOR COMMENT ---\n(.*?)\n--- END UNTRUSTED VISITOR COMMENT ---", re.S)


def extract_comment(prompt: str) -> str:
    m = _BLOCK.search(prompt)
    return m.group(1) if m else ""


def rule_based_responder(comment: str) -> dict:
    """A deterministic keyword stand-in for the model, used ONLY to exercise the harness on synthetic comments. It makes no claim about Gemini."""
    c = comment
    has_arabic = bool(re.search(r"[؀-ۿ]", c))
    if not has_arabic and len(c) < 25:
        return dict(intent="SPAM", topic="GENERAL", sentiment="NEUTRAL", ambiguous=True, confidence="low", reason="no readable content")
    if re.search(r"ترشيح|كلمة العام", c):
        return dict(intent="SUGGESTION", topic="WORD_OF_YEAR_NOMINATION", sentiment="NEUTRAL", ambiguous=False, confidence="medium", reason="explicit nomination")
    if re.search(r"لا يفتح|لا يعمل|لا تفتح|بطيء|خلل", c):
        return dict(intent="COMPLAINT", topic="TECHNICAL_ISSUE", sentiment="NEGATIVE", ambiguous=False, confidence="medium", reason="failure report")
    if re.search(r"غير صحيح|خطأ في تعريف|تصحيح|مكرر", c):
        return dict(intent="SUGGESTION", topic="ENTRY_CORRECTION", sentiment="NEUTRAL", ambiguous=False, confidence="medium", reason="existing entry reported wrong")
    if re.search(r"شكر|جزاكم|بارك|رائع", c):
        return dict(intent="THANKS_PRAISE", topic="GENERAL", sentiment="POSITIVE", ambiguous=False, confidence="high", reason="gratitude")
    if "؟" in c or re.search(r"^(ما معنى|كيف|هل|لماذا)", c.strip()):
        return dict(intent="INQUIRY", topic="GENERAL", sentiment="NEUTRAL", ambiguous=False, confidence="medium", reason="question")
    if re.search(r"أقترح|اقترح|أتمنى|إضافة|اضافة", c):
        topic = "WORD_SUBMISSION" if "كلمة" in c else "DICTIONARY_SUGGESTION"
        return dict(intent="SUGGESTION", topic=topic, sentiment="NEUTRAL", ambiguous=False, confidence="medium", reason="proposal")
    return dict(intent="SPAM", topic="OFF_TOPIC_PERSONAL", sentiment="NEUTRAL", ambiguous=len(c.split()) < 3, confidence="low", reason="no dictionary purpose")


class MockGeminiClient:
    """Local, deterministic, offline. Counts calls; never touches the network. `responder` may return a dict (serialised to JSON), a raw string
    (to inject malformed output) or raise ClientError (to inject API failures)."""

    def __init__(self, responder: Callable[[str], object] = rule_based_responder, *, latency_seconds: float = 0.0):
        self.responder = responder
        self.calls = 0
        self.prompts: list[str] = []
        self.latency_seconds = latency_seconds

    def classify(self, prompt: str, schema: dict, settings: GenerationSettings) -> ClientResponse:
        self.calls += 1
        self.prompts.append(prompt)
        out = self.responder(extract_comment(prompt))
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return ClientResponse(text, Usage(estimate_tokens(prompt), estimate_tokens(text), 0, estimated=True))


class ScriptedMockClient:
    """Plays back a fixed list of outcomes (dict | str | ClientError) in order -- for retry and failure-handling tests."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls = 0

    def classify(self, prompt: str, schema: dict, settings: GenerationSettings) -> ClientResponse:
        self.calls += 1
        item = self.outcomes.pop(0) if self.outcomes else self.outcomes_default()
        if isinstance(item, ClientError):
            raise item
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return ClientResponse(text, Usage(estimate_tokens(prompt), estimate_tokens(text), 0, estimated=True))

    @staticmethod
    def outcomes_default():
        return rule_based_responder("")


# ------------------------------------------------------------------------------------------------------------------------ live (never run in this phase)
class LiveGeminiClient:
    """Interactions API client (structured output via response_format; seed / thinking_level / max_output_tokens in generation_config).

    UNVERIFIED: written from the official documentation, never executed. Construction requires a LiveApproval; the API key is read from the
    environment variable GEMINI_API_KEY only at call time and is never stored or logged."""

    def __init__(self, approval: LiveApproval | None, *, api_key: str | None = None, timeout_seconds: float = 60.0):
        if approval is None:
            raise RealDataNotApproved("LiveGeminiClient requires a written LiveApproval")
        approval.validate()
        self.approval = approval
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = None

    def classify(self, prompt: str, schema: dict, settings: GenerationSettings) -> ClientResponse:  # pragma: no cover - not executed in this phase
        import os

        from google import genai  # lazy: never imported by the offline phase

        if self._client is None:
            key = self._api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ClientError("missing_api_key")
            self._client = genai.Client(api_key=key)
        try:
            interaction = self._client.interactions.create(
                model=settings.model_id,
                input=prompt,
                response_format={"type": "text", "mime_type": "application/json", "schema": schema or RESPONSE_SCHEMA},
                generation_config=settings.to_generation_config(),
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001 - classified into retryable/non-retryable by name only; no message is kept (may contain text)
            name = type(exc).__name__
            retryable = any(k in name for k in ("Timeout", "Unavailable", "ResourceExhausted", "TooManyRequests", "ServerError", "Connection"))
            raise ClientError(f"api_error:{name}", retryable=retryable) from None
        u = getattr(interaction, "usage", None)
        usage = Usage(
            int(getattr(u, "total_input_tokens", 0) or 0),
            int(getattr(u, "total_output_tokens", 0) or 0),
            int(getattr(u, "total_thought_tokens", 0) or 0),
        )
        return ClientResponse(interaction.output_text, usage)
