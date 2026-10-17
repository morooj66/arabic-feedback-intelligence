"""Strict Vendor Jev Topic adapter for the experimental hybrid mode.

There are no retries and no fallback classifier. Every outbound request holds
only the redacted visitor text and the frozen seven-label Topic question.
"""

from __future__ import annotations

import http.client
import json
import math
import ssl
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

from config import settings
from rcs.classification.external_redaction import Redactor, redact_external_text
from rcs.classification.jev_criteria import TOPIC_CRITERIA, TOPIC_INSTRUCTIONS


class JevTopicError(RuntimeError):
    """Safe, non-PII error raised when hybrid Topic classification fails."""


# Wording lives in jev_criteria (aligned with LABELING_GUIDE.md); the label set is the frozen taxonomy.
if set(TOPIC_CRITERIA) != set(settings.TOPIC_SUBTYPES):
    raise RuntimeError("Jev Topic criteria drifted from the frozen taxonomy")


Transport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _https_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict[str, Any]:
    """Exactly one HTTPS POST. It never redirects or retries."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise JevTopicError("Jev Topic endpoint must be HTTPS")
    connection = http.client.HTTPSConnection(
        parts.hostname,
        port=parts.port,
        timeout=timeout,
        context=ssl.create_default_context(),
    )
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    try:
        connection.request("POST", target, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read(2_000_001)
        status = response.status
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise JevTopicError(f"Jev Topic request failed without retry: {type(exc).__name__}") from None
    finally:
        connection.close()
    if len(raw) > 2_000_000:
        raise JevTopicError("Jev Topic response exceeded the size limit")
    if status < 200 or status >= 300:
        raise JevTopicError(f"Jev Topic provider returned HTTP {status}")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JevTopicError("Jev Topic provider returned invalid JSON") from None
    if not isinstance(parsed, dict):
        raise JevTopicError("Jev Topic provider returned a non-object response")
    return parsed


@dataclass(frozen=True)
class JevTopicPrediction:
    label: str
    confidence: float


class JevTopicAdapter:
    def __init__(
        self,
        *,
        api_key: str | None,
        endpoint: str = settings.JEV_TOPIC_ENDPOINT,
        model: str = settings.JEV_TOPIC_MODEL,
        expected_snapshot: str = settings.JEV_TOPIC_EXPECTED_SNAPSHOT,
        expected_provider: str = settings.JEV_TOPIC_EXPECTED_PROVIDER,
        timeout_seconds: float = settings.JEV_TOPIC_REQUEST_TIMEOUT_SECONDS,
        transport: Transport = _https_post,
        redactor: Redactor = redact_external_text,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._model = model
        self._expected_snapshot = expected_snapshot
        self._expected_provider = expected_provider
        self._timeout = timeout_seconds
        self._transport = transport
        self._redactor = redactor

    def _request(self, text: str) -> dict[str, Any]:
        try:
            sanitized = self._redactor(text)
        except Exception as exc:
            if type(exc).__name__ == "ExternalRedactionError":
                raise JevTopicError("Jev Topic redaction failed closed") from None
            raise JevTopicError("Jev Topic redaction failed closed") from None
        return {
            "model": self._model,
            "state": {"visitor_comment_text": sanitized.text},
            "questions": {
                "topic": {
                    "type": "choice",
                    "instructions": TOPIC_INSTRUCTIONS,
                    "criteria": TOPIC_CRITERIA,
                }
            },
        }

    def predict_one(self, text: str) -> JevTopicPrediction:
        if not self._api_key:
            raise JevTopicError("OPENROUTER_API_KEY is required for hybrid_jev_topic")
        if not 5 <= self._timeout <= 120:
            raise JevTopicError("Jev Topic timeout must be between 5 and 120 seconds")
        request = self._request(text)
        body = json.dumps(request, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "User-Agent": "rcs-hybrid-jev-topic/1.0.0",
        }
        raw = self._transport(self._endpoint, headers, body, self._timeout)
        return self._validate(raw)

    def _validate(self, raw: dict[str, Any]) -> JevTopicPrediction:
        if raw.get("provider") != self._expected_provider:
            raise JevTopicError("Jev Topic provider validation failed")
        if raw.get("model") != self._expected_snapshot:
            raise JevTopicError("Jev Topic model snapshot validation failed")
        answers = raw.get("answers")
        if not isinstance(answers, dict) or set(answers) != {"topic"}:
            raise JevTopicError("Jev Topic response must contain exactly one Topic answer")
        answer = answers.get("topic")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise JevTopicError("Jev Topic answer is malformed")
        label = answer.get("choice")
        if label not in settings.TOPIC_SUBTYPES:
            raise JevTopicError("Jev Topic label is outside the frozen taxonomy")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(settings.TOPIC_SUBTYPES):
            raise JevTopicError("Jev Topic probabilities do not match the frozen taxonomy")
        values = [_number(probabilities[name]) for name in settings.TOPIC_SUBTYPES]
        if any(value is None or not 0 <= value <= 1 for value in values):
            raise JevTopicError("Jev Topic probabilities are invalid")
        if abs(sum(values) - 1.0) > 0.02:
            raise JevTopicError("Jev Topic probabilities do not sum to one")
        confidence = _number(answer.get("confidence"))
        if confidence is None or not 0 <= confidence <= 1:
            raise JevTopicError("Jev Topic confidence is invalid")
        return JevTopicPrediction(label=label, confidence=confidence)

    def predict(self, comment_ids: list[str], texts: list[str]) -> dict[str, tuple[str, float]]:
        if len(comment_ids) != len(texts) or len(set(comment_ids)) != len(comment_ids):
            raise JevTopicError("Jev Topic batch IDs and texts are inconsistent")
        results: dict[str, tuple[str, float]] = {}
        for comment_id, text in zip(comment_ids, texts):
            prediction = self.predict_one(text)
            results[comment_id] = (prediction.label, prediction.confidence)
        return results
