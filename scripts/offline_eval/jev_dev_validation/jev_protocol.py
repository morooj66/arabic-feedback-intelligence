"""Jev request construction, strict response validation, and cost estimation.

One request per comment carrying the three frozen-taxonomy questions. The
payload holds ONLY the sanitized comment text: no comment id, gold label,
reviewer data, or provenance. No network access in this module.
"""

from __future__ import annotations

import json
import math
from typing import Any

from jev_common import BLENDED_USD_PER_TOKEN, COST_SAFETY_FACTOR, PRIOR_RUN
from config import settings
from rcs.classification.jev_criteria import QUESTIONS


class JevProtocolError(RuntimeError):
    pass


def build_request(sanitized_text: str, version: str = "v1") -> dict[str, Any]:
    """version 'v1' = the production criteria used for the completed evaluation; 'v2' = the prepared candidate
    (jev_criteria_candidate_v2, not yet evaluated)."""
    if version == "v2":
        from jev_criteria_candidate_v2 import CANDIDATE_QUESTIONS as questions
    elif version == "v1":
        questions = QUESTIONS
    else:
        raise ValueError(f"unknown criteria version {version!r}")
    return {
        "model": settings.JEV_TOPIC_MODEL,
        "state": {"visitor_comment_text": sanitized_text},
        "questions": {
            name: {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
            for name, (instructions, criteria, _labels) in questions.items()
        },
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def validate_response(raw: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """-> {"intent": (label, confidence), "topic": ..., "sentiment": ...}; raises on ANY deviation."""
    if raw.get("provider") != settings.JEV_TOPIC_EXPECTED_PROVIDER:
        raise JevProtocolError("provider validation failed")
    if raw.get("model") != settings.JEV_TOPIC_EXPECTED_SNAPSHOT:
        raise JevProtocolError("model snapshot validation failed")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(QUESTIONS):
        raise JevProtocolError("response must contain exactly the intent, topic and sentiment answers")
    out: dict[str, tuple[str, float]] = {}
    for name, (_i, _c, labels) in QUESTIONS.items():
        ans = answers[name]
        if not isinstance(ans, dict) or ans.get("type") != "choice":
            raise JevProtocolError(f"{name}: malformed answer")
        label = ans.get("choice")
        if label not in labels:
            raise JevProtocolError(f"{name}: label outside the frozen taxonomy")
        probs = ans.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(labels):
            raise JevProtocolError(f"{name}: probabilities do not match the frozen taxonomy")
        vals = [_number(probs[k]) for k in labels]
        if any(v is None or not 0 <= v <= 1 for v in vals) or abs(sum(vals) - 1.0) > 0.02:
            raise JevProtocolError(f"{name}: invalid probabilities")
        conf = _number(ans.get("confidence"))
        if conf is None or not 0 <= conf <= 1:
            raise JevProtocolError(f"{name}: invalid confidence")
        out[name] = (label, conf)
    return out


def estimate_cost(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Token estimate from payload size (Arabic JSON ~2.2 chars/token, conservative) priced at the
    blended $/token measured on the 2026-10-13 run, with a 2x safety factor for the upper bound."""
    n = len(payloads)
    chars = sum(len(json.dumps(p, ensure_ascii=False)) for p in payloads)
    est_in = int(chars / 2.2)
    est_out = n * round(PRIOR_RUN["output_tokens"] / PRIOR_RUN["requests"])
    point = (est_in + est_out) * BLENDED_USD_PER_TOKEN
    return {
        "requests": n,
        "payload_chars": chars,
        "estimated_input_tokens": est_in,
        "estimated_output_tokens": est_out,
        "estimated_usd": round(point, 6),
        "upper_bound_usd": round(point * COST_SAFETY_FACTOR, 6),
        "basis": "blended $/token from the 2026-10-13 exploratory run; upper bound = 2x",
    }
