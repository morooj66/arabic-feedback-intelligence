"""
Phone-number redaction — applied to comment text before it is sent to
Gemini, never applied to what is stored locally.

The stored `original_comment` in review_items is always the verbatim
source text; redaction happens only at prompt-build time, in memory, for
the outbound model call.
"""

from __future__ import annotations

import re

from config import settings

# Matches runs of digits interleaved with common separators (space, dash,
# parentheses, plus sign). A run only gets redacted if it contains enough
# actual digits (settings.PHONE_REDACTION_MIN_DIGITS) — short digit runs
# like a 4-digit year or a single placeholder digit ("S2") are left alone.
_CANDIDATE_RUN_RE = re.compile(r"[\d\-\s()+]{4,}")


def redact_phone_numbers(
    text: str | None,
    min_digits: int = settings.PHONE_REDACTION_MIN_DIGITS,
    placeholder: str = settings.PHONE_REDACTION_PLACEHOLDER,
) -> str | None:
    if text is None:
        return None

    def _replace(match: re.Match) -> str:
        span = match.group(0)
        digit_count = sum(ch.isdigit() for ch in span)
        return placeholder if digit_count >= min_digits else span

    redacted = _CANDIDATE_RUN_RE.sub(_replace, text)
    # Cosmetic only: the candidate run can swallow a word-boundary space on
    # either side of the placeholder (e.g. "call me PLACEHOLDERplease").
    # Re-space around the placeholder so the redacted prompt still reads as
    # separate words; does not change whether/what was redacted.
    redacted = re.sub(rf"\s*{re.escape(placeholder)}\s*", f" {placeholder} ", redacted).strip()
    return redacted
