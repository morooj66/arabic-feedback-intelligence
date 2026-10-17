"""Fail-closed PII redaction for external classification requests.

Only the returned text may cross the external-provider boundary. The caller
never supplies visitor metadata, Gold labels, comment IDs, or provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


class ExternalRedactionError(RuntimeError):
    """The text could not be shown to be safe for external processing."""


@dataclass(frozen=True)
class RedactionResult:
    text: str
    replacements: int


_EMAIL = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.-]{1,30}\b")
_LONG_DIGITS = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]*){7,}(?!\d)")
_ID_WITH_CUE = re.compile(
    r"(?i)(?:رقم\s*(?:الهوية|الإقامة|الطلب|الحساب)|هوية|إقامة|national\s*id|account\s*id)"
    r"\s*[:#-]?\s*[A-Za-z0-9-]{4,}"
)
_PERSON_WITH_CUE = re.compile(
    r"(?i)(?P<cue>\b(?:اسمي|أنا|معك|الاسم|أخوكم|أختكم|المرسل|name\s+is|i\s+am)\b)"
    r"\s*[:،,-]?\s*(?P<name>[A-Za-z\u0600-\u06FF]+(?:\s+[A-Za-z\u0600-\u06FF]+){0,3})"
)
_SIGNATURE = re.compile(
    r"(?im)(?:^|\n)\s*(?:مع\s+التحية|تحياتي|المخلص|الموقّع|الموقع)\s*[:،,-]?\s*"
    r"[A-Za-z\u0600-\u06FF]+(?:\s+[A-Za-z\u0600-\u06FF]+){0,3}\s*$"
)
_ADDRESS = re.compile(
    r"(?i)(?:عنواني|عنوان\s+المنزل|أسكن\s+في|my\s+address|i\s+live\s+at)"
    r"\s*[:،,-]?\s*[^\n.!؟]{2,120}"
)


def _sub_count(pattern: re.Pattern[str], replacement: str, text: str) -> tuple[str, int]:
    return pattern.subn(replacement, text)


def _redact_person(match: re.Match[str]) -> str:
    return f"{match.group('cue')} [PERSON]"


def redact_external_text(text: str | None) -> RedactionResult:
    """Return externally safe text or raise instead of sending uncertain PII."""
    if not isinstance(text, str):
        raise ExternalRedactionError("external text must be a string")
    value = text.strip()
    if not value:
        raise ExternalRedactionError("external text is empty")

    replacements = 0
    for pattern, replacement in (
        (_EMAIL, "[EMAIL]"),
        (_URL, "[URL]"),
        (_HANDLE, "[HANDLE]"),
        (_ID_WITH_CUE, "[ID]"),
        (_LONG_DIGITS, "[ID]"),
        (_SIGNATURE, "\n[PERSON]"),
        (_ADDRESS, "[ADDRESS]"),
    ):
        value, count = _sub_count(pattern, replacement, value)
        replacements += count
    value, count = _PERSON_WITH_CUE.subn(_redact_person, value)
    replacements += count
    value = re.sub(r"[ \t]{2,}", " ", value).strip()

    if not value:
        raise ExternalRedactionError("redaction removed all meaningful text")
    # A second pass is a fail-closed assertion, not another best-effort mask.
    residual_detectors = (_EMAIL, _URL, _HANDLE, _LONG_DIGITS, _ID_WITH_CUE)
    if any(pattern.search(value) for pattern in residual_detectors):
        raise ExternalRedactionError("possible identifying information remains after redaction")
    return RedactionResult(text=value, replacements=replacements)


Redactor = Callable[[str | None], RedactionResult]
