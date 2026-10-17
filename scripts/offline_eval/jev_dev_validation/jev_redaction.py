"""Strict, fail-closed redaction for Jev evaluation inputs.

Layers the project's production redactor (rcs.classification.external_redaction:
emails, URLs, handles, long digit runs, ID phrases, addresses, signatures,
cue-introduced names) with extra detectors for identifying details it does not
cover: titled names, nasab/kunya names, and name-like Latin tokens. Anything
that still looks identifying afterwards QUARANTINES the row (it is never
sent) instead of being sent on a best-effort basis. Pure functions, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rcs.classification.external_redaction import ExternalRedactionError, redact_external_text

_AR = r"[؀-ۿ]"
_TITLED_NAME = re.compile(
    rf"(?P<title>(?:الأستاذ|الاستاذ|الدكتور|الشيخ|السيد|المهندس)(?:ة)?|الأخ|الأخت|د\.|أ\.|م\.)"
    rf"\s*(?P<name>{_AR}+(?:\s+{_AR}+){{0,2}})"
)
_NASAB_NAME = re.compile(rf"(?P<cue>أبو|أبي|أبا|ابن|بن|بنت)\s+(?P<name>{_AR}{{2,}})")
_LATIN_NAME = re.compile(r"(?<![A-Za-z])[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2}(?![A-Za-z])")
_CONTACT_CUE_WITH_DIGITS = re.compile(r"(?i)(واتس(?:اب)?|whats\s*app|تواصل\s*معي|رقمي|جوالي|هاتفي)[^\n]{0,30}\d")
_RESIDUAL_AT = re.compile(r"@")
# Added after the 2026-10-15 canary showed bare Arabic names passing the cue-based rules (see README of this folder):
_ID_ADJACENT_LETTERS = re.compile(r"\[ID\]\s*[\u0600-\u06FF]")  # a name pasted right after a number/ID
_GENEALOGY_CHAIN = re.compile(r"(?:بنت|بن|ابن)\s+\[PERSON\]")      # given names that precede/accompany a nasab chain remain
_FIRST_PERSON_NAME_LEFTOVER = re.compile(r"(?:أنا|انا|اسمي|إسمي)\s+\[PERSON\]\s+[\u0600-\u06FF]")  # surname left after the redacted token

PLACEHOLDERS = ("[EMAIL]", "[URL]", "[HANDLE]", "[ID]", "[PERSON]", "[ADDRESS]")


class Quarantined(Exception):
    """The row cannot be shown to be safe; it must not be sent."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Redaction:
    text: str
    replacements: dict[str, int] = field(default_factory=dict)


def redact_strict(text: str, *, release_latin_names: bool = False) -> Redaction:
    """Return sanitized text or raise Quarantined(reason_code).

    `release_latin_names` is for rows a human has reviewed locally and cleared: name-like Latin
    tokens are then replaced with [PERSON] instead of quarantining the row. Every other
    quarantine reason still applies."""
    try:
        base = redact_external_text(text)
    except ExternalRedactionError:
        raise Quarantined("production_redactor_failed_closed") from None
    value = base.text
    counts: dict[str, int] = {"production_redactor": base.replacements}

    if _CONTACT_CUE_WITH_DIGITS.search(value):
        raise Quarantined("contact_solicitation_with_digits")

    value, n = _TITLED_NAME.subn(lambda m: f"{m.group('title')} [PERSON]", value)
    counts["titled_name"] = n
    value, n = _NASAB_NAME.subn(lambda m: f"{m.group('cue')} [PERSON]", value)
    counts["nasab_name"] = n

    if _ID_ADJACENT_LETTERS.search(value):
        raise Quarantined("id_adjacent_name_like_token")
    if _GENEALOGY_CHAIN.search(value):
        raise Quarantined("genealogy_chain_may_leave_given_names")
    if _FIRST_PERSON_NAME_LEFTOVER.search(value):
        raise Quarantined("first_person_name_leftover")
    if _LATIN_NAME.search(value):
        if not release_latin_names:
            raise Quarantined("latin_name_like_token")
        value, n = _LATIN_NAME.subn("[PERSON]", value)
        counts["latin_name_released_after_human_review"] = n
    if _RESIDUAL_AT.search(value):
        raise Quarantined("residual_at_sign")

    # Final assertion: re-run the production detectors on the final text.
    try:
        again = redact_external_text(value)
    except ExternalRedactionError:
        raise Quarantined("residual_identifier_after_redaction") from None
    if again.text != value and again.replacements:
        raise Quarantined("redaction_not_idempotent")

    value = re.sub(r"[ \t]{2,}", " ", value).strip()
    if not value:
        raise Quarantined("empty_after_redaction")
    return Redaction(text=value, replacements={k: v for k, v in counts.items() if v})
