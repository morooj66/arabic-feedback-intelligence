"""
Phase M6.2 — Identifying Context Policy: the single, canonical decision
of whether identifying metadata (dictionary-entry title/entry_id) may be
supplied to Gemini at all, for THIS comment.

--- Why this module exists --------------------------------------------

The Identifying Context Safety Check and Mechanism experiments
(scripts/offline_eval/identifying_context_safety/,
scripts/offline_eval/identifying_context_mechanism/) established that
identifying metadata, when present, was previously trusted
unconditionally — reaching the prompt regardless of whether the
visitor's own comment gave any reason to believe it was relevant.
Confirmed hijacking cases (all real, human-reviewed): "الله اعلم",
"لا اعلم", and "تتت", each paired with a real, specific dictionary entry
("قناطر") the comment never referenced in any way. Confirmed safe cases:
rows discussing "اعتبار", "اشتد", and "مكبس" — in every one, the
comment's own text corroborates the entry.

This module is the one place that decision is made. No other file may
re-implement it — `rcs.response_intelligence.orchestrator` calls it
exactly once per execution, and every caller of the Response Agent
converges there regardless of how it originally learned the entry's
title/entry_id.

--- Data-contract audit this module is built on (evidence, not
    assumption) ------------------------------------------------------

`original_title` is used for lexical corroboration because, per direct
inspection of the real production data contract:
  - `ReviewerItemDetail.original_title` is already the field
    `rcs.review.excel_review_flow.apply_live_generation` has access to
    today, with 100% fill rate in the audited source workbook.
  - `entry_id` is confirmed schema-overloaded in real data — sometimes
    an internal UUID, sometimes a bare Arabic word purely by
    inconsistent data entry — and is therefore NEVER treated as
    semantic evidence here, regardless of its apparent content.
  - `lemma` (`ReviewItem.lemma`) was found, on audit, to have only ~5%
    fill rate in the real source workbook, to be stored as a
    JSON-array-shaped string (`'["اِعْتِبَار"]'`, not plain text), to
    occasionally disagree with `original_title` for the same row, and
    to not even be exposed on `ReviewerItemDetail` today. Adopting it
    now would be a new product assumption requiring new DTO/repository
    work, not a behavior-preserving correction — explicitly deferred,
    not used here.

--- What this module is not --------------------------------------------

Not a classifier: every input is already-computed, deterministic data
(the visitor's own comment text, and structured title/entry_id already
attached to the row). No Gemini call, no embeddings, no fuzzy/semantic
matching, no machine learning. No dependency on primary_intent,
topic_subtype, sentiment, confidence, comment length, or any frozen
classifier output — `evaluate_identifying_context`'s signature makes
this structurally impossible, not just a convention to remember.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

REASON_NO_IDENTIFYING_METADATA = "NO_IDENTIFYING_METADATA"
REASON_LEXICAL_CORROBORATION = "LEXICAL_CORROBORATION"
REASON_EXPLICIT_ENTRY_REFERENCE = "EXPLICIT_ENTRY_REFERENCE"
REASON_NO_CORROBORATING_REFERENCE = "NO_CORROBORATING_REFERENCE"

# --- Explicit-entry-reference cues: three small, closed word sets,
# combined only as FIXED ADJACENT-TOKEN BIGRAMS (never a single bare
# noun). This is deliberately narrower than "the comment mentions a
# domain noun" -- see module docstring and the negative examples in the
# test suite ("المعنى مهم", "الكلمة جميلة", "أحب هذه المصطلحات" must NOT
# qualify; "هذا المدخل", "هذه الكلمة", "المثال هنا", "التعريف المذكور",
# "الشاهد المذكور", "معنى الكلمة", "تعريف الكلمة" must).
_DEMONSTRATIVES = frozenset({"هذا", "هذه"})
_DEICTIC_MODIFIERS = frozenset({"هنا", "المذكور", "المذكورة"})
_CONSTRUCT_REFERENCE_NOUNS = frozenset({"معنى", "تعريف"})
_SINGULAR_REFERENT_NOUNS = frozenset(
    {"المدخل", "الكلمة", "المثال", "التعريف", "الشاهد", "المصطلح", "اللفظ", "المفردة", "المعنى"}
)

_TOKEN_SPLIT_RE = re.compile(r"[،,.!؟?:;\"'()\[\]{}«»\-]")


@dataclass(frozen=True)
class IdentifyingMetadata:
    """Structured, unrendered identifying metadata for one comment.

    `original_title`: the possible human-readable identifying reference
    -- the only field this policy ever corroborates against, and the
    only field the canonical Response Agent path
    (`rcs.response_intelligence.orchestrator`) ever renders into a
    prompt.

    `entry_id`: internal metadata ONLY. Retained here for trace/lookup/
    future-retrieval use -- never treated as semantic/lexical evidence
    by this policy (see `evaluate_identifying_context`), and never
    rendered into any prompt by the orchestrator regardless of the
    policy's decision or of what `entry_id` actually contains (a UUID,
    an Arabic word, or anything else)."""

    original_title: str | None
    entry_id: str | None


@dataclass(frozen=True)
class IdentifyingContextDecision:
    """The policy's sole output. `matched_reference` is always either
    the matched `original_title` value (public dictionary vocabulary,
    not visitor PII) or one of the small closed reference-cue phrases
    above -- safe to log/trace, never visitor comment text."""

    allowed: bool
    reason: str
    matched_reference: str | None = None


def _strip_arabic_diacritics(text: str) -> str:
    """Unicode category Mn (combining marks -- Arabic tashkeel). The
    same normalization concept `eligibility._lacks_minimum_meaningful_
    content`'s docstring already documents; not a new normalization
    rule invented for this module."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _tokenize(normalized_text: str) -> list[str]:
    """Whitespace/punctuation-separated tokens of an already
    diacritic-stripped string. No stemming, no alef/ta-marbuta folding."""
    cleaned = _TOKEN_SPLIT_RE.sub(" ", normalized_text)
    return cleaned.split()


def _find_explicit_entry_reference(normalized_comment: str) -> str | None:
    """Returns the matched two-word cue phrase, or None. Checks every
    adjacent token pair against the three fixed bigram patterns -- see
    module-level word sets above."""
    tokens = _tokenize(normalized_comment)
    for first, second in zip(tokens, tokens[1:]):
        if first in _DEMONSTRATIVES and second in _SINGULAR_REFERENT_NOUNS:
            return f"{first} {second}"
        if first in _SINGULAR_REFERENT_NOUNS and second in _DEICTIC_MODIFIERS:
            return f"{first} {second}"
        if first in _CONSTRUCT_REFERENCE_NOUNS and second in _SINGULAR_REFERENT_NOUNS:
            return f"{first} {second}"
    return None


def evaluate_identifying_context(
    *,
    comment_text: str,
    identifying_metadata: IdentifyingMetadata | None,
) -> IdentifyingContextDecision:
    """Pure, deterministic, no I/O, no Gemini. Fixed evaluation order,
    first match wins:

      1. NO_IDENTIFYING_METADATA -- metadata absent, or original_title
         blank -- nothing to evaluate.
      2. LEXICAL_CORROBORATION -- the diacritic-stripped original_title
         occurs as a substring of the diacritic-stripped comment.
      3. EXPLICIT_ENTRY_REFERENCE -- the comment contains one of the
         closed adjacent-token reference-cue bigrams.
      4. NO_CORROBORATING_REFERENCE -- metadata exists, neither matched.

    Only `comment_text` and `identifying_metadata` participate --
    intentionally the only two parameters this function accepts."""
    if identifying_metadata is None or not (identifying_metadata.original_title or "").strip():
        return IdentifyingContextDecision(allowed=False, reason=REASON_NO_IDENTIFYING_METADATA)

    normalized_comment = _strip_arabic_diacritics(comment_text or "")
    normalized_title = _strip_arabic_diacritics(identifying_metadata.original_title).strip()

    if normalized_title and normalized_title in normalized_comment:
        return IdentifyingContextDecision(
            allowed=True,
            reason=REASON_LEXICAL_CORROBORATION,
            matched_reference=identifying_metadata.original_title,
        )

    matched_cue = _find_explicit_entry_reference(normalized_comment)
    if matched_cue is not None:
        return IdentifyingContextDecision(
            allowed=True,
            reason=REASON_EXPLICIT_ENTRY_REFERENCE,
            matched_reference=matched_cue,
        )

    return IdentifyingContextDecision(allowed=False, reason=REASON_NO_CORROBORATING_REFERENCE)
