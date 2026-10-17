"""
Phase 3 — deterministic spam/junk prefilter.

Runs entirely in Python, on fields already normalized by Phase 2. No
network/API calls. Never deletes a row: every row that matches a rule is
flagged (`is_prefiltered_junk=True`, `prefilter_reason=<code>`) and routed
to EXCLUDE; everything else passes through untouched, unflagged, ready for
Phase 4 classification.

Seven rules, checked in this fixed priority order (first match wins, so
every flagged row has exactly one reason code):

  1. exact_match_known_bot_string  — content == a known junk string verbatim
  2. null_content / near_empty_content — nothing meaningful was submitted
  3. phone_number_only_stub        — only a pasted phone number, no real text
  4. unfilled_form_template        — the word-entry form's own placeholders
                                      (S1/E1/...) were submitted unedited
  5. safe_test_marker              — content IS (not merely contains) a
                                      test/placeholder token
  6. keyboard_mash_sequence        — three or more whole ASCII tokens that
                                      are only contiguous QWERTY-row runs
  7. burst_repetition              — the exact same content, from the same
                                      submitter, repeated >= BURST_MIN_COUNT
                                      times inside a BURST_WINDOW_HOURS window

Deliberately conservative, per explicit product instruction:
  - Rule 5 only matches content that reduces, after normalization, to
    nothing but test tokens — never a substring match. A real comment that
    happens to mention "تجربة" or "test" as a word inside real prose is
    left untouched for Gemini to judge in Phase 4.
  - Rule 7 requires identical content from the same identified submitter
    (email, or name if email is missing) — it never fires on volume alone,
    and never fires with no identity signal at all (two different
    anonymous visitors who both wrote a short common word are not swept
    together).
  - Exact-duplicate status (`is_exact_duplicate_of`, set in Phase 2) is
    never used as a spam signal here — a duplicate is not automatically
    junk, and junk is not automatically a duplicate. The two are computed
    independently and both are preserved.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Rule 1 — exact known junk strings (config-seeded, extendable over time)
# ---------------------------------------------------------------------------

def detect_known_junk_string(content: str | None, known_strings: tuple[str, ...]) -> str | None:
    if content is None:
        return None
    return "exact_match_known_bot_string" if content in known_strings else None


# ---------------------------------------------------------------------------
# Rule 2 — null / blank / near-empty
# ---------------------------------------------------------------------------

NEAR_EMPTY_MAX_LEN = 2  # characters, after Phase 2 normalization


def detect_null_or_near_empty(content: str | None) -> str | None:
    if content is None:
        return "null_content"
    if len(content) <= NEAR_EMPTY_MAX_LEN:
        return "near_empty_content"
    return None


# ---------------------------------------------------------------------------
# Rule 3 — phone-number-only stub
# ---------------------------------------------------------------------------

_PHONE_LABEL_RE = re.compile(
    r"(Phone\s*Number\s*:\s*[\d\-\s]*|رقم\s*الجوال\s*:\s*[\d\-\s]*|الجوال\s*:\s*[\d\-\s]*)",
    re.IGNORECASE,
)


def detect_phone_number_stub(content: str | None) -> str | None:
    if content is None:
        return None
    if not _PHONE_LABEL_RE.search(content):
        return None  # no phone-field label present at all -> not this rule
    remainder = _PHONE_LABEL_RE.sub("", content).strip()
    if len(remainder) <= 2:
        return "phone_number_only_stub"
    return None


# ---------------------------------------------------------------------------
# Rule 4 — unfilled word-entry form template
# ---------------------------------------------------------------------------
# Confirmed in data discovery: the "add a word" form pre-fills its
# meaning/example fields with literal placeholders (S1/E1/S2/E2/S3/E3, or
# just "S"), and a fraction of visitors submit the form without replacing
# them. Matches only when the REQUIRED "first meaning" field itself is
# empty or one of the literal defaults — a real submission would have
# replaced it with actual text.
#
# Deliberately checks only the meaning field, not the example field: the
# example (المثال الأول) is reasonably left blank by genuine contributors
# submitting a short/simple word, so treating a blank example alone as
# junk would falsely exclude real submissions. A blank/placeholder
# *meaning* field is a much stronger signal, since that's the one piece
# of substantive content the form actually asks for.

_TEMPLATE_STRUCTURE_RE = re.compile(r"الكلمة\s*:.*اللهجة\s*:", re.S)
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"المعنى\s*الأول\s*:\s*(S\d?)?\s*(\n|$)")


def detect_unfilled_template(content: str | None) -> str | None:
    if content is None:
        return None
    if _TEMPLATE_STRUCTURE_RE.search(content) and _TEMPLATE_PLACEHOLDER_RE.search(content):
        return "unfilled_form_template"
    return None


# ---------------------------------------------------------------------------
# Rule 5 — safe exact/near-exact test markers only
# ---------------------------------------------------------------------------
# Whole-content match only, never substring. A comment that merely contains
# "تجربة" or "test" as a word within real prose must NOT match this rule.

_TEST_TOKENS = {"test", "تست", "تجربة", "تجربه", "تيست", "تجريب"}
_TRAILING_NOISE_RE = re.compile(r"[\s\.\!\?\d]+$")


def detect_test_marker(content: str | None) -> str | None:
    if content is None:
        return None
    if len(content) > 20:
        return None  # long content is never "just a test marker"

    # Whole-string case: "test", "Test.", "تجربة!!", "test123"
    stripped = _TRAILING_NOISE_RE.sub("", content).strip().lower()
    if stripped in _TEST_TOKENS:
        return "safe_test_marker"

    # Repeated-line case: "Test\nTest\nTest"
    lines = [ln.strip().lower() for ln in content.split("\n") if ln.strip()]
    if lines and all(_TRAILING_NOISE_RE.sub("", ln).strip() in _TEST_TOKENS for ln in lines):
        return "safe_test_marker"

    return None


# ---------------------------------------------------------------------------
# Rule 6 — narrow keyboard-mash sequence
# ---------------------------------------------------------------------------
# This is intentionally NOT a generic Latin-text or gibberish detector. Such
# a rule would be unsafe for technical terms, proper nouns, English comments,
# and dictionary submissions. It only catches the observed placeholder-like
# pattern where the *entire* comment is three or more whitespace-separated
# runs copied from contiguous keys in one QWERTY row (for example
# ``asdf qwer zxcv``). A single token remains eligible as a possible word
# submission, and any Arabic, punctuation, digits, or ordinary Latin word
# prevents this rule from firing.

_ASCII_WORDS_ONLY_RE = re.compile(r"[A-Za-z]+(?:\s+[A-Za-z]+){2,}\Z")
_QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def detect_keyboard_mash_sequence(content: str | None) -> str | None:
    if content is None or not _ASCII_WORDS_ONLY_RE.fullmatch(content):
        return None
    tokens = content.lower().split()
    if len(tokens) < 3 or any(len(token) < 4 for token in tokens):
        return None
    if all(any(token in row or token in row[::-1] for row in _QWERTY_ROWS) for token in tokens):
        return "keyboard_mash_sequence"
    return None


# ---------------------------------------------------------------------------
# Rule 7 — burst / repetition detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BurstConfig:
    min_count: int
    window_hours: float


def detect_burst_repetition(
    df: pd.DataFrame,
    config: BurstConfig,
    already_flagged: pd.Series,
) -> pd.Series:
    """
    Returns a boolean Series (index-aligned with df) marking rows that are
    part of a qualifying burst: identical normalized content, from the
    same identified submitter (email, falling back to name), with at
    least `config.min_count` occurrences inside some `config.window_hours`
    window.

    Rows already flagged by an earlier rule are excluded from consideration
    (both as candidates and as group members) so this rule only adds NEW
    findings and reason codes stay singular per row.

    Rows with no content, or no submitter identity at all (both email and
    name missing), are never eligible — burst detection requires both a
    real content signal and an identity signal, never volume alone.

    Rows with a missing/unparseable timestamp (NaT) are also never
    eligible: the sliding-window comparison below relies on real datetime
    arithmetic, and a NaT would either sort unpredictably or make the
    `ts[right] - ts[left] > window` comparison silently evaluate to False,
    which could let a timestamp-less row sit inside a window it doesn't
    actually belong to. Excluding NaT rows up front removes that
    possibility entirely; it does not change how any row with a valid
    timestamp is treated.
    """
    result = pd.Series(False, index=df.index)

    candidates = df[
        (~already_flagged)
        & df["original_comment"].notna()
        & (df["original_comment"].str.len() > NEAR_EMPTY_MAX_LEN)
        & (df["visitor_email"].notna() | df["visitor_name"].notna())
        & df["comment_created_at"].notna()
    ]
    if candidates.empty:
        return result

    identity = candidates["visitor_email"].fillna(candidates["visitor_name"])
    group_key = identity.astype(str) + "\x1f" + candidates["original_comment"].astype(str)

    window = dt.timedelta(hours=config.window_hours)

    for _, group_index in candidates.groupby(group_key).groups.items():
        if len(group_index) < config.min_count:
            continue
        times = candidates.loc[group_index, "comment_created_at"].sort_values()
        ts = times.to_list()
        idx_sorted = times.index.to_list()

        # Sliding window over sorted timestamps: flag every row that falls
        # inside at least one window containing >= min_count rows.
        left = 0
        for right in range(len(ts)):
            while ts[right] - ts[left] > window:
                left += 1
            if right - left + 1 >= config.min_count:
                for i in range(left, right + 1):
                    result.loc[idx_sorted[i]] = True

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["comment_id", "original_comment", "visitor_email", "visitor_name", "comment_created_at"]


def apply_prefilter(
    df: pd.DataFrame,
    known_junk_strings: tuple[str, ...],
    burst_config: BurstConfig,
) -> pd.DataFrame:
    """
    Adds `is_prefiltered_junk` (bool) and `prefilter_reason` (str|None) to
    a copy of df. Does not drop or reorder any row.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"apply_prefilter missing required columns: {sorted(missing)}")

    df = df.copy()
    reasons = pd.Series(None, index=df.index, dtype=object)

    # Rules 1-5, in priority order, only considering rows not yet flagged.
    single_row_rules = [
        lambda c: detect_known_junk_string(c, known_junk_strings),
        detect_null_or_near_empty,
        detect_phone_number_stub,
        detect_unfilled_template,
        detect_test_marker,
        detect_keyboard_mash_sequence,
    ]
    for rule in single_row_rules:
        unflagged = reasons.isna()
        applied = df.loc[unflagged, "original_comment"].apply(rule)
        reasons.loc[unflagged] = applied

    # Rule 7 — needs whole-dataframe context; only considers still-unflagged rows.
    already_flagged = reasons.notna()
    burst_hits = detect_burst_repetition(df, burst_config, already_flagged)
    reasons.loc[burst_hits] = "burst_repetition"

    df["prefilter_reason"] = reasons
    df["is_prefiltered_junk"] = reasons.notna()
    return df
