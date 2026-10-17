"""
Phase 2 — cleaning.

Scope, deliberately narrow (Phase 3+ handles spam/junk/content judgment):
  1. Text normalization: whitespace/newline/control-character cleanup only.
     Never touches Arabic diacritics or wording — this is not spam
     filtering, just making the stored text well-formed.
  2. Exact full-row duplicate detection: rows identical across every
     original field (same content, same title, same type, same visitor,
     same timestamp, ...) are almost certainly a double form-submission
     of the same event, not two independent comments. These are flagged
     via `is_exact_duplicate_of` (pointing at the first-seen row's
     comment_id) rather than deleted, so the audit trail is preserved.

     This is intentionally NOT the same thing as the repeated-content
     spam patterns found in data discovery (e.g. the same "test" string
     submitted thousands of times at different timestamps, or template
     placeholders reused by different visitors). Those are judged by
     content + burst patterns, not exact-row identity, and are handled by
     the Phase 3 prefilter, which still sees every row (duplicates are
     flagged, not removed).
  3. comment_age_days / age_group: pure date arithmetic against a
     reference "as_of" timestamp (ingestion time).
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata

import pandas as pd

_CONTROL_CHARS = "".join(chr(c) for c in list(range(0, 9)) + list(range(11, 13)) + list(range(14, 32)) + [127])
_CONTROL_CHARS_RE = re.compile("[" + re.escape(_CONTROL_CHARS) + "]")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")

# Fields that define "the same submission" for exact-duplicate detection.
DUPLICATE_KEY_FIELDS = [
    "original_comment",
    "original_title",
    "original_type",
    "entry_id",
    "lemma",
    "visitor_name",
    "visitor_email",
    "platform",
    "comment_created_at",
]


def normalize_text(value):
    """Whitespace/control-character cleanup. Leaves wording untouched."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if not isinstance(value, str):
        return value

    text = unicodedata.normalize("NFC", value)
    text = text.replace(" ", " ")  # non-breaking space -> regular space
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    text = text.strip()
    return text if text != "" else None


def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
    return df


def flag_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `is_exact_duplicate_of`: null for the first-seen row in each
    duplicate-key group, and the first-seen row's comment_id for every
    subsequent identical row. Original row order is preserved; nothing
    is dropped.
    """
    df = df.copy()
    key_cols = [c for c in DUPLICATE_KEY_FIELDS if c in df.columns]
    # NaN-safe key: fillna with a sentinel so groupby treats missing values
    # consistently rather than never matching (pandas default groupby
    # behavior with NaN keys).
    key = df[key_cols].fillna("\x00__NULL__\x00").astype(str).agg("\x1f".join, axis=1)

    first_seen: dict[str, str] = {}
    dup_of = []
    for k, cid in zip(key, df["comment_id"]):
        if k in first_seen:
            dup_of.append(first_seen[k])
        else:
            first_seen[k] = cid
            dup_of.append(None)
    df["is_exact_duplicate_of"] = dup_of
    return df


class InvalidCommentCreatedAtError(ValueError):
    """`comment_created_at` contains a value that cannot be parsed as a
    date. Deliberately carries no row content in its message -- callers
    surfacing this to a job/API error field must still only ever expose
    the exception type name, never str(exc), matching this codebase's
    existing sanitized-error-message convention (see
    rcs.review.job_runner)."""


def compute_age(df: pd.DataFrame, as_of: dt.datetime, recent_max_days: float, delayed_max_days: float) -> pd.DataFrame:
    """Also normalizes `comment_created_at` itself in place: a native
    Excel date cell is already a real datetime and passes through
    unchanged; a text-formatted date cell is parsed into one here so
    every downstream consumer (persistence included) sees a real
    datetime instead of a leftover string. Blank/missing values become
    NaT (preserved as None once written), exactly as before. A value
    that cannot be parsed as a date at all raises
    InvalidCommentCreatedAtError rather than reaching persistence with
    the wrong type."""
    df = df.copy()
    try:
        created = pd.to_datetime(df["comment_created_at"])
    except (ValueError, TypeError) as exc:
        raise InvalidCommentCreatedAtError(
            "comment_created_at contains a value that could not be parsed as a date"
        ) from exc
    df["comment_created_at"] = created
    if as_of.tzinfo is None:
        as_of_ts = pd.Timestamp(as_of)
    else:
        as_of_ts = pd.Timestamp(as_of).tz_localize(None)
    age_days = (as_of_ts - created).dt.total_seconds() / 86400.0
    df["comment_age_days"] = age_days

    def bucket(days):
        if pd.isna(days):
            return None
        if days <= recent_max_days:
            return "recent"
        if days <= delayed_max_days:
            return "delayed"
        return "old"

    df["age_group"] = df["comment_age_days"].apply(bucket)
    return df


def clean(
    df: pd.DataFrame,
    as_of: dt.datetime,
    recent_max_days: float,
    delayed_max_days: float,
) -> pd.DataFrame:
    """Full Phase 2 cleaning pipeline. Returns a new DataFrame; input is untouched."""
    text_cols = ["original_comment", "original_title", "visitor_name", "original_type", "platform", "lemma"]
    df = normalize_text_columns(df, text_cols)
    df = flag_exact_duplicates(df)
    df = compute_age(df, as_of, recent_max_days, delayed_max_days)
    df["ingested_at"] = as_of
    df["review_status"] = "UNPROCESSED"
    df["ready_to_send"] = False
    return df
