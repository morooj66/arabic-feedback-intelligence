"""
Phase 3 tests: deterministic spam/junk prefilter.

Covers all eight scenarios required for Phase 3 sign-off:
  1. the known 3,875-row bot flood is detected
  2. known unfilled form-template samples are detected
  3. phone-number-only stubs are detected
  4. null/blank rows are handled safely (flagged, not crashed on)
  5. legitimate high-volume submitters are NOT excluded for volume alone
  6. ordinary comments containing "تجربة" as a real word are NOT excluded
  7. re-running Phase 3 is idempotent
  8. no input row is ever deleted
"""

import datetime as dt

import pandas as pd
import pytest

from config import settings
from rcs.ingest.cleaning import clean
from rcs.ingest.loader import load_raw
from rcs.prefilter.spam_rules import (
    BurstConfig,
    apply_prefilter,
    detect_keyboard_mash_sequence,
    detect_known_junk_string,
    detect_null_or_near_empty,
    detect_phone_number_stub,
    detect_test_marker,
    detect_unfilled_template,
)
from rcs.review.repository import (
    apply_prefilter_results,
    get_engine,
    init_db,
    load_for_prefilter,
    upsert_review_items,
)

DEFAULT_BURST = BurstConfig(min_count=settings.BURST_MIN_COUNT, window_hours=settings.BURST_WINDOW_HOURS)


# ---------------------------------------------------------------------------
# Rule 1 — exact known junk string
# ---------------------------------------------------------------------------

def test_known_junk_string_detected():
    bot_string = settings.KNOWN_JUNK_STRINGS[0]
    assert detect_known_junk_string(bot_string, settings.KNOWN_JUNK_STRINGS) == "exact_match_known_bot_string"


def test_known_junk_string_near_miss_not_detected():
    bot_string = settings.KNOWN_JUNK_STRINGS[0]
    assert detect_known_junk_string(bot_string + " extra", settings.KNOWN_JUNK_STRINGS) is None


# ---------------------------------------------------------------------------
# Rule 2 — null / near-empty
# ---------------------------------------------------------------------------

def test_null_content_detected():
    assert detect_null_or_near_empty(None) == "null_content"


def test_near_empty_content_detected():
    assert detect_null_or_near_empty("1") == "near_empty_content"
    assert detect_null_or_near_empty("..") == "near_empty_content"


def test_ordinary_short_word_not_treated_as_near_empty():
    # 3+ characters is a real (if short) word, not near-empty junk
    assert detect_null_or_near_empty("نور") is None


# ---------------------------------------------------------------------------
# Rule 3 — phone-number-only stub
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    "Phone Number:",
    "Phone Number: 0555555555",
    " Phone Number: 0555555555",
    "ص Phone Number: ",
    "الجوال: 0500000001",
    "رقم الجوال: 0500000002",
])
def test_phone_number_only_stub_detected(content):
    assert detect_phone_number_stub(content) == "phone_number_only_stub"


def test_phone_number_with_real_content_not_flagged():
    content = "هذه كلمة جميلة جدا وأتمنى إضافتها للمعجم Phone Number: 0555555555"
    assert detect_phone_number_stub(content) is None


def test_no_phone_label_not_flagged():
    assert detect_phone_number_stub("مرحبا كيف الحال") is None


# ---------------------------------------------------------------------------
# Rule 4 — unfilled form template
# ---------------------------------------------------------------------------

def test_unfilled_template_fully_blank_detected():
    content = (
        "الكلمة: Test\n اللهجة: فصيحة\n المعنى الأول: S1\n المثال الأول: E1\n"
        " المعنى الثاني: S2\n المثال الثاني: E2\n المعنى الثالث: S3\n المثال الثالث: E3"
    )
    assert detect_unfilled_template(content) == "unfilled_form_template"


def test_unfilled_template_empty_meaning_detected():
    content = "الكلمة: \n اللهجة: نجدية\n المعنى الأول: S\n المثال الأول: \n المعنى الثاني: \n"
    assert detect_unfilled_template(content) == "unfilled_form_template"


def test_filled_template_with_real_meaning_not_flagged():
    # Real word, real meaning provided, example intentionally left blank —
    # must NOT be treated as junk just because the optional example is empty.
    content = "الكلمة: نور\n اللهجة: نجدية\n المعنى الأول: الضياء الذي يُبصَر به الأشياء\n المثال الأول:\n"
    assert detect_unfilled_template(content) is None


def test_non_template_content_not_flagged():
    assert detect_unfilled_template("شميم.") is None


# ---------------------------------------------------------------------------
# Rule 5 — safe test markers (conservative, whole-string only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["test", "Test", "TEST", "تجربة", "تست", "test123", "Test\nTest\nTest"])
def test_safe_test_marker_whole_string_detected(content):
    assert detect_test_marker(content) == "safe_test_marker"


def test_test_word_inside_real_sentence_not_flagged():
    """
    Explicit product requirement: a comment containing 'تجربة' as a normal
    word within real prose must remain eligible for classification.
    """
    content = (
        "هذه تجربة رائعة لتعلم اللغة العربية واستخدام المعجم بشكل يومي، "
        "وأتمنى أن تستمر هذه التجربة الجميلة لفترة أطول."
    )
    assert detect_test_marker(content) is None


def test_long_real_comment_mentioning_test_word_not_flagged():
    content = "اختبرت الموقع اليوم test وأعجبني كثيرا، شكرا لكم على هذا الجهد الرائع في تطوير المعجم"
    assert detect_test_marker(content) is None


# ---------------------------------------------------------------------------
# Rule 6 — narrow keyboard-mash sequence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    "asdf qwer zxcv",
    "qwert asdfg zxcvbn",
    "fdsa rewq vcxz",
])
def test_whole_comment_keyboard_mash_sequence_detected(content):
    assert detect_keyboard_mash_sequence(content) == "keyboard_mash_sequence"


@pytest.mark.parametrize("content", [
    "هذا تعليق عربي صحيح",
    "نور",
    "dictionary",
    "qwer",
    "qwer asdf",
    "HTTP API SDK",
    "React Native app",
    "John Michael Smith",
    "الموقع يعرض Error 500",
    "hello world example",
])
def test_keyboard_mash_rule_preserves_valid_arabic_mixed_technical_and_word_submissions(content):
    assert detect_keyboard_mash_sequence(content) is None


def test_apply_prefilter_marks_keyboard_mash_without_dropping_row():
    df = _burst_df([{
        "original_comment": "asdf qwer zxcv",
        "visitor_email": None,
        "visitor_name": None,
        "comment_created_at": dt.datetime(2026, 10, 17),
    }])
    out = apply_prefilter(df, settings.KNOWN_JUNK_STRINGS, DEFAULT_BURST)
    assert len(out) == 1
    assert bool(out.iloc[0]["is_prefiltered_junk"])
    assert out.iloc[0]["prefilter_reason"] == "keyboard_mash_sequence"


# ---------------------------------------------------------------------------
# Rule 6 — burst / repetition (unit-level, synthetic data)
# ---------------------------------------------------------------------------

def _burst_df(rows):
    df = pd.DataFrame(rows)
    df["comment_id"] = [f"id{i}" for i in range(len(df))]
    return df


def test_burst_detects_rapid_identical_submissions():
    base = dt.datetime(2025, 1, 1, 12, 0, 0)
    rows = [
        {"original_comment": "spamtext", "visitor_email": "bot@x.com", "visitor_name": None,
         "comment_created_at": base + dt.timedelta(minutes=i)}
        for i in range(12)
    ]
    df = _burst_df(rows)
    flagged = pd.Series(False, index=df.index)
    result = __import__("rcs.prefilter.spam_rules", fromlist=["detect_burst_repetition"]).detect_burst_repetition(
        df, BurstConfig(min_count=10, window_hours=24), flagged
    )
    assert result.all()


def test_burst_does_not_flag_high_volume_distinct_content():
    """
    Explicit product requirement: many DIFFERENT legitimate submissions
    from the same person must never be treated as a burst.
    """
    base = dt.datetime(2025, 1, 1, 12, 0, 0)
    rows = [
        {"original_comment": f"كلمة رقم {i} ذات معنى مختلف تماما", "visitor_email": "prolific@x.com",
         "visitor_name": None, "comment_created_at": base + dt.timedelta(minutes=i)}
        for i in range(50)
    ]
    df = _burst_df(rows)
    flagged = pd.Series(False, index=df.index)
    result = __import__("rcs.prefilter.spam_rules", fromlist=["detect_burst_repetition"]).detect_burst_repetition(
        df, BurstConfig(min_count=10, window_hours=24), flagged
    )
    assert not result.any()


def test_burst_does_not_flag_identical_content_spread_over_long_time():
    base = dt.datetime(2025, 1, 1, 12, 0, 0)
    rows = [
        {"original_comment": "نفس التعليق", "visitor_email": "person@x.com", "visitor_name": None,
         "comment_created_at": base + dt.timedelta(days=40 * i)}  # far apart: not a burst
        for i in range(12)
    ]
    df = _burst_df(rows)
    flagged = pd.Series(False, index=df.index)
    result = __import__("rcs.prefilter.spam_rules", fromlist=["detect_burst_repetition"]).detect_burst_repetition(
        df, BurstConfig(min_count=10, window_hours=24), flagged
    )
    assert not result.any()


def test_burst_requires_identity_signal():
    """Two different anonymous visitors independently writing the same
    short real word must not be clustered into a burst."""
    base = dt.datetime(2025, 1, 1, 12, 0, 0)
    rows = [
        {"original_comment": "شكرا جزيلا لكم", "visitor_email": None, "visitor_name": None,
         "comment_created_at": base + dt.timedelta(minutes=i)}
        for i in range(20)
    ]
    df = _burst_df(rows)
    flagged = pd.Series(False, index=df.index)
    result = __import__("rcs.prefilter.spam_rules", fromlist=["detect_burst_repetition"]).detect_burst_repetition(
        df, BurstConfig(min_count=10, window_hours=24), flagged
    )
    assert not result.any()


# ---------------------------------------------------------------------------
# apply_prefilter — never drops rows
# ---------------------------------------------------------------------------

def test_apply_prefilter_preserves_row_count_and_order():
    df = _burst_df([
        {"original_comment": "test", "visitor_email": "a@x.com", "visitor_name": None,
         "comment_created_at": dt.datetime(2025, 1, 1)},
        {"original_comment": None, "visitor_email": None, "visitor_name": None,
         "comment_created_at": dt.datetime(2025, 1, 1)},
        {"original_comment": "شميم.", "visitor_email": "b@x.com", "visitor_name": None,
         "comment_created_at": dt.datetime(2025, 1, 1)},
    ])
    out = apply_prefilter(df, settings.KNOWN_JUNK_STRINGS, DEFAULT_BURST)
    assert len(out) == len(df)
    assert list(out["comment_id"]) == list(df["comment_id"])


# ---------------------------------------------------------------------------
# End-to-end against the real historical file / SQLite
# ---------------------------------------------------------------------------

SOURCE_PATH = settings.RAW_DIR / settings.SOURCE_FILE


@pytest.fixture(scope="module")
def prefiltered_real_df():
    if not SOURCE_PATH.exists():
        pytest.skip(f"historical source file not present at {SOURCE_PATH}")
    as_of = dt.datetime(2026, 8, 10, 12, 0, 0)
    raw_df = load_raw(SOURCE_PATH, settings.SOURCE_COLUMNS)
    cleaned = clean(raw_df, as_of=as_of, recent_max_days=14, delayed_max_days=180)
    result = apply_prefilter(cleaned, settings.KNOWN_JUNK_STRINGS, DEFAULT_BURST)
    return raw_df, result


def test_bot_flood_fully_detected(prefiltered_real_df):
    _, result = prefiltered_real_df
    bot_string = settings.KNOWN_JUNK_STRINGS[0]
    bot_rows = result[result["original_comment"] == bot_string]
    assert len(bot_rows) == 3875
    assert (bot_rows["is_prefiltered_junk"]).all()
    assert (bot_rows["prefilter_reason"] == "exact_match_known_bot_string").all()


def test_real_data_no_rows_dropped(prefiltered_real_df):
    raw_df, result = prefiltered_real_df
    assert len(result) == len(raw_df)


def test_real_data_lalita_not_excluded_for_volume(prefiltered_real_df):
    """The 748-row prolific-but-legitimate account from data discovery
    must pass through untouched — different content each time."""
    _, result = prefiltered_real_df
    # The account is identified by the SHA-256 of its address, so the real e-mail is never stored in the repository.
    import hashlib
    prolific = "a9f1c10080e8334c63ad34b542e46186637049706dda83a1311ccb8faf26fab9"
    lalita = result[result["visitor_email"].map(lambda e: hashlib.sha256(str(e).strip().encode("utf-8")).hexdigest()) == prolific]
    assert len(lalita) == 748
    assert not lalita["is_prefiltered_junk"].any()


def test_real_data_reason_counts_are_disjoint_and_sum_correctly(prefiltered_real_df):
    _, result = prefiltered_real_df
    flagged = result[result["is_prefiltered_junk"]]
    assert flagged["prefilter_reason"].notna().all()
    unflagged = result[~result["is_prefiltered_junk"]]
    assert unflagged["prefilter_reason"].isna().all()


def test_pipeline_prefilter_idempotent_via_sqlite(tmp_path, prefiltered_real_df):
    raw_df, result = prefiltered_real_df

    columns = [
        "comment_id", "source_file", "source_row_index", "original_comment",
        "original_title", "original_type", "entry_id", "lemma", "visitor_name",
        "visitor_email", "platform", "comment_created_at", "is_exact_duplicate_of",
        "ingested_at", "comment_age_days", "age_group", "review_status", "ready_to_send",
    ]
    records = result[columns].to_dict(orient="records")

    db_path = tmp_path / "prefilter_test.sqlite"
    engine = get_engine(db_path)
    init_db(engine)
    upsert_review_items(engine, records)

    prefilter_cols = result[["comment_id", "is_prefiltered_junk", "prefilter_reason"]]

    first = apply_prefilter_results(engine, prefilter_cols)
    second = apply_prefilter_results(engine, prefilter_cols)

    assert first == second  # identical excluded/passed counts on both runs

    df_after = load_for_prefilter(engine)
    assert len(df_after) == len(raw_df)  # still no rows lost after two prefilter passes


def test_null_rows_handled_safely_end_to_end(prefiltered_real_df):
    _, result = prefiltered_real_df
    null_rows = result[result["original_comment"].isna()]
    assert len(null_rows) > 0
    assert (null_rows["is_prefiltered_junk"]).all()
    assert (null_rows["prefilter_reason"] == "null_content").all()


def test_duplicate_flag_independent_of_spam_flag(prefiltered_real_df):
    """
    A row can be an exact duplicate without being spam, and vice versa —
    the two signals must not be conflated.
    """
    _, result = prefiltered_real_df
    duplicates_not_junk = result[result["is_exact_duplicate_of"].notna() & (~result["is_prefiltered_junk"])]
    junk_not_duplicate = result[result["is_prefiltered_junk"] & result["is_exact_duplicate_of"].isna()]
    assert len(duplicates_not_junk) > 0
    assert len(junk_not_duplicate) > 0


# ---------------------------------------------------------------------------
# Hardening fix: burst detection must never consider a NaT timestamp
# ---------------------------------------------------------------------------

def test_burst_excludes_rows_with_missing_timestamp():
    """
    A row with no timestamp must never be swept into a burst group, and
    must never itself anchor/extend a window for rows that DO have valid
    timestamps. Content/identity here match the rapid-burst fixture above
    exactly, except one row's timestamp is NaT.
    """
    from rcs.prefilter.spam_rules import detect_burst_repetition

    base = dt.datetime(2025, 1, 1, 12, 0, 0)
    rows = [
        {"original_comment": "spamtext", "visitor_email": "bot@x.com", "visitor_name": None,
         "comment_created_at": (pd.NaT if i == 0 else base + dt.timedelta(minutes=i))}
        for i in range(12)
    ]
    df = _burst_df(rows)
    df["comment_created_at"] = pd.to_datetime(df["comment_created_at"])  # keep NaT as NaT, rest as Timestamp
    flagged = pd.Series(False, index=df.index)

    result = detect_burst_repetition(df, BurstConfig(min_count=10, window_hours=24), flagged)

    assert result.loc[0] == False  # the NaT row itself is never flagged
    # the remaining 11 valid-timestamp rows are one short of min_count=10... they're
    # actually 11, still >= 10, so they must still be flagged on their own merits
    assert result.loc[1:].all()


def test_burst_nat_row_does_not_block_valid_group_below_threshold():
    """
    If excluding the NaT row drops a group below min_count, the remaining
    valid rows must correctly NOT be flagged (proves the NaT row isn't
    silently counted toward the threshold either).
    """
    from rcs.prefilter.spam_rules import detect_burst_repetition

    base = dt.datetime(2025, 1, 1, 12, 0, 0)
    # 9 valid + 1 NaT = 10 rows total, but only 9 have usable timestamps —
    # must NOT reach min_count=10 once the NaT row is correctly excluded.
    rows = [
        {"original_comment": "spamtext2", "visitor_email": "bot2@x.com", "visitor_name": None,
         "comment_created_at": (pd.NaT if i == 0 else base + dt.timedelta(minutes=i))}
        for i in range(10)
    ]
    df = _burst_df(rows)
    df["comment_created_at"] = pd.to_datetime(df["comment_created_at"])
    flagged = pd.Series(False, index=df.index)

    result = detect_burst_repetition(df, BurstConfig(min_count=10, window_hours=24), flagged)
    assert not result.any()


# ---------------------------------------------------------------------------
# Exact-count regression test — real 8,413-row historical dataset
# ---------------------------------------------------------------------------
# These numbers were confirmed by direct inspection of the current
# prefilter output (see conversation record). They are asserted verbatim
# on purpose: this test exists to catch any future change — intentional
# or accidental — to rule behavior, config constants, or the historical
# source file itself. If a legitimate rule change ever needs to move
# these numbers, this test must be updated deliberately, not silently.

EXPECTED_TOTAL = 8413
EXPECTED_EXCLUDED = 4966
EXPECTED_REMAINING = 3447
EXPECTED_REASON_COUNTS = {
    "exact_match_known_bot_string": 3875,
    "phone_number_only_stub": 386,
    "null_content": 368,
    "burst_repetition": 140,
    "unfilled_form_template": 106,
    "safe_test_marker": 49,
    "near_empty_content": 42,
}


def test_real_data_exact_exclusion_counts(prefiltered_real_df):
    raw_df, result = prefiltered_real_df

    assert len(raw_df) == EXPECTED_TOTAL
    assert len(result) == EXPECTED_TOTAL  # no row deleted

    excluded = result[result["is_prefiltered_junk"]]
    remaining = result[~result["is_prefiltered_junk"]]

    assert len(excluded) == EXPECTED_EXCLUDED
    assert len(remaining) == EXPECTED_REMAINING
    assert len(excluded) + len(remaining) == EXPECTED_TOTAL

    actual_reason_counts = excluded["prefilter_reason"].value_counts().to_dict()
    assert actual_reason_counts == EXPECTED_REASON_COUNTS
    assert sum(EXPECTED_REASON_COUNTS.values()) == EXPECTED_EXCLUDED

    # exactly one reason per excluded row, and none for passed-through rows
    assert excluded["prefilter_reason"].notna().all()
    assert (excluded["prefilter_reason"].isin(EXPECTED_REASON_COUNTS.keys())).all()
    assert remaining["prefilter_reason"].isna().all()
