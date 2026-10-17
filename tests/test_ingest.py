"""
Phase 2 tests: loader + cleaning.

Covers:
- text normalization (whitespace/NBSP/control chars, empty -> None)
- exact full-row duplicate flagging (synthetic + real-data cross-check)
- comment_id stability / determinism
- age_days / age_group bucketing at boundaries
- end-to-end: real historical file row counts, duplicate handling,
  idempotent re-run via the SQLite repository
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from config import settings
from rcs.ingest.cleaning import InvalidCommentCreatedAtError, clean, compute_age, flag_exact_duplicates, normalize_text
from rcs.ingest.loader import load_raw, _comment_id
from rcs.review.repository import get_engine, init_db, upsert_review_items, count_rows


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_strips_whitespace():
    assert normalize_text("  hello  \n") == "hello"


def test_normalize_text_nbsp_becomes_space():
    assert normalize_text("hello\xa0world") == "hello world"


def test_normalize_text_control_chars_removed():
    assert normalize_text("hel\x00lo\x0bworld") == "hello world" or "\x00" not in normalize_text("hel\x00lo\x0bworld")


def test_normalize_text_empty_after_strip_becomes_none():
    assert normalize_text("   \n\n  ") is None
    assert normalize_text("") is None


def test_normalize_text_none_passthrough():
    assert normalize_text(None) is None


def test_normalize_text_preserves_arabic_wording():
    text = "شكرًا لتواصلكم معنا"
    assert normalize_text(text) == text


# ---------------------------------------------------------------------------
# comment_id
# ---------------------------------------------------------------------------

def test_comment_id_deterministic():
    a = _comment_id("file.xlsx", 5)
    b = _comment_id("file.xlsx", 5)
    assert a == b


def test_comment_id_varies_by_row_index():
    a = _comment_id("file.xlsx", 5)
    b = _comment_id("file.xlsx", 6)
    assert a != b


def test_comment_id_varies_by_source_file():
    a = _comment_id("file_a.xlsx", 5)
    b = _comment_id("file_b.xlsx", 5)
    assert a != b


# ---------------------------------------------------------------------------
# flag_exact_duplicates
# ---------------------------------------------------------------------------

def _make_df(rows):
    df = pd.DataFrame(rows)
    df["comment_id"] = [f"id{i}" for i in range(len(df))]
    return df


def test_flag_exact_duplicates_marks_repeats_but_not_first():
    df = _make_df([
        {"original_comment": "test", "original_title": "t", "original_type": "x",
         "entry_id": None, "lemma": None, "visitor_name": "a", "visitor_email": "a@x.com",
         "platform": "p", "comment_created_at": pd.Timestamp("2025-01-01")},
        {"original_comment": "test", "original_title": "t", "original_type": "x",
         "entry_id": None, "lemma": None, "visitor_name": "a", "visitor_email": "a@x.com",
         "platform": "p", "comment_created_at": pd.Timestamp("2025-01-01")},
        {"original_comment": "different", "original_title": "t", "original_type": "x",
         "entry_id": None, "lemma": None, "visitor_name": "a", "visitor_email": "a@x.com",
         "platform": "p", "comment_created_at": pd.Timestamp("2025-01-01")},
    ])
    out = flag_exact_duplicates(df)
    assert out.loc[0, "is_exact_duplicate_of"] is None
    assert out.loc[1, "is_exact_duplicate_of"] == "id0"
    assert out.loc[2, "is_exact_duplicate_of"] is None


def test_flag_exact_duplicates_handles_nan_keys_consistently():
    df = _make_df([
        {"original_comment": None, "original_title": None, "original_type": "x",
         "entry_id": None, "lemma": None, "visitor_name": None, "visitor_email": None,
         "platform": "p", "comment_created_at": pd.Timestamp("2025-01-01")},
        {"original_comment": None, "original_title": None, "original_type": "x",
         "entry_id": None, "lemma": None, "visitor_name": None, "visitor_email": None,
         "platform": "p", "comment_created_at": pd.Timestamp("2025-01-01")},
    ])
    out = flag_exact_duplicates(df)
    # two rows with identical (all-null) content are still a legitimate exact-duplicate pair
    assert out.loc[1, "is_exact_duplicate_of"] == "id0"


def test_flag_exact_duplicates_does_not_drop_rows():
    df = _make_df([
        {"original_comment": "a", "original_title": None, "original_type": "x", "entry_id": None,
         "lemma": None, "visitor_name": None, "visitor_email": None, "platform": "p",
         "comment_created_at": pd.Timestamp("2025-01-01")},
        {"original_comment": "a", "original_title": None, "original_type": "x", "entry_id": None,
         "lemma": None, "visitor_name": None, "visitor_email": None, "platform": "p",
         "comment_created_at": pd.Timestamp("2025-01-01")},
    ])
    out = flag_exact_duplicates(df)
    assert len(out) == len(df)  # nothing removed, only flagged


# ---------------------------------------------------------------------------
# compute_age
# ---------------------------------------------------------------------------

def test_compute_age_bucket_boundaries():
    as_of = dt.datetime(2026, 1, 15)
    df = pd.DataFrame({
        "comment_created_at": [
            as_of - dt.timedelta(days=1),     # recent
            as_of - dt.timedelta(days=14),    # recent (boundary, inclusive)
            as_of - dt.timedelta(days=14.1),  # delayed
            as_of - dt.timedelta(days=180),   # delayed (boundary, inclusive)
            as_of - dt.timedelta(days=180.1), # old
        ]
    })
    out = compute_age(df, as_of, recent_max_days=14, delayed_max_days=180)
    assert list(out["age_group"]) == ["recent", "recent", "delayed", "delayed", "old"]


def test_compute_age_null_created_at_yields_null_bucket():
    as_of = dt.datetime(2026, 1, 15)
    df = pd.DataFrame({"comment_created_at": [pd.NaT]})
    out = compute_age(df, as_of, recent_max_days=14, delayed_max_days=180)
    assert out.loc[0, "age_group"] is None


# ---------------------------------------------------------------------------
# compute_age's comment_created_at type normalization (Mongo runtime
# boundary milestone -- fixes the pre-existing defect where age
# calculation read a parsed value but never wrote it back, leaving
# text-formatted dates as raw strings reaching persistence).
# ---------------------------------------------------------------------------


def test_compute_age_native_excel_datetime_passes_through_unchanged():
    """A column that's already real datetimes (native Excel date cells,
    as openpyxl/pandas already produce them) must keep working exactly
    as before -- same dtype, same values, same age/bucket results."""
    as_of = dt.datetime(2026, 1, 15)
    df = pd.DataFrame({"comment_created_at": [as_of - dt.timedelta(days=1)]})
    assert df["comment_created_at"].dtype.kind == "M"  # already datetime64 before the fix runs
    out = compute_age(df, as_of, recent_max_days=14, delayed_max_days=180)
    assert out["comment_created_at"].dtype.kind == "M"
    assert out.loc[0, "comment_created_at"] == pd.Timestamp(as_of - dt.timedelta(days=1))
    assert out.loc[0, "age_group"] == "recent"


def test_compute_age_normalizes_text_formatted_date_in_place():
    """The actual defect: a text-formatted date cell (a plain Python
    str, exactly what openpyxl produces for a text-formatted Excel
    column) must come out the OTHER SIDE of compute_age as a real
    datetime, not merely be parsed internally for the age calculation
    and then discarded."""
    as_of = dt.datetime(2026, 1, 15)
    df = pd.DataFrame({"comment_created_at": ["2026-01-14 09:00:00"]})
    assert df["comment_created_at"].dtype == object  # a plain string column going in
    out = compute_age(df, as_of, recent_max_days=14, delayed_max_days=180)
    assert out["comment_created_at"].dtype.kind == "M"  # datetime64 coming out
    assert out.loc[0, "comment_created_at"] == pd.Timestamp("2026-01-14 09:00:00")
    assert out.loc[0, "age_group"] == "recent"


def test_compute_age_blank_date_stays_null_not_an_error():
    """Missing/blank comment_created_at (None or empty string) must
    keep producing a null age/bucket, never raise -- unchanged by the
    fix."""
    as_of = dt.datetime(2026, 1, 15)
    df = pd.DataFrame({"comment_created_at": [None, "", pd.NaT]})
    out = compute_age(df, as_of, recent_max_days=14, delayed_max_days=180)
    assert out["comment_created_at"].isna().all()
    assert out["age_group"].isna().all() or all(v is None for v in out["age_group"])


def test_compute_age_invalid_date_raises_sanitized_error():
    """A genuinely unparseable date string must fail closed with a
    dedicated, sanitized exception -- not silently coerce to garbage,
    and not leak raw row content: the raised message names no comment
    text, no visitor data, nothing but the fact that the date column
    itself is invalid."""
    as_of = dt.datetime(2026, 1, 15)
    df = pd.DataFrame({"comment_created_at": ["definitely-not-a-date"]})
    with pytest.raises(InvalidCommentCreatedAtError) as exc_info:
        compute_age(df, as_of, recent_max_days=14, delayed_max_days=180)
    assert "definitely-not-a-date" not in str(exc_info.value)


def test_compute_age_invalid_date_reaches_job_runner_as_sanitized_type_name_only(tmp_path, monkeypatch):
    """End-to-end: an invalid comment_created_at value, ingested through
    the real Excel batch-processing entrypoint, must never expose its
    raw pandas exception text through the job's error_message field --
    only the sanitized `processing_failed:<ExceptionTypeName>` form the
    persistence-decoupling milestone's PII-leak fix already guarantees
    for every exception raised inside a processing job."""
    import time

    from config import settings as _settings
    from rcs.review.excel_review_flow import write_fixture_workbook
    from rcs.review.excel_upload import save_uploaded_workbook
    from rcs.review.job_runner import start_or_resume_processing
    from rcs.review.persistence import bundle_from_engine
    from rcs.review.repository import get_engine, init_db
    from rcs.review.sessions import create_session, get_job

    engine = get_engine(tmp_path / "invalid_date.sqlite")
    init_db(engine)
    monkeypatch.setattr(_settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")

    rows = [{
        "original_comment": "هذا تعليق طويل بما يكفي لتجاوز حد الفلترة", "original_title": None,
        "entry_id": None, "lemma": None, "visitor_name": "زائر",
        "visitor_email": "invalid-date-check@example.test", "original_type": "تعليق",
        "comment_created_at": "definitely-not-a-date-either", "platform": "الرياض",
    }]
    source = write_fixture_workbook(tmp_path / "invalid_date.xlsx", rows)
    saved = save_uploaded_workbook("invalid_date.xlsx", source.read_bytes())
    review_session = create_session(engine, original_filename="invalid_date.xlsx", source_file=saved.original_filename, upload_token=saved.token)

    job_id = start_or_resume_processing(engine, review_session.session_id)
    deadline = time.time() + 30
    job = get_job(engine, job_id)
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.1)
        job = get_job(engine, job_id)

    assert job.status == "failed"
    assert job.error_message == "processing_failed:InvalidCommentCreatedAtError"
    assert "definitely-not-a-date-either" not in job.error_message
    assert "invalid-date-check@example.test" not in job.error_message


# ---------------------------------------------------------------------------
# End-to-end against the real historical file
# ---------------------------------------------------------------------------

SOURCE_PATH = settings.RAW_DIR / settings.SOURCE_FILE


@pytest.fixture(scope="module")
def raw_df():
    if not SOURCE_PATH.exists():
        pytest.skip(f"historical source file not present at {SOURCE_PATH}")
    return load_raw(SOURCE_PATH, settings.SOURCE_COLUMNS)


def test_loader_row_count_matches_source(raw_df):
    expected = len(pd.read_excel(SOURCE_PATH))
    assert len(raw_df) == expected


def test_loader_comment_id_is_unique(raw_df):
    assert raw_df["comment_id"].is_unique


def test_loader_source_row_index_matches_original_order(raw_df):
    assert list(raw_df["source_row_index"]) == list(range(len(raw_df)))


def test_cleaning_duplicate_count_matches_independent_pandas_check(raw_df):
    as_of = dt.datetime(2026, 8, 10, 12, 0, 0)
    cleaned = clean(raw_df, as_of=as_of, recent_max_days=14, delayed_max_days=180)

    # Independently recompute exact-duplicate count using plain pandas
    # .duplicated() over the same key fields, so this test doesn't rely on
    # a hardcoded magic number that could silently drift from the data.
    from rcs.ingest.cleaning import DUPLICATE_KEY_FIELDS
    expected_dup_count = int(raw_df[DUPLICATE_KEY_FIELDS].duplicated(keep="first").sum())

    actual_dup_count = int(cleaned["is_exact_duplicate_of"].notna().sum())
    assert actual_dup_count == expected_dup_count


def test_cleaning_does_not_drop_or_add_rows(raw_df):
    as_of = dt.datetime(2026, 8, 10, 12, 0, 0)
    cleaned = clean(raw_df, as_of=as_of, recent_max_days=14, delayed_max_days=180)
    assert len(cleaned) == len(raw_df)


def test_cleaning_sets_unprocessed_status(raw_df):
    as_of = dt.datetime(2026, 8, 10, 12, 0, 0)
    cleaned = clean(raw_df, as_of=as_of, recent_max_days=14, delayed_max_days=180)
    assert (cleaned["review_status"] == "UNPROCESSED").all()
    assert (cleaned["ready_to_send"] == False).all()  # noqa: E712


def test_pipeline_idempotent_upsert(tmp_path, raw_df):
    as_of = dt.datetime(2026, 8, 10, 12, 0, 0)
    cleaned = clean(raw_df, as_of=as_of, recent_max_days=14, delayed_max_days=180)

    columns = [
        "comment_id", "source_file", "source_row_index", "original_comment",
        "original_title", "original_type", "entry_id", "lemma", "visitor_name",
        "visitor_email", "platform", "comment_created_at", "is_exact_duplicate_of",
        "ingested_at", "comment_age_days", "age_group", "review_status", "ready_to_send",
    ]
    records = cleaned[columns].to_dict(orient="records")

    db_path = tmp_path / "test_review.sqlite"
    engine = get_engine(db_path)
    init_db(engine)

    first = upsert_review_items(engine, records)
    assert first["inserted"] == len(records)
    assert first["updated"] == 0
    assert count_rows(engine) == len(records)

    second = upsert_review_items(engine, records)
    assert second["inserted"] == 0
    assert second["updated"] == len(records)
    assert count_rows(engine) == len(records)  # no duplicate rows created
