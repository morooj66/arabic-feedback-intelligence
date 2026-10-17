"""Safe Excel upload handling: filename sanitization, path safety, and
workbook validation. No frozen models are touched by these tests."""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.review.excel_review_flow import write_fixture_workbook
from rcs.review.excel_upload import (
    UploadRejected,
    resolve_uploaded_workbook,
    sanitize_upload_filename,
    save_uploaded_workbook,
    validate_workbook,
)


def test_sanitize_filename_strips_path_components():
    assert sanitize_upload_filename("../../etc/passwd.xlsx") == "passwd.xlsx"
    assert sanitize_upload_filename("/absolute/path/comments.xlsx") == "comments.xlsx"
    # Backslashes aren't POSIX path separators, so on this platform the whole
    # string is one "name" — the safety property under test is that no
    # literal ".." survives into the final filename, not the exact stem.
    assert ".." not in sanitize_upload_filename("..\\..\\windows\\evil.xlsx")


def test_sanitize_filename_keeps_arabic_and_safe_characters():
    name = sanitize_upload_filename("تعليقات الزوار 2026.xlsx")
    assert name.endswith(".xlsx")
    assert "تعليقات" in name


def test_sanitize_filename_defaults_when_empty_or_missing():
    assert sanitize_upload_filename(None) == "upload.xlsx"
    assert sanitize_upload_filename("") == "upload.xlsx"
    assert sanitize_upload_filename("   ") == "upload.xlsx"


def test_save_uploaded_workbook_preserves_bytes_and_original_name(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    data = b"not a real xlsx but bytes are what we're checking here"
    saved = save_uploaded_workbook("visitors report.xlsx", data)
    assert saved.path.read_bytes() == data  # exact bytes preserved, unmodified
    assert saved.original_filename == "visitors_report.xlsx"
    assert saved.path.parent.name == saved.token
    assert len(saved.token) == 32  # opaque uuid4 hex token


def test_save_uploaded_workbook_rejects_non_xlsx(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    with pytest.raises(UploadRejected) as exc:
        save_uploaded_workbook("script.exe", b"data")
    assert exc.value.code == "unsupported_file_type"


def test_save_uploaded_workbook_rejects_empty_and_oversized(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "EXCEL_UPLOAD_MAX_BYTES", 10)

    with pytest.raises(UploadRejected) as exc:
        save_uploaded_workbook("x.xlsx", b"")
    assert exc.value.code == "empty_upload"

    with pytest.raises(UploadRejected) as exc:
        save_uploaded_workbook("x.xlsx", b"0123456789ABCDEF")
    assert exc.value.code == "file_too_large"


def test_resolve_uploaded_workbook_round_trips(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    saved = save_uploaded_workbook("comments.xlsx", b"xlsx-bytes")
    resolved = resolve_uploaded_workbook(saved.token)
    assert resolved == saved.path


def test_resolve_uploaded_workbook_rejects_path_traversal_tokens(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    for bad_token in ("../../etc", "..", "/etc/passwd", "not-a-hex-token"):
        with pytest.raises(UploadRejected) as exc:
            resolve_uploaded_workbook(bad_token)
        assert exc.value.code == "invalid_upload_token"


def test_resolve_uploaded_workbook_rejects_unknown_token(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    with pytest.raises(UploadRejected) as exc:
        resolve_uploaded_workbook("0" * 32)
    assert exc.value.code == "upload_not_found"


def test_validate_workbook_accepts_well_formed_file(tmp_path):
    path = write_fixture_workbook(
        tmp_path / "ok.xlsx",
        [
            {
                "original_comment": "تعليق حقيقي وكافٍ للاختبار",
                "original_title": None,
                "entry_id": None,
                "lemma": None,
                "visitor_name": "زائر",
                "visitor_email": "v@example.com",
                "original_type": "رسالة",
                "comment_created_at": dt.datetime(2026, 8, 1),
                "platform": "الرياض",
            }
        ],
    )
    result = validate_workbook(path)
    assert result.ok is True
    assert result.total_rows == 1
    assert result.errors == []


def test_validate_workbook_rejects_missing_columns(tmp_path):
    import pandas as pd

    path = tmp_path / "bad_columns.xlsx"
    pd.DataFrame({"عمود غير متوقع": ["قيمة"]}).to_excel(path, index=False)
    result = validate_workbook(path)
    assert result.ok is False
    assert any("الأعمدة المتوقعة" in e for e in result.errors)


def test_validate_workbook_rejects_empty_workbook(tmp_path):
    path = write_fixture_workbook(tmp_path / "empty.xlsx", [])
    result = validate_workbook(path)
    assert result.ok is False
    assert result.total_rows == 0


def test_validate_workbook_rejects_unreadable_file(tmp_path):
    path = tmp_path / "not_really_xlsx.xlsx"
    path.write_bytes(b"this is not a zip/xlsx file at all")
    result = validate_workbook(path)
    assert result.ok is False
    assert result.errors
