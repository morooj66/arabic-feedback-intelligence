"""Closed local demo package: ingest → classify persist → generate → review → export."""

from __future__ import annotations

import pandas as pd

from rcs.review.excel_review_flow import (
    _EXPORT_SHEET1_HEADERS,
    _EXPORT_SHEET1_TITLE,
    _EXPORT_SHEET2_HEADERS,
    _EXPORT_SHEET2_TITLE,
    write_fixture_workbook,
)
from rcs.review.local_demo_package import _fixture_rows, run_closed_local_demo


def test_closed_local_demo_package(tmp_path):
    source = write_fixture_workbook(tmp_path / "src.xlsx", _fixture_rows())
    out = tmp_path / "reviewed_responses_output.xlsx"
    report = run_closed_local_demo(
        db_path=tmp_path / "pkg.sqlite",
        export_path=out,
        source_workbook=source,
        live_drafts_db=tmp_path / "missing.sqlite",
    )
    assert out.exists()
    assert report["external_send"] is False
    actions = {row["action"]: row for row in report["actions"]}
    assert actions["SEND_AS_IS"]["outbound_status"] == "PENDING"
    assert actions["EDIT_AND_SEND"]["outbound_status"] == "PENDING"
    assert actions["EDIT_AND_SEND"]["final_response"] != actions["EDIT_AND_SEND"]["suggested_response"]
    assert actions["EDIT_AND_SEND_MANUAL"]["outbound_status"] == "PENDING"
    assert not actions["EDIT_AND_SEND_MANUAL"]["suggested_response"]
    assert actions["NO_RESPONSE_NEEDED"]["outbound_status"] == "NOT_APPLICABLE"
    assert report["final_counts"]["outcomes"]["SEND_AS_IS"] == 1
    assert report["final_counts"]["outcomes"]["EDIT_AND_SEND"] == 2
    assert report["final_counts"]["outcomes"]["NO_RESPONSE_NEEDED"] == 1
    assert report["final_counts"]["pending"] == 0
    # Phase 4: two-sheet export — Sheet 1 business-facing (Arabic headers),
    # Sheet 2 technical/audit. Same row order in both (one row per source
    # row), so a straight column concat reconstructs a combined view.
    sheet1 = pd.read_excel(out, sheet_name=_EXPORT_SHEET1_TITLE, dtype=str)
    sheet2 = pd.read_excel(out, sheet_name=_EXPORT_SHEET2_TITLE, dtype=str)
    assert list(sheet1.columns) == _EXPORT_SHEET1_HEADERS
    assert list(sheet2.columns) == _EXPORT_SHEET2_HEADERS
    frame = pd.concat([sheet1, sheet2], axis=1)
    send = frame[frame["إجراء المراجع"] == "SEND_AS_IS"].iloc[0]
    assert send["outbound_status"] == "PENDING"
    assert send["الرد النهائي"] == send["الرد المقترح"]
    assert (frame["حالة المراجعة"] == "SENT").sum() == 0
