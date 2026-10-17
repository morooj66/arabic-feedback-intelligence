"""
Focused tests for scripts/offline_eval/reply_strategy/error_analysis_v2.py.

Fast and deterministic -- no real model load, no network, no Gemini
call, no DB write, no Excel mutation. Uses small synthetic fixtures in
tmp_path rather than the real (large, real-data) result files, so these
tests do not depend on the actual 25-case evaluation run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STRATEGY_DIR = _ROOT / "scripts" / "offline_eval" / "reply_strategy"
sys.path.insert(0, str(_STRATEGY_DIR))

import error_analysis_v2  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_error_tags_use_only_the_declared_taxonomy():
    for row_index, (tag, _rationale) in error_analysis_v2.ERROR_TAGS.items():
        assert tag in error_analysis_v2.TAXONOMY, f"row {row_index} uses undeclared tag {tag!r}"


def test_load_joined_rows_unblinds_correctly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [
        {"eval_id": "c1", "source_row_index": 1, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
         "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ENGAGE_WITH_CONTENT", "strategy_depth": "SUBSTANTIVE",
         "strategy_key_focus": "اعتبار", "comment_preview": "x", "reply_a": "a1", "reply_b": "b1"},
        {"eval_id": "c2", "source_row_index": 2, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
         "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ENGAGE_WITH_CONTENT", "strategy_depth": "SUBSTANTIVE",
         "strategy_key_focus": "التراكيب المتلازمة", "comment_preview": "y", "reply_a": "a2", "reply_b": "b2"},
    ])
    _write_jsonl(labels_path, [
        {"eval_id": "c1", "decision": "REPLY_1", "winner_variant": "A", "reviewed_at": "x"},
        {"eval_id": "c2", "decision": "REPLY_2", "winner_variant": "B", "reviewed_at": "x"},
    ])
    rows = error_analysis_v2.load_joined_rows(cases_path, labels_path)
    assert [r["source_row_index"] for r in rows] == [1, 2]  # sorted, deterministic order
    assert rows[0]["winner_variant"] == "A"
    assert rows[1]["winner_variant"] == "B"
    # Rows 1 and 2 exist in the real recorded ERROR_TAGS -- confirms the join reaches them.
    assert rows[0]["tag"] in error_analysis_v2.TAXONOMY
    assert rows[1]["tag"] == "GOOD_CONTENT_ENGAGEMENT"


def test_load_joined_rows_skips_unlabeled_cases(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [
        {"eval_id": "c1", "source_row_index": 1, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
         "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ENGAGE_WITH_CONTENT", "strategy_depth": "SUBSTANTIVE",
         "strategy_key_focus": "اعتبار", "comment_preview": "x", "reply_a": "a1", "reply_b": "b1"},
    ])
    _write_jsonl(labels_path, [])  # nothing reviewed yet
    rows = error_analysis_v2.load_joined_rows(cases_path, labels_path)
    assert rows == []


def test_unrecorded_row_falls_back_to_other_not_a_crash(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [
        {"eval_id": "cX", "source_row_index": 999999, "primary_intent": "SUGGESTION", "topic_subtype": "GENERAL",
         "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ACKNOWLEDGMENT_ONLY", "strategy_depth": "BRIEF",
         "strategy_key_focus": "GENERAL_COMMENT", "comment_preview": "z", "reply_a": "a", "reply_b": "b"},
    ])
    _write_jsonl(labels_path, [{"eval_id": "cX", "decision": "REPLY_1", "winner_variant": "A", "reviewed_at": "x"}])
    rows = error_analysis_v2.load_joined_rows(cases_path, labels_path)
    assert rows[0]["tag"] == "OTHER"


def test_win_counts():
    rows = [{"winner_variant": "A"}, {"winner_variant": "A"}, {"winner_variant": "B"}]
    counts = error_analysis_v2.win_counts(rows)
    assert counts["A"] == 2 and counts["B"] == 1


def test_aggregate_by_tag():
    rows = [{"tag": "GOOD_CONTENT_ENGAGEMENT"}, {"tag": "GOOD_CONTENT_ENGAGEMENT"}, {"tag": "OTHER"}]
    counts = error_analysis_v2.aggregate_by_tag(rows)
    assert counts["GOOD_CONTENT_ENGAGEMENT"] == 2
    assert counts["OTHER"] == 1


def test_aggregate_by_field_breaks_down_wins_per_group():
    rows = [
        {"primary_intent": "SUGGESTION", "winner_variant": "A"},
        {"primary_intent": "SUGGESTION", "winner_variant": "B"},
        {"primary_intent": "COMPLAINT", "winner_variant": "A"},
    ]
    grouped = error_analysis_v2.aggregate_by_field(rows, "primary_intent")
    assert grouped["SUGGESTION"]["A"] == 1 and grouped["SUGGESTION"]["B"] == 1
    assert grouped["COMPLAINT"]["A"] == 1


def test_real_result_files_reproduce_the_reported_aggregate_stats():
    """Integration check against the actual completed evaluation: the
    join + aggregation must reproduce the exact numbers already reported
    to the user (16 baseline / 9 V2 wins over 25 cases). Skips cleanly
    if the real result files are not present in this environment."""
    if not error_analysis_v2.CASES_PATH.exists() or not error_analysis_v2.LABELS_PATH.exists():
        import pytest
        pytest.skip("real V2 evaluation result files not present")
    rows = error_analysis_v2.load_joined_rows()
    assert len(rows) == 25
    counts = error_analysis_v2.win_counts(rows)
    assert counts["A"] == 16
    assert counts["B"] == 9


def test_analysis_reads_evaluation_artifacts_without_gemini_or_db_or_excel():
    source = (_STRATEGY_DIR / "error_analysis_v2.py").read_text(encoding="utf-8")
    assert "genai" not in source.lower()
    assert "build_live_gemini_client" not in source
    assert ".generate(" not in source
    assert "to_excel(" not in source
    assert "pymongo" not in source
    assert "sqlalchemy" not in source
