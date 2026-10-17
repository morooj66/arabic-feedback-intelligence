"""Tests for the B.3 human review wizard (no Gradio UI, no Gemini)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rcs.evaluation.b3_labeling import (
    B3_LABELER_ID,
    accept_ai_suggestion_row,
    ai_suggestion_to_form_labels,
    empty_b3_review_state,
    finalize_b3_review,
    import_b3_wizard_labels,
    is_review_complete,
    load_or_init_b3_review_state,
    load_remaining_train_rows,
    save_and_advance_b3_row,
    save_current_b3_row,
)
from rcs.evaluation.gold_dataset import PREEXISTING_AI_ASSISTED_GOLD_IDS, verify_post_b2_b3_invariants


def _mini_train_csv(path: Path) -> None:
    rows = [
        {
            "comment_id": "done1",
            "comment_text": "شكرا",
            "split": "train",
            "annotation_stage": "ai_assisted_train",
            "is_few_shot_example": "False",
            "gold_primary_intent": "THANKS_PRAISE",
            "gold_topic_subtype": "GENERAL",
            "gold_secondary_intent": "",
            "gold_subcategory": "",
            "gold_sentiment": "POSITIVE",
            "gold_annotation_status": "LABELED",
            "labeler_notes": "calibration",
            "labeled_by": "b2_calibration_wizard",
            "labeled_at": "2026-08-17",
            "ai_suggested_primary_intent": "THANKS_PRAISE",
            "ai_suggested_topic_subtype": "GENERAL",
            "ai_suggested_secondary_intent": "",
            "ai_suggested_subcategory": "",
            "ai_suggested_sentiment": "POSITIVE",
            "ai_suggested_confidence": "0.95",
            "ai_suggestion_model": "gemini-3.6-flash:b2-v3",
        },
        {
            "comment_id": "pending1",
            "comment_text": "أضف كلمة جديدة",
            "split": "train",
            "annotation_stage": "ai_assisted_train",
            "is_few_shot_example": "False",
            "gold_primary_intent": "",
            "gold_topic_subtype": "",
            "gold_secondary_intent": "",
            "gold_subcategory": "",
            "gold_sentiment": "",
            "gold_annotation_status": "",
            "labeler_notes": "",
            "labeled_by": "",
            "labeled_at": "",
            "ai_suggested_primary_intent": "SUGGESTION",
            "ai_suggested_topic_subtype": "WORD_SUBMISSION",
            "ai_suggested_secondary_intent": "",
            "ai_suggested_subcategory": "",
            "ai_suggested_sentiment": "NEUTRAL",
            "ai_suggested_confidence": "0.55",
            "ai_suggestion_model": "gemini-3.6-flash:b2-v3",
        },
        {
            "comment_id": "pending2",
            "comment_text": "؟؟",
            "split": "train",
            "annotation_stage": "ai_assisted_train",
            "is_few_shot_example": "False",
            "gold_primary_intent": "",
            "gold_topic_subtype": "",
            "gold_secondary_intent": "",
            "gold_subcategory": "",
            "gold_sentiment": "",
            "gold_annotation_status": "",
            "labeler_notes": "",
            "labeled_by": "",
            "labeled_at": "",
            "ai_suggested_primary_intent": "SPAM",
            "ai_suggested_topic_subtype": "GENERAL",
            "ai_suggested_secondary_intent": "",
            "ai_suggested_subcategory": "",
            "ai_suggested_sentiment": "NEUTRAL",
            "ai_suggested_confidence": "0.90",
            "ai_suggestion_model": "gemini-3.6-flash:b2-v3",
        },
        {
            "comment_id": "protected_val",
            "comment_text": "val row",
            "split": "val",
            "annotation_stage": "human_val",
            "is_few_shot_example": "False",
            "gold_primary_intent": "INQUIRY",
            "gold_topic_subtype": "GENERAL",
            "gold_secondary_intent": "",
            "gold_subcategory": "",
            "gold_sentiment": "NEUTRAL",
            "gold_annotation_status": "LABELED",
            "labeler_notes": "",
            "labeled_by": "human",
            "labeled_at": "2026-01-01",
            "ai_suggested_primary_intent": "",
            "ai_suggested_topic_subtype": "",
            "ai_suggested_secondary_intent": "",
            "ai_suggested_subcategory": "",
            "ai_suggested_sentiment": "",
            "ai_suggested_confidence": "",
            "ai_suggestion_model": "",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


@pytest.fixture()
def mini_b3(tmp_path: Path):
    csv_path = tmp_path / "gold.csv"
    review_path = tmp_path / "b3_review.json"
    marker_path = tmp_path / "complete.json"
    _mini_train_csv(csv_path)
    return csv_path, review_path, marker_path


def test_load_remaining_train_rows_excludes_completed(mini_b3):
    csv_path, _, _ = mini_b3
    rows = load_remaining_train_rows(csv_path)
    assert len(rows) == 2
    assert {r["comment_id"] for r in rows} == {"pending1", "pending2"}
    assert rows[0]["comment_id"] == "pending1"


def test_ai_suggestion_to_form_labels(mini_b3):
    csv_path, _, _ = mini_b3
    row = load_remaining_train_rows(csv_path)[0]
    labels = ai_suggestion_to_form_labels(row)
    assert labels["intent_label"] == "اقتراح"
    assert labels["topic_label"] == "إضافة كلمة"
    assert labels["sentiment_label"] == "محايدة"


def test_accept_ai_writes_gold_without_auto_on_load(mini_b3):
    csv_path, review_path, _ = mini_b3
    state = empty_b3_review_state(csv_path)
    state, error = accept_ai_suggestion_row(state, review_path=review_path)
    assert error is None
    row = state["rows"][0]
    assert row["gold_primary_intent"] == "SUGGESTION"
    assert row["gold_annotation_status"] == "LABELED"
    assert row["labeled_by"] == B3_LABELER_ID
    assert row["labeled_at"].count("-") == 2


def test_manual_save_insufficient_context(mini_b3):
    csv_path, review_path, _ = mini_b3
    state = empty_b3_review_state(csv_path)
    state, error = save_current_b3_row(
        state,
        intent_label="النص غير كافٍ للحكم",
        review_path=review_path,
    )
    assert error is None
    row = state["rows"][0]
    assert row["gold_annotation_status"] == "INSUFFICIENT_CONTEXT"
    assert row["gold_primary_intent"] == ""


def test_finalize_imports_only_wizard_rows(mini_b3):
    csv_path, review_path, marker_path = mini_b3
    state = empty_b3_review_state(csv_path)
    save_and_advance_b3_row(
        state,
        review_path=review_path,
        intent_label="اقتراح",
        topic_label="إضافة كلمة",
        sentiment_label="محايدة",
    )
    state = load_or_init_b3_review_state(review_path, csv_path)
    state, _ = save_current_b3_row(
        state,
        intent_label="النص غير كافٍ للحكم",
        review_path=review_path,
    )
    assert is_review_complete(state)

    out_csv = csv_path.parent / "merged.csv"
    _, marker, summary = finalize_b3_review(
        state,
        output_csv_path=out_csv,
        gold_csv_path=csv_path,
        review_path=review_path,
        marker_path=marker_path,
    )
    merged = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
    verify_post_b2_b3_invariants(merged, require_b3_complete=True)

    done = merged[merged.comment_id == "done1"].iloc[0]
    assert done.gold_primary_intent == "THANKS_PRAISE"
    assert done.labeled_by == "b2_calibration_wizard"

    p1 = merged[merged.comment_id == "pending1"].iloc[0]
    assert p1.gold_primary_intent == "SUGGESTION"
    assert p1.labeled_by == B3_LABELER_ID

    protected = merged[merged.comment_id == "protected_val"].iloc[0]
    assert protected.annotation_stage == "human_val"
    assert protected.gold_primary_intent == "INQUIRY"

    assert marker.exists()
    assert "COMPLETE" in summary


def test_import_refuses_incomplete_review(mini_b3):
    csv_path, review_path, _ = mini_b3
    state = empty_b3_review_state(csv_path)
    review_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(AssertionError, match="incomplete"):
        import_b3_wizard_labels(csv_path.parent / "out.csv", csv_path, review_path)


def test_empty_state_does_not_copy_ai_into_gold(mini_b3):
    csv_path, _, _ = mini_b3
    state = empty_b3_review_state(csv_path)
    row = state["rows"][0]
    assert row["ai_suggested_primary_intent"] == "SUGGESTION"
    assert row["gold_primary_intent"] == ""
    assert row["gold_annotation_status"] == ""
    assert row["annotation_stage"] == "ai_assisted_train"


def test_accept_ai_preserves_locked_csv_fields(mini_b3):
    csv_path, review_path, _ = mini_b3
    state = empty_b3_review_state(csv_path)
    before = dict(state["rows"][0])
    state, error = accept_ai_suggestion_row(state, review_path=review_path)
    assert error is None
    after = state["rows"][0]
    for field in (
        "comment_id",
        "comment_text",
        "split",
        "annotation_stage",
        "ai_suggested_primary_intent",
        "ai_suggested_confidence",
        "ai_suggestion_model",
    ):
        assert after[field] == before[field]


def test_canonical_train_gold_complete_after_b3():
    from rcs.evaluation.calibration_labeling import GOLD_CSV
    from rcs.evaluation.b3_labeling import B3_COMPLETE_MARKER, load_remaining_train_rows

    if not GOLD_CSV.exists():
        pytest.skip("canonical gold CSV not present")
    if not B3_COMPLETE_MARKER.exists():
        pytest.skip("B.3 completion marker not present")

    rows = load_remaining_train_rows(GOLD_CSV)
    gold = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)
    train = gold[gold.annotation_stage == "ai_assisted_train"]
    protected = gold[gold.annotation_stage.isin(["human_val", "human_test", "human_train_seed"])]

    assert len(rows) == 0
    assert len(train) == 300
    assert train.gold_annotation_status.isin(["LABELED", "INSUFFICIENT_CONTEXT"]).all()
    assert (train.gold_annotation_status.astype(str).str.strip() != "").all()

    marker = json.loads(B3_COMPLETE_MARKER.read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["ai_assisted_train_total"] == 300
    assert marker["labeled_count"] + marker["insufficient_context_count"] == 300

    remaining_ids = {r["comment_id"] for r in rows}
    assert remaining_ids.isdisjoint(set(protected.comment_id))
    assert remaining_ids.isdisjoint(PREEXISTING_AI_ASSISTED_GOLD_IDS)


def test_resume_from_first_incomplete(mini_b3):
    csv_path, review_path, _ = mini_b3
    state = empty_b3_review_state(csv_path)
    save_current_b3_row(
        state,
        intent_label="اقتراح",
        topic_label="إضافة كلمة",
        sentiment_label="محايدة",
        review_path=review_path,
    )
    reloaded = load_or_init_b3_review_state(review_path, csv_path)
    assert reloaded["current_index"] == 1
