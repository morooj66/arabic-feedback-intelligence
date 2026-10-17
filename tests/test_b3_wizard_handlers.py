"""Gradio handler output-contract tests for the B.3 review wizard."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from rcs.evaluation.b3_labeling import (
    accept_ai_suggestion_row,
    empty_b3_review_state,
    save_and_advance_b3_row,
)
from rcs.evaluation.calibration_labeling import current_comment_id
from scripts.run_b3_wizard import (
    HANDLER_OUTPUT_COUNT,
    RENDER_OUTPUT_COUNT,
    SAVE_OK_MESSAGE,
    _ai_suggestion_card_html,
    _pack_response,
    _render_row,
    on_prev,
    on_save_next,
    on_show_manual_form,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_REVIEW_JSON = PROJECT_ROOT / "data" / "gold" / "b3_human_review.json"


def _write_mini_csv(path: Path) -> None:
    rows = [
        {
            "comment_id": "p1",
            "comment_text": "test",
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
            "ai_suggested_topic_subtype": "GENERAL",
            "ai_suggested_secondary_intent": "",
            "ai_suggested_subcategory": "",
            "ai_suggested_sentiment": "NEUTRAL",
            "ai_suggested_confidence": "0.7",
            "ai_suggestion_model": "gemini-3.6-flash:b2-v3",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _is_gradio_update(value: object) -> bool:
    return isinstance(value, dict) and value.get("__type__") == "update"


def assert_handler_output_contract(packed: tuple) -> None:
    assert len(packed) == HANDLER_OUTPUT_COUNT
    (
        progress,
        comment,
        ai_panel,
        manual_form_u,
        intent_u,
        topic_u,
        sentiment_u,
        subcat_u,
        secondary_u,
        status,
        state,
        show_secondary,
        show_manual_form,
    ) = packed
    assert isinstance(progress, str)
    assert "progress_panel" in progress or "التعليق" in progress
    assert isinstance(comment, str)
    assert isinstance(ai_panel, str)
    assert "ai_card" in ai_panel
    assert "الغرض" in ai_panel
    assert _is_gradio_update(manual_form_u)
    assert _is_gradio_update(intent_u)
    assert _is_gradio_update(topic_u)
    assert _is_gradio_update(sentiment_u)
    assert _is_gradio_update(subcat_u)
    assert _is_gradio_update(secondary_u)
    assert isinstance(status, str)
    assert isinstance(state, dict)
    assert isinstance(show_secondary, bool)
    assert isinstance(show_manual_form, bool)


def test_b3_handler_output_contract(tmp_path: Path):
    csv_path = tmp_path / "gold.csv"
    review_path = tmp_path / "review.json"
    _write_mini_csv(csv_path)
    state = empty_b3_review_state(csv_path)
    row = state["rows"][0]
    card = _ai_suggestion_card_html(row)
    assert 'id="ai_card"' in card
    assert "background:#0f172a" in card
    assert "0.7" in card

    assert len(_render_row(state, False, False)) == RENDER_OUTPUT_COUNT
    collapsed = _render_row(state, False, False)
    assert collapsed[3]["visible"] is False
    assert collapsed[4]["visible"] is False
    assert_handler_output_contract(_pack_response(state, False, False))

    manual_render = _render_row(state, False, True)
    assert manual_render[3]["visible"] is True

    state_after, error = save_and_advance_b3_row(
        copy.deepcopy(state),
        review_path=review_path,
        intent_label="النص غير كافٍ للحكم",
    )
    assert error is None
    assert_handler_output_contract(_pack_response(state_after, False, False, SAVE_OK_MESSAGE))

    incomplete, error = save_and_advance_b3_row(
        copy.deepcopy(state),
        review_path=review_path,
        intent_label="اقتراح",
        topic_label="",
        sentiment_label="",
    )
    assert error
    assert_handler_output_contract(_pack_response(incomplete, False, False, error))


def _write_two_row_csv(path: Path) -> None:
    rows = [
        {
            "comment_id": "p1",
            "comment_text": "first",
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
            "ai_suggested_topic_subtype": "GENERAL",
            "ai_suggested_secondary_intent": "",
            "ai_suggested_subcategory": "",
            "ai_suggested_sentiment": "NEUTRAL",
            "ai_suggested_confidence": "0.7",
            "ai_suggestion_model": "gemini-3.6-flash:b2-v3",
        },
        {
            "comment_id": "p2",
            "comment_text": "second",
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
            "ai_suggested_primary_intent": "INQUIRY",
            "ai_suggested_topic_subtype": "GENERAL",
            "ai_suggested_secondary_intent": "",
            "ai_suggested_subcategory": "",
            "ai_suggested_sentiment": "NEUTRAL",
            "ai_suggested_confidence": "0.8",
            "ai_suggestion_model": "gemini-3.6-flash:b2-v3",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_edit_button_reveals_manual_form_and_roundtrip(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "gold.csv"
    review_path = tmp_path / "review.json"
    _write_two_row_csv(csv_path)
    state = empty_b3_review_state(csv_path)
    start_id = current_comment_id(state)

    def _save_advance(state_in, **kwargs):
        return save_and_advance_b3_row(state_in, review_path=review_path, **kwargs)

    def _go_prev(state_in):
        from rcs.evaluation.b3_labeling import go_previous_b3_row

        return go_previous_b3_row(state_in, review_path=review_path)

    monkeypatch.setattr("scripts.run_b3_wizard.save_and_advance_b3_row", _save_advance)
    monkeypatch.setattr("scripts.run_b3_wizard.go_previous_b3_row", _go_prev)

    hidden = _pack_response(state, False, False)
    assert hidden[12] is False
    assert hidden[3]["visible"] is False
    assert hidden[4]["visible"] is False

    opened = on_show_manual_form(state, False, False)
    assert opened[12] is True
    assert opened[3]["visible"] is True
    assert opened[4]["visible"] is True
    assert current_comment_id(opened[10]) == start_id

    opened_again = on_show_manual_form(opened[10], opened[11], opened[12])
    assert opened_again[12] is True
    assert opened_again[3]["visible"] is True

    saved_next = on_save_next(
        opened[10],
        opened[11],
        opened[12],
        "اقتراح",
        "موضوع عام",
        "محايدة",
        "",
        "",
    )
    assert saved_next[9] == SAVE_OK_MESSAGE
    assert saved_next[10]["current_index"] == 1
    assert saved_next[12] is False
    assert saved_next[3]["visible"] is False

    restored = on_prev(saved_next[10], saved_next[11], saved_next[12])
    assert restored[10]["current_index"] == 0
    assert current_comment_id(restored[10]) == start_id
    assert restored[10]["rows"][0]["gold_primary_intent"] == "SUGGESTION"
    assert restored[10]["rows"][0]["gold_topic_subtype"] == "GENERAL"
    assert restored[10]["rows"][0]["gold_sentiment"] == "NEUTRAL"

    reopened = on_show_manual_form(restored[10], restored[11], restored[12])
    assert reopened[4]["visible"] is True
    assert reopened[4]["value"] == "اقتراح"
    assert reopened[5]["visible"] is True
    assert reopened[5]["value"] == "موضوع عام"
    assert reopened[6]["visible"] is True
    assert reopened[6]["value"] == "محايدة"


def test_save_next_advances_comment_30_after_accept_ai_hidden_form(
    tmp_path: Path,
    monkeypatch,
):
    if not LIVE_REVIEW_JSON.exists():
        pytest.skip("live B.3 review artifact not present")

    live_before = LIVE_REVIEW_JSON.read_bytes()
    state = json.loads(LIVE_REVIEW_JSON.read_text(encoding="utf-8"))
    completed = sum(
        1
        for row in state["rows"]
        if row.get("gold_annotation_status") in ("LABELED", "INSUFFICIENT_CONTEXT")
    )
    if completed != 29 or state.get("current_index") != 29:
        pytest.skip("progress state no longer at comment 30 with 29 completed")

    snapshot_29 = copy.deepcopy(state["rows"][:29])
    comment_30_id = state["rows"][29]["comment_id"]
    tmp_review = tmp_path / "review.json"

    def _save_advance(state_in, **kwargs):
        return save_and_advance_b3_row(state_in, review_path=tmp_review, **kwargs)

    def _go_prev(state_in):
        from rcs.evaluation.b3_labeling import go_previous_b3_row

        return go_previous_b3_row(state_in, review_path=tmp_review)

    monkeypatch.setattr("scripts.run_b3_wizard.save_and_advance_b3_row", _save_advance)
    monkeypatch.setattr("scripts.run_b3_wizard.go_previous_b3_row", _go_prev)

    work = copy.deepcopy(state)
    work, error = accept_ai_suggestion_row(work, review_path=tmp_review)
    assert error is None
    assert work["rows"][29]["gold_annotation_status"] == "LABELED"

    stuck_form = ("", "", "", "", "")
    advanced = on_save_next(work, False, False, *stuck_form)
    assert advanced[9] == SAVE_OK_MESSAGE
    assert advanced[10]["current_index"] == 30
    assert current_comment_id(advanced[10]) != comment_30_id

    restored = on_prev(advanced[10], False, False)
    assert restored[10]["current_index"] == 29
    assert restored[10]["rows"][29]["comment_id"] == comment_30_id
    assert restored[10]["rows"][29]["gold_primary_intent"] == "SUGGESTION"
    assert restored[10]["rows"][29]["gold_topic_subtype"] == "WORD_SUBMISSION"
    assert restored[10]["rows"][29]["gold_sentiment"] == "NEUTRAL"
    assert restored[10]["rows"][:29] == snapshot_29
    assert LIVE_REVIEW_JSON.read_bytes() == live_before
