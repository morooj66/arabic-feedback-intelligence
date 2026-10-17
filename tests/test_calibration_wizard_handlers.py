"""Gradio handler output-contract tests for the calibration wizard."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rcs.evaluation.calibration_labeling import (
    CALIBRATION_MANIFEST,
    CALIBRATION_XLSX,
    GOLD_CSV,
    REVIEW_JSON,
    apply_human_answers,
    count_completed_rows,
    current_comment_id,
    empty_review_state,
    first_incomplete_index,
    go_previous_row,
    load_or_init_review_state,
    row_to_form_labels,
    save_and_advance_row,
)
from scripts.run_calibration_wizard import (
    HANDLER_OUTPUT_COUNT,
    RENDER_OUTPUT_COUNT,
    SAVE_OK_MESSAGE,
    _pack_response,
    _render_row,
)


def _is_gradio_update(value: object) -> bool:
    return isinstance(value, dict) and value.get("__type__") == "update"


def assert_handler_output_contract(packed: tuple) -> None:
    assert len(packed) == HANDLER_OUTPUT_COUNT
    progress, comment, intent_u, topic_u, sentiment_u, subcat_u, secondary_u, status, state, show_secondary = packed
    assert isinstance(progress, str)
    assert isinstance(comment, str)
    assert _is_gradio_update(intent_u)
    assert _is_gradio_update(topic_u)
    assert _is_gradio_update(sentiment_u)
    assert _is_gradio_update(subcat_u)
    assert _is_gradio_update(secondary_u)
    assert isinstance(status, str)
    assert isinstance(state, dict)
    assert isinstance(show_secondary, bool)
    assert "rows" in state
    assert "current_index" in state
    for value in (progress, comment, status):
        assert "خطأ" not in value


def test_load_first_incomplete_row_after_existing_progress():
    if not REVIEW_JSON.exists():
        pytest.skip("no persisted human review artifact")
    before = REVIEW_JSON.read_text(encoding="utf-8")
    before_row0 = json.loads(before)["rows"][0]
    state = load_or_init_review_state()
    idx = first_incomplete_index(state["rows"])
    assert state["current_index"] == idx
    assert count_completed_rows(state) >= 1
    assert state["rows"][0]["gold_annotation_status"] == "INSUFFICIENT_CONTEXT"

    packed = _pack_response(state, False)
    assert_handler_output_contract(packed)
    assert isinstance(packed[1], str)
    assert packed[1] == state["rows"][idx]["comment_text"]
    assert len(_render_row(state, False)) == RENDER_OUTPUT_COUNT

    after = REVIEW_JSON.read_text(encoding="utf-8")
    assert after == before
    assert json.loads(after)["rows"][0] == before_row0


def test_save_and_previous_handler_contract_with_isolated_review(tmp_path: Path):
    review_copy = tmp_path / "review.json"
    state = empty_review_state(CALIBRATION_XLSX, CALIBRATION_MANIFEST, GOLD_CSV)
    state["rows"][0] = apply_human_answers(state["rows"][0], intent_label="النص غير كافٍ للحكم")
    state["current_index"] = 1

    row2_id_before = current_comment_id(state)
    state, error = save_and_advance_row(
        state,
        review_path=review_copy,
        intent_label="اقتراح",
        topic_label="ترشيح كلمة العام",
        sentiment_label="إيجابية",
    )
    assert error is None
    save_packed = _pack_response(state, False, SAVE_OK_MESSAGE)
    assert_handler_output_contract(save_packed)
    assert state["current_index"] == 2
    assert current_comment_id(state) != row2_id_before

    state = go_previous_row(state, review_copy)
    prev_packed = _pack_response(state, False)
    assert_handler_output_contract(prev_packed)
    assert state["current_index"] == 1
    labels = row_to_form_labels(state["rows"][1])
    assert prev_packed[2]["value"] == labels["intent_label"]
    assert prev_packed[3]["value"] == labels["topic_label"]
    assert prev_packed[4]["value"] == labels["sentiment_label"]


def test_no_handler_output_contains_error_sentinel(tmp_path: Path):
    state = empty_review_state(CALIBRATION_XLSX, CALIBRATION_MANIFEST, GOLD_CSV)
    review_copy = tmp_path / "review.json"

    assert_handler_output_contract(_pack_response(state, False))

    state_after, error = save_and_advance_row(
        copy.deepcopy(state),
        review_path=review_copy,
        intent_label="اقتراح",
        topic_label="",
        sentiment_label="",
    )
    incomplete = _pack_response(state_after, False, error or "")
    assert_handler_output_contract(incomplete)
    assert "اختر موضوع التعليق." in incomplete[7]

    sufficient_state, error = save_and_advance_row(
        copy.deepcopy(state),
        review_path=review_copy,
        intent_label="النص غير كافٍ للحكم",
    )
    assert error is None
    assert_handler_output_contract(_pack_response(sufficient_state, False, SAVE_OK_MESSAGE))
