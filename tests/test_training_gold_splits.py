"""Tests for training gold-split loading (no ML dependencies required)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import settings
from rcs.evaluation.gold_dataset import verify_post_b2_b3_invariants
from rcs.training.gold_splits import (
    SUPERVISED_STATUS,
    TEST_STAGE,
    TRAIN_STAGES,
    VAL_STAGE,
    load_supervised_gold,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CSV = PROJECT_ROOT / "data" / "gold" / "gold_dataset_final_training.csv"


def _synthetic_gold_csv(path: Path) -> None:
    rows = []
    for i, (stage, split, status, intent, topic, sentiment, *rest) in enumerate(
        [
            ("human_train_seed", "train", "LABELED", "SUGGESTION", "WORD_SUBMISSION", "NEUTRAL"),
            ("ai_assisted_train", "train", "LABELED", "COMPLAINT", "ENTRY_CORRECTION", "NEGATIVE", "MeaningFault"),
            ("ai_assisted_train", "train", "INSUFFICIENT_CONTEXT", "", "", ""),
            ("human_val", "val", "LABELED", "INQUIRY", "GENERAL", "NEUTRAL"),
            ("human_test", "test", "LABELED", "THANKS_PRAISE", "GENERAL", "POSITIVE"),
        ]
    ):
        rows.append(
            {
                "comment_id": f"id{i}",
                "comment_text": f"text {i}",
                "gold_primary_intent": intent,
                "gold_topic_subtype": topic,
                "gold_secondary_intent": "",
                "gold_subcategory": rest[0] if rest else "",
                "gold_sentiment": sentiment,
                "gold_annotation_status": status,
                "labeler_notes": "",
                "labeled_by": "test" if status in ("LABELED", "INSUFFICIENT_CONTEXT") else "",
                "labeled_at": "2026-01-01" if status in ("LABELED", "INSUFFICIENT_CONTEXT") else "",
                "split": split,
                "ai_suggested_primary_intent": "SUGGESTION" if stage == "ai_assisted_train" else "",
                "ai_suggested_topic_subtype": "GENERAL" if stage == "ai_assisted_train" else "",
                "ai_suggested_secondary_intent": "",
                "ai_suggested_subcategory": "",
                "ai_suggested_sentiment": "NEUTRAL" if stage == "ai_assisted_train" else "",
                "ai_suggested_confidence": "0.9" if stage == "ai_assisted_train" else "",
                "ai_suggestion_model": "test" if stage == "ai_assisted_train" else "",
                "annotation_stage": stage,
                "is_few_shot_example": str(stage == "human_train_seed"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_load_supervised_gold_synthetic(tmp_path):
    csv_path = tmp_path / "gold.csv"
    _synthetic_gold_csv(csv_path)
    # Bypass B.3-complete invariant for synthetic mini set.
    gold = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    verify_post_b2_b3_invariants(gold, require_b3_complete=False)

    # Monkeypatch by writing a marker-complete subset manually via direct loader logic
    from rcs.training import gold_splits as gs

    original = gs._load_gold_frame

    def _load(path):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        return frame

    gs._load_gold_frame = _load
    try:
        data = load_supervised_gold(csv_path, include_test=True)
    finally:
        gs._load_gold_frame = original

    assert data.train_rows == 2
    assert data.val_rows == 1
    assert data.test_rows == 1
    assert data.train_excluded_insufficient == 1
    assert len(data.intent.train) == 2
    assert len(data.intent.val) == 1
    assert all(ex.annotation_stage in TRAIN_STAGES for ex in data.intent.train)
    assert all(ex.annotation_stage == VAL_STAGE for ex in data.intent.val)
    assert all(ex.annotation_stage == TEST_STAGE for ex in data.intent.test)


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="canonical gold CSV not present")
def test_canonical_supervised_counts():
    from rcs.training import gold_splits as gs

    data = load_supervised_gold(CANONICAL_CSV, include_test=False)
    assert data.train_rows == 455
    assert data.val_rows == 108
    assert data.train_excluded_insufficient == 37
    assert len(data.intent.train) == data.train_rows
    assert all(ex.label in settings.PRIMARY_INTENTS for ex in data.intent.train)


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="canonical gold CSV not present")
def test_test_split_not_loaded_by_default():
    data = load_supervised_gold(CANONICAL_CSV, include_test=False)
    assert data.test_rows == 0
    assert len(data.intent.test) == 0
