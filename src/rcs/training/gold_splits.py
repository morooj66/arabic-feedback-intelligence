"""
Load frozen gold splits for supervised training and validation.

Rules (post B.3):
  TRAIN  — human_train_seed + ai_assisted_train rows with gold_annotation_status=LABELED
  VAL    — human_val rows only (never used for training)
  TEST   — human_test rows (loaded only when explicitly requested; never for tuning)

INSUFFICIENT_CONTEXT rows are excluded from all supervised train/val/test loads.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from config import settings
from rcs.evaluation.gold_dataset import (
    verify_no_split_leakage,
    verify_post_b2_b3_invariants,
    _canonical_intent,
)

TRAINING_GOLD_CSV = settings.DATA_DIR / "gold" / "gold_dataset_final_training.csv"

TaskName = Literal["intent", "topic", "sentiment"]

TASK_LABEL_COLUMN = {
    "intent": "gold_primary_intent",
    "topic": "gold_topic_subtype",
    "sentiment": "gold_sentiment",
}

TASK_LABELS = {
    "intent": list(settings.PRIMARY_INTENTS),
    "topic": list(settings.TOPIC_SUBTYPES),
    "sentiment": list(settings.SENTIMENTS),
}

TRAIN_STAGES = frozenset({"human_train_seed", "ai_assisted_train"})
VAL_STAGE = "human_val"
TEST_STAGE = "human_test"
SUPERVISED_STATUS = "LABELED"


@dataclass(frozen=True)
class LabeledExample:
    comment_id: str
    text: str
    label: str
    annotation_stage: str
    split: str


@dataclass
class TaskSplitBundle:
    task: TaskName
    train: list[LabeledExample] = field(default_factory=list)
    val: list[LabeledExample] = field(default_factory=list)
    test: list[LabeledExample] = field(default_factory=list)

    def train_distribution(self) -> dict[str, int]:
        return dict(Counter(ex.label for ex in self.train))

    def val_distribution(self) -> dict[str, int]:
        return dict(Counter(ex.label for ex in self.val))


@dataclass
class SupervisedGoldData:
    """All three tasks over the same frozen row sets."""

    train_rows: int
    val_rows: int
    test_rows: int
    train_excluded_insufficient: int
    val_excluded_insufficient: int
    val_excluded_unlabeled: int
    val_excluded_details: list[dict[str, str]]
    intent: TaskSplitBundle
    topic: TaskSplitBundle
    sentiment: TaskSplitBundle
    imbalance_flags: dict[str, list[str]] = field(default_factory=dict)


def describe_excluded_val_rows(gold_csv_path: Path | None = None) -> list[dict[str, str]]:
    """Document human_val rows excluded from supervised validation (never modifies labels)."""
    path = Path(gold_csv_path or TRAINING_GOLD_CSV)
    gold = pd.read_csv(path, dtype=str, keep_default_na=False)
    val_pool = gold[gold["annotation_stage"] == VAL_STAGE]
    excluded = val_pool[val_pool["gold_annotation_status"] != SUPERVISED_STATUS]
    details: list[dict[str, str]] = []
    for _, row in excluded.iterrows():
        status = str(row["gold_annotation_status"]).strip()
        missing: list[str] = []
        if status != SUPERVISED_STATUS:
            missing.append(
                "gold_annotation_status"
                if not status
                else f"gold_annotation_status={status!r} (requires LABELED)"
            )
        for col in (
            "gold_primary_intent",
            "gold_topic_subtype",
            "gold_sentiment",
            "labeled_by",
            "labeled_at",
        ):
            if not str(row[col]).strip():
                missing.append(col)
        details.append(
            {
                "comment_id": str(row["comment_id"]),
                "gold_annotation_status": status or "(empty)",
                "gold_primary_intent": str(row["gold_primary_intent"]).strip(),
                "gold_topic_subtype": str(row["gold_topic_subtype"]).strip(),
                "gold_sentiment": str(row["gold_sentiment"]).strip(),
                "exclusion_reason": "; ".join(missing),
            }
        )
    return details


def _load_gold_frame(gold_csv_path: Path | None = None) -> pd.DataFrame:
    gold_csv_path = Path(gold_csv_path or TRAINING_GOLD_CSV)
    if not gold_csv_path.exists():
        raise FileNotFoundError(f"canonical gold CSV not found: {gold_csv_path}")
    gold = pd.read_csv(gold_csv_path, dtype=str, keep_default_na=False)
    verify_no_split_leakage(gold)
    verify_post_b2_b3_invariants(gold, require_b3_complete=True)
    return gold


def _examples_from_frame(
    frame: pd.DataFrame,
    task: TaskName,
) -> list[LabeledExample]:
    label_col = TASK_LABEL_COLUMN[task]
    examples: list[LabeledExample] = []
    for _, row in frame.iterrows():
        label = str(row[label_col]).strip()
        if task == "intent":
            label = _canonical_intent(label)
        if not label:
            continue
        examples.append(
            LabeledExample(
                comment_id=str(row["comment_id"]),
                text=str(row["comment_text"]),
                label=label,
                annotation_stage=str(row["annotation_stage"]),
                split=str(row["split"]),
            )
        )
    return examples


def _imbalance_flags(distribution: dict[str, int], *, min_count: int = 5) -> list[str]:
    flags: list[str] = []
    if not distribution:
        return ["empty split"]
    total = sum(distribution.values())
    for label, count in sorted(distribution.items()):
        if count < min_count:
            flags.append(f"{label}: {count} (<{min_count})")
        elif count / total < 0.03:
            flags.append(f"{label}: {count} ({100 * count / total:.1f}% — rare)")
    return flags


def load_supervised_gold(
    gold_csv_path: Path | None = None,
    *,
    include_test: bool = False,
) -> SupervisedGoldData:
    """
    Build train/val/(optional test) bundles for intent, topic, and sentiment.

    Raises if train or val supervised sets are empty.
    """
    gold = _load_gold_frame(gold_csv_path)

    train_pool = gold[
        gold["annotation_stage"].isin(TRAIN_STAGES)
        & (gold["split"] == "train")
    ]
    val_pool = gold[gold["annotation_stage"] == VAL_STAGE]
    test_pool = gold[gold["annotation_stage"] == TEST_STAGE] if include_test else gold.iloc[0:0]

    train_labeled = train_pool[train_pool["gold_annotation_status"] == SUPERVISED_STATUS]
    val_labeled = val_pool[val_pool["gold_annotation_status"] == SUPERVISED_STATUS]
    test_labeled = test_pool[test_pool["gold_annotation_status"] == SUPERVISED_STATUS]

    train_excluded = int(
        (train_pool["gold_annotation_status"] == "INSUFFICIENT_CONTEXT").sum()
    )
    val_excluded = int(
        (val_pool["gold_annotation_status"] == "INSUFFICIENT_CONTEXT").sum()
    )
    val_excluded_unlabeled = int(
        (val_pool["gold_annotation_status"].astype(str).str.strip() == "").sum()
    )
    val_excluded_details = describe_excluded_val_rows(gold_csv_path)

    if train_labeled.empty:
        raise AssertionError("no LABELED train rows available for supervised training")
    if val_labeled.empty:
        raise AssertionError("no LABELED human_val rows available for validation")

    bundles: dict[TaskName, TaskSplitBundle] = {}
    imbalance: dict[str, list[str]] = {}
    for task in ("intent", "topic", "sentiment"):
        bundle = TaskSplitBundle(task=task)
        bundle.train = _examples_from_frame(train_labeled, task)
        bundle.val = _examples_from_frame(val_labeled, task)
        if include_test:
            bundle.test = _examples_from_frame(test_labeled, task)
        bundles[task] = bundle
        imbalance[f"{task}_train"] = _imbalance_flags(bundle.train_distribution())
        imbalance[f"{task}_val"] = _imbalance_flags(bundle.val_distribution())

    return SupervisedGoldData(
        train_rows=len(train_labeled),
        val_rows=len(val_labeled),
        test_rows=len(test_labeled) if include_test else 0,
        train_excluded_insufficient=train_excluded,
        val_excluded_insufficient=val_excluded,
        val_excluded_unlabeled=val_excluded_unlabeled,
        val_excluded_details=val_excluded_details,
        intent=bundles["intent"],
        topic=bundles["topic"],
        sentiment=bundles["sentiment"],
        imbalance_flags=imbalance,
    )
