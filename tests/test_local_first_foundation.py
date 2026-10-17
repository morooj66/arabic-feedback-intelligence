"""
Phase A tests — LOCAL_FIRST_ARCHITECTURE_PLAN.md foundation.

No real gold data, no model training, no Gemini calls anywhere in this
file — Phase A proves the plumbing (unified schema, new persistence
columns, evaluation math, split-assignment logic) is correct against
synthetic/dummy data, on purpose, before any real labeling starts.

Covers:
  1. ClassificationResult — the unified per-tier contract shape
  2. metrics.evaluate() — Macro-F1 / per-class precision-recall / confusion
     matrix, verified against a hand-computed example, plus input-validation
     error paths
  3. gold_dataset.assign_splits() — reproducibility (same seed -> same
     split), stratification correctness, and input-validation error paths
  4. repository.apply_classification_results() — new
     ai_classification_source/ai_needs_review columns round-trip correctly
     when present, and the function stays backward-compatible with the
     existing Gemini-only classify_row output shape (no reference to the
     new keys) that predates this phase
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.classification.classifier import ClassificationOutcome
from rcs.classification.schema import ClassificationResult
from rcs.evaluation.gold_dataset import DEFAULT_SPLIT_RATIOS, assign_splits
from rcs.evaluation.metrics import evaluate
from rcs.review.repository import (
    apply_classification_results,
    get_engine,
    init_db,
    upsert_review_items,
)


# ---------------------------------------------------------------------------
# 1. ClassificationResult — unified per-tier contract
# ---------------------------------------------------------------------------

def test_classification_result_holds_the_four_contract_fields():
    result = ClassificationResult(
        category="THANKS_PRAISE", confidence=0.92,
        classification_source="RULE_BASED", needs_review=False,
    )
    assert result.category == "THANKS_PRAISE"
    assert result.confidence == 0.92
    assert result.classification_source == "RULE_BASED"
    assert result.needs_review is False


def test_classification_result_allows_none_category_for_a_declining_tier():
    # A tier that isn't confident enough to answer at all still returns a
    # well-formed result — category=None, needs_review=True — rather than
    # raising or being omitted.
    result = ClassificationResult(
        category=None, confidence=0.0,
        classification_source="RULE_BASED", needs_review=True,
    )
    assert result.category is None
    assert result.needs_review is True


def test_classification_result_is_frozen():
    result = ClassificationResult(
        category="SPAM", confidence=0.99,
        classification_source="TFIDF_SVM", needs_review=False,
    )
    with pytest.raises(Exception):
        result.category = "SUGGESTION"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. metrics.evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_matches_hand_computed_macro_f1_and_confusion_matrix():
    # true:      A A B B B C
    # predicted: A B B B A C
    y_true = ["A", "A", "B", "B", "B", "C"]
    y_pred = ["A", "B", "B", "B", "A", "C"]

    report = evaluate(y_true, y_pred, labels=["A", "B", "C"])

    # confusion_matrix[i][j] = count of true=labels[i], predicted=labels[j]
    assert report.confusion_matrix == [
        [1, 1, 0],  # true A: 1 predicted A, 1 predicted B
        [1, 2, 0],  # true B: 1 predicted A, 2 predicted B
        [0, 0, 1],  # true C: 1 predicted C
    ]

    assert report.per_class_precision["A"] == pytest.approx(0.5)
    assert report.per_class_recall["A"] == pytest.approx(0.5)
    assert report.per_class_f1["A"] == pytest.approx(0.5)

    assert report.per_class_precision["B"] == pytest.approx(2 / 3)
    assert report.per_class_recall["B"] == pytest.approx(2 / 3)
    assert report.per_class_f1["B"] == pytest.approx(2 / 3)

    assert report.per_class_precision["C"] == pytest.approx(1.0)
    assert report.per_class_recall["C"] == pytest.approx(1.0)
    assert report.per_class_f1["C"] == pytest.approx(1.0)

    assert report.per_class_support == {"A": 2, "B": 3, "C": 1}
    # macro-F1 = unweighted mean of per-class F1
    assert report.macro_f1 == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)


def test_evaluate_perfect_predictions_yield_macro_f1_of_one():
    y = ["SPAM", "THANKS_PRAISE", "COMPLAINT", "SPAM"]
    report = evaluate(y, y)
    assert report.macro_f1 == pytest.approx(1.0)
    for label in report.labels:
        assert report.per_class_precision[label] == pytest.approx(1.0)
        assert report.per_class_recall[label] == pytest.approx(1.0)


def test_evaluate_zero_division_handled_as_zero_not_a_crash():
    # A class that never appears in y_true (only in y_pred) has undefined
    # recall in the usual formula (0/0) — must resolve to 0.0, not raise.
    report = evaluate(["A", "A"], ["A", "B"], labels=["A", "B"])
    assert report.per_class_recall["B"] == 0.0  # B never a true label
    assert report.per_class_precision["B"] == 0.0  # 1 predicted B, 0 correct


def test_evaluate_infers_labels_when_not_provided():
    report = evaluate(["X", "Y"], ["X", "X"])
    assert report.labels == ["X", "Y"]


def test_evaluate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate(["A", "B"], ["A"])


def test_evaluate_rejects_empty_input():
    with pytest.raises(ValueError):
        evaluate([], [])


def test_evaluate_rejects_label_outside_fixed_set():
    with pytest.raises(ValueError):
        evaluate(["A"], ["B"], labels=["A"])  # "B" not in the fixed label set


# ---------------------------------------------------------------------------
# 3. gold_dataset.assign_splits()
# ---------------------------------------------------------------------------

def test_assign_splits_reproducible_with_same_seed():
    ids = [f"c{i}" for i in range(200)]
    labels = ["SUGGESTION", "COMPLAINT", "SPAM"] * 66 + ["SUGGESTION", "COMPLAINT"]

    first = assign_splits(ids, labels=labels, seed=42)
    second = assign_splits(ids, labels=labels, seed=42)

    assert first["split"].tolist() == second["split"].tolist()


def test_assign_splits_differs_with_different_seed():
    ids = [f"c{i}" for i in range(200)]
    labels = ["SUGGESTION", "COMPLAINT", "SPAM"] * 66 + ["SUGGESTION", "COMPLAINT"]

    a = assign_splits(ids, labels=labels, seed=1)
    b = assign_splits(ids, labels=labels, seed=2)

    assert a["split"].tolist() != b["split"].tolist()


def test_assign_splits_is_stratified_per_label():
    # A deliberately imbalanced label distribution — 180 of one class,
    # 20 of a rare one — mirroring the kind of imbalance flagged in the
    # original data-discovery report.
    ids = [f"c{i}" for i in range(200)]
    labels = ["COMMON"] * 180 + ["RARE"] * 20

    result = assign_splits(ids, labels=labels, seed=7, ratios={"train": 0.6, "val": 0.2, "test": 0.2})

    for label, count in [("COMMON", 180), ("RARE", 20)]:
        subset = result[result["label"] == label]
        assert len(subset) == count
        split_counts = subset["split"].value_counts().to_dict()
        # every split must contain at least one example of even the rare class
        assert split_counts.get("train", 0) > 0
        assert split_counts.get("val", 0) > 0
        assert split_counts.get("test", 0) > 0
        # proportions roughly match the requested ratios (rounding tolerance)
        assert split_counts.get("train", 0) == pytest.approx(count * 0.6, abs=2)
        assert split_counts.get("val", 0) == pytest.approx(count * 0.2, abs=2)
        assert split_counts.get("test", 0) == pytest.approx(count * 0.2, abs=2)


def test_assign_splits_covers_every_id_exactly_once():
    ids = [f"c{i}" for i in range(97)]  # deliberately not evenly divisible
    labels = ["A", "B", "C"] * 32 + ["A"]

    result = assign_splits(ids, labels=labels, seed=3)

    assert set(result["comment_id"]) == set(ids)
    assert len(result) == len(ids)
    assert set(result["split"]) <= set(DEFAULT_SPLIT_RATIOS.keys())


def test_assign_splits_unstratified_when_no_labels_given():
    ids = [f"c{i}" for i in range(50)]
    result = assign_splits(ids, seed=1)
    assert set(result["comment_id"]) == set(ids)
    assert len(result) == 50


def test_assign_splits_rejects_ratios_not_summing_to_one():
    with pytest.raises(ValueError):
        assign_splits(["a", "b"], ratios={"train": 0.5, "val": 0.4})


def test_assign_splits_rejects_mismatched_labels_length():
    with pytest.raises(ValueError):
        assign_splits(["a", "b", "c"], labels=["X", "Y"])


def test_assign_splits_rejects_duplicate_comment_ids():
    with pytest.raises(ValueError):
        assign_splits(["a", "a", "b"])


# ---------------------------------------------------------------------------
# 4. repository.apply_classification_results — new columns
# ---------------------------------------------------------------------------

def _seed_one_row(tmp_path):
    db_path = tmp_path / "phase_a_test.sqlite"
    engine = get_engine(db_path)
    init_db(engine)
    now = dt.datetime(2026, 8, 12, 9, 0, 0)
    upsert_review_items(engine, [{
        "comment_id": "row-1",
        "source_file": "f.xlsx", "source_row_index": 0,
        "original_comment": "تجربة", "original_title": None, "original_type": None,
        "entry_id": None, "lemma": None, "visitor_name": None, "visitor_email": None,
        "platform": "الرياض", "comment_created_at": now, "is_exact_duplicate_of": None,
        "ingested_at": now, "comment_age_days": 1.0, "age_group": "recent",
        "review_status": "UNPROCESSED", "ready_to_send": False,
    }])
    return engine


def test_ai_classification_source_and_needs_review_persist_when_present(tmp_path):
    engine = _seed_one_row(tmp_path)
    result = [{
        "comment_id": "row-1",
        **ClassificationOutcome(
            success=True, ai_primary_intent="SUGGESTION", ai_topic_subtype="WORD_SUBMISSION",
            ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="ACKNOWLEDGE_THEN_REVIEW",
            ai_classification_reason="rule matched", ai_confidence=0.8,
        ).__dict__,
        "ai_classification_source": "RULE_BASED",
        "ai_needs_review": False,
    }]
    apply_classification_results(engine, result, "rule-based-v1", None, "n/a", "0.1.0", "run-a")

    from sqlalchemy.orm import Session
    from rcs.review.models import ReviewItem
    with Session(engine) as session:
        item = session.get(ReviewItem, "row-1")
        assert item.ai_classification_source == "RULE_BASED"
        assert item.ai_needs_review is False
        assert item.classification_source == "RULE_BASED"
        assert item.needs_review is False


def test_ai_classification_source_stays_null_when_absent_backward_compat(tmp_path):
    # This is exactly the shape classify_row/ClassificationOutcome produces
    # today (Phase A doesn't touch classifier.py) — no
    # ai_classification_source/ai_needs_review keys at all. Must not raise,
    # and the new columns must simply stay null.
    engine = _seed_one_row(tmp_path)
    result = [{
        "comment_id": "row-1",
        **ClassificationOutcome(
            success=True, ai_primary_intent="THANKS_PRAISE", ai_topic_subtype="GENERAL",
            ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="AUTO_REPLY_CANDIDATE",
            ai_classification_reason="gratitude", ai_confidence=0.9,
        ).__dict__,
    }]
    assert "ai_classification_source" not in result[0]
    assert "ai_needs_review" not in result[0]

    write_result = apply_classification_results(engine, result, "gemini-2.5-flash", None, "v1", "0.1.0", "run-b")
    assert write_result == {"classified": 1, "failed": 0, "skipped_already_classified": 0}

    from sqlalchemy.orm import Session
    from rcs.review.models import ReviewItem
    with Session(engine) as session:
        item = session.get(ReviewItem, "row-1")
        assert item.ai_primary_intent == "THANKS_PRAISE"
        assert item.ai_classification_source is None
        assert item.ai_needs_review is None
