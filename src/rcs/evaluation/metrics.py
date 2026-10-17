"""
Evaluation metrics harness — LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase A.

One generic function, usable to score ANY classifier tier's predictions
against gold labels — rule-based, TF-IDF, embedding-similarity, a local
transformer, or Gemini. Per the architecture plan's evaluation discipline
(Section 6): Macro-F1 (not accuracy — this dataset's classes are known to
be imbalanced from the original data-discovery report), per-class
precision/recall/F1/support, and a confusion matrix. No tier's numbers are
comparable to another's unless they're computed by this same function.

Deliberately dependency-light: a pure-Python implementation (stdlib only),
not scikit-learn. This keeps the evaluation harness installable and
testable with zero extra packages — scikit-learn is deferred to Tier 1
(TF-IDF/Logistic-Regression/Linear-SVM), where it's actually needed for
the classifier itself, per the architecture plan's incremental-dependency
principle. The `evaluate()` interface below matches what a scikit-learn-
backed implementation would return, so swapping the internals later (if
ever needed) would not require changing any caller.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class EvaluationReport:
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion_matrix: List[List[int]]  # confusion_matrix[i][j] = count of true=labels[i], predicted=labels[j]
    labels: List[str]


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> EvaluationReport:
    """
    Pure function over two aligned sequences of labels (same length,
    position i in y_true is the gold label for the same item as position i
    in y_pred). No side effects, no file I/O, no randomness.

    `labels` fixes which classes appear in the report and the row/column
    order of the confusion matrix. If omitted, inferred as the sorted union
    of everything seen in y_true and y_pred. Passing it explicitly is
    recommended once a fixed taxonomy exists (e.g. settings.PRIMARY_INTENTS)
    so a class with zero occurrences in a given split still appears in the
    report with support=0, rather than silently vanishing.

    Raises ValueError if y_true/y_pred are misaligned in length, if either
    is empty, or if a value outside `labels` appears — the last case is
    treated as a caller error (mismatched taxonomy) rather than silently
    ignored, since silently dropping label mismatches would hide exactly
    the kind of bug this harness exists to catch.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be the same length (got {len(y_true)} and {len(y_pred)})")
    if not y_true:
        raise ValueError("cannot evaluate on an empty set")

    if labels is None:
        resolved_labels = sorted(set(y_true) | set(y_pred))
    else:
        resolved_labels = list(labels)
    label_index = {label: i for i, label in enumerate(resolved_labels)}

    n = len(resolved_labels)
    confusion = [[0] * n for _ in range(n)]
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in label_index or pred_label not in label_index:
            raise ValueError(
                f"label outside the provided `labels`: true={true_label!r} pred={pred_label!r}, "
                f"labels={resolved_labels!r}"
            )
        confusion[label_index[true_label]][label_index[pred_label]] += 1

    true_support = Counter(y_true)
    predicted_support = Counter(y_pred)

    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    f1: Dict[str, float] = {}
    support: Dict[str, int] = {}

    for label in resolved_labels:
        i = label_index[label]
        true_positive = confusion[i][i]
        false_positive = predicted_support[label] - true_positive
        false_negative = true_support[label] - true_positive

        p = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        r = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precision[label] = p
        recall[label] = r
        f1[label] = f
        support[label] = true_support[label]

    macro_f1 = sum(f1.values()) / len(resolved_labels) if resolved_labels else 0.0

    return EvaluationReport(
        macro_f1=macro_f1,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        per_class_support=support,
        confusion_matrix=confusion,
        labels=resolved_labels,
    )
