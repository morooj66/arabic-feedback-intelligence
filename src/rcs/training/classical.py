"""TF-IDF + linear classifiers as classical benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rcs.evaluation.metrics import EvaluationReport, evaluate
from rcs.training.gold_splits import LabeledExample, TaskName, TASK_LABELS

ClassifierKind = Literal["svm", "logistic"]


@dataclass
class ClassicalResult:
    classifier: ClassifierKind
    task: TaskName
    train_size: int
    val_size: int
    report: EvaluationReport
    y_true: list[str]
    y_pred: list[str]
    comment_ids: list[str]


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the classical baseline. "
            "Install with: pip install scikit-learn"
        ) from exc


def train_evaluate_classical(
    train: list[LabeledExample],
    val: list[LabeledExample],
    *,
    task: TaskName,
    classifier: ClassifierKind = "svm",
    random_state: int = 42,
) -> ClassicalResult:
    _require_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    labels = TASK_LABELS[task]
    x_train = [ex.text for ex in train]
    y_train = [ex.label for ex in train]
    x_val = [ex.text for ex in val]
    y_true = [ex.label for ex in val]
    comment_ids = [ex.comment_id for ex in val]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,
    )

    if classifier == "svm":
        clf = LinearSVC(random_state=random_state)
    else:
        clf = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        )

    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_val).tolist()

    report = evaluate(y_true, y_pred, labels=labels)
    return ClassicalResult(
        classifier=classifier,
        task=task,
        train_size=len(train),
        val_size=len(val),
        report=report,
        y_true=y_true,
        y_pred=y_pred,
        comment_ids=comment_ids,
    )
