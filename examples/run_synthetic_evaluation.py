"""Runs the repository's REAL classical training + evaluation code (TF-IDF + linear classifier, macro-F1, confusion matrix) on the SYNTHETIC labelled data.

    python examples/make_synthetic_data.py && python examples/run_synthetic_evaluation.py

This demonstrates the evaluation METHOD (stratified split, macro-F1, per-class report). The synthetic sentences are generated from a few templates, so the scores are
meaningless as accuracy evidence: they only prove the pipeline runs end to end. The real classifiers were evaluated on restricted data; see docs/ML_EVALUATION.md."""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent)]

from rcs.training.classical import train_evaluate_classical  # noqa: E402
from rcs.training.gold_splits import LabeledExample, TASK_LABELS  # noqa: E402


def main() -> None:
    rows = list(csv.DictReader(open(HERE / "synthetic_labeled_comments.csv", encoding="utf-8")))
    rnd = random.Random(42)
    rnd.shuffle(rows)
    cut = int(len(rows) * 0.8)
    for task in ("intent", "topic", "sentiment"):
        mk = lambda rs, split: [LabeledExample(r["comment_id"], r["text"], r[task], "synthetic", split) for r in rs]
        res = train_evaluate_classical(mk(rows[:cut], "train"), mk(rows[cut:], "val"), task=task, classifier="logistic")
        print(f"{task:9s} train={res.train_size:4d} val={res.val_size:3d} macro-F1={res.report.macro_f1:.3f}  (SYNTHETIC data: demonstrates the method, not accuracy)")


if __name__ == "__main__":
    main()
