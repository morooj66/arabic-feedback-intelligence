"""Shared data loading, folds and metrics for the classification v2 study. Development data only (gold train + val, LABELED).
Never reads sealed human_test or the 75 blind labels."""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
GOLD = ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
OUT = ROOT / "data" / "evaluation" / "classification_v2"
OUT.mkdir(parents=True, exist_ok=True)

INTENTS = ["THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM"]
TOPICS = ["WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION", "TECHNICAL_ISSUE", "WORD_OF_YEAR_NOMINATION", "OFF_TOPIC_PERSONAL", "GENERAL"]
SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
TASKS = {"intent": ("gold_primary_intent", INTENTS), "topic": ("gold_topic_subtype", TOPICS), "sentiment": ("gold_sentiment", SENTIMENTS)}


def load_dev() -> list[dict]:
    """563 labelled development rows, original gold labels untouched. Sealed `test` rows are dropped on read."""
    with open(GOLD, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] in ("train", "val") and r["gold_annotation_status"] == "LABELED"]
    return rows


def gold_sha() -> str:
    return hashlib.sha256(GOLD.read_bytes()).hexdigest()


def cv_splits(y, *, seeds=(11, 22, 33), k=5):
    """Identical (seed, fold) splits for every model of a task: stratified on the task label, rare classes handled by folding the
    smallest classes with min(k, count) so every row is tested exactly once per repeat."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    counts = Counter(y)
    out = []
    for seed in seeds:
        yy = np.array([c if counts[c] >= k else "__RARE__" for c in y])    # classes with < k members are pooled only for stratification
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(yy)), yy)):
            out.append((seed, fold, tr, te))
    return out


def prf(gold, pred, labels):
    per = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred)); pp = sum(p == c for p in pred); s = sum(g == c for g in gold)
        P = tp / pp if pp else 0.0; R = tp / s if s else None
        F = 2 * P * R / (P + R) if (R and P) else 0.0
        per[c] = dict(support=s, predicted=pp, precision=round(P, 3), recall=None if R is None else round(R, 3), f1=round(F, 3) if s else None)
    f = [v["f1"] for v in per.values() if v["support"]]
    acc = sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold))
    return dict(accuracy=round(acc, 4), macro_f1=round(sum(f) / max(1, len(f)), 4), per_class=per)


def confusion(gold, pred, labels):
    return {g: {p: sum(1 for a, b in zip(gold, pred) if a == g and b == p) for p in labels} for g in labels if any(a == g for a in gold)}


def macro_f1(gold, pred, labels):
    return prf(gold, pred, labels)["macro_f1"]
