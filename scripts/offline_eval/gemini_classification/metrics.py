"""Pure, dependency-light evaluation functions (no I/O). Every function takes plain label lists so it can be unit-tested against hand-computed values.

Macro-F1 is always reported twice: over ALL taxonomy classes (zero-support classes count as F1 = 0, as in the frozen reports) and over SUPPORTED
classes only (support > 0); zero-support classes are listed separately so neither number hides them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def confusion_matrix(labels: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]) -> list[list[int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            raise ValueError(f"label outside the taxonomy: {t!r} / {p!r}")
        cm[idx[t]][idx[p]] += 1
    return cm


@dataclass(frozen=True)
class ClassStats:
    label: str
    support: int
    predicted: int
    precision: float
    recall: float
    f1: float


def per_class(labels: Sequence[str], cm: list[list[int]]) -> list[ClassStats]:
    out = []
    for i, label in enumerate(labels):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out.append(ClassStats(label, support, predicted, p, r, f))
    return out


def macro_f1(stats: Sequence[ClassStats]) -> dict:
    allc = [s.f1 for s in stats]
    sup = [s.f1 for s in stats if s.support > 0]
    return {
        "macro_f1_all_classes": sum(allc) / len(allc) if allc else 0.0,
        "macro_f1_supported_classes": sum(sup) / len(sup) if sup else 0.0,
        "supported_classes": len(sup),
        "total_classes": len(allc),
    }


def zero_support_report(stats: Sequence[ClassStats]) -> dict:
    return {
        "zero_support_classes": [s.label for s in stats if s.support == 0],
        "predicted_without_support": {s.label: s.predicted for s in stats if s.support == 0 and s.predicted > 0},
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def task_report(labels: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    cm = confusion_matrix(labels, y_true, y_pred)
    stats = per_class(labels, cm)
    n = len(y_true)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    majority = max((sum(row) for row in cm), default=0) / n if n else 0.0
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "accuracy_wilson95": wilson_interval(correct, n),
        "majority_class_baseline_accuracy": majority,
        **macro_f1(stats),
        "per_class": [s.__dict__ for s in stats],
        "confusion_matrix": {"labels": list(labels), "rows_true_cols_pred": cm},
        **zero_support_report(stats),
    }


# ------------------------------------------------------------------------------------------------------------------------ paired comparison
def _macro_all(labels, yt, yp) -> float:
    return macro_f1(per_class(labels, confusion_matrix(labels, yt, yp)))["macro_f1_all_classes"]


def _macro_sup(labels, yt, yp) -> float:
    return macro_f1(per_class(labels, confusion_matrix(labels, yt, yp)))["macro_f1_supported_classes"]


def mcnemar_exact(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    """Exact two-sided McNemar test on the discordant pairs (only A right / only B right)."""
    b = sum(1 for t, a, c in zip(y_true, pred_a, pred_b) if a == t and c != t)   # only A correct
    c_ = sum(1 for t, a, c in zip(y_true, pred_a, pred_b) if a != t and c == t)  # only B correct
    n = b + c_
    if n == 0:
        return {"only_a_correct": b, "only_b_correct": c_, "p_value": 1.0}
    k = min(b, c_)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))
    return {"only_a_correct": b, "only_b_correct": c_, "p_value": p}


def paired_bootstrap(labels: Sequence[str], y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str], *,
                     n_resamples: int = 2000, seed: int = 0) -> dict:
    """B minus A on identical rows: accuracy and both macro-F1 flavours, with percentile 95% intervals. Positive values favour B."""
    n = len(y_true)
    if not (n == len(pred_a) == len(pred_b)):
        raise ValueError("all systems must be scored on the same rows")
    rng = random.Random(seed)
    metrics: dict[str, Callable] = {
        "accuracy": lambda t, p: accuracy(t, p),
        "macro_f1_all_classes": lambda t, p: _macro_all(labels, t, p),
        "macro_f1_supported_classes": lambda t, p: _macro_sup(labels, t, p),
    }
    point = {k: f(y_true, pred_b) - f(y_true, pred_a) for k, f in metrics.items()}
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    for _ in range(n_resamples):
        ix = [rng.randrange(n) for _ in range(n)]
        t = [y_true[i] for i in ix]
        a = [pred_a[i] for i in ix]
        b = [pred_b[i] for i in ix]
        for k, f in metrics.items():
            draws[k].append(f(t, b) - f(t, a))
    out = {}
    for k, v in draws.items():
        v.sort()
        out[k] = {"difference_b_minus_a": point[k], "ci95": (v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))])}
    out["mcnemar_exact_accuracy"] = mcnemar_exact(y_true, pred_a, pred_b)
    return out


# ------------------------------------------------------------------------------------------------------------------------ routing & ambiguity
def routing_report(gold_routing: Sequence[str], pred_routing: Sequence[str]) -> dict:
    """Operational correctness. 'false_exclusion' = a comment that should reach a person is routed to EXCLUDE (the costly error);
    'missed_exclusion' = a comment that should be excluded is sent on for a reply."""
    if len(gold_routing) != len(pred_routing):
        raise ValueError("length mismatch")
    n = len(gold_routing)
    agree = sum(g == p for g, p in zip(gold_routing, pred_routing))
    false_ex = sum(1 for g, p in zip(gold_routing, pred_routing) if p == "EXCLUDE" and g != "EXCLUDE")
    missed_ex = sum(1 for g, p in zip(gold_routing, pred_routing) if g == "EXCLUDE" and p != "EXCLUDE")
    gold_ex = sum(1 for g in gold_routing if g == "EXCLUDE")
    return {
        "n": n, "routing_agreement": agree / n if n else 0.0,
        "false_exclusion": false_ex, "false_exclusion_rate": false_ex / max(1, n - gold_ex),
        "missed_exclusion": missed_ex, "missed_exclusion_rate": missed_ex / max(1, gold_ex),
    }


def ambiguity_report(gold_ambiguous: Sequence[bool], pred_ambiguous: Sequence[bool]) -> dict:
    tp = sum(1 for g, p in zip(gold_ambiguous, pred_ambiguous) if g and p)
    fp = sum(1 for g, p in zip(gold_ambiguous, pred_ambiguous) if not g and p)
    fn = sum(1 for g, p in zip(gold_ambiguous, pred_ambiguous) if g and not p)
    return {"n": len(gold_ambiguous), "gold_ambiguous": sum(map(bool, gold_ambiguous)), "flagged": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0}
