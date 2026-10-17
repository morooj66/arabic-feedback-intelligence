"""Scoring for the Jev vs frozen-RCS development comparison. Pure; no I/O, no network.

For each task (intent, topic, sentiment): accuracy, macro-F1 (over classes with
gold support, and over the full frozen label set), per-class precision / recall /
F1 / support, plus the specific questions asked of this evaluation: Complaint
recall, Technical-Issue recall, the suggestion-subtype confusion, the sentiment
confusion. Bootstrap 95% intervals are reported because the development
validation set is small (a handful of rows in several classes).
"""

from __future__ import annotations

import random
from typing import Sequence

from config import settings

TASK_LABELS = {
    "intent": tuple(settings.PRIMARY_INTENTS),
    "topic": tuple(settings.TOPIC_SUBTYPES),
    "sentiment": tuple(settings.SENTIMENTS),
}
SUGGESTION_SUBTYPES = ("WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def per_class(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict[str, dict]:
    out = {}
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        p, r, f = _prf(tp, fp, fn)
        out[lab] = {"precision": p, "recall": r, "f1": f, "support": tp + fn, "predicted": tp + fp}
    return out


def macro_f1(pc: dict[str, dict], *, only_supported: bool) -> float:
    vals = [v["f1"] for v in pc.values() if (v["support"] > 0 or not only_supported)]
    return sum(vals) / len(vals) if vals else 0.0


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0


def confusion(gold: Sequence[str], pred: Sequence[str], rows: Sequence[str], cols: Sequence[str]) -> dict:
    """rows = gold labels, cols = predicted labels; any prediction outside `cols` is counted under 'OTHER'."""
    cols_x = list(cols) + ["OTHER"]
    m = {r: {c: 0 for c in cols_x} for r in rows}
    for g, p in zip(gold, pred):
        if g in m:
            m[g][p if p in cols else "OTHER"] += 1
    return m


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def bootstrap_ci(gold, pred, labels, *, n_boot: int = 1000, seed: int = 20261014) -> dict:
    rng = random.Random(seed)
    n = len(gold)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        accs.append(accuracy(g, p))
        f1s.append(macro_f1(per_class(g, p, labels), only_supported=True))
    accs.sort(); f1s.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"accuracy": (accs[lo], accs[hi]), "macro_f1": (f1s[lo], f1s[hi])}


def score_task(task: str, gold: Sequence[str], pred: Sequence[str]) -> dict:
    labels = TASK_LABELS[task]
    pc = per_class(gold, pred, labels)
    return {
        "n": len(gold),
        "accuracy": accuracy(gold, pred),
        "macro_f1_supported_classes": macro_f1(pc, only_supported=True),
        "macro_f1_all_classes": macro_f1(pc, only_supported=False),
        "per_class": pc,
        "ci95": bootstrap_ci(list(gold), list(pred), labels) if gold else None,
    }


def evaluate(gold: dict[str, dict], predictions: dict[str, dict[str, str]]) -> dict:
    """gold: eval_id -> {intent, topic, sentiment}; predictions: eval_id -> same keys.
    Rows lacking a prediction are listed as `missing` and excluded (never imputed)."""
    ids = [i for i in gold if i in predictions]
    missing = [i for i in gold if i not in predictions]
    res: dict = {"rows_scored": len(ids), "rows_missing_prediction": len(missing), "tasks": {}}
    for task in TASK_LABELS:
        g = [gold[i][task] for i in ids]
        p = [predictions[i][task] for i in ids]
        res["tasks"][task] = score_task(task, g, p)
    gi = [gold[i]["intent"] for i in ids]
    pi = [predictions[i]["intent"] for i in ids]
    gt = [gold[i]["topic"] for i in ids]
    pt = [predictions[i]["topic"] for i in ids]
    gs = [gold[i]["sentiment"] for i in ids]
    ps = [predictions[i]["sentiment"] for i in ids]
    c_k = sum(1 for g, p in zip(gi, pi) if g == "COMPLAINT" and p == "COMPLAINT")
    c_n = sum(1 for g in gi if g == "COMPLAINT")
    t_k = sum(1 for g, p in zip(gt, pt) if g == "TECHNICAL_ISSUE" and p == "TECHNICAL_ISSUE")
    t_n = sum(1 for g in gt if g == "TECHNICAL_ISSUE")
    res["complaint_recall"] = {"hits": c_k, "support": c_n, "recall": c_k / c_n if c_n else None,
                               "wilson95": wilson(c_k, c_n)}
    res["technical_issue_recall"] = {"hits": t_k, "support": t_n, "recall": t_k / t_n if t_n else None,
                                     "wilson95": wilson(t_k, t_n)}
    sub = [(g, p) for g, p in zip(gt, pt) if g in SUGGESTION_SUBTYPES]
    res["suggestion_subtype_confusion"] = confusion([g for g, _ in sub], [p for _, p in sub],
                                                    SUGGESTION_SUBTYPES, SUGGESTION_SUBTYPES)
    res["sentiment_confusion"] = confusion(gs, ps, TASK_LABELS["sentiment"], TASK_LABELS["sentiment"])
    return res


def to_markdown(name: str, result: dict) -> str:
    out = [f"## {name}", "", f"Rows scored: {result['rows_scored']} (missing predictions: {result['rows_missing_prediction']})", ""]
    out += ["| Task | Accuracy | Macro-F1 (supported) | Macro-F1 (all classes) | 95% CI acc | 95% CI macro-F1 |", "|---|---|---|---|---|---|"]
    for t, r in result["tasks"].items():
        ci = r["ci95"] or {"accuracy": (0, 0), "macro_f1": (0, 0)}
        out.append(f"| {t} | {r['accuracy']:.3f} | {r['macro_f1_supported_classes']:.3f} | {r['macro_f1_all_classes']:.3f} | "
                   f"{ci['accuracy'][0]:.2f}-{ci['accuracy'][1]:.2f} | {ci['macro_f1'][0]:.2f}-{ci['macro_f1'][1]:.2f} |")
    for t, r in result["tasks"].items():
        out += ["", f"### {t}: per-class", "", "| Class | P | R | F1 | support | predicted |", "|---|---|---|---|---|---|"]
        for lab, v in r["per_class"].items():
            out.append(f"| {lab} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} | {v['predicted']} |")
    for key, title in (("complaint_recall", "Complaint recall"), ("technical_issue_recall", "Technical-Issue recall")):
        v = result[key]
        rec = "n/a" if v["recall"] is None else f"{v['recall']:.2f}"
        out += ["", f"**{title}:** {v['hits']}/{v['support']} = {rec} (Wilson 95%: {v['wilson95'][0]:.2f}-{v['wilson95'][1]:.2f})"]
    for key, title in (("suggestion_subtype_confusion", "Suggestion-subtype confusion (rows = gold, cols = predicted)"),
                       ("sentiment_confusion", "Sentiment confusion (rows = gold, cols = predicted)")):
        m = result[key]
        cols = list(next(iter(m.values())).keys())
        out += ["", f"### {title}", "", "| gold \\ pred | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for r, row in m.items():
            out.append(f"| {r} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"
