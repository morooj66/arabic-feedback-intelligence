"""Isolated test of the confirmed Arabic-diacritics tokenisation defect in the frozen Intent classifier (TF-IDF + balanced logistic regression).

The frozen pipeline's default token pattern splits a word at every combining mark, so a fully diacritised word yields no tokens. Two PREDEFINED
candidates differ from the frozen pipeline ONLY by a text preprocessor (same vectorizer settings, same classifier, same seed):
  C1  strip diacritics and tatweel only (the minimal fix)
  C2  rcs.classification.arabic_text.normalize_arabic (diacritics, tatweel, alef variants, digits, elongation) - the variant already tried in the error analysis
Data: existing gold TRAIN (455) / VAL (108) rows only. The sealed human_test split is never read. Prints aggregates only; no comment text.

    python -m scripts.offline_eval.intent_tokenizer_fix.evaluate
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rcs.classification.arabic_text import normalize_arabic  # noqa: E402
from rcs.routing.engine import determine_routing  # noqa: E402
from scripts.offline_eval.gemini_classification import metrics as m  # noqa: E402

GOLD = ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
INTENTS = ("THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM")
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]")
_TOKEN = re.compile(r"(?u)\b\w\w+\b")                    # sklearn's default token pattern: what the frozen vectorizer uses
FROZEN_VAL_MACRO_F1 = 0.31538918597742127                 # data/models/selection/FROZEN_MODEL_SELECTION.json


def strip_diacritics(text: str) -> str:
    return _DIAC.sub("", str(text or ""))


PREPROCESSORS = {"frozen": None, "C1_strip_diacritics_tatweel": strip_diacritics, "C2_normalize_arabic": normalize_arabic}


def make_pipeline(preprocessor):
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50_000, sublinear_tf=True, preprocessor=None if preprocessor is None else preprocessor)
    if preprocessor is None:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50_000, sublinear_tf=True)
    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"))])


def load_rows(splits=("train", "val")):
    with open(GOLD, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] in splits and r["gold_annotation_status"] == "LABELED"]
    return rows


def has_diacritics(t: str) -> bool:
    return bool(_DIAC.search(t)) and bool(re.search(r"[ً-ٰٟ]", t))


def routing_of(intent: str, topic: str) -> str:
    return determine_routing(primary_intent=intent, topic_subtype=topic, subcategory=None, secondary_intent=None, is_content_meaningful=True).routing_decision


def score(y_true, y_pred, topics):
    rep = m.task_report(INTENTS, y_true, y_pred)
    r = m.routing_report([routing_of(i, t) for i, t in zip(y_true, topics)], [routing_of(i, t) for i, t in zip(y_pred, topics)])
    return rep, r


def tokens_report(texts, preprocessor):
    empty = sum(1 for t in texts if not _TOKEN.findall(t if preprocessor is None else preprocessor(t)))
    return {"rows": len(texts), "rows_with_no_tokens": empty, "mean_tokens": sum(len(_TOKEN.findall(t if preprocessor is None else preprocessor(t))) for t in texts) / max(1, len(texts))}


def main() -> int:
    rows = load_rows()
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    X = lambda rs: [r["comment_text"] for r in rs]
    Y = lambda rs: [r["gold_primary_intent"] for r in rs]
    T = lambda rs: [r["gold_topic_subtype"] for r in rs]
    out: dict = {"n_train": len(train), "n_val": len(val)}
    out["diacritised_rows"] = {"train": sum(has_diacritics(t) for t in X(train)), "val": sum(has_diacritics(t) for t in X(val))}
    out["tokenisation"] = {name: tokens_report(X(rows), fn) for name, fn in PREPROCESSORS.items()}
    # ---------------- A. train -> val, identical rows
    preds = {}
    for name, fn in PREPROCESSORS.items():
        pipe = make_pipeline(fn).fit(X(train), Y(train))
        preds[name] = pipe.predict(X(val)).tolist()
    base = m.task_report(INTENTS, Y(val), preds["frozen"])["macro_f1_all_classes"]
    out["frozen_val_macro_f1_reproduced"] = {"value": base, "reported": FROZEN_VAL_MACRO_F1, "exact": abs(base - FROZEN_VAL_MACRO_F1) < 1e-9}
    dia_idx = [i for i, t in enumerate(X(val)) if has_diacritics(t)]
    out["val"] = {}
    for name in PREPROCESSORS:
        rep, rt = score(Y(val), preds[name], T(val))
        out["val"][name] = {"accuracy": rep["accuracy"], "macro_f1_all": rep["macro_f1_all_classes"], "macro_f1_supported": rep["macro_f1_supported_classes"],
                            "per_class": {c["label"]: [round(c["precision"], 3), round(c["recall"], 3), round(c["f1"], 3), c["support"], c["predicted"]] for c in rep["per_class"]},
                            "routing": {k: rt[k] for k in ("routing_agreement", "false_exclusion", "missed_exclusion")},
                            "diacritised_subset_accuracy": (sum(Y(val)[i] == preds[name][i] for i in dia_idx) / len(dia_idx)) if dia_idx else None, "diacritised_subset_n": len(dia_idx)}
        if name != "frozen":
            out["val"][name]["paired_vs_frozen"] = m.paired_bootstrap(INTENTS, Y(val), preds["frozen"], preds[name], n_resamples=2000, seed=5)
    # ---------------- B. repeated 5-fold CV on all development rows (train + val), identical folds
    X_all, Y_all, T_all = X(rows), Y(rows), T(rows)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=11)
    fold_scores = {n: [] for n in PREPROCESSORS}
    pooled = {n: [[None] * len(rows) for _ in range(3)] for n in PREPROCESSORS}
    for k, (tr, te) in enumerate(cv.split(X_all, Y_all)):
        rep_i = k // 5
        for name, fn in PREPROCESSORS.items():
            pipe = make_pipeline(fn).fit([X_all[i] for i in tr], [Y_all[i] for i in tr])
            pr = pipe.predict([X_all[i] for i in te]).tolist()
            for i, p in zip(te, pr):
                pooled[name][rep_i][i] = p
            fold_scores[name].append(m.task_report(INTENTS, [Y_all[i] for i in te], pr)["macro_f1_supported_classes"])
    dia_all = [i for i, t in enumerate(X_all) if has_diacritics(t)]
    out["cv"] = {"rows": len(rows), "diacritised_rows": len(dia_all)}
    for name in PREPROCESSORS:
        accs, macs, fes, dia_acc = [], [], [], []
        for rep_i in range(3):
            p = pooled[name][rep_i]
            rep, rt = score(Y_all, p, T_all)
            accs.append(rep["accuracy"]); macs.append(rep["macro_f1_supported_classes"]); fes.append(rt["false_exclusion"])
            dia_acc.append(sum(Y_all[i] == p[i] for i in dia_all) / len(dia_all))
        out["cv"][name] = {"accuracy": sum(accs) / 3, "macro_f1_supported": sum(macs) / 3, "false_exclusion_mean": sum(fes) / 3, "diacritised_accuracy": sum(dia_acc) / 3,
                           "fold_macro_f1_mean": sum(fold_scores[name]) / len(fold_scores[name])}
    for name in ("C1_strip_diacritics_tatweel", "C2_normalize_arabic"):
        d = [a - b for a, b in zip(fold_scores[name], fold_scores["frozen"])]
        mean = sum(d) / len(d)
        sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        out["cv"][name]["fold_diff_vs_frozen"] = {"mean": mean, "sd": sd, "folds_better": sum(x > 0 for x in d), "folds_worse": sum(x < 0 for x in d), "folds": len(d)}
    res = ROOT / "scripts" / "offline_eval" / "intent_tokenizer_fix" / "results"
    res.mkdir(exist_ok=True)
    (res / "intent_tokenizer_fix_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
