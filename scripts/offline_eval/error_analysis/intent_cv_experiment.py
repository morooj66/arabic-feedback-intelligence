"""Dev-only cross-validation: which BASE-model changes help Intent (esp. COMPLAINT)? No blind data, no thresholds tuned on blind.

Variants (all learned models; the lexicon enters only as input features, never as a decision rule):
  V0 frozen-like  : word 1-2gram TF-IDF + balanced LogisticRegression (the frozen configuration)
  V1 + normalise  : Arabic normalisation (diacritics/tatweel removed; alef/ya/ta-marbuta unified) before TF-IDF
  V2 + char n-gram: V1 + char_wb 2-5 n-grams (dialect spelling / morphology)
  V3 + features   : V2 + failure-expression families and contrast cues (how-to / feature request / lexical-error) as learned features
  V4 + augmentation: V3 trained with the synthetic candidates ADDED TO TRAINING FOLDS ONLY (never scored)
Metrics are out-of-fold on the 563 labelled development rows, 3 repeats x 5 folds.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import common as C  # noqa: E402

CLASSES = ["THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM"]
FAM = list(C.FAMILIES) + list(C.CONTRAST)


def feats(texts):
    return csr_matrix(np.array([[1.0 if C.hits(t, (C.FAMILIES | C.CONTRAST)[k]) else 0.0 for k in FAM] for t in texts]) * 2.0)


class Model:
    def __init__(self, normalise, char, features, C_=1.0):
        self.normalise, self.char, self.features, self.C = normalise, char, features, C_

    def _p(self, texts):
        return [C.norm_ar(t) for t in texts] if self.normalise else list(texts)

    def fit(self, texts, y):
        p = self._p(texts)
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=50000)
        parts = [self.w.fit_transform(p)]
        if self.char:
            self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)
            parts.append(self.c.fit_transform(p))
        if self.features:
            parts.append(feats(texts))
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.clf.fit(hstack(parts).tocsr(), y)
        return self

    def proba(self, texts):
        p = self._p(texts)
        parts = [self.w.transform(p)]
        if self.char:
            parts.append(self.c.transform(p))
        if self.features:
            parts.append(feats(texts))
        P = self.clf.predict_proba(hstack(parts).tocsr())
        order = [list(self.clf.classes_).index(c) for c in CLASSES]
        return P[:, order]


def prf(y, pred, cls):
    tp = sum(1 for a, b in zip(y, pred) if a == cls and b == cls); fp = sum(1 for a, b in zip(y, pred) if a != cls and b == cls)
    fn = sum(1 for a, b in zip(y, pred) if a == cls and b != cls)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    dev = C.load_dev()
    texts = [r["comment_text"] for r in dev]; y = [r["gold_primary_intent"] for r in dev]
    syn = list(csv.DictReader(open(C.OUT / "synthetic_augmentation_candidates.csv", encoding="utf-8")))
    stext = [r["text"] for r in syn]; sy = [r["proposed_intent"] for r in syn]
    variants = {"V0 frozen-like": (Model(False, False, False), False), "V1 + normalise": (Model(True, False, False), False),
                "V2 + char n-grams": (Model(True, True, False), False), "V3 + expression features": (Model(True, True, True), False),
                "V4 + synthetic augmentation (train folds only)": (Model(True, True, True), True)}
    res = {}
    for name, (proto, aug) in variants.items():
        runs = []
        for seed in (0, 1, 2):
            pred = [None] * len(y); prob = np.zeros((len(y), 5))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y):
                    tt = [texts[i] for i in tr]; yy = [y[i] for i in tr]
                    if aug:
                        tt += stext; yy += sy
                    m = Model(proto.normalise, proto.char, proto.features).fit(tt, yy)
                    P = m.proba([texts[i] for i in te])
                    for k, i in enumerate(te):
                        prob[i] = P[k]; pred[i] = CLASSES[int(P[k].argmax())]
            per = {c: prf(y, pred, c) for c in CLASSES}
            ap = average_precision_score([1 if a == "COMPLAINT" else 0 for a in y], prob[:, CLASSES.index("COMPLAINT")])
            runs.append({"macro_f1": float(np.mean([per[c][2] for c in CLASSES])), "acc": float(np.mean([a == b for a, b in zip(y, pred)])),
                         "complaint_ap": float(ap), **{f"{c}_{k}": per[c][i] for c in CLASSES for i, k in enumerate("PRF")}})
        res[name] = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
    L = ["| variant | accuracy | macro-F1 | COMPLAINT P / R / F1 | COMPLAINT avg-precision | INQUIRY F1 | SUGGESTION F1 | SPAM F1 |", "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        L.append(f"| {n} | {r['acc'][0]:.3f} | {r['macro_f1'][0]:.3f}±{r['macro_f1'][1]:.3f} | {r['COMPLAINT_P'][0]:.2f} / {r['COMPLAINT_R'][0]:.2f} / {r['COMPLAINT_F'][0]:.2f} | "
                 f"{r['complaint_ap'][0]:.3f} | {r['INQUIRY_F'][0]:.2f} | {r['SUGGESTION_F'][0]:.2f} | {r['SPAM_F'][0]:.2f} |")
    (C.OUT / "intent_cv_experiment.md").write_text("\n".join(L), encoding="utf-8")
    (C.OUT / "intent_cv_experiment.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))
    print("n dev rows:", len(y), "| COMPLAINT rows:", y.count("COMPLAINT"), "| base rate", round(y.count("COMPLAINT") / len(y), 3))


if __name__ == "__main__":
    main()
