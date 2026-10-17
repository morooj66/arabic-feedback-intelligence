"""Dev-only diagnostics for the frozen pretrained sentiment model (CAMeLBERT-DA, inference only). Blind labels: aggregate counts only."""

from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import common as C  # noqa: E402
from config import settings  # noqa: E402
from rcs.training.gold_splits import LabeledExample  # noqa: E402
from rcs.training.sentiment_pretrained import evaluate_pretrained_sentiment  # noqa: E402

ORDER = list(settings.SENTIMENTS)  # POSITIVE, NEGATIVE, NEUTRAL
DIAC = re.compile(r"[ً-ٰٟـ]")
THANKS = ["شكرا", "شكراً", "جزاكم الله", "بارك الله", "ممتاز", "رائع", "جهد", "مشكورين", "الله يعطيكم", "احسنتم", "أحسنتم", "تسلمون", "ما قصرتم"]
COURTESY = ["السلام عليكم", "تحية طيبة", "تحيه طيبه", "بعد التحية"]


def f1s(y, pred):
    out = {}
    for c in ORDER:
        tp = sum(1 for a, b in zip(y, pred) if a == c and b == c); fp = sum(1 for a, b in zip(y, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, pred) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return out


def main() -> None:
    dev = C.load_dev()
    y = [r["gold_sentiment"] for r in dev]
    ex = [LabeledExample(comment_id=r["comment_id"], text=r["comment_text"], label="NEUTRAL", annotation_stage="dev", split="dev") for r in dev]
    res = evaluate_pretrained_sentiment(ex, model_id="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment", batch_size=16)
    P = np.array(res.probabilities)
    pred = list(res.y_pred)
    out = {"n": len(y)}
    print("gold dist:", dict(Counter(y)), "| predicted dist:", dict(Counter(pred)))
    cm = {g: dict(Counter(p for gg, p in zip(y, pred) if gg == g)) for g in ORDER}
    print("confusion (rows gold):", cm)
    pr = f1s(y, pred); print("per-class P/R/F1:", {c: tuple(round(v, 2) for v in pr[c]) for c in ORDER})
    out["raw"] = {"acc": float(np.mean([a == b for a, b in zip(y, pred)])), "macro_f1": float(np.mean([pr[c][2] for c in ORDER])), "cm": cm}
    # slices: formal / diacritised text vs plain; thanks lexicon; courtesy opening
    sl = {}
    for name, fn in (("has_diacritics", lambda t: bool(DIAC.search(t))), ("thanks_or_praise_words", lambda t: any(w in t for w in THANKS)),
                     ("courtesy_opening", lambda t: any(w in t for w in COURTESY)), ("long_text_>300ch", lambda t: len(t) > 300)):
        idx = [i for i, r in enumerate(dev) if fn(r["comment_text"])]
        yy = [y[i] for i in idx]; pp = [pred[i] for i in idx]
        sl[name] = {"n": len(idx), "gold": dict(Counter(yy)), "pred": dict(Counter(pp)),
                    "NEUTRAL->NEGATIVE": sum(1 for a, b in zip(yy, pp) if a == "NEUTRAL" and b == "NEGATIVE"),
                    "POSITIVE_recall": (sum(1 for a, b in zip(yy, pp) if a == "POSITIVE" and b == "POSITIVE") / max(1, yy.count("POSITIVE")))}
        print(name, sl[name])
    out["slices"] = sl
    # POSITIVE rows: what do they look like (lexicon share), dev vs blind (counts only)
    pos_dev = [r for r in dev if r["gold_sentiment"] == "POSITIVE"]
    blind = C.load_blind(); pos_b = [r for r in blind if r["sentiment"] == "POSITIVE"]
    share = lambda rows: (sum(any(w in r["comment_text"] for w in THANKS) for rows_ in [rows] for r in rows_), sum(any(w in r["comment_text"] for w in COURTESY) for r in rows), len(rows))
    out["positive_lexicon_share"] = {"dev": share(pos_dev), "blind": share(pos_b)}
    print("POSITIVE rows (thanks-lexicon, courtesy-opening, total): dev", share(pos_dev), "| blind", share(pos_b))
    # recalibration: learned LR on the 3 probabilities (+ log length, diacritics flag), repeated CV, development rows only
    X = np.column_stack([np.log(P + 1e-6), [np.log1p(len(r["comment_text"])) for r in dev], [1.0 if DIAC.search(r["comment_text"]) else 0.0 for r in dev]])
    scores = {"raw": [], "recal_probs_only": [], "recal_probs+length+diacritics": []}
    for seed in (0, 1, 2):
        for name, cols in (("recal_probs_only", [0, 1, 2]), ("recal_probs+length+diacritics", [0, 1, 2, 3, 4])):
            predc = [None] * len(y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X[tr][:, cols], [y[i] for i in tr])
                    for i, p in zip(te, m.predict(X[te][:, cols])):
                        predc[i] = p
            f = f1s(y, predc); scores[name].append((float(np.mean([a == b for a, b in zip(y, predc)])), float(np.mean([f[c][2] for c in ORDER])), f["NEGATIVE"][1], f["POSITIVE"][1]))
    scores["raw"] = [(out["raw"]["acc"], out["raw"]["macro_f1"], pr["NEGATIVE"][1], pr["POSITIVE"][1])] * 3
    print("\nvariant | accuracy | macro-F1 | NEGATIVE recall | POSITIVE recall (dev, repeated 5-fold CV)")
    for n, v in scores.items():
        a = np.mean(v, axis=0); print(f"{n:32} {a[0]:.3f} {a[1]:.3f} {a[2]:.2f} {a[3]:.2f}")
    out["recalibration_cv"] = {n: [float(x) for x in np.mean(v, axis=0)] for n, v in scores.items()}
    (C.OUT / "sentiment_diagnostics.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
