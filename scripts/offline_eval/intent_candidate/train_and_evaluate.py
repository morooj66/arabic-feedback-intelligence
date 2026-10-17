"""Train and evaluate the EXPERIMENTAL Intent candidate (rcs.classification.intent_candidate). OFFLINE. Nothing is activated.

DATA RULES (enforced here):
  * Real development labels only (gold train/val, LABELED). The sealed human_test rows are dropped on read. The 75 blind records are never
    loaded: there is no import of any blind file in this script (a test asserts it).
  * Preprocessing, vectorisers and every learned parameter are fitted INSIDE each training fold / training split.
  * Synthetic contrast examples are added to TRAINING data only, tagged, never scored in the real-data metrics, and evaluated separately
    with theme-grouped folds so a contrast pair (failure report / how-to question / feature request on the same theme) is never split.
  * Hyper-parameters are fixed in advance (intent_candidate.DEFAULT_PARAMS). The sensitivity grid is reported, never used to choose.

PROTOCOLS
  A  (primary)      repeated stratified 5-fold CV on the 455 TRAIN rows only (validation rows never touched)
  B  (frozen split) train on the 455 train rows -> score on the 108 validation rows (the protocol the frozen model was selected with)
  C  (supplementary) repeated 5-fold CV on all 563 development rows (validation rows appear in training folds of OTHER folds)
  S  (synthetic)    theme-grouped hold-out of the synthetic examples (separate, optimistic, not evidence of real accuracy)

Variants on IDENTICAL folds:  FROZEN (refit of the frozen configuration) | FROZEN + synthetic | CANDIDATE real-only | CANDIDATE + synthetic.
"""

from __future__ import annotations

import os

# Single-threaded math libraries: on small sparse problems multi-threaded BLAS oversubscribes the CPU (a 0.6 s fit took 24 s) and
# thread scheduling is a source of run-to-run numerical noise. Must be set BEFORE numpy / scikit-learn are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import hashlib
import json
import platform
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src", ROOT):
    sys.path.insert(0, str(p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from rcs.classification import intent_candidate as IC  # noqa: E402

CLASSES = list(IC.CLASSES)
GOLD = ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
SYN = ROOT / "data" / "evaluation" / "error_analysis" / "synthetic_augmentation_candidates.csv"
OUT = ROOT / "data" / "evaluation" / "intent_candidate_v1"
MODEL_DIR = ROOT / "data" / "models" / "experimental" / "intent_candidate_v1"
SEEDS = (0, 1, 2)
N_SPLITS = 5


# ------------------------------------------------------------------------------------------------------- data


def load_real() -> list[dict]:
    rows = []
    with open(GOLD, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in ("train", "val") and r["gold_annotation_status"] == "LABELED" and r["gold_primary_intent"] in CLASSES:
                rows.append({"id": r["comment_id"], "text": r["comment_text"], "y": r["gold_primary_intent"], "split": r["split"]})
    return rows


_THEMES = (("search_results", ("ابحث", "البحث", "نتيجه", "نتائج", "يطلع")), ("login", ("دخول", "تسجيل")),
           ("page_open", ("يفتح", "تفتح", "الصفحه", "الرابط", "الموقع")), ("app", ("التطبيق", "تطبيق", "الجوال")),
           ("content", ("تعريف", "الكلمه", "المعنى", "مدخل", "مثال", "كلمه")), ("feature", ("خاصيه", "قسم", "تحميل", "اضافه", "اضيف")))


def theme_of(text: str) -> str:
    m = IC.match_form(text)
    for name, keys in _THEMES:
        if any(k in m for k in keys):
            return name
    return "other"


def load_synthetic() -> list[dict]:
    out = []
    for r in csv.DictReader(open(SYN, encoding="utf-8")):
        out.append({"text": r["text"], "y": r["proposed_intent"], "group": theme_of(r["text"]), "family": r["family"]})
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------- models


class FrozenConfig:
    """Refit of the FROZEN configuration: raw text, default word TF-IDF 1-2 grams, balanced LogisticRegression (same hyper-parameters)."""

    def fit(self, texts, y, sample_weight=None):
        self.v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
        self.c = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
        self.c.fit(self.v.fit_transform(list(texts)), list(y), sample_weight=None if sample_weight is None else np.asarray(sample_weight))
        self.order = [list(self.c.classes_).index(c) for c in CLASSES]
        return self

    def predict_proba(self, texts):
        return self.c.predict_proba(self.v.transform(list(texts)))[:, self.order]


def make(kind: str, **params):
    return FrozenConfig() if kind.startswith("frozen") else IC.IntentCandidate(**params)


VARIANTS = {  # name -> (kind, uses_synthetic)
    "FROZEN configuration (refit)": ("frozen", False),
    "FROZEN configuration + synthetic": ("frozen", True),
    "CANDIDATE, real data only": ("candidate", False),
    "CANDIDATE + synthetic": ("candidate", True),
}


def fit_variant(kind, use_syn, tr_texts, tr_y, syn, params=None):
    params = params or {}
    texts, y, w = list(tr_texts), list(tr_y), [1.0] * len(tr_texts)
    if use_syn and syn:
        sw = params.get("synthetic_weight", IC.DEFAULT_PARAMS["synthetic_weight"])
        if sw > 0:
            texts += [s["text"] for s in syn]; y += [s["y"] for s in syn]; w += [sw] * len(syn)
    p = {k: v for k, v in params.items() if k != "synthetic_weight"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make(kind, **p).fit(texts, y, sample_weight=w)


# ----------------------------------------------------------------------------------------------------- metrics


def prf(y, pred):
    out = {}
    for c in CLASSES:
        tp = sum(1 for a, b in zip(y, pred) if a == c and b == c); fp = sum(1 for a, b in zip(y, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, pred) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    return out


def summary(y, pred):
    pc = prf(y, pred)
    present = [c for c in CLASSES if pc[c]["support"] > 0]
    cm = {g: {p: sum(1 for a, b in zip(y, pred) if a == g and b == p) for p in CLASSES} for g in CLASSES}
    return {"accuracy": float(np.mean([a == b for a, b in zip(y, pred)])), "macro_f1": float(np.mean([pc[c]["F1"] for c in present])),
            "per_class": pc, "confusion": cm, "complaint_recall": pc["COMPLAINT"]["R"], "complaint_fp": pc["COMPLAINT"]["fp"],
            "complaint_precision": pc["COMPLAINT"]["P"]}


def cv_protocol(rows, syn, params=None, repeats=SEEDS, variants=VARIANTS):
    texts = [r["text"] for r in rows]; y = [r["y"] for r in rows]
    res = {v: {"repeats": [], "folds": []} for v in variants}
    oof0 = {}
    for seed in repeats:
        folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(texts, y))     # identical for every variant
        for v, (kind, use_syn) in variants.items():
            pred = [None] * len(y)
            for tr, te in folds:
                m = fit_variant(kind, use_syn, [texts[i] for i in tr], [y[i] for i in tr], syn, params)
                P = m.predict_proba([texts[i] for i in te])
                for k, i in enumerate(te):
                    pred[i] = CLASSES[int(P[k].argmax())]
                fy = [y[i] for i in te]; fp_ = [pred[i] for i in te]
                s = summary(fy, fp_)
                res[v]["folds"].append({"seed": seed, "macro_f1": s["macro_f1"], "accuracy": s["accuracy"],
                                        "complaint_recall": s["complaint_recall"], "complaint_support": s["per_class"]["COMPLAINT"]["support"]})
            res[v]["repeats"].append(summary(y, pred))
            if seed == repeats[0]:
                oof0[v] = list(pred)
    return res, oof0, y


def agg(res_v):
    reps = res_v["repeats"]; folds = res_v["folds"]
    m = lambda k: (float(np.mean([r[k] for r in reps])), float(np.std([r[k] for r in reps])))
    out = {"accuracy": m("accuracy"), "macro_f1": m("macro_f1"), "complaint_recall": m("complaint_recall"),
           "complaint_precision": m("complaint_precision"), "complaint_fp_per_run": float(np.mean([r["complaint_fp"] for r in reps])),
           "per_class": {c: {k: float(np.mean([r["per_class"][c][k] for r in reps])) for k in ("P", "R", "F1")} | {"support": reps[0]["per_class"][c]["support"]} for c in CLASSES},
           "confusion_mean": {g: {p: float(np.mean([r["confusion"][g][p] for r in reps])) for p in CLASSES} for g in CLASSES},
           "fold_macro_f1": {"mean": float(np.mean([f["macro_f1"] for f in folds])), "sd": float(np.std([f["macro_f1"] for f in folds])),
                             "min": float(min(f["macro_f1"] for f in folds)), "max": float(max(f["macro_f1"] for f in folds)), "n_folds": len(folds)},
           "fold_complaint_recall": {"mean": float(np.mean([f["complaint_recall"] for f in folds if f["complaint_support"]])),
                                     "sd": float(np.std([f["complaint_recall"] for f in folds if f["complaint_support"]])),
                                     "folds_with_complaint_detected": int(sum(1 for f in folds if f["complaint_support"] and f["complaint_recall"] > 0)),
                                     "folds_with_complaints": int(sum(1 for f in folds if f["complaint_support"]))}}
    return out


def paired(res, oof0, y, a="FROZEN configuration (refit)", b="CANDIDATE + synthetic", n_boot=2000):
    fa, fb = res[a]["folds"], res[b]["folds"]
    d = [x["macro_f1"] - z["macro_f1"] for x, z in zip(fb, fa)]
    rng = random.Random(20261017)
    da, df = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(len(y)) for _ in range(len(y))]
        g = [y[i] for i in idx]; pa = [oof0[a][i] for i in idx]; pb = [oof0[b][i] for i in idx]
        da.append(np.mean([u == v for u, v in zip(g, pb)]) - np.mean([u == v for u, v in zip(g, pa)]))
        df.append(summary(g, pb)["macro_f1"] - summary(g, pa)["macro_f1"])
    da.sort(); df.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"fold_macro_f1_diff_mean": float(np.mean(d)), "fold_macro_f1_diff_sd": float(np.std(d)),
            "folds_candidate_better": int(sum(1 for x in d if x > 0)), "folds_total": len(d),
            "record_bootstrap_accuracy_diff_ci95": [float(da[lo]), float(da[hi])], "record_bootstrap_macro_f1_diff_ci95": [float(df[lo]), float(df[hi])]}


# ----------------------------------------------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = load_real(); syn = load_synthetic()
    train = [r for r in real if r["split"] == "train"]; val = [r for r in real if r["split"] == "val"]
    prov = {"gold_csv_sha256": sha256_file(GOLD), "synthetic_csv_sha256": sha256_file(SYN), "n_real_total": len(real), "n_train": len(train),
            "n_val": len(val), "n_synthetic": len(syn), "train_ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in train)).encode()).hexdigest(),
            "val_ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in val)).encode()).hexdigest(),
            "class_counts_train": dict(Counter(r["y"] for r in train)), "class_counts_val": dict(Counter(r["y"] for r in val)),
            "synthetic_class_counts": dict(Counter(s["y"] for s in syn)), "synthetic_groups": dict(Counter(s["group"] for s in syn)),
            "sealed_human_test_read": False, "blind_75_used": False}
    results = {"provenance": prov}

    print("Protocol A (train rows only, repeated CV)...", flush=True)
    resA, oofA, yA = cv_protocol(train, syn)
    results["A"] = {v: agg(resA[v]) for v in VARIANTS}; results["A_paired"] = paired(resA, oofA, yA)
    print("Protocol C (all 563 dev rows, repeated CV)...", flush=True)
    resC, oofC, yC = cv_protocol(real, syn)
    results["C"] = {v: agg(resC[v]) for v in VARIANTS}; results["C_paired"] = paired(resC, oofC, yC)

    print("Protocol B (train -> validation) + final model...", flush=True)
    B = {}
    Xtr = [r["text"] for r in train]; ytr = [r["y"] for r in train]; Xva = [r["text"] for r in val]; yva = [r["y"] for r in val]
    final = None
    for v, (kind, use_syn) in VARIANTS.items():
        m = fit_variant(kind, use_syn, Xtr, ytr, syn)
        P = m.predict_proba(Xva); pred = [CLASSES[int(i)] for i in P.argmax(1)]
        B[v] = summary(yva, pred)
        if v == "CANDIDATE + synthetic":
            final = m
    frozen_art = joblib.load(ROOT / "data" / "models" / "final_test" / "intent_tfidf_logistic.joblib")
    Pf = frozen_art.predict_proba(Xva); cls = list(frozen_art.classes_)
    B["FROZEN production artifact (as shipped)"] = summary(yva, [cls[int(i)] for i in Pf.argmax(1)])
    results["B"] = B

    print("Protocol S (synthetic, theme-grouped)...", flush=True)
    groups = sorted({s["group"] for s in syn}); S = {"per_group": {}, "groups": groups}
    hits = {"CANDIDATE + synthetic (other themes in training)": [], "CANDIDATE real-only (control)": [], "FROZEN configuration (control)": []}
    grouped_pairs = []
    ctrl_c = fit_variant("candidate", False, Xtr, ytr, syn); ctrl_f = fit_variant("frozen", False, Xtr, ytr, syn)
    for g in groups:
        held = [s for s in syn if s["group"] == g]; rest = [s for s in syn if s["group"] != g]
        m = fit_variant("candidate", True, Xtr, ytr, rest)
        pred = [CLASSES[int(i)] for i in m.predict_proba([s["text"] for s in held]).argmax(1)]
        yy = [s["y"] for s in held]
        S["per_group"][g] = {"n": len(held), "classes": dict(Counter(yy)), "accuracy_candidate_synth": float(np.mean([a == b for a, b in zip(yy, pred)]))}
        hits["CANDIDATE + synthetic (other themes in training)"] += [a == b for a, b in zip(yy, pred)]
        grouped_pairs += list(zip(yy, pred))
    allp_c = [CLASSES[int(i)] for i in ctrl_c.predict_proba([s["text"] for s in syn]).argmax(1)]
    allp_f = [CLASSES[int(i)] for i in ctrl_f.predict_proba([s["text"] for s in syn]).argmax(1)]
    sy = [s["y"] for s in syn]
    hits["CANDIDATE real-only (control)"] = [a == b for a, b in zip(sy, allp_c)]; hits["FROZEN configuration (control)"] = [a == b for a, b in zip(sy, allp_f)]
    S["accuracy"] = {k: float(np.mean(v)) for k, v in hits.items()}
    S["complaint_recall_on_synthetic_failure_reports"] = {
        "CANDIDATE + synthetic (grouped)": float(np.mean([p == "COMPLAINT" for a, p in grouped_pairs if a == "COMPLAINT"])), "CANDIDATE real-only (control)": float(np.mean([p == "COMPLAINT" for p, a in zip(allp_c, sy) if a == "COMPLAINT"])),
        "FROZEN configuration (control)": float(np.mean([p == "COMPLAINT" for p, a in zip(allp_f, sy) if a == "COMPLAINT"]))}
    results["S"] = S

    print("Sensitivity grid (protocol A, one repeat; NOT used to choose)...", flush=True)
    sens = {}
    for Cv in (0.3, 1.0, 3.0):
        for sw in (0.0, 0.5, 1.0):
            r, _, _ = cv_protocol(train, syn, {"C": Cv, "synthetic_weight": sw}, repeats=(0,), variants={"c": ("candidate", True)})
            a = agg(r["c"]); sens[f"C={Cv} synth_weight={sw}"] = {"macro_f1": a["macro_f1"][0], "complaint_recall": a["complaint_recall"][0], "complaint_precision": a["complaint_precision"][0]}
    results["sensitivity_protocol_A_single_repeat"] = sens

    # ---- save the experimental model (separate from every frozen artifact)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mp = MODEL_DIR / "model.joblib"
    joblib.dump(final, mp)
    cfg = {**final.describe(), "status": "EXPERIMENTAL - NOT ACTIVATED - not imported by any runtime path",
           "trained_on": "455 real train rows + 36 synthetic contrast examples (weight 1.0); validation rows and blind records NOT used",
           "preprocessing": "rcs.classification.intent_candidate.normalize_arabic (mild) / match_form (lexicon matching only)"}
    (MODEL_DIR / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    (MODEL_DIR / "provenance.json").write_text(json.dumps(prov, indent=1))
    code_sha = sha256_file(ROOT / "src" / "rcs" / "classification" / "intent_candidate.py")
    repro = {"command": "python scripts/offline_eval/intent_candidate/train_and_evaluate.py", "seeds_cv": list(SEEDS), "n_splits": N_SPLITS,
             "model_random_state": IC.DEFAULT_PARAMS["random_state"], "python": platform.python_version(), "sklearn": sklearn.__version__,
             "numpy": np.__version__, "joblib": joblib.__version__, "platform": platform.platform(), "intent_candidate_py_sha256": code_sha,
             "train_script_sha256": sha256_file(Path(__file__)), "deterministic_refit": "same data + same code + same versions reproduce the model"}
    (MODEL_DIR / "reproducibility.json").write_text(json.dumps(repro, indent=1))
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"A": results["A"]["CANDIDATE + synthetic"], "B_validation": results["B"]["CANDIDATE + synthetic"]}, indent=1, default=str))
    sums = {f.name: sha256_file(f) for f in sorted(MODEL_DIR.iterdir()) if f.name != "SHA256SUMS"}
    (MODEL_DIR / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    results["model_artifact"] = {"dir": str(MODEL_DIR.relative_to(ROOT)), "model_sha256": sums["model.joblib"], "files": sums}
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    for f in list(MODEL_DIR.iterdir()) + list(OUT.iterdir()):
        f.chmod(0o600)
    print("done; model sha256", sums["model.joblib"])


if __name__ == "__main__":
    main()
