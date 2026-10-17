"""Classification v2 study (offline, development data only, no network, no labelling, no Gold writes, nothing activated).

Tasks: intent, topic, sentiment. Identical record splits per task:
  A  repeated stratified 5-fold CV x3 seeds over the 563 development rows (preprocessing/vectorizers fitted inside each training fold)
  B  train 455 -> validation 108 (the split the frozen models were selected on)
Frozen references: intent = frozen config refit (A) + the actual frozen artifact (B); topic = the actual frozen MARBERTv2 validation predictions (B only:
base weights are not cached offline, so a CV refit is impossible without a download); sentiment = pretrained CAMeLBERT (inference only; A and B).
Original gold labels are never changed. Policy-conflicting rows are excluded from TRAINING only (recorded), never from evaluation.
Synthetic rows (36 proposed candidates) are training-only options and never appear in any evaluation.
Selection rule is fixed in `qualifies()` below before results are looked at."""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from scripts.offline_eval.classification_v2.common import *  # noqa: F401,F403
from scripts.offline_eval.classification_v2.common import ROOT, OUT, TASKS, INTENTS, TOPICS, SENTIMENTS, confusion, cv_splits, gold_sha, load_dev, prf
from rcs.classification.arabic_text import has_question, normalize_arabic
from rcs.classification.lexical_classifier import LexicalClassifier
import csv
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

rows = load_dev()
N = len(rows)
ids = [r["comment_id"] for r in rows]
texts = [r["comment_text"] for r in rows]
is_train = np.array([r["split"] == "train" for r in rows])
TR, VA = np.where(is_train)[0], np.where(~is_train)[0]

# ---------------------------------------------------------------- policy-conflict rows (excluded from TRAINING only, recorded)
_DEF = re.compile(r"(?:^|\s)(?:معناها|معناه|يعني|تعني|بمعني|بمعنى|المعني|المعنى)(?:\s|:|$)")


def intent_policy_conflict(r) -> str | None:
    """Approved policy: 'a bare word followed by its meaning, with no question, is NOT an inquiry' (it is a word suggestion), and spam is not a
    dictionary-style word+meaning entry. Rows labelled INQUIRY or SPAM that have NO question cue and a meaning marker contradict it; the form is
    labelled three different ways across annotation rounds. They are excluded from candidate training (not relabelled)."""
    if r["gold_primary_intent"] in ("INQUIRY", "SPAM") and not has_question(r["comment_text"]) and _DEF.search(normalize_arabic(r["comment_text"])):
        return "R-INTENT-DEF: no question cue + meaning marker, labelled " + r["gold_primary_intent"]
    return None


CONFLICT = {r["comment_id"]: intent_policy_conflict(r) for r in rows}
CONFLICT = {k: v for k, v in CONFLICT.items() if v}
excluded_intent = np.array([i in CONFLICT for i in ids])

# ---------------------------------------------------------------- frozen sentiment probabilities (cached local inference, no labels used)
cam = json.load(open(OUT / "camelbert_probs_dev563.json"))
assert cam["comment_ids"] == ids
CAM = np.array(cam["probabilities"])                       # columns: POSITIVE, NEGATIVE, NEUTRAL
CAM_LOG = np.log(np.clip(CAM, 1e-6, 1))

E5 = np.load(OUT / "e5_small_dev563.npy")
E5_SYN = np.load(OUT / "e5_small_synthetic36.npy")
CAM_SYN_LOG = np.log(np.clip(np.load(OUT / "camelbert_probs_synthetic36.npy"), 1e-6, 1))
FEAT = {"camel": (CAM_LOG, CAM_SYN_LOG), "e5": (E5, E5_SYN)}          # frozen, label-free features: (development rows, synthetic rows)

# ---------------------------------------------------------------- synthetic (training-only option)
syn = list(csv.DictReader(open(ROOT / "data/evaluation/error_analysis/synthetic_augmentation_candidates.csv", encoding="utf-8")))


def synthetic_for(task):
    col = {"intent": "proposed_intent", "topic": "proposed_topic", "sentiment": "proposed_sentiment"}[task]
    return [s["text"] for s in syn], [s[col] for s in syn]


# ---------------------------------------------------------------- variants
def frozen_intent_refit():
    return dict(vec=lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True),
                clf=lambda: LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"))


def fit_predict(task, spec, tr, te):
    """-> (n_test, n_labels) probabilities. `tr` indices already filtered by the caller for exclusions."""
    labels = TASKS[task][1]
    col = TASKS[task][0]
    y = np.array([rows[i][col] for i in range(N)])
    kind = spec["kind"]
    if kind == "camel":                                        # frozen pretrained inference
        return CAM[te]
    tt = [texts[i] for i in tr]; yy = list(y[tr]); sw = None
    names = spec.get("extra", ())
    extra_tr = np.hstack([FEAT[n][0][tr] for n in names]) if names else None
    extra_te = np.hstack([FEAT[n][0][te] for n in names]) if names else None
    if spec.get("synthetic"):
        st, sy = synthetic_for(task)
        tt, yy = tt + st, yy + sy
        sw = [1.0] * len(tr) + [spec["synthetic"]] * len(st)
        if names:
            extra_tr = np.vstack([extra_tr, np.hstack([FEAT[n][1] for n in names])])
    if kind == "frozen_refit":
        f = frozen_intent_refit(); v = f["vec"](); X = v.fit_transform(tt); c = f["clf"]().fit(X, yy)
        raw = c.predict_proba(v.transform([texts[i] for i in te]))
        out = np.zeros((len(te), len(labels)))
        for j, lab in enumerate(labels):
            if lab in c.classes_:
                out[:, j] = raw[:, list(c.classes_).index(lab)]
        return out
    if kind == "camel_calibrated":                              # multinomial LR on the pretrained model's own log-probabilities (+ length, diacritics)
        from rcs.classification.lexical_classifier import text_feature_block
        A = np.hstack([CAM_LOG[tr], text_feature_block([texts[i] for i in tr])[:, 1:]]); B = np.hstack([CAM_LOG[te], text_feature_block([texts[i] for i in te])[:, 1:]])
        c = LogisticRegression(C=spec.get("C", 1.0), class_weight=spec.get("class_weight", "balanced"), max_iter=2000, random_state=42).fit(A, list(y[tr]))
        raw = c.predict_proba(B); out = np.zeros((len(te), len(labels)))
        for j, lab in enumerate(labels):
            if lab in c.classes_:
                out[:, j] = raw[:, list(c.classes_).index(lab)]
        return out
    m = LexicalClassifier(labels, **spec.get("params", {})).fit(tt, yy, extra=(None if extra_tr is None else extra_tr), sample_weight=sw)
    return m.predict_proba([texts[i] for i in te], extra=extra_te)


VARIANTS = {
    "intent": {
        "FROZEN config (refit)": dict(kind="frozen_refit"),
        "I1 norm word+char": dict(kind="lex", params=dict(C=1.0)),
        "I2 I1 + policy exclusions": dict(kind="lex", params=dict(C=1.0), exclude=True),
        "I3 I2 + question/length features": dict(kind="lex", params=dict(C=1.0, text_features=True), exclude=True),
        "I4 I3 + synthetic(0.5)": dict(kind="lex", params=dict(C=1.0, text_features=True), exclude=True, synthetic=0.5),
        "I5 I3 C=3": dict(kind="lex", params=dict(C=3.0, text_features=True), exclude=True),
        "I6 I1 + question/length (no exclusion)": dict(kind="lex", params=dict(C=1.0, text_features=True)),
        "I7 I1 + CAMeL logprobs": dict(kind="lex", params=dict(C=1.0), extra=("camel",)),
        "I8 e5 embeddings only": dict(kind="lex", params=dict(C=3.0, use_tfidf=False, text_features=True, dense_scale=1.0), extra=("e5",)),
        "I9 I1 + e5": dict(kind="lex", params=dict(C=1.0, dense_scale=2.0), extra=("e5",)),
        "I10 I1 + e5 + CAMeL": dict(kind="lex", params=dict(C=1.0, dense_scale=2.0), extra=("e5", "camel")),
        "I11 I10 + synthetic(0.5)": dict(kind="lex", params=dict(C=1.0, dense_scale=2.0), extra=("e5", "camel"), synthetic=0.5),
    },
    "topic": {
        "T1 norm word+char C=1": dict(kind="lex", params=dict(C=1.0)),
        "T2 norm word+char C=10": dict(kind="lex", params=dict(C=10.0)),
        "T3 word only C=10": dict(kind="lex", params=dict(C=10.0, use_char=False)),
        "T4 T2 + question/length": dict(kind="lex", params=dict(C=10.0, text_features=True)),
        "T5 T2 + synthetic(0.5)": dict(kind="lex", params=dict(C=10.0), synthetic=0.5),
        "T6 T1 unweighted": dict(kind="lex", params=dict(C=10.0, class_weight=None)),
        "T7 e5 embeddings only": dict(kind="lex", params=dict(C=3.0, use_tfidf=False, text_features=True), extra=("e5",)),
        "T8 T2 + e5": dict(kind="lex", params=dict(C=10.0, dense_scale=2.0), extra=("e5",)),
        "T9 T8 + synthetic(0.5)": dict(kind="lex", params=dict(C=10.0, dense_scale=2.0), extra=("e5",), synthetic=0.5),
    },
    "sentiment": {
        "FROZEN CAMeLBERT (inference)": dict(kind="camel"),
        "S1 calibrated (balanced)": dict(kind="camel_calibrated", class_weight="balanced"),
        "S1b calibrated (unweighted)": dict(kind="camel_calibrated", class_weight=None),
        "S2 tfidf LR": dict(kind="lex", params=dict(C=3.0)),
        "S3 tfidf + CAMeL logprobs": dict(kind="lex", params=dict(C=3.0), extra=("camel",)),
        "S4 S2 + synthetic(0.5)": dict(kind="lex", params=dict(C=3.0), synthetic=0.5),
        "S5 e5 + CAMeL (no tfidf)": dict(kind="lex", params=dict(C=3.0, use_tfidf=False, text_features=True), extra=("e5", "camel")),
        "S6 tfidf + e5 + CAMeL": dict(kind="lex", params=dict(C=3.0, dense_scale=2.0), extra=("e5", "camel")),
        "S7 S6 + synthetic(0.5)": dict(kind="lex", params=dict(C=3.0, dense_scale=2.0), extra=("e5", "camel"), synthetic=0.5),
        "S8 S3 C=1": dict(kind="lex", params=dict(C=1.0), extra=("camel",)),
    },
}


def train_idx(spec, base):
    if spec.get("exclude"):
        return np.array([i for i in base if not excluded_intent[i]])
    return base


def run_protocols(task, seeds=(11, 22, 33), names=None):
    labels = TASKS[task][1]; col = TASKS[task][0]
    y = np.array([r[col] for r in rows])
    splits = cv_splits(y, seeds=seeds)
    res = {}
    for name, spec in VARIANTS[task].items():
        if names is not None and name not in names:
            continue
        oof = {s: np.zeros((N, len(labels))) for s in {s[0] for s in splits}}
        fold_mf1 = []
        for seed, fold, tr, te in splits:
            P = fit_predict(task, spec, train_idx(spec, tr), te)
            oof[seed][te] = P
            fold_mf1.append(prf(list(y[te]), [labels[k] for k in P.argmax(1)], labels)["macro_f1"])
        preds = {s: [labels[k] for k in oof[s].argmax(1)] for s in oof}
        A = {s: prf(list(y), preds[s], labels) for s in preds}
        PB = fit_predict(task, spec, train_idx(spec, TR), VA)
        predB = [labels[k] for k in PB.argmax(1)]
        res[name] = dict(
            A_preds={str(s): preds[s] for s in preds},
            A=dict(accuracy=float(np.mean([a["accuracy"] for a in A.values()])), macro_f1=float(np.mean([a["macro_f1"] for a in A.values()])),
                   macro_f1_by_repeat=[a["macro_f1"] for a in A.values()], fold_macro_f1_mean=float(np.mean(fold_mf1)), fold_macro_f1_sd=float(np.std(fold_mf1)),
                   per_class={c: dict(precision=float(np.mean([A[s]["per_class"][c]["precision"] for s in A])), recall=(None if A[next(iter(A))]["per_class"][c]["recall"] is None else float(np.mean([A[s]["per_class"][c]["recall"] for s in A]))),
                                      f1=float(np.mean([A[s]["per_class"][c]["f1"] or 0 for s in A])), support=A[next(iter(A))]["per_class"][c]["support"]) for c in labels},
                   confusion=confusion(list(y), preds[next(iter(preds))], labels)),
            B_pred=predB, B=prf(list(y[VA]), predB, labels), B_confusion=confusion(list(y[VA]), predB, labels))
    return res, y


def frozen_references(task, y):
    labels = TASKS[task][1]
    if task == "intent":
        pipe = joblib.load(ROOT / "data/models/final_test/intent_tfidf_logistic.joblib")
        pr = list(pipe.predict([texts[i] for i in VA]))
        return {"FROZEN artifact": dict(B_pred=pr, B=prf(list(y[VA]), pr, labels), B_confusion=confusion(list(y[VA]), pr, labels))}
    if task == "topic":
        d = json.load(open(ROOT / "data/models/selection/expanded_val_108_2026-08-25/marbertv2_topic/val_probs.json"))
        pos = {c: i for i, c in enumerate(d["comment_ids"])}
        pr = [d["y_pred"][pos[ids[i]]] for i in VA]
        return {"FROZEN MARBERTv2 (saved val predictions)": dict(B_pred=pr, B=prf(list(y[VA]), pr, labels), B_confusion=confusion(list(y[VA]), pr, labels))}
    return {}


def boot_diff(y, labels, pa, pb, n=1000, seed=7):
    """Paired record bootstrap of macro-F1(b) - macro-F1(a), averaged over the repeats of A (pa/pb: {seed: preds})."""
    rng = np.random.RandomState(seed); y = np.asarray(y); out = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        d = [prf(list(y[idx]), list(np.asarray(pb[s])[idx]), labels)["macro_f1"] - prf(list(y[idx]), list(np.asarray(pa[s])[idx]), labels)["macro_f1"] for s in pa]
        out.append(np.mean(d))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def qualifies(cand, ref, refA, refB, labels, ci):
    """FIXED selection rule. A candidate replaces the frozen model only if ALL hold:
    1. protocol A (when a frozen reference exists): paired macro-F1 gain with 95% CI lower bound > 0
    2. protocol B: macro-F1 not below the frozen model's by more than 0.01
    3. no class with enough support loses more than 0.10 recall vs frozen (A: support>=15, B: support>=8)
    Returns (ok, reasons)."""
    why = []
    if refA is not None:
        if not (ci[0] > 0): why.append(f"A macro-F1 gain CI lower bound {ci[0]:+.3f} <= 0")
    if cand["B"]["macro_f1"] < refB["B"]["macro_f1"] - 0.01: why.append(f"B macro-F1 {cand['B']['macro_f1']:.3f} < frozen {refB['B']['macro_f1']:.3f} - 0.01")
    for c in labels:
        pb, rb = cand["B"]["per_class"][c], refB["B"]["per_class"][c]
        if rb["support"] >= 8 and rb["recall"] is not None and (pb["recall"] or 0) < rb["recall"] - 0.10: why.append(f"B recall drop on {c}: {rb['recall']:.2f}->{(pb['recall'] or 0):.2f}")
        if refA is not None:
            pa, ra = cand["A"]["per_class"][c], refA["A"]["per_class"][c]
            if ra["support"] >= 15 and ra["recall"] is not None and (pa["recall"] or 0) < ra["recall"] - 0.10: why.append(f"A recall drop on {c}: {ra['recall']:.2f}->{(pa['recall'] or 0):.2f}")
    return (not why), why


def main():
    results = {"meta": dict(gold_sha256=gold_sha(), n_dev=N, n_train=int(len(TR)), n_val=int(len(VA)), policy_conflict_rows_excluded_from_training=len(CONFLICT),
                            blind_75_used=False, sealed_human_test_read=False, synthetic_in_evaluation=False)}
    exc = [{"comment_id": k, "rule": v, "original_gold_intent": next(r["gold_primary_intent"] for r in rows if r["comment_id"] == k)} for k, v in sorted(CONFLICT.items())]
    with open(OUT / "training_exclusions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["comment_id", "rule", "original_gold_intent"]); w.writeheader(); w.writerows(exc)
    for task in ("intent", "topic", "sentiment"):
        labels = TASKS[task][1]
        res, y = run_protocols(task)
        refs = frozen_references(task, y)
        results[task] = dict(variants=res, frozen_reference=refs)
        # frozen reference selection for comparison
        if task == "intent": refA_name, refB = "FROZEN config (refit)", refs["FROZEN artifact"]
        elif task == "sentiment": refA_name, refB = "FROZEN CAMeLBERT (inference)", res["FROZEN CAMeLBERT (inference)"]
        else: refA_name, refB = None, refs["FROZEN MARBERTv2 (saved val predictions)"]
        refA = res[refA_name] if refA_name else None
        sel = {}
        for name, v in res.items():
            if name == refA_name: continue
            ci = boot_diff(y, labels, refA["A_preds"], v["A_preds"]) if refA else [None, None]
            ok, why = qualifies(v, refA, refA, refB, labels, ci if refA else [1, 1])
            sel[name] = dict(qualifies=ok, reasons=why, A_macro_f1_gain_ci95=ci,
                             A_macro_f1_gain=(v["A"]["macro_f1"] - refA["A"]["macro_f1"]) if refA else None, B_macro_f1_gain=v["B"]["macro_f1"] - refB["B"]["macro_f1"])
        results[task]["selection"] = sel
        results[task]["frozen_reference_name_A"] = refA_name
    for t in ("intent", "topic", "sentiment"):                      # strip bulky per-record predictions from the summary copy, keep them in a side file
        pass
    (OUT / "study_results_full.json").write_text(json.dumps(results, ensure_ascii=False))
    slim = json.loads(json.dumps(results))
    for t in ("intent", "topic", "sentiment"):
        for v in slim[t]["variants"].values():
            v.pop("A_preds", None); v.pop("B_pred", None)
        for v in slim[t]["frozen_reference"].values():
            v.pop("B_pred", None)
    (OUT / "study_results.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1))
    print("written")


if __name__ == "__main__":
    main()
