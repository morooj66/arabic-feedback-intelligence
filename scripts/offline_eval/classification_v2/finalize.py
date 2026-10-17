"""Builds the experimental artifacts for the best candidate of each task, the comparison tables and the decision record.
Reads study_results_full.json + confirmation_results.json; fits the chosen specs on ALL 563 development rows (exclusions apply to training only);
saves under data/models/experimental/classification_v2/<task>/ with config, provenance and SHA256SUMS. Activates nothing."""
import hashlib
import json
import platform
import sys

import joblib
import numpy as np
import sklearn

from scripts.offline_eval.classification_v2.run_study import (CAM_LOG, CONFLICT, E5, FEAT, N, ROOT, TASKS, VARIANTS, excluded_intent, gold_sha, ids, rows, synthetic_for, texts)
from scripts.offline_eval.classification_v2.common import OUT, confusion, prf
from rcs.classification.lexical_classifier import LexicalClassifier

CHOICE = {"intent": "I10 I1 + e5 + CAMeL", "topic": "T9 T8 + synthetic(0.5)", "sentiment": "S5 e5 + CAMeL (no tfidf)"}
FROZEN_NAME = {"intent": "FROZEN config (refit)", "topic": None, "sentiment": "FROZEN CAMeLBERT (inference)"}
full = json.load(open(OUT / "study_results_full.json"))
conf = json.load(open(OUT / "confirmation_results.json"))
ART = ROOT / "data/models/experimental/classification_v2"


def build(task):
    spec = VARIANTS[task][CHOICE[task]]
    labels = TASKS[task][1]; col = TASKS[task][0]
    keep = [i for i in range(N) if not (spec.get("exclude") and excluded_intent[i])]
    tt = [texts[i] for i in keep]; yy = [rows[i][col] for i in keep]
    names = tuple(spec.get("extra", ())); extra = np.hstack([FEAT[n][0][keep] for n in names]) if names else None
    sw = None
    if spec.get("synthetic"):
        st, sy = synthetic_for(task); tt += st; yy += sy; sw = [1.0] * len(keep) + [spec["synthetic"]] * len(st)
        if names: extra = np.vstack([extra, np.hstack([FEAT[n][1] for n in names])])
    m = LexicalClassifier(labels, **spec.get("params", {})).fit(tt, yy, extra=extra, sample_weight=sw)
    d = ART / task; d.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, d / "model.joblib")
    cfg = {"status": "EXPERIMENTAL - NOT ACTIVATED", "task": task, "variant": CHOICE[task], "labels": labels, "extra_features": list(names),
           "classifier": m.describe(), "synthetic_weight": spec.get("synthetic", 0), "policy_exclusions_applied": bool(spec.get("exclude"))}
    prov = {"gold_csv_sha256": gold_sha(), "training_rows": len(keep), "excluded_from_training": (len(CONFLICT) if spec.get("exclude") else 0),
            "synthetic_rows": (len(sy) if spec.get("synthetic") else 0), "feature_extractors": {
                "e5": "intfloat/multilingual-e5-small @ 614241f622f53c4eeff9890bdc4f31cfecc418b3 (passage embeddings, max_length 64)" if "e5" in names else None,
                "camel": "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment (inference only, log-probabilities)" if "camel" in names else None},
            "blind_75_used": False, "sealed_human_test_read": False, "frozen_models_modified": False}
    repro = {"python": sys.version.split()[0], "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__, "random_state": 42,
             "study_seeds_A": [11, 22, 33], "confirmation_seeds_A": [44, 55, 66]}
    r = full[task]["variants"][CHOICE[task]]
    metrics = {"A_macro_f1": r["A"]["macro_f1"], "A_accuracy": r["A"]["accuracy"], "B_macro_f1": r["B"]["macro_f1"], "B_accuracy": r["B"]["accuracy"],
               "note": "metrics come from cross-validation / train->validation runs, NOT from the all-563 refit saved here"}
    for n, o in (("config.json", cfg), ("provenance.json", prov), ("reproducibility.json", repro), ("metrics.json", metrics)):
        (d / n).write_text(json.dumps(o, indent=1, ensure_ascii=False))
    (d / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in ("config.json", "metrics.json", "model.joblib", "provenance.json", "reproducibility.json")))
    return hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()


def fmt_pc(pc, labels):
    return " | ".join(f"{c[:5]} {('-' if pc[c]['recall'] is None else format(pc[c]['recall'], '.2f'))}/{pc[c]['precision']:.2f}" for c in labels)


def flips(task, cand, ref_preds_by_seed, y):
    """records right under the frozen reference but wrong under the candidate (A, first repeat) and the reverse."""
    s = next(iter(ref_preds_by_seed)); a, b = ref_preds_by_seed[s], full[task]["variants"][cand]["A_preds"][s]
    return sum(1 for g, p, q in zip(y, a, b) if p == g and q != g), sum(1 for g, p, q in zip(y, a, b) if p != g and q == g)


def main():
    shas = {t: build(t) for t in CHOICE}
    lines = []
    decision = {}
    for task in ("intent", "topic", "sentiment"):
        labels = TASKS[task][1]; col = TASKS[task][0]; y = [r[col] for r in rows]
        v = full[task]["variants"][CHOICE[task]]; sel = full[task]["selection"].get(CHOICE[task], {})
        refname = FROZEN_NAME[task]
        refA = full[task]["variants"][refname] if refname else None
        refB = (full[task]["variants"][refname] if task == "sentiment" else list(full[task]["frozen_reference"].values())[0]) if (refname or task == "topic") else None
        if task == "intent": refB = full[task]["frozen_reference"]["FROZEN artifact"]
        conf_c = conf[task].get(CHOICE[task], {})
        lines += [f"## {task.capitalize()}: best candidate `{CHOICE[task]}`", ""]
        lines.append(f"- Protocol A (3x5-fold CV, 563 rows): accuracy {v['A']['accuracy']:.3f}, macro-F1 {v['A']['macro_f1']:.3f} (fold sd {v['A']['fold_macro_f1_sd']:.3f}); "
                     + (f"frozen: {refA['A']['accuracy']:.3f} / {refA['A']['macro_f1']:.3f}" if refA else "frozen: not computable offline (base MARBERT weights are not cached; a CV refit would need a download)"))
        if sel and refA:
            lines.append(f"- Paired macro-F1 gain {sel['A_macro_f1_gain']:+.3f} (95% CI {sel['A_macro_f1_gain_ci95'][0]:+.3f}..{sel['A_macro_f1_gain_ci95'][1]:+.3f}); confirmation run with fresh seeds: "
                         f"{conf_c.get('gain_vs_frozen', float('nan')):+.3f} (CI {conf_c.get('gain_ci95', [float('nan')]*2)[0]:+.3f}..{conf_c.get('gain_ci95', [float('nan')]*2)[1]:+.3f}).")
        lines.append(f"- Protocol B (train 455 -> validation 108): accuracy {v['B']['accuracy']:.3f}, macro-F1 {v['B']['macro_f1']:.3f}; frozen {refB['B']['accuracy']:.3f} / {refB['B']['macro_f1']:.3f}")
        lines += ["", "| class | support(A) | frozen R/P (A) | candidate R/P (A) | frozen R/P (B) | candidate R/P (B) |", "|---|---|---|---|---|---|"]
        for c in labels:
            a = v["A"]["per_class"][c]; b = v["B"]["per_class"][c]
            ra = refA["A"]["per_class"][c] if refA else None; rb = refB["B"]["per_class"][c]
            g = lambda d: "-" if d is None or d["recall"] is None else f"{d['recall']:.2f}/{d['precision']:.2f}"
            lines.append(f"| {c} | {a['support']} | {g(ra)} | {g(a)} | {g(rb)} | {g(b)} |")
        lines += ["", "Confusion matrix, candidate, Protocol A (first repeat; rows = gold):", "", "| gold \\ pred | " + " | ".join(c[:6] for c in labels) + " |", "|---|" + "---|" * len(labels)]
        for g_, row in v["A"]["confusion"].items():
            lines.append(f"| {g_[:12]} | " + " | ".join(str(row[p]) for p in labels) + " |")
        if refA:
            w, b = flips(task, CHOICE[task], refA["A_preds"], y)
            lines.append(f"\nRegression risk (A, first repeat): {w} records correct under frozen but wrong under the candidate; {b} the reverse.")
        reg = [c for c in labels if refB["B"]["per_class"][c]["recall"] is not None and (v["B"]["per_class"][c]["recall"] or 0) < refB["B"]["per_class"][c]["recall"] - 0.0001 and refB["B"]["per_class"][c]["support"] >= 3]
        lines.append(f"Classes with lower validation recall than frozen: {', '.join(reg) if reg else 'none'}.")
        lines.append(f"Selection-rule outcome: {'QUALIFIES' if sel.get('qualifies') else ('NOT EVALUABLE against a frozen CV baseline' if not refA else 'does not qualify')}" + (f" - {sel['reasons']}" if sel.get('reasons') else "") + "\n")
        decision[task] = dict(candidate=CHOICE[task], model_sha256=shas[task], qualifies=sel.get("qualifies") if refA else None)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1))
    (OUT / "comparison_tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
