"""Confirmation run: FRESH cross-validation seeds (44, 55, 66) for the finalists chosen from the first study, with no change to any model.
Reduces (does not remove) selection optimism: the same 563 rows are used, but with different fold assignments."""
import json
from scripts.offline_eval.classification_v2.run_study import TASKS, boot_diff, run_protocols
from scripts.offline_eval.classification_v2.common import OUT

FINALISTS = {
    "intent": ["FROZEN config (refit)", "I1 norm word+char", "I9 I1 + e5", "I10 I1 + e5 + CAMeL", "I11 I10 + synthetic(0.5)"],
    "topic": ["T2 norm word+char C=10", "T8 T2 + e5", "T9 T8 + synthetic(0.5)"],
    "sentiment": ["FROZEN CAMeLBERT (inference)", "S1 calibrated (balanced)", "S3 tfidf + CAMeL logprobs", "S5 e5 + CAMeL (no tfidf)", "S6 tfidf + e5 + CAMeL"],
}
REF = {"intent": "FROZEN config (refit)", "sentiment": "FROZEN CAMeLBERT (inference)"}
out = {}
for task, names in FINALISTS.items():
    res, y = run_protocols(task, seeds=(44, 55, 66), names=names)
    labels = TASKS[task][1]
    out[task] = {}
    for n, v in res.items():
        d = dict(A_accuracy=v["A"]["accuracy"], A_macro_f1=v["A"]["macro_f1"], A_macro_f1_by_repeat=v["A"]["macro_f1_by_repeat"], fold_sd=v["A"]["fold_macro_f1_sd"],
                 A_per_class_recall={c: v["A"]["per_class"][c]["recall"] for c in labels})
        if task in REF and n != REF[task]:
            d["gain_vs_frozen"] = v["A"]["macro_f1"] - res[REF[task]]["A"]["macro_f1"]
            d["gain_ci95"] = boot_diff(y, labels, res[REF[task]]["A_preds"], v["A_preds"])
        out[task][n] = d
        print(task, n, round(d["A_macro_f1"], 3), d.get("gain_vs_frozen"), d.get("gain_ci95"))
(OUT / "confirmation_results.json").write_text(json.dumps(out, indent=1))
