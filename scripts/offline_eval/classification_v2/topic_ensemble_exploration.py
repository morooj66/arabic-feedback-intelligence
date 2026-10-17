"""EXPLORATORY only (validation 108, the split the frozen MARBERT checkpoint was early-stopped on, so it favours the frozen model): average of the
frozen MARBERTv2 saved validation probabilities and the lightweight T9 probabilities. Not usable for selection (single small split, no CV for MARBERT)."""
import json
import numpy as np
from scripts.offline_eval.classification_v2.run_study import ROOT, TASKS, TR, VA, VARIANTS, fit_predict, ids, rows
from scripts.offline_eval.classification_v2.common import OUT, TOPICS, prf

d = json.load(open(ROOT / "data/models/selection/expanded_val_108_2026-08-25/marbertv2_topic/val_probs.json"))
assert d["labels"] == TOPICS
pos = {c: i for i, c in enumerate(d["comment_ids"])}
PM = np.array([d["probabilities"][pos[ids[i]]] for i in VA])
PL = fit_predict("topic", VARIANTS["topic"]["T9 T8 + synthetic(0.5)"], TR, VA)
y = [rows[i]["gold_topic_subtype"] for i in VA]
out = {}
for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    P = w * PM + (1 - w) * PL
    r = prf(y, [TOPICS[k] for k in P.argmax(1)], TOPICS)
    out[f"w_marbert={w}"] = dict(accuracy=r["accuracy"], macro_f1=r["macro_f1"], recall={c: v["recall"] for c, v in r["per_class"].items()})
    print(w, r["accuracy"], r["macro_f1"], out[f"w_marbert={w}"]["recall"])
(OUT / "topic_ensemble_exploration.json").write_text(json.dumps(out, indent=1))
