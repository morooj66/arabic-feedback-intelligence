"""Paired Jev v1 vs v2 comparison on the unseen development holdout (pure functions + CLI; no network).

DECISION CRITERIA ARE PRE-REGISTERED HERE, BEFORE ANY v2 PREDICTION EXISTS, so v2 cannot be judged after the fact:

  C1  topic boundary   : delta recall(WORD_SUBMISSION) >= +0.15  AND  delta recall(ENTRY_CORRECTION) >= -0.15
  C2  complaint        : delta precision(COMPLAINT) >= +0.05     AND  delta recall(COMPLAINT) >= -0.10
  C3  no broad regress.: for intent and topic, the 95% paired-bootstrap LOWER bound of the accuracy delta and of
                         the macro-F1 delta is >= -0.05
  C4  noise control    : sentiment wording is IDENTICAL in v1 and v2, so any sentiment flips measure run-to-run
                         noise; changes smaller than that flip rate are not interpretable

  Verdict:  REJECT        if C3 fails
            PROMISING     if C1 and C3 hold and (C2 holds or COMPLAINT true positives did not fall)
            INCONCLUSIVE  otherwise
  PROMISING only means "worth an independent test on Gold Test v2"; it is NOT production accuracy and never
  authorises a prompt or model change by itself. TECHNICAL_ISSUE cannot be evaluated here (holdout support 0).

Run (after both prediction files exist):  python scripts/offline_eval/jev_dev_validation/compare_v1_v2.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_common  # noqa: E402,F401
import jev_metrics as M  # noqa: E402
from jev_common import V2_HOLDOUT_DIR  # noqa: E402

TASKS = ("intent", "topic", "sentiment")
TARGETS = (("topic", "WORD_SUBMISSION"), ("topic", "ENTRY_CORRECTION"), ("topic", "DICTIONARY_SUGGESTION"),
           ("intent", "COMPLAINT"), ("topic", "TECHNICAL_ISSUE"), ("sentiment", "NEGATIVE"))
C1_WS, C1_EC, C2_P, C2_R, C3_LB = 0.15, -0.15, 0.05, -0.10, -0.05


def _paired_boot(gold, a, b, labels, n_boot=2000, seed=20261015):
    rng = random.Random(seed)
    n = len(gold)
    da, df = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        pa, pb = [a[i] for i in idx], [b[i] for i in idx]
        da.append(M.accuracy(g, pb) - M.accuracy(g, pa))
        df.append(M.macro_f1(M.per_class(g, pb, labels), only_supported=True)
                  - M.macro_f1(M.per_class(g, pa, labels), only_supported=True))
    da.sort(); df.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return (da[lo], da[hi]), (df[lo], df[hi])


def compare(gold: dict, p1: dict, p2: dict) -> dict:
    ids = [i for i in gold if i in p1 and i in p2]
    out: dict = {"n": len(ids), "missing": len(gold) - len(ids), "tasks": {}, "targets": {}, "flips": {}}
    for t in TASKS:
        labels = M.TASK_LABELS[t]
        g = [gold[i][t] for i in ids]
        a = [p1[i][t] for i in ids]
        b = [p2[i][t] for i in ids]
        s1, s2 = M.score_task(t, g, a), M.score_task(t, g, b)
        (alo, ahi), (flo, fhi) = _paired_boot(g, a, b, labels)
        out["tasks"][t] = {
            "acc_v1": s1["accuracy"], "acc_v2": s2["accuracy"], "f1_v1": s1["macro_f1_supported_classes"],
            "f1_v2": s2["macro_f1_supported_classes"], "d_acc_ci": (alo, ahi), "d_f1_ci": (flo, fhi),
            "per_class_v1": s1["per_class"], "per_class_v2": s2["per_class"],
        }
        regress = [i for i, x, y, z in zip(ids, g, a, b) if y == x and z != x]
        gain = [i for i, x, y, z in zip(ids, g, a, b) if y != x and z == x]
        changed = [i for i, y, z in zip(ids, a, b) if y != z]
        out["flips"][t] = {"v1_right_v2_wrong": regress, "v1_wrong_v2_right": gain, "changed": len(changed),
                           "changed_rate": len(changed) / len(ids) if ids else 0.0}
    for t, cls in TARGETS:
        v1, v2 = out["tasks"][t]["per_class_v1"][cls], out["tasks"][t]["per_class_v2"][cls]
        out["targets"][f"{t}:{cls}"] = {"support": v1["support"], "v1": v1, "v2": v2,
                                       "d_recall": v2["recall"] - v1["recall"], "d_precision": v2["precision"] - v1["precision"]}
    gt = [gold[i]["topic"] for i in ids]
    for name, p in (("v1", p1), ("v2", p2)):
        sub = [(g, p[i]["topic"]) for g, i in zip(gt, ids) if g in M.SUGGESTION_SUBTYPES]
        out[f"suggestion_confusion_{name}"] = M.confusion([g for g, _ in sub], [q for _, q in sub],
                                                          M.SUGGESTION_SUBTYPES, M.SUGGESTION_SUBTYPES)
    comp = {}
    for name, p in (("v1", p1), ("v2", p2)):
        flag = [p[i]["intent"] == "COMPLAINT" for i in ids]
        comp[name] = {"flagged": sum(flag), "tp": sum(1 for g, f in zip([gold[i]["intent"] for i in ids], flag) if f and g == "COMPLAINT"),
                      "support": sum(1 for i in ids if gold[i]["intent"] == "COMPLAINT")}
    out["complaint_flags"] = comp
    out["verdict"] = verdict(out)
    return out


def verdict(res: dict) -> dict:
    tg = res["targets"]
    ws, ec = tg["topic:WORD_SUBMISSION"], tg["topic:ENTRY_CORRECTION"]
    c1 = ws["d_recall"] >= C1_WS and ec["d_recall"] >= C1_EC
    cm = tg["intent:COMPLAINT"]
    c2 = cm["d_precision"] >= C2_P and cm["d_recall"] >= C2_R
    c3 = all(res["tasks"][t]["d_acc_ci"][0] >= C3_LB and res["tasks"][t]["d_f1_ci"][0] >= C3_LB for t in ("intent", "topic"))
    tp_ok = res["complaint_flags"]["v2"]["tp"] >= res["complaint_flags"]["v1"]["tp"]
    if not c3:
        v = "REJECT"
    elif c1 and (c2 or tp_ok):
        v = "PROMISING"
    else:
        v = "INCONCLUSIVE"
    return {"C1_topic_boundary": c1, "C2_complaint": c2, "C3_no_broad_regression": c3, "complaint_tp_not_lower": tp_ok,
            "C4_sentiment_noise_flip_rate": res["flips"]["sentiment"]["changed_rate"], "verdict": v,
            "note": "PROMISING = worth an independent Gold Test v2 evaluation only; not production accuracy."}


def to_markdown(res: dict) -> str:
    L = [f"# Jev v1 vs v2 on the unseen development holdout (n={res['n']}; missing predictions {res['missing']})", "",
         "| Task | v1 acc | v2 acc | delta acc 95% CI | v1 macro-F1 | v2 macro-F1 | delta macro-F1 95% CI | v1 right -> v2 wrong | v1 wrong -> v2 right | rows changed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        r, f = res["tasks"][t], res["flips"][t]
        L.append(f"| {t} | {r['acc_v1']:.3f} | {r['acc_v2']:.3f} | {r['d_acc_ci'][0]:+.2f}..{r['d_acc_ci'][1]:+.2f} | {r['f1_v1']:.3f} | "
                 f"{r['f1_v2']:.3f} | {r['d_f1_ci'][0]:+.2f}..{r['d_f1_ci'][1]:+.2f} | {len(f['v1_right_v2_wrong'])} | "
                 f"{len(f['v1_wrong_v2_right'])} | {f['changed']} ({f['changed_rate']:.0%}) |")
    L += ["", "Sentiment wording is identical in v1 and v2, so its 'rows changed' rate is the run-to-run noise floor (C4).", "",
          "## Targeted classes", "", "| class | support | recall v1 -> v2 | precision v1 -> v2 | F1 v1 -> v2 |", "|---|---|---|---|---|"]
    for key, v in res["targets"].items():
        if v["support"] == 0:
            L.append(f"| {key} | 0 | not evaluable (no gold rows in the holdout) | | |")
            continue
        L.append(f"| {key} | {v['support']} | {v['v1']['recall']:.2f} -> {v['v2']['recall']:.2f} | {v['v1']['precision']:.2f} -> "
                 f"{v['v2']['precision']:.2f} | {v['v1']['f1']:.2f} -> {v['v2']['f1']:.2f} |")
    for name in ("v1", "v2"):
        m = res[f"suggestion_confusion_{name}"]
        cols = list(next(iter(m.values())).keys())
        L += ["", f"Suggestion-subtype confusion {name} (rows gold, cols predicted)", "", "| gold | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for g, row in m.items():
            L.append(f"| {g} | " + " | ".join(str(row[c]) for c in cols) + " |")
    cf = res["complaint_flags"]
    L += ["", f"COMPLAINT flags: v1 flagged {cf['v1']['flagged']} (TP {cf['v1']['tp']}/{cf['v1']['support']}); "
          f"v2 flagged {cf['v2']['flagged']} (TP {cf['v2']['tp']}/{cf['v2']['support']}).", "", "## Pre-registered verdict", ""]
    for k, v in res["verdict"].items():
        L.append(f"- {k}: {v}")
    return "\n".join(L) + "\n"


def _load_preds(path: Path) -> dict:
    return {r["eval_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> None:
    d = V2_HOLDOUT_DIR
    gold = json.loads((d / "gold_labels.local.json").read_text())
    res = compare(gold, _load_preds(d / "jev_v1_predictions.jsonl"), _load_preds(d / "jev_v2_predictions.jsonl"))
    (d / "v1_vs_v2_report.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))


if __name__ == "__main__":
    main()
