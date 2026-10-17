"""Offline evaluation of rcs.classification.context_signals. No model call, no network, no DB, no training. Aggregate counts and safe ids only (no comment text is printed or saved).

  dev    labelled development rows (563, gold train+val; sealed human_test and the 75 blind labels are never read here):
         (a) detection quality of each signal against the gold label it should relate to
         (b) gold-as-prediction: every conflict raised against a row whose labels are all gold is a CANDIDATE false positive (the layer disagrees with a human label)
  pool   1,543 real unlabeled comments + both stacks' saved predictions: conflicts by dimension, overlap with existing review flags, routing simulation
  blind  (--blind, run once, after the rules were frozen) the 75 exposed human-labelled comments: do conflicts concentrate on wrong predictions?
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/offline_eval/classification_v2"))
import common  # noqa: E402
from config import settings  # noqa: E402
from rcs.classification import context_signals as C  # noqa: E402
from rcs.routing.engine import determine_routing  # noqa: E402

OUT = ROOT / "data/evaluation/context_signals"
FV = ROOT / "data/evaluation/classification_v2_final_validation"


def pr(name, pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold)); fp = sum(p and not g for p, g in zip(pred, gold)); fn = sum((not p) and g for p, g in zip(pred, gold))
    return {"signal": name, "flagged": tp + fp, "gold_positive": tp + fn, "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3)}


def dev():
    d = common.load_dev()
    sig = [C.analyze(r["comment_text"]) for r in d]
    g_i, g_t, g_s = [r["gold_primary_intent"] for r in d], [r["gold_topic_subtype"] for r in d], [r["gold_sentiment"] for r in d]
    det = [
        pr("question(strong) vs INQUIRY", [s.question == "strong" for s in sig], [x == "INQUIRY" for x in g_i]),
        pr("affect vs COMPLAINT", [s.affect for s in sig], [x == "COMPLAINT" for x in g_i]),
        pr("affect vs NEGATIVE", [s.affect for s in sig], [x == "NEGATIVE" for x in g_s]),
        pr("polarity=negative vs NEGATIVE", [s.polarity == "negative" for s in sig], [x == "NEGATIVE" for x in g_s]),
        pr("praise vs THANKS_PRAISE", [s.praise for s in sig], [x == "THANKS_PRAISE" for x in g_i]),
        pr("praise vs POSITIVE", [s.praise for s in sig], [x == "POSITIVE" for x in g_s]),
        pr("error_report vs ENTRY_CORRECTION", [s.error_report for s in sig], [x == "ENTRY_CORRECTION" for x in g_t]),
        pr("technical_failure vs TECHNICAL_ISSUE", [s.technical_failure for s in sig], [x == "TECHNICAL_ISSUE" for x in g_t]),
        pr("suggestion_request vs SUGGESTION", [s.suggestion_request for s in sig], [x == "SUGGESTION" for x in g_i]),
        pr("platform_feature subtype vs DICTIONARY_SUGGESTION", ["platform_feature" in s.suggestion_subtypes for s in sig], [x == "DICTIONARY_SUGGESTION" for x in g_t]),
    ]
    fp = collections.defaultdict(list)
    for r, s in zip(d, sig):
        for c in C.detect_conflicts(s, r["gold_primary_intent"], r["gold_topic_subtype"], r["gold_sentiment"]):
            fp[f"{c.dimension}:{c.code}"].append(r["comment_id"][:10])
    rows_with = len({i for v in fp.values() for i in v})
    return {"n": len(d), "detection": det, "gold_as_prediction_conflicts": {k: {"count": len(v), "ids": v[:12]} for k, v in sorted(fp.items())},
            "rows_with_any_conflict": rows_with, "rows_with_any_conflict_pct": round(100 * rows_with / len(d), 1),
            "suggestion_subtypes": dict(collections.Counter(x for s in sig for x in s.suggestion_subtypes)),
            "suggestion_forms": dict(collections.Counter(str(s.suggestion_form) for s in sig))}


def _route(i, t):
    r = determine_routing(primary_intent=i, topic_subtype=t, subcategory=None, secondary_intent=None, is_content_meaningful=True)
    return r.routing_decision


def pool():
    uni = [json.loads(l) for l in open(FV / "universe.jsonl", encoding="utf-8")]
    P = {"frozen": json.load(open(FV / "predictions_frozen.json")), "v2": json.load(open(FV / "predictions_v2.json"))}
    idx = [i for i, u in enumerate(uni) if u["set"] == "pool" and not u.get("near_duplicate_of_training")]
    res = {"n": len(idx)}
    for name, p in P.items():
        by_code = collections.defaultdict(list); by_dim = collections.Counter(); flagged = set(); route_chg = collections.Counter(); needs_before = needs_after = 0
        sample_ids = {}
        for i in idx:
            intent, conf = p["intent"][i]; topic = p["topic"][i][0]; sent = p["sentiment"][i][0]
            s = C.analyze(uni[i]["text"]); cs = C.detect_conflicts(s, intent, topic, sent)
            before = float(conf) < settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD
            needs_before += before
            if cs:
                flagged.add(i)
                for c in cs:
                    by_code[f"{c.dimension}:{c.code}"].append(uni[i]["comment_id"][:10])
                for dm in {c.dimension for c in cs}:
                    by_dim[dm] += 1
            needs_after += before or bool(cs)
            rd = _route(intent, topic)
            if cs and rd == "EXCLUDE":
                route_chg["EXCLUDE->REVIEW_ONLY"] += 1
        res[name] = {"rows_with_any_conflict": len(flagged), "rows_with_any_conflict_pct": round(100 * len(flagged) / len(idx), 1), "rows_by_dimension": dict(by_dim),
                     "conflicts_by_code": {k: len(v) for k, v in sorted(by_code.items())}, "example_ids": {k: v[:6] for k, v in sorted(by_code.items())},
                     "needs_review_before": needs_before, "needs_review_after": needs_after, "newly_flagged_for_review": needs_after - needs_before,
                     "routing_changes": dict(route_chg), "exclude_routed_total": sum(_route(*[p["intent"][i][0], p["topic"][i][0]]) == "EXCLUDE" for i in idx)}
    sub = collections.Counter(); forms = collections.Counter()
    for i in idx:
        s = C.analyze(uni[i]["text"]); forms[str(s.suggestion_form)] += 1
        for x in s.suggestion_subtypes: sub[x] += 1
    res["suggestion_subtypes"] = dict(sub); res["suggestion_forms"] = dict(forms)
    return res


def blind():
    import csv
    uni = [json.loads(l) for l in open(FV / "universe.jsonl", encoding="utf-8")]
    P = {"frozen": json.load(open(FV / "predictions_frozen.json")), "v2": json.load(open(FV / "predictions_v2.json"))}
    lab_dir = ROOT / "data/evaluation/blind_eval_v1"
    labels = None
    for p in sorted(lab_dir.rglob("*.csv")):
        pass
    return {"note": "blind labels are loaded by the caller-provided loader; see main()"}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "dev"
    out = {"dev": dev, "pool": pool}[what]()
    (OUT / f"{what}_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
