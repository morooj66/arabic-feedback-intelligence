"""FINAL QUALITY VALIDATION (offline, read-only): frozen baseline vs the selected v2 pilot (Intent I10, frozen MARBERTv2 topic, Sentiment S5) on the same real
visitor comments. Nothing is trained, written to a database, sent to Gemini or activated. Results go to data/evaluation/classification_v2_final_validation/.

  python -m scripts.offline_eval.classification_v2.final_validation prepare
  python -m scripts.offline_eval.classification_v2.final_validation run          (orchestrates predict --stack frozen / v2 in fresh subprocesses, then analyses)
Data: the eligible-comment universe built by the blind-evaluation tooling (prefilter, duplicates and privacy policy applied; sealed human_test, Gold/dev ids and
candidate-round ids excluded by id, dev texts excluded by text). The 75 exposed blind comments are kept apart and used only as a clearly-labelled secondary check."""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.offline_eval.classification_v2.common import ROOT, load_dev, prf, confusion, INTENTS, TOPICS, SENTIMENTS  # noqa: F401  (sets sys.path)

OUT = ROOT / "data" / "evaluation" / "classification_v2_final_validation"
OUT.mkdir(parents=True, exist_ok=True)
THRESH = 0.45                                      # settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD - NOT changed by this test


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


# ------------------------------------------------------------------------------------------------------------------------ prepare
def prepare():
    sys.path.insert(0, str(ROOT / "scripts" / "offline_eval" / "blind_eval"))
    import build_sample as BS                                                    # the existing, privacy-aware universe builder
    from rcs.classification.arabic_text import normalize_arabic
    from sklearn.feature_extraction.text import TfidfVectorizer

    log = {}
    universe = BS.build_universe(log)
    blind = {r["comment_id"]: r for r in csv.DictReader(open(ROOT / "data/evaluation/blind_eval_v1/coordinator_only/blind_eval_mapping.local.csv", encoding="utf-8"))}
    rows = []
    for u in universe:
        comp = blind[u["sid"]]["component"] if u["sid"] in blind else "pool"
        rows.append({"comment_id": u["sid"], "text": u["text"], "set": comp, "blind_record_id": blind[u["sid"]]["record_id"] if u["sid"] in blind else None})
    missing_blind = [k for k in blind if k not in {r["comment_id"] for r in rows}]
    # overlap with everything the v2 candidates were trained on (563 development rows + 36 synthetic proposals)
    dev = [normalize_arabic(r["comment_text"]) for r in load_dev()]
    syn = [normalize_arabic(s["text"]) for s in csv.DictReader(open(ROOT / "data/evaluation/error_analysis/synthetic_augmentation_candidates.csv", encoding="utf-8"))]
    train_texts = dev + syn
    exact = sum(1 for r in rows if normalize_arabic(r["text"]) in set(train_texts))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(train_texts + [normalize_arabic(r["text"]) for r in rows])
    A = vec.transform(train_texts); B = vec.transform([normalize_arabic(r["text"]) for r in rows])
    sims = (B @ A.T).max(axis=1).toarray().ravel()
    overlap = {"exact_normalized_text_matches_with_training": exact, "near_duplicates_char_cosine_ge_0.90": int((sims >= 0.90).sum()),
               "near_duplicates_char_cosine_ge_0.80": int((sims >= 0.80).sum()), "max_cosine": round(float(sims.max()), 3)}
    flag_near = {rows[i]["comment_id"] for i in np.where(sims >= 0.90)[0]}
    for r in rows:
        r["near_duplicate_of_training"] = r["comment_id"] in flag_near
    with open(OUT / "universe.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # real blank / malformed comments from the raw export (what the prefilter removes upstream) for the robustness check
    import datetime as dt
    from config import settings
    from rcs.ingest.cleaning import clean
    from rcs.ingest.loader import load_raw
    from rcs.prefilter.spam_rules import BurstConfig, apply_prefilter
    raw = load_raw(ROOT / "data/raw/historical-export.xlsx", settings.SOURCE_COLUMNS)
    df = apply_prefilter(clean(raw, dt.datetime(2026, 10, 16), settings.AGE_RECENT_MAX_DAYS, settings.AGE_DELAYED_MAX_DAYS), settings.KNOWN_JUNK_STRINGS,
                         BurstConfig(min_count=settings.BURST_MIN_COUNT, window_hours=settings.BURST_WINDOW_HOURS))
    bad = df[df["prefilter_reason"].astype(str).isin(["null_content", "near_empty_content"])]
    malformed = [{"comment_id": r.comment_id, "text": ("" if r.original_comment is None or str(r.original_comment) in ("nan", "None") else str(r.original_comment)),
                  "reason": r.prefilter_reason} for r in bad.itertuples()]
    json.dump(malformed, open(OUT / "real_blank_or_malformed.json", "w", encoding="utf-8"), ensure_ascii=False)
    summary = {"data_source": "data/raw/historical-export.xlsx (real visitor comments) -> eligible universe from the blind-evaluation builder",
               "universe_log": log, "unique_comments_total": len(rows), "by_set": dict(Counter(r["set"] for r in rows)),
               "blind_comments_not_found_in_universe": len(missing_blind), "reference_labels": {"unlabeled_pool": False, "exposed_blind_75": True},
               "overlap_with_training": overlap, "real_blank_or_malformed_rows": len(malformed)}
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------------------------------ predict
def predict(stack: str):
    from config import settings
    settings.CLASSIFICATION_MODE = "local_v2_pilot" if stack == "v2" else "legacy_rcs"
    from rcs.classification import runtime
    rows = [json.loads(l) for l in open(OUT / "universe.jsonl", encoding="utf-8")]
    ids, texts = [r["comment_id"] for r in rows], [r["text"] for r in rows]
    malformed = json.load(open(OUT / "real_blank_or_malformed.json", encoding="utf-8"))
    m_ids, m_texts = [f"m{i}" for i in range(len(malformed))], [m["text"] for m in malformed]
    t = {}
    t0 = time.perf_counter()
    if stack == "v2":
        from rcs.classification.v2_stack import get_stack
        get_stack().load()
        t["cold_load_s"] = round(time.perf_counter() - t0, 2)
    out = {}
    for name, fn in (("intent", lambda: runtime.predict_intent(texts)), ("topic", lambda: runtime.predict_topic(ids, texts)),
                     ("sentiment", lambda: runtime.predict_sentiment(ids, texts))):
        s = time.perf_counter(); out[name] = fn(); t[name + "_s"] = round(time.perf_counter() - s, 2)
    t["total_s_including_load"] = round(time.perf_counter() - t0, 2)
    t["peak_rss_mb"] = rss_mb()
    # real blank / malformed rows (normally removed by the prefilter) through the same runtime
    errs = []
    mal = {}
    try:
        mal = {"intent": runtime.predict_intent(m_texts), "topic": runtime.predict_topic(m_ids, m_texts), "sentiment": runtime.predict_sentiment(m_ids, m_texts)}
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        errs.append(f"{type(exc).__name__}: {str(exc)[:200]}")
    res = {"stack": stack, "mode": settings.CLASSIFICATION_MODE, "n": len(rows), "timing": t, "runtime_errors": errs,
           "intent": [[a, float(b)] for a, b in out["intent"]],
           "topic": [[out["topic"][i][0], None if out["topic"][i][1] is None else float(out["topic"][i][1])] for i in ids],
           "sentiment": [[out["sentiment"][i][0], float(out["sentiment"][i][1])] for i in ids],
           "malformed": ({k: ([[a, float(b)] for a, b in v] if k == "intent" else [[v[i][0], None if v[i][1] is None else float(v[i][1])] for i in m_ids]) for k, v in mal.items()} if mal else None),
           "malformed_reasons": [m["reason"] for m in malformed], "malformed_n": len(malformed)}
    (OUT / f"predictions_{stack}.json").write_text(json.dumps(res))
    print(stack, json.dumps(t))


# ------------------------------------------------------------------------------------------------------------------------ analyse
def analyse():
    from rcs.response_generation.eligibility import EligibilityInput, determine_response_eligibility
    from rcs.review.priority import compute_priority
    from rcs.routing.engine import determine_routing
    from rcs.classification.arabic_text import has_question
    sys.path.insert(0, str(ROOT / "scripts/offline_eval/error_analysis"))
    import common as EA

    rows = [json.loads(l) for l in open(OUT / "universe.jsonl", encoding="utf-8")]
    F = json.load(open(OUT / "predictions_frozen.json")); V = json.load(open(OUT / "predictions_v2.json"))
    n = len(rows); assert F["n"] == V["n"] == n
    pool = [i for i, r in enumerate(rows) if r["set"] == "pool" and not r["near_duplicate_of_training"]]   # near-duplicates of training rows are reported, not analysed
    res = {"n_total": n, "n_pool_unlabeled": len(pool), "near_duplicates_of_training_excluded_from_pool_analysis": sum(1 for r in rows if r["set"] == "pool" and r["near_duplicate_of_training"])}
    res["runtime"] = {"frozen": {**F["timing"], "errors": F["runtime_errors"]}, "v2": {**V["timing"], "errors": V["runtime_errors"]},
                      "topic_identical_in_both_stacks": F["topic"] == V["topic"]}
    # ---- distributions and cross-stack disagreement (pool only; agreement is NOT correctness)
    for task, labels in (("intent", INTENTS), ("topic", TOPICS), ("sentiment", SENTIMENTS)):
        fa = [F[task][i][0] for i in pool]; va = [V[task][i][0] for i in pool]
        res[task] = {"distribution_frozen": dict(Counter(fa)), "distribution_v2": dict(Counter(va)),
                     "disagreement_rate": round(sum(a != b for a, b in zip(fa, va)) / len(pool), 3),
                     "transitions_frozen_to_v2": {f"{a}->{b}": c for (a, b), c in Counter((a, b) for a, b in zip(fa, va) if a != b).most_common(12)},
                     "mean_confidence": {"frozen": round(float(np.mean([F[task][i][1] for i in pool if F[task][i][1] is not None])), 3), "v2": round(float(np.mean([V[task][i][1] for i in pool if V[task][i][1] is not None])), 3)}}
    # ---- confidence / review / priority / routing (thresholds untouched)
    def derive(P):
        d = []
        for i in range(n):
            intent, conf = P["intent"][i]; topic = P["topic"][i][0]; sent = P["sentiment"][i][0]
            needs = conf < THRESH
            route = determine_routing(primary_intent=intent, topic_subtype=topic, subcategory=None, secondary_intent=None, is_content_meaningful=True)
            pr, why = compute_priority(primary_intent=intent, topic_subtype=topic, generation_routing=None, needs_review=needs, confidence=conf)
            pr0, _ = compute_priority(primary_intent=intent, topic_subtype=topic, generation_routing=None, needs_review=False, confidence=None)
            e = determine_response_eligibility(EligibilityInput(comment_text=rows[i]["text"], primary_intent=intent, topic_subtype=topic, trusted_grounding_available=False,
                                                                classification_routing_decision=route.routing_decision))
            d.append(dict(needs=needs, route=route.routing_decision, prio=pr, prio_without_confidence=pr0, elig=e.routing_decision, elig_reason=e.reason_code, intent=intent, topic=topic, conf=conf))
        return d
    DF, DV = derive(F), derive(V)
    def summ(D):
        P = [D[i] for i in pool]
        spam = [x for x in P if x["intent"] == "SPAM"]
        return {"needs_review_share": round(sum(x["needs"] for x in P) / len(P), 3), "routing": dict(Counter(x["route"] for x in P)),
                "priority_counts": {str(k): v for k, v in sorted(Counter(x["prio"] for x in P).items())},
                "priority_changed_by_confidence_share": round(sum(x["prio"] != x["prio_without_confidence"] for x in P) / len(P), 3),
                "priority1_share": round(sum(x["prio"] == 1 for x in P) / len(P), 3),
                "response_eligibility": dict(Counter(x["elig"] for x in P)),
                "single_spam_predictions": len(spam),
                "single_spam_routed_to_human_review": sum(x["elig"] == "REVIEW_ONLY" for x in spam),
                "spam_corroborated_by_topic_model_no_response": sum(x["elig"] == "NO_RESPONSE_RECOMMENDED" and x["intent"] == "SPAM" for x in P),
                "silently_suppressed_by_a_single_label": sum(1 for x in P if x["intent"] == "SPAM" and x["topic"] != "OFF_TOPIC_PERSONAL" and x["elig"] == "NO_RESPONSE_RECOMMENDED"),
                "high_confidence_(>=0.70)_share": round(sum(x["conf"] >= 0.70 for x in P) / len(P), 3)}
    res["safety"] = {"frozen": summ(DF), "v2": summ(DV)}
    res["needs_review_transitions"] = {"flagged_in_frozen_not_in_v2": sum(DF[i]["needs"] and not DV[i]["needs"] for i in pool), "flagged_in_v2_not_in_frozen": sum(DV[i]["needs"] and not DF[i]["needs"] for i in pool)}
    res["priority_transitions"] = {f"{DF[i]['prio']}->{DV[i]['prio']}": c for (i0), c in [] } or {f"{a}->{b}": c for (a, b), c in Counter((DF[i]["prio"], DV[i]["prio"]) for i in pool).items()}
    # ---- high-impact cue-based risk flags (cues SURFACE candidates for human review; they are not labels)
    texts = {i: rows[i]["text"] for i in pool}
    fail_cue = [i for i in pool if EA.any_failure(texts[i])]
    ques = [i for i in pool if has_question(texts[i])]
    def dist(idx, P, task="intent"): return dict(Counter(P[task][i][0] for i in idx))
    res["cue_risk"] = {"failure_expression_comments": len(fail_cue), "failure_cue_intent_frozen": dist(fail_cue, F), "failure_cue_intent_v2": dist(fail_cue, V),
                       "failure_cue_topic_frozen": dist(fail_cue, F, "topic"), "failure_cue_topic_v2": dist(fail_cue, V, "topic"),
                       "question_comments": len(ques), "question_intent_frozen": dist(ques, F), "question_intent_v2": dist(ques, V),
                       "predicted_TECHNICAL_ISSUE_frozen": sum(F["topic"][i][0] == "TECHNICAL_ISSUE" for i in pool), "predicted_TECHNICAL_ISSUE_v2": sum(V["topic"][i][0] == "TECHNICAL_ISSUE" for i in pool),
                       "predicted_COMPLAINT_frozen": sum(F["intent"][i][0] == "COMPLAINT" for i in pool), "predicted_COMPLAINT_v2": sum(V["intent"][i][0] == "COMPLAINT" for i in pool)}
    risks = {
        "v2_high_conf_(>=0.70)_THANKS_or_SPAM_on_a_question": [i for i in ques if V["intent"][i][0] in ("THANKS_PRAISE", "SPAM") and V["intent"][i][1] >= 0.70],
        "v2_high_conf_(>=0.70)_SPAM_on_failure_cue": [i for i in fail_cue if V["intent"][i][0] == "SPAM" and V["intent"][i][1] >= 0.70],
        "v2_not_COMPLAINT_on_failure_cue_(complaint_missed?)": [i for i in fail_cue if V["intent"][i][0] != "COMPLAINT"],
        "frozen_not_COMPLAINT_on_failure_cue": [i for i in fail_cue if F["intent"][i][0] != "COMPLAINT"],
        "v2_INQUIRY_to_other_vs_frozen_INQUIRY": [i for i in pool if F["intent"][i][0] == "INQUIRY" and V["intent"][i][0] != "INQUIRY"],
        "v2_NEW_INQUIRY_vs_frozen_other": [i for i in pool if F["intent"][i][0] != "INQUIRY" and V["intent"][i][0] == "INQUIRY"],
        "v2_NEW_COMPLAINT_vs_frozen_other": [i for i in pool if F["intent"][i][0] != "COMPLAINT" and V["intent"][i][0] == "COMPLAINT"],
        "frozen_COMPLAINT_lost_in_v2": [i for i in pool if F["intent"][i][0] == "COMPLAINT" and V["intent"][i][0] != "COMPLAINT"],
        "v2_SUGGESTION_to_other_vs_frozen_SUGGESTION": [i for i in pool if F["intent"][i][0] == "SUGGESTION" and V["intent"][i][0] != "SUGGESTION"],
        "sentiment_NEGATIVE_flip": [i for i in pool if (F["sentiment"][i][0] == "NEGATIVE") != (V["sentiment"][i][0] == "NEGATIVE")],
    }
    res["risk_counts"] = {k: len(v) for k, v in risks.items()}
    import random
    random.seed(7)
    ex = {}
    for k, idx in risks.items():
        pick = random.sample(idx, min(6, len(idx)))
        ex[k] = [{"text": rows[i]["text"][:170], "frozen": [F["intent"][i][0], round(F["intent"][i][1], 2), F["topic"][i][0], F["sentiment"][i][0]],
                  "v2": [V["intent"][i][0], round(V["intent"][i][1], 2), V["topic"][i][0], V["sentiment"][i][0]]} for i in pick]
    (OUT / "examples_of_disagreement.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1))
    # ---- blank / malformed
    mal = {}
    for name, P in (("frozen", F), ("v2", V)):
        m = P["malformed"]
        mal[name] = None if not m else {"n": P["malformed_n"], "all_labels_in_taxonomy": all(a in INTENTS for a, _ in m["intent"]) and all(a in SENTIMENTS for a, _ in m["sentiment"]),
                                        "intent_distribution": dict(Counter(a for a, _ in m["intent"])), "mean_intent_confidence": round(float(np.mean([b for _, b in m["intent"]])), 3),
                                        "any_runtime_error": bool(P["runtime_errors"])}
    res["blank_or_malformed_real_rows"] = {"source": "raw export rows the prefilter removes (null / near-empty)", "reasons": dict(Counter(F["malformed_reasons"])), **mal}
    # ---- secondary: exposed blind comments with human labels (NOT used for selection; small; natural and targeted never pooled)
    labs = {r["record_id"]: r for r in json.load(open(ROOT / "data/evaluation/blind_eval_v1/returned/labels_verbatim_from_numbers.json"))}
    sec = {}
    for comp in ("natural", "targeted"):
        idx = [i for i, r in enumerate(rows) if r["set"] == comp and r["blind_record_id"] in labs and labs[r["blind_record_id"]].get("intent")]
        comp_res = {"n": len(idx)}
        for task, labels in (("intent", INTENTS), ("topic", TOPICS), ("sentiment", SENTIMENTS)):
            g = [labs[rows[i]["blind_record_id"]][task] for i in idx]
            row = {}
            for nm, P in (("frozen", F), ("v2", V)):
                p = [P[task][i][0] for i in idx]; m = prf(g, p, labels)
                row[nm] = {"accuracy": m["accuracy"], "macro_f1_supported": m["macro_f1"], "recall": {c: v["recall"] for c, v in m["per_class"].items() if v["support"]},
                           "precision": {c: v["precision"] for c, v in m["per_class"].items() if v["predicted"]}, "confusion": confusion(g, p, labels)}
            row["paired_changes"] = {"correct_to_wrong": sum(1 for gi, a, b in zip(g, [F[task][i][0] for i in idx], [V[task][i][0] for i in idx]) if a == gi and b != gi),
                                     "wrong_to_correct": sum(1 for gi, a, b in zip(g, [F[task][i][0] for i in idx], [V[task][i][0] for i in idx]) if a != gi and b == gi)}
            comp_res[task] = row
        sec[comp] = comp_res
    res["secondary_exposed_blind_75_with_human_labels"] = {"note": "independent of training; but exposed during earlier analysis and small - indicative only, never pooled, not used for any decision rule", **sec}
    (OUT / "final_validation_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


def run():
    for stack in ("frozen", "v2"):
        p = subprocess.run([sys.executable, "-m", "scripts.offline_eval.classification_v2.final_validation", "predict", "--stack", stack], cwd=str(ROOT), capture_output=True, text=True)
        print(p.stdout[-400:], p.stderr[-600:] if p.returncode else "")
        if p.returncode:
            raise SystemExit(f"{stack} run failed")
    analyse()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prepare", "predict", "analyse", "run"]); ap.add_argument("--stack")
    a = ap.parse_args()
    {"prepare": prepare, "predict": lambda: predict(a.stack), "analyse": analyse, "run": run}[a.cmd]()
