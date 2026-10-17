"""Offline, deterministic quality audit of the EXISTING system. No network, no model API, no training, no DB, no Gold writes.
Prints/saves aggregates only (no comment text). Sections:
  C1  frozen Intent on the 108 validation rows (held out from the 455-row training set): accuracy, macro-F1, per-class recall, confusion
  C2  confidence analysis: what the reported 'confidence' means (max predict_proba of a class-balanced LR) vs. actual accuracy
  C3  routing/eligibility consequence of intent errors (what a wrong Intent does downstream)
  C4  frozen RCS on the 75 blind labels (stored predictions): Intent/Topic/Sentiment accuracy/macro-F1/confusion
  R1  eligibility rule table (intent x topic x grounding)
  R2  draft-validator adversarial battery (synthetic drafts) - detection rates of unsupported claims
  R3  stored human-reviewed drafts: re-validation + grounding mode + reviewer verdicts
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import csv, glob, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402
from rcs.classification import frozen_inference  # noqa: E402
from rcs.response_generation.eligibility import EligibilityInput, determine_response_eligibility  # noqa: E402
from rcs.response_generation.validate import validate_draft  # noqa: E402
from rcs.routing.engine import determine_routing  # noqa: E402

OUT = ROOT / "data" / "evaluation" / "system_audit"; OUT.mkdir(parents=True, exist_ok=True)
INTENTS = ["THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM"]
res: dict = {}


def prf(gold, pred, labels):
    out = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred)); pp = sum(p == c for p in pred); s = sum(g == c for g in gold)
        P = tp / pp if pp else 0.0; R = tp / s if s else None
        F = 2 * P * R / (P + R) if (R and P) else 0.0
        out[c] = dict(support=s, predicted=pp, precision=round(P, 3), recall=None if R is None else round(R, 3), f1=round(F, 3) if s else None)
    f = [v["f1"] for v in out.values() if v["support"]]
    return out, round(sum(f) / len(f), 3)


def confusion(gold, pred, labels):
    return {g: dict(Counter(p for gg, p in zip(gold, pred) if gg == g)) for g in labels if any(gg == g for gg in gold)}


# ---------------------------------------------------------------- C1/C2/C3 : validation rows
val = [r for r in csv.DictReader(open(ROOT / "data/gold/gold_dataset_final_training.csv", newline="", encoding="utf-8-sig"))
       if r["split"] == "val" and r["gold_annotation_status"] == "LABELED" and r["gold_primary_intent"]]
texts = [r["comment_text"] for r in val]; gold = [r["gold_primary_intent"] for r in val]
pred = frozen_inference.predict_intent(texts)
pl = [p[0] for p in pred]; conf = [p[1] for p in pred]
pc, mf = prf(gold, pl, INTENTS)
res["C1_val_intent"] = dict(n=len(val), accuracy=round(sum(g == p for g, p in zip(gold, pl)) / len(val), 3), macro_f1=mf, per_class=pc, confusion=confusion(gold, pl, INTENTS))
bins = [(0, .3), (.3, .45), (.45, .6), (.6, 1.01)]
cal = []
for lo, hi in bins:
    idx = [i for i, c in enumerate(conf) if lo <= c < hi]
    cal.append(dict(bin=f"{lo:.2f}-{min(hi,1):.2f}", n=len(idx), mean_conf=round(sum(conf[i] for i in idx) / len(idx), 3) if idx else None,
                    accuracy=round(sum(gold[i] == pl[i] for i in idx) / len(idx), 3) if idx else None))
thr = settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD
flag = [c < thr for c in conf]
by_pred = {c: dict(n=sum(p == c for p in pl), mean_conf=round(sum(cf for p, cf in zip(pl, conf) if p == c) / max(1, sum(p == c for p in pl)), 3)) for c in INTENTS}
wrong = [g != p for g, p in zip(gold, pl)]
res["C2_confidence"] = dict(threshold=thr, calibration_bins=cal, flagged_needs_review=sum(flag), flagged_share=round(sum(flag) / len(flag), 3),
                            errors_total=sum(wrong), errors_flagged=sum(w and f for w, f in zip(wrong, flag)),
                            errors_not_flagged=sum(w and not f for w, f in zip(wrong, flag)),
                            correct_flagged=sum((not w) and f for w, f in zip(wrong, flag)), conf_min=round(min(conf), 3), conf_max=round(max(conf), 3),
                            mean_conf_by_predicted_class=by_pred, mean_conf_correct=round(sum(c for c, w in zip(conf, wrong) if not w) / max(1, wrong.count(False)), 3),
                            mean_conf_wrong=round(sum(c for c, w in zip(conf, wrong) if w) / max(1, wrong.count(True)), 3))

def downstream(intent, topic=None):
    e = determine_response_eligibility(EligibilityInput(comment_text="نص كاف للمعالجة", primary_intent=intent, topic_subtype=topic, confidence=None))
    r = determine_routing(intent, topic, None, None, True)
    return r.routing_decision, e.routing_decision
cons = Counter()
for g, p in zip(gold, pl):
    if g == p: continue
    cons[(g, p, *downstream(p))] += 1
res["C3_val_error_consequences"] = [dict(gold=k[0], predicted=k[1], routing=k[2], eligibility=k[3], n=v) for k, v in cons.most_common()]
silenced = sum(1 for g, p in zip(gold, pl) if p == "SPAM" and g != "SPAM"); drafted_spam = sum(1 for g, p in zip(gold, pl) if g == "SPAM" and p != "SPAM")
res["C3_summary"] = dict(real_comments_silenced_as_spam=silenced, of_non_spam_gold=sum(g != "SPAM" for g in gold),
                         gold_spam_that_would_get_a_draft=drafted_spam, of_spam_gold=sum(g == "SPAM" for g in gold),
                         complaints_not_flagged_complaint=sum(g == "COMPLAINT" and p != "COMPLAINT" for g, p in zip(gold, pl)), complaints=sum(g == "COMPLAINT" for g in gold))

# ---------------------------------------------------------------- C5 : queue priority produced by predicted labels vs gold labels
from rcs.review.priority import compute_priority  # noqa: E402
def prio(intent, conf_):
    return compute_priority(primary_intent=intent, topic_subtype=None, generation_routing=None, needs_review=conf_ < thr, confidence=conf_)[0]
pp = [prio(p, c) for p, c in zip(pl, conf)]
gp = [prio(g, 0.9) for g in gold]   # what the priority would be if the label were right and confident
comp_idx = [i for i, g in enumerate(gold) if g == "COMPLAINT"]
res["C5_priority"] = dict(system_priority_counts=dict(Counter(pp)), complaints_n=len(comp_idx),
    complaints_priority_assigned=dict(Counter(pp[i] for i in comp_idx)), complaints_would_be_priority_1_if_detected=len(comp_idx),
    non_complaints_given_priority_1=sum(1 for i, g in enumerate(gold) if g != "COMPLAINT" and pp[i] == 1),
    priority_tier_changed_by_uncertainty_flag=sum(1 for p, c in zip(pl, conf) if prio(p, c) != prio(p, 0.9)), n=len(pl))

# ---------------------------------------------------------------- C4 : blind 75 (stored frozen RCS predictions)
bl = {r["record_id"]: r for r in json.load(open(ROOT / "data/evaluation/blind_eval_v1/returned/labels_verbatim_from_numbers.json")) if not r.get("insufficient_context")}
bp = json.load(open(ROOT / "data/evaluation/blind_eval_v1/results/rcs_predictions.json"))
comp = {r["record_id"]: r["component"] for r in csv.DictReader(open(ROOT / "data/evaluation/blind_eval_v1/coordinator_only/blind_eval_mapping.local.csv", encoding="utf-8"))}
TOP = list(settings.TOPIC_SUBTYPES); SEN = list(settings.SENTIMENTS)
c4 = {}
for sample in ("natural", "targeted"):
    ids = [i for i in sorted(bl) if comp[i] == sample and i in bp and bl[i].get("intent")]
    for task, labs in (("intent", INTENTS), ("topic", TOP), ("sentiment", SEN)):
        g = [bl[i][task] for i in ids]; p = [bp[i][task] for i in ids]
        per, m = prf(g, p, labs)
        c4[f"{sample}_{task}"] = dict(n=len(ids), accuracy=round(sum(a == b for a, b in zip(g, p)) / len(ids), 3), macro_f1_supported=m,
                                     per_class_recall={k: v["recall"] for k, v in per.items() if v["support"]}, predicted_share={k: round(v["predicted"] / len(ids), 2) for k, v in per.items() if v["predicted"]},
                                     confusion=confusion(g, p, labs))
res["C4_blind_frozen_rcs"] = c4

# ---------------------------------------------------------------- R1 eligibility table
tab = []
for intent in INTENTS:
    for topic in (None, "TECHNICAL_ISSUE", "OFF_TOPIC_PERSONAL", "WORD_OF_YEAR_NOMINATION"):
        for gr in (False, True):
            e = determine_response_eligibility(EligibilityInput(comment_text="نص كاف للمعالجة", primary_intent=intent, topic_subtype=topic, trusted_grounding_available=gr, confidence=0.2, needs_review=True))
            tab.append(dict(intent=intent, topic=topic, grounding=gr, decision=e.routing_decision, rule=e.rule_id))
res["R1_eligibility_table"] = tab
res["R1_facts"] = dict(low_confidence_changes_eligibility=len({(t['decision']) for t in tab if t['intent']=='SUGGESTION'}) == 1,
                       note="confidence=0.2 and needs_review=True were passed for every row; decisions depend only on intent/topic/grounding")

# ---------------------------------------------------------------- R2 validator batteries, BEFORE (original gates) vs AFTER (claim gates)
from rcs.response_generation import claim_gates as _cg  # noqa: E402
from rcs.response_generation.validate import validate_reviewer_edit  # noqa: E402
from scripts.offline_eval.system_audit.reply_safety_battery import HELD_OUT_SAFE, HELD_OUT_UNSAFE, SAFE, UNSAFE  # noqa: E402

def _accepted(items, fn):
    return [bool(fn(d, comment_text=k)[0]) for _c, d, k in items]

def _measure():
    out = {}
    for prof, fn in (("generated", validate_draft), ("reviewer_edit", validate_reviewer_edit)):
        out[prof] = dict(
            unsafe_accepted=f"{sum(_accepted(UNSAFE, fn))}/{len(UNSAFE)}", held_out_unsafe_accepted=f"{sum(_accepted(HELD_OUT_UNSAFE, fn))}/{len(HELD_OUT_UNSAFE)}",
            safe_rejected=f"{len(SAFE) - sum(_accepted(SAFE, fn))}/{len(SAFE)}", held_out_safe_rejected=f"{len(HELD_OUT_SAFE) - sum(_accepted(HELD_OUT_SAFE, fn))}/{len(HELD_OUT_SAFE)}")
    return out

_orig_gate = _cg.first_failing_claim_gate
_cg.first_failing_claim_gate = lambda *a, **k: None
before = _measure()
_cg.first_failing_claim_gate = _orig_gate
res["R2_validator_before_after"] = dict(before_original_gates=before, after_claim_gates=_measure(),
    note="held-out set was used for two rounds of refinement after its first clean measurement, so it is not an independent estimate any more")

# ---------------------------------------------------------------- R3 stored reviewed drafts
st = []
for f in sorted(glob.glob(str(ROOT / "data/processed/agent_reply_review/*.jsonl"))):
    st += [json.loads(l) for l in open(f) if l.strip()]
gen = [r for r in st if r.get("generation_status") == "SUCCEEDED"]
revalid = [validate_draft(r["generated_reply"], trusted_grounding_text=None, comment_text=r["sanitized_comment"]) for r in gen]
res["R3_stored_replies"] = dict(total=len(st), generated=len(gen), no_response_recommended=sum(r.get("routing_decision") == "NO_RESPONSE_RECOMMENDED" for r in st),
    grounding_mode=dict(Counter(r.get("grounding_mode") for r in gen)), official_knowledge_used=dict(Counter(str(r.get("official_knowledge_used")) for r in st)),
    human_judgment_all=dict(Counter(str(r.get("human_judgment")) for r in st)), human_judgment_on_generated=dict(Counter(str(r.get("human_judgment")) for r in gen)),
    revalidation_pass=sum(ok for ok, _ in revalid), revalidation_fail=sum(not ok for ok, _ in revalid),
    approved_but_revalidation_fails=sum(1 for r, (ok, _) in zip(gen, revalid) if r.get("human_judgment") == "APPROVE" and not ok),
    rejected_but_validator_passes=sum(1 for r, (ok, _) in zip(gen, revalid) if r.get("human_judgment") == "REJECT" and ok),
    generated_by_intent=dict(Counter(r["intent"] for r in gen)), no_response_by_intent=dict(Counter(r["intent"] for r in st if r.get("routing_decision") == "NO_RESPONSE_RECOMMENDED")))
(OUT / "audit_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("R1_eligibility_table",)}, ensure_ascii=False, indent=1))
