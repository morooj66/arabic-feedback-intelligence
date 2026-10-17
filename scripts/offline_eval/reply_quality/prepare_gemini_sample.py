"""Prepare (offline, no Gemini) a <= 20-comment diverse sample for the live rg-v14.2 quality run. Source: the eligible-comment universe of the final validation (already
privacy-sanitised by the existing name policy, sealed human_test and dev/Gold ids excluded). Categories come from the v2 pilot stack's PREDICTED labels plus declared cues - they
are sampling aids, not labels. Excluded: the 75 exposed blind comments, near-duplicates of training, anything with a mask/placeholder, questions or predicted INQUIRY/SPAM (a factual
dictionary answer would be needed and there is no approved grounding). Deterministic (seed 20261017)."""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from rcs.classification.arabic_text import has_question  # noqa: E402
from rcs.response_generation.eligibility import EligibilityInput, asks_for_facts, determine_response_eligibility  # noqa: E402

V = ROOT / "data/evaluation/classification_v2_final_validation"
OUT = ROOT / "data/evaluation/reply_quality"; OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(V / "universe.jsonl", encoding="utf-8")]
P = json.load(open(V / "predictions_v2.json"))
sys.path.insert(0, str(ROOT / "scripts/offline_eval/error_analysis"))
import common as EA  # noqa: E402  (declared failure-expression families)
THANKS = ("شكر", "جزاك", "جزى الله", "بارك الله", "يعطيكم العافية", "ممتاز", "رائع", "جهود", "مشكور")
TECH = ("لا يفتح", "ما يفتح", "لا يعمل", "ما يعمل", "لا تعمل", "لا يظهر", "ما يظهر", "ما طلع", "يعلق", "تطبيق", "التطبيق", "تسجيل الدخول", "الرابط", "البحث لا", "الموقع لا")
FEATURE = ("أتمنى", "اتمنى", "خاصية", "ميزة", "تضيفون", "أقترح", "اقترح")
random.seed(20261017)


def eligible(i):
    r = rows[i]
    t = r["text"]
    if r["set"] != "pool" or r["near_duplicate_of_training"]:
        return False
    if re.search(r"\[[A-Z_]{3,}\]|https?://|@\w|\d{5,}", t) or len(t) < 25 or len(t) > 400:
        return False                                         # masks / links / handles / long digit runs / too short or long
    if has_question(t) or asks_for_facts(t):
        return False                                         # would need an unavailable factual answer -> stays with a human
    intent, topic = P["intent"][i][0], P["topic"][i][0]
    if intent in ("INQUIRY", "SPAM"):
        return False
    e = determine_response_eligibility(EligibilityInput(comment_text=t, primary_intent=intent, topic_subtype=topic, trusted_grounding_available=False))
    return e.routing_decision == "GENERATE_DRAFT"            # only comments the pipeline would actually send to Gemini


ok = [i for i in range(len(rows)) if eligible(i)]
cat = {
    "complaint": [i for i in ok if P["intent"][i][0] == "COMPLAINT"],
    "suggestion_feature": [i for i in ok if P["intent"][i][0] == "SUGGESTION" and any(c in rows[i]["text"] for c in FEATURE)],
    "word_submission": [i for i in ok if P["topic"][i][0] == "WORD_SUBMISSION" and P["intent"][i][0] == "SUGGESTION"],
    "entry_correction": [i for i in ok if P["topic"][i][0] == "ENTRY_CORRECTION"],
    "thanks_praise": [i for i in ok if any(c in rows[i]["text"] for c in THANKS)],
    "technical_report": [i for i in ok if EA.any_failure(rows[i]["text"]) or any(c in rows[i]["text"] for c in TECH[:12])],
}
quota = {"complaint": 4, "suggestion_feature": 3, "word_submission": 4, "entry_correction": 3, "thanks_praise": 3, "technical_report": 3}
chosen, used = [], set()
for k, n in quota.items():
    pool = [i for i in cat[k] if i not in used]
    random.shuffle(pool)
    for i in pool[:n]:
        used.add(i); chosen.append((k, i))
chosen = chosen[:20]
sample = [{"n": j + 1, "category": k, "comment_id": rows[i]["comment_id"], "comment_redacted": rows[i]["text"],
           "intent": P["intent"][i][0], "topic": P["topic"][i][0], "sentiment": P["sentiment"][i][0], "classification_source": "v2 pilot (I10 / frozen MARBERTv2 / S5)"}
          for j, (k, i) in enumerate(chosen)]
(OUT / "gemini_sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1))
print(json.dumps({"candidates_after_exclusions": len(ok), "available_per_category": {k: len(v) for k, v in cat.items()}, "selected": len(sample),
                  "selected_per_category": dict(Counter(s["category"] for s in sample))}, indent=1))
