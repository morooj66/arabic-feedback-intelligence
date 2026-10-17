"""Offline, deterministic evaluation of EXISTING Gemini drafts stored in a local review database (read-only; no Gemini, no network, no writes).
Measures only what code can measure honestly: grounding availability, unsupported-claim findings under the current gates, escalation correctness under the
current eligibility policy, and structural tone / relevance PROXIES. Relevance and tone quality themselves, and factual correctness of definitions, need
human judgement or live generation and are reported as such. Output: aggregates only (no comment or reply text)."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from rcs.classification.arabic_text import normalize_arabic  # noqa: E402
from rcs.response_generation import claim_gates as C  # noqa: E402
from rcs.response_generation.eligibility import EligibilityInput, asks_for_facts, determine_response_eligibility  # noqa: E402
from rcs.response_generation.validate import collect_reply_findings, validate_draft  # noqa: E402

DB = ROOT / "data/db/excel_review_demo.sqlite"
OUT = ROOT / "data/evaluation/reply_quality"; OUT.mkdir(parents=True, exist_ok=True)
APOLOGY = ("نعتذر", "نأسف", "عذرا", "نتفهم", "نفهم")
ASK_DETAILS = ("نرجو", "يرجى", "تزويدنا", "توضيح", "تفاصيل", "اخبارنا")


def toks(s):
    return set(C._content_tokens(C.normalize(s)))


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute("""select comment_id, original_comment, coalesce(primary_intent, ai_primary_intent) intent, coalesce(topic_subtype, ai_topic_subtype) topic,
                        suggested_final_reply, generation_status, generation_routing_decision, generation_reason_code, generation_prompt_version, review_status
                        from review_items where generation_status is not null""").fetchall()
    drafts = [r for r in rows if r["generation_status"] == "SUCCEEDED" and (r["suggested_final_reply"] or "").strip()]
    res = {"source": "local review database (read-only)", "rows_with_generation": len(rows), "generated_drafts": len(drafts),
           "grounding": {"official_knowledge_configured": False, "drafts_grounded_in_official_knowledge": 0,
                         "note": "no approved knowledge source exists locally; every stored draft was generated without factual grounding"}}
    by_status = Counter(r["generation_status"] for r in rows); res["generation_status"] = dict(by_status)
    by_ver = defaultdict(lambda: Counter())
    gate_reasons = Counter(); cat = Counter(); per_intent = defaultdict(Counter)
    esc = Counter(); tone = Counter(); echo = []
    for r in drafts:
        v = r["generation_prompt_version"] or "unknown"
        d, q = r["suggested_final_reply"], r["original_comment"] or ""
        ok, why = validate_draft(d, comment_text=q, trusted_grounding_text=None)
        f = collect_reply_findings(d, comment_text=q)
        by_ver[v]["n"] += 1
        by_ver[v]["passes_current_gates"] += int(ok)
        gate_reasons[why or "pass"] += 1
        per_intent[r["intent"] or "none"]["n"] += 1; per_intent[r["intent"] or "none"]["pass"] += int(ok)
        for code in f.blocking + f.attestable:
            cat[code] += 1
        # escalation under the CURRENT eligibility policy (no grounding, model labels as stored)
        e = determine_response_eligibility(EligibilityInput(comment_text=q, primary_intent=r["intent"], topic_subtype=r["topic"], trusted_grounding_available=False))
        esc[("generated_then", e.routing_decision)] += 1
        if e.routing_decision != "GENERATE_DRAFT":
            esc[("generated_but_now_" + e.routing_decision.lower(), e.reason_code)] += 1
        if asks_for_facts(q):
            esc["draft_for_a_question_without_grounding"] += 1
        # tone / structure proxies (NOT quality judgements)
        n = normalize_arabic(d)
        if r["intent"] == "COMPLAINT" or r["topic"] == "TECHNICAL_ISSUE":
            tone["complaint_or_technical_drafts"] += 1
            tone["  ...with apology/empathy marker"] += int(any(normalize_arabic(a) in n for a in APOLOGY))
            tone["  ...asking for details"] += int(any(normalize_arabic(a) in n for a in ASK_DETAILS))
        ov = len(toks(d) & toks(q)) / max(1, len(toks(d)))
        echo.append(ov)
        tone["mean_content_overlap_with_comment_x100"] = tone.get("mean_content_overlap_with_comment_x100", 0)
        tone["length_chars_total"] += len(d)
    # What is left in the assertion gate: split flagged sentences into detached evaluative praise of the visitor's contribution ("ملاحظة قيمة ...",
    # the register human reviewers rejected) versus everything else (factual statements about the dictionary / platform / words).
    praise_nouns = ("ملاحظه", "مشاركه", "اقتراح", "توضيح", "مساهمه", "اضافه", "تنبيه", "استشهاد", "صياغه", "مقترح", "طرح")
    praise_adj = ("قيم", "دقيق", "موفق", "مثري", "مقدر", "وجيه", "لافت", "جميل", "مهم", "رائع", "ممتاز", "ثري", "لغوي")
    flagged = Counter()
    for r in drafts:
        ct = toks(r["original_comment"] or "")
        for s in C._sentences(r["suggested_final_reply"]):
            s2 = " ".join(C._tokens(s))
            if s2 and not C._is_safe_speech_act(s2, ct):
                w = s2.split()
                if w and w[0] in praise_nouns and any(w[1].startswith(a) for a in praise_adj if len(w) > 1):
                    flagged["detached evaluative praise of the visitor's contribution"] += 1
                else:
                    flagged["other declarative statement"] += 1
    res["assertion_gate_flagged_sentences"] = dict(flagged)
    res["by_prompt_version"] = {v: dict(k, pass_rate=round(k["passes_current_gates"] / k["n"], 3)) for v, k in sorted(by_ver.items())}
    res["first_failing_gate"] = dict(gate_reasons.most_common())
    res["findings_any_gate_counts"] = dict(cat.most_common())
    res["pass_rate_by_intent"] = {k: dict(v, pass_rate=round(v["pass"] / v["n"], 3)) for k, v in per_intent.items()}
    res["escalation_replay"] = {str(k): v for k, v in esc.items()}
    res["tone_proxies"] = {k: v for k, v in tone.items() if k not in ("mean_content_overlap_with_comment_x100", "length_chars_total")}
    res["mean_draft_length_chars"] = round(tone["length_chars_total"] / max(1, len(drafts)), 1)
    res["mean_content_overlap_draft_vs_comment"] = round(sum(echo) / max(1, len(echo)), 3)
    # skipped rows: would the current policy still skip them?
    skipped = [r for r in rows if r["generation_status"] == "SKIPPED"]
    sk = Counter()
    for r in skipped:
        e = determine_response_eligibility(EligibilityInput(comment_text=r["original_comment"] or "", primary_intent=r["intent"], topic_subtype=r["topic"], trusted_grounding_available=False))
        sk[(r["generation_routing_decision"], "->", e.routing_decision)] += 1
    res["skipped_rows_replay"] = {" ".join(map(str, k)): v for k, v in sk.most_common()}
    # Second source: drafts reviewed by a human in the agent-reply review sessions (later prompt versions; human verdicts where given).
    import glob
    stored = []
    for p in sorted(glob.glob(str(ROOT / "data/processed/agent_reply_review/*.jsonl"))):
        stored += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    gen = [s for s in stored if s.get("generation_status") == "SUCCEEDED"]
    agg = defaultdict(Counter)
    for s in gen:
        ok, why = validate_draft(s["generated_reply"], comment_text=s["sanitized_comment"], trusted_grounding_text=None)
        key = s.get("generation_prompt_version") or "pre-rg-v13 sessions (version not stored)"
        agg[key]["n"] += 1; agg[key]["pass_current_gates"] += int(ok)
        agg[key]["human_" + str(s.get("human_judgment"))] += 1
        agg[key]["human_" + str(s.get("human_judgment")) + ("_and_passes" if ok else "_and_fails")] += 1
    res["human_reviewed_sessions"] = {k: dict(v) for k, v in agg.items()}
    (OUT / "existing_drafts_evaluation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
