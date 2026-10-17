"""Label-policy adjudication of the 94 development-validation records. LOCAL, READ-ONLY.

Every entry is a REVIEWER PROPOSAL, never a Gold change: this module reads gold, never writes it, and
`main()` re-checks that data/gold/gold_dataset_final_training.csv is byte-identical to its fingerprint.
Buckets (per record AND task):
  LABEL_ERROR  gold contradicts explicit text of LABELING_GUIDE.md (probable annotation error; human confirms)
  AMBIGUOUS    the guide is silent or self-conflicting, or near-identical rows carry different gold labels
  MODEL_ERROR  gold is consistent with the guide and a model contradicts it
Eval ids refer to the local sanitized inputs (dev-NNNN); they map to comment ids only in id_map.local.json.

Run:  python scripts/offline_eval/jev_dev_validation/adjudication.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_common  # noqa: E402,F401
import jev_metrics as M  # noqa: E402
from jev_common import GOLD_CSV, OUT_DIR  # noqa: E402

LE, AM, ME = "LABEL_ERROR", "AMBIGUOUS", "MODEL_ERROR"
IC = "INSUFFICIENT_CONTEXT"

RULES = {
    "R1": "Bare lexical submission (word + meaning/example, no framing): INQUIRY vs SUGGESTION",
    "R2": "Entry-error reports: COMPLAINT vs SUGGESTION vs INQUIRY (what is 'tone'?)",
    "R3": "TECHNICAL_ISSUE boundary (site fault vs content fault)",
    "R4": "WORD_SUBMISSION vs ENTRY_CORRECTION when the text only supplies lexical content",
    "R5": "NEGATIVE vs NEUTRAL (nonsense/test messages; blunt error reports)",
    "R6": "POSITIVE vs NEUTRAL (courtesy formulas and praise inside lexical content)",
    "R7": "Single word / fragment rows labelled instead of INSUFFICIENT_CONTEXT (guide section 5)",
    "R8": "Test messages and bare greetings (SPAM vs INQUIRY / THANKS_PRAISE)",
    "R9": "Proposals to contribute a whole dictionary (DICTIONARY_SUGGESTION boundary)",
    "M1": "RCS reads plain thanks as SUGGESTION / WORD_SUBMISSION",
    "M2": "RCS misses plain problem reports (site access / display) as thanks or suggestion",
    "M3": "Jev reads non-thanks chatter or greetings as THANKS_PRAISE / COMPLAINT",
    "M4": "RCS reads neutral formal dictionary text as NEGATIVE",
}

# (eval number, task, bucket, rule, proposed label or None, short note)
E: list[tuple[str, str, str, str, str | None, str]] = []


def add(nums, task, bucket, rule, proposed=None, note=""):
    for n in nums.split():
        E.append((f"dev-{int(n):04d}", task, bucket, rule, proposed, note))


# ---- INTENT: labelling errors against explicit guide text
add("0003", "intent", LE, "R8", "THANKS_PRAISE", "pure praise, not spam")
add("0055", "intent", LE, "R1", "INQUIRY", "asks which sources were used: a genuine question")
add("0024", "intent", LE, "R8", "SPAM", "'test of an add-word request' = test noise (cf. 0049, 0069 labelled SPAM)")
add("0021", "intent", LE, "R2", "SUGGESTION", "states an entry is wrong (feminine inflection); not spam")
add("0011 0017 0020 0029 0030 0031 0048 0053 0057 0059 0064 0067 0068 0070", "intent", LE, "R1", "SUGGESTION",
    "word + meaning statement, no question; same form as gold-SUGGESTION rows 0005 0022 0027 0032 0047 0060")
add("0085 0086 0087 0088 0091", "intent", LE, "R2", "SUGGESTION", "calm, polite correction: guide s.1 says closer to SUGGESTION; not a question")
add("0095 0102 0108", "intent", LE, "R2", "SUGGESTION", "calm correction/'please fix', no blame: guide prefers SUGGESTION when 50/50")
add("0007 0010 0036 0038 0041 0052 0066 0092", "intent", LE, "R7", IC, "single word / fragment: guide s.5 says INSUFFICIENT_CONTEXT, not a forced label")
# ---- INTENT: ambiguous (guide silent or near-identical rows labelled differently)
add("0014 0037 0058", "intent", AM, "R2", None, "one-line assertion that something is wrong; gold INQUIRY, SUGGESTION or COMPLAINT all defensible")
add("0073 0082 0101 0107", "intent", AM, "R2", None, "blunt criticism of an entry; COMPLAINT vs SUGGESTION depends on how 'tone' is judged")
add("0076 0077 0078 0079 0080 0081 0093", "intent", AM, "R9", None, "offers/proposals to contribute a dictionary: 4 labelled SUGGESTION, 3 INQUIRY")
add("0013 0039 0063 0100", "intent", AM, "R1", None, "gloss/translation request/ornate letter: no clear rule")
# ---- INTENT: clear model errors
add("0098 0099 0104 0105", "intent", ME, "M1", "THANKS_PRAISE", "RCS: 'شكرا' variants predicted SUGGESTION")
add("0074 0094 0096", "intent", ME, "M2", None, "RCS: problem reports predicted THANKS_PRAISE/SUGGESTION")
add("0015 0045 0054 0062", "intent", ME, "M3", "SPAM", "Jev: greeting / dialect remark / quotation predicted THANKS_PRAISE")
add("0097 0012 0033", "intent", ME, "M3", None, "Jev: thanks or off-topic chatter predicted COMPLAINT")
# ---- TOPIC
add("0004", "topic", LE, "R4", "ENTRY_CORRECTION", "criticises an existing definition")
add("0058", "topic", LE, "R3", "ENTRY_CORRECTION", "duplicated definition is a content fault, not a technical fault")
add("0007 0010 0036 0038 0041 0052 0066 0092", "topic", LE, "R7", IC, "single word / fragment")
add("0005 0022 0027 0032 0040 0047 0060 0029 0030 0031 0048 0053 0057 0059 0064 0067 0068 0070", "topic", AM, "R4", None,
    "lexical content only; whether the entry exists is not decidable from the text (guide s.3), v1 Jev sent 19 to ENTRY_CORRECTION")
add("0076 0077 0078 0079 0080 0081 0093", "topic", AM, "R9", None, "dictionary-contribution proposals: guide defines DICTIONARY_SUGGESTION as a platform feature/process idea")
add("0023 0055 0063 0100 0103 0013", "topic", AM, "R9", None, "mixed praise/question/letter content with no single obvious topic")
add("0098 0099 0104 0105 0003", "topic", ME, "M1", "GENERAL", "RCS: plain thanks predicted WORD_SUBMISSION/OFF_TOPIC")
add("0074 0094 0096", "topic", ME, "M2", "TECHNICAL_ISSUE", "RCS: access/display problem predicted GENERAL/WORD_SUBMISSION")
# ---- SENTIMENT
add("0003", "sentiment", LE, "R6", "POSITIVE", "'ابداع منقطع النظير' is praise; gold NEUTRAL")
add("0001 0002 0012 0049 0062 0065", "sentiment", AM, "R5", None,
    "nonsense/test/off-topic labelled NEGATIVE, while ~20 comparable SPAM rows are NEUTRAL; the guide ties NEGATIVE to an irritated/angry tone")
add("0021 0037 0073 0107 0108", "sentiment", AM, "R5", None,
    "blunt error reports labelled NEGATIVE; near-identical 0101/0102/0082/0095 labelled NEUTRAL")
add("0076 0077 0078 0079 0080 0081 0093 0103", "sentiment", AM, "R6", None,
    "formal courteous letters labelled POSITIVE; courtesy formulas alone are not warmth")
add("0005 0011 0020 0030 0031 0053 0067", "sentiment", ME, "M4", "NEUTRAL", "RCS: neutral dictionary prose predicted NEGATIVE")

TASKS = ("intent", "topic", "sentiment")


def validate_entries() -> None:
    seen = set()
    for i, t, b, r, p, _ in E:
        assert t in TASKS and b in (LE, AM, ME) and r in RULES, (i, t, b, r)
        assert (i, t, b) not in seen or True
        if p is not None:
            assert p == IC or p in M.TASK_LABELS[t], (i, t, p)
    by = defaultdict(set)
    for i, t, b, *_ in E:
        by[(i, t)].add(b)
    # a record/task is allowed several buckets only when they are rule-distinct notes; flag true contradictions
    for k, bs in by.items():
        assert not ({LE, ME} <= bs), f"contradictory buckets for {k}"


def load():
    gold = json.loads((OUT_DIR / "gold_labels.local.json").read_text())
    rcs = json.loads((OUT_DIR / "rcs_predictions.json").read_text())
    jev = {json.loads(l)["eval_id"]: json.loads(l) for l in (OUT_DIR / "jev_predictions.jsonl").read_text().splitlines()}
    txt = {json.loads(l)["eval_id"]: " ".join(json.loads(l)["text"].split()) for l in
           (OUT_DIR / "sanitized_inputs.jsonl").read_text(encoding="utf-8").splitlines()}
    return gold, rcs, jev, txt


def sensitivity(gold, preds, name):
    """Metrics under (a) original gold, (b) proposals applied (hypothetical), (c) clean core = drop LABEL_ERROR+AMBIGUOUS."""
    prop = {(i, t): p for i, t, b, r, p, _ in E if b == LE and p and p != IC}
    drop_ic = {(i, t) for i, t, b, r, p, _ in E if b == LE and p == IC}
    contested = {(i, t) for i, t, b, *_ in E if b in (LE, AM)}
    out = {}
    for t in TASKS:
        ids = list(gold)
        g0 = [gold[i][t] for i in ids]
        p0 = [preds[i][t] for i in ids]
        row = {"original": M.score_task(t, g0, p0)}
        ids_b = [i for i in ids if (i, t) not in drop_ic]
        g1 = [prop.get((i, t), gold[i][t]) for i in ids_b]
        row["proposals_applied"] = M.score_task(t, g1, [preds[i][t] for i in ids_b])
        ids_c = [i for i in ids if (i, t) not in contested]
        row["clean_core"] = M.score_task(t, [gold[i][t] for i in ids_c], [preds[i][t] for i in ids_c])
        out[t] = row
    return out


def main() -> None:
    validate_entries()
    sha_before = (Path(OUT_DIR.parents[2] / "data/gold/gold_dataset_final_training.csv").read_bytes())
    gold, rcs, jev, txt = load()
    for i, *_ in E:
        assert i in gold, f"{i} not in the 94"
    L = ["# Label-policy adjudication (development records only; proposals, NOT Gold changes)", "",
         "Source: the 94 sanitized development-validation records, read one by one against LABELING_GUIDE.md. "
         "Every proposed label below is a reviewer suggestion for a human to confirm; Gold was not modified.", ""]
    # counts
    cnt = defaultdict(set)
    for i, t, b, *_ in E:
        cnt[(t, b)].add(i)
    L += ["## Summary: records per task and bucket", "", "| Task | LABEL_ERROR (probable) | AMBIGUOUS | MODEL_ERROR (clear) |", "|---|---|---|---|"]
    for t in TASKS:
        L.append(f"| {t} | {len(cnt[(t, LE)])} | {len(cnt[(t, AM)])} | {len(cnt[(t, ME)])} |")
    L.append("")
    for rule, title in RULES.items():
        items = [e for e in E if e[3] == rule]
        if not items:
            continue
        L += [f"## {rule}. {title}", ""]
        for bucket in (LE, AM, ME):
            sub = [e for e in items if e[2] == bucket]
            if not sub:
                continue
            L.append(f"**{bucket}** ({len({(e[0], e[1]) for e in sub})} record-tasks)")
            for task in TASKS:
                tsub = [e for e in sub if e[1] == task]
                if not tsub:
                    continue
                prop = Counter(e[4] for e in tsub if e[4])
                L.append(f"- {task}: " + ", ".join(e[0][4:] for e in tsub)
                         + (f" -> proposed {dict(prop)}" if prop else "") + f". {tsub[0][5]}")
            # ids only: comment text is deliberately NOT quoted in reports (some development records contain personal data)
            L.append("")
    # sensitivity
    L += ["## Impact: what the labelling issues do to the measured accuracy (hypothetical; Gold untouched)", "",
          "`original` = current Gold; `proposals applied` = LABEL_ERROR rows relabelled as proposed (INSUFFICIENT_CONTEXT rows dropped); "
          "`clean core` = LABEL_ERROR and AMBIGUOUS record-tasks removed (only rows where the guide gives one defensible answer).", ""]
    rep = {}
    for name, preds in (("Frozen RCS", rcs), ("Jev v1", jev)):
        s = sensitivity(gold, preds, name)
        rep[name] = {t: {k: {"n": v["n"], "accuracy": v["accuracy"], "macro_f1": v["macro_f1_supported_classes"]} for k, v in row.items()} for t, row in s.items()}
    L += ["| Task | Model | original (n) acc / macro-F1 | proposals applied (n) | clean core (n) |", "|---|---|---|---|---|"]
    for t in TASKS:
        for name in ("Frozen RCS", "Jev v1"):
            r = rep[name][t]
            cell = lambda k: f"({r[k]['n']}) {r[k]['accuracy']:.2f} / {r[k]['macro_f1']:.2f}"
            L.append(f"| {t} | {name} | {cell('original')} | {cell('proposals_applied')} | {cell('clean_core')} |")
    L += ["", "Reading: the gap between `original` and `clean core` is the part of the measured error that comes from labelling policy, "
          "not from the models. The clean core is a small, hand-selected subset, so it is a diagnostic, not an accuracy estimate.", ""]
    # complaint / TI against guide-consistent rows
    clear_c = ["dev-0073", "dev-0074", "dev-0094", "dev-0107"]
    clear_t = ["dev-0074", "dev-0094", "dev-0096"]
    L += ["## Rare-class recall restricted to guide-consistent gold rows", ""]
    for name, preds in (("Frozen RCS", rcs), ("Jev v1", jev)):
        c = sum(preds[i]["intent"] == "COMPLAINT" for i in clear_c)
        t = sum(preds[i]["topic"] == "TECHNICAL_ISSUE" for i in clear_t)
        L.append(f"- {name}: COMPLAINT {c}/{len(clear_c)} on rows whose COMPLAINT label the guide supports "
                 f"(0073, 0074, 0094, 0107); TECHNICAL_ISSUE {t}/{len(clear_t)} (0074, 0094, 0096).")
    L.append("")
    (OUT_DIR / "label_adjudication_report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT_DIR / "adjudication_proposals.local.json").write_text(json.dumps(
        [dict(eval_id=i, task=t, bucket=b, rule=r, proposed_label=p, note=n) for i, t, b, r, p, n in E],
        ensure_ascii=False, indent=1))
    (OUT_DIR / "adjudication_sensitivity.json").write_text(json.dumps(rep, indent=1))
    sha_after = Path(OUT_DIR.parents[2] / "data/gold/gold_dataset_final_training.csv").read_bytes()
    assert sha_before == sha_after
    print("wrote label_adjudication_report.md;", "gold unchanged:", hashlib.sha256(sha_after).hexdigest()[:16])


if __name__ == "__main__":
    main()
