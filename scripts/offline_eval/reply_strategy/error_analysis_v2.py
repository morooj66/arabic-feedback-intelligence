"""
Read-only forensic analysis over the completed V2 blinded A/B evaluation
(results/strategy_v2_ab_cases.jsonl + results/strategy_v2_ab_labels.jsonl).

Does NOT call Gemini, does NOT write to any database, does NOT modify
the source Excel, does NOT modify any production module, and does NOT
modify the existing cases/labels files.

The per-case WIN/LOSS REASON tags below (`ERROR_TAGS`) are a fixed,
recorded analyst judgment made by reading all 25 unblinded case pairs
once (see results/strategy_v2_error_analysis.md for the full reasoning
per case) -- classifying "why did the human prefer A or B" is a
qualitative read of two short Arabic replies, not something a
determinstic string rule can safely automate. What IS deterministic and
tested here is: (1) the join/unblinding of cases+labels+tags, and (2)
the aggregation math over that fixed, recorded dataset. Re-running this
module against the same three inputs always reproduces the same table.

Usage:
    python scripts/offline_eval/reply_strategy/error_analysis_v2.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CASES_PATH = _HERE / "results" / "strategy_v2_ab_cases.jsonl"
LABELS_PATH = _HERE / "results" / "strategy_v2_ab_labels.jsonl"
REPORT_PATH = _HERE / "results" / "strategy_v2_error_analysis.md"

TAXONOMY = (
    "OVERPLANNED_SIMPLE_REPLY",
    "GENERIC_ACKNOWLEDGMENT",
    "EVALUATOR_VOICE",
    "OVERPRAISE",
    "UNSUPPORTED_FACTUAL_FRAMING",
    "MISSED_CORE_POINT",
    "GOOD_CONTENT_ENGAGEMENT",
    "GOOD_CLARIFICATION",
    "INQUIRY_WITHOUT_GROUNDING",
    "STRATEGY_MISMATCH",
    "SOURCE_CLASSIFICATION_MISMATCH",
    "CONTEXT_EXTRACTION_SUSPECT",
    "OTHER",
)

# source_row_index -> (primary_taxonomy_tag, one-line rationale)
# Recorded once by reading all 25 unblinded case pairs; see the
# generated report's per-case table for the reply text each rationale
# refers to.
ERROR_TAGS: dict[int, tuple[str, str]] = {
    1: ("OTHER", "Both substantively engage «اعتبار»; near-tie, A marginally more precise on the meaning contrast."),
    2: ("GOOD_CONTENT_ENGAGEMENT", "B engages the specific collocation-vs-syntax distinction cleanly."),
    9: ("OTHER", "Near-identical paraphrases of the same acknowledgment."),
    10: ("GENERIC_ACKNOWLEDGMENT", "B drops the specific 'wrongly attributed to the Quran' detail A keeps."),
    15: ("OVERPLANNED_SIMPLE_REPLY", "B is more formal/literary ('ثنائكم') than warranted for a one-word compliment."),
    16: ("GOOD_CONTENT_ENGAGEMENT", "B is concise; A adds redundant generic praise not present in the source comment."),
    18: ("OTHER", "Both engage the same specific point («إنسانيًا» ambiguity); near-tie."),
    19: ("OTHER", "Near-identical paraphrases of the same linguistic point."),
    22: ("OVERPLANNED_SIMPLE_REPLY", "B adds an unneeded 'لإثراء المعجم' flourish for an 8-character submission."),
    25: ("OTHER", "Both give a generic, near-equal invitation to elaborate."),
    63: ("GOOD_CONTENT_ENGAGEMENT", "B engages the hadith citation directly; note 'إثراء المعجم' phrasing is borderline (see report)."),
    67: ("INQUIRY_WITHOUT_GROUNDING", "No Official Knowledge exists for this etymology; both variants safely acknowledge the visitor's OWN citation without endorsing it. B more concise."),
    84: ("INQUIRY_WITHOUT_GROUNDING", "No confirmation of a specialized dictionary is available; both acknowledge the request safely. A warmer in tone."),
    92: ("MISSED_CORE_POINT", "Multi-item word list; key_focus fallback gives no single anchor, so BOTH variants stay generic instead of engaging any one entry."),
    112: ("CONTEXT_EXTRACTION_SUSPECT", "B names 'ابرايل الله' as if it were the submitted word -- traced to identifying_context (original_title), not the visitor's actual comment. See forensic section."),
    146: ("OTHER", "Both restate the same acknowledgment of the definition; near-tie."),
    180: ("OTHER", "Both acknowledge the embedded feature suggestion; near-tie, A slightly warmer."),
    211: ("GOOD_CONTENT_ENGAGEMENT", "Both handle the vocalization note correctly; B phrasing preferred. (Underlying COMPLAINT label is itself questionable -- see classification observations.)"),
    362: ("SOURCE_CLASSIFICATION_MISMATCH", "Comment is a lament/supplication, not praise; THANKS_PRAISE is a likely mislabel. Notably, BOTH replies ignored the derived ACKNOWLEDGE_PRAISE strategy and responded as if to a concern -- generation did not blindly follow a bad strategy input."),
    603: ("GOOD_CONTENT_ENGAGEMENT", "B retains the specific morphological point (اسم الذات vs اسم المكان) that A only gestures at."),
    845: ("GOOD_CLARIFICATION", "Sparse/near-empty submission ('حُرّ' with no filled-in meaning); A correctly asks for clarification, B gives a generic acknowledgment instead."),
    987: ("GOOD_CONTENT_ENGAGEMENT", "A's loss reason is EVALUATOR_VOICE + OVERPRAISE ('يمثل طرحكم ... إثراءً مهماً للمفاهيم اللغوية المتخصصة'); B is a direct, natural thanks referencing the actual definition content."),
    991: ("GENERIC_ACKNOWLEDGMENT", "B drops the specific quoted example (the الرفق/زان shahid) that A retains."),
    1074: ("GOOD_CONTENT_ENGAGEMENT", "Both engage the etymology of «ديالا» well; near-tie, B marginally more specific."),
    1191: ("GOOD_CONTENT_ENGAGEMENT", "Real quoted key_focus ('الذَّكالي') drives specific engagement in both; B marginally more complete."),
}

assert set(tag for tag, _ in ERROR_TAGS.values()) <= set(TAXONOMY)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_joined_rows(cases_path: Path = CASES_PATH, labels_path: Path = LABELS_PATH) -> list[dict]:
    """Deterministic join: case metadata + human winner_variant (already
    unblinded by review_ab_v2.py at label-write time) + the recorded
    error-taxonomy tag, sorted by source_row_index for reproducibility."""
    cases = {c["eval_id"]: c for c in _load_jsonl(cases_path)}
    labels = {l["eval_id"]: l for l in _load_jsonl(labels_path)}

    rows = []
    for eval_id, case in cases.items():
        label = labels.get(eval_id)
        if label is None:
            continue
        row_index = case["source_row_index"]
        tag, rationale = ERROR_TAGS.get(row_index, ("OTHER", "not yet tagged"))
        rows.append({
            "source_row_index": row_index,
            "eval_id": eval_id,
            "winner_variant": label["winner_variant"],
            "primary_intent": case["primary_intent"],
            "topic_subtype": case["topic_subtype"],
            "strategy_goal": case.get("strategy_goal"),
            "strategy_mode": case.get("strategy_mode"),
            "strategy_depth": case.get("strategy_depth"),
            "strategy_key_focus": case.get("strategy_key_focus"),
            "comment_preview": case.get("comment_preview"),
            "reply_a": case.get("reply_a"),
            "reply_b": case.get("reply_b"),
            "tag": tag,
            "rationale": rationale,
        })
    rows.sort(key=lambda r: r["source_row_index"])
    return rows


def aggregate_by_tag(rows: list[dict]) -> Counter:
    return Counter(r["tag"] for r in rows)


def aggregate_by_field(rows: list[dict], field: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for r in rows:
        key = r.get(field) or "UNKNOWN"
        out.setdefault(key, Counter())[r["winner_variant"]] += 1
    return out


def win_counts(rows: list[dict]) -> Counter:
    return Counter(r["winner_variant"] for r in rows)


if __name__ == "__main__":
    joined = load_joined_rows()
    print(f"{len(joined)} labeled cases")
    print("Win counts:", dict(win_counts(joined)))
    print("Tag counts:", dict(aggregate_by_tag(joined)))
