"""
Analysis over strategy_v2_ab_labels.jsonl (human decisions) joined with
strategy_v2_ab_cases.jsonl (case metadata). Read-only. Prints a report;
writes nothing. Does not touch V1's result files.

Unblinding: review_ab_v2.py already stores `winner_variant` ("A" =
baseline/rg-v10, "B" = Strategy V2, "TIE", "NEITHER") at label-write
time, computed from the per-case deterministic swap -- this module does
not need to re-derive the swap, only read `winner_variant` back.

Honest-by-construction: every statistic below is computed ONLY from
labels that actually exist, and is explicitly labeled exploratory human
preference evidence -- no significance claim is made from a small
sample.

Usage:
    python scripts/offline_eval/reply_strategy/analyze_v2.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CASES_PATH = _HERE / "results" / "strategy_v2_ab_cases.jsonl"
LABELS_PATH = _HERE / "results" / "strategy_v2_ab_labels.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(cases_path: Path = CASES_PATH, labels_path: Path = LABELS_PATH) -> str:
    all_cases = _load_jsonl(cases_path)
    cases_by_id = {c["eval_id"]: c for c in all_cases}
    eligible_cases = [c for c in all_cases if c.get("reply_a") is not None and c.get("reply_b") is not None]

    labels = _load_jsonl(labels_path)
    decided = [
        row for row in labels
        if row.get("decision") in ("REPLY_1", "REPLY_2", "TIE", "NEITHER") and row["eval_id"] in cases_by_id
    ]

    lines = []
    lines.append("EXPLORATORY HUMAN PREFERENCE EVIDENCE -- not a statistical significance claim.")
    lines.append(f"Eligible A/B cases: {len(eligible_cases)}")
    lines.append(f"Total reviewed (decided): {len(decided)}")

    if not decided:
        lines.append("")
        lines.append("INSUFFICIENT DATA: zero human A/B labels exist yet.")
        lines.append(f"Run: python scripts/offline_eval/reply_strategy/review_ab_v2.py")
        return "\n".join(lines)

    baseline_wins = [row for row in decided if row["winner_variant"] == "A"]
    strategy_wins = [row for row in decided if row["winner_variant"] == "B"]
    ties = [row for row in decided if row["winner_variant"] == "TIE"]
    neither = [row for row in decided if row["winner_variant"] == "NEITHER"]
    skipped_or_unreviewed = len(eligible_cases) - len(decided)

    n = len(decided)
    lines.append(f"Baseline (rg-v10) wins: {len(baseline_wins)}/{n} = {len(baseline_wins)/n:.1%}")
    lines.append(f"Strategy V2 wins:       {len(strategy_wins)}/{n} = {len(strategy_wins)/n:.1%}")
    lines.append(f"Ties:                   {len(ties)}/{n} = {len(ties)/n:.1%}")
    lines.append(f"Neither acceptable:     {len(neither)}/{n} = {len(neither)/n:.1%}")
    lines.append(f"Skipped/unreviewed:     {skipped_or_unreviewed}")

    def _breakdown(key_fn, title):
        lines.append("")
        lines.append(f"--- By {title} ---")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in decided:
            case = cases_by_id[row["eval_id"]]
            groups[key_fn(case)].append(row)
        for key in sorted(groups):
            rows = groups[key]
            gn = len(rows)
            gb = sum(1 for r in rows if r["winner_variant"] == "A")
            gs = sum(1 for r in rows if r["winner_variant"] == "B")
            gt = sum(1 for r in rows if r["winner_variant"] == "TIE")
            gne = sum(1 for r in rows if r["winner_variant"] == "NEITHER")
            lines.append(f"  {key}: n={gn}  baseline={gb}  strategy_v2={gs}  tie={gt}  neither={gne}")

    _breakdown(lambda c: c.get("primary_intent") or "UNKNOWN", "primary_intent")
    _breakdown(lambda c: c.get("topic_subtype") or "UNKNOWN", "topic_subtype")
    _breakdown(lambda c: c.get("strategy_goal") or "UNKNOWN", "strategy communicative_goal")
    _breakdown(lambda c: c.get("strategy_depth") or "UNKNOWN", "response_depth")
    _breakdown(lambda c: c.get("strategy_mode") or "UNKNOWN", "response_mode")

    lines.append("")
    lines.append("--- Difficult linguistic subset: SUGGESTION + ENTRY_CORRECTION ---")
    difficult = [
        row for row in decided
        if cases_by_id[row["eval_id"]].get("primary_intent") == "SUGGESTION"
        and cases_by_id[row["eval_id"]].get("topic_subtype") == "ENTRY_CORRECTION"
    ]
    if not difficult:
        lines.append("  No reviewed cases in this subset yet.")
    else:
        dn = len(difficult)
        db = sum(1 for r in difficult if r["winner_variant"] == "A")
        ds = sum(1 for r in difficult if r["winner_variant"] == "B")
        dt_ = sum(1 for r in difficult if r["winner_variant"] == "TIE")
        dne = sum(1 for r in difficult if r["winner_variant"] == "NEITHER")
        lines.append(f"  n={dn}  baseline={db}  strategy_v2={ds}  tie={dt_}  neither={dne}")
        for row in difficult:
            case = cases_by_id[row["eval_id"]]
            lines.append(f"    row={case.get('source_row_index')}  winner={row['winner_variant']}")

    lines.append("")
    lines.append(
        f"Sample size is small (n={n}); this is exploratory human preference evidence only, "
        "not a statistically significant result. No production promotion decision should be "
        "made from this alone."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze())
