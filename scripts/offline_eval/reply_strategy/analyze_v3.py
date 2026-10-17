"""
Analysis over strategy_v3_ab_labels.jsonl (human decisions) joined with
strategy_v3_ab_cases.jsonl (case metadata, including the gate decision
`strategy_enabled`). Read-only. Prints a report; writes nothing. Does
not touch V1/V2 result files.

Unblinding: review_ab_v3.py already stores `winner_variant` ("A" =
baseline/rg-v10, "V3" = the gated Strategy V3 variant, "TIE", "NEITHER")
at label-write time, computed from the per-case deterministic swap --
this module only reads it back.

The key breakdown this analyzer adds beyond analyze_v2.py: results split
by `strategy_enabled` (gate-positive vs gate-negative). For
gate-negative cases, V3's reply is expected to be indistinguishable from
baseline (byte-identical prompt) -- a high TIE rate or near-50/50 split
there is the expected, healthy signal, not evidence "V3 is only as good
as baseline"; a strong V3-wins skew on gate-negative cases would be
suspicious and worth investigating (it would mean the gate is not truly
inert for those cases).

Honest-by-construction: every statistic below is computed ONLY from
labels that actually exist, and is explicitly labeled exploratory human
preference evidence -- no significance claim is made from a small
sample.

Usage:
    python scripts/offline_eval/reply_strategy/analyze_v3.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CASES_PATH = _HERE / "results" / "strategy_v3_ab_cases.jsonl"
LABELS_PATH = _HERE / "results" / "strategy_v3_ab_labels.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(cases_path: Path = CASES_PATH, labels_path: Path = LABELS_PATH) -> str:
    all_cases = _load_jsonl(cases_path)
    cases_by_id = {c["eval_id"]: c for c in all_cases}
    eligible_cases = [c for c in all_cases if c.get("reply_a") is not None and c.get("reply_v3") is not None]

    labels = _load_jsonl(labels_path)
    decided = [
        row for row in labels
        if row.get("decision") in ("REPLY_1", "REPLY_2", "TIE", "NEITHER") and row["eval_id"] in cases_by_id
    ]

    lines = []
    lines.append("EXPLORATORY HUMAN PREFERENCE EVIDENCE -- not a statistical significance claim.")
    lines.append(f"Eligible A/B cases: {len(eligible_cases)}")
    gate_positive_n = sum(1 for c in eligible_cases if c.get("strategy_enabled"))
    gate_negative_n = len(eligible_cases) - gate_positive_n
    lines.append(f"  gate-positive (strategy applied): {gate_positive_n}")
    lines.append(f"  gate-negative (baseline-equivalent): {gate_negative_n}")
    lines.append(f"Total reviewed (decided): {len(decided)}")

    if not decided:
        lines.append("")
        lines.append("INSUFFICIENT DATA: zero human A/B labels exist yet.")
        lines.append(f"Run: python scripts/offline_eval/reply_strategy/review_ab_v3.py")
        return "\n".join(lines)

    baseline_wins = [row for row in decided if row["winner_variant"] == "A"]
    v3_wins = [row for row in decided if row["winner_variant"] == "V3"]
    ties = [row for row in decided if row["winner_variant"] == "TIE"]
    neither = [row for row in decided if row["winner_variant"] == "NEITHER"]
    skipped_or_unreviewed = len(eligible_cases) - len(decided)

    n = len(decided)
    lines.append(f"Baseline (rg-v10) wins: {len(baseline_wins)}/{n} = {len(baseline_wins)/n:.1%}")
    lines.append(f"Strategy V3 wins:       {len(v3_wins)}/{n} = {len(v3_wins)/n:.1%}")
    lines.append(f"Ties:                   {len(ties)}/{n} = {len(ties)/n:.1%}")
    lines.append(f"Neither acceptable:     {len(neither)}/{n} = {len(neither)/n:.1%}")
    lines.append(f"Skipped/unreviewed:     {skipped_or_unreviewed}")

    lines.append("")
    lines.append("--- By gate state (the key V3 question) ---")
    for gate_label, gate_value in (("gate-positive (strategy applied)", True), ("gate-negative (baseline-equivalent)", False)):
        rows = [row for row in decided if bool(cases_by_id[row["eval_id"]].get("strategy_enabled")) == gate_value]
        gn = len(rows)
        if gn == 0:
            lines.append(f"  {gate_label}: n=0")
            continue
        gb = sum(1 for r in rows if r["winner_variant"] == "A")
        gv3 = sum(1 for r in rows if r["winner_variant"] == "V3")
        gt = sum(1 for r in rows if r["winner_variant"] == "TIE")
        gne = sum(1 for r in rows if r["winner_variant"] == "NEITHER")
        lines.append(f"  {gate_label}: n={gn}  baseline={gb}  v3={gv3}  tie={gt}  neither={gne}")
        if not gate_value and gv3 > gb + gt + gne:
            lines.append(
                "    NOTE: V3 wins outnumber all other outcomes combined on gate-negative cases -- "
                "unexpected if the gate is truly inert; worth investigating before drawing conclusions."
            )

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
            gv3 = sum(1 for r in rows if r["winner_variant"] == "V3")
            gt = sum(1 for r in rows if r["winner_variant"] == "TIE")
            gne = sum(1 for r in rows if r["winner_variant"] == "NEITHER")
            lines.append(f"  {key}: n={gn}  baseline={gb}  v3={gv3}  tie={gt}  neither={gne}")

    _breakdown(lambda c: c.get("primary_intent") or "UNKNOWN", "primary_intent")
    _breakdown(lambda c: c.get("topic_subtype") or "UNKNOWN", "topic_subtype")
    _breakdown(lambda c: c.get("strategy_depth") or "UNKNOWN", "response_depth")

    lines.append("")
    lines.append(
        f"Sample size is small (n={n}); this is exploratory human preference evidence only, "
        "not a statistically significant result. No production promotion decision should be "
        "made from this alone."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze())
