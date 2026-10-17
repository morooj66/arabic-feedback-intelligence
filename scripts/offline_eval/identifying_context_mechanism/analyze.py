"""
Analysis over identifying_context_mechanism_labels.jsonl (human
decisions) joined with identifying_context_mechanism_cases.jsonl (case
metadata: comment_class, context_specificity, role). Read-only. Prints a
report; writes nothing.

Reports hijacking rate broken down by comment-vacuousness class and by
context-specificity class, an interaction table between the two, the
count of unrelated (D) failures, and the raw case IDs for every C
(hijacking) result -- exactly what is needed to answer the three
mechanism questions this experiment was built for:
  - did hijacking appear only when context specificity increased?
  - were short/vacuous comments more vulnerable?
  - did substantive comments remain stable?

Every statistic is computed ONLY from labels that actually exist, and
explicitly labeled exploratory -- this experiment has at most 16 labels
possible (18 generated variants minus 2 NONE baselines that are never
independently judged), so no significance claim is ever made.

Usage:
    python scripts/offline_eval/identifying_context_mechanism/analyze.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CASES_PATH = _HERE / "results" / "identifying_context_mechanism_cases.jsonl"
LABELS_PATH = _HERE / "results" / "identifying_context_mechanism_labels.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(cases_path: Path = CASES_PATH, labels_path: Path = LABELS_PATH) -> str:
    cases_by_id = {c["eval_id"]: c for c in _load_jsonl(cases_path)}
    labels = _load_jsonl(labels_path)
    decided = [
        row for row in labels
        if row.get("decision") in ("IMPROVED_RELEVANCE", "NO_MEANINGFUL_EFFECT", "CONTEXT_HIJACKING", "UNACCEPTABLE_OTHER")
        and row["eval_id"] in cases_by_id
    ]

    lines = []
    lines.append("EXPLORATORY MECHANISM EVIDENCE -- not a statistical significance claim (small n by design).")
    lines.append(f"Reviewable decisions: {len(decided)}")

    if not decided:
        lines.append("")
        lines.append("INSUFFICIENT DATA: zero human decisions exist yet.")
        lines.append("Run: python scripts/offline_eval/identifying_context_mechanism/review.py")
        return "\n".join(lines)

    n = len(decided)
    counts = Counter(row["decision"] for row in decided)
    for label in ("IMPROVED_RELEVANCE", "NO_MEANINGFUL_EFFECT", "CONTEXT_HIJACKING", "UNACCEPTABLE_OTHER"):
        lines.append(f"  {label}: {counts.get(label, 0)}/{n}")

    def _hijack_rate_by(key_fn, title):
        lines.append("")
        lines.append(f"--- Hijacking rate by {title} ---")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in decided:
            case = cases_by_id[row["eval_id"]]
            groups[key_fn(case)].append(row)
        for key in sorted(groups):
            rows = groups[key]
            hijacked = sum(1 for r in rows if r["decision"] == "CONTEXT_HIJACKING")
            lines.append(f"  {key}: {hijacked}/{len(rows)}")

    _hijack_rate_by(lambda c: c["comment_class"], "comment-vacuousness class")
    _hijack_rate_by(lambda c: c["context_specificity"], "context-specificity class")

    lines.append("")
    lines.append("--- Interaction table (comment_class x context_specificity): hijacked/total ---")
    interaction: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in decided:
        case = cases_by_id[row["eval_id"]]
        interaction[(case["comment_class"], case["context_specificity"])].append(row)
    for (cclass, spec) in sorted(interaction):
        rows = interaction[(cclass, spec)]
        hijacked = sum(1 for r in rows if r["decision"] == "CONTEXT_HIJACKING")
        lines.append(f"  {cclass:<26} x {spec:<10}: {hijacked}/{len(rows)}")

    unrelated = [row for row in decided if row["decision"] == "UNACCEPTABLE_OTHER"]
    lines.append("")
    lines.append(f"Unrelated failures (D, not identifying-context-attributable): {len(unrelated)}/{n}")
    for row in unrelated:
        lines.append(f"  {row['eval_id']}")

    hijacked_rows = [row for row in decided if row["decision"] == "CONTEXT_HIJACKING"]
    lines.append("")
    lines.append(f"Raw case IDs for every CONTEXT_HIJACKING (C) result ({len(hijacked_rows)}):")
    for row in hijacked_rows:
        case = cases_by_id[row["eval_id"]]
        lines.append(f"  {row['eval_id']}  (comment_class={case['comment_class']}, specificity={case['context_specificity']})")
    if not hijacked_rows:
        lines.append("  (none)")

    # --- The three mechanism questions, answered ONLY from what is present ---
    lines.append("")
    lines.append("--- Mechanism observations (descriptive only, n too small for significance) ---")
    vacuous_rows = [row for row in decided if cases_by_id[row["eval_id"]]["comment_class"] == "VERY_SHORT_NONCOMMITTAL"]
    substantive_rows = [row for row in decided if cases_by_id[row["eval_id"]]["comment_class"] == "SUBSTANTIVE"]
    short_meaningful_rows = [row for row in decided if cases_by_id[row["eval_id"]]["comment_class"] == "SHORT_MEANINGFUL"]
    generic_rows = [row for row in decided if cases_by_id[row["eval_id"]]["context_specificity"] == "GENERIC"]
    specific_rows = [row for row in decided if cases_by_id[row["eval_id"]]["context_specificity"] == "SPECIFIC"]

    vacuous_hijacked = sum(1 for r in vacuous_rows if r["decision"] == "CONTEXT_HIJACKING")
    substantive_hijacked = sum(1 for r in substantive_rows if r["decision"] == "CONTEXT_HIJACKING")
    short_meaningful_hijacked = sum(1 for r in short_meaningful_rows if r["decision"] == "CONTEXT_HIJACKING")
    generic_hijacked = sum(1 for r in generic_rows if r["decision"] == "CONTEXT_HIJACKING")
    specific_hijacked = sum(1 for r in specific_rows if r["decision"] == "CONTEXT_HIJACKING")

    lines.append(
        f"Did hijacking appear only when specificity increased? "
        f"GENERIC hijacked {generic_hijacked}/{len(generic_rows)}; SPECIFIC hijacked {specific_hijacked}/{len(specific_rows)}."
    )
    lines.append(
        f"Were short/vacuous comments more vulnerable? "
        f"VERY_SHORT_NONCOMMITTAL hijacked {vacuous_hijacked}/{len(vacuous_rows)}; "
        f"SHORT_MEANINGFUL hijacked {short_meaningful_hijacked}/{len(short_meaningful_rows)}."
    )
    lines.append(
        f"Did substantive comments remain stable? "
        f"SUBSTANTIVE hijacked {substantive_hijacked}/{len(substantive_rows)}."
    )

    lines.append("")
    lines.append(
        f"Sample size is intentionally small (n={n} decisions across a controlled factorial, not a benchmark); "
        "this is exploratory mechanism-isolation evidence only, not a statistically significant result. "
        "No suppression rule should be designed from this alone without further confirmation."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze())
