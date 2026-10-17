"""
Analysis over routing_safety_labels.jsonl (human decisions) joined with
routing_safety_sample.jsonl (classifier evidence). Read-only. Prints a
report; writes nothing.

Honest-by-construction: every statistic below is computed ONLY from
labels that actually exist. If zero cases have been labeled, this
reports that explicitly rather than fabricating a conclusion.

Usage:
    python scripts/offline_eval/routing_safety/analyze.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SAMPLE_PATH = _HERE / "results" / "routing_safety_sample.jsonl"
LABELS_PATH = _HERE / "results" / "routing_safety_labels.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(sample_path: Path = SAMPLE_PATH, labels_path: Path = LABELS_PATH) -> str:
    sample_by_id = {row["eval_id"]: row for row in _load_jsonl(sample_path)}
    labels = _load_jsonl(labels_path)
    decided = [row for row in labels if row.get("decision") in ("APPROPRIATE", "WRONG")]

    lines = []
    lines.append(f"Sample size: {len(sample_by_id)}")
    lines.append(f"Human-labeled (APPROPRIATE/WRONG): {len(decided)}")

    if not decided:
        lines.append("")
        lines.append("INSUFFICIENT DATA: zero human routing labels exist yet.")
        lines.append(f"Run: python scripts/offline_eval/routing_safety/review.py")
        lines.append("No accuracy/threshold conclusion can be drawn without human labels.")
        return "\n".join(lines)

    correct = [row for row in decided if row["decision"] == "APPROPRIATE"]
    wrong = [row for row in decided if row["decision"] == "WRONG"]
    lines.append(f"Route accuracy (of labeled cases): {len(correct)}/{len(decided)} = {len(correct)/len(decided):.1%}")
    lines.append(f"Rejected (WRONG) count: {len(wrong)}")

    off_topic_ids = {cid for cid, row in sample_by_id.items() if row["topic_prediction"] == "OFF_TOPIC_PERSONAL"}
    off_topic_labeled = [row for row in decided if row["eval_id"] in off_topic_ids]
    off_topic_fn = [row for row in off_topic_labeled if row["decision"] == "WRONG"]
    lines.append(f"OFF_TOPIC_PERSONAL cases labeled: {len(off_topic_labeled)}")
    lines.append(f"OFF_TOPIC_PERSONAL false negatives (WRONG route): {len(off_topic_fn)}")

    reason_counts: dict[str, int] = {}
    for row in wrong:
        cat = row.get("reason_category") or "unspecified"
        reason_counts[cat] = reason_counts.get(cat, 0) + 1
    if reason_counts:
        lines.append(f"Rejection reasons: {reason_counts}")

    def _confs(rows):
        return [r["topic_confidence"] for r in rows if r.get("topic_confidence") is not None]

    def _tokens(rows):
        return [r["total_topic_tokens"] for r in rows if r.get("total_topic_tokens") is not None]

    correct_confs, wrong_confs = _confs(correct), _confs(wrong)
    correct_tokens, wrong_tokens = _tokens(correct), _tokens(wrong)

    lines.append("")
    lines.append("Topic-confidence distribution:")
    lines.append(f"  correct routes: n={len(correct_confs)} " + (f"mean={statistics.mean(correct_confs):.3f}" if correct_confs else "n/a"))
    lines.append(f"  wrong routes:   n={len(wrong_confs)} " + (f"mean={statistics.mean(wrong_confs):.3f}" if wrong_confs else "n/a"))

    lines.append("Token-length distribution:")
    lines.append(f"  correct routes: n={len(correct_tokens)} " + (f"mean={statistics.mean(correct_tokens):.0f}" if correct_tokens else "n/a"))
    lines.append(f"  wrong routes:   n={len(wrong_tokens)} " + (f"mean={statistics.mean(wrong_tokens):.0f}" if wrong_tokens else "n/a"))

    lines.append("")
    if len(decided) < 15:
        lines.append(
            f"SAMPLE TOO SMALL ({len(decided)} labeled) to support any confidence threshold recommendation. "
            "No threshold is proposed from this data."
        )
    elif not wrong_confs or not correct_confs:
        lines.append("One of the two groups (correct/wrong) has zero labeled cases -- no threshold comparison is possible.")
    else:
        overlap = max(0.0, min(max(correct_confs), max(wrong_confs)) - max(min(correct_confs), min(wrong_confs)))
        lines.append(
            f"Confidence-range overlap between correct and wrong groups: {overlap:.3f}. "
            "A threshold is only plausibly useful if this overlap is small relative to the group ranges -- "
            "judge this from the raw numbers above, not from this heuristic alone."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze())
