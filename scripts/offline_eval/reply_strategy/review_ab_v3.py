"""
Human A/B review CLI for the Response-Strategy V3 (selective-gate) experiment.

Does NOT read or write V1/V2 files (results/strategy_ab_*,
results/strategy_v2_ab_*). Reads results/strategy_v3_ab_cases.jsonl,
writes results/strategy_v3_ab_labels.jsonl.

Shows each eligible case's two replies (baseline vs V3) in a per-case
RANDOMIZED (but deterministic, reproducible) order labeled "Reply
1"/"Reply 2" -- which one is baseline vs V3, and whether V3's gate was
even open for this case, is not shown until after the decision is
recorded, so the label itself never hints at an answer. This applies
identically to gate-negative cases (V3 should be indistinguishable from
baseline there) and gate-positive cases.

[V] View full redacted comment -- same as the routing-safety and V2
reviewers: looks up the full original comment (read-only, by
source_row_index) and prints it through the SAME redact_for_generation()
safeguard production uses, purely to stdout. Never written to the
labels file; after viewing, the reviewer is asked the same question
again for the SAME case.

Resumable: cases already decided are skipped on a re-run; SKIP does not
count as decided. EVALUATION ONLY -- writes only to
results/strategy_v3_ab_labels.jsonl (local, append-only). No database
write of any kind.

Usage:
    python scripts/offline_eval/reply_strategy/review_ab_v3.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parents[2]))

CASES_PATH = _HERE / "results" / "strategy_v3_ab_cases.jsonl"
LABELS_PATH = _HERE / "results" / "strategy_v3_ab_labels.jsonl"

_CHOICE_LABEL = {"A": "REPLY_1", "B": "REPLY_2", "T": "TIE", "N": "NEITHER"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _already_decided(labels: list[dict]) -> set[str]:
    return {row["eval_id"] for row in labels if row.get("decision") in ("REPLY_1", "REPLY_2", "TIE", "NEITHER")}


def _swap_for(eval_id: str) -> bool:
    """Deterministic per-case coin flip -- reproducible across runs,
    not revealed to the reviewer."""
    return random.Random(eval_id).choice([True, False])


def _print_case(index: int, total: int, case: dict, swap: bool) -> None:
    reply_1 = case["reply_v3"] if swap else case["reply_a"]
    reply_2 = case["reply_a"] if swap else case["reply_v3"]
    print(f"\nCase {index} / {total}  (eval_id={case['eval_id']})")
    print("Visitor comment (redacted preview):")
    print(case["comment_preview"] or "(empty)")
    print(f"Intent: {case['primary_intent']}  Topic: {case['topic_subtype']}  Sentiment: {case['sentiment']}")
    print(f"Route: {case['routing_decision']}")
    print("\nReply 1:")
    print(reply_1 or "(no reply / validation failed)")
    print("\nReply 2:")
    print(reply_2 or "(no reply / validation failed)")


def _prompt_choice(prompt_fn: Callable[[str], str]) -> str:
    while True:
        raw = prompt_fn(
            "[A] Reply 1 better   [B] Reply 2 better   [T] Tie   [N] Neither acceptable   "
            "[V] View full redacted comment   [S] Skip   [Q] Quit\n> "
        ).strip().upper()
        if raw in ("A", "B", "T", "N", "V", "S", "Q"):
            return raw
        print("الرجاء اختيار A أو B أو T أو N أو V أو S أو Q.")


def _build_default_comment_lookup() -> Callable[[int], str]:
    """Lazily loads the source workbook ONCE, read-only. Never writes
    anything back to it; the raw text is only ever passed through
    redact_for_generation() and printed, never stored."""
    from config import settings
    from rcs.ingest.loader import load_raw

    source_file = _HERE.parents[2] / "data" / "raw" / "historical-export.xlsx"
    df = load_raw(source_file, settings.SOURCE_COLUMNS)
    by_row = {int(row["source_row_index"]): str(row["original_comment"]) for _, row in df.iterrows()}
    return lambda source_row_index: by_row.get(source_row_index, "")


def _show_full_redacted_comment(case: dict, lookup_fn: Callable[[int], str]) -> None:
    from rcs.response_generation.prompt import redact_for_generation

    raw_text = lookup_fn(case.get("source_row_index"))
    redacted = redact_for_generation(raw_text) if raw_text else ""
    print("\n--- Full comment (redacted; not persisted) ---")
    print(redacted or "(unavailable)")
    print("--- end of full comment ---\n")


def run_review(
    cases_path: Path = CASES_PATH,
    labels_path: Path = LABELS_PATH,
    prompt_fn: Callable[[str], str] = input,
    comment_lookup_fn: Optional[Callable[[int], str]] = None,
) -> Path:
    all_cases = _load_jsonl(cases_path)
    cases = [c for c in all_cases if c.get("reply_a") is not None and c.get("reply_v3") is not None]
    if not cases:
        raise FileNotFoundError(f"No eligible A/B cases found at {cases_path} -- run experiment_v3.py first.")
    existing_labels = _load_jsonl(labels_path)
    decided = _already_decided(existing_labels)
    pending = [c for c in cases if c["eval_id"] not in decided]
    print(f"{len(cases)} eligible cases, {len(decided)} already decided, {len(pending)} pending.")

    lookup_cache: list[Callable[[int], str]] = [comment_lookup_fn] if comment_lookup_fn else []

    def _lookup(source_row_index: int) -> str:
        if not lookup_cache:
            lookup_cache.append(_build_default_comment_lookup())
        return lookup_cache[0](source_row_index)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(pending, start=1):
        swap = _swap_for(case["eval_id"])
        _print_case(i, len(pending), case, swap)
        choice = _prompt_choice(prompt_fn)
        while choice == "V":
            _show_full_redacted_comment(case, _lookup)
            choice = _prompt_choice(prompt_fn)
        if choice == "Q":
            print("تم إنهاء الجلسة.")
            break
        if choice == "S":
            continue

        shown_choice = _CHOICE_LABEL[choice]
        if shown_choice == "REPLY_1":
            winner_variant = "V3" if swap else "A"
        elif shown_choice == "REPLY_2":
            winner_variant = "A" if swap else "V3"
        else:
            winner_variant = shown_choice  # TIE or NEITHER

        row = {
            "eval_id": case["eval_id"],
            "decision": shown_choice,
            "winner_variant": winner_variant,
            "reviewed_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Human A/B review of Response-Strategy V3 replies.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args()
    run_review(args.cases, args.labels)


if __name__ == "__main__":
    main()
