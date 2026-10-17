"""
Human review CLI for the long-comment routing-safety sample.

Shows ONE sampled case at a time (from sample.py's output) and asks for
a judgment on ROUTE APPROPRIATENESS ONLY -- not whether every individual
classifier label (intent/topic/sentiment) is perfect. Resumable: cases
already decided APPROPRIATE or WRONG in a previous run are not asked
again; SKIP does not mark a case as decided, so it will be offered again
on the next run.

EVALUATION ONLY. Reads sample.py's JSONL (already privacy-safe -- only a
short redacted preview, no raw PII). Writes ONLY to
results/routing_safety_labels.jsonl (append-only). No database write of
any kind, anywhere in this file.

[V] View full redacted comment -- an optional, non-deciding action
available alongside A/R/S/Q. It looks up the full original comment
(read-only, by source_row_index) and prints it through the SAME
redact_for_generation() safeguard production uses, purely to stdout.
The full comment is NEVER written to the labels file or anywhere else;
after viewing, the reviewer is asked A/R/S/Q again for the SAME case.

Usage:
    python scripts/offline_eval/routing_safety/review.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parents[2]))

SAMPLE_PATH = _HERE / "results" / "routing_safety_sample.jsonl"
LABELS_PATH = _HERE / "results" / "routing_safety_labels.jsonl"

_REASON_CATEGORIES = {
    "1": "relevant_linguistic_content_incorrectly_dropped",
    "2": "off_topic_content_incorrectly_allowed",
    "3": "insufficient_content_error",
    "4": "other",
}

_DECISION_LABEL = {"A": "APPROPRIATE", "R": "WRONG"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _already_decided(labels: list[dict]) -> set[str]:
    """SKIP is deliberately NOT included -- a skipped case is offered
    again on the next run, unlike APPROPRIATE/WRONG which are final."""
    return {row["eval_id"] for row in labels if row.get("decision") in ("APPROPRIATE", "WRONG")}


def _print_case(index: int, total: int, case: dict) -> None:
    print(f"\nCase {index} / {total}  (eval_id={case['eval_id']})")
    print("Comment preview (redacted, truncated):")
    print(case["comment_preview"] or "(empty)")
    print(f"Topic: {case['topic_prediction']} (p={case['topic_confidence']})", end="")
    if case.get("topic_second_best"):
        print(f"  | 2nd: {case['topic_second_best']} (p={case['topic_second_best_confidence']})")
    else:
        print()
    print(f"Intent: {case['intent_prediction']} (p={case['intent_confidence']})")
    print(f"Sentiment: {case['sentiment_prediction']}")
    print(f"Total Topic tokens (untruncated): {case['total_topic_tokens']}")
    print(f"Route: {case['routing_decision']} ({case['routing_reason_code']})")
    print(f"Would call Gemini: {case['would_call_gemini']}")
    if case.get("window_beginning_topic"):
        print(
            f"Windows -- beginning: {case['window_beginning_topic']}({case['window_beginning_confidence']})  "
            f"middle: {case['window_middle_topic']}({case['window_middle_confidence']})  "
            f"ending: {case['window_ending_topic']}({case['window_ending_confidence']})  "
            f"averaged-top: {case['window_averaged_top_topic']}({case['window_averaged_top_confidence']})"
        )


def _prompt_decision(prompt_fn: Callable[[str], str]) -> str:
    while True:
        raw = prompt_fn(
            "[A] Appropriate route   [R] Wrong route   [V] View full redacted comment   "
            "[S] Skip   [Q] Quit\n> "
        ).strip().upper()
        if raw in ("A", "R", "V", "S", "Q"):
            return raw
        print("الرجاء اختيار A أو R أو V أو S أو Q.")


def _build_default_comment_lookup() -> Callable[[int], str]:
    """Lazily loads the source workbook ONCE, read-only, and returns a
    source_row_index -> raw original_comment lookup. Never writes
    anything back to the workbook; the raw text it returns is only ever
    passed through redact_for_generation() and printed, never stored."""
    from config import settings
    from rcs.ingest.loader import load_raw

    source_file = _HERE.parents[2] / "data" / "raw" / "historical-export.xlsx"
    df = load_raw(source_file, settings.SOURCE_COLUMNS)
    by_row = {int(row["source_row_index"]): str(row["original_comment"]) for _, row in df.iterrows()}
    return lambda source_row_index: by_row.get(source_row_index, "")


def _show_full_redacted_comment(case: dict, lookup_fn: Callable[[int], str]) -> None:
    """Prints the full comment through the same redaction safeguard used
    in production generation. Prints only -- never returns/persists the
    text anywhere the caller could accidentally write out."""
    from rcs.response_generation.prompt import redact_for_generation

    raw_text = lookup_fn(case["source_row_index"])
    redacted = redact_for_generation(raw_text) if raw_text else ""
    print("\n--- Full comment (redacted; not persisted) ---")
    print(redacted or "(unavailable)")
    print("--- end of full comment ---\n")


def _prompt_reason(prompt_fn: Callable[[str], str]) -> Optional[str]:
    while True:
        raw = prompt_fn(
            "Reason: [1] relevant content dropped  [2] off-topic allowed  "
            "[3] insufficient-content error  [4] other\n> "
        ).strip()
        if raw in _REASON_CATEGORIES:
            return _REASON_CATEGORIES[raw]
        print("الرجاء اختيار 1 أو 2 أو 3 أو 4.")


def run_review(
    sample_path: Path = SAMPLE_PATH,
    labels_path: Path = LABELS_PATH,
    prompt_fn: Callable[[str], str] = input,
    comment_lookup_fn: Optional[Callable[[int], str]] = None,
) -> Path:
    cases = _load_jsonl(sample_path)
    if not cases:
        raise FileNotFoundError(f"No sample found at {sample_path} -- run sample.py first.")
    existing_labels = _load_jsonl(labels_path)
    decided = _already_decided(existing_labels)
    pending = [c for c in cases if c["eval_id"] not in decided]
    print(f"{len(cases)} total cases, {len(decided)} already decided, {len(pending)} pending.")

    # Lazy, cached lookup: only built (and only ever reads the source
    # workbook) if the reviewer actually presses [V] at least once.
    lookup_cache: list[Callable[[int], str]] = [comment_lookup_fn] if comment_lookup_fn else []

    def _lookup(source_row_index: int) -> str:
        if not lookup_cache:
            lookup_cache.append(_build_default_comment_lookup())
        return lookup_cache[0](source_row_index)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(pending, start=1):
        _print_case(i, len(pending), case)
        decision = _prompt_decision(prompt_fn)
        while decision == "V":
            _show_full_redacted_comment(case, _lookup)
            decision = _prompt_decision(prompt_fn)
        if decision == "Q":
            print("تم إنهاء الجلسة.")
            break
        if decision == "S":
            continue
        reason = _prompt_reason(prompt_fn) if decision == "R" else None
        row = {
            "eval_id": case["eval_id"],
            "decision": _DECISION_LABEL[decision],
            "reason_category": reason,
            "topic_confidence": case["topic_confidence"],
            "total_topic_tokens": case["total_topic_tokens"],
            "reviewed_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Human review of long-comment routing appropriateness.")
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args()
    run_review(args.sample, args.labels)


if __name__ == "__main__":
    main()
