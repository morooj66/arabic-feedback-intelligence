#!/usr/bin/env python3
"""
CLI entrypoint for Phase B.2 — AI-assisted gold-label suggestion.

Intended to be run LOCALLY, on a machine whose network can actually reach
generativelanguage.googleapis.com (Cowork's sandbox cannot — its egress
proxy blocks that domain at the allowlist level, confirmed directly; this
is not fixable from inside that sandbox). See PHASE_B2_LOCAL_RUNBOOK.md for
the exact one-command setup.

Deliberately separate from scripts/run_classification.py (Phase 4's live
classification, which writes to review.sqlite/review_items) — this script
only ever reads/writes data/gold/gold_dataset_round1.csv, and only ever
writes to the 7 ai_suggested_* columns, and only ever for rows where
annotation_stage == "ai_assisted_train".

RESUME / IDEMPOTENCY: a target row is skipped if it already has a
non-blank ai_suggested_primary_intent — i.e. it was already successfully
suggested by a prior run. A row that failed last time (network error,
validation failure, etc.) is left blank on failure, so it is automatically
retried the next time this same command is run. Re-running the exact same
command after an interruption (Ctrl-C, lost connection, hit a quota) is
therefore always safe and picks up where it left off — pass --force to
override this and re-suggest rows that already have a value.

CHECKPOINTING: the CSV is rewritten after every row (atomically — written
to a temp file, then renamed over the real path, so a crash mid-write
never corrupts data/gold/gold_dataset_round1.csv) rather than only once at
the end. A run interrupted after row 150 of 300 has already saved those
150 suggestions to disk.

Refuses to run without a real GEMINI_API_KEY. Loads .env automatically
(simple KEY=VALUE parser below, not python-dotenv, no new dependency) if
GEMINI_API_KEY isn't already set in the environment — so the one command
in the runbook is exactly `python3 scripts/run_ai_suggestions.py`, nothing
to export manually first.

Before EVERY checkpoint write, this script re-verifies (not just trusts
its own bookkeeping) that:
  - every gold_*/labeler_*/comment_id/split/annotation_stage/
    is_few_shot_example/comment_text value is byte-identical to the
    original input file, for every row (Phase B.2 must never touch a gold
    label);
  - every row not yet processed this run has its ai_suggested_* columns
    unchanged from the input;
  - every human_val/human_test/human_train_seed row has ALL SEVEN
    ai_suggested_* columns still blank — a second, independent check on
    top of the query filter, not a restatement of it.

Any of these failing raises AssertionError and that checkpoint's write is
skipped (prior checkpoints already on disk are unaffected).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_if_present(env_path: Path) -> None:
    """Minimal .env loader: sets os.environ for KEY=VALUE lines that
    aren't already set in the real environment. Silently does nothing if
    the file doesn't exist. Never overrides a value already set in the
    environment (a real `export` always wins over .env)."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_if_present(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from rcs.evaluation.genai_client import GenAIClient  # noqa: E402
from rcs.evaluation.ai_suggester import suggest_row  # noqa: E402
from rcs.evaluation.few_shot import select_few_shot_examples  # noqa: E402
from rcs.evaluation.gold_dataset import AI_SUGGESTED_COLUMNS  # noqa: E402

GOLD_DIR = settings.DATA_DIR / "gold"
GOLD_CSV = GOLD_DIR / "gold_dataset_round1.csv"

_PRESERVED_COLUMNS = [
    "comment_id", "comment_text", "gold_primary_intent", "gold_topic_subtype",
    "gold_secondary_intent", "gold_subcategory", "gold_sentiment",
    "gold_annotation_status", "labeler_notes", "labeled_by", "labeled_at",
    "split", "annotation_stage", "is_few_shot_example",
]


def _safety_check(original: pd.DataFrame, updated: pd.DataFrame, not_yet_processed_ids: set[str]) -> None:
    orig_idx = original.set_index("comment_id")
    upd_idx = updated.set_index("comment_id")

    if not orig_idx[_PRESERVED_COLUMNS[1:]].equals(upd_idx[_PRESERVED_COLUMNS[1:]]):
        raise AssertionError("a gold_*/labeler_*/split/annotation_stage/comment_text value changed — refusing to write")

    if not_yet_processed_ids:
        ids = list(not_yet_processed_ids)
        if not orig_idx.loc[ids, list(AI_SUGGESTED_COLUMNS)].equals(upd_idx.loc[ids, list(AI_SUGGESTED_COLUMNS)]):
            raise AssertionError("an ai_suggested_* value changed on a row not yet processed this run")

    protected = updated[updated["annotation_stage"] != "ai_assisted_train"]
    for col in AI_SUGGESTED_COLUMNS:
        written = protected[col].astype(str).str.strip() != ""
        if written.any():
            raise AssertionError(
                f"{col} has {int(written.sum())} value(s) on non-ai_assisted_train row(s) — "
                f"human_val/human_test/human_train_seed must never receive AI suggestions"
            )


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, path)  # atomic on the same filesystem


def run(
    limit: int | None = None,
    low_confidence_threshold: float = 0.6,
    client=None,
    gold_csv_path: Path | None = None,
    force: bool = False,
    sleep_seconds: float = 0.0,
    progress_every: int = 20,
) -> dict:
    """
    `client`/`gold_csv_path` are injectable purely for testing (see
    tests/test_ai_suggestion.py) — the CLI entrypoint below never passes
    them, so a real run always constructs a real GenAIClient (requiring a
    real GEMINI_API_KEY) against the real gold_dataset_round1.csv.
    """
    csv_path = gold_csv_path or GOLD_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    original = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    few_shot_examples = select_few_shot_examples(original)
    if not few_shot_examples:
        raise RuntimeError(
            "no eligible few-shot examples found (need is_few_shot_example=True AND "
            "gold_annotation_status=LABELED rows) — refusing to run"
        )

    all_targets = original[original["annotation_stage"] == "ai_assisted_train"].copy()
    if force:
        targets = all_targets
        already_done_count = 0
    else:
        already_done_mask = all_targets["ai_suggested_primary_intent"].astype(str).str.strip() != ""
        already_done_count = int(already_done_mask.sum())
        targets = all_targets[~already_done_mask]
    if limit is not None:
        targets = targets.head(limit)

    if client is None:
        client = GenAIClient(api_key=settings.GEMINI_API_KEY, model_name=settings.AI_SUGGESTION_MODEL_NAME)
    model_tag = f"{settings.AI_SUGGESTION_MODEL_NAME}:{settings.AI_SUGGESTION_PROMPT_VERSION}"

    updated = original.copy()
    succeeded, failed = [], []
    target_ids_ordered = list(targets["comment_id"])
    remaining_ids = set(target_ids_ordered)

    print(f"already_suggested (skipped, resume): {already_done_count}")
    print(f"to_process_this_run: {len(target_ids_ordered)}")

    for n, (_, row) in enumerate(targets.iterrows(), start=1):
        cid = row["comment_id"]
        outcome = suggest_row(row.to_dict(), few_shot_examples, client, max_retries=settings.GEMINI_MAX_RETRIES)

        if outcome.success:
            idx = updated.index[updated["comment_id"] == cid][0]
            updated.at[idx, "ai_suggested_primary_intent"] = outcome.primary_intent
            updated.at[idx, "ai_suggested_topic_subtype"] = outcome.topic_subtype
            updated.at[idx, "ai_suggested_secondary_intent"] = outcome.secondary_intent or ""
            updated.at[idx, "ai_suggested_subcategory"] = outcome.subcategory or ""
            updated.at[idx, "ai_suggested_sentiment"] = outcome.sentiment
            updated.at[idx, "ai_suggested_confidence"] = str(outcome.confidence)
            updated.at[idx, "ai_suggestion_model"] = model_tag
            succeeded.append({
                "comment_id": cid, "primary_intent": outcome.primary_intent,
                "topic_subtype": outcome.topic_subtype, "sentiment": outcome.sentiment,
                "confidence": outcome.confidence,
            })
        else:
            failed.append({"comment_id": cid, "error": outcome.error})

        remaining_ids.discard(cid)

        _safety_check(original, updated, remaining_ids)
        _atomic_write_csv(updated, csv_path)

        if progress_every and n % progress_every == 0:
            print(f"progress: {n}/{len(target_ids_ordered)} processed "
                  f"({len(succeeded)} succeeded, {len(failed)} failed so far)")

        if sleep_seconds:
            time.sleep(sleep_seconds)

    intent_dist = Counter(s["primary_intent"] for s in succeeded)
    topic_dist = Counter(s["topic_subtype"] for s in succeeded)
    sentiment_dist = Counter(s["sentiment"] for s in succeeded)
    low_confidence = [s for s in succeeded if s["confidence"] < low_confidence_threshold]

    return {
        "few_shot_examples_used": len(few_shot_examples),
        "already_suggested_skipped": already_done_count,
        "targets_attempted": len(target_ids_ordered),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "failures": failed,
        "primary_intent_distribution": dict(intent_dist),
        "topic_subtype_distribution": dict(topic_dist),
        "sentiment_distribution": dict(sentiment_dist),
        "low_confidence_threshold": low_confidence_threshold,
        "low_confidence_count": len(low_confidence),
        "low_confidence_examples": low_confidence[:20],
        "safety_checks": "all passed (re-verified before every checkpoint write)",
        "output_csv": str(csv_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B.2: AI-assisted gold-label suggestion (live Gemini calls). Run locally — see PHASE_B2_LOCAL_RUNBOOK.md.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N remaining ai_assisted_train rows this invocation.")
    parser.add_argument("--force", action="store_true", help="Re-suggest rows that already have an ai_suggested_primary_intent (default: skip them — this is what makes re-running the same command a safe resume).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls (rate-limit friendliness).")
    parser.add_argument("--progress-every", type=int, default=20, help="Print a progress line every N processed rows (0 to disable).")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6, help="Reporting-only cutoff for flagging low-confidence suggestions; not a validated production threshold.")
    args = parser.parse_args()

    result = run(
        limit=args.limit,
        low_confidence_threshold=args.low_confidence_threshold,
        force=args.force,
        sleep_seconds=args.sleep,
        progress_every=args.progress_every,
    )
    print("\n--- FINAL REPORT ---")
    for key, value in result.items():
        print(f"{key}: {value}")
