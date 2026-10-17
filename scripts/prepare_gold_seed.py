#!/usr/bin/env python3
"""
Phase B.1 — REVISED_ARCHITECTURE_PLAN.md v2, Section 6/7.

Extends the FROZEN data/gold/gold_dataset_round1.csv (500 rows, split
already assigned in Phase B) with the new leakage-conscious annotation
columns, and selects the ~50-row human_train_seed from the existing
`train` split only.

Guarantees, enforced by assertion before anything is written:
  - every existing comment_id is preserved, none added or removed
  - every existing `split` value is preserved exactly (train stays train,
    val stays val, test stays test — no row moves between splits)
  - zero ai_suggested_* values written (that's Phase B.2)
  - human_train_seed is drawn ONLY from `train`, never val/test
  - no val/test row is ever marked as a few-shot example

No model calls. No training. No inference. Read-only against
review.sqlite (for original_type/age_group metadata used to stratify the
seed selection — same proxy covariates as the original 500-row sample).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from rcs.evaluation.gold_dataset import (  # noqa: E402
    AI_SUGGESTED_COLUMNS,
    assign_annotation_stage,
    verify_gold_workbook_invariants,
)
from rcs.evaluation.sampling import (  # noqa: E402
    DEFAULT_MIN_PER_STRATUM,
    DEFAULT_STRATIFY_COLUMNS,
    select_sample_for_labeling,
)
from rcs.review.repository import get_engine, init_db, load_for_gold_sampling

GOLD_DIR = settings.DATA_DIR / "gold"
GOLD_CSV = GOLD_DIR / "gold_dataset_round1.csv"

NEW_HUMAN_COLUMNS = ["gold_sentiment", "gold_annotation_status"]
NEW_GENERATED_COLUMNS = ["annotation_stage", "is_few_shot_example"]


def prepare(seed_size: int, seed: int, min_per_stratum: int = 1) -> pd.DataFrame:
    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"{GOLD_CSV} not found — Phase B's frozen 500-row sample must exist first")

    existing = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)
    original_ids = set(existing["comment_id"])
    original_split_map = dict(zip(existing["comment_id"], existing["split"]))

    engine = get_engine(settings.DB_PATH)
    init_db(engine)
    eligible_meta = load_for_gold_sampling(engine)

    train_ids = set(existing.loc[existing["split"] == "train", "comment_id"])
    train_pool = eligible_meta[eligible_meta["comment_id"].isin(train_ids)].copy()
    if len(train_pool) != len(train_ids):
        missing = train_ids - set(train_pool["comment_id"])
        raise RuntimeError(
            f"{len(missing)} train-split comment_id(s) from the frozen gold file are no longer in the "
            f"current eligible pool (review.sqlite may have changed since Phase B) — aborting rather than "
            f"silently proceeding: {sorted(missing)[:5]}"
        )

    # min_per_stratum=1 here (not the DEFAULT_MIN_PER_STRATUM=3 used for the
    # original 500-row sample) is deliberate: the 350-row train pool alone
    # already contains 23 distinct (original_type, age_group) strata, and a
    # floor of 3 across that many strata forces the total selection well
    # past the requested ~50 (68, in an actual run) before any random
    # supplement is even added. The seed's job is a reasonably diverse
    # STARTING few-shot pool sized close to what was asked for; guaranteeing
    # real coverage of all 6 PRIMARY_INTENT / 7 TOPIC_SUBTYPE values is
    # Phase B.1b's job, checked against actual gold labels (not this proxy
    # metadata) after the seed is labeled — and topped up then, only if
    # genuinely needed.
    seed_sample = select_sample_for_labeling(
        train_pool,
        sample_size=seed_size,
        stratify_columns=DEFAULT_STRATIFY_COLUMNS,
        min_per_stratum=min_per_stratum,
        random_supplement_fraction=0.2,
        seed=seed,
    )
    seed_ids = set(seed_sample["comment_id"])

    updated = existing.copy()
    for col in NEW_HUMAN_COLUMNS + list(AI_SUGGESTED_COLUMNS):
        if col not in updated.columns:
            updated[col] = ""

    updated = assign_annotation_stage(updated, seed_ids)

    # --- integrity checks: comment_id/split are exactly preserved ---
    if set(updated["comment_id"]) != original_ids:
        raise AssertionError("comment_id set changed during Phase B.1 — refusing to write output")
    for cid in original_ids:
        new_split = updated.loc[updated["comment_id"] == cid, "split"].iloc[0]
        if new_split != original_split_map[cid]:
            raise AssertionError(f"split value changed for {cid}: {original_split_map[cid]!r} -> {new_split!r}")

    verify_gold_workbook_invariants(updated)

    return updated, seed_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B.1: extend the gold workbook and select the human_train_seed.")
    parser.add_argument("--seed-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-stratum", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Compute and validate, but do not write the CSV.")
    args = parser.parse_args()

    updated, seed_ids = prepare(args.seed_size, args.seed, args.min_per_stratum)

    if not args.dry_run:
        updated.to_csv(GOLD_CSV, index=False, encoding="utf-8-sig")

    print(f"total_rows: {len(updated)}")
    print(f"annotation_stage_counts: {updated['annotation_stage'].value_counts().to_dict()}")
    print(f"human_train_seed_selected: {len(seed_ids)}")
    print(f"dry_run: {args.dry_run}")
    print(f"output_csv: {GOLD_CSV}")


if __name__ == "__main__":
    main()
