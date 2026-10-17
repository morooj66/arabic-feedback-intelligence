#!/usr/bin/env python3
"""
CLI entrypoint for LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase B — sample
selection ONLY. Does not label anything, does not call Gemini or any
model, does not train anything.

Reads the Phase-3-eligible pool from review.sqlite (read-only), selects a
sample for human labeling via rcs.evaluation.sampling, assigns each
selected row a train/val/test split via
rcs.evaluation.gold_dataset.assign_splits_incremental (frozen-once
semantics — safe to re-run for a later labeling round without disturbing
round 1's assignments), and writes the result to
data/gold/gold_dataset_round1.csv — a plain CSV with the gold_* columns
present but empty, ready for a human to open (or convert to .xlsx) and
fill in.

Usage:
    python scripts/select_gold_sample.py --sample-size 500 --seed 42
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
    assign_splits_incremental,
    verify_no_split_leakage,
)
from rcs.evaluation.sampling import (  # noqa: E402
    DEFAULT_MIN_PER_STRATUM,
    DEFAULT_RANDOM_SUPPLEMENT_FRACTION,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    DEFAULT_STRATIFY_COLUMNS,
    select_sample_for_labeling,
)
from rcs.review.repository import get_engine, init_db, load_for_gold_sampling

GOLD_DIR = settings.DATA_DIR / "gold"
OUTPUT_PATH = GOLD_DIR / "gold_dataset_round1.csv"

GOLD_COLUMNS = [
    "comment_id",
    "comment_text",
    "gold_primary_intent",
    "gold_topic_subtype",
    "gold_secondary_intent",
    "gold_subcategory",
    "labeler_notes",
    "labeled_by",
    "labeled_at",
    "split",
]


def build_gold_template(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    stratify_columns=DEFAULT_STRATIFY_COLUMNS,
    random_supplement_fraction: float = DEFAULT_RANDOM_SUPPLEMENT_FRACTION,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
) -> pd.DataFrame:
    engine = get_engine(settings.DB_PATH)
    init_db(engine)
    eligible = load_for_gold_sampling(engine)

    sample = select_sample_for_labeling(
        eligible,
        sample_size=sample_size,
        stratify_columns=stratify_columns,
        random_supplement_fraction=random_supplement_fraction,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    split_df = assign_splits_incremental(
        existing=None,  # round 1: no prior gold data to preserve
        new_comment_ids=sample["comment_id"].tolist(),
        new_labels=None,  # no gold labels exist yet; split by presence only (see note below)
        seed=seed,
    )
    verify_no_split_leakage(split_df)

    merged = sample.merge(split_df[["comment_id", "split"]], on="comment_id", how="left")

    template = pd.DataFrame({
        "comment_id": merged["comment_id"],
        "comment_text": merged["original_comment"],
        "gold_primary_intent": "",
        "gold_topic_subtype": "",
        "gold_secondary_intent": "",
        "gold_subcategory": "",
        "labeler_notes": "",
        "labeled_by": "",
        "labeled_at": "",
        "split": merged["split"],
    })
    return template[GOLD_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B: select the gold-dataset labeling sample (no labels, no model calls).")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    template = build_gold_template(sample_size=args.sample_size, seed=args.seed)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    template.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"sample_size_requested: {args.sample_size}")
    print(f"sample_size_actual: {len(template)}")
    print(f"split_counts: {template['split'].value_counts().to_dict()}")
    print(f"output_path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
