#!/usr/bin/env python3
"""
Phase B.1b — Targeted Train-Only Top-Up (approved by user).

The 50-row human_train_seed selected in Phase B.1 turned out to have zero
examples of PRIMARY_INTENT={THANKS_PRAISE, DUA_BLESSING, COMPLAINT} and zero
examples of TOPIC_SUBTYPE={ENTRY_CORRECTION, TECHNICAL_ISSUE,
WORD_OF_YEAR_NOMINATION} (plus DICTIONARY_SUGGESTION at only 1 example).
This script adds a small, targeted batch of NEW candidate rows — sourced
only from `original_type` as a **sampling proxy** (never as a gold label,
never used to infer intent/topic automatically) — for a human to label.

Guarantees, enforced by assertion before anything is written:
  - every existing comment_id/split/annotation_stage/gold_* value in the
    canonical gold_dataset_round1.csv is preserved byte-for-byte
  - new rows are sourced ONLY from the eligible pool, ONLY from
    comment_ids not already present anywhere in the existing 500-row file
  - new rows are always split="train" (never val/test)
  - new rows get annotation_stage="human_train_seed",
    is_few_shot_example=True (recomputed via the same
    assign_annotation_stage() used in Phase B.1 — not hand-set)
  - no human_val/human_test row is touched
  - zero ai_suggested_* values written (no model involved)
  - gold_* for the new rows is left blank (human labeling happens after
    this script runs, not before)

No model calls. No training. No inference. Read-only against
review.sqlite except for the metadata needed to pick candidates.
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
    verify_no_split_leakage,
)
from rcs.review.repository import get_engine, init_db, load_for_gold_sampling  # noqa: E402

GOLD_DIR = settings.DATA_DIR / "gold"
GOLD_CSV = GOLD_DIR / "gold_dataset_round1.csv"

WOTY_PATTERN = "كلمة العام"
REPORT_TYPES = {"الإبلاغ عن مدخل", "ما أصل هذه الكلمة وهل وردت في المعاجم القديمة بالمعنى نفسه"}
CONTACT_TYPE = "رسالة تواصل"

NON_GOLD_HUMAN_COLUMNS = ["labeler_notes", "labeled_by", "labeled_at"]
GOLD_COLUMNS = [
    "gold_primary_intent", "gold_topic_subtype", "gold_secondary_intent",
    "gold_subcategory", "gold_sentiment", "gold_annotation_status",
]


def _pick(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(pool) <= n:
        return pool.copy()
    return pool.sample(n=n, random_state=seed).copy()


def prepare(n_woty: int, n_report: int, n_contact: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"{GOLD_CSV} not found")

    existing = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)
    existing_ids = set(existing["comment_id"])
    old_split_map = dict(zip(existing["comment_id"], existing["split"]))
    old_stage_map = dict(zip(existing["comment_id"], existing["annotation_stage"]))
    old_gold_snapshot = existing.set_index("comment_id")[GOLD_COLUMNS + NON_GOLD_HUMAN_COLUMNS].copy()

    old_seed_ids = set(existing.loc[existing["annotation_stage"] == "human_train_seed", "comment_id"])

    engine = get_engine(settings.DB_PATH)
    init_db(engine)
    eligible = load_for_gold_sampling(engine)  # comment_id, original_comment, original_type, age_group, ...

    fresh_pool = eligible[~eligible["comment_id"].isin(existing_ids)].copy()

    woty_pool = fresh_pool[fresh_pool["original_type"].str.contains(WOTY_PATTERN, na=False)]
    report_pool = fresh_pool[fresh_pool["original_type"].isin(REPORT_TYPES)]
    contact_pool = fresh_pool[fresh_pool["original_type"] == CONTACT_TYPE]

    woty_sel = _pick(woty_pool, n_woty, seed)
    report_sel = _pick(report_pool, n_report, seed)
    contact_sel = _pick(contact_pool, n_contact, seed)

    selected = pd.concat([woty_sel, report_sel, contact_sel], ignore_index=True)
    selected = selected.drop_duplicates(subset="comment_id")

    if selected["comment_id"].isin(existing_ids).any():
        raise AssertionError("a selected top-up candidate is already present in the existing gold file")

    new_rows = pd.DataFrame({
        "comment_text": selected["original_comment"],
        "gold_primary_intent": "",
        "gold_topic_subtype": "",
        "gold_sentiment": "",
        "gold_annotation_status": "",
        "gold_secondary_intent": "",
        "gold_subcategory": "",
        "labeler_notes": "",
        "labeled_by": "",
        "labeled_at": "",
        "comment_id": selected["comment_id"],
        "split": "train",
        "annotation_stage": "",  # recomputed below
        "is_few_shot_example": "",  # recomputed below
    })
    for col in AI_SUGGESTED_COLUMNS:
        new_rows[col] = ""
    new_rows = new_rows[existing.columns.tolist()]

    combined = pd.concat([existing, new_rows], ignore_index=True)

    # --- integrity checks on the OLD rows: byte-identical, nothing reshuffled ---
    for cid in existing_ids:
        row = combined.loc[combined["comment_id"] == cid].iloc[0]
        if row["split"] != old_split_map[cid]:
            raise AssertionError(f"split changed for existing row {cid}")
        if row["annotation_stage"] != old_stage_map[cid]:
            raise AssertionError(f"annotation_stage changed for existing row {cid}")
    old_now = combined.set_index("comment_id").loc[list(existing_ids), GOLD_COLUMNS + NON_GOLD_HUMAN_COLUMNS]
    if not old_now.loc[old_gold_snapshot.index].equals(old_gold_snapshot):
        raise AssertionError("an existing gold_*/labeler_* value changed during top-up — refusing to write")

    new_ids = set(new_rows["comment_id"])
    all_seed_ids = old_seed_ids | new_ids
    combined = assign_annotation_stage(combined, all_seed_ids)

    # re-verify old rows once more, post-recompute (defensive double-check)
    for cid in existing_ids:
        row = combined.loc[combined["comment_id"] == cid].iloc[0]
        if row["annotation_stage"] != old_stage_map[cid]:
            raise AssertionError(f"assign_annotation_stage altered an EXISTING row's stage: {cid}")

    for cid in new_ids:
        row = combined.loc[combined["comment_id"] == cid].iloc[0]
        if row["annotation_stage"] != "human_train_seed":
            raise AssertionError(f"new top-up row {cid} did not land in human_train_seed as expected")
        if row["is_few_shot_example"] != True and str(row["is_few_shot_example"]).lower() != "true":  # noqa: E712
            raise AssertionError(f"new top-up row {cid} missing is_few_shot_example=True")

    verify_no_split_leakage(combined)
    verify_gold_workbook_invariants(combined)

    return combined, new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B.1b: targeted train-only top-up for human_train_seed.")
    parser.add_argument("--n-woty", type=int, default=6)
    parser.add_argument("--n-report", type=int, default=9)
    parser.add_argument("--n-contact", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    combined, new_rows = prepare(args.n_woty, args.n_report, args.n_contact, args.seed)

    if not args.dry_run:
        combined.to_csv(GOLD_CSV, index=False, encoding="utf-8-sig")

    print(f"new_rows_added: {len(new_rows)}")
    print(f"total_rows_after: {len(combined)}")
    print(f"annotation_stage_counts: {combined['annotation_stage'].value_counts().to_dict()}")
    print(f"new_comment_ids: {new_rows['comment_id'].tolist()}")
    print(f"dry_run: {args.dry_run}")
    print(f"output_csv: {GOLD_CSV}")


if __name__ == "__main__":
    main()
