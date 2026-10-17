#!/usr/bin/env python3
"""
Phase B.1b — Targeted Train-Only Top-Up, Round 2 (approved by user).

After Round 1's top-up (scripts/prepare_gold_topup.py, +24 rows) was
labeled, coverage of the 74-row human_train_seed still had:
  - DUA_BLESSING (renamed from PRAYER): 0 examples
  - THANKS_PRAISE: 1 example
  - DICTIONARY_SUGGESTION: 1 example
  - TECHNICAL_ISSUE: 1 example

This script adds a small second batch, sourced via keyword search and
`original_type` — used ONLY as a sampling proxy to surface candidates,
never as a gold label and never used to auto-classify anything. DUA_BLESSING
is prioritized (currently zero examples); the other three get a smaller,
deliberately modest share — the user does not want these rare classes
artificially inflated, only reasonably represented in the seed.

Guarantees, enforced by assertion before anything is written:
  - every existing comment_id/split/annotation_stage/gold_* value in the
    canonical gold_dataset_round1.csv is preserved byte-for-byte
  - new rows sourced ONLY from ids not already present anywhere in the file
  - new rows are always split="train"
  - new rows get annotation_stage="human_train_seed",
    is_few_shot_example=True via the same assign_annotation_stage() used
    throughout Phase B.1/B.1b
  - no human_val/human_test row is touched
  - zero ai_suggested_* values written
  - gold_* for the new rows is left blank

No model calls. No training. No inference.
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

# Keyword lists are a SAMPLING PROXY ONLY — they pick candidate rows for a
# human to label, they do not assign or imply any label.
DUA_BLESSING_KEYWORDS = [
    "جزاك الله", "جزاكم الله", "بارك الله", "الله يعطيك العافية",
    "الله يعطيكم العافية", "الله يوفقكم", "ربي يوفقكم", "اللهم",
    "دعواتي", "الله يجزاكم", "يعطيك العافية", "يعطيكم العافية",
]
THANKS_PRAISE_KEYWORDS = [
    "شكرا", "شكراً", "الموقع رائع", "عمل رائع", "جهد رائع", "ممتاز",
    "أحسنتم", "جهد مميز", "موقع مميز",
]
TECHNICAL_ISSUE_KEYWORDS = [
    "لا يعمل", "لا يفتح", "مشكلة تقنية", "خطأ برمجي", "لا يظهر",
    "التطبيق لا", "الصفحة لا تعمل", "خلل تقني", "لا يمكنني الدخول",
]
DICTIONARY_SUGGESTION_TYPE = "إقتراح معجم"


def _pick(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(pool) <= n:
        return pool.copy()
    return pool.sample(n=n, random_state=seed).copy()


def _keyword_mask(texts: pd.Series, keywords: list[str]) -> pd.Series:
    return texts.fillna("").apply(lambda t: any(k in t for k in keywords))


def prepare(n_dua: int, n_thanks: int, n_dict: int, n_tech: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"{GOLD_CSV} not found")

    existing = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)
    existing_ids = set(existing["comment_id"])
    old_split_map = dict(zip(existing["comment_id"], existing["split"]))
    old_stage_map = dict(zip(existing["comment_id"], existing["annotation_stage"]))
    old_gold_cols = [c for c in existing.columns if c.startswith("gold_")] + ["labeler_notes", "labeled_by", "labeled_at"]
    old_gold_snapshot = existing.set_index("comment_id")[old_gold_cols].copy()
    old_seed_ids = set(existing.loc[existing["annotation_stage"] == "human_train_seed", "comment_id"])

    engine = get_engine(settings.DB_PATH)
    init_db(engine)
    eligible = load_for_gold_sampling(engine)
    fresh_pool = eligible[~eligible["comment_id"].isin(existing_ids)].copy()

    dua_pool = fresh_pool[_keyword_mask(fresh_pool["original_comment"], DUA_BLESSING_KEYWORDS)]
    dua_sel = _pick(dua_pool, n_dua, seed)
    claimed = set(dua_sel["comment_id"])

    thanks_pool = fresh_pool[
        _keyword_mask(fresh_pool["original_comment"], THANKS_PRAISE_KEYWORDS)
        & ~fresh_pool["comment_id"].isin(claimed)
    ]
    thanks_sel = _pick(thanks_pool, n_thanks, seed)
    claimed |= set(thanks_sel["comment_id"])

    dict_pool = fresh_pool[
        (fresh_pool["original_type"] == DICTIONARY_SUGGESTION_TYPE)
        & ~fresh_pool["comment_id"].isin(claimed)
    ]
    dict_sel = _pick(dict_pool, n_dict, seed)
    claimed |= set(dict_sel["comment_id"])

    tech_pool = fresh_pool[
        _keyword_mask(fresh_pool["original_comment"], TECHNICAL_ISSUE_KEYWORDS)
        & ~fresh_pool["comment_id"].isin(claimed)
    ]
    tech_sel = _pick(tech_pool, n_tech, seed)

    pool_counts = {
        "dua_blessing_pool": len(dua_pool), "dua_blessing_selected": len(dua_sel),
        "thanks_praise_pool": len(thanks_pool), "thanks_praise_selected": len(thanks_sel),
        "dictionary_suggestion_pool": len(dict_pool), "dictionary_suggestion_selected": len(dict_sel),
        "technical_issue_pool": len(tech_pool), "technical_issue_selected": len(tech_sel),
    }

    selected = pd.concat([dua_sel, thanks_sel, dict_sel, tech_sel], ignore_index=True)
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
        "annotation_stage": "",
        "is_few_shot_example": "",
    })
    for col in AI_SUGGESTED_COLUMNS:
        new_rows[col] = ""
    new_rows = new_rows[existing.columns.tolist()]

    combined = pd.concat([existing, new_rows], ignore_index=True)

    for cid in existing_ids:
        row = combined.loc[combined["comment_id"] == cid].iloc[0]
        if row["split"] != old_split_map[cid]:
            raise AssertionError(f"split changed for existing row {cid}")
        if row["annotation_stage"] != old_stage_map[cid]:
            raise AssertionError(f"annotation_stage changed for existing row {cid}")
    old_now = combined.set_index("comment_id").loc[list(existing_ids), old_gold_cols]
    if not old_now.loc[old_gold_snapshot.index].equals(old_gold_snapshot):
        raise AssertionError("an existing gold_*/labeler_* value changed during top-up — refusing to write")

    new_ids = set(new_rows["comment_id"])
    all_seed_ids = old_seed_ids | new_ids
    combined = assign_annotation_stage(combined, all_seed_ids)

    for cid in existing_ids:
        row = combined.loc[combined["comment_id"] == cid].iloc[0]
        if row["annotation_stage"] != old_stage_map[cid]:
            raise AssertionError(f"assign_annotation_stage altered an EXISTING row's stage: {cid}")
    for cid in new_ids:
        row = combined.loc[combined["comment_id"] == cid].iloc[0]
        if row["annotation_stage"] != "human_train_seed":
            raise AssertionError(f"new top-up row {cid} did not land in human_train_seed as expected")

    verify_no_split_leakage(combined)
    verify_gold_workbook_invariants(combined)

    return combined, new_rows, pool_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B.1b Round 2: targeted top-up for DUA_BLESSING/THANKS_PRAISE/DICTIONARY_SUGGESTION/TECHNICAL_ISSUE.")
    parser.add_argument("--n-dua", type=int, default=5)
    parser.add_argument("--n-thanks", type=int, default=3)
    parser.add_argument("--n-dict", type=int, default=3)
    parser.add_argument("--n-tech", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    combined, new_rows, pool_counts = prepare(args.n_dua, args.n_thanks, args.n_dict, args.n_tech, args.seed)

    if not args.dry_run:
        combined.to_csv(GOLD_CSV, index=False, encoding="utf-8-sig")

    print(f"pool_counts: {pool_counts}")
    print(f"new_rows_added: {len(new_rows)}")
    print(f"total_rows_after: {len(combined)}")
    print(f"annotation_stage_counts: {combined['annotation_stage'].value_counts().to_dict()}")
    print(f"new_comment_ids: {new_rows['comment_id'].tolist()}")
    print(f"dry_run: {args.dry_run}")
    print(f"output_csv: {GOLD_CSV}")


if __name__ == "__main__":
    main()
