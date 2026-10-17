"""
Phase B tests — LOCAL_FIRST_ARCHITECTURE_PLAN.md sample selection and
anti-leakage mechanism. All synthetic data — no real gold labels, no
model/Gemini calls, no training anywhere in this file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rcs.evaluation.gold_dataset import (
    AI_SUGGESTED_COLUMNS,
    assign_annotation_stage,
    assign_splits,
    assign_splits_incremental,
    verify_gold_workbook_invariants,
    verify_no_split_leakage,
)
from rcs.evaluation.sampling import select_sample_for_labeling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_pool(n=1000, n_types=5, n_ages=3, duplicate_rate=0.02, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    types = [f"TYPE_{i}" for i in range(n_types)]
    ages = ["recent", "delayed", "old"][:n_ages]

    ids = [f"c{i}" for i in range(n)]
    df = pd.DataFrame({
        "comment_id": ids,
        "original_comment": [f"comment text {i}" for i in range(n)],
        "original_type": rng.choice(types, size=n),
        "age_group": rng.choice(ages, size=n),
        "platform": rng.choice(["الرياض", "سوار"], size=n),
        "is_exact_duplicate_of": [None] * n,
    })

    # Wire up a controlled number of exact-duplicate rows pointing back at
    # an earlier row's comment_id.
    n_dupes = int(n * duplicate_rate)
    dupe_targets = rng.choice(ids[: n // 2], size=n_dupes, replace=True)
    dupe_rows = rng.choice(range(n // 2, n), size=n_dupes, replace=False)
    for row_idx, target in zip(dupe_rows, dupe_targets):
        df.loc[row_idx, "is_exact_duplicate_of"] = target

    return df


# ---------------------------------------------------------------------------
# select_sample_for_labeling
# ---------------------------------------------------------------------------

def test_sample_size_respected_when_pool_is_larger():
    pool = _synthetic_pool(n=1000)
    sample = select_sample_for_labeling(pool, sample_size=200, seed=1)
    assert len(sample) == 200


def test_sample_returns_full_pool_when_smaller_than_requested_size():
    pool = _synthetic_pool(n=50)
    sample = select_sample_for_labeling(pool, sample_size=500, seed=1)
    # duplicates may reduce this slightly below 50 — but never below the
    # deduplicated candidate count, and never raises
    assert len(sample) <= 50
    assert len(sample) > 0


def test_sample_is_deterministic_given_same_seed():
    pool = _synthetic_pool(n=1000)
    a = select_sample_for_labeling(pool, sample_size=150, seed=7)
    b = select_sample_for_labeling(pool, sample_size=150, seed=7)
    assert sorted(a["comment_id"]) == sorted(b["comment_id"])


def test_sample_differs_with_different_seed():
    pool = _synthetic_pool(n=1000)
    a = select_sample_for_labeling(pool, sample_size=150, seed=1)
    b = select_sample_for_labeling(pool, sample_size=150, seed=2)
    assert sorted(a["comment_id"]) != sorted(b["comment_id"])


def test_sample_never_selects_two_rows_from_the_same_duplicate_cluster():
    pool = _synthetic_pool(n=1000, duplicate_rate=0.1, seed=3)
    sample = select_sample_for_labeling(pool, sample_size=400, seed=3)

    # Rebuild the same cluster_id logic the function uses internally, and
    # confirm no two selected rows share a cluster.
    cluster_id = pool.set_index("comment_id")["is_exact_duplicate_of"]
    resolved = {
        cid: (cluster_id[cid] if pd.notna(cluster_id[cid]) else cid)
        for cid in sample["comment_id"]
    }
    assert len(set(resolved.values())) == len(resolved)


def test_sample_covers_rare_strata_via_floor():
    # One stratum with only 4 rows total, everything else large — the
    # floor must still surface at least some of the rare stratum.
    import numpy as np
    n = 1000
    rng = np.random.default_rng(5)
    types = ["COMMON"] * (n - 4) + ["RARE"] * 4
    df = pd.DataFrame({
        "comment_id": [f"c{i}" for i in range(n)],
        "original_comment": ["x"] * n,
        "original_type": types,
        "age_group": ["old"] * n,
        "platform": ["الرياض"] * n,
        "is_exact_duplicate_of": [None] * n,
    })
    sample = select_sample_for_labeling(
        df, sample_size=100, stratify_columns=("original_type",),
        random_supplement_fraction=0.0, min_per_stratum=3, seed=9,
    )
    rare_in_sample = sample[sample["original_type"] == "RARE"]
    assert len(rare_in_sample) >= 3


def test_sample_random_supplement_zero_is_fully_stratified():
    pool = _synthetic_pool(n=1000)
    sample = select_sample_for_labeling(pool, sample_size=200, random_supplement_fraction=0.0, seed=1)
    assert len(sample) == 200  # still respects requested size


def test_sample_rejects_unknown_stratify_column():
    pool = _synthetic_pool(n=50)
    with pytest.raises(ValueError):
        select_sample_for_labeling(pool, sample_size=10, stratify_columns=("not_a_real_column",))


def test_sample_rejects_bad_random_supplement_fraction():
    pool = _synthetic_pool(n=50)
    with pytest.raises(ValueError):
        select_sample_for_labeling(pool, sample_size=10, random_supplement_fraction=1.5)


# ---------------------------------------------------------------------------
# Anti-leakage: assign_splits_incremental / verify_no_split_leakage
# ---------------------------------------------------------------------------

def test_incremental_split_never_moves_existing_rows():
    round1_ids = [f"c{i}" for i in range(100)]
    round1_labels = ["A", "B"] * 50
    round1 = assign_splits(round1_ids, labels=round1_labels, seed=42)

    round2_ids = [f"c{i}" for i in range(100, 150)]
    round2_labels = ["A", "B"] * 25
    combined = assign_splits_incremental(round1, round2_ids, round2_labels, seed=42)

    # every round-1 id keeps EXACTLY the split it was given in round 1
    round1_lookup = dict(zip(round1["comment_id"], round1["split"]))
    combined_lookup = dict(zip(combined["comment_id"], combined["split"]))
    for cid, split in round1_lookup.items():
        assert combined_lookup[cid] == split


def test_incremental_split_adds_new_ids():
    round1 = assign_splits([f"c{i}" for i in range(50)], seed=1)
    combined = assign_splits_incremental(round1, [f"n{i}" for i in range(20)], seed=1)
    assert len(combined) == 70
    assert set(combined["comment_id"]) == set(round1["comment_id"]) | {f"n{i}" for i in range(20)}


def test_incremental_split_first_round_with_no_existing_data():
    combined = assign_splits_incremental(None, [f"c{i}" for i in range(30)], seed=1)
    assert len(combined) == 30
    assert set(combined["split"]) <= {"train", "val", "test"}


def test_incremental_split_rejects_reassigning_existing_id():
    round1 = assign_splits([f"c{i}" for i in range(10)], seed=1)
    with pytest.raises(ValueError):
        assign_splits_incremental(round1, ["c0", "c99"], seed=1)  # c0 already has a frozen split


def test_verify_no_split_leakage_passes_on_clean_data():
    gold = assign_splits([f"c{i}" for i in range(50)], seed=1)
    verify_no_split_leakage(gold)  # must not raise


def test_verify_no_split_leakage_catches_duplicate_comment_id():
    gold = pd.DataFrame({
        "comment_id": ["a", "a", "b"],
        "label": ["X", "X", "Y"],
        "split": ["train", "test", "train"],
    })
    with pytest.raises(AssertionError):
        verify_no_split_leakage(gold)


def test_verify_no_split_leakage_catches_unrecognized_split_value():
    gold = pd.DataFrame({
        "comment_id": ["a", "b"],
        "label": ["X", "Y"],
        "split": ["train", "holdout_v2"],  # typo / invalid value
    })
    with pytest.raises(AssertionError):
        verify_no_split_leakage(gold)


# ---------------------------------------------------------------------------
# Phase B.1: assign_annotation_stage / verify_gold_workbook_invariants
# ---------------------------------------------------------------------------

def _synthetic_gold(n_train=350, n_val=75, n_test=75):
    ids = (
        [f"train{i}" for i in range(n_train)]
        + [f"val{i}" for i in range(n_val)]
        + [f"test{i}" for i in range(n_test)]
    )
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    df = pd.DataFrame({"comment_id": ids, "split": splits})
    for col in AI_SUGGESTED_COLUMNS:
        df[col] = ""
    return df


def test_assign_annotation_stage_basic_mapping():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    seed_ids = {"train0", "train1"}
    result = assign_annotation_stage(gold, seed_ids)

    assert set(result.loc[result["split"] == "val", "annotation_stage"]) == {"human_val"}
    assert set(result.loc[result["split"] == "test", "annotation_stage"]) == {"human_test"}
    assert result.loc[result["comment_id"] == "train0", "annotation_stage"].iloc[0] == "human_train_seed"
    assert result.loc[result["comment_id"] == "train1", "annotation_stage"].iloc[0] == "human_train_seed"
    assert result.loc[result["comment_id"] == "train2", "annotation_stage"].iloc[0] == "ai_assisted_train"


def test_assign_annotation_stage_is_few_shot_flag_matches_seed_only():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    seed_ids = {"train0", "train1"}
    result = assign_annotation_stage(gold, seed_ids)

    few_shot = set(result.loc[result["is_few_shot_example"], "comment_id"])
    assert few_shot == seed_ids


def test_assign_annotation_stage_never_touches_comment_id_or_split():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    before = gold[["comment_id", "split"]].copy()
    result = assign_annotation_stage(gold, {"train0"})
    pd.testing.assert_frame_equal(before, result[["comment_id", "split"]])


def test_assign_annotation_stage_rejects_unknown_split():
    gold = pd.DataFrame({"comment_id": ["a"], "split": ["holdout"]})
    with pytest.raises(ValueError):
        assign_annotation_stage(gold, set())


def test_verify_gold_workbook_invariants_passes_on_clean_data():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    result = assign_annotation_stage(gold, {"train0", "train1", "train2"})
    verify_gold_workbook_invariants(result)  # must not raise


def test_verify_gold_workbook_invariants_catches_val_marked_as_few_shot():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    result = assign_annotation_stage(gold, {"train0"})
    # simulate the exact leakage bug this phase exists to prevent
    result.loc[result["comment_id"] == "val0", "is_few_shot_example"] = True
    with pytest.raises(AssertionError, match="leakage"):
        verify_gold_workbook_invariants(result)


def test_verify_gold_workbook_invariants_catches_test_marked_as_few_shot():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    result = assign_annotation_stage(gold, {"train0"})
    result.loc[result["comment_id"] == "test0", "is_few_shot_example"] = True
    with pytest.raises(AssertionError, match="leakage"):
        verify_gold_workbook_invariants(result)


def test_verify_gold_workbook_invariants_catches_few_shot_flag_on_non_seed_row():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    result = assign_annotation_stage(gold, {"train0"})
    result.loc[result["comment_id"] == "train5", "is_few_shot_example"] = True  # not actually a seed row
    with pytest.raises(AssertionError):
        verify_gold_workbook_invariants(result)


def test_verify_gold_workbook_invariants_catches_missing_seed():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    result = assign_annotation_stage(gold, set())  # no seed selected at all
    with pytest.raises(AssertionError, match="seed selection did not run"):
        verify_gold_workbook_invariants(result)


def test_verify_gold_workbook_invariants_catches_premature_ai_suggested_write():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    result = assign_annotation_stage(gold, {"train0"})
    result.loc[result["comment_id"] == "train5", "ai_suggested_primary_intent"] = "SUGGESTION"
    with pytest.raises(AssertionError, match="ai_suggested_primary_intent"):
        verify_gold_workbook_invariants(result)


def test_verify_gold_workbook_invariants_catches_unrecognized_annotation_stage():
    gold = _synthetic_gold(n_train=10, n_val=3, n_test=3)
    result = assign_annotation_stage(gold, {"train0"})
    result.loc[result["comment_id"] == "train5", "annotation_stage"] = "typo_stage"
    with pytest.raises(AssertionError):
        verify_gold_workbook_invariants(result)
