"""
Gold-dataset SAMPLE SELECTION — LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase B.

Answers "which comment_ids should a human label?" This is deliberately
separate from gold_dataset.py's assign_splits()/assign_splits_incremental(),
which answer "once labeled, which split (train/val/test) does each id
belong to?" — two different concerns, two different functions.

No gold labels exist yet at this point — that is the entire reason this
module exists. It operates purely on Phase 2/3 output already sitting in
review_items (original_type, age_group, is_exact_duplicate_of), which
DESIGN_SPEC.md explicitly treats as unverified contextual metadata, never
ground truth about intent or topic. Here it is used ONLY as a diversity
proxy for sample selection, never as a label source, and never fed to a
model — no Gemini call, no training, anywhere in this file.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = 42
DEFAULT_STRATIFY_COLUMNS = ("original_type", "age_group")
DEFAULT_RANDOM_SUPPLEMENT_FRACTION = 0.2  # 20% pure random, 80% stratified
DEFAULT_MIN_PER_STRATUM = 3


def _collapse_duplicate_clusters(pool: pd.DataFrame) -> pd.DataFrame:
    """
    Exact-duplicate rows (same is_exact_duplicate_of chain, from Phase 2)
    are collapsed to ONE representative per cluster before any selection
    happens. This is the first of the anti-leakage layers described in
    LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase B item 4: if a duplicate is
    never even a selection candidate, it can never later land in a
    different split than its twin. It also avoids spending labeling effort
    on text a human has already labeled once.

    `is_exact_duplicate_of` points to the first-seen row's comment_id (or
    is null for a row that IS the first-seen representative / has no
    duplicates) — grouping on that value (falling back to the row's own
    comment_id when null) gives the cluster key.
    """
    pool = pool.copy()
    pool["cluster_id"] = pool["is_exact_duplicate_of"].fillna(pool["comment_id"])
    return pool.drop_duplicates(subset="cluster_id", keep="first")


def select_sample_for_labeling(
    eligible: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    stratify_columns: Sequence[str] = DEFAULT_STRATIFY_COLUMNS,
    random_supplement_fraction: float = DEFAULT_RANDOM_SUPPLEMENT_FRACTION,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    `eligible`: one row per candidate comment, e.g.
    repository.load_for_gold_sampling()'s output — must contain
    comment_id, is_exact_duplicate_of, and every column named in
    `stratify_columns`.

    Returns a subset of `eligible` (original columns preserved, duplicate
    clusters collapsed), up to `sample_size` rows, selected as:

      1. Duplicate clusters collapsed to one representative each (see
         `_collapse_duplicate_clusters`).
      2. `(1 - random_supplement_fraction)` of the sample chosen via
         PROPORTIONAL STRATIFIED sampling over `stratify_columns`, with a
         floor of `min_per_stratum` per non-empty stratum so a real but
         small stratum (e.g. a rare original_type) isn't silently invisible
         in the labeled set. A stratum smaller than the floor contributes
         everything it has, never more than it has.
      3. The remaining `random_supplement_fraction` chosen via simple
         random sampling from whatever wasn't already selected in step 2 —
         a deliberate safeguard against `stratify_columns` being an
         imperfect proxy for what actually matters (intent/topic), since
         `original_type`/`age_group` are explicitly NOT ground truth.

    If the deduplicated candidate pool has fewer rows than `sample_size`,
    every candidate is returned (no error) — this is expected to be rare
    at the current eligible-pool size, but not treated as invalid.

    Deterministic: identical `eligible` + parameters + `seed` always
    produce the identical selection. Uses a local `numpy.random.Generator`,
    never global random state.
    """
    if not (0.0 <= random_supplement_fraction <= 1.0):
        raise ValueError(f"random_supplement_fraction must be in [0, 1], got {random_supplement_fraction}")
    for col in stratify_columns:
        if col not in eligible.columns:
            raise ValueError(f"stratify column {col!r} not present in `eligible`")

    candidates = _collapse_duplicate_clusters(eligible)

    if sample_size >= len(candidates):
        return candidates.drop(columns="cluster_id").sort_values("comment_id").reset_index(drop=True)

    rng = np.random.default_rng(seed)

    n_random = round(sample_size * random_supplement_fraction)
    n_stratified = sample_size - n_random

    stratum_key = candidates[list(stratify_columns)].astype(str).agg("|".join, axis=1)
    candidates = candidates.assign(_stratum=stratum_key)
    stratum_sizes = candidates["_stratum"].value_counts()

    allocation = _allocate_stratified_counts(stratum_sizes, n_stratified, min_per_stratum)

    stratified_ids: list = []
    for stratum, n in allocation.items():
        if n <= 0:
            continue
        pool_ids = candidates.loc[candidates["_stratum"] == stratum, "comment_id"].to_numpy()
        chosen = rng.choice(pool_ids, size=min(n, len(pool_ids)), replace=False)
        stratified_ids.extend(chosen.tolist())

    remaining_pool = candidates.loc[~candidates["comment_id"].isin(stratified_ids), "comment_id"].to_numpy()
    n_random_actual = min(n_random, len(remaining_pool))
    random_ids = (
        rng.choice(remaining_pool, size=n_random_actual, replace=False).tolist()
        if n_random_actual > 0
        else []
    )

    selected_ids = set(stratified_ids) | set(random_ids)
    result = candidates[candidates["comment_id"].isin(selected_ids)].drop(columns=["_stratum", "cluster_id"])
    return result.sort_values("comment_id").reset_index(drop=True)


def _allocate_stratified_counts(
    stratum_sizes: pd.Series,
    total: int,
    min_per_stratum: int,
) -> pd.Series:
    """
    Proportional allocation of `total` slots across strata (sized by
    `stratum_sizes`), with a floor of `min_per_stratum` per non-empty
    stratum and a ceiling of the stratum's own size (never allocate more
    than exists). The floor/ceiling adjustments can push the raw total
    away from `total` exactly, so a final rebalancing pass nudges the
    largest strata up or down by one until the sum matches `total` (or
    every stratum is already at its ceiling/floor and no further
    adjustment is possible, in which case the achievable total is
    returned as-is rather than raising).
    """
    if stratum_sizes.empty or total <= 0:
        return pd.Series(dtype=int)

    proportional = (stratum_sizes / stratum_sizes.sum() * total).round().astype(int)
    floored = proportional.clip(lower=min_per_stratum)
    capped = np.minimum(floored, stratum_sizes)

    diff = total - int(capped.sum())
    if diff == 0:
        return capped

    # Largest-strata-first in BOTH directions: when adding, spreads the
    # remainder over strata with the most headroom; when removing, drains
    # from the strata with the most slack first, which is what protects a
    # small stratum's floor for as long as any larger stratum still has
    # room to absorb the adjustment instead.
    order = stratum_sizes.sort_values(ascending=False).index.tolist()
    result = capped.copy()
    i = 0
    max_iterations = len(order) * (abs(diff) + 1) + 10
    iterations = 0
    while diff != 0 and iterations < max_iterations and order:
        s = order[i % len(order)]
        if diff > 0 and result[s] < stratum_sizes[s]:
            result[s] += 1
            diff -= 1
        elif diff < 0 and result[s] > min_per_stratum and result[s] > 0:
            result[s] -= 1
            diff += 1
        i += 1
        iterations += 1
    return result
