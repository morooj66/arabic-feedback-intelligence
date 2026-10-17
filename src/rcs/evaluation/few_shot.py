"""
Phase B.2 — the ONLY few-shot example source, per REVISED_ARCHITECTURE_PLAN.md
v2 Section 7/8 (leakage-conscious annotation strategy, reaffirmed by the
user at Phase B.1b close):

    is_few_shot_example == True   AND   gold_annotation_status == "LABELED"

`INSUFFICIENT_CONTEXT` rows are excluded entirely, even if flagged
is_few_shot_example (a row can be selected as part of the seed and still
turn out, on human review, to lack enough context to label — such a row
must never appear in a few-shot prompt). `human_val`/`human_test` rows are
never eligible in the first place (is_few_shot_example is False for all of
them, enforced by rcs.evaluation.gold_dataset.assign_annotation_stage/
verify_gold_workbook_invariants) — this module doesn't need to re-check
split, only re-check the two conditions above, defensively.

Selection is a FIXED set, computed once per run and reused for every
target row's prompt — not a per-row nearest-neighbor/RAG lookup
(REVISED_ARCHITECTURE_PLAN.md v2: "No RAG unless the actual requirements
justify it" — they don't, here). Deterministic and reproducible: same
gold dataframe + same seed always produces the same few-shot set.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

DEFAULT_MAX_PER_INTENT = 2
DEFAULT_SEED = 42

_EXAMPLE_FIELDS = (
    "comment_id", "comment_text", "gold_primary_intent", "gold_topic_subtype",
    "gold_secondary_intent", "gold_subcategory", "gold_sentiment",
)


def eligible_few_shot_pool(gold: pd.DataFrame) -> pd.DataFrame:
    """The full set of rows that are ALLOWED to appear in a few-shot prompt.
    Pure filter — no selection/sampling here."""
    is_seed = gold["is_few_shot_example"].astype(str).str.strip().str.lower() == "true"
    is_labeled = gold["gold_annotation_status"].astype(str).str.strip() == "LABELED"
    return gold[is_seed & is_labeled].copy()


def select_few_shot_examples(
    gold: pd.DataFrame,
    max_per_intent: int = DEFAULT_MAX_PER_INTENT,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """
    Builds the fixed few-shot set: up to `max_per_intent` examples per
    gold_primary_intent value found in the eligible pool, deterministically
    chosen. A class with fewer than `max_per_intent` eligible examples
    simply contributes all of them (never padded, never duplicated).

    Returns a list of dicts (one per selected example) with the fields in
    _EXAMPLE_FIELDS — ready for suggestion_prompt.py to format, and
    small/stable enough to log verbatim in a run report.
    """
    pool = eligible_few_shot_pool(gold)
    if pool.empty:
        return []

    selected_frames = []
    for intent, group in pool.groupby("gold_primary_intent", sort=True):
        group_sorted = group.sort_values("comment_id").reset_index(drop=True)
        if len(group_sorted) <= max_per_intent:
            selected_frames.append(group_sorted)
        else:
            selected_frames.append(
                group_sorted.sample(n=max_per_intent, random_state=seed).sort_values("comment_id")
            )

    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["gold_primary_intent", "comment_id"]
    )
    return selected[list(_EXAMPLE_FIELDS)].to_dict(orient="records")
