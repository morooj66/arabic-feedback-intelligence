"""
Gold dataset split-assignment logic — LOCAL_FIRST_ARCHITECTURE_PLAN.md
Phase A only.

Phase A proves the split mechanism is correct (reproducible, stratified)
BEFORE any real labeling starts. Building the actual gold dataset —
sampling real rows for manual labeling, writing data/gold/gold_dataset.csv,
recording labeled_by/labeled_at — is Phase B (scripts/build_gold_dataset.py,
not implemented yet). This module intentionally does not touch any real
data, read any file, or make any labeling decision.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import settings

DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIOS: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
SPLIT_NAMES = frozenset(DEFAULT_SPLIT_RATIOS.keys())

# REVISED_ARCHITECTURE_PLAN.md Section 6/7 (v2, leakage-conscious annotation
# strategy). Every row's provenance, one of four disjoint values:
#   human_val         -- frozen `val` split, pure human, zero AI involvement ever
#   human_test        -- frozen `test` split, pure human, zero AI involvement ever
#   human_train_seed  -- a subset of `train`, pure human, the ONLY few-shot source
#   ai_assisted_train -- the rest of `train`, AI-suggested then human-confirmed
ANNOTATION_STAGES = frozenset({"human_val", "human_test", "human_train_seed", "ai_assisted_train"})

AI_SUGGESTED_COLUMNS = (
    "ai_suggested_primary_intent",
    "ai_suggested_topic_subtype",
    "ai_suggested_secondary_intent",
    "ai_suggested_subcategory",
    "ai_suggested_sentiment",
    "ai_suggested_confidence",
    "ai_suggestion_model",
)

GOLD_LABEL_COLUMNS = (
    "gold_primary_intent",
    "gold_topic_subtype",
    "gold_secondary_intent",
    "gold_subcategory",
    "gold_sentiment",
    "gold_annotation_status",
)
HUMAN_TRACE_COLUMNS = ("labeler_notes", "labeled_by", "labeled_at")
GOLD_ANNOTATION_STATUSES = frozenset({"LABELED", "UNCERTAIN", "INSUFFICIENT_CONTEXT"})
REVIEWER_EDITABLE_COLUMNS = GOLD_LABEL_COLUMNS + HUMAN_TRACE_COLUMNS

# This row was already human-labeled before it was assigned to the B.2 target
# set. It has no labeled_by/labeled_at trace because the older workbook did not
# require those fields. The B.3 workbook highlights it for explicit review.
PREEXISTING_AI_ASSISTED_GOLD_IDS = frozenset({
    "00bb4fc8146ff830b88709f0ddab7a164cd18312461b35c13e8f9d621c18089d",
})
TRAIN_GOLD_COMPLETE_MARKER = settings.DATA_DIR / "gold" / "train_gold_complete.json"


def _boolean_mask(values: pd.Series, column_name: str) -> pd.Series:
    """Normalize bool/string workbook values without treating "False" as true."""
    normalized = values.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false"})
    if not valid.all():
        bad = sorted(set(values.loc[~valid].astype(str)))
        raise AssertionError(f"{column_name} contains non-boolean value(s): {bad[:5]}")
    return normalized == "true"


def _is_nonblank(value: object) -> bool:
    return pd.notna(value) and str(value).strip() != ""


# Approved 2026-08-24 merge: DUA_BLESSING is not a live intent. Historical
# B.2 suggestion columns (and any leftover gold fields) still may store it.
_INTENT_MERGE_ALIASES = {"DUA_BLESSING": "THANKS_PRAISE"}


def _canonical_intent(value: str) -> str:
    stripped = str(value).strip()
    return _INTENT_MERGE_ALIASES.get(stripped, stripped)


def _canonical_optional_intent(value: object, primary: str) -> str | None:
    if not _is_nonblank(value):
        return None
    mapped = _canonical_intent(str(value))
    if mapped == primary:
        return None
    return mapped


def _validate_gold_label_values(row: pd.Series, context: str) -> None:
    primary = _canonical_intent(str(row["gold_primary_intent"]))
    topic = str(row["gold_topic_subtype"]).strip()
    secondary = _canonical_optional_intent(row["gold_secondary_intent"], primary) or ""
    subcategory = str(row["gold_subcategory"]).strip()
    sentiment = str(row["gold_sentiment"]).strip()
    status = str(row["gold_annotation_status"]).strip()

    if primary and primary not in settings.PRIMARY_INTENTS:
        raise AssertionError(f"{context}: invalid gold_primary_intent {primary!r}")
    if topic and topic not in settings.TOPIC_SUBTYPES:
        raise AssertionError(f"{context}: invalid gold_topic_subtype {topic!r}")
    if secondary and secondary not in settings.PRIMARY_INTENTS:
        raise AssertionError(f"{context}: invalid gold_secondary_intent {secondary!r}")
    if subcategory and subcategory not in settings.SUBCATEGORIES:
        raise AssertionError(f"{context}: invalid gold_subcategory {subcategory!r}")
    if sentiment and sentiment not in settings.SENTIMENTS:
        raise AssertionError(f"{context}: invalid gold_sentiment {sentiment!r}")
    if status and status not in GOLD_ANNOTATION_STATUSES:
        raise AssertionError(f"{context}: invalid gold_annotation_status {status!r}")

    if secondary and not primary:
        raise AssertionError(f"{context}: secondary intent requires a primary intent")
    if secondary and secondary == primary:
        raise AssertionError(f"{context}: secondary intent must differ from primary intent")

    if topic in settings.TOPICS_REQUIRING_SUBCATEGORY and not subcategory:
        raise AssertionError(f"{context}: subcategory is required for topic {topic!r}")
    if subcategory and topic not in settings.TOPICS_REQUIRING_SUBCATEGORY:
        raise AssertionError(f"{context}: subcategory is not allowed for topic {topic!r}")

    if status in {"LABELED", "UNCERTAIN"} and not all((primary, topic, sentiment)):
        raise AssertionError(f"{context}: {status} requires primary intent, topic, and sentiment")


def assign_splits(
    comment_ids: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    seed: int = DEFAULT_SEED,
    ratios: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Deterministically assigns each comment_id to a split ("train"/"val"/
    "test" by default, or whatever keys `ratios` uses).

    If `labels` is provided (same length and order as comment_ids), the
    split is stratified per label — each class gets its own proportional
    train/val/test split, so a rare class isn't accidentally absent from
    val or test. This matters here because the dataset's classes are known
    to be imbalanced (per the original data-discovery report). If `labels`
    is omitted, a single unstratified split is used across all ids.

    Reproducibility: uses a `numpy.random.Generator` seeded explicitly with
    `seed` (not global random state), so the same comment_ids + labels +
    seed always produce the exact same assignment, and calling this
    function does not affect any other code's randomness.

    Raises ValueError if `ratios` doesn't sum to 1.0 (within floating-point
    tolerance), if `labels` is provided with a different length than
    `comment_ids`, or if `comment_ids` contains duplicates.
    """
    ratios = dict(ratios) if ratios is not None else dict(DEFAULT_SPLIT_RATIOS)
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios} (sum={ratio_sum})")

    comment_ids = list(comment_ids)
    if len(comment_ids) != len(set(comment_ids)):
        raise ValueError("comment_ids must not contain duplicates")

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(comment_ids):
            raise ValueError(
                f"labels must be the same length as comment_ids (got {len(labels)} and {len(comment_ids)})"
            )
    else:
        labels = ["__all__"] * len(comment_ids)

    id_to_label = dict(zip(comment_ids, labels))
    # group ids by label, preserving nothing about original order (shuffled below)
    ids_by_label: Dict[str, List[str]] = {}
    for cid, label in id_to_label.items():
        ids_by_label.setdefault(label, []).append(cid)

    split_names = list(ratios.keys())
    split_fracs = [ratios[name] for name in split_names]
    cum_fracs = np.cumsum(split_fracs)

    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}
    for label in sorted(ids_by_label):  # sorted: label iteration order never affects the result
        ids = list(ids_by_label[label])
        rng.shuffle(ids)  # in-place, deterministic given `seed`
        n = len(ids)
        boundaries = [0] + [int(round(f * n)) for f in cum_fracs]
        boundaries[-1] = n  # guarantee every id in this label group is assigned despite rounding
        for i, split_name in enumerate(split_names):
            for cid in ids[boundaries[i]:boundaries[i + 1]]:
                assignment[cid] = split_name

    return pd.DataFrame({
        "comment_id": comment_ids,
        "label": labels,
        "split": [assignment[cid] for cid in comment_ids],
    })


# ---------------------------------------------------------------------------
# Anti-leakage mechanism — LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase B item 4.
#
# Three independent layers guard against train/val/test leakage:
#   1. Sample selection never selects more than one row per exact-duplicate
#      cluster (src/rcs/evaluation/sampling.py's _collapse_duplicate_clusters)
#      — a duplicate can't leak across splits if it's never a candidate for
#      more than one split assignment in the first place.
#   2. assign_splits_incremental() below: once a comment_id has a `split`
#      value, it is FROZEN — later labeling rounds can only assign splits
#      for ids not already present, never move or recompute an existing
#      one. This is what makes it safe to grow the gold dataset over
#      multiple rounds without silently invalidating any evaluation already
#      run against the existing test split.
#   3. verify_no_split_leakage() below: a defensive invariant check to run
#      against any gold dataset file before it's used for training/
#      evaluation.
# ---------------------------------------------------------------------------

def assign_splits_incremental(
    existing: Optional[pd.DataFrame],
    new_comment_ids: Sequence[str],
    new_labels: Optional[Sequence[str]] = None,
    seed: int = DEFAULT_SEED,
    ratios: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Grows a gold dataset's split assignment by exactly the new ids, without
    ever touching an existing id's split.

    `existing`: a DataFrame with at least comment_id/label/split columns
    (e.g. a previous call's return value, or None/empty for a first
    labeling round). Every row in it is copied through UNCHANGED.

    `new_comment_ids`/`new_labels`: ids not yet present in `existing`.
    Split for these is computed via assign_splits() (same stratified,
    seeded logic), then appended.

    Raises ValueError if any id in `new_comment_ids` already has a frozen
    split in `existing` — that would be an ambiguous request (re-labeling
    an id is a different operation than assigning it a first split, and
    must not silently reassign an existing split).
    """
    ratios = ratios if ratios is not None else dict(DEFAULT_SPLIT_RATIOS)

    if existing is None or len(existing) == 0:
        existing = pd.DataFrame(columns=["comment_id", "label", "split"])

    new_comment_ids = list(new_comment_ids)
    overlap = set(new_comment_ids) & set(existing["comment_id"])
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(
            f"{len(overlap)} comment_id(s) already have a frozen split assignment and cannot be "
            f"reassigned (first few: {preview}) — this function only assigns splits for NEW ids."
        )

    if not new_comment_ids:
        return existing.reset_index(drop=True)

    new_split = assign_splits(new_comment_ids, labels=new_labels, seed=seed, ratios=ratios)
    combined = pd.concat([existing, new_split], ignore_index=True)
    return combined


def verify_no_split_leakage(gold: pd.DataFrame, valid_splits: Optional[Sequence[str]] = None) -> None:
    """
    Defensive invariant check — run this against any gold dataset
    DataFrame/file before it's used for training or evaluation.

    Raises AssertionError (never silently returns False, so a caller can't
    accidentally ignore the result) if:
      - any comment_id appears more than once (would mean it's assigned to
        more than one split — structurally impossible via assign_splits/
        assign_splits_incremental, but worth checking against a
        hand-edited or externally-modified gold file);
      - the `split` column contains any value outside `valid_splits`
        (default: the keys of DEFAULT_SPLIT_RATIOS) — catches typos from
        manual editing.

    Does NOT and cannot, on its own, detect semantic near-duplicate
    leakage (two different comment_ids with near-identical but not
    byte-identical text landing in different splits) — that risk is
    reduced, not eliminated, by exact-duplicate collapsing at the sampling
    stage (see sampling.py) and is a known residual limitation, not solved
    by this function.
    """
    valid_splits = set(valid_splits) if valid_splits is not None else set(DEFAULT_SPLIT_RATIOS.keys())

    duplicated_ids = gold["comment_id"][gold["comment_id"].duplicated()]
    if not duplicated_ids.empty:
        raise AssertionError(
            f"comment_id appears more than once — would mean it's in more than one split: "
            f"{duplicated_ids.tolist()[:5]}"
        )

    bad_splits = set(gold["split"].unique()) - valid_splits
    if bad_splits:
        raise AssertionError(f"unrecognized split value(s) found: {sorted(bad_splits)}")


# ---------------------------------------------------------------------------
# Phase B.1 — REVISED_ARCHITECTURE_PLAN.md Section 6/7 (v2).
#
# Annotation-stage assignment is a PRESENTATION/provenance concern layered
# on top of the already-frozen split column — it never changes comment_id
# or split, only records which of the four disjoint annotation stages a row
# belongs to and whether it's part of the few-shot seed pool.
# ---------------------------------------------------------------------------

def assign_annotation_stage(gold: pd.DataFrame, train_seed_ids: Iterable[str]) -> pd.DataFrame:
    """
    Sets `annotation_stage` (and the derived `is_few_shot_example` flag) for
    every row, based ONLY on its already-frozen `split` and, for `train`
    rows, whether its comment_id is in `train_seed_ids`.

    Never reads or writes `comment_id`/`split` — those are untouched
    upstream inputs. Idempotent: calling this again with the same
    `train_seed_ids` reproduces the same assignment.

    Raises ValueError if `gold["split"]` contains anything outside
    {"train", "val", "test"} (defensive — should be unreachable given
    assign_splits/assign_splits_incremental's own guarantees).
    """
    train_seed_ids = set(train_seed_ids)
    gold = gold.copy()

    unknown_splits = set(gold["split"].unique()) - {"train", "val", "test"}
    if unknown_splits:
        raise ValueError(f"unrecognized split value(s): {sorted(unknown_splits)}")

    def _stage(row) -> str:
        if row["split"] == "val":
            return "human_val"
        if row["split"] == "test":
            return "human_test"
        return "human_train_seed" if row["comment_id"] in train_seed_ids else "ai_assisted_train"

    gold["annotation_stage"] = gold.apply(_stage, axis=1)
    gold["is_few_shot_example"] = gold["annotation_stage"] == "human_train_seed"
    return gold


def verify_gold_workbook_invariants(gold: pd.DataFrame) -> None:
    """
    Phase B.1 safety checks — layered on top of verify_no_split_leakage(),
    which this function also runs. Raises AssertionError (never silently
    returns False) on the first violation found:

      - split-leakage invariants (via verify_no_split_leakage);
      - every `annotation_stage` value is one of ANNOTATION_STAGES;
      - no `human_val`/`human_test` row is ever marked
        `is_few_shot_example = True` — this is the specific leakage
        guarantee this phase exists to enforce;
      - `is_few_shot_example = True` only ever occurs on
        `annotation_stage == "human_train_seed"` rows;
      - at least one `human_train_seed` row exists (catches a no-op/failed
        seed-selection run silently producing an unlabeled-provenance file);
      - every `human_train_seed` row's `split` is actually `"train"` (a
        seed can only ever be drawn from train, never val/test);
      - none of AI_SUGGESTED_COLUMNS has ANY non-empty value anywhere in
        the frame — Phase B.1 writes zero ai_suggested_* values; that's
        Phase B.2's job, not this one's.
    """
    verify_no_split_leakage(gold)

    bad_stage = set(gold["annotation_stage"].dropna().unique()) - ANNOTATION_STAGES
    if bad_stage:
        raise AssertionError(f"unrecognized annotation_stage value(s): {sorted(bad_stage)}")

    few_shot_mask = _boolean_mask(gold["is_few_shot_example"], "is_few_shot_example")

    leaked = gold[gold["annotation_stage"].isin(["human_val", "human_test"]) & few_shot_mask]
    if not leaked.empty:
        raise AssertionError(
            f"val/test row(s) marked as few-shot examples — this is exactly the leakage this "
            f"phase must prevent: {leaked['comment_id'].tolist()}"
        )

    non_seed_few_shot = gold[few_shot_mask & (gold["annotation_stage"] != "human_train_seed")]
    if not non_seed_few_shot.empty:
        raise AssertionError(
            f"is_few_shot_example=True on non-seed row(s): {non_seed_few_shot['comment_id'].tolist()}"
        )

    seed_rows = gold[gold["annotation_stage"] == "human_train_seed"]
    if seed_rows.empty:
        raise AssertionError("no human_train_seed rows found — seed selection did not run (or produced nothing)")

    seed_outside_train = seed_rows[seed_rows["split"] != "train"]
    if not seed_outside_train.empty:
        raise AssertionError(
            f"human_train_seed row(s) found outside the train split: {seed_outside_train['comment_id'].tolist()}"
        )

    for col in AI_SUGGESTED_COLUMNS:
        if col not in gold.columns:
            continue
        written = gold[col].notna() & (gold[col].astype(str).str.strip() != "")
        if written.any():
            raise AssertionError(
                f"{col} has {int(written.sum())} value(s) written — no ai_suggested_* writes are "
                f"allowed in Phase B.1 (that's Phase B.2)"
            )


def merge_partial_reviewer_labels(
    canonical: pd.DataFrame,
    labels_by_id: dict[str, dict[str, str]],
    *,
    require_empty_existing_gold: bool = True,
) -> pd.DataFrame:
    """
    Apply reviewer-editable fields for an explicit comment_id subset only.

    Protected/non-editable columns must remain byte-identical. Used to import
    completed B.2 calibration human labels before the remaining B.3 review.
    """
    verify_post_b2_b3_invariants(canonical, require_b3_complete=False)
    merged = canonical.copy()
    index = merged.set_index("comment_id")
    for cid, labels in labels_by_id.items():
        if cid not in index.index:
            raise AssertionError(f"unknown comment_id in partial import: {cid}")
        if index.at[cid, "annotation_stage"] != "ai_assisted_train":
            raise AssertionError(
                f"comment_id={cid}: partial human import allowed only on ai_assisted_train rows"
            )
        if require_empty_existing_gold:
            for col in GOLD_LABEL_COLUMNS:
                if _is_nonblank(index.at[cid, col]):
                    raise AssertionError(
                        f"comment_id={cid}: refusing to overwrite existing gold field {col!r}"
                    )
            if _is_nonblank(index.at[cid, "gold_annotation_status"]):
                raise AssertionError(
                    f"comment_id={cid}: refusing to overwrite existing gold_annotation_status"
                )
        for col in REVIEWER_EDITABLE_COLUMNS:
            index.at[cid, col] = str(labels.get(col, "")).strip()
    merged = index.loc[canonical["comment_id"]].reset_index()
    merged = merged[canonical.columns]

    noneditable = [col for col in canonical.columns if col not in REVIEWER_EDITABLE_COLUMNS]
    if not canonical[noneditable].equals(merged[noneditable]):
        raise AssertionError("partial import changed a non-reviewer-editable canonical value")
    protected_rows = canonical["annotation_stage"] != "ai_assisted_train"
    if not canonical.loc[protected_rows].equals(merged.loc[protected_rows]):
        raise AssertionError("partial import changed a protected human_val/human_test/seed row")
    verify_post_b2_b3_invariants(merged, require_b3_complete=False)
    return merged


def verify_post_b2_b3_invariants(
    gold: pd.DataFrame,
    *,
    require_b3_complete: bool = False,
) -> None:
    """
    Validate a canonical gold dataset after B.2 and during/after B.3.

    B.2 suggestions must be complete and schema-valid on ai_assisted_train
    rows only. Human val/test/seed rows must remain free of AI suggestions.
    Any new gold values on an AI-assisted row require an explicit human trace;
    the one documented pre-B.2 anomaly is exempt until B.3 reviews it.

    Set require_b3_complete=True only after human review is expected to be
    finished; it rejects any still-unreviewed ai_assisted_train row.
    """
    required_columns = {
        "comment_id", "comment_text", "split", "annotation_stage",
        "is_few_shot_example", *GOLD_LABEL_COLUMNS, *HUMAN_TRACE_COLUMNS,
        *AI_SUGGESTED_COLUMNS,
    }
    missing = sorted(required_columns - set(gold.columns))
    if missing:
        raise AssertionError(f"gold dataset is missing required column(s): {missing}")

    verify_no_split_leakage(gold)

    bad_stage = set(gold["annotation_stage"].unique()) - ANNOTATION_STAGES
    if bad_stage:
        raise AssertionError(f"unrecognized annotation_stage value(s): {sorted(bad_stage)}")

    expected_split = {
        "human_val": "val",
        "human_test": "test",
        "human_train_seed": "train",
        "ai_assisted_train": "train",
    }
    wrong_split = gold[
        gold.apply(lambda row: row["split"] != expected_split[row["annotation_stage"]], axis=1)
    ]
    if not wrong_split.empty:
        raise AssertionError(
            "annotation_stage/split mismatch on row(s): "
            f"{wrong_split['comment_id'].tolist()[:5]}"
        )

    few_shot_mask = _boolean_mask(gold["is_few_shot_example"], "is_few_shot_example")
    expected_few_shot = gold["annotation_stage"] == "human_train_seed"
    if not few_shot_mask.equals(expected_few_shot):
        bad_ids = gold.loc[few_shot_mask != expected_few_shot, "comment_id"].tolist()
        raise AssertionError(
            "is_few_shot_example must be true exactly for human_train_seed rows: "
            f"{bad_ids[:5]}"
        )

    protected = gold["annotation_stage"] != "ai_assisted_train"
    for col in AI_SUGGESTED_COLUMNS:
        leaked = protected & gold[col].map(_is_nonblank)
        if leaked.any():
            raise AssertionError(
                f"{col} populated on protected row(s): "
                f"{gold.loc[leaked, 'comment_id'].tolist()[:5]}"
            )

    from rcs.evaluation.ai_suggestion_schema import AISuggestionResponse

    targets = gold[gold["annotation_stage"] == "ai_assisted_train"]
    if targets.empty:
        raise AssertionError("no ai_assisted_train rows found")

    required_suggestion_columns = (
        "ai_suggested_primary_intent",
        "ai_suggested_topic_subtype",
        "ai_suggested_sentiment",
        "ai_suggested_confidence",
        "ai_suggestion_model",
    )
    for _, row in targets.iterrows():
        cid = str(row["comment_id"])
        context = f"comment_id={cid}"

        missing_suggestions = [
            col for col in required_suggestion_columns if not _is_nonblank(row[col])
        ]
        if missing_suggestions:
            raise AssertionError(
                f"{context}: incomplete B.2 suggestion fields {missing_suggestions}"
            )
        try:
            primary = _canonical_intent(str(row["ai_suggested_primary_intent"]))
            secondary = _canonical_optional_intent(
                row["ai_suggested_secondary_intent"], primary
            )
            AISuggestionResponse(
                primary_intent=primary,
                topic_subtype=str(row["ai_suggested_topic_subtype"]).strip(),
                secondary_intent=secondary,
                subcategory=str(row["ai_suggested_subcategory"]).strip() or None,
                sentiment=str(row["ai_suggested_sentiment"]).strip(),
                confidence=float(row["ai_suggested_confidence"]),
            )
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"{context}: invalid B.2 suggestion: {exc}") from exc

        _validate_gold_label_values(row, context)
        gold_populated = any(_is_nonblank(row[col]) for col in GOLD_LABEL_COLUMNS)
        status = str(row["gold_annotation_status"]).strip()
        has_trace = (
            _is_nonblank(row["labeled_by"])
            and _is_nonblank(row["labeled_at"])
        )
        is_preexisting = cid in PREEXISTING_AI_ASSISTED_GOLD_IDS

        if gold_populated and not status and not is_preexisting:
            raise AssertionError(
                f"{context}: gold values require gold_annotation_status"
            )
        if gold_populated and not has_trace and not is_preexisting:
            raise AssertionError(
                f"{context}: reviewed gold values require labeled_by and labeled_at"
            )
        if has_trace:
            labeled_at = str(row["labeled_at"]).strip()
            try:
                parsed_date = dt.date.fromisoformat(labeled_at)
            except ValueError as exc:
                raise AssertionError(
                    f"{context}: labeled_at must use YYYY-MM-DD"
                ) from exc
            if parsed_date.isoformat() != labeled_at:
                raise AssertionError(f"{context}: labeled_at must use YYYY-MM-DD")
        if require_b3_complete and not status:
            raise AssertionError(f"{context}: B.3 review is incomplete")


def mark_train_gold_complete(
    gold: pd.DataFrame,
    marker_path: Path = TRAIN_GOLD_COMPLETE_MARKER,
) -> Path:
    """Freeze the train gold dataset after every ai_assisted_train row is reviewed."""
    verify_post_b2_b3_invariants(gold, require_b3_complete=True)
    targets = gold[gold["annotation_stage"] == "ai_assisted_train"]
    status = targets["gold_annotation_status"].astype(str).str.strip()
    if not (status.isin({"LABELED", "INSUFFICIENT_CONTEXT", "UNCERTAIN"})).all():
        raise AssertionError("train gold freeze requires a status on every ai_assisted_train row")
    payload = {
        "status": "complete",
        "phase": "B.3",
        "ai_assisted_train_rows": int(len(targets)),
        "labeled": int((status == "LABELED").sum()),
        "insufficient_context": int((status == "INSUFFICIENT_CONTEXT").sum()),
        "uncertain": int((status == "UNCERTAIN").sum()),
        "protected_stages_untouched": True,
        "next_phase": "AraBERT",
        "marked_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "comment_id_sha256": hashlib.sha256(
            "\n".join(targets["comment_id"].astype(str).tolist()).encode("utf-8")
        ).hexdigest(),
    }
    marker_path = Path(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, marker_path)
    return marker_path
