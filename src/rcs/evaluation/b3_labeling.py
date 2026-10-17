"""
Phase B.3 — human review wizard for remaining ai_assisted_train rows.

Loads unlabeled rows from the canonical gold CSV (calibration-imported and
pre-existing gold rows are excluded). Persists human decisions to a separate
review JSON artifact; on completion imports via merge_partial_reviewer_labels.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from rcs.evaluation.calibration_labeling import (
    COMPLETED_ANNOTATION_STATUSES,
    GOLD_CSV,
    INTENT_CODE_TO_LABEL,
    SENTIMENT_CODE_TO_LABEL,
    SUBCATEGORY_CODE_TO_LABEL,
    TOPIC_CODE_TO_LABEL,
    _atomic_write_json,
    apply_human_answers,
    count_completed_rows,
    current_row,
    ensure_resume_index,
    first_incomplete_index,
    go_previous_row,
    is_review_complete,
    row_to_form_labels,
    save_review_state,
    topic_requires_subcategory,
    validate_human_answers,
)
from rcs.evaluation.gold_dataset import (
    mark_train_gold_complete,
    merge_partial_reviewer_labels,
    verify_post_b2_b3_invariants,
)

B3_REVIEW_JSON = settings.DATA_DIR / "gold" / "b3_human_review.json"
B3_COMPLETE_MARKER = settings.DATA_DIR / "gold" / "b3_train_gold_complete.json"
B3_LABELER_ID = "b3_review_wizard"
B3_ACCEPT_AI_NOTES = "اعتماد اقتراح b2-v3 (Gemini)"
B3_MANUAL_NOTES = "تصنيف بشري عبر معالج B.3"

B3_ROW_FIELDS = (
    "comment_id",
    "comment_text",
    "split",
    "annotation_stage",
    "is_few_shot_example",
    "ai_suggested_primary_intent",
    "ai_suggested_topic_subtype",
    "ai_suggested_secondary_intent",
    "ai_suggested_subcategory",
    "ai_suggested_sentiment",
    "ai_suggested_confidence",
    "ai_suggestion_model",
    "gold_primary_intent",
    "gold_topic_subtype",
    "gold_secondary_intent",
    "gold_subcategory",
    "gold_sentiment",
    "gold_annotation_status",
    "labeler_notes",
    "labeled_by",
    "labeled_at",
)

CSV_LOCKED_FIELDS = (
    "comment_id",
    "comment_text",
    "split",
    "annotation_stage",
    "is_few_shot_example",
    "ai_suggested_primary_intent",
    "ai_suggested_topic_subtype",
    "ai_suggested_secondary_intent",
    "ai_suggested_subcategory",
    "ai_suggested_sentiment",
    "ai_suggested_confidence",
    "ai_suggestion_model",
)

AI_DISPLAY_FIELDS = (
    "ai_suggested_primary_intent",
    "ai_suggested_topic_subtype",
    "ai_suggested_secondary_intent",
    "ai_suggested_subcategory",
    "ai_suggested_sentiment",
    "ai_suggested_confidence",
)


def _row_from_csv_record(record: pd.Series) -> dict[str, str]:
    row = {field: str(record.get(field, "")).strip() for field in B3_ROW_FIELDS}
    if row["annotation_stage"] != "ai_assisted_train":
        raise AssertionError(
            f"B.3 wizard cannot load protected stage {row['annotation_stage']!r}"
        )
    return row


def load_remaining_train_rows(gold_csv_path: Path = GOLD_CSV) -> list[dict[str, str]]:
    """Return ai_assisted_train rows that still need human gold labels."""
    if not gold_csv_path.exists():
        raise FileNotFoundError(f"canonical gold CSV not found: {gold_csv_path}")
    df = pd.read_csv(gold_csv_path, dtype=str, keep_default_na=False)
    subset = df[df["annotation_stage"] == "ai_assisted_train"].copy()
    pending = subset[
        ~subset["gold_annotation_status"].astype(str).str.strip().isin(COMPLETED_ANNOTATION_STATUSES)
    ].copy()
    pending["_conf_sort"] = pd.to_numeric(pending["ai_suggested_confidence"], errors="coerce")
    pending = pending.sort_values("_conf_sort", ascending=True, na_position="last")
    rows = [_row_from_csv_record(record) for _, record in pending.iterrows()]
    return rows


def empty_b3_review_state(gold_csv_path: Path = GOLD_CSV) -> dict[str, Any]:
    rows = load_remaining_train_rows(gold_csv_path)
    for row in rows:
        row["gold_primary_intent"] = ""
        row["gold_topic_subtype"] = ""
        row["gold_secondary_intent"] = ""
        row["gold_subcategory"] = ""
        row["gold_sentiment"] = ""
        row["gold_annotation_status"] = ""
        row["labeler_notes"] = ""
        row["labeled_by"] = ""
        row["labeled_at"] = ""
    return {
        "version": 1,
        "source_csv": str(gold_csv_path),
        "current_index": 0,
        "rows": rows,
        "updated_at": "",
        "completed_at": "",
    }


def load_or_init_b3_review_state(
    review_path: Path = B3_REVIEW_JSON,
    gold_csv_path: Path = GOLD_CSV,
) -> dict[str, Any]:
    source_rows = load_remaining_train_rows(gold_csv_path)
    source_by_id = {row["comment_id"]: row for row in source_rows}

    if review_path.exists():
        state = json.loads(review_path.read_text(encoding="utf-8"))
        merged_rows: list[dict[str, str]] = []
        for saved in state.get("rows", []):
            cid = saved["comment_id"]
            if cid not in source_by_id:
                raise AssertionError(
                    f"B.3 review artifact references row no longer pending review: {cid}"
                )
            merged = dict(source_by_id[cid])
            for field in B3_ROW_FIELDS:
                if field in CSV_LOCKED_FIELDS:
                    continue
                merged[field] = str(saved.get(field, "")).strip()
            merged_rows.append(merged)
        if len(merged_rows) != len(source_rows):
            raise AssertionError(
                "B.3 review artifact row count does not match pending train rows"
            )
        state["rows"] = merged_rows
        state["current_index"] = min(int(state.get("current_index", 0)), len(merged_rows) - 1)
        return ensure_resume_index(state)

    return empty_b3_review_state(gold_csv_path)


def save_b3_review_state(state: dict[str, Any], review_path: Path = B3_REVIEW_JSON) -> None:
    save_review_state(state, review_path)


def ai_suggestion_to_form_labels(row: dict[str, str]) -> dict[str, str]:
    return {
        "intent_label": INTENT_CODE_TO_LABEL.get(row.get("ai_suggested_primary_intent", ""), ""),
        "topic_label": TOPIC_CODE_TO_LABEL.get(row.get("ai_suggested_topic_subtype", ""), ""),
        "sentiment_label": SENTIMENT_CODE_TO_LABEL.get(row.get("ai_suggested_sentiment", ""), ""),
        "subcategory_label": SUBCATEGORY_CODE_TO_LABEL.get(row.get("ai_suggested_subcategory", ""), ""),
        "secondary_intent_label": INTENT_CODE_TO_LABEL.get(
            row.get("ai_suggested_secondary_intent", ""), ""
        ),
    }


def format_ai_suggestion_markdown(row: dict[str, str]) -> str:
    labels = ai_suggestion_to_form_labels(row)
    conf = row.get("ai_suggested_confidence", "") or "—"
    lines = [
        "### اقتراح Gemini (b2-v3) — للمساعدة فقط",
        f"- **الغرض:** {labels['intent_label'] or '—'}",
        f"- **الموضوع:** {labels['topic_label'] or '—'}",
        f"- **النبرة:** {labels['sentiment_label'] or '—'}",
    ]
    if labels["subcategory_label"]:
        lines.append(f"- **التصنيف الفرعي:** {labels['subcategory_label']}")
    if labels["secondary_intent_label"]:
        lines.append(f"- **نية ثانية:** {labels['secondary_intent_label']}")
    lines.append(f"- **الثقة:** {conf}")
    lines.append("")
    lines.append("_لن يُنسخ الاقتراح إلى الذهب إلا عند الضغط على «اعتماد اقتراح الذكاء»._")
    return "\n".join(lines)


def _apply_b3_labels(
    row: dict[str, str],
    *,
    intent_label: str,
    topic_label: str = "",
    sentiment_label: str = "",
    subcategory_label: str = "",
    secondary_intent_label: str = "",
    accepted_ai: bool = False,
) -> dict[str, str]:
    notes = B3_ACCEPT_AI_NOTES if accepted_ai else B3_MANUAL_NOTES
    return apply_human_answers(
        row,
        intent_label=intent_label,
        topic_label=topic_label,
        sentiment_label=sentiment_label,
        subcategory_label=subcategory_label,
        secondary_intent_label=secondary_intent_label,
        labeled_by=B3_LABELER_ID,
        labeler_notes=notes,
        use_date_only_labeled_at=True,
    )


def save_current_b3_row(
    state: dict[str, Any],
    *,
    intent_label: str,
    topic_label: str = "",
    sentiment_label: str = "",
    subcategory_label: str = "",
    secondary_intent_label: str = "",
    accepted_ai: bool = False,
    review_path: Path = B3_REVIEW_JSON,
) -> tuple[dict[str, Any], str | None]:
    error = validate_human_answers(
        intent_label=intent_label,
        topic_label=topic_label,
        sentiment_label=sentiment_label,
        subcategory_label=subcategory_label,
        secondary_intent_label=secondary_intent_label,
    )
    if error:
        return state, error

    state = dict(state)
    rows = [dict(row) for row in state["rows"]]
    index = state["current_index"]
    rows[index] = _apply_b3_labels(
        rows[index],
        intent_label=intent_label,
        topic_label=topic_label,
        sentiment_label=sentiment_label,
        subcategory_label=subcategory_label,
        secondary_intent_label=secondary_intent_label,
        accepted_ai=accepted_ai,
    )
    state["rows"] = rows
    save_b3_review_state(state, review_path)
    return state, None


def save_and_advance_b3_row(
    state: dict[str, Any],
    *,
    intent_label: str,
    topic_label: str = "",
    sentiment_label: str = "",
    subcategory_label: str = "",
    secondary_intent_label: str = "",
    accepted_ai: bool = False,
    review_path: Path = B3_REVIEW_JSON,
) -> tuple[dict[str, Any], str | None]:
    state, error = save_current_b3_row(
        state,
        intent_label=intent_label,
        topic_label=topic_label,
        sentiment_label=sentiment_label,
        subcategory_label=subcategory_label,
        secondary_intent_label=secondary_intent_label,
        accepted_ai=accepted_ai,
        review_path=review_path,
    )
    if error:
        return state, error

    state = dict(state)
    if state["current_index"] < len(state["rows"]) - 1:
        state["current_index"] += 1
        save_b3_review_state(state, review_path)
    return state, None


def accept_ai_suggestion_row(
    state: dict[str, Any],
    review_path: Path = B3_REVIEW_JSON,
) -> tuple[dict[str, Any], str | None]:
    row = current_row(state)
    labels = ai_suggestion_to_form_labels(row)
    if not labels["intent_label"]:
        return state, "اقتراح الذكاء غير صالح أو غير مكتمل (الغرض)."
    return save_current_b3_row(
        state,
        intent_label=labels["intent_label"],
        topic_label=labels["topic_label"],
        sentiment_label=labels["sentiment_label"],
        subcategory_label=labels["subcategory_label"],
        secondary_intent_label=labels["secondary_intent_label"],
        accepted_ai=True,
        review_path=review_path,
    )


def go_previous_b3_row(state: dict[str, Any], review_path: Path = B3_REVIEW_JSON) -> dict[str, Any]:
    return go_previous_row(state, review_path)


def b3_human_labels_by_id(state: dict[str, Any]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for row in state["rows"]:
        status = row.get("gold_annotation_status", "")
        if status not in COMPLETED_ANNOTATION_STATUSES:
            continue
        labels[row["comment_id"]] = {
            "gold_primary_intent": row.get("gold_primary_intent", ""),
            "gold_topic_subtype": row.get("gold_topic_subtype", ""),
            "gold_secondary_intent": row.get("gold_secondary_intent", ""),
            "gold_subcategory": row.get("gold_subcategory", ""),
            "gold_sentiment": row.get("gold_sentiment", ""),
            "gold_annotation_status": status,
            "labeler_notes": row.get("labeler_notes", ""),
            "labeled_by": row.get("labeled_by", "") or B3_LABELER_ID,
            "labeled_at": row.get("labeled_at", ""),
        }
    return labels


def prepare_b3_wizard_import(
    gold_csv_path: Path = GOLD_CSV,
    review_path: Path = B3_REVIEW_JSON,
) -> pd.DataFrame:
    if not review_path.exists():
        raise FileNotFoundError(f"B.3 review artifact not found: {review_path}")
    state = json.loads(review_path.read_text(encoding="utf-8"))
    if not is_review_complete(state):
        raise AssertionError("B.3 review is incomplete — finalize only when all rows are done")
    labels = b3_human_labels_by_id(state)
    canonical = pd.read_csv(gold_csv_path, dtype=str, keep_default_na=False)
    return merge_partial_reviewer_labels(
        canonical,
        labels,
        require_empty_existing_gold=True,
    )


def import_b3_wizard_labels(
    output_csv_path: Path,
    gold_csv_path: Path = GOLD_CSV,
    review_path: Path = B3_REVIEW_JSON,
) -> Path:
    merged = prepare_b3_wizard_import(gold_csv_path, review_path)
    verify_post_b2_b3_invariants(merged, require_b3_complete=True)

    output_csv_path = Path(output_csv_path)
    if not output_csv_path.parent.exists():
        raise FileNotFoundError(
            f"output directory does not exist: {output_csv_path.parent}"
        )

    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=output_csv_path.parent,
            prefix=f".{output_csv_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            temp_name = tmp.name
            merged.to_csv(tmp, index=False)
        os.replace(temp_name, output_csv_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return output_csv_path


def write_b3_complete_marker(
    state: dict[str, Any],
    gold_csv_path: Path = GOLD_CSV,
    marker_path: Path = B3_COMPLETE_MARKER,
) -> Path:
    gold = pd.read_csv(gold_csv_path, dtype=str, keep_default_na=False)
    verify_post_b2_b3_invariants(gold, require_b3_complete=True)
    mark_train_gold_complete(gold, marker_path)
    train = gold[gold["annotation_stage"] == "ai_assisted_train"]
    labeled = int((train["gold_annotation_status"] == "LABELED").sum())
    insufficient = int((train["gold_annotation_status"] == "INSUFFICIENT_CONTEXT").sum())
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "wizard_rows": len(state["rows"]),
            "source_csv": str(gold_csv_path),
            "review_json": str(B3_REVIEW_JSON),
            "labeled_count": labeled,
            "insufficient_context_count": insufficient,
            "ai_assisted_train_total": int(len(train)),
        }
    )
    _atomic_write_json(marker_path, payload)
    return marker_path


def finalize_b3_review(
    state: dict[str, Any],
    *,
    output_csv_path: Path = GOLD_CSV,
    gold_csv_path: Path = GOLD_CSV,
    review_path: Path = B3_REVIEW_JSON,
    marker_path: Path = B3_COMPLETE_MARKER,
) -> tuple[Path, Path, str]:
    if not is_review_complete(state):
        raise AssertionError("all pending B.3 rows must be reviewed before finalization")

    state = dict(state)
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    save_b3_review_state(state, review_path)

    csv_path = import_b3_wizard_labels(
        output_csv_path,
        gold_csv_path=gold_csv_path,
        review_path=review_path,
    )
    marker = write_b3_complete_marker(state, gold_csv_path=csv_path, marker_path=marker_path)

    train = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    train = train[train["annotation_stage"] == "ai_assisted_train"]
    labeled = int((train["gold_annotation_status"] == "LABELED").sum())
    insufficient = int((train["gold_annotation_status"] == "INSUFFICIENT_CONTEXT").sum())
    summary = (
        f"B.3 Train Gold Dataset — COMPLETE\n"
        f"ai_assisted_train rows: {len(train)}\n"
        f"LABELED: {labeled}\n"
        f"INSUFFICIENT_CONTEXT: {insufficient}\n"
        f"Canonical CSV: {csv_path}\n"
        f"Completion marker: {marker}"
    )
    return csv_path, marker, summary


def count_status_breakdown(state: dict[str, Any]) -> dict[str, int]:
    labeled = 0
    insufficient = 0
    for row in state["rows"]:
        status = row.get("gold_annotation_status", "")
        if status == "LABELED":
            labeled += 1
        elif status == "INSUFFICIENT_CONTEXT":
            insufficient += 1
    return {
        "total": len(state["rows"]),
        "completed": count_completed_rows(state),
        "labeled": labeled,
        "insufficient_context": insufficient,
        "remaining": len(state["rows"]) - count_completed_rows(state),
    }
