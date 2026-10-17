"""
Phase 2 — Human Feedback Capture Layer.

This module does NOT train, fine-tune, or change any model. It only turns
already-persisted ReviewItem state into a clean, privacy-minimized,
canonical-aware read view of completed reviewer decisions — a trustworthy
source for future feedback analytics (Phase 3+).

Nothing here writes to review_items. Every field either already existed
on ReviewItem (see the mapping in list_feedback_examples' docstring) or is
computed at read time (draft_accepted_as_is / was_edited /
no_response_needed / edit_distance / edit_ratio) — never persisted, never
fed back into the reviewer-facing record, so historical rg-v7 metadata
and reviewer text are never rewritten.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from rcs.response_generation.prompt import redact_for_generation
from rcs.review.models import ReviewItem

FEEDBACK_CSV_COLUMNS = [
    "feedback_id",
    "comment_id",
    "source_reference",
    "comment_text",
    "intent",
    "topic",
    "sentiment",
    "ai_draft",
    "generation_provider",
    "generation_model",
    "generation_prompt_version",
    "reviewer_action",
    "reviewer_edit",
    "final_response",
    "draft_accepted_as_is",
    "was_edited",
    "no_response_needed",
    "edit_distance",
    "edit_ratio",
    "reviewer_id",
    "reviewer_display_name",
    "reviewed_at",
]


@dataclass(frozen=True)
class FeedbackExample:
    """One completed, canonical reviewer decision. No visitor PII fields
    (no visitor name/email/phone) — see redact_comment_text below for how
    comment_text itself is scrubbed of embedded phone/email substrings."""

    feedback_id: str  # == comment_id; a canonical ReviewItem produces exactly one example
    comment_id: str
    source_reference: str | None
    comment_text: str | None
    intent: str | None
    topic: str | None
    sentiment: str | None
    ai_draft: str | None
    generation_provider: str | None
    generation_model: str | None
    generation_prompt_version: str | None
    reviewer_action: str
    reviewer_edit: str | None
    final_response: str | None
    draft_accepted_as_is: bool
    was_edited: bool
    no_response_needed: bool
    edit_distance: int | None
    edit_ratio: float | None
    reviewer_id: str | None
    reviewer_display_name: str | None
    reviewed_at: datetime | None


def levenshtein_distance(a: str, b: str) -> int:
    """Classic character-level Levenshtein edit distance (insertions,
    deletions, substitutions) via the standard O(len(a)*len(b)) dynamic
    program. Deterministic and dependency-free.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    return prev[-1]


def similarity_ratio(a: str, b: str) -> float:
    """difflib.SequenceMatcher's standard similarity ratio in [0, 1] — 1.0
    means identical text, 0.0 means no matching content at all. Stdlib,
    deterministic, documented at
    https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.ratio
    """
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


def redact_comment_text(text: str | None) -> str | None:
    """Reuses the SAME phone/email redaction already applied before
    visitor text reaches Gemini (rcs.response_generation.prompt.
    redact_for_generation) — applied here only to the in-memory feedback
    view, exactly like the existing precedent documented in
    rcs.classification.redact ("never applied to what is stored
    locally"). The stored original_comment on ReviewItem is never
    modified."""
    if text is None:
        return None
    return redact_for_generation(text)


def _to_feedback_example(item: ReviewItem) -> FeedbackExample:
    action = item.reviewer_action
    accepted_as_is = action == "SEND_AS_IS"
    was_edited = action == "EDIT_AND_SEND"
    no_response_needed = action == "NO_RESPONSE_NEEDED"

    edit_distance: int | None = None
    edit_ratio: float | None = None
    if was_edited and item.suggested_final_reply and item.final_reply:
        edit_distance = levenshtein_distance(item.suggested_final_reply, item.final_reply)
        edit_ratio = similarity_ratio(item.suggested_final_reply, item.final_reply)

    return FeedbackExample(
        feedback_id=item.comment_id,
        comment_id=item.comment_id,
        source_reference=item.source_reference,
        comment_text=redact_comment_text(item.original_comment),
        intent=item.ai_primary_intent,
        topic=item.ai_topic_subtype,
        sentiment=item.ai_sentiment,
        ai_draft=item.suggested_final_reply,
        generation_provider=item.generation_provider,
        generation_model=item.generation_model_name,
        generation_prompt_version=item.generation_prompt_version,
        reviewer_action=action,
        reviewer_edit=item.reviewer_edit,
        final_response=item.final_reply,
        draft_accepted_as_is=accepted_as_is,
        was_edited=was_edited,
        no_response_needed=no_response_needed,
        edit_distance=edit_distance,
        edit_ratio=edit_ratio,
        reviewer_id=item.reviewer_id,
        reviewer_display_name=item.reviewer_display_name,
        reviewed_at=item.reviewed_at,
    )


def list_feedback_examples(engine, *, source_file: str | None = None) -> list[FeedbackExample]:
    """Canonical-aware read view of every COMPLETED reviewer decision.

    Field provenance — every field already existed on ReviewItem; nothing
    new was added to the schema for this list:
      comment_text          <- original_comment (redacted at read time)
      intent/topic/sentiment <- ai_primary_intent/ai_topic_subtype/ai_sentiment
                                 (the immutable, first-write-wins AI fields —
                                 not the reviewer-mutable primary_intent/
                                 topic_subtype working columns)
      ai_draft               <- suggested_final_reply (never rewritten after
                                 generation — edit_and_send explicitly
                                 preserves it)
      generation_provider/generation_model/generation_prompt_version
                              <- generation_provider/generation_model_name/
                                 generation_prompt_version
      reviewer_action/reviewer_edit/final_response
                              <- reviewer_action/reviewer_edit/final_reply
      reviewer_id/reviewer_display_name/reviewed_at
                              <- same-named columns
    draft_accepted_as_is/was_edited/no_response_needed/edit_distance/
    edit_ratio are computed here, never persisted.

    Canonical-only, by construction: only rows with reviewer_action set
    are included, and non-canonical DUPLICATE_CONTENT rows never receive
    a reviewer_action (they never enter the reviewer queue at all — see
    rcs.review.excel_review_flow.run_content_deduplication). The
    dedupe_canonical_id check below is an explicit, defense-in-depth
    restatement of that same guarantee, not a new mechanism.
    """
    with Session(engine) as session:
        stmt = select(ReviewItem).where(
            ReviewItem.reviewer_action.is_not(None),
            or_(
                ReviewItem.dedupe_canonical_id.is_(None),
                ReviewItem.dedupe_canonical_id == ReviewItem.comment_id,
            ),
        )
        if source_file is not None:
            stmt = stmt.where(ReviewItem.source_file == source_file)
        stmt = stmt.order_by(ReviewItem.source_row_index, ReviewItem.comment_id)
        items = session.execute(stmt).scalars().all()
        return [_to_feedback_example(item) for item in items]


def export_feedback_dataset(engine, dest: Path, *, source_file: str | None = None) -> Path:
    """Writes the internal (non-reviewer-facing) feedback dataset as CSV —
    one row per completed canonical reviewer decision, no visitor PII,
    no duplicate source-row inflation. Does not touch the normal
    reviewer-facing Excel export (rcs.review.excel_review_flow.
    export_reviewed_workbook) or the source workbook."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    examples = list_feedback_examples(engine, source_file=source_file)
    with dest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_CSV_COLUMNS)
        writer.writeheader()
        for ex in examples:
            row = asdict(ex)
            row["reviewed_at"] = ex.reviewed_at.isoformat() if ex.reviewed_at else ""
            writer.writerow(row)
    return dest


@dataclass(frozen=True)
class FeedbackSummary:
    completed_total: int
    send_as_is_count: int
    edit_and_send_count: int
    no_response_needed_count: int
    acceptance_as_is_rate: float | None
    edit_rate: float | None
    no_response_rate: float | None
    average_edit_ratio: float | None


def summarize_feedback(examples: Sequence[FeedbackExample]) -> FeedbackSummary:
    """Aggregate metrics over an already-canonical, already-completed
    example list. Pure function — no DB access, no PII in the output."""
    total = len(examples)
    send_as_is = sum(1 for e in examples if e.draft_accepted_as_is)
    edited = sum(1 for e in examples if e.was_edited)
    no_response = sum(1 for e in examples if e.no_response_needed)
    ratios = [e.edit_ratio for e in examples if e.edit_ratio is not None]
    return FeedbackSummary(
        completed_total=total,
        send_as_is_count=send_as_is,
        edit_and_send_count=edited,
        no_response_needed_count=no_response,
        acceptance_as_is_rate=(send_as_is / total) if total else None,
        edit_rate=(edited / total) if total else None,
        no_response_rate=(no_response / total) if total else None,
        average_edit_ratio=(sum(ratios) / len(ratios)) if ratios else None,
    )
