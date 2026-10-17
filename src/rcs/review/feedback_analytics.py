"""
Phase 3 — feedback analytics over the Phase 2 canonical feedback layer
(rcs.review.feedback). Read-only aggregation; never writes to
review_items, never changes generation behavior.

Canonical-only by construction: every function here operates on
FeedbackExample objects already produced by
rcs.review.feedback.list_feedback_examples(), which already excludes
non-canonical DUPLICATE_CONTENT rows (see that module's docstring) — so
duplicate source rows can never inflate a rate or count computed here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from rcs.review.feedback import FeedbackExample

# Below this many reviews, a group's rates are statistically unreliable —
# flagged, never silently presented as a real signal. Chosen because the
# whole current corpus is ~12 reviews total, so most per-group breakdowns
# will be small by definition; this threshold exists to make that
# explicit rather than to imply 5+ is "enough" for a real conclusion.
MIN_RELIABLE_SAMPLE_SIZE = 5

UNCLASSIFIED_KEY = "غير مصنف"  # used when the grouping field is None

# Edit-magnitude buckets over edit_ratio (difflib similarity, 1.0 =
# identical). Purely descriptive of HOW MUCH text changed — not a
# judgment of quality. Thresholds are deliberately simple and documented
# here (the only place they're defined):
#   ratio >= 0.95        -> "unchanged"       (whitespace/punctuation-level)
#   0.70 <= ratio < 0.95  -> "light_edit"
#   ratio < 0.70          -> "substantial_edit"
EDIT_BUCKET_UNCHANGED_MIN = 0.95
EDIT_BUCKET_LIGHT_MIN = 0.70


def edit_bucket(edit_ratio: float | None) -> str | None:
    """Deterministic bucket label for an edit_ratio, or None if not
    applicable (edit_ratio is None — i.e. not an EDIT_AND_SEND case).
    Always keep the raw edit_ratio alongside this — the bucket is a
    convenience label, not a replacement metric."""
    if edit_ratio is None:
        return None
    if edit_ratio >= EDIT_BUCKET_UNCHANGED_MIN:
        return "unchanged"
    if edit_ratio >= EDIT_BUCKET_LIGHT_MIN:
        return "light_edit"
    return "substantial_edit"


@dataclass(frozen=True)
class GroupMetrics:
    key: str
    reviewed_count: int
    accepted_as_is_count: int
    accepted_as_is_rate: float | None
    edited_count: int
    edited_rate: float | None
    no_response_count: int
    no_response_rate: float | None
    average_edit_ratio: float | None
    median_edit_ratio: float | None
    small_sample: bool


def _group_metrics(key: str, examples: Sequence[FeedbackExample]) -> GroupMetrics:
    total = len(examples)
    accepted = sum(1 for e in examples if e.draft_accepted_as_is)
    edited = sum(1 for e in examples if e.was_edited)
    no_response = sum(1 for e in examples if e.no_response_needed)
    ratios = [e.edit_ratio for e in examples if e.edit_ratio is not None]
    return GroupMetrics(
        key=key,
        reviewed_count=total,
        accepted_as_is_count=accepted,
        accepted_as_is_rate=(accepted / total) if total else None,
        edited_count=edited,
        edited_rate=(edited / total) if total else None,
        no_response_count=no_response,
        no_response_rate=(no_response / total) if total else None,
        average_edit_ratio=(sum(ratios) / len(ratios)) if ratios else None,
        median_edit_ratio=statistics.median(ratios) if ratios else None,
        small_sample=total < MIN_RELIABLE_SAMPLE_SIZE,
    )


def _grouped(examples: Sequence[FeedbackExample], key_fn) -> list[GroupMetrics]:
    buckets: dict[str, list[FeedbackExample]] = {}
    for ex in examples:
        key = key_fn(ex) or UNCLASSIFIED_KEY
        buckets.setdefault(key, []).append(ex)
    return [_group_metrics(key, items) for key, items in sorted(buckets.items())]


@dataclass(frozen=True)
class FeedbackAnalytics:
    overall: GroupMetrics
    by_intent: list[GroupMetrics] = field(default_factory=list)
    by_topic: list[GroupMetrics] = field(default_factory=list)
    by_sentiment: list[GroupMetrics] = field(default_factory=list)
    by_prompt_version: list[GroupMetrics] = field(default_factory=list)


def compute_feedback_analytics(examples: Sequence[FeedbackExample]) -> FeedbackAnalytics:
    """Pure function — no DB access. `examples` must already be the
    canonical, completed-only list from list_feedback_examples()."""
    overall = _group_metrics("overall", examples)
    return FeedbackAnalytics(
        overall=overall,
        by_intent=_grouped(examples, lambda e: e.intent),
        by_topic=_grouped(examples, lambda e: e.topic),
        by_sentiment=_grouped(examples, lambda e: e.sentiment),
        by_prompt_version=_grouped(examples, lambda e: e.generation_prompt_version),
    )


def build_feedback_analytics(engine, *, source_file: str | None = None) -> FeedbackAnalytics:
    from rcs.review.feedback import list_feedback_examples

    examples = list_feedback_examples(engine, source_file=source_file)
    return compute_feedback_analytics(examples)
