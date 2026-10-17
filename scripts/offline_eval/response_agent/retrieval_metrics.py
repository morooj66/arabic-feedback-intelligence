"""
Phase M5F — retrieval metric implementations for the offline evaluation
harness. Deliberately mirrors scripts/benchmarks/retrieval/metrics.py's
exact definitions (Recall@K = binary hit, MRR = mean reciprocal rank) --
duplicated rather than cross-imported because these are two independent,
self-contained script trees by this project's own established
convention (see M5B.2's own module docstring: "not an installed
package"). Pure stdlib, no side effects.

Official Knowledge and Approved Response Memory metrics are computed
SEPARATELY (two independent `AggregateMetrics` instances) — never
merged into one blended retrieval score, per instruction.
"""

from __future__ import annotations

from dataclasses import dataclass


def hit_at_k(ranked_candidate_ids: list[str], relevant_ids: frozenset[str], k: int) -> bool:
    if k <= 0:
        raise ValueError("k must be positive")
    return any(cid in relevant_ids for cid in ranked_candidate_ids[:k])


def reciprocal_rank(ranked_candidate_ids: list[str], relevant_ids: frozenset[str]) -> float:
    for position, cid in enumerate(ranked_candidate_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True)
class AggregateMetrics:
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    mrr: float | None
    num_queries: int


NO_DATA_METRICS = AggregateMetrics(recall_at_1=None, recall_at_3=None, recall_at_5=None, mrr=None, num_queries=0)


def compute_aggregate_metrics(
    per_query_rankings: list[list[str]],
    per_query_relevant: list[frozenset[str]],
) -> AggregateMetrics:
    """Zero-division-safe: an empty input returns `NO_DATA_METRICS`
    (all-None) rather than raising or dividing by zero -- distinct from
    scripts/benchmarks/retrieval/metrics.py's stricter benchmark-only
    version, since an evaluation harness must degrade safely when a
    corpus/query set is empty, not crash the whole run."""
    if len(per_query_rankings) != len(per_query_relevant):
        raise ValueError("per_query_rankings and per_query_relevant must be the same length")
    if not per_query_rankings:
        return NO_DATA_METRICS

    n = len(per_query_rankings)
    r1 = sum(hit_at_k(r, rel, 1) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    r3 = sum(hit_at_k(r, rel, 3) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    r5 = sum(hit_at_k(r, rel, 5) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    mrr = sum(reciprocal_rank(r, rel) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    return AggregateMetrics(recall_at_1=r1, recall_at_3=r3, recall_at_5=r5, mrr=mrr, num_queries=n)
