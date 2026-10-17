"""
Phase M5B.2 — retrieval metric implementations for the benchmark ONLY.

Definitions (recorded explicitly per instruction):

Recall@K (a.k.a. Hit Rate@K / Success@K in retrieval-QA literature):
    For one query, 1.0 if AT LEAST ONE relevant candidate appears in the
    top-K ranked results, else 0.0. Averaged over all queries. This is
    the correct definition for "did retrieval surface a usable
    example," which is this project's actual future use case (a
    generation step needs ONE good grounding/style example, not every
    relevant one) -- not the classic "fraction of all relevant items
    retrieved" definition, which would behave identically here anyway
    for every query with exactly one relevant ID, and is handled
    correctly (via `any()` over the relevant set) for the one
    deliberately multi-relevant query in dataset.py (Q28).

MRR (Mean Reciprocal Rank):
    For one query, 1 / (rank of the FIRST relevant candidate in the
    ranked results), or 0.0 if no relevant candidate appears at all.
    Averaged over all queries.

Both metrics take a ranked list of candidate_ids (already sorted by
whatever scoring the candidate model produced -- this module has no
opinion on how that ranking was produced) and a set of relevant IDs.
"""

from __future__ import annotations

from dataclasses import dataclass


def hit_at_k(ranked_candidate_ids: list[str], relevant_ids: frozenset[str], k: int) -> bool:
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = ranked_candidate_ids[:k]
    return any(cid in relevant_ids for cid in top_k)


def reciprocal_rank(ranked_candidate_ids: list[str], relevant_ids: frozenset[str]) -> float:
    for position, cid in enumerate(ranked_candidate_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True)
class AggregateMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    num_queries: int


def compute_aggregate_metrics(
    per_query_rankings: list[list[str]],
    per_query_relevant: list[frozenset[str]],
) -> AggregateMetrics:
    """`per_query_rankings[i]` is the full ranked candidate_id list a
    candidate model produced for `per_query_relevant[i]`'s query, in
    the SAME order across both lists. Every query in the benchmark
    contributes to every metric -- no query is silently dropped."""
    if len(per_query_rankings) != len(per_query_relevant):
        raise ValueError("per_query_rankings and per_query_relevant must be the same length")
    if not per_query_rankings:
        raise ValueError("cannot compute metrics over zero queries")

    n = len(per_query_rankings)
    r1 = sum(hit_at_k(r, rel, 1) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    r3 = sum(hit_at_k(r, rel, 3) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    r5 = sum(hit_at_k(r, rel, 5) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    mrr = sum(reciprocal_rank(r, rel) for r, rel in zip(per_query_rankings, per_query_relevant)) / n
    return AggregateMetrics(recall_at_1=r1, recall_at_3=r3, recall_at_5=r5, mrr=mrr, num_queries=n)
