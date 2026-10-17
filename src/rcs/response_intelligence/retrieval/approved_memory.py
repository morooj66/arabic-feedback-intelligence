"""
Phase M5B.3 — Approved Response Memory retrieval: a real, in-memory,
E5-embedding-backed implementation of
`rcs.response_intelligence.retrieval.contracts.ApprovedResponseMemoryRetriever`.

Not wired into orchestrator.py, response_generation/service.py, Gemini
prompt construction, or the frontend in this phase — this module is
independently callable and independently tested (see
tests/test_approved_response_memory.py). Runtime integration is a later
phase (M5C), per instruction.

--- Index lifecycle (explicit, no hidden state) --------------------------

`build_approved_response_memory_retriever(...)` is the only way to get a
working retriever. It:
  1. Runs `corpus.build_governed_corpus` once (pure, deterministic).
  2. If the governed corpus is non-empty, loads the embedder (if not
     already loaded) and embeds every corpus record's response_text
     exactly once, building a normalized embedding matrix.
  3. Returns an `ApprovedResponseMemoryRetrieverImpl` wrapping that
     already-built index.

There is no lazy/implicit rebuild inside `.retrieve()` — embeddings are
never recomputed per query. To refresh from newer governed data, call
`build_approved_response_memory_retriever(...)` again with new raw
candidates; this produces a brand-new retriever/index rather than
mutating anything in place, so there is no hidden mutable global index.

--- TIMEOUT semantics (M5B.3a correction) ---------------------------------

`ApprovedResponseMemoryRetrieverImpl.retrieve` NEVER returns `TIMEOUT`.

M5B.3's first implementation measured elapsed wall-clock time AFTER a
synchronous embed_query()+rank() call had already returned, and reported
`TIMEOUT` if that exceeded a budget. That was removed here because it is
dishonest about what actually happened: the call was never bounded or
cancellable in the first place, so retroactively labeling an unusually
slow *completed* call "TIMEOUT" implies a protection that does not
exist. This implementation is a synchronous, in-process Python call with
no thread/process boundary; Python has no cooperative-cancellation
mechanism for a running CPU-bound call, and building real preemptive
cancellation (a worker thread/process with a hard kill) is exactly the
kind of complexity the M5B.3a instruction says not to add merely to
satisfy the name "TIMEOUT."

`UNAVAILABLE` (embedding model/resource unusable) and `ERROR` (sanitized
unexpected failure) are this implementation's actual reachable failure
outcomes — see `retrieve` below. `TIMEOUT` remains a valid value in the
shared `contracts.RetrievalOutcome` (untouched by this phase) for a
future provider that can genuinely enforce a deadline — e.g. a
network-bound or subprocess-isolated implementation with a real
cancellable boundary. Nothing about this decision touches contracts.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from rcs.response_intelligence.retrieval.contracts import (
    ERROR,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    UNAVAILABLE,
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryResult,
    ApprovedResponseMemoryRetrievalResult,
)
from rcs.response_intelligence.retrieval.corpus import (
    ApprovedResponseCorpusRecord,
    RawApprovedResponseCandidate,
    build_governed_corpus,
)
from rcs.response_intelligence.retrieval.embeddings import EmbeddingAdapter, EmbeddingModelUnavailable


@dataclass(frozen=True)
class ApprovedResponseIndex:
    """An already-built (or explicitly unavailable) index. `records[i]`
    corresponds to row `i` of `embeddings` -- built once, together, by
    `build_index`; nothing else constructs or mutates this pairing."""

    records: tuple[ApprovedResponseCorpusRecord, ...] = ()
    embeddings: np.ndarray | None = None
    unavailable_reason: str | None = None

    @property
    def record_count(self) -> int:
        return len(self.records)

    def filtered_indices(
        self, *, intent: str | None, topic: str | None, sentiment: str | None
    ) -> list[int]:
        """Structured filtering happens BEFORE any semantic ranking --
        an unmatched filter yields an empty list here, and the caller
        must treat that as SUCCESS_EMPTY without ever invoking the
        embedder for the query (see ApprovedResponseMemoryRetrieverImpl.
        retrieve)."""
        out = []
        for i, record in enumerate(self.records):
            if intent is not None and record.primary_intent != intent:
                continue
            if topic is not None and record.topic_subtype != topic:
                continue
            if sentiment is not None and record.sentiment != sentiment:
                continue
            out.append(i)
        return out


def build_index(
    records: tuple[ApprovedResponseCorpusRecord, ...],
    embedder: EmbeddingAdapter,
) -> ApprovedResponseIndex:
    """Explicit, one-shot index construction. Embeds every record's
    response_text exactly once. If the governed corpus is empty, the
    embedder is never touched at all -- there is nothing to embed, and
    no reason to require the model to even be available."""
    if not records:
        return ApprovedResponseIndex(records=(), embeddings=None)

    try:
        if not embedder.is_loaded:
            embedder.load()
        embeddings = embedder.embed_passages([r.response_text for r in records])
    except EmbeddingModelUnavailable as exc:
        return ApprovedResponseIndex(
            records=(), embeddings=None, unavailable_reason=str(exc) or "embedding model unavailable"
        )

    return ApprovedResponseIndex(records=records, embeddings=embeddings)


def _rank(
    index: ApprovedResponseIndex, candidate_indices: list[int], query_vector: np.ndarray, max_results: int
) -> list[tuple[ApprovedResponseCorpusRecord, float]]:
    """Deterministic ranking: descending score, ties broken by
    review_item_id ascending (a stable id) -- never insertion order,
    never floating-point-comparison luck."""
    scored = []
    for i in candidate_indices:
        score = float(index.embeddings[i] @ query_vector)
        scored.append((index.records[i], score))
    scored.sort(key=lambda pair: (-pair[1], pair[0].review_item_id))
    return scored[:max_results]


def _to_result(record: ApprovedResponseCorpusRecord, score: float) -> ApprovedResponseMemoryResult:
    return ApprovedResponseMemoryResult(
        review_item_id=record.review_item_id,
        final_reply_id=record.final_reply_id,
        response_text=record.response_text,
        reviewer_action=record.reviewer_action,
        approved_at=record.approved_at,
        retrieval_score=score,
        primary_intent=record.primary_intent,
        topic_subtype=record.topic_subtype,
    )


@dataclass
class ApprovedResponseMemoryRetrieverImpl:
    """Structurally satisfies `contracts.ApprovedResponseMemoryRetriever`
    (a plain Protocol -- no explicit inheritance needed, matching this
    codebase's established convention). Holds an already-built index;
    `.retrieve()` never mutates it and never rebuilds embeddings."""

    index: ApprovedResponseIndex
    embedder: EmbeddingAdapter

    def retrieve(self, query: ApprovedResponseMemoryQuery) -> ApprovedResponseMemoryRetrievalResult:
        if self.index.unavailable_reason is not None:
            return ApprovedResponseMemoryRetrievalResult(
                outcome=UNAVAILABLE, failure_detail=self.index.unavailable_reason
            )

        if self.index.record_count == 0:
            return ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)

        candidate_indices = self.index.filtered_indices(
            intent=query.intent, topic=query.topic, sentiment=query.sentiment
        )
        if not candidate_indices:
            return ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)

        # No elapsed-time-based timeout classification here -- see the
        # module docstring's timeout-semantics note. This call is
        # synchronous and in-process with no enforceable deadline;
        # UNAVAILABLE and ERROR below are this implementation's only
        # reachable failure outcomes.
        try:
            query_vector = self.embedder.embed_query(query.query_text)
            ranked = _rank(self.index, candidate_indices, query_vector, query.max_results)
        except EmbeddingModelUnavailable:
            return ApprovedResponseMemoryRetrievalResult(
                outcome=UNAVAILABLE,
                failure_detail="approved response memory embedding model unavailable",
            )
        except Exception:  # noqa: BLE001 -- never leak a raw exception/path through the public contract
            return ApprovedResponseMemoryRetrievalResult(
                outcome=ERROR,
                failure_detail="approved response memory retrieval failed unexpectedly",
            )

        if not ranked:
            return ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)

        results = tuple(_to_result(record, score) for record, score in ranked)
        return ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=results)


def build_approved_response_memory_retriever(
    raw_candidates: Iterable[RawApprovedResponseCandidate],
    embedder: EmbeddingAdapter,
    *,
    allowlist: frozenset[str] | None = None,
) -> ApprovedResponseMemoryRetrieverImpl:
    """The one explicit build/rebuild entry point. `allowlist` defaults
    to `config.settings.APPROVED_RESPONSE_SOURCE_ALLOWLIST` (empty by
    default -- fails closed) only when not given explicitly, so tests
    can exercise a non-empty allowlist without touching real settings."""
    if allowlist is None:
        from config import settings

        allowlist = settings.APPROVED_RESPONSE_SOURCE_ALLOWLIST

    governed_records = build_governed_corpus(raw_candidates, allowlist=allowlist)
    index = build_index(governed_records, embedder)
    return ApprovedResponseMemoryRetrieverImpl(index=index, embedder=embedder)
