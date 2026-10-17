"""
Phase M5B.4 — Official Knowledge retrieval: a real implementation of
`rcs.response_intelligence.retrieval.contracts.OfficialKnowledgeRetriever`.

Not wired into orchestrator.py, response_generation/service.py, Gemini
prompt construction, or the frontend in this phase — independently
callable and independently tested (see
tests/test_official_knowledge.py). Runtime integration is a later
phase, per instruction.

Reuses the exact E5 embedding adapter from embeddings.py (no
duplication) and the same explicit, one-shot index-build lifecycle
established in M5B.3's approved_memory.py.

--- Structured lookup vs. semantic retrieval ------------------------------

`contracts.OfficialKnowledgeQuery` (M5B.1, unmodified — this phase does
not touch contracts.py) has no separate `entry_id`/`lemma` field, only
`query_text`. Rather than add one (which would touch a protected file
without absolute need), a structured/exact lookup is expressed through
the SAME `query_text` channel: if `query_text` (stripped) exactly
matches a candidate record's `source_id`, `entry_id`, or `lemma`, that
counts as a structured hit and is returned WITHOUT ever calling the
embedder — satisfying "prefer structured lookup first" and "do not run
embeddings unnecessarily when an exact match is available" without any
contract change. A caller with only free-text context (not an exact
key) naturally falls through to semantic ranking, since ordinary
sentences essentially never equal a stored source_id/entry_id/lemma
exactly.

--- Staleness / conflict policy (explicit, not invented) -----------------

`effective_from`/`effective_to` are checked at query time (via an
injectable `clock`, defaulting to `datetime.datetime.now`) as a
deterministic temporal-validity filter: a record not yet effective, or
already superseded, is excluded from that call's candidates entirely —
this is the one "deterministic source precedence" this phase implements,
because it is the one rule that is genuinely explicit and unambiguous
(a plain date-range check), unlike an invented cross-source priority
ranking. If, after that filter, a structured exact-key lookup still
matches more than one currently-effective record (a genuine
simultaneous conflict with no explicit organizational precedence rule
to break it), ALL of them are returned as separate, individually
distinguishable results (deterministically ordered by source_id) rather
than the implementation silently picking a "winner" by score or by
guesswork — the caller sees the ambiguity instead of it being hidden.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from rcs.response_intelligence.retrieval.contracts import (
    ERROR,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    UNAVAILABLE,
    OfficialKnowledgeQuery,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
)
from rcs.response_intelligence.retrieval.embeddings import EmbeddingAdapter, EmbeddingModelUnavailable
from rcs.response_intelligence.retrieval.official_corpus import (
    OfficialCorpusRecord,
    OfficialSourceLoadError,
    OfficialSourceProvider,
    RawOfficialSourceRecord,
    build_governed_official_corpus,
)


@dataclass(frozen=True)
class OfficialKnowledgeIndex:
    """An already-built (or explicitly unavailable) index. `records[i]`
    corresponds to row `i` of `embeddings` -- built once, together, by
    `build_index`; nothing else constructs or mutates this pairing."""

    records: tuple[OfficialCorpusRecord, ...] = ()
    embeddings: np.ndarray | None = None
    unavailable_reason: str | None = None

    @property
    def record_count(self) -> int:
        return len(self.records)

    def filtered_indices(self, *, intent: str | None, topic: str | None, now: dt.datetime) -> list[int]:
        """Structured intent/topic filters AND the temporal-validity
        check both happen BEFORE any exact-match or semantic ranking --
        see module docstring's staleness/conflict policy."""
        out = []
        for i, record in enumerate(self.records):
            if intent is not None and record.intent != intent:
                continue
            if topic is not None and record.topic != topic:
                continue
            if record.effective_from > now:
                continue
            if record.effective_to is not None and record.effective_to <= now:
                continue
            out.append(i)
        return out

    def exact_match_indices(self, candidate_indices: list[int], query_text: str) -> list[int]:
        key = (query_text or "").strip()
        if not key:
            return []
        out = []
        for i in candidate_indices:
            record = self.records[i]
            if key == record.source_id or key == record.entry_id or key == record.lemma:
                out.append(i)
        return out


def build_index(records: tuple[OfficialCorpusRecord, ...], embedder: EmbeddingAdapter) -> OfficialKnowledgeIndex:
    """Explicit, one-shot index construction -- embeds every record's
    `content` exactly once. If the governed corpus is empty, the
    embedder is never touched at all."""
    if not records:
        return OfficialKnowledgeIndex(records=(), embeddings=None)

    try:
        if not embedder.is_loaded:
            embedder.load()
        embeddings = embedder.embed_passages([r.content for r in records])
    except EmbeddingModelUnavailable as exc:
        return OfficialKnowledgeIndex(
            records=(), embeddings=None, unavailable_reason=str(exc) or "embedding model unavailable"
        )

    return OfficialKnowledgeIndex(records=records, embeddings=embeddings)


def _rank_exact(
    index: OfficialKnowledgeIndex, indices: list[int], max_results: int
) -> list[tuple[OfficialCorpusRecord, float]]:
    """1.0 denotes an exact structured match -- not a semantic
    similarity score. Deterministic order: source_id ascending."""
    ordered = sorted((index.records[i] for i in indices), key=lambda r: r.source_id)
    return [(r, 1.0) for r in ordered[:max_results]]


def _rank_semantic(
    index: OfficialKnowledgeIndex, candidate_indices: list[int], query_vector: np.ndarray, max_results: int
) -> list[tuple[OfficialCorpusRecord, float]]:
    """Same deterministic ranking discipline as approved_memory.py's
    _rank: descending score, ties broken by source_id ascending."""
    scored = []
    for i in candidate_indices:
        score = float(index.embeddings[i] @ query_vector)
        scored.append((index.records[i], score))
    scored.sort(key=lambda pair: (-pair[1], pair[0].source_id))
    return scored[:max_results]


def _to_result(record: OfficialCorpusRecord, score: float) -> OfficialKnowledgeResult:
    return OfficialKnowledgeResult(
        source_id=record.source_id,
        source_title=record.source_title,
        source_type=record.source_type,
        content=record.content,
        version=record.version,
        effective_from=record.effective_from,
        authoritative=True,
        retrieval_score=score,
        effective_to=record.effective_to,
    )


@dataclass
class OfficialKnowledgeRetrieverImpl:
    """Structurally satisfies `contracts.OfficialKnowledgeRetriever` (a
    plain Protocol -- no explicit inheritance, matching this codebase's
    convention). Holds an already-built index; `.retrieve()` never
    mutates it and never rebuilds embeddings. `clock` is injectable so
    tests can pin "now" for deterministic temporal-validity checks."""

    index: OfficialKnowledgeIndex
    embedder: EmbeddingAdapter
    clock: Callable[[], dt.datetime] = field(default=dt.datetime.now)

    def retrieve(self, query: OfficialKnowledgeQuery) -> OfficialKnowledgeRetrievalResult:
        if self.index.unavailable_reason is not None:
            return OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, failure_detail=self.index.unavailable_reason)

        if self.index.record_count == 0:
            return OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)

        now = self.clock()
        candidate_indices = self.index.filtered_indices(intent=query.intent, topic=query.topic, now=now)
        if not candidate_indices:
            return OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)

        exact_indices = self.index.exact_match_indices(candidate_indices, query.query_text)
        if exact_indices:
            ranked = _rank_exact(self.index, exact_indices, query.max_results)
            results = tuple(_to_result(record, score) for record, score in ranked)
            return OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=results)

        # No enforceable deadline exists for this synchronous, in-process
        # call -- consistent with M5B.3a's corrected timeout semantics,
        # no elapsed-time-based classification is performed here either.
        try:
            query_vector = self.embedder.embed_query(query.query_text)
            ranked = _rank_semantic(self.index, candidate_indices, query_vector, query.max_results)
        except EmbeddingModelUnavailable:
            return OfficialKnowledgeRetrievalResult(
                outcome=UNAVAILABLE,
                failure_detail="official knowledge embedding model unavailable",
            )
        except Exception:  # noqa: BLE001 -- never leak a raw exception/path through the public contract
            return OfficialKnowledgeRetrievalResult(
                outcome=ERROR,
                failure_detail="official knowledge retrieval failed unexpectedly",
            )

        if not ranked:
            return OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)

        results = tuple(_to_result(record, score) for record, score in ranked)
        return OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=results)


def build_official_knowledge_retriever(
    source: OfficialSourceProvider | None,
    embedder: EmbeddingAdapter,
) -> OfficialKnowledgeRetrieverImpl:
    """The one explicit build/rebuild entry point.

    `source=None` (the default posture in this repository today, since
    no trustworthy local official source exists — see official_corpus.py's
    audit note) means no source is even attempted: the resulting
    retriever always returns SUCCESS_EMPTY, and the embedder is never
    touched. Passing a real `OfficialSourceProvider` that raises
    `OfficialSourceLoadError` produces an UNAVAILABLE-flagged index
    instead of crashing or falling back to Approved Response Memory."""
    if source is None:
        raw_records: Iterable[RawOfficialSourceRecord] = ()
    else:
        try:
            raw_records = source.fetch_records()
        except OfficialSourceLoadError as exc:
            unavailable_index = OfficialKnowledgeIndex(records=(), embeddings=None, unavailable_reason=str(exc))
            return OfficialKnowledgeRetrieverImpl(index=unavailable_index, embedder=embedder)

    governed_records = build_governed_official_corpus(raw_records)
    index = build_index(governed_records, embedder)
    return OfficialKnowledgeRetrieverImpl(index=index, embedder=embedder)
