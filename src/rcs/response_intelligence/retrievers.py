"""
Phase M5C — dependency-injection defaults for the two M5B retrievers.

`generate_response_with_orchestrator` accepts an explicit retriever for
each corpus (real DI: any caller — production, an eval script, a test —
can pass its own, including a fake). When a caller passes neither, this
module supplies a process-lifetime-cached default so a normal request
never loads the E5 model itself and never rebuilds either index.

--- Why a module-level cache, not "no cache at all" ------------------------

Per instruction: "Use dependency injection for retrievers. Do not load
E5 per request." Those two constraints together require SOME shared
instance across requests when a caller doesn't inject its own — the
alternative (every caller must always pass explicit retrievers) would
mean touching every existing call site of
`generate_response_with_orchestrator` (production code in
`rcs.review.excel_review_flow`, several test files) just to keep
today's behavior unchanged. The cache below holds only immutable,
already-built retriever objects (never mutated after construction);
building it lazily on first use, once, and reusing it afterward, is a
memoization cache, not hidden business-logic state — `retrieve()` calls
against a cached retriever remain pure and stateless.

With TODAY's configuration (no `APPROVED_RESPONSE_SOURCE_ALLOWLIST`, no
`OFFICIAL_KNOWLEDGE_SOURCE_PATH`), building either default touches
`E5EmbeddingAdapter` but never actually loads/downloads it — both
`build_index`/`build_governed_*_corpus` skip the embedder entirely for
an empty corpus (see approved_memory.py/official_knowledge.py).
"""

from __future__ import annotations

from config import settings
from rcs.response_intelligence.retrieval.approved_memory import build_approved_response_memory_retriever
from rcs.response_intelligence.retrieval.contracts import ApprovedResponseMemoryRetriever, OfficialKnowledgeRetriever
from rcs.response_intelligence.retrieval.embeddings import E5EmbeddingAdapter, EmbeddingAdapter
from rcs.response_intelligence.retrieval.official_corpus import JsonFileOfficialSourceProvider
from rcs.response_intelligence.retrieval.official_knowledge import build_official_knowledge_retriever

_shared_embedder: EmbeddingAdapter | None = None
_official_knowledge_default: OfficialKnowledgeRetriever | None = None
_approved_response_default: ApprovedResponseMemoryRetriever | None = None


def _get_shared_embedder() -> EmbeddingAdapter:
    global _shared_embedder
    if _shared_embedder is None:
        _shared_embedder = E5EmbeddingAdapter()
    return _shared_embedder


def default_official_knowledge_retriever() -> OfficialKnowledgeRetriever:
    global _official_knowledge_default
    if _official_knowledge_default is None:
        path = settings.OFFICIAL_KNOWLEDGE_SOURCE_PATH
        source = JsonFileOfficialSourceProvider(path) if path else None
        _official_knowledge_default = build_official_knowledge_retriever(source, _get_shared_embedder())
    return _official_knowledge_default


def default_approved_response_memory_retriever() -> ApprovedResponseMemoryRetriever:
    global _approved_response_default
    if _approved_response_default is None:
        # No repository-side adapter exists yet (known limitation carried
        # over from M5B.3's final report) -- an empty tuple is the
        # honest current corpus, not a placeholder standing in for real
        # data. `APPROVED_RESPONSE_SOURCE_ALLOWLIST` being empty would
        # make any real candidates ineligible anyway.
        _approved_response_default = build_approved_response_memory_retriever((), _get_shared_embedder())
    return _approved_response_default


def reset_default_retrievers_for_testing() -> None:
    """Test-only escape hatch: clears the module-level cache so a test
    can observe a fresh build (e.g. to count embedder calls from zero).
    Never called by production code."""
    global _shared_embedder, _official_knowledge_default, _approved_response_default
    _shared_embedder = None
    _official_knowledge_default = None
    _approved_response_default = None
