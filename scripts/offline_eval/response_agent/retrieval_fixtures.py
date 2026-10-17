"""
Phase M5F — controlled fixture corpora for evaluating the retrieval
METRIC MACHINERY, not the real E5 embedding model (that question was
already answered in M5B.2 -- see scripts/benchmarks/retrieval/, which
found intfloat/multilingual-e5-small the leading candidate on its own
28-query synthetic set, with no embedding downloaded here).

Today's actual production corpora (Official Knowledge and Approved
Response Memory) are BOTH empty/unconfigured (see config.settings.
OFFICIAL_KNOWLEDGE_SOURCE_PATH / APPROVED_RESPONSE_SOURCE_ALLOWLIST) --
this is documented explicitly, not silently worked around. This module
exists so Recall@K/MRR can still be exercised end-to-end against a
small, hand-labeled, clearly-synthetic ground truth, using a
deterministic keyword-overlap ranker (NOT a semantic model) purely to
prove the metric computation itself is correct.
"""

from __future__ import annotations

# --- Official Knowledge fixture corpus + labeled queries -------------------

_OK_CORPUS = {
    "OK-A": "طريقة اقتراح كلمة جديدة عبر نموذج الإضافة في الموقع",
    "OK-B": "سياسة مراجعة الملاحظات وتحويلها إلى الفريق المختص",
    "OK-C": "خطوات الإبلاغ عن خطأ إملائي في مدخل قائم",
    "OK-D": "شروط ترشيح كلمة لتكون كلمة العام",
}

OK_QUERIES = (
    ("OKQ1", "كيف أقترح كلمة جديدة؟", frozenset({"OK-A"})),
    ("OKQ2", "هل ملاحظتي تُحول لفريق مختص؟", frozenset({"OK-B"})),
    ("OKQ3", "وجدت خطأ إملائي في الكلمة", frozenset({"OK-C"})),
    ("OKQ4", "كيف أرشح كلمة العام؟", frozenset({"OK-D"})),
)

# --- Approved Response Memory fixture corpus + labeled queries -------------

_ARM_CORPUS = {
    "ARM-A": "شكرًا لملاحظتكم القيّمة سنطلع عليها قريبًا",
    "ARM-B": "نعتذر عن الإزعاج ونقدر صبركم",
    "ARM-C": "نشكر تواصلكم ونقدر اهتمامكم بالمعجم",
}

ARM_QUERIES = (
    ("ARMQ1", "شكرًا على المجهود", frozenset({"ARM-A", "ARM-C"})),
    ("ARMQ2", "أعتذر إذا كان هناك إزعاج", frozenset({"ARM-B"})),
)


def _keyword_overlap_rank(query_text: str, corpus: dict[str, str]) -> list[str]:
    """Deterministic, non-semantic: ranks by shared-character-bigram
    count with the query -- a controlled-fixture stand-in ranker ONLY,
    never the real embedding retriever. Ties broken by candidate id for
    determinism."""

    def bigrams(text: str) -> set[str]:
        return {text[i : i + 2] for i in range(len(text) - 1)}

    query_bigrams = bigrams(query_text)
    scored = [(cid, len(query_bigrams & bigrams(text))) for cid, text in corpus.items()]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [cid for cid, _ in scored]


def rank_official_knowledge_fixture(query_text: str) -> list[str]:
    return _keyword_overlap_rank(query_text, _OK_CORPUS)


def rank_approved_response_memory_fixture(query_text: str) -> list[str]:
    return _keyword_overlap_rank(query_text, _ARM_CORPUS)
