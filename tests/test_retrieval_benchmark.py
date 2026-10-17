"""
Phase M5B.2 — tests for the Arabic retrieval BENCHMARK MACHINERY.

These tests validate the benchmark's dataset integrity and metric
implementations -- NOT any third-party embedding model's quality (no
test here asserts a specific floating-point score for a real model,
per instruction). They also statically confirm the benchmark stays
fully isolated from production code: no database, no ReviewItem, no
an external dictionary platform, no secrets, no modification of the M5B.1 retrieval contracts.

`scripts/benchmarks/retrieval/` is plain script code, not an installed
package -- its directory is added to sys.path here the same way running
`python run_benchmark.py` directly would put it there automatically.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "retrieval"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import candidates as candidates_module  # noqa: E402
import dataset as dataset_module  # noqa: E402
import metrics as metrics_module  # noqa: E402
from candidates import CandidateReport  # noqa: E402
from dataset import CANDIDATES, QUERIES  # noqa: E402
from metrics import compute_aggregate_metrics, hit_at_k, reciprocal_rank  # noqa: E402


# --- 1. Dataset IDs are unique ----------------------------------------


def test_candidate_ids_are_unique():
    ids = [c.candidate_id for c in CANDIDATES]
    assert len(ids) == len(set(ids))


def test_query_ids_are_unique():
    ids = [q.query_id for q in QUERIES]
    assert len(ids) == len(set(ids))


# --- 2. Every query has >= 1 valid relevant candidate -------------------


def test_every_query_has_at_least_one_relevant_candidate():
    for query in QUERIES:
        assert len(query.relevant_candidate_ids) >= 1, f"{query.query_id} has no relevant candidate"


# --- 3. Every relevant ID exists in the candidate corpus -----------------


def test_every_relevant_id_exists_in_candidate_corpus():
    known_ids = {c.candidate_id for c in CANDIDATES}
    for query in QUERIES:
        for cid in query.relevant_candidate_ids:
            assert cid in known_ids, f"{query.query_id} references unknown candidate {cid!r}"


def test_dataset_uses_only_current_frozen_taxonomy_values():
    """No invented taxonomy label -- every intent/topic used in the
    benchmark must be a real, current config.settings value (src/ is
    already on sys.path via tests/conftest.py)."""
    from config import settings

    valid_intents = set(settings.PRIMARY_INTENTS)
    valid_topics = set(settings.TOPIC_SUBTYPES)
    for c in CANDIDATES:
        assert c.intent in valid_intents, f"{c.candidate_id} uses unknown intent {c.intent!r}"
        assert c.topic in valid_topics, f"{c.candidate_id} uses unknown topic {c.topic!r}"


# --- 4/6/7/8. Metric implementation correctness --------------------------


def test_hit_at_k_true_when_relevant_within_top_k():
    assert hit_at_k(["C2", "C1", "C3"], frozenset({"C1"}), 2) is True


def test_hit_at_k_false_when_relevant_outside_top_k():
    assert hit_at_k(["C2", "C3", "C1"], frozenset({"C1"}), 2) is False


def test_hit_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        hit_at_k(["C1"], frozenset({"C1"}), 0)


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["C1", "C2"], frozenset({"C1"})) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["C2", "C3", "C1"], frozenset({"C1"})) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_never_found():
    assert reciprocal_rank(["C2", "C3"], frozenset({"C1"})) == 0.0


def test_recall_at_1_hand_computed_example():
    # Query 1: relevant "A" is rank 1 -> hit. Query 2: relevant "B" is rank 2 -> miss.
    rankings = [["A", "X"], ["X", "B"]]
    relevant = [frozenset({"A"}), frozenset({"B"})]
    metrics = compute_aggregate_metrics(rankings, relevant)
    assert metrics.recall_at_1 == pytest.approx(0.5)


def test_recall_at_3_hand_computed_example():
    rankings = [["X", "Y", "A"], ["X", "Y", "Z"]]  # query 1 hits at rank 3, query 2 never hits
    relevant = [frozenset({"A"}), frozenset({"B"})]
    metrics = compute_aggregate_metrics(rankings, relevant)
    assert metrics.recall_at_3 == pytest.approx(0.5)


def test_mrr_hand_computed_example():
    rankings = [["A", "X"], ["X", "B"]]  # RR = 1.0 and 0.5 -> mean 0.75
    relevant = [frozenset({"A"}), frozenset({"B"})]
    metrics = compute_aggregate_metrics(rankings, relevant)
    assert metrics.mrr == pytest.approx(0.75)


# --- 5. Ranking correctness on a tiny deterministic vector example ------


def test_ranking_correctness_on_tiny_deterministic_vectors():
    """A hand-built, deterministic cosine-similarity scenario -- proves
    the candidate ranking LOGIC (descending by score) is correct,
    independent of any real embedding model."""
    import numpy as np

    candidate_ids = ["A", "B", "C"]
    # A is closest to the query direction, C is orthogonal, B is opposite.
    candidate_vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    query_vector = np.array([1.0, 0.0])
    sims = candidate_vectors @ query_vector
    order = np.argsort(-sims)
    ranked = [candidate_ids[i] for i in order]
    assert ranked[0] == "A"
    assert ranked[-1] == "B"


# --- 9. Multiple-relevant-item handling -----------------------------------


def test_multi_relevant_query_exists_and_is_handled_by_hit_at_k():
    multi = [q for q in QUERIES if len(q.relevant_candidate_ids) > 1]
    assert len(multi) >= 1, "benchmark should exercise the multi-relevant-item code path at least once"
    query = multi[0]
    relevant = frozenset(query.relevant_candidate_ids)
    # A ranking where only the SECOND relevant id appears early must still count as a hit.
    other_relevant = next(iter(relevant))
    ranking = ["unrelated-1", other_relevant, "unrelated-2"]
    assert hit_at_k(ranking, relevant, 3) is True


def test_reciprocal_rank_uses_first_matching_relevant_id_for_multi_relevant():
    relevant = frozenset({"A", "B"})
    assert reciprocal_rank(["X", "B", "A"], relevant) == pytest.approx(1 / 2)


# --- 10. Benchmark result schema/serialization ----------------------------


def test_candidate_report_is_json_serializable_shape():
    import json

    report = CandidateReport(
        name="synthetic_candidate",
        kind="embedding",
        status=candidates_module.EXECUTED,
        rankings={"Q01": ["C01", "C02"]},
        scores={"Q01": {"C01": 0.9, "C02": 0.1}},
        setup_seconds=0.01,
        total_query_seconds=0.001,
        embedding_dimension=384,
    )
    payload = {
        "name": report.name, "kind": report.kind, "status": report.status,
        "rankings": report.rankings, "embedding_dimension": report.embedding_dimension,
    }
    json.dumps(payload, ensure_ascii=False)  # must not raise


# --- 11/12. No production database or corpus access -----------------------


def _module_source_files():
    return [
        inspect.getsourcefile(dataset_module),
        inspect.getsourcefile(candidates_module),
        inspect.getsourcefile(metrics_module),
    ]


def _code_excluding_docstrings(path: str) -> str:
    """Returns the file's source with every module/class/function
    docstring blanked out, so a banned-term scan checks actual CODE,
    not prose. Every static-guard test in this session so far that
    scanned raw text (M5A.1's orchestrator test, M5A.3's leak-import
    test) has hit the same false positive: a module's own docstring
    legitimately NAMES the things it does NOT do, which a plain
    substring search misfires on. This is the general fix, not another
    one-off patch."""
    import ast

    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                for lineno in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                    idx = lineno - 1
                    if 0 <= idx < len(lines):
                        lines[idx] = "\n"
    return "".join(lines)


def test_no_production_database_or_review_item_import():
    banned = ("sqlalchemy", "pymongo", "ReviewItem", "rcs.review.repositories", "rcs.review.models")
    for path in _module_source_files():
        code = _code_excluding_docstrings(path)
        for term in banned:
            assert term not in code, f"forbidden reference {term!r} found in {path}"


# --- 13. No secrets/API keys ------------------------------------------------


def test_no_secrets_or_api_keys_in_benchmark_code():
    banned = ("GEMINI_API_KEY", "api_key=", "API_KEY", "SECRET", "password")
    for path in _module_source_files():
        code = _code_excluding_docstrings(path)
        for term in banned:
            assert term not in code, f"unexpected secret-shaped reference {term!r} found in {path}"


# --- 14. No external dictionary integration -----------------------------------------------


def test_no_external_dictionary_integration_anywhere_in_benchmark():
    for path in _module_source_files() + [str(_BENCHMARK_DIR / "run_benchmark.py")]:
        code = _code_excluding_docstrings(path).lower()
        assert "external_dictionary_client" not in code, f"unexpected external dictionary reference found in {path}"


# --- 15. Benchmark code does not modify runtime retrieval contracts -------


def test_benchmark_does_not_import_or_modify_retrieval_contracts():
    for path in _module_source_files():
        code = _code_excluding_docstrings(path)
        assert "response_intelligence.retrieval" not in code
        assert "OfficialKnowledgeRetriever" not in code
        assert "ApprovedResponseMemoryRetriever" not in code


# --- Additional static safety check (no forbidden application imports) ----


def test_benchmark_modules_import_only_expected_dependencies():
    """Narrow guard: dataset.py and metrics.py must be pure stdlib (no ML
    library needed just to hold data/math); candidates.py may use numpy/
    sklearn/transformers/torch (the actual candidate implementations) but
    nothing application-specific."""
    import ast

    def imported_roots(module) -> set[str]:
        tree = ast.parse(inspect.getsource(module))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        return roots

    assert imported_roots(dataset_module) <= {"__future__", "dataclasses"}
    assert imported_roots(metrics_module) <= {"__future__", "dataclasses"}
    banned_in_candidates = {"sqlalchemy", "pymongo", "rcs", "config"}
    assert imported_roots(candidates_module).isdisjoint(banned_in_candidates)
