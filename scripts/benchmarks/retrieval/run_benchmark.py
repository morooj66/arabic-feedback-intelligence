"""
Phase M5B.2 — Arabic retrieval benchmark entrypoint.

Run:
    .venv/bin/python scripts/benchmarks/retrieval/run_benchmark.py

Writes a small JSON report to
scripts/benchmarks/retrieval/results/m5b2_report.json (aggregate
metrics + per-query ranks + status + latency -- no raw vectors, no
embeddings committed) and prints a human-readable summary + qualitative
failure highlights to stdout.

Isolation guarantees (see also test_retrieval_benchmark.py):
  - No import of sqlalchemy, pymongo, rcs.review.models, any production
    repository, external-dictionary client, application auth, or Gemini generation.
  - No network call except each embedding candidate's own model
    download/load through the `transformers` library (documented, not
    hidden) -- there is no external-dictionary/organization-database network call
    anywhere in this file.
  - No write to any database. Output is one local JSON file under this
    same experimental directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Run as a plain script (`python scripts/benchmarks/retrieval/run_benchmark.py`):
# Python puts this file's own directory on sys.path[0] automatically, so the
# sibling modules below import by bare name -- no path manipulation, no
# dependency on `rcs.*`/`src/` at all, since this benchmark is fully isolated
# from the application package.
from candidates import (  # noqa: E402
    EXECUTED,
    FAILED_RUNTIME,
    NOT_EXECUTED_DEPENDENCY_MISSING,
    NOT_EXECUTED_DOWNLOAD_BLOCKED,
    CandidateReport,
    TfidfLexicalBaseline,
    TransformerMeanPoolEmbeddingCandidate,
)
from dataset import CANDIDATES, QUERIES  # noqa: E402
from metrics import compute_aggregate_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORT_PATH = RESULTS_DIR / "m5b2_report.json"


def _build_candidate_runners():
    candidate_ids = [c.candidate_id for c in CANDIDATES]
    candidate_texts = [c.text for c in CANDIDATES]
    return [
        TfidfLexicalBaseline(candidate_ids, candidate_texts),
        TransformerMeanPoolEmbeddingCandidate(
            name="multilingual_e5_small",
            model_identifier="intfloat/multilingual-e5-small",
            model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            # intfloat/multilingual-e5-* models are documented as requiring
            # "query: "/"passage: " prefixes for asymmetric retrieval --
            # this is not optional preprocessing, it is the model's own
            # documented usage contract (E5 paper / model card).
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        TransformerMeanPoolEmbeddingCandidate(
            name="paraphrase_multilingual_minilm_l12_v2",
            model_identifier="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            # No documented prefix requirement for this model (symmetric
            # sentence-similarity training) -- both query and passage use
            # the same, empty prefix, per its model card.
            query_prefix="",
            passage_prefix="",
        ),
    ]


def _run_candidate(runner) -> CandidateReport:
    report = CandidateReport(name=runner.name, kind=runner.kind, status=EXECUTED)
    try:
        report.setup_seconds = runner.setup()
    except ImportError as exc:
        report.status = NOT_EXECUTED_DEPENDENCY_MISSING
        report.error_detail = f"{type(exc).__name__}: {exc}"
        return report
    except Exception as exc:  # noqa: BLE001 -- classify below, never silently drop the candidate
        message = str(exc).lower()
        if any(term in message for term in ("connection", "timed out", "resolve", "ssl", "certificate", "network")):
            report.status = NOT_EXECUTED_DOWNLOAD_BLOCKED
        else:
            report.status = FAILED_RUNTIME
        report.error_detail = f"{type(exc).__name__}: {exc}"
        return report

    report.model_identifier = getattr(runner, "model_identifier", None)
    report.model_revision = getattr(runner, "model_revision", None)
    report.embedding_dimension = getattr(runner, "embedding_dimension", None)

    query_seconds_total = 0.0
    try:
        for query in QUERIES:
            start = time.perf_counter()
            ranked_ids, scores = runner.rank(query.query_text)
            query_seconds_total += time.perf_counter() - start
            report.rankings[query.query_id] = ranked_ids
            report.scores[query.query_id] = scores
    except Exception as exc:  # noqa: BLE001 -- a runtime failure mid-run must still be reported, not hidden
        report.status = FAILED_RUNTIME
        report.error_detail = f"{type(exc).__name__}: {exc}"
        return report

    report.total_query_seconds = query_seconds_total
    return report


def _relevant_sets():
    return {q.query_id: frozenset(q.relevant_candidate_ids) for q in QUERIES}


def _print_qualitative_failures(report: CandidateReport, relevant_by_query: dict, top_n: int = 5) -> None:
    if report.status != EXECUTED:
        print(f"  [{report.name}] not executed ({report.status}): {report.error_detail}")
        return
    misses = []
    for query in QUERIES:
        relevant = relevant_by_query[query.query_id]
        ranked = report.rankings.get(query.query_id, [])
        if not relevant:
            continue
        if not any(cid in relevant for cid in ranked[:3]):
            top_hit = ranked[0] if ranked else None
            misses.append((query.query_id, query.query_text, sorted(relevant), top_hit, query.variation_type))
    print(f"  [{report.name}] misses outside top-3: {len(misses)} / {len(QUERIES)} queries")
    for query_id, text, relevant_ids, top_hit, variation in misses[:top_n]:
        print(f"    - {query_id} ({variation}): {text!r} expected {relevant_ids}, top result was {top_hit!r}")


def main() -> None:
    print(f"M5B.2 Arabic retrieval benchmark: {len(CANDIDATES)} candidates, {len(QUERIES)} queries")
    relevant_by_query = _relevant_sets()
    reports: list[CandidateReport] = []

    for runner in _build_candidate_runners():
        print(f"\nRunning candidate: {runner.name} ...")
        report = _run_candidate(runner)
        reports.append(report)
        if report.status != EXECUTED:
            print(f"  status: {report.status} ({report.error_detail})")
            continue

        rankings_in_order = [report.rankings[q.query_id] for q in QUERIES]
        relevant_in_order = [relevant_by_query[q.query_id] for q in QUERIES]
        metrics = compute_aggregate_metrics(rankings_in_order, relevant_in_order)
        print(
            f"  status: EXECUTED | setup: {report.setup_seconds:.2f}s | "
            f"query total: {report.total_query_seconds:.3f}s | "
            f"dim: {report.embedding_dimension}"
        )
        print(
            f"  Recall@1={metrics.recall_at_1:.3f}  Recall@3={metrics.recall_at_3:.3f}  "
            f"Recall@5={metrics.recall_at_5:.3f}  MRR={metrics.mrr:.3f}"
        )

    print("\n--- Qualitative failure analysis (misses outside top-3) ---")
    for report in reports:
        _print_qualitative_failures(report, relevant_by_query)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        "dataset_version": "m5b2-synthetic-v1",
        "num_candidates": len(CANDIDATES),
        "num_queries": len(QUERIES),
        "candidates": [
            {
                "name": r.name,
                "kind": r.kind,
                "status": r.status,
                "model_identifier": r.model_identifier,
                "model_revision": r.model_revision,
                "embedding_dimension": r.embedding_dimension,
                "setup_seconds": r.setup_seconds,
                "total_query_seconds": r.total_query_seconds,
                "error_detail": r.error_detail,
                "metrics": (
                    vars(compute_aggregate_metrics(
                        [r.rankings[q.query_id] for q in QUERIES],
                        [relevant_by_query[q.query_id] for q in QUERIES],
                    ))
                    if r.status == EXECUTED
                    else None
                ),
                "rankings": r.rankings if r.status == EXECUTED else {},
            }
            for r in reports
        ],
        "queries": [
            {"query_id": q.query_id, "query_text": q.query_text, "relevant_candidate_ids": list(q.relevant_candidate_ids)}
            for q in QUERIES
        ],
    }
    REPORT_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
