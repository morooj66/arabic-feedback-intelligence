# M5B.2 Arabic retrieval benchmark

Experimental evaluation only — not production code, not wired into
`rcs.response_intelligence.retrieval` or any served path.

## Run

```bash
.venv/bin/python scripts/benchmarks/retrieval/run_benchmark.py
```

## Dependencies

**No new dependency was added to `requirements.txt`.** Embedding
candidates use the already-installed `transformers`/`torch` (both
already production dependencies, used for the frozen classifiers) via
raw `AutoTokenizer`/`AutoModel` + manual mean-pooling. The
`sentence-transformers` package is deliberately not installed or
imported anywhere in this benchmark, specifically so it can never
create a dependency-ownership question for the production manifest.
The TF-IDF lexical baseline uses `scikit-learn`, also already a
production dependency.

## Reproducibility

- Dataset: `dataset.py`, version tag `m5b2-synthetic-v1` (hand-authored, see module docstring).
- Candidate models are pinned to an exact commit SHA (see `run_benchmark.py`'s `_build_candidate_runners`), not a moving branch:
  - `intfloat/multilingual-e5-small` @ `614241f622f53c4eeff9890bdc4f31cfecc418b3`
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` @ `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
- Metrics: `metrics.py` (Recall@1/3/5, MRR — definitions documented in that module's own docstring).
- No randomness is used anywhere in this benchmark (deterministic dataset, deterministic ranking by descending score).
- Model weights download to the standard Hugging Face cache (`~/.cache/huggingface/hub`), never into this repository.

## Output

`results/m5b2_report.json` — aggregate metrics, per-query rankings (candidate IDs only, no vectors), status, and latency per candidate. This file is small and safe to keep in version control; it contains no embeddings and no production data.
