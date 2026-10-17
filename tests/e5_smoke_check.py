"""
Phase M5B.3a — OPTIONAL, model-backed smoke test for the real E5
embedding adapter. This is the one place a Hugging Face model may
actually be downloaded/loaded, so it must never be part of the default
backend regression baseline.

Isolation (two independent layers):

1. PRIMARY: this file is deliberately named `e5_smoke_check.py` --
   neither `test_*.py` nor `*_test.py`, pytest's default `python_files`
   discovery patterns. A plain `pytest` / `pytest tests/` run never even
   collects this file, so it contributes NOTHING to the default suite's
   pass/skip counts (unlike tests/test_mongo_repository_live.py's
   module-level pytest.mark.skipif approach, which IS collected by
   default and therefore permanently adds one skip -- deliberately not
   reused here, per the M5B.3a instruction that this smoke test must not
   do that).
2. SECONDARY safety net: even when explicitly targeted, this module
   still requires RUN_E5_SMOKE_TEST=1 -- so an accidental
   `pytest tests/e5_smoke_check.py` with no env var set skips instead of
   downloading a model.

Opt-in command (both conditions satisfied):
    RUN_E5_SMOKE_TEST=1 .venv/bin/python -m pytest tests/e5_smoke_check.py -q
"""

from __future__ import annotations

import os

import pytest

from rcs.response_intelligence.retrieval.embeddings import E5EmbeddingAdapter

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E5_SMOKE_TEST") != "1",
    reason="SKIPPED by default: set RUN_E5_SMOKE_TEST=1 to download/load the real "
    "intfloat/multilingual-e5-small model and exercise it end-to-end. Never run "
    "as part of the default backend regression suite.",
)


def test_e5_adapter_loads_and_embeds_real_text():
    adapter = E5EmbeddingAdapter()
    adapter.load()
    assert adapter.is_loaded
    assert adapter.embedding_dimension == 384

    passages = adapter.embed_passages(["شكرًا لتواصلكم معنا.", "سنراجع طلبكم قريبًا."])
    assert passages.shape == (2, 384)

    query_vec = adapter.embed_query("هل تم مراجعة طلبي؟")
    assert query_vec.shape == (384,)

    import numpy as np

    for vec in list(passages) + [query_vec]:
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
