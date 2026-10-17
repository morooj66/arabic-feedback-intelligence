"""
Phase M5F — reproducibility metadata for one evaluation run.

No secrets: only names/versions/booleans/timestamps, never an API key,
connection string, or credential of any kind.
"""

from __future__ import annotations

import datetime as dt

from config import settings
from scripts.offline_eval.response_agent.dataset import DATASET_VERSION


def build_run_metadata() -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "model_name": settings.RESPONSE_GENERATION_MODEL_NAME,
        "provider": settings.RESPONSE_GENERATION_PROVIDER,
        "prompt_version": settings.RESPONSE_GENERATION_PROMPT_VERSION,
        "retrieval_configuration": {
            "official_knowledge_source_configured": bool(settings.OFFICIAL_KNOWLEDGE_SOURCE_PATH),
            "approved_response_allowlist_size": len(settings.APPROVED_RESPONSE_SOURCE_ALLOWLIST),
            "official_knowledge_max_results_per_query": settings.OFFICIAL_KNOWLEDGE_MAX_RESULTS_PER_QUERY,
            "approved_response_memory_max_results_per_query": settings.APPROVED_RESPONSE_MEMORY_MAX_RESULTS_PER_QUERY,
        },
        "seed": None,
        "determinism_note": (
            "This offline run uses fully deterministic, scripted fake Gemini clients and "
            "fake retrievers -- results are exactly reproducible run-to-run. A live-Gemini "
            "evaluation (see run_live_optional.py, opt-in only) would NOT share this "
            "guarantee: real Gemini generation is not guaranteed deterministic even with "
            "fixed inputs, and that limitation is not resolved by this harness."
        ),
    }
