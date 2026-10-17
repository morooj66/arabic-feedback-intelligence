"""Opt-in live Gemini smoke test. Never part of the default suite.

Run with: RCS_LIVE_GEMINI=1 GEMINI_API_KEY=... pytest tests/test_response_generation_live.py
"""

from __future__ import annotations

import os

import pytest

from rcs.response_generation.service import (
    GenerationRequest,
    build_live_gemini_client,
    generate_suggested_response,
)


@pytest.mark.skipif(os.environ.get("RCS_LIVE_GEMINI") != "1", reason="opt-in live Gemini only")
def test_live_thanks_draft_smoke():
    client = build_live_gemini_client()
    result = generate_suggested_response(
        GenerationRequest(
            comment_id="live-smoke",
            comment_text="شكرًا لكم على المعجم",
            primary_intent="THANKS_PRAISE",
            topic_subtype="GENERAL",
            confidence=0.9,
        ),
        client,
        max_retries=0,
    )
    assert result.review_required is True
    if result.generation_status == "SUCCEEDED":
        assert result.suggested_response
        assert result.routing_decision == "GENERATE_DRAFT"
