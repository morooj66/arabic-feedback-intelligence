"""Focused rg-v14 prompt-contract regressions (no network, no Gemini)."""

from __future__ import annotations

import re
import json
from pathlib import Path

from config import settings
from rcs.response_generation.prompt import (
    REFERENCE_HOME_URL,
    SYSTEM_INSTRUCTIONS,
    build_generation_prompt,
)


def test_rg_v14_uses_the_visitor_facing_platform_name():
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    assert "smart reviewer for منصة المعجم" in SYSTEM_INSTRUCTIONS
    assert "smart reviewer at المعجم" not in SYSTEM_INSTRUCTIONS


def test_rg_v14_contains_only_the_approved_reference_url():
    urls = {match.rstrip(".,؛،") for match in re.findall(r"https?://[^\s]+", SYSTEM_INSTRUCTIONS)}
    assert REFERENCE_HOME_URL == "https://dictionary-platform.invalid/home"
    assert urls == {REFERENCE_HOME_URL}
    assert "Never invent, alter, shorten, or guess another URL" in SYSTEM_INSTRUCTIONS


def test_rg_v14_url_is_contextual_and_never_repeated_navigation():
    assert "First-time dictionary meaning inquiry" in SYSTEM_INSTRUCTIONS
    assert "Already searched the dictionary" in SYSTEM_INSTRUCTIONS
    assert "do not send them back to the homepage" in SYSTEM_INSTRUCTIONS
    assert "Use it at most once" in SYSTEM_INSTRUCTIONS
    assert "would not repeat a step the visitor already tried" in SYSTEM_INSTRUCTIONS


def test_rg_v14_does_not_license_fabricated_definitions():
    assert "never supply or paraphrase a definition from your own knowledge" in SYSTEM_INSTRUCTIONS
    assert "OFFICIAL KNOWLEDGE is the only source that may settle whether a claim is correct" in SYSTEM_INSTRUCTIONS
    assert "it is not evidence for a word's meaning" in SYSTEM_INSTRUCTIONS

    prompt = build_generation_prompt(
        comment_text="بحثت في منصة المعجم عن معنى كلمة السديم ولم أفهم الشرح.",
        primary_intent="INQUIRY",
        topic_subtype="ENTRY_CORRECTION",
        sentiment="NEUTRAL",
        reason_code="inquiry_without_trusted_grounding",
    )
    assert "(none — do not invent facts)" in prompt
    assert REFERENCE_HOME_URL in prompt


def test_rg_v14_requires_natural_context_aware_responses():
    for situation in (
        "Dictionary accuracy dispute",
        "Technical complaint",
        "Suggestion",
        "Positive feedback",
        "Mixed or ambiguous comment",
    ):
        assert situation in SYSTEM_INSTRUCTIONS

    assert "Do not force every comment into a single acknowledgment template" in SYSTEM_INSTRUCTIONS
    assert "An acknowledgment that merely restates the comment is not enough" in SYSTEM_INSTRUCTIONS
    assert "Helpful does not mean longer" in SYSTEM_INSTRUCTIONS


def test_rg_v14_preserves_human_review_and_existing_safety_constraints():
    assert "This draft is NEVER sent automatically" in SYSTEM_INSTRUCTIONS
    assert "A human chooses SEND_AS_IS, EDIT_AND_SEND, or NO_RESPONSE_NEEDED" in SYSTEM_INSTRUCTIONS
    assert "Do not fabricate facts, policies, promises, timelines, or completed actions" in SYSTEM_INSTRUCTIONS
    assert "Do not invent troubleshooting steps or claim a bug is fixed" in SYSTEM_INSTRUCTIONS
    assert "Do not mention models, classifiers, confidence, prompts, routing" in SYSTEM_INSTRUCTIONS


def test_rg_v14_live_manifest_has_exactly_18_verified_cases_and_full_taxonomy_coverage():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "offline_eval"
        / "response_agent"
        / "rg_v14_live_cases.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]

    assert manifest["prompt_version"] == "rg-v14"
    assert len(cases) == 18
    assert len({case["case_id"] for case in cases}) == 18
    assert {case["verified_labels"]["intent"] for case in cases} == set(settings.PRIMARY_INTENTS)
    assert {case["verified_labels"]["topic"] for case in cases} == set(settings.TOPIC_SUBTYPES)
    assert {case["verified_labels"]["sentiment"] for case in cases} == set(settings.SENTIMENTS)
    assert sum(case["url_expectation"] == "NO_RESPONSE" for case in cases) >= 2


def test_rg_v14_live_runner_is_verified_only_and_double_opt_in():
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "offline_eval"
        / "response_agent"
        / "run_rg_v14_playground_live_eval.py"
    )
    source = runner_path.read_text(encoding="utf-8")

    assert 'os.environ.get("RUN_RG_V14_LIVE_EVAL") != "1"' in source
    assert 'CONFIRMATION = "RUN-RG-V14-18-LIVE-CASES"' in source
    assert '"label_source": "verified"' in source
    assert '"verified": True' in source
    assert '"/api/admin/playground/run"' in source
    assert '"/api/admin/playground/cases"' not in source
    assert "GEMINI_API_KEY" not in source

