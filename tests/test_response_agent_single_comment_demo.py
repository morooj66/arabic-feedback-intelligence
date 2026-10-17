"""
Tests for scripts/response_agent_single_comment_demo.py -- the
single-comment live-demonstration wrapper.

No live Gemini call and no real classifier load here: the classifier
and Gemini client are injected exactly the way
tests/test_review_agent_replies.py already does for the multi-comment
CLI (a scripted fake client, a fake classifier) -- this proves the
wrapper's WIRING (it reuses classify_and_route/generate_reply/
SourceRow unmodified, it never calls Gemini for an ineligible comment,
it never prints secrets), not classifier/Gemini behavior itself.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import scripts.review_agent_replies as cli
import scripts.response_agent_single_comment_demo as demo
from rcs.response_intelligence.inspector import ClassificationInspection


def _classification(**overrides) -> ClassificationInspection:
    base = dict(
        primary_intent="THANKS_PRAISE",
        primary_intent_confidence=0.9,
        topic_subtype="GENERAL",
        topic_subtype_confidence=0.9,
        sentiment="POSITIVE",
        sentiment_confidence=0.9,
    )
    base.update(overrides)
    return ClassificationInspection(**base)


class ScriptedGeminiClient:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.call_count += 1
        return self._responses[min(self.call_count - 1, len(self._responses) - 1)]


class NeverCallGeminiClient:
    def generate(self, prompt: str, response_schema: dict) -> str:
        raise AssertionError("Gemini must never be called for an ineligible comment")


def _draft_json(text: str) -> str:
    return json.dumps({"suggested_response": text}, ensure_ascii=False)


_VALID_DRAFT = "شكرًا لملاحظتكم القيّمة حول هذا المدخل."


# --- 1. Uses the real orchestration functions, not a second Gemini path ----


def test_single_comment_uses_classify_and_route_and_generate_reply_unmodified(monkeypatch):
    """Structural + behavioral proof: run_single_comment must call the
    SAME classify_and_route/generate_reply the multi-comment CLI uses --
    spied here to confirm the exact call, not a re-implementation."""
    calls = {"classify_and_route": 0, "generate_reply": 0}
    real_classify = demo.classify_and_route
    real_generate = demo.generate_reply

    def spy_classify(*args, **kwargs):
        calls["classify_and_route"] += 1
        return real_classify(*args, **kwargs)

    def spy_generate(*args, **kwargs):
        calls["generate_reply"] += 1
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(demo, "classify_and_route", lambda text: spy_classify(text, classifier=lambda t: _classification()))
    monkeypatch.setattr(demo, "generate_reply", spy_generate)
    monkeypatch.setattr(demo, "build_live_gemini_client", lambda: ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]))

    routing, exec_result = demo.run_single_comment("شكرًا جزيلاً على هذا العمل الرائع.")

    assert calls["classify_and_route"] == 1
    assert calls["generate_reply"] == 1
    assert exec_result is not None
    assert exec_result.generation_result.suggested_response == _VALID_DRAFT


def test_ineligible_comment_never_calls_gemini_or_builds_a_client(monkeypatch):
    monkeypatch.setattr(demo, "classify_and_route", lambda text: cli.classify_and_route(text, classifier=lambda t: _classification(primary_intent="SPAM")))

    def _explode():
        raise AssertionError("build_live_gemini_client must never be called for an ineligible comment")

    monkeypatch.setattr(demo, "build_live_gemini_client", _explode)
    routing, exec_result = demo.run_single_comment("سبام سبام غير ذي صلة")
    assert exec_result is None
    assert routing.routing_decision != "GENERATE_DRAFT"


def test_wrapper_module_imports_the_cli_functions_not_new_logic():
    """Static guard: the wrapper must import classify_and_route/
    generate_reply/SourceRow from the existing CLI, never define its
    own classification or generation logic."""
    import inspect

    source = inspect.getsource(demo)
    assert "from scripts.review_agent_replies import" in source
    assert "def determine_response_eligibility" not in source
    assert "def run_agent_pipeline" not in source
    # No second/direct Gemini prompt path: the only client construction
    # in this file is the exact same build_live_gemini_client production
    # already uses.
    assert "genai.Client(" not in source
    assert "GenAIClient(" not in source


# --- 2. Output never contains secrets/raw prompts ---------------------------


def test_report_never_prints_secrets_api_keys_or_raw_prompts(monkeypatch):
    monkeypatch.setattr(demo, "classify_and_route", lambda text: cli.classify_and_route(text, classifier=lambda t: _classification()))
    monkeypatch.setattr(demo, "build_live_gemini_client", lambda: ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]))

    routing, exec_result = demo.run_single_comment("شكرًا جزيلاً على هذا العمل الرائع.")
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo._print_report("شكرًا جزيلاً على هذا العمل الرائع.", routing, exec_result)
    printed = buf.getvalue()

    for forbidden in (
        "GEMINI_API_KEY", "api_key", "Bearer ", "postgresql://", "mongodb://",
        "OFFICIAL KNOWLEDGE", "IDENTIFYING CONTEXT", "system_instruction",
    ):
        assert forbidden not in printed
    assert printed.count("Prompt version:") == 1


def test_report_format_matches_the_documented_shape(monkeypatch):
    monkeypatch.setattr(demo, "classify_and_route", lambda text: cli.classify_and_route(text, classifier=lambda t: _classification()))
    monkeypatch.setattr(demo, "build_live_gemini_client", lambda: ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]))
    routing, exec_result = demo.run_single_comment("شكرًا جزيلاً على هذا العمل الرائع.")
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo._print_report("شكرًا جزيلاً على هذا العمل الرائع.", routing, exec_result)
    printed = buf.getvalue()
    for expected in (
        "Response Agent", "Comment:", "Classification:", "Intent:", "Topic:",
        "Sentiment:", "Eligibility:", "Proposed response:", "Trace:", "Prompt version:",
    ):
        assert expected in printed


def test_pii_in_typed_comment_is_redacted_before_printing(monkeypatch):
    monkeypatch.setattr(demo, "classify_and_route", lambda text: cli.classify_and_route(text, classifier=lambda t: _classification(primary_intent="INQUIRY")))
    monkeypatch.setattr(demo, "build_live_gemini_client", lambda: ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]))
    comment = "راسلوني على 0501234567 أو visitor@example.com."
    routing, exec_result = demo.run_single_comment(comment)
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo._print_report(comment, routing, exec_result)
    printed = buf.getvalue()
    assert "0501234567" not in printed
    assert "visitor@example.com" not in printed


# --- 3. CLI wiring ------------------------------------------------------


def test_cli_accepts_comment_argument():
    import inspect

    from pathlib import Path

    sig = inspect.signature(demo.main)
    assert sig.parameters == {}
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "--comment" in source
    assert "required=True" in source


# --- 4. No DB/Mongo write path is reachable ---------------------------------


def test_no_db_repository_or_service_writes_invoked():
    import ast
    from pathlib import Path

    source = Path(demo.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    banned_prefixes = (
        "rcs.review.repositories",
        "rcs.review.repository",
        "rcs.review.service",
        "rcs.review.workflow",
        "rcs.review.excel_review_flow",
        "sqlalchemy",
        "pymongo",
    )
    for module in imported_modules:
        for banned in banned_prefixes:
            assert not module.startswith(banned), f"forbidden import {module!r} found in {demo.__file__}"


def test_single_comment_writes_no_file_at_all(tmp_path, monkeypatch):
    """Unlike the multi-comment CLI, the demo wrapper never persists
    anything -- it only prints. Confirmed by running inside an empty
    tmp cwd and asserting nothing was created."""
    monkeypatch.setattr(demo, "classify_and_route", lambda text: cli.classify_and_route(text, classifier=lambda t: _classification()))
    monkeypatch.setattr(demo, "build_live_gemini_client", lambda: ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]))
    monkeypatch.chdir(tmp_path)
    demo.run_single_comment("شكرًا جزيلاً على هذا العمل الرائع.")
    assert list(tmp_path.rglob("*")) == []
