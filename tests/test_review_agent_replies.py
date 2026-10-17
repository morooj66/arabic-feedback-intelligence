"""
Tests for scripts/review_agent_replies.py — the developer CLI for human
review of real Response Agent output.

No Hugging Face/frozen-model download or load happens here: the
classifier is injected as a fake (`classify_and_route`'s `classifier`
parameter). The Gemini client is a scripted fake satisfying
SupportsGenerate — using a fake client in a UNIT TEST does not
contradict "use the real Gemini client" (a runtime requirement for
actually running the CLI); every other orchestrator test in this
codebase follows the same convention.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.review_agent_replies as cli
from rcs.response_intelligence.inspector import ClassificationInspection
from rcs.response_intelligence.orchestrator import run_agent_pipeline

_SYNTHETIC_HEADERS = [
    "العنوان",
    "معرف المدخل",
    "اللما",
    "المحتوى",
    "اسم الزائر",
    " البريد الالكتروني للزائر",
    "النوع",
    "التاريخ",
    "المنصة",
]


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def _write_fixture_workbook(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=_SYNTHETIC_HEADERS)
    df.to_excel(path, index=False)


def _row(**overrides) -> dict:
    base = {
        "العنوان": "كلمة تجريبية",
        "معرف المدخل": "E-1",
        "اللما": "تجربة",
        "المحتوى": "شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        "اسم الزائر": "زائر تجريبي",
        " البريد الالكتروني للزائر": "synthetic.visitor@example.test",
        "النوع": "ثناء",
        "التاريخ": dt.datetime(2026, 1, 1),
        "المنصة": "test",
    }
    base.update(overrides)
    return base


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


def _scripted_prompt(*answers: str):
    it = iter(answers)

    def prompt_fn(_message: str) -> str:
        return next(it)

    return prompt_fn


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. Source Excel is never modified --------------------------------


def test_source_excel_never_modified(tmp_path):
    workbook = tmp_path / "visitors.xlsx"
    _write_fixture_workbook(workbook, [_row()])
    before_bytes = workbook.read_bytes()
    before_mtime = workbook.stat().st_mtime

    out_path = tmp_path / "out" / "session.jsonl"
    cli.run_cli(
        workbook,
        limit=1,
        start_row=0,
        client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
        out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("A"),
    )

    assert workbook.read_bytes() == before_bytes
    assert workbook.stat().st_mtime == before_mtime


def test_script_never_calls_excel_writers():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "to_excel(" not in source
    assert ".save(" not in source


# --- 2. No DB repository/service writes are ever reachable -------------


def test_no_db_repository_or_service_writes_invoked():
    """Structural guarantee: the CLI's own import graph contains no name
    from any module capable of writing to PostgreSQL/Supabase/MongoDB or
    mutating review_items/assignment."""
    source = Path(cli.__file__).read_text(encoding="utf-8")
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
            assert not module.startswith(banned), f"forbidden import {module!r} found in {cli.__file__}"


def test_no_seed_or_assignment_functions_referenced():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    for banned in ("seed_demo_db", "assign_batch", "upsert_review_items", "apply_generation_result", "apply_classification_results"):
        assert banned not in source


# --- 3. Ineligible comments never invoke Gemini -------------------------


def test_ineligible_comment_does_not_invoke_gemini(tmp_path):
    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-1",
        source_row_index=0,
        original_comment="محتوى غير ذي صلة تمامًا سبام سبام",
        original_title=None,
        entry_id=None,
    )
    decision = cli.process_row(
        row,
        index=1,
        total=1,
        client=NeverCallGeminiClient(),  # raises if .generate() is ever called
        out_path=out_path,
        classifier=lambda text: _classification(primary_intent="SPAM"),
        prompt_fn=_scripted_prompt("S"),
    )
    assert decision == "S"
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row_json = json.loads(lines[0])
    assert row_json["routing_decision"] == "NO_RESPONSE_RECOMMENDED"
    assert row_json["generated_reply"] is None
    assert row_json["generation_status"] is None


# --- 4. Eligible comments use the real, unmodified orchestrator --------


def test_eligible_comment_uses_the_real_orchestrator(tmp_path, monkeypatch):
    calls = []
    real_run_agent_pipeline = cli.run_agent_pipeline

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_run_agent_pipeline(*args, **kwargs)

    monkeypatch.setattr(cli, "run_agent_pipeline", spy)

    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-2",
        source_row_index=1,
        original_comment="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        original_title="كلمة",
        entry_id="E-1",
    )
    decision = cli.process_row(
        row,
        index=1,
        total=1,
        client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
        out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("A"),
    )
    assert decision == "A"
    assert len(calls) == 1
    # Proves the REAL orchestrator ran (not a stub): the request it
    # received carries the CLI's namespaced comment_id and structured
    # identifying_metadata built from original_title/entry_id (Phase
    # M6.2: the orchestrator's own Context/Trust Policies step decides
    # whether this is actually used).
    (request,), kwargs = calls[0]
    assert request.comment_id == "cli-eval-row1"
    assert request.identifying_metadata is not None
    assert request.identifying_metadata.original_title == "كلمة"
    assert request.identifying_metadata.entry_id == "E-1"
    assert kwargs.get("client") is not None
    assert "audit_repository" not in kwargs  # never opts into audit persistence

    row_json = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row_json["generation_status"] == "SUCCEEDED"
    assert row_json["generated_reply"] == _VALID_DRAFT


def test_real_orchestrator_validation_gates_actually_run(tmp_path):
    """Not a stub: a draft that fails the REAL boilerplate gate must be
    rejected exactly as it would be in production."""
    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-3", source_row_index=2,
        original_comment="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        original_title=None, entry_id=None,
    )
    canned = "أهلًا بك، نشكرك على تواصلك واهتمامك بالمعجم."
    cli.process_row(
        row, index=1, total=1,
        client=ScriptedGeminiClient([_draft_json(canned)]),
        out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("R"),
    )
    row_json = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row_json["generation_status"] == "FAILED_REVIEW_ONLY"
    assert row_json["validation_reason_code"] == "generic_boilerplate"


# --- 5. A/R/S decisions only append to the local JSONL -------------------


def test_a_e_r_s_decisions_each_append_exactly_one_line(tmp_path):
    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-4", source_row_index=3,
        original_comment="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        original_title=None, entry_id=None,
    )
    # "E" needs a second scripted answer (the edited-response text).
    for prompt_fn in (
        _scripted_prompt("A"),
        _scripted_prompt("E", "نص الرد بعد التعديل البشري."),
        _scripted_prompt("R"),
        _scripted_prompt("S"),
    ):
        cli.process_row(
            row, index=1, total=1,
            client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
            out_path=out_path,
            classifier=lambda text: _classification(),
            prompt_fn=prompt_fn,
        )
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    judgments = [json.loads(line)["human_judgment"] for line in lines]
    assert judgments == ["APPROVE_AS_IS", "EDIT", "REJECT", "SKIP"]


def test_edit_preserves_original_agent_response_alongside_the_human_edit(tmp_path):
    """The exact requirement: EDIT must never overwrite the original
    Agent output -- both must be present, distinct, in the persisted
    row."""
    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-edit", source_row_index=9,
        original_comment="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        original_title=None, entry_id=None,
    )
    edited_text = "نسخة معدَّلة من الرد كتبها المراجع البشري."
    cli.process_row(
        row, index=1, total=1,
        client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
        out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("E", edited_text),
    )
    row_json = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row_json["human_judgment"] == "EDIT"
    assert row_json["generated_reply"] == _VALID_DRAFT  # original Agent output, untouched
    assert row_json["edited_response"] == edited_text
    assert row_json["generated_reply"] != row_json["edited_response"]


def test_no_other_file_created_besides_the_jsonl(tmp_path):
    out_path = tmp_path / "eval" / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-5", source_row_index=4,
        original_comment="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        original_title=None, entry_id=None,
    )
    cli.process_row(
        row, index=1, total=1,
        client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
        out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("A"),
    )
    created = list(tmp_path.rglob("*"))
    files = [p for p in created if p.is_file()]
    assert files == [out_path]


# --- 6. No PII in the evaluation output ----------------------------------


def test_pii_not_stored_in_output(tmp_path):
    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-6", source_row_index=5,
        original_comment="تواصلوا معي على 0501234567 أو راسلوني على visitor@example.com من فضلكم.",
        original_title=None, entry_id=None,
    )
    cli.process_row(
        row, index=1, total=1,
        client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
        out_path=out_path,
        classifier=lambda text: _classification(primary_intent="INQUIRY"),
        prompt_fn=_scripted_prompt("A"),
    )
    raw_line = out_path.read_text(encoding="utf-8").splitlines()[0]
    assert "0501234567" not in raw_line
    assert "visitor@example.com" not in raw_line
    row_json = json.loads(raw_line)
    assert "visitor_name" not in row_json
    assert "visitor_email" not in row_json
    assert set(row_json.keys()) == {
        "comment_id", "source_row_index", "sanitized_comment", "intent", "topic",
        "sentiment", "routing_decision", "routing_reason_code", "generated_reply",
        "generation_status", "validation_reason_code", "grounding_mode",
        "official_knowledge_used", "approved_response_memory_used",
        "refinement_attempted", "generation_provider", "generation_model_name",
        "generation_prompt_version", "generated_at",
        "human_judgment", "edited_response", "reviewed_at",
    }


def test_loader_drops_visitor_columns_before_any_processing(tmp_path):
    workbook = tmp_path / "visitors.xlsx"
    _write_fixture_workbook(workbook, [_row(**{"اسم الزائر": "اسم حقيقي حساس", " البريد الالكتروني للزائر": "real.person@example.test"})])
    rows = cli._load_rows(workbook, start_row=0, limit=None)
    assert len(rows) == 1
    for field_name in dataclasses_fields_of(rows[0]):
        assert "real.person" not in str(getattr(rows[0], field_name))
        assert "اسم حقيقي حساس" not in str(getattr(rows[0], field_name))


def dataclasses_fields_of(instance):
    import dataclasses

    return [f.name for f in dataclasses.fields(instance)]


# --- 7. Q exits cleanly without appending ---------------------------------


def test_quit_exits_cleanly_without_appending(tmp_path):
    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-7", source_row_index=6,
        original_comment="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        original_title=None, entry_id=None,
    )
    decision = cli.process_row(
        row, index=1, total=1,
        client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
        out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("Q"),
    )
    assert decision == "Q"
    assert not out_path.exists()


def test_quit_stops_the_run_loop_immediately(tmp_path):
    workbook = tmp_path / "visitors.xlsx"
    _write_fixture_workbook(workbook, [_row(), _row(**{"معرف المدخل": "E-2"})])
    out_path = tmp_path / "session.jsonl"
    client = ScriptedGeminiClient([_draft_json(_VALID_DRAFT)])
    cli.run_cli(
        workbook, limit=None, start_row=0,
        client=client, out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("Q"),  # quits on the FIRST comment
    )
    assert not out_path.exists()
    # Q is asked AFTER the row is shown (the reviewer needs to see the
    # reply before deciding to quit) -- exactly one generation call for
    # row 1, and the loop must stop there, never reaching row 2.
    assert client.call_count == 1


# --- CLI argument wiring ---------------------------------------------------


def test_cli_accepts_file_limit_start_row_arguments():
    import inspect

    sig = inspect.signature(cli.main)
    assert sig.parameters == {}  # argparse handled inside main(); verify parser wiring instead
    parser_source = Path(cli.__file__).read_text(encoding="utf-8")
    for flag in ("--file", "--limit", "--start-row", "--rows", "--out", "--auto"):
        assert flag in parser_source


# --- 8. Provenance: version/model come from live config, never hardcoded ---


def test_generation_provenance_is_read_from_live_config_not_hardcoded(tmp_path, monkeypatch):
    """The exact requirement: generation_prompt_version/generation_model_name
    must reflect whatever config.settings currently reports -- proven
    here by changing that config value and confirming the persisted row
    changes with it, rather than a string baked into this script."""
    from config import settings as cfg

    monkeypatch.setattr(cfg, "RESPONSE_GENERATION_PROMPT_VERSION", "rg-vTEST-99")
    out_path = tmp_path / "session.jsonl"
    row = cli.SourceRow(
        comment_id="synthetic-prov", source_row_index=8,
        original_comment="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        original_title=None, entry_id=None,
    )
    cli.process_row(
        row, index=1, total=1,
        client=ScriptedGeminiClient([_draft_json(_VALID_DRAFT)]),
        out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("A"),
    )
    row_json = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row_json["generation_prompt_version"] == "rg-vTEST-99"
    assert row_json["generation_model_name"] == cfg.RESPONSE_GENERATION_MODEL_NAME
    assert row_json["generation_provider"] == cfg.RESPONSE_GENERATION_PROVIDER
    assert row_json["generated_at"] is not None


def test_script_never_hardcodes_a_prompt_version_string():
    """Static guard: this file must read the prompt version off
    GenerationResult.prompt_version (itself sourced from live config in
    service.py) -- it must never assign a literal "rg-v..." string to
    generation_prompt_version itself."""
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert 'generation_prompt_version"] = "rg-' not in source
    assert "generation_prompt_version = \"rg-" not in source


# --- 9. Explicit --rows selection ------------------------------------------


def test_explicit_rows_argument_selects_exact_non_contiguous_positions(tmp_path):
    workbook = tmp_path / "visitors.xlsx"
    _write_fixture_workbook(
        workbook,
        [_row(**{"معرف المدخل": f"E-{i}", "المحتوى": f"تعليق رقم {i}"}) for i in range(10)],
    )
    out_path = tmp_path / "session.jsonl"
    client = ScriptedGeminiClient([_draft_json(_VALID_DRAFT)] * 10)
    cli.run_cli(
        workbook, limit=None, start_row=0, rows=[2, 7, 9],
        client=client, out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_scripted_prompt("S", "S", "S"),
    )
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source_row_index"] for row in lines] == [2, 7, 9]
    assert [row["sanitized_comment"] for row in lines] == ["تعليق رقم 2", "تعليق رقم 7", "تعليق رقم 9"]


# --- 10. Non-interactive (--auto) mode: no prompt, blank human_judgment ----


def test_auto_mode_never_calls_prompt_fn_and_leaves_judgment_blank(tmp_path):
    workbook = tmp_path / "visitors.xlsx"
    _write_fixture_workbook(workbook, [_row(), _row(**{"معرف المدخل": "E-2"})])
    out_path = tmp_path / "session.jsonl"
    client = ScriptedGeminiClient([_draft_json(_VALID_DRAFT)] * 2)

    def _explode(_message: str) -> str:
        raise AssertionError("prompt_fn must never be called in non-interactive (--auto) mode")

    cli.run_cli(
        workbook, limit=None, start_row=0,
        client=client, out_path=out_path,
        classifier=lambda text: _classification(),
        prompt_fn=_explode,
        interactive=False,
    )
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert all(row["human_judgment"] is None for row in lines)
    assert all(row["edited_response"] is None for row in lines)
    # The real Agent still ran for both real comments -- non-interactive
    # means "no decision prompt", never "no generation".
    assert all(row["generated_reply"] == _VALID_DRAFT for row in lines)


# --- 11. New artifacts never collide with / overwrite historical ones ------


def test_default_output_path_is_dated_versioned_and_distinct_from_historical_session_files():
    """Historical artifacts (session_<timestamp>.jsonl, from before this
    provenance-hardening change) must never be touched by a new run's
    default path -- proven structurally by the naming pattern itself,
    plus a check that the version segment is read from live config."""
    path = cli._default_output_path()
    assert path.parent == cli.OUTPUT_DIR
    assert not path.name.startswith("session_")  # the old, pre-hardening pattern
    assert "human_verification" in path.name
    from config import settings as cfg

    assert cfg.RESPONSE_GENERATION_PROMPT_VERSION.replace("-", "_") in path.name


def test_append_never_truncates_an_existing_artifact(tmp_path):
    """append_evaluation_row must only ever add a line -- proven by
    writing one row, then a second, and confirming the first line is
    still there unchanged (guards the "never overwrite historical
    files" requirement at the file-write level, not just the naming
    level)."""
    out_path = tmp_path / "existing_history.jsonl"
    cli.append_evaluation_row(out_path, {"marker": "historical-row-must-survive"})
    before = out_path.read_text(encoding="utf-8")
    cli.append_evaluation_row(out_path, {"marker": "new-row"})
    after = out_path.read_text(encoding="utf-8")
    assert after.startswith(before)
    assert after.count("\n") == 2
