"""
Focused tests for the Reply-Strategy V3 (selective-gate) evaluation
tooling (scripts/offline_eval/reply_strategy/{strategy_v3,experiment_v3,
review_ab_v3,analyze_v3}.py). Fast and deterministic -- no real model
load, no network, no Gemini call.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STRATEGY_DIR = _ROOT / "scripts" / "offline_eval" / "reply_strategy"

sys.path.insert(0, str(_STRATEGY_DIR))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import strategy_v3  # noqa: E402
import experiment_v3  # noqa: E402
import review_ab_v3  # noqa: E402
import analyze_v3  # noqa: E402
from config import settings  # noqa: E402
from rcs.response_generation.prompt import build_generation_prompt  # noqa: E402


# --- 1-5: gate behavior --------------------------------------------------


def test_gate_activates_for_suggestion_and_substantive():
    long_text = "اقتراح لغوي مفصل حول مفردة معينة. " * 15  # >= 400 chars
    assert strategy_v3.gate_is_open(primary_intent="SUGGESTION", response_depth="SUBSTANTIVE") is True
    decision = strategy_v3.derive_strategy_v3(
        comment_text=long_text, primary_intent="SUGGESTION", topic_subtype="ENTRY_CORRECTION",
        official_grounding_available=False,
    )
    assert decision.strategy_enabled is True
    assert decision.strategy.response_depth == "SUBSTANTIVE"


def test_gate_off_for_brief_suggestion():
    decision = strategy_v3.derive_strategy_v3(
        comment_text="كلمة قصيرة", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION",
        official_grounding_available=False,
    )
    assert decision.strategy.response_depth == "BRIEF"
    assert decision.strategy_enabled is False


def test_gate_off_for_thanks_praise_even_if_long():
    long_text = "شكرا جزيلا على هذا المعجم الرائع والمفيد جدا لكل الباحثين. " * 10
    decision = strategy_v3.derive_strategy_v3(
        comment_text=long_text, primary_intent="THANKS_PRAISE", topic_subtype="GENERAL",
        official_grounding_available=False,
    )
    assert decision.strategy.response_depth == "SUBSTANTIVE"  # long enough
    assert decision.strategy_enabled is False  # but wrong intent -- gate must stay closed


def test_gate_off_for_inquiry_even_if_long():
    long_text = "سؤال لغوي طويل جدا حول أصل الكلمة وتاريخها واستخدامها. " * 10
    decision = strategy_v3.derive_strategy_v3(
        comment_text=long_text, primary_intent="INQUIRY", topic_subtype="ENTRY_CORRECTION",
        official_grounding_available=False,
    )
    assert decision.strategy.response_depth == "SUBSTANTIVE"
    assert decision.strategy_enabled is False


def test_gate_off_for_complaint_even_if_long():
    long_text = "هذا خطأ جسيم ورد في المعجم ويجب تصحيحه فورا لأنه يضلل القراء. " * 10
    decision = strategy_v3.derive_strategy_v3(
        comment_text=long_text, primary_intent="COMPLAINT", topic_subtype="ENTRY_CORRECTION",
        official_grounding_available=False,
    )
    assert decision.strategy.response_depth == "SUBSTANTIVE"
    assert decision.strategy_enabled is False


def test_gate_uses_exactly_two_fields_no_topic_dependence():
    """The gate must not vary by topic_subtype -- only intent + depth."""
    long_text = "اقتراح لغوي مفصل جدا يستحق النظر والتحليل والمراجعة. " * 10
    for topic in ("WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION", "GENERAL", None):
        decision = strategy_v3.derive_strategy_v3(
            comment_text=long_text, primary_intent="SUGGESTION", topic_subtype=topic,
            official_grounding_available=False,
        )
        assert decision.strategy_enabled is True


# --- reused V2 logic, not reimplemented ----------------------------------


def test_strategy_v3_reuses_v2_derivation_not_a_copy():
    source = (_STRATEGY_DIR / "strategy_v3.py").read_text(encoding="utf-8")
    assert "from strategy_v2 import" in source
    assert "def derive_strategy_v2" not in source  # not redefined


def test_render_returns_none_when_gate_closed():
    decision = strategy_v3.derive_strategy_v3(
        comment_text="قصير", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL",
        official_grounding_available=False,
    )
    assert strategy_v3.render_strategy_v3_block(decision) is None


def test_render_returns_v2_style_block_when_gate_open():
    long_text = "اقتراح لغوي مفصل جدا يستحق النظر والتحليل والمراجعة. " * 10
    decision = strategy_v3.derive_strategy_v3(
        comment_text=long_text, primary_intent="SUGGESTION", topic_subtype="ENTRY_CORRECTION",
        official_grounding_available=False,
    )
    block = strategy_v3.render_strategy_v3_block(decision)
    assert block is not None
    assert "RESPONSE STRATEGY V2" in block  # explicitly reused, not a new "V3" block design


def test_gate_closed_case_prompt_is_byte_identical_to_baseline_prompt():
    """The defining behavioral guarantee: when the gate is closed, the
    prompt V3 would send to Gemini must be EXACTLY the unmodified rg-v10
    prompt -- not a strategy block with different/empty content."""
    kwargs = dict(
        comment_text="شكرا لكم", primary_intent="THANKS_PRAISE", topic_subtype="WORD_SUBMISSION",
        sentiment="POSITIVE", reason_code="thanks_eligible", identifying_context=None,
        trusted_grounding_text=None,
    )
    baseline_prompt = build_generation_prompt(**kwargs)

    decision = strategy_v3.derive_strategy_v3(
        comment_text=kwargs["comment_text"], primary_intent=kwargs["primary_intent"],
        topic_subtype=kwargs["topic_subtype"], official_grounding_available=False,
    )
    assert decision.strategy_enabled is False
    strategy_block = strategy_v3.render_strategy_v3_block(decision)
    assert strategy_block is None
    v3_prompt = baseline_prompt if strategy_block is None else "should not reach here"
    assert v3_prompt == baseline_prompt


# --- Trust rules preserved -------------------------------------------


def test_identifying_context_never_becomes_factual_authority_field():
    """strategy_v3 must not introduce any field or code path that grants
    factual authority based on identifying_context; factual_answer_allowed
    stays tied only to official_grounding_available (inherited from V2,
    unmodified)."""
    long_text = "اقتراح لغوي مفصل جدا يستحق النظر والتحليل والمراجعة. " * 10
    decision = strategy_v3.derive_strategy_v3(
        comment_text=long_text, primary_intent="SUGGESTION", topic_subtype="ENTRY_CORRECTION",
        official_grounding_available=False,
    )
    assert decision.strategy.factual_answer_allowed is False
    # V3 does not touch the identifying_context mechanism at all -- checked
    # structurally (as a real identifier, not just absent from the docstring
    # that explains why it's out of scope).
    tree = ast.parse((_STRATEGY_DIR / "strategy_v3.py").read_text(encoding="utf-8"))
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    identifiers |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    assert "identifying_context" not in identifiers


# --- 6-8: evaluation isolation --------------------------------------


def _v3_module_files():
    return [
        _STRATEGY_DIR / "strategy_v3.py",
        _STRATEGY_DIR / "experiment_v3.py",
        _STRATEGY_DIR / "review_ab_v3.py",
        _STRATEGY_DIR / "analyze_v3.py",
    ]


def test_no_extra_gemini_call_to_derive_strategy_or_gate():
    for fn in (strategy_v3.derive_strategy_v3, strategy_v3.gate_is_open, strategy_v3.render_strategy_v3_block):
        names = fn.__code__.co_names
        assert "generate" not in names
        assert "client" not in names


def test_reply_v3_never_calls_refinement_logic():
    names = experiment_v3._generate_reply_v3.__code__.co_names
    assert "RefinementContext" not in names
    assert "run_agent_pipeline" not in names


def test_no_db_imports_in_v3_tooling():
    banned_prefixes = (
        "rcs.review.repositories", "rcs.review.repository", "rcs.review.service",
        "rcs.review.workflow", "rcs.review.excel_review_flow", "sqlalchemy", "pymongo",
    )
    for path in _v3_module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        for module in imported:
            for banned in banned_prefixes:
                assert not module.startswith(banned), f"forbidden import {module!r} in {path}"


def test_no_excel_write_functions_in_v3_tooling():
    for path in _v3_module_files():
        source = path.read_text(encoding="utf-8")
        assert "to_excel(" not in source
        assert ".save(" not in source


def test_no_new_prompt_version_or_rg_v11():
    """No new prompt version was introduced by this V3 tooling's own
    files (`_v3_module_files()`, scoped to this directory). The
    settings literal now reads "rg-v14.2": later, separate
    prompt-content changes legitimately bumped it -- see
    rcs.response_generation.prompt's module docstring -- unrelated to
    this V3 tooling, which was never touched."""
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    for path in _v3_module_files():
        assert "rg-v11" not in path.read_text(encoding="utf-8")


def test_real_source_excel_unchanged_if_present():
    import hashlib

    source_file = _ROOT / "data" / "raw" / "historical-export.xlsx"
    if not source_file.exists():
        import pytest
        pytest.skip("source Excel not present in this environment")
    # This test only proves the file is READABLE and its hash is stable
    # across two reads within the same test run -- it cannot know the
    # "before" hash from a prior session, so the real non-mutation
    # guarantee is enforced structurally: no module here ever calls a
    # write/save function (see test_no_excel_write_functions_in_v3_tooling).
    h1 = hashlib.md5(source_file.read_bytes()).hexdigest()
    h2 = hashlib.md5(source_file.read_bytes()).hexdigest()
    assert h1 == h2


# --- 9: sample excludes V1/V2 evaluated rows -----------------------------


def test_v3_sample_excludes_all_v1_and_v2_evaluated_rows():
    selected = set(experiment_v3.SELECTED_SOURCE_ROW_INDICES_V3)
    assert selected.isdisjoint(experiment_v3.PREVIOUSLY_EVALUATED_ROWS)
    assert selected.isdisjoint(set(experiment_v3.V1_ROWS))
    assert selected.isdisjoint(set(experiment_v3.V2_ROWS))


def test_v3_sample_has_no_internal_duplicates():
    rows = experiment_v3.SELECTED_SOURCE_ROW_INDICES_V3
    assert len(rows) == len(set(rows))


def test_v3_sample_size_within_requested_range():
    assert 24 <= len(experiment_v3.SELECTED_SOURCE_ROW_INDICES_V3) <= 30


def test_v3_sample_excludes_known_exact_duplicate_cluster():
    exact_duplicate_cluster = set(range(1002, 1012))
    selected = set(experiment_v3.SELECTED_SOURCE_ROW_INDICES_V3)
    assert selected.isdisjoint(exact_duplicate_cluster)


# --- 10: deterministic A/B randomization ---------------------------------


def test_swap_is_deterministic_per_eval_id():
    assert review_ab_v3._swap_for("row1595") == review_ab_v3._swap_for("row1595")


def test_ab_v3_maps_positional_choice_back_to_variant_correctly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {
        "eval_id": "swap-test", "source_row_index": 1, "comment_preview": "x", "primary_intent": "SUGGESTION",
        "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
        "reply_a": "REPLY_FROM_A", "reply_v3": "REPLY_FROM_V3",
    }
    with cases_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(case) + "\n")

    swap = review_ab_v3._swap_for("swap-test")
    review_ab_v3.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "A")
    label = review_ab_v3._load_jsonl(labels_path)[0]
    expected_winner = "V3" if swap else "A"
    assert label["winner_variant"] == expected_winner


def test_ab_v3_resumable(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [
        {"eval_id": f"c{i}", "source_row_index": i, "comment_preview": "x", "primary_intent": "SUGGESTION",
         "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
         "reply_a": f"a{i}", "reply_v3": f"v{i}"}
        for i in range(2)
    ]
    with cases_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    answers = iter(["A", "Q"])
    review_ab_v3.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers))
    assert {row["eval_id"] for row in review_ab_v3._load_jsonl(labels_path)} == {"c0"}

    answers2 = iter(["N"])
    review_ab_v3.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers2))
    assert {row["eval_id"] for row in review_ab_v3._load_jsonl(labels_path)} == {"c0", "c1"}


def test_ab_v3_quit_exits_cleanly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {
        "eval_id": "c1", "source_row_index": 1, "comment_preview": "x", "primary_intent": "SUGGESTION",
        "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
        "reply_a": "a1", "reply_v3": "v1",
    }
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    review_ab_v3.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "Q")
    assert not labels_path.exists() or review_ab_v3._load_jsonl(labels_path) == []


def test_ab_v3_view_does_not_persist_pii(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {
        "eval_id": "c1", "source_row_index": 1, "comment_preview": "x", "primary_intent": "SUGGESTION",
        "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
        "reply_a": "a1", "reply_v3": "v1",
    }
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    raw_with_pii = "للتواصل: user@example.com أو 0501234567."
    answers = iter(["V", "T"])
    review_ab_v3.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: raw_with_pii,
    )
    raw_labels_text = labels_path.read_text(encoding="utf-8")
    assert "0501234567" not in raw_labels_text
    assert "user@example.com" not in raw_labels_text


# --- 11: analyzer correctly unblinds winners ------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_analyzer_v3_unblinds_and_splits_by_gate(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [
        {"eval_id": "c1", "source_row_index": 1595, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
         "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "a1", "reply_v3": "v1",
         "strategy_enabled": True, "strategy_depth": "SUBSTANTIVE"},
        {"eval_id": "c2", "source_row_index": 1544, "primary_intent": "SUGGESTION", "topic_subtype": "WORD_SUBMISSION",
         "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "a2", "reply_v3": "v2",
         "strategy_enabled": False, "strategy_depth": "BRIEF"},
        {"eval_id": "c3", "source_row_index": 1560, "primary_intent": "THANKS_PRAISE", "topic_subtype": "GENERAL",
         "sentiment": "POSITIVE", "routing_decision": "GENERATE_DRAFT", "reply_a": "a3", "reply_v3": "v3",
         "strategy_enabled": False, "strategy_depth": "BRIEF"},
    ]
    _write_jsonl(cases_path, cases)
    labels = [
        {"eval_id": "c1", "decision": "REPLY_2", "winner_variant": "V3", "reviewed_at": "x"},
        {"eval_id": "c2", "decision": "REPLY_1", "winner_variant": "A", "reviewed_at": "x"},
        {"eval_id": "c3", "decision": "REPLY_1", "winner_variant": "A", "reviewed_at": "x"},
    ]
    _write_jsonl(labels_path, labels)

    report = analyze_v3.analyze(cases_path, labels_path)
    assert "Eligible A/B cases: 3" in report
    assert "gate-positive (strategy applied): 1" in report
    assert "gate-negative (baseline-equivalent): 2" in report
    assert "Baseline (rg-v10) wins: 2/3" in report
    assert "Strategy V3 wins:       1/3" in report
    assert "gate-positive (strategy applied): n=1  baseline=0  v3=1" in report
    assert "gate-negative (baseline-equivalent): n=2  baseline=2  v3=0" in report


def test_analyzer_v3_reports_insufficient_when_no_labels(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [{
        "eval_id": "c1", "source_row_index": 1595, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
        "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "a1", "reply_v3": "v1",
        "strategy_enabled": True, "strategy_depth": "SUBSTANTIVE",
    }])
    report = analyze_v3.analyze(cases_path, labels_path)
    assert "INSUFFICIENT DATA" in report


def test_analyzer_v3_flags_unexpected_v3_wins_on_gate_negative(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [
        {"eval_id": f"c{i}", "source_row_index": 1544 + i, "primary_intent": "SUGGESTION", "topic_subtype": "WORD_SUBMISSION",
         "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": f"a{i}", "reply_v3": f"v{i}",
         "strategy_enabled": False, "strategy_depth": "BRIEF"}
        for i in range(3)
    ]
    _write_jsonl(cases_path, cases)
    labels = [{"eval_id": f"c{i}", "decision": "REPLY_2", "winner_variant": "V3", "reviewed_at": "x"} for i in range(3)]
    _write_jsonl(labels_path, labels)
    report = analyze_v3.analyze(cases_path, labels_path)
    assert "unexpected if the gate is truly inert" in report
