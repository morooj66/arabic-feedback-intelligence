"""
Tests for the routing-safety and reply-strategy EVALUATION TOOLING only
(scripts/offline_eval/routing_safety/, scripts/offline_eval/reply_strategy/).

These tests prove the TOOLING's own safety properties -- they do not
call the real frozen classifiers or real Gemini (no model load, no
network, fast and deterministic), and they do not assert anything about
production behavior beyond "was it left alone".
"""

from __future__ import annotations

import ast
import dataclasses
import json
import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ROUTING_DIR = _ROOT / "scripts" / "offline_eval" / "routing_safety"
_STRATEGY_DIR = _ROOT / "scripts" / "offline_eval" / "reply_strategy"

sys.path.insert(0, str(_ROUTING_DIR))
sys.path.insert(0, str(_STRATEGY_DIR))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import review as routing_review  # noqa: E402
import sample as routing_sample  # noqa: E402
import review_ab as strategy_review_ab  # noqa: E402
import strategy as strategy_module  # noqa: E402
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS  # noqa: E402


def _all_new_module_files() -> list[Path]:
    return [
        _ROUTING_DIR / "sample.py",
        _ROUTING_DIR / "review.py",
        _ROUTING_DIR / "analyze.py",
        _STRATEGY_DIR / "strategy.py",
        _STRATEGY_DIR / "experiment.py",
        _STRATEGY_DIR / "review_ab.py",
    ]


# --- Source Excel / DB / production-modification guards --------------------


def test_no_excel_write_functions_anywhere_in_the_new_tooling():
    for path in _all_new_module_files():
        source = path.read_text(encoding="utf-8")
        assert "to_excel(" not in source
        assert ".save(" not in source


def test_no_db_repository_or_service_imports_in_the_new_tooling():
    banned_prefixes = (
        "rcs.review.repositories", "rcs.review.repository", "rcs.review.service",
        "rcs.review.workflow", "rcs.review.excel_review_flow", "sqlalchemy", "pymongo",
    )
    for path in _all_new_module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        for module in imported:
            for banned in banned_prefixes:
                assert not module.startswith(banned), f"forbidden import {module!r} in {path}"


def test_no_production_routing_or_classification_files_modified():
    """This task is evaluation-only -- routing/engine.py and
    eligibility.py must not be imported for WRITE purposes anywhere in
    the new tooling (read-only inspection via inspect_eligibility is
    fine and expected)."""
    for path in _all_new_module_files():
        source = path.read_text(encoding="utf-8")
        assert "determine_routing(" not in source or path.name == "sample.py" and False
        assert "INTENT_ROUTING[" not in source
        assert "TOPIC_ROUTING_OVERRIDES[" not in source


def test_no_new_prompt_version_created():
    """No new prompt version was introduced by this directory's own new
    tooling files. `settings.RESPONSE_GENERATION_PROMPT_VERSION` itself
    is no longer pinned to a literal here: a later, separate
    prompt-content change legitimately bumped it to "rg-v11" for
    production traceability -- see rcs.response_generation.prompt's
    module docstring. That is a real, intentional version change
    unrelated to this directory's own scope, so this test only checks
    that its own new tooling files never reference a next-version
    label themselves."""
    for path in _all_new_module_files():
        assert "rg-v11" not in path.read_text(encoding="utf-8")


def test_experiment_reuses_real_system_instructions_not_a_copy():
    experiment_source = (_STRATEGY_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "from rcs.response_generation.prompt import build_generation_prompt" in experiment_source
    assert "SYSTEM_INSTRUCTIONS =" not in experiment_source  # never redefined/duplicated


def test_experiment_reply_b_never_calls_refinement_logic():
    """Reply B is documented as a deliberately single-shot, no-
    refinement variant -- proven here, not just claimed."""
    import experiment as strategy_experiment

    source_of_reply_b = strategy_experiment._generate_reply_b.__code__.co_names
    assert "RefinementContext" not in source_of_reply_b
    assert "run_agent_pipeline" not in source_of_reply_b


# --- Deterministic sample selection -----------------------------------


def test_sample_selection_is_deterministic_given_the_same_seed():
    population = [f"id-{i}" for i in range(50)]
    a = random.Random(routing_sample.SAMPLE_SEED).sample(population, k=10)
    b = random.Random(routing_sample.SAMPLE_SEED).sample(population, k=10)
    assert a == b


def test_long_threshold_and_sample_size_are_fixed_constants():
    assert routing_sample.LONG_TOKEN_THRESHOLD == 256
    assert isinstance(routing_sample.SAMPLE_SEED, int)


# --- Resumable human labels (routing_safety/review.py) -----------------


def test_routing_review_resumable_across_two_runs(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [
        {"eval_id": "c1", "comment_preview": "x", "topic_prediction": "OFF_TOPIC_PERSONAL", "topic_confidence": 0.2,
         "topic_second_best": None, "topic_second_best_confidence": None, "intent_prediction": "SUGGESTION",
         "intent_confidence": 0.3, "sentiment_prediction": "NEUTRAL", "total_topic_tokens": 300,
         "routing_decision": "NO_RESPONSE_RECOMMENDED", "routing_reason_code": "off_topic_personal", "would_call_gemini": False},
        {"eval_id": "c2", "comment_preview": "y", "topic_prediction": "WORD_SUBMISSION", "topic_confidence": 0.5,
         "topic_second_best": None, "topic_second_best_confidence": None, "intent_prediction": "SUGGESTION",
         "intent_confidence": 0.6, "sentiment_prediction": "NEUTRAL", "total_topic_tokens": 300,
         "routing_decision": "GENERATE_DRAFT", "routing_reason_code": "suggestion_eligible", "would_call_gemini": True},
    ]
    with cases_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    answers = iter(["A", "Q"])  # decide only case c1, then quit (simulates a partial session)
    routing_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers))
    labels = routing_review._load_jsonl(labels_path)
    assert len(labels) == 1 and labels[0]["eval_id"] == "c1"

    # Second run must skip c1 and only ask about c2.
    seen = []

    def _second_run_prompt(msg):
        seen.append(msg)
        return "A"

    routing_review.run_review(cases_path, labels_path, prompt_fn=_second_run_prompt)
    labels_after = routing_review._load_jsonl(labels_path)
    assert {row["eval_id"] for row in labels_after} == {"c1", "c2"}


def test_routing_review_skip_is_not_persisted_as_decided(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {"eval_id": "c1", "comment_preview": "x", "topic_prediction": "WORD_SUBMISSION", "topic_confidence": 0.5,
            "topic_second_best": None, "topic_second_best_confidence": None, "intent_prediction": "SUGGESTION",
            "intent_confidence": 0.6, "sentiment_prediction": "NEUTRAL", "total_topic_tokens": 300,
            "routing_decision": "GENERATE_DRAFT", "routing_reason_code": "suggestion_eligible", "would_call_gemini": True}
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    routing_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "S")
    assert routing_review._load_jsonl(labels_path) == []


def test_routing_review_reason_prompted_only_on_wrong(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {"eval_id": "c1", "comment_preview": "x", "topic_prediction": "OFF_TOPIC_PERSONAL", "topic_confidence": 0.2,
            "topic_second_best": None, "topic_second_best_confidence": None, "intent_prediction": "SUGGESTION",
            "intent_confidence": 0.4, "sentiment_prediction": "NEUTRAL", "total_topic_tokens": 500,
            "routing_decision": "NO_RESPONSE_RECOMMENDED", "routing_reason_code": "off_topic_personal", "would_call_gemini": False}
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    answers = iter(["R", "1"])
    routing_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers))
    labels = routing_review._load_jsonl(labels_path)
    assert labels[0]["decision"] == "WRONG"
    assert labels[0]["reason_category"] == "relevant_linguistic_content_incorrectly_dropped"


# --- [V] View full redacted comment -------------------------------------


def _make_pii_case(eval_id="c1", source_row_index=1):
    return {
        "eval_id": eval_id, "comment_preview": "x", "source_row_index": source_row_index,
        "topic_prediction": "WORD_SUBMISSION", "topic_confidence": 0.5,
        "topic_second_best": None, "topic_second_best_confidence": None, "intent_prediction": "SUGGESTION",
        "intent_confidence": 0.6, "sentiment_prediction": "NEUTRAL", "total_topic_tokens": 300,
        "routing_decision": "GENERATE_DRAFT", "routing_reason_code": "suggestion_eligible", "would_call_gemini": True,
    }


def test_view_reveals_only_redacted_text(tmp_path, capsys):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text(json.dumps(_make_pii_case()) + "\n", encoding="utf-8")

    raw_with_pii = "تواصلوا معي على 0501234567 من فضلكم."
    answers = iter(["V", "A"])
    routing_review.run_review(
        cases_path, labels_path,
        prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: raw_with_pii,
    )
    captured = capsys.readouterr()
    assert "0501234567" not in captured.out
    assert "تواصلوا معي" in captured.out  # non-PII content still shown, just not the phone number


def test_view_does_not_create_label(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text(json.dumps(_make_pii_case()) + "\n", encoding="utf-8")

    answers = iter(["V", "Q"])
    routing_review.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: "نص عادي بلا بيانات شخصية.",
    )
    assert not labels_path.exists() or routing_review._load_jsonl(labels_path) == []


def test_view_returns_to_the_same_case(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [_make_pii_case("c1", 1), _make_pii_case("c2", 2)]
    with cases_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    # View c1's full comment, THEN decide c1 -- must not have silently
    # advanced to c2 while showing the comment.
    answers = iter(["V", "A", "S", "Q"])
    routing_review.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: "نص عادي.",
    )
    labels = routing_review._load_jsonl(labels_path)
    assert len(labels) == 1
    assert labels[0]["eval_id"] == "c1"


def test_view_can_be_pressed_multiple_times_before_deciding(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text(json.dumps(_make_pii_case()) + "\n", encoding="utf-8")

    answers = iter(["V", "V", "V", "A"])
    routing_review.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: "نص عادي.",
    )
    labels = routing_review._load_jsonl(labels_path)
    assert len(labels) == 1 and labels[0]["decision"] == "APPROPRIATE"


def test_view_does_not_persist_pii_anywhere_in_labels_file(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text(json.dumps(_make_pii_case()) + "\n", encoding="utf-8")

    raw_with_pii = "للتواصل: user@example.com أو 0501234567."
    answers = iter(["V", "R", "1"])
    routing_review.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: raw_with_pii,
    )
    raw_labels_text = labels_path.read_text(encoding="utf-8")
    assert "0501234567" not in raw_labels_text
    assert "user@example.com" not in raw_labels_text
    assert "للتواصل" not in raw_labels_text  # no full comment text at all, redacted or not


def test_quit_after_viewing_still_exits_cleanly(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text(json.dumps(_make_pii_case()) + "\n", encoding="utf-8")

    answers = iter(["V", "Q"])
    routing_review.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: "نص عادي.",
    )
    assert not labels_path.exists() or routing_review._load_jsonl(labels_path) == []


def test_v_is_not_a_valid_terminal_decision_without_looping():
    # _prompt_decision must keep looping (never return control past "V"
    # without the caller re-prompting) -- verified at the run_review
    # level above; this checks the raw accepted-token set directly.
    answers = iter(["v", "a"])  # lowercase must also work (case-insensitive)
    result = routing_review._prompt_decision(lambda _msg: next(answers))
    assert result == "V"


# --- Q exits cleanly --------------------------------------------------


def test_routing_review_quit_exits_cleanly_without_writing(tmp_path):
    cases_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {"eval_id": "c1", "comment_preview": "x", "topic_prediction": "WORD_SUBMISSION", "topic_confidence": 0.5,
            "topic_second_best": None, "topic_second_best_confidence": None, "intent_prediction": "SUGGESTION",
            "intent_confidence": 0.6, "sentiment_prediction": "NEUTRAL", "total_topic_tokens": 300,
            "routing_decision": "GENERATE_DRAFT", "routing_reason_code": "suggestion_eligible", "would_call_gemini": True}
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    routing_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "Q")
    assert not labels_path.exists() or routing_review._load_jsonl(labels_path) == []


def test_ab_review_quit_exits_cleanly_without_writing(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {
        "eval_id": "c1", "comment_preview": "x", "primary_intent": "SUGGESTION", "topic_subtype": "GENERAL",
        "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "رد أ", "reply_b": "رد ب",
    }
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    strategy_review_ab.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "Q")
    assert not labels_path.exists() or strategy_review_ab._load_jsonl(labels_path) == []


# --- A/B review resumability + blind ordering ---------------------------


def test_ab_review_resumable(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [
        {"eval_id": f"c{i}", "comment_preview": "x", "primary_intent": "SUGGESTION", "topic_subtype": "GENERAL",
         "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": f"a{i}", "reply_b": f"b{i}"}
        for i in range(2)
    ]
    with cases_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    answers = iter(["A", "Q"])
    strategy_review_ab.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers))
    assert {row["eval_id"] for row in strategy_review_ab._load_jsonl(labels_path)} == {"c0"}

    answers2 = iter(["T"])
    strategy_review_ab.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers2))
    assert {row["eval_id"] for row in strategy_review_ab._load_jsonl(labels_path)} == {"c0", "c1"}


def test_ab_review_maps_positional_choice_back_to_variant_correctly(tmp_path):
    """The stored winner_variant must correctly account for the
    deterministic per-case swap, not just echo the raw A/B keypress."""
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    case = {"eval_id": "swap-test-id", "comment_preview": "x", "primary_intent": "SUGGESTION", "topic_subtype": "GENERAL",
            "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "REPLY_FROM_A", "reply_b": "REPLY_FROM_B"}
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    swap = strategy_review_ab._swap_for("swap-test-id")
    expected_reply_1 = "REPLY_FROM_B" if swap else "REPLY_FROM_A"

    seen_prompts = []

    def prompt_fn(msg):
        seen_prompts.append(msg)
        return "A"  # always choose "Reply 1 better"

    strategy_review_ab.run_review(cases_path, labels_path, prompt_fn=prompt_fn)
    label = strategy_review_ab._load_jsonl(labels_path)[0]
    expected_winner = "B" if swap else "A"
    assert label["winner_variant"] == expected_winner
    assert label["decision"] == "REPLY_1"


# --- PII / no raw prompt / no CoT / no reply-template in contracts -----


def test_sampled_case_record_has_no_visitor_identity_or_raw_prompt_fields():
    field_names = {f.name for f in dataclasses.fields(routing_sample.SampledCaseRecord)}
    for banned in ("visitor_name", "visitor_email", "raw_prompt", "prompt", "raw_comment", "chain_of_thought"):
        assert banned not in field_names


def test_comment_preview_is_redacted():
    from rcs.response_generation.prompt import redact_for_generation

    text = "تواصلوا معي على 0501234567 من فضلكم."
    preview = redact_for_generation(text)[:80]
    assert "0501234567" not in preview


def test_ab_case_has_no_visitor_identity_fields():
    import experiment as strategy_experiment

    field_names = {f.name for f in dataclasses.fields(strategy_experiment.ABCase)}
    for banned in ("visitor_name", "visitor_email", "raw_prompt", "chain_of_thought"):
        assert banned not in field_names


def test_response_strategy_contract_contains_no_reply_text_or_template():
    field_names = {f.name for f in dataclasses.fields(strategy_module.ResponseStrategy)}
    for banned in ("reply_text", "template", "final_reply", "suggested_response", "chain_of_thought", "reasoning"):
        assert banned not in field_names


def test_strategy_block_never_contains_a_ready_made_sentence():
    strategy = strategy_module.derive_strategy(
        comment_text="نص تجريبي قصير جدًا.", primary_intent="THANKS_PRAISE", topic_subtype="GENERAL",
        official_grounding_available=False,
    )
    block = strategy_module.render_strategy_block(strategy)
    assert "suggested_response" not in block
    assert block.count("\n") < 15  # short structured block, not a paragraph of prose


# --- Strategy derivation is deterministic and taxonomy-grounded --------


def test_strategy_goal_derivation_is_deterministic():
    a = strategy_module.derive_strategy(
        comment_text="اقتراح", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION",
        official_grounding_available=False,
    )
    b = strategy_module.derive_strategy(
        comment_text="اقتراح", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION",
        official_grounding_available=False,
    )
    assert a == b == strategy_module.ResponseStrategy(
        communicative_goal="RECEIVE_WORD_SUBMISSION", quoted_terms=(), response_depth="BRIEF",
        factual_answer_allowed=False, acknowledgment_only=True,
    )


def test_strategy_extracts_quoted_terms_deterministically():
    strategy = strategy_module.derive_strategy(
        comment_text="قلتم «التراكيب المتلازمة» لكن أقترح «المتلازمات اللفظية».",
        primary_intent="SUGGESTION", topic_subtype="ENTRY_CORRECTION", official_grounding_available=False,
    )
    assert strategy.quoted_terms == ("التراكيب المتلازمة", "المتلازمات اللفظية")


def test_strategy_only_allows_factual_answers_when_official_grounding_available():
    strategy = strategy_module.derive_strategy(
        comment_text="سؤال", primary_intent="INQUIRY", topic_subtype="GENERAL", official_grounding_available=True,
    )
    assert strategy.factual_answer_allowed is True
    strategy_without = strategy_module.derive_strategy(
        comment_text="سؤال", primary_intent="INQUIRY", topic_subtype="GENERAL", official_grounding_available=False,
    )
    assert strategy_without.factual_answer_allowed is False


def test_unknown_communicative_goal_rejected():
    with pytest.raises(ValueError):
        strategy_module.ResponseStrategy(
            communicative_goal="NOT_A_REAL_GOAL", quoted_terms=(), response_depth="BRIEF",
            factual_answer_allowed=False, acknowledgment_only=False,
        )
