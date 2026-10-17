"""
Focused tests for the Reply-Strategy V2 evaluation tooling
(scripts/offline_eval/reply_strategy/{strategy_v2,experiment_v2,
review_ab_v2,analyze_v2}.py). Fast and deterministic -- no real model
load, no network, no Gemini call.

These tests prove the TOOLING's own properties (intent precedence,
factual safety, determinism, isolation, duplicate-sampling discipline,
analyzer correctness) -- they say nothing about production behavior
beyond "was it left alone", which is asserted explicitly below.
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

import strategy_v2  # noqa: E402
import experiment_v2  # noqa: E402
import review_ab_v2  # noqa: E402
import analyze_v2  # noqa: E402
from config import settings  # noqa: E402


# --- A. Intent precedence ------------------------------------------------


def test_thanks_praise_word_submission_not_word_submission_goal():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="أحسنتم",
        primary_intent="THANKS_PRAISE",
        topic_subtype="WORD_SUBMISSION",
        official_grounding_available=False,
    )
    assert strategy.communicative_goal == "ACKNOWLEDGE_PRAISE"
    assert strategy.communicative_goal != "RECEIVE_WORD_SUBMISSION"


def test_inquiry_word_submission_remains_inquiry_oriented():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="ما معنى هذه الكلمة بالضبط؟",
        primary_intent="INQUIRY",
        topic_subtype="WORD_SUBMISSION",
        official_grounding_available=False,
    )
    assert strategy.communicative_goal == "ANSWER_INQUIRY"


def test_complaint_word_submission_remains_complaint_oriented():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="هذا خطأ في المعجم يجب تصحيحه",
        primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION",
        official_grounding_available=False,
    )
    assert strategy.communicative_goal == "ADDRESS_COMPLAINT"
    assert strategy.response_mode == "ACKNOWLEDGE_CONCERN"


def test_suggestion_entry_correction_remains_suggestion_oriented():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="لا يعمل الرابط الصوتي في صفحة هذه المادة عندي منذ يومين«اعتبار» يبدو أنها تُهمَل في معجمكم. " * 10,
        primary_intent="SUGGESTION",
        topic_subtype="ENTRY_CORRECTION",
        official_grounding_available=False,
    )
    assert strategy.communicative_goal == "ENGAGE_SUGGESTION"
    assert strategy.response_mode == "ENGAGE_WITH_CONTENT"  # long enough to be SUBSTANTIVE
    assert strategy.key_focus == "اعتبار"


def test_goal_is_intent_first_regardless_of_topic_across_all_topics():
    """Broader sweep: for a fixed intent, the goal must never change as
    topic_subtype varies -- this is the exact defect V1 had."""
    for topic in ("WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION", "GENERAL", None):
        s = strategy_v2.derive_strategy_v2(
            comment_text="نص تجريبي", primary_intent="COMPLAINT", topic_subtype=topic,
            official_grounding_available=False,
        )
        assert s.communicative_goal == "ADDRESS_COMPLAINT"


# --- B. Factual safety ----------------------------------------------------


def test_no_official_knowledge_factual_answer_not_allowed():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="هل هذا صحيح؟", primary_intent="INQUIRY", topic_subtype="GENERAL",
        official_grounding_available=False,
    )
    assert strategy.factual_answer_allowed is False
    assert strategy.evidence_requirement == "NONE_AVAILABLE_ACKNOWLEDGE_ONLY"


def test_official_knowledge_available_allows_factual_answer():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="هل هذا صحيح؟", primary_intent="INQUIRY", topic_subtype="GENERAL",
        official_grounding_available=True,
    )
    assert strategy.factual_answer_allowed is True
    assert strategy.evidence_requirement == "OFFICIAL_KNOWLEDGE_ONLY"


def test_strategy_cannot_mark_visitor_claims_authoritative_without_grounding():
    for intent in ("SUGGESTION", "INQUIRY"):
        strategy = strategy_v2.derive_strategy_v2(
            comment_text="هذا استشهاد لغوي مهم", primary_intent=intent, topic_subtype="ENTRY_CORRECTION",
            official_grounding_available=False,
        )
        assert strategy.acknowledge_claim_without_confirming is True
        assert strategy.factual_answer_allowed is False


def test_acknowledge_claim_flag_false_once_grounding_exists():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="هذا استشهاد لغوي مهم", primary_intent="SUGGESTION", topic_subtype="ENTRY_CORRECTION",
        official_grounding_available=True,
    )
    assert strategy.acknowledge_claim_without_confirming is False


def test_rendered_block_never_grants_authority_to_non_official_sources():
    strategy = strategy_v2.derive_strategy_v2(
        comment_text="اقتراح", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION",
        official_grounding_available=False,
    )
    block = strategy_v2.render_strategy_v2_block(strategy)
    assert "OFFICIAL KNOWLEDGE" in block
    assert "Approved Responses" in block or "Identifying Context" in block


# --- C. Determinism ---------------------------------------------------


def test_same_inputs_produce_identical_v2_strategy():
    kwargs = dict(
        comment_text="قلتم «التراكيب المتلازمة» وهي ترجمة تقابل collocation." * 3,
        primary_intent="SUGGESTION", topic_subtype="ENTRY_CORRECTION", official_grounding_available=False,
    )
    a = strategy_v2.derive_strategy_v2(**kwargs)
    b = strategy_v2.derive_strategy_v2(**kwargs)
    assert a == b


def test_unknown_goal_or_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        strategy_v2.ResponseStrategyV2(
            communicative_goal="NOT_REAL", response_mode="ACKNOWLEDGMENT_ONLY", response_depth="BRIEF",
            factual_answer_allowed=False, acknowledge_claim_without_confirming=False,
            key_focus="x", evidence_requirement="NONE_AVAILABLE_ACKNOWLEDGE_ONLY",
        )
    with pytest.raises(ValueError):
        strategy_v2.ResponseStrategyV2(
            communicative_goal="ADDRESS_COMPLAINT", response_mode="NOT_REAL", response_depth="BRIEF",
            factual_answer_allowed=False, acknowledge_claim_without_confirming=False,
            key_focus="x", evidence_requirement="NONE_AVAILABLE_ACKNOWLEDGE_ONLY",
        )


def test_no_reply_text_or_chain_of_thought_field_in_contract():
    field_names = {f.name for f in dataclasses.fields(strategy_v2.ResponseStrategyV2)}
    for banned in ("reply_text", "template", "final_reply", "suggested_response", "chain_of_thought", "reasoning"):
        assert banned not in field_names


# --- D. Evaluation isolation --------------------------------------------


def _v2_module_files():
    return [
        _STRATEGY_DIR / "strategy_v2.py",
        _STRATEGY_DIR / "experiment_v2.py",
        _STRATEGY_DIR / "review_ab_v2.py",
        _STRATEGY_DIR / "analyze_v2.py",
    ]


def test_no_db_imports_in_v2_tooling():
    banned_prefixes = (
        "rcs.review.repositories", "rcs.review.repository", "rcs.review.service",
        "rcs.review.workflow", "rcs.review.excel_review_flow", "sqlalchemy", "pymongo",
    )
    for path in _v2_module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        for module in imported:
            for banned in banned_prefixes:
                assert not module.startswith(banned), f"forbidden import {module!r} in {path}"


def test_no_excel_write_functions_in_v2_tooling():
    for path in _v2_module_files():
        source = path.read_text(encoding="utf-8")
        assert "to_excel(" not in source
        assert ".save(" not in source


def test_no_new_prompt_version_or_rg_v11():
    """No new prompt version was introduced by this V2 tooling's own
    files (`_v2_module_files()`, scoped to this directory). The
    settings literal now reads "rg-v14.2": later, separate
    prompt-content changes legitimately bumped it -- see
    rcs.response_generation.prompt's module docstring -- unrelated to
    this V2 tooling, which was never touched."""
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    for path in _v2_module_files():
        assert "rg-v11" not in path.read_text(encoding="utf-8")


def test_experiment_v2_reuses_real_prompt_builder_not_a_copy():
    source = (_STRATEGY_DIR / "experiment_v2.py").read_text(encoding="utf-8")
    assert "from rcs.response_generation.prompt import build_generation_prompt" in source
    assert "SYSTEM_INSTRUCTIONS =" not in source  # never redefined/duplicated


def test_reply_b_v2_makes_no_extra_llm_call_to_derive_strategy():
    """derive_strategy_v2 must be a pure function with no client/generate
    call anywhere in its bytecode names."""
    names = strategy_v2.derive_strategy_v2.__code__.co_names
    assert "generate" not in names
    assert "client" not in names


def test_reply_b_v2_never_calls_refinement_logic():
    names = experiment_v2._generate_reply_b.__code__.co_names
    assert "RefinementContext" not in names
    assert "run_agent_pipeline" not in names


def test_v2_case_dataclass_has_no_visitor_identity_fields():
    field_names = {f.name for f in dataclasses.fields(experiment_v2.ABCaseV2)}
    for banned in ("visitor_name", "visitor_email", "raw_prompt", "chain_of_thought"):
        assert banned not in field_names


# --- E. Duplicate sampling -------------------------------------------


def test_v2_selected_rows_exclude_known_exact_duplicate_cluster():
    exact_duplicate_cluster = set(range(1002, 1012))  # the "مِضْرَادٌ" cluster, 10 rows
    selected = set(experiment_v2.SELECTED_SOURCE_ROW_INDICES_V2)
    assert selected.isdisjoint(exact_duplicate_cluster)


def test_v2_selected_rows_include_at_most_one_of_near_duplicate_pair():
    near_duplicate_pair = {1190, 1191}
    selected = set(experiment_v2.SELECTED_SOURCE_ROW_INDICES_V2)
    assert len(selected & near_duplicate_pair) <= 1


def test_v2_selected_rows_have_no_internal_duplicates():
    rows = experiment_v2.SELECTED_SOURCE_ROW_INDICES_V2
    assert len(rows) == len(set(rows))


def test_v2_sample_size_within_requested_range():
    assert 20 <= len(experiment_v2.SELECTED_SOURCE_ROW_INDICES_V2) <= 30


# --- F. Analyzer -------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_analyzer_reports_insufficient_when_no_labels(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [{
        "eval_id": "c1", "source_row_index": 1, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
        "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "a", "reply_b": "b",
        "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ENGAGE_WITH_CONTENT", "strategy_depth": "SUBSTANTIVE",
    }])
    report = analyze_v2.analyze(cases_path, labels_path)
    assert "INSUFFICIENT DATA" in report


def test_analyzer_unblinds_and_counts_correctly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [
        {"eval_id": "c1", "source_row_index": 1, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
         "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "a1", "reply_b": "b1",
         "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ENGAGE_WITH_CONTENT", "strategy_depth": "SUBSTANTIVE"},
        {"eval_id": "c2", "source_row_index": 2, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
         "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "a2", "reply_b": "b2",
         "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ENGAGE_WITH_CONTENT", "strategy_depth": "SUBSTANTIVE"},
        {"eval_id": "c3", "source_row_index": 15, "primary_intent": "THANKS_PRAISE", "topic_subtype": "WORD_SUBMISSION",
         "sentiment": "POSITIVE", "routing_decision": "GENERATE_DRAFT", "reply_a": "a3", "reply_b": "b3",
         "strategy_goal": "ACKNOWLEDGE_PRAISE", "strategy_mode": "ACKNOWLEDGMENT_ONLY", "strategy_depth": "BRIEF"},
        {"eval_id": "c4", "source_row_index": 999, "primary_intent": "SPAM", "topic_subtype": "OFF_TOPIC_PERSONAL",
         "sentiment": "NEUTRAL", "routing_decision": "NO_RESPONSE_RECOMMENDED", "reply_a": None, "reply_b": None,
         "strategy_goal": "", "strategy_mode": "", "strategy_depth": ""},
    ]
    _write_jsonl(cases_path, cases)
    labels = [
        {"eval_id": "c1", "decision": "REPLY_2", "winner_variant": "B", "reviewed_at": "2026-01-01T00:00:00+00:00"},
        {"eval_id": "c2", "decision": "REPLY_1", "winner_variant": "A", "reviewed_at": "2026-01-01T00:00:00+00:00"},
        {"eval_id": "c3", "decision": "TIE", "winner_variant": "TIE", "reviewed_at": "2026-01-01T00:00:00+00:00"},
    ]
    _write_jsonl(labels_path, labels)

    report = analyze_v2.analyze(cases_path, labels_path)
    assert "Eligible A/B cases: 3" in report  # c4 excluded (not eligible: reply_a/b are None)
    assert "Total reviewed (decided): 3" in report
    assert "Baseline (rg-v10) wins: 1/3" in report
    assert "Strategy V2 wins:       1/3" in report
    assert "Ties:                   1/3" in report
    assert "Skipped/unreviewed:     0" in report
    assert "SUGGESTION + ENTRY_CORRECTION" in report
    assert "n=2  baseline=1  strategy_v2=1" in report


def test_analyzer_never_claims_significance():
    cases_path_content = [{
        "eval_id": "c1", "source_row_index": 1, "primary_intent": "SUGGESTION", "topic_subtype": "ENTRY_CORRECTION",
        "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT", "reply_a": "a", "reply_b": "b",
        "strategy_goal": "ENGAGE_SUGGESTION", "strategy_mode": "ENGAGE_WITH_CONTENT", "strategy_depth": "SUBSTANTIVE",
    }]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cases_path = Path(d) / "cases.jsonl"
        labels_path = Path(d) / "labels.jsonl"
        _write_jsonl(cases_path, cases_path_content)
        _write_jsonl(labels_path, [{"eval_id": "c1", "decision": "REPLY_1", "winner_variant": "A", "reviewed_at": "x"}])
        report = analyze_v2.analyze(cases_path, labels_path)
    assert "exploratory" in report.lower()
    assert "not a statistically significant" in report.lower() or "not a statistical significance" in report.lower()


# --- [V] view option parity with routing-safety tool --------------------


def test_review_ab_v2_view_does_not_create_label(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [{
        "eval_id": "c1", "source_row_index": 1, "comment_preview": "x", "primary_intent": "SUGGESTION",
        "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
        "reply_a": "a1", "reply_b": "b1",
    }])
    answers = iter(["V", "Q"])
    review_ab_v2.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: "نص عادي بلا بيانات شخصية.",
    )
    assert not labels_path.exists() or review_ab_v2._load_jsonl(labels_path) == []


def test_review_ab_v2_view_reveals_only_redacted_text(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [{
        "eval_id": "c1", "source_row_index": 1, "comment_preview": "x", "primary_intent": "SUGGESTION",
        "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
        "reply_a": "a1", "reply_b": "b1",
    }])
    raw_with_pii = "تواصلوا معي على 0501234567 من فضلكم."
    answers = iter(["V", "T"])
    review_ab_v2.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: raw_with_pii,
    )
    captured = capsys.readouterr()
    assert "0501234567" not in captured.out


def test_review_ab_v2_quit_exits_cleanly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [{
        "eval_id": "c1", "source_row_index": 1, "comment_preview": "x", "primary_intent": "SUGGESTION",
        "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
        "reply_a": "a1", "reply_b": "b1",
    }])
    review_ab_v2.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "Q")
    assert not labels_path.exists() or review_ab_v2._load_jsonl(labels_path) == []


def test_review_ab_v2_resumable(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases = [
        {"eval_id": f"c{i}", "source_row_index": i, "comment_preview": "x", "primary_intent": "SUGGESTION",
         "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
         "reply_a": f"a{i}", "reply_b": f"b{i}"}
        for i in range(2)
    ]
    _write_jsonl(cases_path, cases)

    answers = iter(["A", "Q"])
    review_ab_v2.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers))
    assert {row["eval_id"] for row in review_ab_v2._load_jsonl(labels_path)} == {"c0"}

    answers2 = iter(["N"])
    review_ab_v2.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers2))
    assert {row["eval_id"] for row in review_ab_v2._load_jsonl(labels_path)} == {"c0", "c1"}


def test_review_ab_v2_does_not_persist_pii(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [{
        "eval_id": "c1", "source_row_index": 1, "comment_preview": "x", "primary_intent": "SUGGESTION",
        "topic_subtype": "ENTRY_CORRECTION", "sentiment": "NEUTRAL", "routing_decision": "GENERATE_DRAFT",
        "reply_a": "a1", "reply_b": "b1",
    }])
    raw_with_pii = "للتواصل: user@example.com أو 0501234567."
    answers = iter(["V", "A"])
    review_ab_v2.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: raw_with_pii,
    )
    raw_labels_text = labels_path.read_text(encoding="utf-8")
    assert "0501234567" not in raw_labels_text
    assert "user@example.com" not in raw_labels_text
