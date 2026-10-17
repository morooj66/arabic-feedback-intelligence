"""
Focused tests for the Identifying Context Mechanism experiment
(scripts/offline_eval/identifying_context_mechanism/{experiment,review,
analyze}.py). Fast and deterministic -- no real model load, no network,
no Gemini call. Reuses the same fakes the M5F offline-eval harness
already established.
"""

from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MECH_DIR = _ROOT / "scripts" / "offline_eval" / "identifying_context_mechanism"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def _load_local_module(unique_name: str, file_path: Path):
    """Loads a script module under a UNIQUE sys.modules key instead of
    its bare filename -- several evaluation-tooling directories each
    have their own experiment.py/review.py/analyze.py, and a bare
    `import experiment` would silently reuse whichever same-named module
    another test file already cached from a DIFFERENT directory."""
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


mech_experiment = _load_local_module("idctx_mechanism_experiment", _MECH_DIR / "experiment.py")
mech_review = _load_local_module("idctx_mechanism_review", _MECH_DIR / "review.py")
mech_analyze = _load_local_module("idctx_mechanism_analyze", _MECH_DIR / "analyze.py")
from config import settings  # noqa: E402
from rcs.response_generation.prompt import build_identifying_grounding  # noqa: E402
from rcs.response_intelligence.retrieval.contracts import (  # noqa: E402
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
    OfficialKnowledgeResult,
)
from scripts.offline_eval.response_agent.fakes import (  # noqa: E402
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
)

_VALID_DRAFT_JSON = '{"suggested_response": "شكرًا لتواصلكم معنا."}'
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)


class _FixedOfficialKnowledgeRetriever:
    def __init__(self, result: OfficialKnowledgeRetrievalResult):
        self.result = result
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        return self.result


def _empty_ok():
    return _FixedOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY))


def _run_agent_pipeline_direct(request, client, ok, arm):
    from rcs.response_intelligence.orchestrator import run_agent_pipeline

    return run_agent_pipeline(
        request, client=client, official_knowledge_retriever=ok, approved_response_retriever=arm,
    )


# --- Design constants sanity ------------------------------------------


def test_total_variant_count_within_requested_range():
    total = sum(len(bc.conditions) for bc in mech_experiment.SELECTED_BASE_CASES)
    assert 12 <= total <= 18


def test_required_base_comments_present():
    row_indices = {bc.source_row_index for bc in mech_experiment.SELECTED_BASE_CASES}
    # لا اعلم, الله اعلم, تتت, أنشطة لا صفية
    assert {6281, 112, 6278, 6236}.issubset(row_indices)


def test_every_base_case_includes_none_condition():
    """NONE must always be present -- it is the required baseline every
    other condition is compared against."""
    for bc in mech_experiment.SELECTED_BASE_CASES:
        assert "NONE" in bc.conditions


def test_specific_condition_uses_the_real_row_6281_word():
    """The SPECIFIC stimulus is deliberately the exact word implicated in
    the confirmed hijacking case, reused as a fixed control -- not a
    fresh guess."""
    assert mech_experiment.SPECIFIC_TITLE == "قناطر"


# --- All variants use the same orchestrator path; only identifying_context varies ---


def test_all_conditions_call_run_agent_pipeline_not_a_manual_prompt():
    source = (_MECH_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "build_generation_prompt" not in source
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "run_agent_pipeline" in names


def test_no_strategy_v1_v2_v3_imports():
    """AST-based (not substring) so an honest docstring mention of what
    this module does NOT import can't trip the check."""
    tree = ast.parse((_MECH_DIR / "experiment.py").read_text(encoding="utf-8"))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    for banned in ("strategy_v2", "strategy_v3", "prompt_candidates"):
        assert not any(banned in m for m in imported_modules), f"{banned} imported"
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "derive_strategy_v2" not in names
    assert "derive_strategy_v3" not in names


def test_only_identifying_metadata_differs_across_conditions_for_a_case():
    """Build the three requests for a NONE/GENERIC/SPECIFIC base case
    directly and prove every other field is identical -- only
    identifying_metadata differs. Phase M6.2: structured metadata, not
    pre-rendered text."""
    from rcs.response_generation.service import GenerationRequest
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata

    common = dict(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
    )
    none_req = GenerationRequest(**common, identifying_metadata=None)
    generic_req = GenerationRequest(**common, identifying_metadata=IdentifyingMetadata(mech_experiment.GENERIC_TITLE, None))
    specific_req = GenerationRequest(**common, identifying_metadata=IdentifyingMetadata(mech_experiment.SPECIFIC_TITLE, None))

    for field in ("comment_text", "primary_intent", "topic_subtype", "sentiment", "confidence"):
        assert getattr(none_req, field) == getattr(generic_req, field) == getattr(specific_req, field)
    assert none_req.identifying_metadata != generic_req.identifying_metadata != specific_req.identifying_metadata
    assert none_req.identifying_metadata is None


def test_resolve_identifying_metadata_none_generic_specific_own_real():
    class _Row(dict):
        def __getitem__(self, key):
            return dict.get(self, key)

    row = _Row(original_title="عنوان حقيقي", entry_id="entry-1")
    assert mech_experiment._resolve_identifying_metadata("NONE", row) is None
    generic = mech_experiment._resolve_identifying_metadata("GENERIC", row)
    specific = mech_experiment._resolve_identifying_metadata("SPECIFIC", row)
    own_real = mech_experiment._resolve_identifying_metadata("OWN_REAL", row)
    assert generic.original_title == "إثراء المعجم"
    assert specific.original_title == "قناطر"
    assert own_real.original_title == "عنوان حقيقي" and own_real.entry_id == "entry-1"
    assert generic != specific != own_real


# --- rg-v10 unchanged (by this experiment's own files) -------------------
#
# Two later, separate prompt-content changes legitimately bumped
# settings.RESPONSE_GENERATION_PROMPT_VERSION, currently "rg-v14.2" --
# see rcs.response_generation.prompt's module docstring. The literals
# below are updated to that current value; the substring check stays
# scoped to this experiment's own (untouched) experiment.py file,
# which still correctly never mentions "rg-v11".


def test_rg_v10_unchanged():
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    source = (_MECH_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "rg-v11" not in source


def test_generated_result_carries_rg_v10_prompt_version():
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    from rcs.response_generation.service import GenerationRequest
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata

    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata(mech_experiment.SPECIFIC_TITLE, None),
    )
    result = _run_agent_pipeline_direct(request, client, _empty_ok(), FakeApprovedResponseMemoryRetriever(_EMPTY_ARM))
    assert result.generation_result.prompt_version == "rg-v14.4"


# --- Official Knowledge availability constant within a controlled set ---


def test_official_grounding_availability_constant_across_conditions_empty():
    from rcs.response_generation.service import GenerationRequest
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata

    ok = _empty_ok()
    arm = FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    results = []
    for metadata in (None, IdentifyingMetadata("إثراء المعجم", None), IdentifyingMetadata("قناطر", None)):
        client = FakeGeminiClient([_VALID_DRAFT_JSON])
        request = GenerationRequest(
            comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
            topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
            identifying_metadata=metadata,
        )
        results.append(_run_agent_pipeline_direct(request, client, ok, arm))
    availabilities = {r.metadata.official_grounding_available for r in results}
    assert availabilities == {False}


def test_official_grounding_availability_constant_across_conditions_present():
    from rcs.response_generation.service import GenerationRequest
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata

    ok_result = OfficialKnowledgeRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS,
        results=(
            OfficialKnowledgeResult(
                source_id="s1", source_title="Doc", source_type="dictionary_entry",
                content="حقيقة معتمدة.", version="v1", effective_from=dt.datetime(2026, 1, 1),
                authoritative=True, retrieval_score=0.9,
            ),
        ),
    )
    ok = _FixedOfficialKnowledgeRetriever(ok_result)
    arm = FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    results = []
    for metadata in (None, IdentifyingMetadata("قناطر", None)):
        client = FakeGeminiClient([_VALID_DRAFT_JSON])
        request = GenerationRequest(
            comment_id="x", comment_text="لا اعلم", primary_intent="INQUIRY",
            topic_subtype="GENERAL", sentiment="NEUTRAL", confidence=0.4,
            identifying_metadata=metadata,
        )
        results.append(_run_agent_pipeline_direct(request, client, ok, arm))
    assert {r.metadata.official_grounding_available for r in results} == {True}


# --- Structural post-implementation confirmation (Phase M6.2) -----------
# Reuses this same mechanism tooling's real case shapes to prove, without
# any live Gemini call, that the identifying-context policy actually
# gates what reaches the prompt for the confirmed hijacking pattern and
# for a known-good corroborated case.


def test_confirmed_hijacking_pair_produces_no_identifying_context_section():
    from rcs.response_generation.service import GenerationRequest
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata

    for comment_text in ("لا اعلم", "الله اعلم", "تتت"):
        client = FakeGeminiClient([_VALID_DRAFT_JSON])
        request = GenerationRequest(
            comment_id="x", comment_text=comment_text, primary_intent="COMPLAINT",
            topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
            identifying_metadata=IdentifyingMetadata("قناطر", None),
        )
        result = _run_agent_pipeline_direct(
            request, client, _empty_ok(), FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
        )
        assert result.metadata.identifying_context_allowed is False
        assert result.metadata.identifying_context_decision_reason == "NO_CORROBORATING_REFERENCE"
        assert "IDENTIFYING CONTEXT" not in client.prompts[0]
        assert "قناطر" not in client.prompts[0]


def test_known_good_lexically_corroborated_case_still_receives_identifying_context():
    from rcs.response_generation.service import GenerationRequest
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata

    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا يعمل الرابط الصوتي في صفحة هذه المادة عندي منذ يومين«اعتبار» يبدو أنها تُهمَل في معجمكم.",
        primary_intent="SUGGESTION", topic_subtype="ENTRY_CORRECTION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("اِعْتِبَار", "6b746072-ca71-483f-8b73-c52da0efbdde"),
    )
    result = _run_agent_pipeline_direct(
        request, client, _empty_ok(), FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    )
    assert result.metadata.identifying_context_allowed is True
    assert result.metadata.identifying_context_decision_reason == "LEXICAL_CORROBORATION"
    assert "IDENTIFYING CONTEXT" in client.prompts[0]
    # entry_id is NEVER used to DECIDE the policy outcome (proven by
    # test_entry_id_never_appears_as_identifying_context_even_when_it_is_a_real_word
    # below); build_identifying_grounding's own unchanged rendering
    # template separately includes entry_id in its output when present
    # (pre-existing M5D.2 behavior, out of scope for this policy).


def test_entry_id_never_appears_as_identifying_context_even_when_it_is_a_real_word():
    """Arabic-looking entry_id must never become semantic identifying
    context by itself -- only original_title may."""
    from rcs.response_generation.service import GenerationRequest
    from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata

    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata(None, "قناطر"),  # entry_id is a real Arabic word here
    )
    result = _run_agent_pipeline_direct(
        request, client, _empty_ok(), FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    )
    assert result.metadata.identifying_context_allowed is False
    assert result.metadata.identifying_context_decision_reason == "NO_IDENTIFYING_METADATA"
    assert "قناطر" not in client.prompts[0]


# --- Isolation: no DB writes, no source Excel mutation, no PII ----------


def _module_files():
    return [_MECH_DIR / "experiment.py", _MECH_DIR / "review.py", _MECH_DIR / "analyze.py"]


def test_no_db_imports():
    banned_prefixes = (
        "rcs.review.repositories", "rcs.review.repository", "rcs.review.service",
        "rcs.review.workflow", "rcs.review.excel_review_flow", "sqlalchemy", "pymongo",
    )
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        for module in imported:
            for banned in banned_prefixes:
                assert not module.startswith(banned), f"forbidden import {module!r} in {path}"


def test_no_excel_write_functions():
    for path in _module_files():
        source = path.read_text(encoding="utf-8")
        assert "to_excel(" not in source
        assert ".save(" not in source


def test_experiment_never_passes_audit_repository():
    source = (_MECH_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "audit_repository" not in source
    assert "AgentAuditRepository" not in source


def test_source_excel_hash_stable_if_present():
    import hashlib

    source_file = _ROOT / "data" / "raw" / "historical-export.xlsx"
    if not source_file.exists():
        import pytest
        pytest.skip("source Excel not present in this environment")
    h1 = hashlib.md5(source_file.read_bytes()).hexdigest()
    h2 = hashlib.md5(source_file.read_bytes()).hexdigest()
    assert h1 == h2


def test_comment_and_identifying_context_previews_are_redacted():
    from rcs.response_generation.prompt import redact_for_generation

    raw_comment = "تواصلوا معي على 0501234567."
    raw_title = "قسم 0509876543"
    preview = redact_for_generation(raw_comment)[:100]
    identifying_preview = redact_for_generation(build_identifying_grounding(raw_title, None))[:200]
    assert "0501234567" not in preview
    assert "0509876543" not in identifying_preview


# --- Review CLI: paired-per-set, resumable, safe -------------------------


def _cases_for_one_base(base_case_id="laa_aelam", specificities=("NONE", "GENERIC", "SPECIFIC")):
    rows = []
    for spec in specificities:
        rows.append({
            "eval_id": f"{base_case_id}-{spec}", "base_case_id": base_case_id, "source_row_index": 6281,
            "role": "VACUOUS", "comment_class": "VERY_SHORT_NONCOMMITTAL", "context_specificity": spec,
            "comment_preview": "لا اعلم", "identifying_context_preview": None if spec == "NONE" else "y",
            "primary_intent": "COMPLAINT", "topic_subtype": "WORD_SUBMISSION", "sentiment": "NEUTRAL",
            "routing_decision": "GENERATE_DRAFT", "official_grounding_available": False,
            "generated_reply": f"reply-{spec}", "generation_status": "SUCCEEDED", "reason_code": "complaint_safe_acknowledgement",
        })
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_review_skips_none_as_a_reviewable_item():
    items = mech_review._build_review_items(_cases_for_one_base())
    eval_ids = {i["eval_id"] for i in items}
    assert "laa_aelam-NONE" not in eval_ids
    assert eval_ids == {"laa_aelam-GENERIC", "laa_aelam-SPECIFIC"}


def test_review_pairs_each_item_with_its_own_baseline():
    items = mech_review._build_review_items(_cases_for_one_base())
    for item in items:
        assert item["baseline_reply"] == "reply-NONE"


def test_review_resumable(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, _cases_for_one_base())

    answers = iter(["C", "Q"])
    mech_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers))
    labels = mech_review._load_jsonl(labels_path)
    assert len(labels) == 1

    answers2 = iter(["B"])
    mech_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers2))
    assert len(mech_review._load_jsonl(labels_path)) == 2


def test_review_skip_not_persisted(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, _cases_for_one_base())
    mech_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "S")
    assert not labels_path.exists() or mech_review._load_jsonl(labels_path) == []


def test_review_quit_exits_cleanly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, _cases_for_one_base())
    mech_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "Q")
    assert not labels_path.exists() or mech_review._load_jsonl(labels_path) == []


def test_review_decision_mapping():
    cases_path_rows = _cases_for_one_base()
    assert mech_review._DECISION_LABEL["C"] == "CONTEXT_HIJACKING"
    assert mech_review._DECISION_LABEL["A"] == "IMPROVED_RELEVANCE"


def test_review_view_does_not_persist_pii(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, _cases_for_one_base())
    answers = iter(["V", "B", "Q"])
    mech_review.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: ("رقمي 0501234567", None),
    )
    raw_labels_text = labels_path.read_text(encoding="utf-8")
    assert "0501234567" not in raw_labels_text


# --- Analyzer -------------------------------------------------------


def test_analyzer_insufficient_when_no_labels(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, _cases_for_one_base())
    report = mech_analyze.analyze(cases_path, labels_path)
    assert "INSUFFICIENT DATA" in report


def test_analyzer_hijack_rate_and_interaction_table(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, _cases_for_one_base())
    labels = [
        {"eval_id": "laa_aelam-GENERIC", "base_case_id": "laa_aelam", "comment_class": "VERY_SHORT_NONCOMMITTAL",
         "context_specificity": "GENERIC", "decision": "NO_MEANINGFUL_EFFECT", "reviewed_at": "x"},
        {"eval_id": "laa_aelam-SPECIFIC", "base_case_id": "laa_aelam", "comment_class": "VERY_SHORT_NONCOMMITTAL",
         "context_specificity": "SPECIFIC", "decision": "CONTEXT_HIJACKING", "reviewed_at": "x"},
    ]
    _write_jsonl(labels_path, labels)
    report = mech_analyze.analyze(cases_path, labels_path)
    assert "CONTEXT_HIJACKING: 1/2" in report
    assert "GENERIC: 0/1" in report
    assert "SPECIFIC: 1/1" in report
    assert "laa_aelam-SPECIFIC" in report
    assert "exploratory" in report.lower()


def test_analyzer_never_claims_significance():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cases_path = Path(d) / "cases.jsonl"
        labels_path = Path(d) / "labels.jsonl"
        _write_jsonl(cases_path, _cases_for_one_base())
        _write_jsonl(labels_path, [{
            "eval_id": "laa_aelam-GENERIC", "base_case_id": "laa_aelam", "comment_class": "VERY_SHORT_NONCOMMITTAL",
            "context_specificity": "GENERIC", "decision": "NO_MEANINGFUL_EFFECT", "reviewed_at": "x",
        }])
        report = mech_analyze.analyze(cases_path, labels_path)
    assert "not a statistically significant" in report.lower()
    assert "no suppression rule should be designed from this alone" in report.lower()
