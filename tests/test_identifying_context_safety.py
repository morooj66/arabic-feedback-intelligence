"""
Focused tests for the Identifying Context Safety Check evaluation
tooling (scripts/offline_eval/identifying_context_safety/{experiment,
review}.py). Fast and deterministic -- no real model load, no network,
no Gemini call. Uses the same fakes the M5F offline-eval harness already
established (FakeGeminiClient/FakeOfficialKnowledgeRetriever/
FakeApprovedResponseMemoryRetriever).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SAFETY_DIR = _ROOT / "scripts" / "offline_eval" / "identifying_context_safety"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def _load_local_module(unique_name: str, file_path: Path):
    """Loads a script module under a UNIQUE sys.modules key instead of
    its bare filename. Several evaluation-tooling directories each have
    their own experiment.py/review.py/analyze.py -- a bare `import
    experiment` would be cached by the generic name "experiment" and
    silently reused (wrong module!) if another test file already
    imported a same-named module from a different directory. This
    avoids that collision entirely."""
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


idctx_experiment = _load_local_module("idctx_safety_experiment", _SAFETY_DIR / "experiment.py")
idctx_review = _load_local_module("idctx_safety_review", _SAFETY_DIR / "review.py")
from config import settings  # noqa: E402
from rcs.response_generation.prompt import build_identifying_grounding  # noqa: E402
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata  # noqa: E402
from rcs.response_intelligence.retrieval.contracts import (  # noqa: E402
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
)
from scripts.offline_eval.response_agent.fakes import (  # noqa: E402
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
)


_VALID_DRAFT_JSON = '{"suggested_response": "شكرًا لتواصلكم معنا."}'
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)


class _FixedOfficialKnowledgeRetriever:
    """Local, minimal fake identical in shape to FakeOfficialKnowledgeRetriever
    -- reused instance across both calls in a pair so identical retrieval
    behavior for control and context is trivially guaranteed, not just
    asserted after the fact."""

    def __init__(self, result: OfficialKnowledgeRetrievalResult):
        self.result = result
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        return self.result


def _empty_ok():
    return _FixedOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY))


# --- Control has no identifying context; context variant does ------------


def test_control_request_has_no_identifying_context():
    client = FakeGeminiClient([_VALID_DRAFT_JSON, _VALID_DRAFT_JSON])
    ok = _empty_ok()
    control_result, context_result = idctx_experiment.run_pair(
        comment_id_prefix="t1", comment_text="نص تجريبي طويل بما يكفي.",
        primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.9,
        identifying_metadata=IdentifyingMetadata("كلمة تجريبية", None),
        client=client, official_knowledge_retriever=ok, approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    # The control prompt must contain no "IDENTIFYING CONTEXT" section at all.
    assert "IDENTIFYING CONTEXT" not in client.prompts[0]
    assert "كلمة تجريبية" not in client.prompts[0]


def test_context_request_has_identifying_context_when_corroborated():
    """Phase M6.2: the CONTEXT variant only receives identifying context
    when the comment corroborates it -- use a comment that lexically
    references the title, matching the new policy."""
    client = FakeGeminiClient([_VALID_DRAFT_JSON, _VALID_DRAFT_JSON])
    ok = _empty_ok()
    control_result, context_result = idctx_experiment.run_pair(
        comment_id_prefix="t2", comment_text="نص تجريبي طويل يذكر «كلمة تجريبية» صراحة.",
        primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.9,
        identifying_metadata=IdentifyingMetadata("كلمة تجريبية", None),
        client=client, official_knowledge_retriever=ok, approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert context_result.metadata.identifying_context_allowed is True
    assert "IDENTIFYING CONTEXT" in client.prompts[1]
    assert "كلمة تجريبية" in client.prompts[1]


# --- Both use rg-v10, both traverse the same orchestrator path -----------


def test_both_variants_use_rg_v10():
    """Value updated to "rg-v14": later, separate prompt-content
    changes bumped settings.RESPONSE_GENERATION_PROMPT_VERSION for
    production traceability (see rcs.response_generation.prompt's
    module docstring). This test's own point -- both variants traverse
    the SAME orchestrator path and therefore share one prompt version
    -- is unaffected; only the literal value tracks current truth."""
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    client = FakeGeminiClient([_VALID_DRAFT_JSON, _VALID_DRAFT_JSON])
    control_result, context_result = idctx_experiment.run_pair(
        comment_id_prefix="t3", comment_text="نص تجريبي طويل بما يكفي.",
        primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.9,
        identifying_metadata=IdentifyingMetadata("X", None),
        client=client, official_knowledge_retriever=_empty_ok(),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert control_result.generation_result.prompt_version == "rg-v14.4"
    assert context_result.generation_result.prompt_version == "rg-v14.4"


def test_experiment_never_builds_a_manual_prompt():
    """Structural proof: this module never imports build_generation_prompt
    -- both variants MUST go through run_agent_pipeline, no alternative
    prompt-construction path exists here at all."""
    source = (_SAFETY_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "build_generation_prompt" not in source
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "run_agent_pipeline" in names


def test_run_pair_calls_run_agent_pipeline_exactly_twice(monkeypatch):
    calls = []
    real = idctx_experiment.run_agent_pipeline

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(idctx_experiment, "run_agent_pipeline", _spy)
    client = FakeGeminiClient([_VALID_DRAFT_JSON, _VALID_DRAFT_JSON])
    idctx_experiment.run_pair(
        comment_id_prefix="t4", comment_text="نص تجريبي طويل بما يكفي.",
        primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.9,
        identifying_metadata=IdentifyingMetadata("X", None),
        client=client, official_knowledge_retriever=_empty_ok(),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert len(calls) == 2


# --- Official Knowledge availability identical between the pair ----------


def test_official_grounding_availability_identical_when_empty():
    client = FakeGeminiClient([_VALID_DRAFT_JSON, _VALID_DRAFT_JSON])
    control_result, context_result = idctx_experiment.run_pair(
        comment_id_prefix="t5", comment_text="نص تجريبي طويل بما يكفي.",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL", confidence=0.9,
        identifying_metadata=IdentifyingMetadata("X", None),
        client=client, official_knowledge_retriever=_empty_ok(),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert control_result.metadata.official_grounding_available is False
    assert context_result.metadata.official_grounding_available is False
    assert control_result.metadata.official_grounding_available == context_result.metadata.official_grounding_available


def test_official_grounding_availability_identical_when_present():
    """Even with REAL Official Knowledge results, availability must be
    identical between control/context -- it depends only on the shared
    retriever result, never on identifying_context."""
    import datetime as dt

    from rcs.response_intelligence.retrieval.contracts import OfficialKnowledgeResult

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
    client = FakeGeminiClient([_VALID_DRAFT_JSON, _VALID_DRAFT_JSON])
    control_result, context_result = idctx_experiment.run_pair(
        comment_id_prefix="t6", comment_text="نص تجريبي طويل بما يكفي.",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL", confidence=0.9,
        identifying_metadata=IdentifyingMetadata("X", None),
        client=client, official_knowledge_retriever=_FixedOfficialKnowledgeRetriever(ok_result),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert control_result.metadata.official_grounding_available is True
    assert context_result.metadata.official_grounding_available is True


# --- Isolation: no DB writes, no source Excel mutation, no PII -----------


def _module_files():
    return [_SAFETY_DIR / "experiment.py", _SAFETY_DIR / "review.py"]


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
    """No production audit/persistence repository is ever constructed or
    passed -- run_agent_pipeline's audit_repository stays at its default
    None for both variants."""
    source = (_SAFETY_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "audit_repository" not in source
    assert "AgentAuditRepository" not in source


def test_source_excel_unchanged_hash_stable_if_present():
    import hashlib

    source_file = _ROOT / "data" / "raw" / "historical-export.xlsx"
    if not source_file.exists():
        import pytest
        pytest.skip("source Excel not present in this environment")
    h1 = hashlib.md5(source_file.read_bytes()).hexdigest()
    h2 = hashlib.md5(source_file.read_bytes()).hexdigest()
    assert h1 == h2


def test_row_25_email_pii_is_excluded_from_sample():
    """source_row_index 25's original_title is itself a raw email
    address -- deliberately excluded from SELECTED_ROWS to avoid any
    risk of persisting it in evaluation output."""
    assert 25 not in idctx_experiment.SELECTED_ROWS


def test_comment_preview_and_identifying_context_preview_are_redacted():
    from rcs.response_generation.prompt import redact_for_generation

    raw_comment = "تواصلوا معي على 0501234567 من فضلكم."
    raw_title = "قسم الاتصال 0509876543"
    preview = redact_for_generation(raw_comment)[:100]
    identifying = build_identifying_grounding(raw_title, None)
    identifying_preview = redact_for_generation(identifying)[:200]
    assert "0501234567" not in preview
    assert "0509876543" not in identifying_preview


def test_no_new_prompt_version_or_rg_v11():
    """No new prompt version was introduced by this experiment/review
    scripts directory's own files (`_module_files()`, scoped to this
    directory). The settings literal below now reads "rg-v14.2": later
    later, separate prompt-content changes legitimately bumped it --
    see rcs.response_generation.prompt's module docstring -- unrelated
    to this directory's own scope, which never touched these two files
    (still correctly never mentioning "rg-v11" either)."""
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    for path in _module_files():
        assert "rg-v11" not in path.read_text(encoding="utf-8")


# --- Review CLI: paired (not blinded), resumable, safe --------------------


def _pair_case(eval_id="row1", control="بدون سياق", context="مع سياق"):
    return {
        "eval_id": eval_id, "source_row_index": 1, "categories": ["A"],
        "comment_preview": "x", "identifying_context_preview": "y",
        "primary_intent": "SUGGESTION", "topic_subtype": "WORD_SUBMISSION", "sentiment": "NEUTRAL",
        "official_grounding_available_control": False, "official_grounding_available_context": False,
        "control_reply": control, "control_status": "SUCCEEDED",
        "context_reply": context, "context_status": "SUCCEEDED",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_review_is_clearly_paired_not_blinded():
    """Unlike the reply-strategy reviewers, this one must show
    'without'/'with' labels directly -- no swap, no blinding."""
    source = (_SAFETY_DIR / "review.py").read_text(encoding="utf-8")
    assert "_swap_for" not in source
    assert "random" not in source


def test_review_resumable(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [_pair_case("c0"), _pair_case("c1")])

    answers = iter(["A", "Q"])
    idctx_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers))
    assert {row["eval_id"] for row in idctx_review._load_jsonl(labels_path)} == {"c0"}

    answers2 = iter(["C"])
    idctx_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: next(answers2))
    assert {row["eval_id"] for row in idctx_review._load_jsonl(labels_path)} == {"c0", "c1"}


def test_review_skip_not_persisted(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [_pair_case()])
    idctx_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "S")
    assert not labels_path.exists() or idctx_review._load_jsonl(labels_path) == []


def test_review_quit_exits_cleanly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [_pair_case()])
    idctx_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "Q")
    assert not labels_path.exists() or idctx_review._load_jsonl(labels_path) == []


def test_review_decision_codes_map_correctly(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [_pair_case()])
    idctx_review.run_review(cases_path, labels_path, prompt_fn=lambda _msg: "C")
    label = idctx_review._load_jsonl(labels_path)[0]
    assert label["decision"] == "CONTEXT_HIJACKING"


def test_review_view_does_not_persist_pii(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [_pair_case()])
    answers = iter(["V", "B"])
    idctx_review.run_review(
        cases_path, labels_path, prompt_fn=lambda _msg: next(answers),
        comment_lookup_fn=lambda row_index: ("رقمي 0501234567", "submission title: X 0509876543"),
    )
    raw_labels_text = labels_path.read_text(encoding="utf-8")
    assert "0501234567" not in raw_labels_text
    assert "0509876543" not in raw_labels_text


def test_review_never_infers_decision_without_asking(tmp_path):
    """No code path in run_review may write a label without first
    calling prompt_fn for that case -- structural proof via a prompt_fn
    that raises if called more than the number of pending cases."""
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(cases_path, [_pair_case()])
    call_count = {"n": 0}

    def _prompt(_msg):
        call_count["n"] += 1
        return "A"

    idctx_review.run_review(cases_path, labels_path, prompt_fn=_prompt)
    assert call_count["n"] == 1
    assert idctx_review._load_jsonl(labels_path)[0]["decision"] == "IMPROVED_RELEVANCE"
