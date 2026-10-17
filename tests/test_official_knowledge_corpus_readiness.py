"""
Tests for Official Knowledge curated-corpus integration readiness.

An analysis-only audit recommended the curated-JSON-corpus strategy
over an external dictionary API (no accessible credential -- confirmed 401 on the
one safe smoke GET) and over the one uploaded "المعجم الطلابي" file
(unverified provenance, a differently-named product from "معجم
الرياض"). This work makes the EXISTING governed-corpus architecture
(contracts.py, official_corpus
.py, official_knowledge.py, retrievers.py, evidence.py, orchestrator.py
-- none rewritten here) technically ready for a future organization-
approved export, without populating one:

  1. `scripts/build_official_knowledge_corpus.py` -- a one-time,
     offline converter from an approved export into the EXISTING,
     documented `RawOfficialSourceRecord` JSON schema. Never infers
     `authoritative`/`provenance_tag` from the input file itself --
     only from an explicit `--confirm-approved` operator flag.
  2. `tests/fixtures/official_knowledge_test_corpus.json` -- a small,
     unmistakably synthetic governed corpus (see its own content: every
     source_title is literally tagged "[SYNTHETIC TEST FIXTURE — NOT A
     REAL DICTIONARY]") used ONLY by this test file, never referenced
     by any settings/config default.

This file does not modify or re-test the CORE retrieval modules'
existing behavior (already covered by tests/test_official_knowledge.py)
-- it proves the REAL provider -> retriever -> evidence -> orchestrator
wiring against a real on-disk JSON file, which no existing test does
(test_official_knowledge.py deliberately never loads a real file --
see its own `test_no_production_data_fixture_loaded_by_this_test_file`).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS
from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES
from rcs.response_generation.service import GenerationRequest
from rcs.response_intelligence.identifying_context_policy import (
    REASON_NO_CORROBORATING_REFERENCE,
    IdentifyingMetadata,
    evaluate_identifying_context,
)
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    UNAVAILABLE,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeQuery,
)
from rcs.response_intelligence.retrieval.official_corpus import (
    JsonFileOfficialSourceProvider,
    RawOfficialSourceRecord,
    build_governed_official_corpus,
)
from rcs.response_intelligence.retrieval.official_knowledge import build_official_knowledge_retriever
from scripts.offline_eval.response_agent.fakes import FakeApprovedResponseMemoryRetriever, FakeGeminiClient

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _ROOT / "tests" / "fixtures" / "official_knowledge_test_corpus.json"
_SCRIPT_PATH = _ROOT / "scripts" / "build_official_knowledge_corpus.py"
_FIXTURE_NOW = dt.datetime(2026, 6, 1)
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)
_VALID_DRAFT_JSON = '{"suggested_response": "شكرًا لتواصلكم معنا ولاهتمامكم."}'


class FakeEmbeddingAdapter:
    """Mirrors tests/test_official_knowledge.py's own double exactly --
    not shared across files since this codebase has no shared
    test-utils module (same convention already established there)."""

    def __init__(self, vectors=None, dim=4):
        self._vectors = vectors or {}
        self._dim = dim
        self._loaded = False
        self.load_calls = 0
        self.embed_passages_calls = 0
        self.embed_query_calls = 0

    def load(self):
        self._loaded = True
        self.load_calls += 1

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def embedding_dimension(self):
        return self._dim if self._loaded else None

    def embed_passages(self, texts):
        self.embed_passages_calls += 1
        return np.stack([self._lookup(t) for t in texts])

    def embed_query(self, text):
        self.embed_query_calls += 1
        return self._lookup(text)

    def _lookup(self, text):
        for key, vec in self._vectors.items():
            if key in text:
                return vec
        return np.zeros(self._dim)


def _real_retriever_from_fixture(vectors=None):
    provider = JsonFileOfficialSourceProvider(str(_FIXTURE_PATH))
    embedder = FakeEmbeddingAdapter(vectors or {})
    retriever = build_official_knowledge_retriever(provider, embedder)
    retriever.clock = lambda: _FIXTURE_NOW
    return retriever, embedder


def _run_converter(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *args],
        capture_output=True, text=True, cwd=str(_ROOT),
    )


def _write_sample_export(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_excel(path, index=False)


_VALID_SAMPLE_ROWS = [
    {"الكلمة": "اختبارية-أ", "قسم الكلام": "اسم ذات", "التعريف": "تعريف اصطناعي أول.", "الشاهد": "شاهد اصطناعي أول."},
    {"الكلمة": "اختبارية-ب", "قسم الكلام": "اسم ذات", "التعريف": "تعريف اصطناعي ثانٍ.", "الشاهد": ""},
]


# --- 1-9. Converter / preparation tool ------------------------------------


def test_converter_normalizes_valid_input(tmp_path):
    input_path = tmp_path / "sample_export.xlsx"
    _write_sample_export(input_path, _VALID_SAMPLE_ROWS)
    proc = _run_converter(["--input", str(input_path), "--validate-only"])
    assert proc.returncode == 0
    summary = json.loads(proc.stdout)
    assert summary["input_rows"] == 2
    assert summary["valid_rows"] == 2
    assert summary["rejected_rows"] == 0
    assert summary["approved"] is False


def test_converter_fails_closed_without_confirm_approved(tmp_path):
    input_path = tmp_path / "sample_export.xlsx"
    output_path = tmp_path / "out.json"
    _write_sample_export(input_path, _VALID_SAMPLE_ROWS)
    proc = _run_converter([
        "--input", str(input_path), "--output", str(output_path),
        "--source-title", "معجم اختباري", "--source-type", "dictionary_entry",
        "--version", "v1", "--effective-from", "2026-01-01",
    ])
    assert proc.returncode != 0
    assert "confirm-approved" in proc.stderr
    assert not output_path.exists()


def test_converter_fails_closed_without_full_metadata_even_when_approved(tmp_path):
    input_path = tmp_path / "sample_export.xlsx"
    output_path = tmp_path / "out.json"
    _write_sample_export(input_path, _VALID_SAMPLE_ROWS)
    proc = _run_converter([
        "--input", str(input_path), "--output", str(output_path), "--confirm-approved",
    ])
    assert proc.returncode != 0
    assert "missing required metadata" in proc.stderr
    assert not output_path.exists()


def test_converter_never_infers_authoritative_from_filename_or_content(tmp_path):
    """A filename that says "المعجم" and "رسمي" (official) proves
    nothing by itself -- without --confirm-approved, records are never
    marked authoritative, no matter how official the file looks."""
    input_path = tmp_path / "معجم_الرياض_رسمي_معتمد.xlsx"
    _write_sample_export(input_path, _VALID_SAMPLE_ROWS)
    proc = _run_converter(["--input", str(input_path), "--validate-only"])
    assert proc.returncode == 0
    summary = json.loads(proc.stdout)
    assert summary["approved"] is False


def test_converter_rejects_missing_required_columns(tmp_path):
    input_path = tmp_path / "bad_columns.xlsx"
    _write_sample_export(input_path, [{"الكلمة": "س", "وصف": "عمود غير متوقع"}])
    proc = _run_converter(["--input", str(input_path), "--validate-only"])
    assert proc.returncode != 0
    assert "missing expected column" in proc.stderr


def test_converter_rejects_blank_content(tmp_path):
    input_path = tmp_path / "blank_content.xlsx"
    _write_sample_export(input_path, [
        {"الكلمة": "كلمة", "قسم الكلام": "اسم ذات", "التعريف": "", "الشاهد": ""},
    ])
    proc = _run_converter(["--input", str(input_path), "--validate-only"])
    summary = json.loads(proc.stdout)
    assert summary["valid_rows"] == 0
    assert summary["rejected_missing_content"] == 1
    assert summary["rejections"] == [{"row": 2, "reason": "missing_required_content"}]


def test_converter_rejects_duplicate_source_ids(tmp_path):
    input_path = tmp_path / "dup_ids.xlsx"
    _write_sample_export(input_path, [
        {"الكلمة": "أ", "معرف": "DUP-1", "التعريف": "تعريف أول.", "الشاهد": ""},
        {"الكلمة": "ب", "معرف": "DUP-1", "التعريف": "تعريف ثانٍ.", "الشاهد": ""},
    ])
    proc = _run_converter([
        "--input", str(input_path), "--validate-only", "--source-id-column", "معرف",
    ])
    summary = json.loads(proc.stdout)
    assert summary["valid_rows"] == 1
    assert summary["rejected_duplicate_source_id"] == 1


def test_converter_rejects_malformed_effective_dates(tmp_path):
    input_path = tmp_path / "bad_dates.xlsx"
    _write_sample_export(input_path, [
        {"الكلمة": "أ", "تاريخ_النفاذ": "not-a-date", "التعريف": "تعريف.", "الشاهد": ""},
    ])
    proc = _run_converter([
        "--input", str(input_path), "--validate-only", "--effective-from-column", "تاريخ_النفاذ",
    ])
    summary = json.loads(proc.stdout)
    assert summary["valid_rows"] == 0
    assert summary["rejected_invalid_effective_date"] == 1


def test_converter_summary_never_contains_row_content(tmp_path):
    """Governance/privacy requirement: the validation summary reports
    counts and row NUMBERS only -- never the actual cell text."""
    input_path = tmp_path / "sample_export.xlsx"
    _write_sample_export(input_path, [
        {"الكلمة": "سر", "قسم الكلام": "اسم ذات", "التعريف": "نص لا يجب أن يظهر في الملخص", "الشاهد": ""},
    ])
    proc = _run_converter(["--input", str(input_path), "--validate-only"])
    assert "نص لا يجب أن يظهر في الملخص" not in proc.stdout


def test_converter_output_matches_raw_official_source_record_contract(tmp_path):
    input_path = tmp_path / "sample_export.xlsx"
    output_path = tmp_path / "out.json"
    _write_sample_export(input_path, _VALID_SAMPLE_ROWS)
    proc = _run_converter([
        "--input", str(input_path), "--output", str(output_path), "--confirm-approved",
        "--source-title", "معجم اختباري", "--source-type", "dictionary_entry",
        "--version", "v1", "--effective-from", "2026-01-01",
    ])
    assert proc.returncode == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(written) == 2
    allowed_keys = {f.name for f in fields(RawOfficialSourceRecord)}
    for record in written:
        assert set(record.keys()) <= allowed_keys
        assert record["authoritative"] is True
        assert record["provenance_tag"] == "official_approved"


def test_json_file_provider_can_consume_generated_converter_output(tmp_path):
    """Real integration: converter output, loaded by the REAL
    JsonFileOfficialSourceProvider, passes the REAL governance check."""
    input_path = tmp_path / "sample_export.xlsx"
    output_path = tmp_path / "out.json"
    _write_sample_export(input_path, _VALID_SAMPLE_ROWS)
    proc = _run_converter([
        "--input", str(input_path), "--output", str(output_path), "--confirm-approved",
        "--source-title", "معجم اختباري", "--source-type", "dictionary_entry",
        "--version", "v1", "--effective-from", "2026-01-01",
    ])
    assert proc.returncode == 0

    provider = JsonFileOfficialSourceProvider(str(output_path))
    raw = provider.fetch_records()
    assert len(raw) == 2
    governed = build_governed_official_corpus(raw)
    assert len(governed) == 2
    assert {r.lemma for r in governed} == {"اختبارية-أ", "اختبارية-ب"}


# --- 10-16. Retrieval integration against the REAL on-disk fixture --------


def test_exact_entry_id_lookup_against_real_fixture_file():
    retriever, embedder = _real_retriever_from_fixture()
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="TEST-ENTRY-1", max_results=5))
    assert result.outcome == "SUCCESS_WITH_RESULTS"
    assert [r.source_id for r in result.results] == ["TEST-SYNTHETIC-OK-1"]
    assert embedder.embed_query_calls == 0


def test_exact_lemma_lookup_against_real_fixture_file():
    retriever, embedder = _real_retriever_from_fixture()
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="تجريبي-كلمة-ب", max_results=5))
    assert result.outcome == "SUCCESS_WITH_RESULTS"
    assert [r.source_id for r in result.results] == ["TEST-SYNTHETIC-OK-2"]
    assert embedder.embed_query_calls == 0


def test_semantic_fallback_against_real_fixture_file():
    unit = np.array([1.0, 0.0, 0.0, 0.0])
    retriever, embedder = _real_retriever_from_fixture(vectors={"SEMANTICKEY": unit, "استفسار حول SEMANTICKEY": unit})
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="استفسار حول SEMANTICKEY", max_results=5))
    assert result.outcome == "SUCCESS_WITH_RESULTS"
    assert result.results[0].source_id == "TEST-SYNTHETIC-SEMANTIC-1"
    assert embedder.embed_query_calls == 1


def test_multiple_currently_effective_senses_returned_deterministically():
    retriever, embedder = _real_retriever_from_fixture()
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="TEST-ENTRY-SHARED", max_results=5))
    assert result.outcome == "SUCCESS_WITH_RESULTS"
    assert [r.source_id for r in result.results] == ["TEST-SYNTHETIC-OK-3-SENSE-A", "TEST-SYNTHETIC-OK-4-SENSE-B"]
    assert embedder.embed_query_calls == 0


def test_not_yet_effective_record_excluded_from_real_fixture():
    """The retriever's semantic fallback has no minimum-similarity
    threshold (by design -- see official_knowledge.py's own tests): it
    still ranks and returns the closest OTHER currently-effective
    candidate when the queried key itself is filtered out. The
    invariant this proves is narrower and still correct: the
    not-yet-effective record's OWN result can never be one of them."""
    retriever, _ = _real_retriever_from_fixture()
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="TEST-ENTRY-NOT-YET-EFFECTIVE", max_results=10))
    assert "TEST-SYNTHETIC-NOT-YET-EFFECTIVE" not in [r.source_id for r in result.results]


def test_expired_record_excluded_from_real_fixture():
    retriever, _ = _real_retriever_from_fixture()
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="TEST-ENTRY-EXPIRED", max_results=10))
    assert "TEST-SYNTHETIC-EXPIRED" not in [r.source_id for r in result.results]


def test_non_authoritative_record_excluded_from_real_fixture():
    """authoritative=false is excluded at governance/corpus-build time
    -- it never becomes an OfficialCorpusRecord at all, so it can never
    appear in the index or any result, for any query."""
    retriever, _ = _real_retriever_from_fixture()
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="TEST-ENTRY-NOT-AUTHORITATIVE", max_results=10))
    assert "TEST-SYNTHETIC-NOT-AUTHORITATIVE" not in [r.source_id for r in result.results]


def test_non_approved_provenance_record_excluded_from_real_fixture():
    """Wrong provenance_tag is excluded at governance/corpus-build time
    -- same guarantee as the authoritative check above."""
    retriever, _ = _real_retriever_from_fixture()
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="TEST-ENTRY-NOT-APPROVED", max_results=10))
    assert "TEST-SYNTHETIC-NOT-APPROVED" not in [r.source_id for r in result.results]


def test_no_result_returns_success_empty_from_real_fixture():
    """A genuine "no result" case in the current (unmodified) retriever
    design is produced by the structured intent/topic filter emptying
    candidate_indices entirely -- see filtered_indices, checked BEFORE
    any exact/semantic matching. None of this fixture's records set
    intent, so an intent filter that requires one excludes every
    record, deterministically reaching SUCCESS_EMPTY without depending
    on any embedding score threshold (there isn't one -- see the known
    limitation noted in the final report)."""
    retriever, _ = _real_retriever_from_fixture()
    result = retriever.retrieve(
        OfficialKnowledgeQuery(query_text="أي نص عام", max_results=5, intent="COMPLAINT")
    )
    assert result.outcome == SUCCESS_EMPTY


def test_malformed_corpus_file_fails_closed_not_crash(tmp_path):
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not valid json", encoding="utf-8")
    provider = JsonFileOfficialSourceProvider(str(malformed_path))
    embedder = FakeEmbeddingAdapter()
    retriever = build_official_knowledge_retriever(provider, embedder)
    result = retriever.retrieve(OfficialKnowledgeQuery(query_text="أي نص", max_results=5))
    assert result.outcome == UNAVAILABLE
    assert result.results == ()


# --- 17-22. Agent end-to-end (real provider + real retriever + FakeGeminiClient) --


def _agent_req(comment_text: str, **overrides) -> GenerationRequest:
    # primary_intent/topic_subtype default to None deliberately: the
    # fixture's records carry no topic/intent (a real curated export
    # need not set them either -- both are optional structured filters
    # per contracts.py), and the orchestrator forwards whatever
    # classification signals it was given straight into
    # OfficialKnowledgeQuery's structured intent/topic filter, which
    # runs BEFORE any exact/semantic matching -- a non-None value here
    # that the fixture never matches would exclude every candidate.
    base = dict(
        comment_id="x", comment_text=comment_text, primary_intent=None,
        topic_subtype=None, sentiment="NEUTRAL", confidence=0.5,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def test_agent_e2e_trusted_result_reaches_official_knowledge_prompt_block():
    retriever, _ = _real_retriever_from_fixture(vectors={
        "SEMANTICKEY": np.array([1.0, 0.0, 0.0, 0.0]),
        "هل يمكنكم توضيح معنى SEMANTICKEY؟": np.array([1.0, 0.0, 0.0, 0.0]),
    })
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    result = run_agent_pipeline(
        _agent_req("هل يمكنكم توضيح معنى SEMANTICKEY؟"),
        client=client,
        official_knowledge_retriever=retriever,
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert result.metadata.official_grounding_available is True
    assert result.metadata.grounding_mode == "FACTUAL"
    # No relevance threshold exists in the current (unmodified) ranker
    # -- with no topic/intent filter it returns up to
    # OFFICIAL_KNOWLEDGE_MAX_RESULTS_PER_QUERY candidates, ranked by
    # score. The genuine semantic match must rank first.
    assert result.metadata.official_result_count >= 1
    assert "OFFICIAL KNOWLEDGE" in client.prompts[0]
    assert "محتوى اصطناعي يحتوي على رمز الاختبار SEMANTICKEY" in client.prompts[0]
    assert "(none — do not invent facts)" not in client.prompts[0]


def test_agent_e2e_no_result_never_invents_facts():
    """Genuine "no result" through the real orchestrator path: the
    fixture's records set no `intent`, so a request classified with a
    primary_intent the corpus never matches empties
    `filtered_indices` BEFORE any exact/semantic ranking runs -- see
    the retriever-level test's docstring for why an unrelated query
    text alone does not reach SUCCESS_EMPTY in the current (unmodified)
    ranker."""
    retriever, _ = _real_retriever_from_fixture()
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    result = run_agent_pipeline(
        _agent_req("عبارة عامة لا ترتبط بأي سجل اختباري في المتن", primary_intent="COMPLAINT"),
        client=client,
        official_knowledge_retriever=retriever,
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert result.metadata.official_grounding_available is False
    assert result.metadata.grounding_mode == "IDENTIFYING_ONLY"
    assert "(none — do not invent facts)" in client.prompts[0]


def test_agent_e2e_unavailable_corpus_does_not_crash_and_stays_unavailable(tmp_path):
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("not json at all", encoding="utf-8")
    provider = JsonFileOfficialSourceProvider(str(malformed_path))
    retriever = build_official_knowledge_retriever(provider, FakeEmbeddingAdapter())
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    result = run_agent_pipeline(
        _agent_req("أي تعليق عادي من زائر"),
        client=client,
        official_knowledge_retriever=retriever,
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert result.metadata.official_grounding_available is False
    assert result.metadata.official_retrieval_outcome == UNAVAILABLE
    assert result.generation_result.generation_status == "SUCCEEDED"
    assert "(none — do not invent facts)" in client.prompts[0]


def test_no_additional_gemini_call_for_official_knowledge_retrieval():
    retriever, _ = _real_retriever_from_fixture(vectors={
        "SEMANTICKEY": np.array([1.0, 0.0, 0.0, 0.0]),
        "سؤال عن SEMANTICKEY": np.array([1.0, 0.0, 0.0, 0.0]),
    })
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    run_agent_pipeline(
        _agent_req("سؤال عن SEMANTICKEY"),
        client=client,
        official_knowledge_retriever=retriever,
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert client.call_count == 1


def test_rg_v11_unchanged():
    """Value updated to "rg-v14": later, separate prompt-content
    change bumped it for production traceability -- see
    rcs.response_generation.prompt's module docstring. This file's own
    point (Official Knowledge readiness work never touched the prompt
    version) is unaffected."""
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"


def test_identifying_context_policy_unchanged():
    decision = evaluate_identifying_context(
        comment_text="أحسنتم", identifying_metadata=IdentifyingMetadata("إِطَاعَة", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_ambiguous_intent_gate_still_present():
    assert "unsupported_visitor_intent_claim" in REFINEMENT_ELIGIBLE_REASON_CODES
    assert "Do not attribute a specific action, request, or proposal to the visitor" in SYSTEM_INSTRUCTIONS


def test_no_db_dependency_introduced_by_new_script():
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    for banned in ("sqlalchemy", "pymongo", "rcs.review.models", "psycopg"):
        assert banned not in source


def test_new_script_never_calls_gemini_or_orchestrator():
    """Checks actual import/call shapes, not the bare word "orchestrator"
    -- the module's own docstring legitimately mentions it in prose
    (describing why the tool exists), which is not a wiring concern."""
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    for banned in (
        "google.genai", "genai_client", "GenerationRequest(", "run_agent_pipeline(",
        "from rcs.response_intelligence.orchestrator",
    ):
        assert banned not in source


def test_fixture_is_unmistakably_synthetic():
    text = _FIXTURE_PATH.read_text(encoding="utf-8")
    assert "SYNTHETIC TEST FIXTURE" in text
    assert "NOT A REAL DICTIONARY" in text


def test_fixture_not_referenced_by_any_settings_default():
    assert settings.OFFICIAL_KNOWLEDGE_SOURCE_PATH is None
    settings_source = (_ROOT / "config" / "settings.py").read_text(encoding="utf-8")
    assert "official_knowledge_test_corpus" not in settings_source
    env_example = (_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "official_knowledge_test_corpus" not in env_example
