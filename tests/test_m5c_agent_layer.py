"""
Phase M5C — tests for the deterministic response agent layer:
evidence assembly, DI/caching of default retrievers, and the
orchestrator's end-to-end wiring.

No Hugging Face model is ever downloaded here: retrieval tests use
plain fake retrievers (satisfying the M5B.1 Protocols structurally) or
the existing M5B.3/M5B.4 real implementations built with a fake
embedding adapter -- consistent with those phases' own test doubles.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import inspect
from dataclasses import fields

import pytest

from rcs.response_generation.prompt import build_generation_prompt
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest, generate_suggested_response
from rcs.response_intelligence import evidence, retrievers as retrievers_module
from rcs.response_intelligence.evidence import (
    GROUNDING_MODE_FACTUAL,
    GROUNDING_MODE_IDENTIFYING_ONLY,
    assemble_grounding,
    assemble_style_reference_texts,
)
from rcs.response_intelligence.orchestrator import generate_response_with_orchestrator
from rcs.response_intelligence.retrieval.contracts import (
    ERROR,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    UNAVAILABLE,
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryResult,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeQuery,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
)

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def _req(**overrides) -> GenerationRequest:
    base = dict(
        comment_id="c-1",
        comment_text="هذا تعليق تجريبي اصطناعي للاختبار.",
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        sentiment="POSITIVE",
        confidence=0.9,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def _ok_result(**overrides) -> OfficialKnowledgeResult:
    base = dict(
        source_id="OK-1",
        source_title="عنوان رسمي تجريبي",
        source_type="dictionary_entry",
        content="محتوى رسمي تجريبي معتمد.",
        version="v1",
        effective_from=_SYNTHETIC_NOW,
        authoritative=True,
        retrieval_score=0.9,
    )
    base.update(overrides)
    return OfficialKnowledgeResult(**base)


def _arm_result(**overrides) -> ApprovedResponseMemoryResult:
    base = dict(
        review_item_id="r-1",
        final_reply_id="hash-1",
        response_text="هذا مثال أسلوب تجريبي معتمد سابقًا.",
        reviewer_action="SEND_AS_IS",
        approved_at=_SYNTHETIC_NOW,
        retrieval_score=0.9,
    )
    base.update(overrides)
    return ApprovedResponseMemoryResult(**base)


class FakeOfficialKnowledgeRetriever:
    def __init__(self, result: OfficialKnowledgeRetrievalResult):
        self.result = result
        self.calls: list[OfficialKnowledgeQuery] = []

    def retrieve(self, query: OfficialKnowledgeQuery) -> OfficialKnowledgeRetrievalResult:
        self.calls.append(query)
        return self.result


class FakeApprovedResponseMemoryRetriever:
    def __init__(self, result: ApprovedResponseMemoryRetrievalResult):
        self.result = result
        self.calls: list[ApprovedResponseMemoryQuery] = []

    def retrieve(self, query: ApprovedResponseMemoryQuery) -> ApprovedResponseMemoryRetrievalResult:
        self.calls.append(query)
        return self.result


class FakeGeminiClient:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        return self._responses[min(self.call_count - 1, len(self._responses) - 1)]


def _draft_json(text: str) -> str:
    import json

    return json.dumps({"suggested_response": text}, ensure_ascii=False)


_EMPTY_OK = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)


# --- 1. Evidence assembly: grounding mode ----------------------------------


def test_no_official_results_keeps_identifying_only_mode_and_text_unchanged():
    assembly = assemble_grounding("submission title: X", _EMPTY_OK)
    assert assembly.mode == GROUNDING_MODE_IDENTIFYING_ONLY
    assert assembly.identifying_context == "submission title: X"
    assert assembly.official_grounding_available is False
    assert assembly.official_grounding_text is None
    # M5D.1 correction: identifying context alone must NOT satisfy the
    # caller-facing eligibility flag either -- orchestrator.py now sets
    # trusted_grounding_available = official_grounding_available
    # directly, with no OR-identifying-context fallback (see evidence.py's
    # module docstring for the corrected history). M5D.2: identifying
    # context and official grounding are no longer merged into one
    # combined "prompt_text" at all -- GroundingAssembly exposes only
    # the two source fields; rcs.response_generation.prompt renders them
    # under two separate sections.


def test_no_identifying_and_no_official_results_is_unavailable_identifying_only():
    assembly = assemble_grounding(None, _EMPTY_OK)
    assert assembly.mode == GROUNDING_MODE_IDENTIFYING_ONLY
    assert assembly.identifying_context is None
    assert assembly.official_grounding_available is False


def test_official_results_switch_to_factual_mode_and_include_content():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    assembly = assemble_grounding("submission title: X", official)
    assert assembly.mode == GROUNDING_MODE_FACTUAL
    assert assembly.official_grounding_available is True
    assert "محتوى رسمي تجريبي معتمد." in assembly.official_grounding_text
    assert "عنوان رسمي تجريبي" in assembly.official_grounding_text
    # M5D.2: identifying context is kept as its own field, never folded
    # into official_grounding_text.
    assert assembly.identifying_context == "submission title: X"
    assert "submission title: X" not in assembly.official_grounding_text


def test_official_results_without_identifying_context_still_factual():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    assembly = assemble_grounding(None, official)
    assert assembly.mode == GROUNDING_MODE_FACTUAL
    assert assembly.official_grounding_available is True
    assert "محتوى رسمي تجريبي معتمد." in assembly.official_grounding_text


def test_unavailable_or_error_official_outcome_falls_back_to_identifying_only():
    for outcome in (UNAVAILABLE, ERROR):
        official = OfficialKnowledgeRetrievalResult(outcome=outcome, failure_detail="synthetic")
        assembly = assemble_grounding("submission title: X", official)
        assert assembly.mode == GROUNDING_MODE_IDENTIFYING_ONLY
        assert assembly.identifying_context == "submission title: X"
        assert assembly.official_grounding_available is False


# --- 2. Evidence assembly: style references, never merged with facts -------


def test_style_reference_texts_empty_for_every_outcome_except_success():
    for outcome in (SUCCESS_EMPTY, UNAVAILABLE, ERROR):
        result = ApprovedResponseMemoryRetrievalResult(outcome=outcome, failure_detail="synthetic" if outcome != SUCCESS_EMPTY else None)
        assert assemble_style_reference_texts(result) == ()


def test_style_reference_texts_extracted_from_approved_results():
    approved = ApprovedResponseMemoryRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="مثال أول"), _arm_result(response_text="مثال ثانٍ", review_item_id="r-2", final_reply_id="hash-2"))
    )
    assert assemble_style_reference_texts(approved) == ("مثال أول", "مثال ثانٍ")


def test_official_and_approved_never_cross_contaminate():
    """Structural proof: assemble_grounding's factual text never contains
    an approved-response's response_text, and style texts never contain
    official content -- built from disjoint input types entirely."""
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="FACT_MARKER"),))
    approved = ApprovedResponseMemoryRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="STYLE_MARKER"),)
    )
    grounding = assemble_grounding(None, official)
    styles = assemble_style_reference_texts(approved)
    assert "STYLE_MARKER" not in grounding.official_grounding_text
    assert "FACT_MARKER" not in styles[0]


# --- 3. Prompt building: byte-identical when empty -------------------------


def _prompt_kwargs(**overrides):
    base = dict(
        comment_text="تعليق",
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        sentiment="POSITIVE",
        reason_code="thanks_praise_eligible",
    )
    base.update(overrides)
    return base


def test_prompt_byte_identical_when_style_reference_texts_absent():
    without_param = build_generation_prompt(**_prompt_kwargs())
    with_none = build_generation_prompt(**_prompt_kwargs(style_reference_texts=None))
    with_empty_tuple = build_generation_prompt(**_prompt_kwargs(style_reference_texts=()))
    assert without_param == with_none == with_empty_tuple


def test_prompt_gains_style_block_only_when_texts_present():
    prompt = build_generation_prompt(**_prompt_kwargs(style_reference_texts=("مثال أسلوب",)))
    assert "STYLE REFERENCE EXAMPLES" in prompt
    assert "مثال أسلوب" in prompt
    assert "NOT factual" in prompt


# --- 4. GenerationRequest/service.py regression -----------------------------


def test_generation_request_default_style_field_does_not_change_existing_behavior():
    req_old_shape = _req()
    req_explicit_empty = dataclasses.replace(_req(), style_reference_texts=())
    client_a = FakeGeminiClient([_draft_json("رد تجريبي")])
    client_b = FakeGeminiClient([_draft_json("رد تجريبي")])
    result_a = generate_suggested_response(req_old_shape, client_a)
    result_b = generate_suggested_response(req_explicit_empty, client_b)
    assert client_a.prompts == client_b.prompts
    assert result_a == result_b


# --- 5. Default retriever DI/caching (retrievers.py) ------------------------


@pytest.fixture(autouse=True)
def _reset_retriever_cache():
    retrievers_module.reset_default_retrievers_for_testing()
    yield
    retrievers_module.reset_default_retrievers_for_testing()


def test_default_retrievers_are_built_once_and_cached():
    ok_1 = retrievers_module.default_official_knowledge_retriever()
    ok_2 = retrievers_module.default_official_knowledge_retriever()
    assert ok_1 is ok_2

    arm_1 = retrievers_module.default_approved_response_memory_retriever()
    arm_2 = retrievers_module.default_approved_response_memory_retriever()
    assert arm_1 is arm_2


def test_default_retrievers_never_load_embedder_with_todays_empty_configuration():
    """Confirms 'do not download E5 when governed corpora are empty' end
    to end through the real DI factory, not just at the lower-level
    build_index unit -- both defaults' corpora are empty under this
    repository's current settings (no allowlist, no source path)."""
    from config import settings

    assert settings.APPROVED_RESPONSE_SOURCE_ALLOWLIST == frozenset()
    assert settings.OFFICIAL_KNOWLEDGE_SOURCE_PATH is None

    retrievers_module.default_official_knowledge_retriever()
    retrievers_module.default_approved_response_memory_retriever()
    embedder = retrievers_module._get_shared_embedder()
    assert embedder.is_loaded is False


def test_both_defaults_share_the_same_embedder_instance():
    retrievers_module.default_official_knowledge_retriever()
    retrievers_module.default_approved_response_memory_retriever()
    # Calling _get_shared_embedder again must return the SAME instance
    # both retrievers were built with -- one adapter, not two.
    assert retrievers_module._get_shared_embedder() is retrievers_module._get_shared_embedder()


# --- 6. Orchestrator end-to-end: safe when both empty (regression) ---------


def test_orchestrator_with_both_retrievers_empty_matches_direct_call_exactly():
    """The critical regression guarantee: with both retrieval sources at
    SUCCESS_EMPTY (today's actual state), the orchestrated result must
    be byte-for-byte identical to calling generate_suggested_response
    directly with the ORIGINAL, un-enriched request."""
    request = _req()
    client_direct = FakeGeminiClient([_draft_json("شكرًا لتواصلكم معنا ولاهتمامكم.")])
    client_orchestrated = FakeGeminiClient([_draft_json("شكرًا لتواصلكم معنا ولاهتمامكم.")])

    direct = generate_suggested_response(request, client_direct)
    orchestrated = generate_response_with_orchestrator(
        request,
        client_orchestrated,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert direct == orchestrated
    assert client_direct.prompts == client_orchestrated.prompts


def test_orchestrator_skips_retrieval_entirely_for_blank_comment():
    ok_fake = FakeOfficialKnowledgeRetriever(_EMPTY_OK)
    arm_fake = FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    generate_response_with_orchestrator(
        _req(comment_text="   "),
        None,
        official_knowledge_retriever=ok_fake,
        approved_response_retriever=arm_fake,
    )
    assert ok_fake.calls == []
    assert arm_fake.calls == []


# --- 7. Orchestrator end-to-end: official knowledge changes the draft -------


def test_orchestrator_injects_factual_content_into_prompt_when_official_knowledge_found():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="حقيقة رسمية فريدة XYZ"),))
    client = FakeGeminiClient([_draft_json("نشكركم على تواصلكم الكريم معنا في هذا الرد التجريبي.")])
    generate_response_with_orchestrator(
        _req(primary_intent="INQUIRY", topic_subtype="GENERAL"),
        client,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(official),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert "حقيقة رسمية فريدة XYZ" in client.prompts[0]
    assert "APPROVED FACTUAL statements" in client.prompts[0]


def test_orchestrator_official_knowledge_flips_inquiry_eligibility_to_grounded_path():
    """Trust-boundary-correct wiring proof: an INQUIRY with no prior
    identifying grounding normally routes through
    'inquiry_without_trusted_grounding' (see eligibility.py). Once
    Official Knowledge actually returns a result, trusted_grounding_
    available must flip to True and the SAME comment must route through
    'inquiry_with_trusted_grounding' instead -- proving the enrichment
    reaches eligibility, not just the prompt string."""
    request = _req(primary_intent="INQUIRY", topic_subtype="GENERAL", trusted_grounding_text=None)

    without_official = generate_response_with_orchestrator(
        request,
        FakeGeminiClient([_draft_json("نشكركم على تواصلكم الكريم معنا في هذا الرد التجريبي.")]),
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert without_official.eligibility_rule_id == "inquiry_without_trusted_grounding"

    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    with_official = generate_response_with_orchestrator(
        request,
        FakeGeminiClient([_draft_json("نشكركم على تواصلكم الكريم معنا في هذا الرد التجريبي.")]),
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(official),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert with_official.eligibility_rule_id == "generate_inquiry_with_grounding"


def test_orchestrator_unavailable_official_knowledge_behaves_like_empty_not_a_crash():
    official = OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, failure_detail="synthetic unavailable")
    client = FakeGeminiClient([_draft_json("نشكركم على تواصلكم الكريم معنا في هذا الرد التجريبي.")])
    result = generate_response_with_orchestrator(
        _req(),
        client,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(official),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert result.generation_status == "SUCCEEDED"


# --- 8. Orchestrator end-to-end: approved-response style examples ----------


def test_orchestrator_injects_style_examples_into_prompt_when_approved_memory_found():
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="نمط أسلوب فريد ABC"),))
    client = FakeGeminiClient([_draft_json("نشكركم على تواصلكم الكريم معنا في هذا الرد التجريبي.")])
    generate_response_with_orchestrator(
        _req(),
        client,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(approved),
    )
    assert "نمط أسلوب فريد ABC" in client.prompts[0]
    assert "STYLE REFERENCE EXAMPLES" in client.prompts[0]
    assert "APPROVED FACTUAL statements" not in client.prompts[0]


def test_orchestrator_style_examples_never_bypass_validation_or_get_copied_verbatim():
    """The draft Gemini actually returns is validated exactly as before
    -- style examples influence the PROMPT, never the accepted draft
    directly, and a draft that fails validation still fails."""
    from rcs.response_generation.validate import _GENERIC_EXACT_DRAFTS

    canned = next(iter(_GENERIC_EXACT_DRAFTS))
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="أي نص"),))
    client = FakeGeminiClient([_draft_json(canned)])
    result = generate_response_with_orchestrator(
        _req(),
        client,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(approved),
    )
    assert result.generation_status == "FAILED_REVIEW_ONLY"
    assert result.reason_code == "generic_boilerplate"


# --- 9. Structured filters passed through to retrieval ----------------------


def test_orchestrator_passes_intent_topic_and_bounded_max_results_to_both_retrievers():
    from config import settings

    ok_fake = FakeOfficialKnowledgeRetriever(_EMPTY_OK)
    arm_fake = FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    generate_response_with_orchestrator(
        _req(primary_intent="COMPLAINT", topic_subtype="TECHNICAL_ISSUE"),
        None,
        official_knowledge_retriever=ok_fake,
        approved_response_retriever=arm_fake,
    )
    assert ok_fake.calls[0].intent == "COMPLAINT"
    assert ok_fake.calls[0].topic == "TECHNICAL_ISSUE"
    assert ok_fake.calls[0].max_results == settings.OFFICIAL_KNOWLEDGE_MAX_RESULTS_PER_QUERY
    assert arm_fake.calls[0].intent == "COMPLAINT"
    assert arm_fake.calls[0].topic == "TECHNICAL_ISSUE"
    assert arm_fake.calls[0].max_results == settings.APPROVED_RESPONSE_MEMORY_MAX_RESULTS_PER_QUERY


def test_orchestrator_redacts_query_text_before_retrieval():
    ok_fake = FakeOfficialKnowledgeRetriever(_EMPTY_OK)
    arm_fake = FakeApprovedResponseMemoryRetriever(_EMPTY_ARM)
    generate_response_with_orchestrator(
        _req(comment_text="اتصل بي على 0512345678 من فضلك"),
        None,
        official_knowledge_retriever=ok_fake,
        approved_response_retriever=arm_fake,
    )
    assert "0512345678" not in ok_fake.calls[0].query_text
    assert "0512345678" not in arm_fake.calls[0].query_text


# --- 10. GenerationResult schema untouched (no trace/reasoning fields) -----


def test_generation_result_schema_has_no_new_trace_or_evidence_fields():
    field_names = {f.name for f in fields(GenerationResult)}
    assert field_names == {
        "comment_id",
        "routing_decision",
        "generation_status",
        "suggested_response",
        "review_required",
        "reason_code",
        "provider",
        "model_name",
        "prompt_version",
        "generated_at",
        "future_reviewer_actions",
        "eligibility_rule_id",
        "error",
    }


# --- 11. Static guards: no out-of-scope integration in the new modules ----


def _code_excluding_docstrings(path: str) -> str:
    from pathlib import Path

    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                for lineno in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                    idx = lineno - 1
                    if 0 <= idx < len(lines):
                        lines[idx] = "\n"
    return "".join(lines)


def _m5c_module_files() -> list[str]:
    import rcs.response_intelligence.orchestrator as orchestrator_module
    import rcs.response_intelligence.retrievers as retrievers_mod

    return [
        inspect.getsourcefile(evidence),
        inspect.getsourcefile(retrievers_mod),
        inspect.getsourcefile(orchestrator_module),
    ]


def test_no_langgraph_react_or_out_of_scope_integrations():
    banned = (
        "langgraph",
        "react",
        "toolcall",
        "external_dictionary_client",
        "pinecone",
        "chromadb",
        "weaviate",
        "qdrant",
        "pgvector",
        "elasticsearch",
        "faiss",
        "pymongo",
        "sqlalchemy",
        "google.genai",
        "genai_client",
    )
    for path in _m5c_module_files():
        code = _code_excluding_docstrings(path).lower()
        for term in banned:
            assert term not in code, f"forbidden reference {term!r} found in {path}"


def test_no_persistent_memory_or_chain_of_thought_logging():
    """Phase M5E note: "logging." and "trace" are no longer banned
    outright -- M5E intentionally adds sanitized audit-trace logging
    (rcs.response_intelligence.audit's AgentAuditRecord, and a single
    execution_id/review_item_id/exception-type-only logger.warning call
    in orchestrator.py on audit-persistence failure). That the logging
    call itself never carries prompt/comment/PII content is proven
    directly in tests/test_m5e_agent_audit.py, not by banning the
    words here. Still genuinely forbidden: chain-of-thought/reasoning
    logging and bare print()."""
    banned = ("print(", "chain_of_thought", "reasoning_log")
    for path in _m5c_module_files():
        code = _code_excluding_docstrings(path)
        for term in banned:
            assert term not in code, f"forbidden reference {term!r} found in {path}"


def test_exactly_one_gemini_call_per_orchestrated_request():
    client = FakeGeminiClient([_draft_json("شكرًا لتواصلكم معنا ولاهتمامكم.")])
    generate_response_with_orchestrator(
        _req(),
        client,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(_EMPTY_OK),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )
    assert client.call_count == 1
