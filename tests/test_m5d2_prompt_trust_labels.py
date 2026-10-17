"""
Phase M5D.2 — prompt trust-label closure.

M5D.1 corrected ELIGIBILITY (only Official Knowledge sets
`trusted_grounding_available`), but the PROMPT text itself still
concatenated identifying context and official evidence into one block
labeled "TRUSTED GROUNDING" -- misleading whenever that block held only
identifying context. This phase splits the prompt into three honestly
labeled, never-merged sections: IDENTIFYING CONTEXT, OFFICIAL KNOWLEDGE,
and (from M5C) STYLE REFERENCE EXAMPLES.
"""

from __future__ import annotations

import pytest
import datetime as dt
import json
from dataclasses import fields

from rcs.response_generation.prompt import build_generation_prompt, build_identifying_grounding
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest, generate_suggested_response
from rcs.response_generation.validate import validate_draft
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryQuery,
    ApprovedResponseMemoryResult,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeQuery,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
)

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)
_EMPTY_OK = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)
_VALID_DRAFT = "شكرًا لملاحظتكم القيّمة حول هذا الموضوع."
_TOO_SHORT_DRAFT = "شكرًا"


pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def _req(**overrides) -> GenerationRequest:
    base = dict(
        comment_id="c-1",
        comment_text="كيف يمكنني اقتراح كلمة جديدة للمعجم؟",
        primary_intent="INQUIRY",
        topic_subtype="GENERAL",
        sentiment="NEUTRAL",
        confidence=0.9,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def _ok_result(**overrides) -> OfficialKnowledgeResult:
    base = dict(
        source_id="OK-1", source_title="عنوان رسمي", source_type="dictionary_entry",
        content="محتوى رسمي معتمد.", version="v1", effective_from=_SYNTHETIC_NOW,
        authoritative=True, retrieval_score=0.9,
    )
    base.update(overrides)
    return OfficialKnowledgeResult(**base)


def _arm_result(**overrides) -> ApprovedResponseMemoryResult:
    base = dict(
        review_item_id="r-1", final_reply_id="hash-1", response_text="مثال أسلوب معتمد.",
        reviewer_action="SEND_AS_IS", approved_at=_SYNTHETIC_NOW, retrieval_score=0.9,
    )
    base.update(overrides)
    return ApprovedResponseMemoryResult(**base)


class FakeOfficialKnowledgeRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query: OfficialKnowledgeQuery):
        return self.result


class FakeApprovedResponseMemoryRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query: ApprovedResponseMemoryQuery):
        return self.result


class FakeGeminiClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        return self._responses[min(self.call_count - 1, len(self._responses) - 1)]


def _draft_json(text: str) -> str:
    return json.dumps({"suggested_response": text}, ensure_ascii=False)


def _run(request, ok_result=_EMPTY_OK, arm_result=_EMPTY_ARM, responses=(_draft_json(_VALID_DRAFT),)):
    client = FakeGeminiClient(list(responses))
    exec_result = run_agent_pipeline(
        request, client, 0,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(ok_result),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(arm_result),
    )
    return exec_result, client


# --- 1-4: identifying context lives in its own section, never Official Knowledge


def test_1_identifying_context_appears_under_its_own_section():
    """Phase M6.2: identifying context is policy-gated -- the comment
    must corroborate the title. See
    rcs.response_intelligence.identifying_context_policy."""
    _exec, client = _run(_req(
        comment_text="لدي استفسار حول «عنوان فريد ABC» بالتحديد.",
        identifying_metadata=IdentifyingMetadata("عنوان فريد ABC", "ENTRY-XYZ"),
    ))
    assert "IDENTIFYING CONTEXT" in client.prompts[0]
    assert "عنوان فريد ABC" in client.prompts[0]


def test_2_title_never_placed_in_official_knowledge_section():
    identifying = build_identifying_grounding("عنوان فريد ABC", None)
    _exec, client = _run(_req(trusted_grounding_text=identifying))
    prompt = client.prompts[0]
    official_section = prompt.split("--- OFFICIAL KNOWLEDGE")[1].split("--- END OFFICIAL KNOWLEDGE")[0]
    assert "عنوان فريد ABC" not in official_section


def test_3_entry_id_never_placed_in_official_knowledge_section():
    identifying = build_identifying_grounding(None, "ENTRY-XYZ-UNIQUE")
    _exec, client = _run(_req(trusted_grounding_text=identifying))
    prompt = client.prompts[0]
    official_section = prompt.split("--- OFFICIAL KNOWLEDGE")[1].split("--- END OFFICIAL KNOWLEDGE")[0]
    assert "ENTRY-XYZ-UNIQUE" not in official_section


def test_4_lemma_shaped_text_never_placed_in_official_knowledge_section():
    """Phase M6.2: title corroborated by the comment so the policy
    allows it through -- proving even this lemma-shaped value never
    lands in the OFFICIAL KNOWLEDGE section once rendered."""
    lemma_shaped_title = "كلمة-فريدة-XYZ"
    _exec, client = _run(_req(
        comment_text="سؤال حول كلمة-فريدة-XYZ تحديدًا.",
        identifying_metadata=IdentifyingMetadata(lemma_shaped_title, None),
    ))
    prompt = client.prompts[0]
    official_section = prompt.split("--- OFFICIAL KNOWLEDGE")[1].split("--- END OFFICIAL KNOWLEDGE")[0]
    assert "كلمة-فريدة-XYZ" not in official_section
    assert "IDENTIFYING CONTEXT" in prompt
    assert "كلمة-فريدة-XYZ" in prompt


# --- 5: identifying context never labeled trusted/official/authoritative --


def test_5_identifying_context_never_labeled_factual_authority():
    _exec, client = _run(_req(
        comment_text="سؤال حول هذا العنوان تحديدًا.",
        identifying_metadata=IdentifyingMetadata("العنوان", "ENTRY-1"),
    ))
    prompt = client.prompts[0]
    identifying_section = prompt.split("--- IDENTIFYING CONTEXT")[1].split("--- END IDENTIFYING CONTEXT")[0]
    for banned in ("TRUSTED GROUNDING", "OFFICIAL KNOWLEDGE", "factual evidence", "authoritative"):
        assert banned not in identifying_section


# --- 6-7: Official Knowledge section content ---------------------------------


def test_6_official_knowledge_results_appear_only_in_official_section():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="حقيقة UNIQUE_FACT_123"),))
    _exec, client = _run(_req(), ok_result=official)
    prompt = client.prompts[0]
    assert "UNIQUE_FACT_123" in prompt
    official_section = prompt.split("--- OFFICIAL KNOWLEDGE")[1].split("--- END OFFICIAL KNOWLEDGE")[0]
    assert "UNIQUE_FACT_123" in official_section


def test_7_official_section_shows_none_placeholder_when_success_empty():
    _exec, client = _run(_req(), ok_result=_EMPTY_OK)
    prompt = client.prompts[0]
    official_section = prompt.split("--- OFFICIAL KNOWLEDGE")[1].split("--- END OFFICIAL KNOWLEDGE")[0]
    assert "(none — do not invent facts)" in official_section


# --- 8-9: style examples stay separate ---------------------------------------


def test_8_style_examples_remain_separately_labeled_non_factual():
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="نمط UNIQUE_STYLE_456"),))
    _exec, client = _run(_req(), arm_result=approved)
    prompt = client.prompts[0]
    assert "STYLE REFERENCE EXAMPLES" in prompt
    assert "NOT factual" in prompt
    assert "UNIQUE_STYLE_456" in prompt


def test_9_style_examples_never_appear_in_official_knowledge_section():
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="نمط UNIQUE_STYLE_789"),))
    _exec, client = _run(_req(), arm_result=approved)
    prompt = client.prompts[0]
    official_section = prompt.split("--- OFFICIAL KNOWLEDGE")[1].split("--- END OFFICIAL KNOWLEDGE")[0]
    assert "UNIQUE_STYLE_789" not in official_section


# --- 10-11: both initial and refinement prompts preserve the 3 trust classes


def test_10_initial_prompt_preserves_three_trust_classes():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="حقيقة A"),))
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="نمط B"),))
    _exec, client = _run(
        _req(
            comment_text="سؤال حول هذا العنوان تحديدًا.",
            identifying_metadata=IdentifyingMetadata("العنوان", "ENTRY-1"),
        ),
        ok_result=official, arm_result=approved,
    )
    prompt = client.prompts[0]
    assert "IDENTIFYING CONTEXT" in prompt
    assert "OFFICIAL KNOWLEDGE" in prompt
    assert "STYLE REFERENCE EXAMPLES" in prompt


def test_11_refinement_prompt_preserves_three_trust_classes():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="حقيقة A"),))
    approved = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_arm_result(response_text="نمط B"),))
    _exec, client = _run(
        _req(
            comment_text="سؤال حول هذا العنوان تحديدًا.",
            identifying_metadata=IdentifyingMetadata("العنوان", "ENTRY-1"),
        ),
        ok_result=official, arm_result=approved,
        responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)],
    )
    assert client.call_count == 2
    refinement_prompt = client.prompts[1]
    assert "IDENTIFYING CONTEXT" in refinement_prompt
    assert "OFFICIAL KNOWLEDGE" in refinement_prompt
    assert "STYLE REFERENCE EXAMPLES" in refinement_prompt
    assert "REFINEMENT INSTRUCTION" in refinement_prompt


# --- 12: no official knowledge + identifying context => no implied authority


def test_12_no_official_knowledge_identifying_context_implies_no_factual_authority():
    identifying = build_identifying_grounding("عنوان", "ENTRY-1")
    exec_result, client = _run(_req(trusted_grounding_text=identifying), ok_result=_EMPTY_OK)
    assert exec_result.metadata.grounding_mode == "IDENTIFYING_ONLY"
    prompt = client.prompts[0]
    official_section = prompt.split("--- OFFICIAL KNOWLEDGE")[1].split("--- END OFFICIAL KNOWLEDGE")[0]
    assert "(none — do not invent facts)" in official_section


# --- 13-15: routing/grounding semantics unchanged from M5D.1 ----------------


def test_13_official_grounding_available_semantics_unchanged():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    exec_result, _ = _run(_req(), ok_result=official)
    assert exec_result.generation_result.eligibility_rule_id == "generate_inquiry_with_grounding"

    identifying = build_identifying_grounding("عنوان", "ENTRY-1")
    exec_result2, _ = _run(_req(trusted_grounding_text=identifying), ok_result=_EMPTY_OK)
    assert exec_result2.generation_result.eligibility_rule_id == "inquiry_without_trusted_grounding"


def test_14_identifying_only_inquiry_still_uses_ungrounded_route():
    identifying = build_identifying_grounding("عنوان", "ENTRY-1")
    exec_result, _ = _run(_req(trusted_grounding_text=identifying))
    assert exec_result.generation_result.eligibility_rule_id == "inquiry_without_trusted_grounding"


def test_15_real_official_knowledge_still_uses_grounded_route():
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    exec_result, _ = _run(_req(), ok_result=official)
    assert exec_result.generation_result.eligibility_rule_id == "generate_inquiry_with_grounding"


# --- 16: unsupported-claim validation behavior unchanged --------------------


def test_16_unsupported_claim_validation_behavior_unchanged():
    claim = "تم تحويل ملاحظتك إلى فريق المحتوى."
    # Official evidence supports it -> passes.
    official = OfficialKnowledgeRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(content="تم تحويل الملاحظات دائمًا إلى فريق المحتوى."),)
    )
    exec_result, _ = _run(_req(), ok_result=official, responses=[_draft_json(claim)])
    assert exec_result.generation_result.generation_status == "SUCCEEDED"

    # No official evidence, identifying context only -> still fails (unchanged).
    identifying = build_identifying_grounding("عنوان", "ENTRY-1")
    exec_result2, _ = _run(
        _req(trusted_grounding_text=identifying), ok_result=_EMPTY_OK,
        responses=[_draft_json(claim), _draft_json(claim)],
    )
    assert exec_result2.generation_result.generation_status == "FAILED_REVIEW_ONLY"
    assert exec_result2.generation_result.reason_code == "unsupported_operational_claim"

    # Direct validate_draft call, unmodified function, unaffected by this phase.
    ok, why = validate_draft(claim, trusted_grounding_text="تم تحويل الملاحظات دائمًا إلى فريق المحتوى.")
    assert ok and why is None


# --- 17: refinement max stays <= 2 -------------------------------------------


def test_17_refinement_max_remains_two():
    exec_result, client = _run(_req(), responses=[_draft_json(_TOO_SHORT_DRAFT), _draft_json(_VALID_DRAFT)])
    assert client.call_count == 2
    assert exec_result.metadata.logical_generation_count == 2


# --- 18: GenerationResult schema unchanged -----------------------------------


def test_18_generation_result_schema_unchanged():
    field_names = {f.name for f in fields(GenerationResult)}
    assert field_names == {
        "comment_id", "routing_decision", "generation_status", "suggested_response",
        "review_required", "reason_code", "provider", "model_name", "prompt_version",
        "generated_at", "future_reviewer_actions", "eligibility_rule_id", "error",
    }


# --- 19: persistence / human-review protections unchanged ------------------


def test_19_persistence_and_human_review_protections_unchanged(tmp_path):
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    engine = get_engine(tmp_path / "m5d2_persistence.sqlite")
    init_db(engine)
    upsert_review_items(engine, [dict(
        comment_id="c-1", source_file="f", source_row_index=0,
        review_status="APPROVED", is_prefiltered_junk=False, ready_to_send=True, row_version=1,
        reviewer_action="SEND_AS_IS", final_reply="رد المراجع المعتمد.",
    )])
    official = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_WITH_RESULTS, results=(_ok_result(),))
    exec_result, _ = _run(_req(), ok_result=official)
    svc = ReviewWorkflowService(engine)
    detail = svc.apply_generation_result(
        exec_result.generation_result, intent="INQUIRY", topic="GENERAL", sentiment="NEUTRAL"
    )
    assert detail.review_status == "APPROVED"
    assert detail.final_response == "رد المراجع المعتمد."


# --- 20: no frontend changes -------------------------------------------------


def test_20_no_frontend_coupling_in_m5d2_modules():
    import inspect

    from rcs.response_generation import prompt as prompt_module
    from rcs.response_intelligence import evidence as evidence_module

    for module in (prompt_module, evidence_module):
        code = inspect.getsource(module)
        assert "web/" not in code
        assert "ReviewerWorkspace" not in code


# --- Stale-label scan: TRUSTED GROUNDING must be gone from the live prompt --


def test_no_stale_trusted_grounding_label_in_live_prompt_module():
    import inspect

    from rcs.response_generation import prompt as prompt_module

    source = inspect.getsource(prompt_module)
    assert "TRUSTED GROUNDING" not in source


def test_direct_caller_trusted_grounding_text_now_renders_as_official_knowledge():
    """rcs.response_generation.prompt_candidates.py (a frozen, NOT-
    production-wired research module for offline eval) is explicitly
    out of scope for this phase and still uses its own old label --
    this test only proves the LIVE build_generation_prompt is clean."""
    prompt = build_generation_prompt(
        comment_text="تعليق", primary_intent="INQUIRY", topic_subtype="GENERAL",
        sentiment="NEUTRAL", reason_code="inquiry_without_trusted_grounding",
        trusted_grounding_text="نص حقيقي معتمد.",
    )
    assert "OFFICIAL KNOWLEDGE" in prompt
    assert "TRUSTED GROUNDING" not in prompt
