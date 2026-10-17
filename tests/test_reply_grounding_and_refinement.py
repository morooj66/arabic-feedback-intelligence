"""Grounding readiness (no approved source exists locally) and bounded, gate-validated refinement for the new reply-safety codes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from config import settings
from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES, REFINEMENT_GUIDANCE, is_refinement_eligible
from rcs.response_generation.service import GenerationRequest
from rcs.response_generation.validate import validate_draft
from rcs.response_intelligence.retrieval.contracts import SUCCESS_EMPTY, ApprovedResponseMemoryRetrievalResult, OfficialKnowledgeRetrievalResult
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from scripts.offline_eval.response_agent.fakes import FakeApprovedResponseMemoryRetriever, FakeGeminiClient, FakeOfficialKnowledgeRetriever

ROOT = Path(__file__).resolve().parents[1]


def _draft_json(text):
    return json.dumps({"suggested_response": text}, ensure_ascii=False)


def _run(request, client):
    return run_agent_pipeline(
        request, client=client,
        official_knowledge_retriever=FakeOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)),
        approved_response_retriever=FakeApprovedResponseMemoryRetriever(ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)),
    )


def _req(comment, intent="SUGGESTION", topic="DICTIONARY_SUGGESTION"):
    return GenerationRequest(comment_id="c", comment_text=comment, primary_intent=intent, topic_subtype=topic, sentiment="NEUTRAL", confidence=0.5)


# ------------------------------------------------------------------------------------------- no approved source exists
def test_no_official_knowledge_source_is_configured_by_default(monkeypatch):
    from rcs.response_intelligence import retrievers

    monkeypatch.setattr(settings, "OFFICIAL_KNOWLEDGE_SOURCE_PATH", None)
    retrievers.reset_default_retrievers_for_testing()
    try:
        r = retrievers.default_official_knowledge_retriever()
        from rcs.response_intelligence.retrieval.contracts import OfficialKnowledgeQuery

        res = r.retrieve(OfficialKnowledgeQuery(query_text="ما معنى كلمة سوار", max_results=3, topic="GENERAL"))
        assert not getattr(res, "results", ())                              # nothing is retrievable, nothing is invented
    finally:
        retrievers.reset_default_retrievers_for_testing()


def test_corpus_builder_never_writes_official_records_without_explicit_operator_approval(tmp_path):
    import pandas as pd

    src = tmp_path / "export.xlsx"
    pd.DataFrame({"الكلمة": ["كلمة"], "التعريف": ["تعريف تجريبي"]}).to_excel(src, index=False)
    out = tmp_path / "corpus.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/build_official_knowledge_corpus.py"), "--input", str(src), "--output", str(out),
                        "--headword-column", "الكلمة", "--definition-column", "التعريف"], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0 and not out.exists()                          # fails closed: approval is never inferred from the file


def test_factual_question_without_a_source_stays_with_a_human_and_never_reaches_gemini():
    client = FakeGeminiClient([_draft_json("معنى كلمة «سوار» هو حلية تلبس في المعصم.")])
    result = _run(_req("ما معنى كلمة سوار؟", "INQUIRY", "GENERAL"), client).generation_result
    assert result.routing_decision == "REVIEW_ONLY" and result.suggested_response is None and client.call_count == 0


# -------------------------------------------------------------------------------------------- gate refinements (measured)
@pytest.mark.parametrize("text", [
    "شاكرين لكم طرح هذه الفكرة اللغوية.",
    "في حال رغبتكم في تقديم اقتراح أو ملاحظة لغوية يسعدنا استقبالها.",
    "نشير إلى استفساركم بشأن الكلمة، ونرجو تزويدنا بمزيد من التفاصيل.",
    "لا يمكن تقديم تبسيط دقيق من المعلومات الموثوقة المتاحة لدينا.",
])
def test_plain_courtesy_and_limitation_wording_is_not_mistaken_for_a_claim(text):
    assert validate_draft(text, comment_text="استفسار بشأن الكلمة") == (True, None)


@pytest.mark.parametrize("text,code", [
    ("وصلت رسالتكم إلى فريق المعجم.", "unsupported_commitment"),                      # delivery claim nothing can vouch for
    ("شكرًا لتواصلكم، وصل اقتراحي الذي كتبته إلى الإدارة المعنية.", "unsupported_commitment"),
    ("تنبيه دقيق بشأن الخطأ في عزو الشاهد.", "unsupported_correctness_claim"),
])
def test_unverifiable_delivery_and_accuracy_verdicts_are_rejected(text, code):
    assert validate_draft(text, comment_text="أقترح تصحيح الخطأ في الشاهد واقتراحي مهم") == (False, code)


def test_accuracy_verdict_on_the_visitors_observation_is_a_correctness_claim_for_reviewer_text_too():
    from rcs.response_generation.validate import collect_reply_findings

    f = collect_reply_findings("نشكركم، وهذه ملاحظة دقيقة بشأن الشاهد.", comment_text="أرى خطأ في الشاهد")
    assert "unsupported_correctness_claim" in f.attestable and not f.blocking


# --------------------------------------------------------------------------------------- bounded, validated refinement
def test_new_codes_are_refinable_and_leak_pii_are_not():
    for code in ("unsupported_assertion", "unsupported_commitment", "unsupported_policy_claim", "unsupported_definition",
                 "unsupported_correctness_claim", "unsupported_number", "unapproved_url"):
        assert is_refinement_eligible("FAILED_REVIEW_ONLY", code) and code in REFINEMENT_GUIDANCE
    assert not is_refinement_eligible("FAILED_REVIEW_ONLY", "internal_leak") and not is_refinement_eligible("FAILED_REVIEW_ONLY", "pii_in_draft")
    assert set(REFINEMENT_GUIDANCE) <= REFINEMENT_ELIGIBLE_REASON_CODES


def test_a_commitment_is_repaired_once_by_a_rewrite_that_passes_every_gate():
    safe = "نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر."
    client = FakeGeminiClient([_draft_json("شكرًا لكم، سنقوم بإضافة الخاصية قريبًا."), _draft_json(safe)])
    out = _run(_req("أتمنى إضافة خاصية البحث بالجذر"), client)
    r = out.generation_result
    assert r.generation_status == "SUCCEEDED" and r.suggested_response == safe
    assert out.metadata.refinement_attempted is True and client.call_count == 2
    second_prompt = client.prompts[1]
    assert "Required correction:" in second_prompt and "Remove every promise" in second_prompt
    assert validate_draft(r.suggested_response, comment_text="أتمنى إضافة خاصية البحث بالجذر") == (True, None)


def test_an_unrepaired_rewrite_still_fails_closed_with_no_draft_and_at_most_two_calls():
    client = FakeGeminiClient([_draft_json("شكرًا لكم، سنقوم بإضافة الخاصية قريبًا."), _draft_json("سنضيف الخاصية في التحديث القادم إن شاء الله.")])
    r = _run(_req("أتمنى إضافة خاصية البحث بالجذر"), client).generation_result
    assert r.generation_status == "FAILED_REVIEW_ONLY" and r.suggested_response is None and client.call_count == 2


def test_leaked_internal_information_is_never_sent_through_a_second_call():
    client = FakeGeminiClient([_draft_json("حسب تصنيف النظام confidence 0.9 فهذا اقتراح."), _draft_json("نشكر لكم اقتراحكم.")])
    r = _run(_req("أتمنى إضافة خاصية البحث بالجذر"), client).generation_result
    assert r.generation_status == "FAILED_REVIEW_ONLY" and client.call_count == 1


def test_refinement_block_for_legacy_codes_is_unchanged():
    from rcs.response_generation.prompt import _build_refinement_instruction_block
    from rcs.response_generation.refinement import RefinementContext

    block = _build_refinement_instruction_block(RefinementContext(previous_draft="مسودة سابقة", failure_reason_code="generic_boilerplate"))
    assert "Required correction" not in block and "failed validation because of: generic_boilerplate" in block
