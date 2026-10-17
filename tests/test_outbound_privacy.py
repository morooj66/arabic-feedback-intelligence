"""Unit and integration tests of the final outbound privacy boundary (rcs.response_generation.outbound_privacy). No network."""
from __future__ import annotations

import json

import pytest

from rcs.response_generation import outbound_privacy as OP
from rcs.response_generation.outbound_privacy import OutboundPrivacyError, guard_prompt, sanitize_text
from rcs.response_generation.prompt import REFERENCE_HOME_URL, build_generation_prompt
from rcs.response_generation.refinement import RefinementContext
from rcs.response_generation.service import GenerationRequest, generate_once_with_draft
from scripts.offline_eval.response_agent.fakes import FakeGeminiClient


def _prompt(comment, **kw):
    return build_generation_prompt(comment_text=comment, primary_intent="SUGGESTION", topic_subtype="DICTIONARY_SUGGESTION", sentiment="NEUTRAL",
                                   reason_code="suggestion_eligible", **kw)


# ------------------------------------------------------------------------------------------------------------ rules
@pytest.mark.parametrize("text,gone", [
    ("أنا محمد بن عبدالله العتيبي وأقترح كلمة", ["محمد", "العتيبي"]),
    ("اسمي فاطمة الزهراني وأقترح كلمة", ["فاطمة", "الزهراني"]),
    ("اقتراح جيد. تحياتي خالد السبيعي", ["خالد", "السبيعي"]),
    ("جدتي نورة بنت سعد بن فهد تقولها", ["نورة", "فهد"]),
    ("أنا وليد وأقترح إضافة كلمة", ["وليد"]),
    ("أقترح إضافة كلمة جديدة\nمع التحية سعد الحربي", ["سعد", "الحربي"]),                                 # a name that merely starts with waw is still a name
    ("My name is John Smith and I suggest a word", ["John", "Smith"]),
    ("راسلوني sara@example.com أو @sara_acc أو https://example.org/p", ["sara@example.com", "@sara_acc", "example.org"]),
])
def test_identifying_text_is_masked(text, gone):
    r = sanitize_text(text)
    assert r.text is not None and not [g for g in gone if g in r.text]


@pytest.mark.parametrize("text,keep", [
    ("أقترح إضافة كلمة محمد إلى المعجم", "كلمة محمد"),
    ("الكلمة: نوف اسم علم", "الكلمة: نوف"),
    ("أنا أقترح استخدام مفردة الشابكة", "أقترح استخدام مفردة الشابكة"),
    ("أنا أرى أن التعريف غير دقيق", "أرى أن التعريف"),
    ("أنا واقترح تعديل التعريف", "واقترح تعديل التعريف"),
    ("أمي تقول هذه الكلمة دائما", "تقول هذه الكلمة"),
    ("ترجمة السفور إلى toilet غير صحيحة", "toilet"),
    ("قال ابن منظور في لسان العرب", "ابن منظور"),
])
def test_dictionary_content_is_kept(text, keep):
    assert keep in sanitize_text(text).text


def test_sanitiser_is_idempotent():
    once = sanitize_text("أنا خالد القحطاني 0501234567 sara@example.com وأقترح كلمة").text
    assert sanitize_text(once).text == once


@pytest.mark.parametrize("text,reason", [
    ("زوجي طلق ابنة عمي بسبب قضية في المحكمة", "sensitive_personal_circumstance"),
    ("الجوال: 0501234567", "only_identity_or_empty_content"),
    ("   ", "empty"),
])
def test_unsafe_or_content_free_text_is_blocked_not_sent(text, reason):
    r = sanitize_text(text)
    assert r.text is None and r.blocked_reason == reason


@pytest.mark.parametrize("text", ["الموقع بطيء جدًا وتجربة سيئة", "الموقع ما يفتح عندي", "الموقع لا يعمل", "الموقع جميل جدا"])
def test_short_comments_starting_with_the_word_website_are_not_mistaken_for_signatures(text):
    assert sanitize_text(text).text == text


def test_form_boilerplate_is_dropped_but_content_kept():
    r = sanitize_text("الجوال: 0501234567, نص الرسالة: أقترح إضافة خاصية البحث بالجذر")
    assert r.text == "أقترح إضافة خاصية البحث بالجذر"


# ------------------------------------------------------------------------------------------------ boundary behaviour
def test_system_instructions_and_the_one_approved_url_are_untouched():
    p = _prompt("أقترح إضافة خاصية البحث بالجذر")
    assert guard_prompt(p) == p and REFERENCE_HOME_URL in guard_prompt(p)


def test_every_variable_block_is_sanitised_including_metadata_and_style_and_previous_draft():
    p = _prompt("أنا سلمان الغامدي وأقترح كلمة", identifying_context='submission title: "كلمة 0501234567" mail a@b.org',
                style_reference_texts=("نشكر لكم تواصلكم. اتصلوا 0501234567",),
                refinement_context=RefinementContext(previous_draft="شكرا لك، راسلونا على a@b.org أو 0501234567 أو https://example.org/x", failure_reason_code="generic_boilerplate"))
    out = guard_prompt(p)
    assert "سلمان" not in out.split("--- UNTRUSTED VISITOR COMMENT")[1].split("--- END UNTRUSTED")[0]
    for leaked in ("0501234567", "a@b.org", "example.org"):
        assert leaked not in out, leaked


def test_approved_knowledge_keeps_lexical_names_but_never_identifiers():
    k = "- (مصدر معتمد) محمد: اسم علم مذكر. للتواصل info@example.org"
    out = guard_prompt(_prompt("أقترح كلمة", trusted_grounding_text=k))
    assert "محمد: اسم علم" in out and "info@example.org" not in out


def test_a_prompt_without_a_visitor_block_is_refused():
    with pytest.raises(OutboundPrivacyError):
        guard_prompt("some prompt without the delimited visitor block")


def test_residual_identifier_in_the_fixed_text_fails_closed():
    p = _prompt("أقترح كلمة") + "\ncontact: leak@example.org"
    with pytest.raises(OutboundPrivacyError, match="residual_identifier_in_prompt"):
        guard_prompt(p)


# ------------------------------------------------------ the service: initial request, retry and refinement are all guarded
def _req(comment):
    return GenerationRequest(comment_id="c", comment_text=comment, primary_intent="SUGGESTION", topic_subtype="DICTIONARY_SUGGESTION", sentiment="NEUTRAL", confidence=0.5)


def _j(t):
    return json.dumps({"suggested_response": t}, ensure_ascii=False)


COMMENT = "أنا محمد الدوسري وأقترح إضافة خاصية البحث بالجذر 0501234567 sara@example.com"


def test_initial_request_is_sanitised():
    client = FakeGeminiClient([_j("نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر.")])
    generate_once_with_draft(_req(COMMENT), client)
    assert client.call_count == 1 and all(x not in client.prompts[0] for x in ("الدوسري", "0501234567", "sara@example.com"))


def test_api_retry_requests_are_sanitised(monkeypatch):
    import httpx
    from rcs.response_generation import service

    monkeypatch.setattr(service, "_sleep", lambda s: None)
    client = FakeGeminiClient([httpx.ReadTimeout("t"), httpx.ReadTimeout("t"), _j("نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر.")])
    generate_once_with_draft(_req(COMMENT), client, max_retries=2)
    assert client.call_count == 3 and all("الدوسري" not in p and "0501234567" not in p for p in client.prompts)


def test_refinement_request_is_sanitised_even_when_the_previous_draft_echoes_identifiers():
    from rcs.response_intelligence.orchestrator import run_agent_pipeline
    from rcs.response_intelligence.retrieval.contracts import SUCCESS_EMPTY, ApprovedResponseMemoryRetrievalResult, OfficialKnowledgeRetrievalResult
    from scripts.offline_eval.response_agent.fakes import FakeApprovedResponseMemoryRetriever, FakeOfficialKnowledgeRetriever

    client = FakeGeminiClient([_j("شكرا لك، سنقوم بإضافة الخاصية وسنراسلك على sara@example.com أو https://example.org/x"), _j("نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر.")])
    out = run_agent_pipeline(_req(COMMENT), client=client,
                             official_knowledge_retriever=FakeOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)),
                             approved_response_retriever=FakeApprovedResponseMemoryRetriever(ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)))
    assert client.call_count == 2 and out.metadata.refinement_attempted
    assert all("الدوسري" not in p and "sara@example.com" not in p and "0501234567" not in p for p in client.prompts)


def test_a_blocked_comment_makes_no_external_request_and_fails_closed_without_retry():
    client = FakeGeminiClient([_j("نشكر لكم اقتراحكم.")])
    result, draft = generate_once_with_draft(_req("زوجي طلق ابنة عمي بسبب قضية في المحكمة وأقترح كلمة جديدة"), client)
    assert client.call_count == 0 and draft is None
    assert result.generation_status == "FAILED_REVIEW_ONLY" and result.reason_code == "outbound_privacy_blocked" and result.suggested_response is None


def test_privacy_block_is_never_refinable():
    from rcs.response_generation.refinement import is_refinement_eligible

    assert not is_refinement_eligible("FAILED_REVIEW_ONLY", "outbound_privacy_blocked")
