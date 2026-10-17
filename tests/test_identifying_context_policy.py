"""
Phase M6.2 — focused tests for
src/rcs/response_intelligence/identifying_context_policy.py (the
canonical Identifying Context Policy) and its wiring into
run_agent_pipeline. Fast, deterministic -- no real model load, no
network, no Gemini call.

Root cause under test: identifying metadata (dictionary-entry title/
entry_id) was previously trusted unconditionally by the Response Agent,
letting it become a false topic anchor for comments that never
established any reference to the entry ("لا اعلم" + "قناطر" -> context
hijacking, confirmed by human review across three evaluation rounds).
"""

from __future__ import annotations

import datetime as dt

from rcs.response_generation.service import GenerationRequest
from rcs.response_intelligence.identifying_context_policy import (
    REASON_EXPLICIT_ENTRY_REFERENCE,
    REASON_LEXICAL_CORROBORATION,
    REASON_NO_CORROBORATING_REFERENCE,
    REASON_NO_IDENTIFYING_METADATA,
    IdentifyingContextDecision,
    IdentifyingMetadata,
    evaluate_identifying_context,
)
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
    OfficialKnowledgeResult,
)
from scripts.offline_eval.response_agent.fakes import (
    FakeApprovedResponseMemoryRetriever,
    FakeGeminiClient,
    FakeOfficialKnowledgeRetriever,
)

_VALID_DRAFT_JSON = '{"suggested_response": "شكرًا لتواصلكم معنا."}'
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)


def _empty_ok():
    return FakeOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY))


def _run(request, client, ok=None, arm=None):
    return run_agent_pipeline(
        request, client=client,
        official_knowledge_retriever=ok or _empty_ok(),
        approved_response_retriever=arm or FakeApprovedResponseMemoryRetriever(_EMPTY_ARM),
    )


# --- 1. Documented positive/negative policy cases -------------------------


def test_allah_aelam_qanatir_not_allowed():
    decision = evaluate_identifying_context(
        comment_text="الله اعلم", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_laa_aelam_qanatir_not_allowed():
    decision = evaluate_identifying_context(
        comment_text="لا اعلم", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_ttt_qanatir_not_allowed():
    decision = evaluate_identifying_context(
        comment_text="تتت", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_anshita_la_safiya_qanatir_not_allowed():
    decision = evaluate_identifying_context(
        comment_text="أنشطة لا صفية", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_ahsantum_unrelated_title_not_allowed():
    decision = evaluate_identifying_context(
        comment_text="أحسنتم", identifying_metadata=IdentifyingMetadata("إِطَاعَة", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_makbis_lexical_corroboration_allowed():
    decision = evaluate_identifying_context(
        comment_text="مكبس بكسر الميم", identifying_metadata=IdentifyingMetadata("مكبس", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_LEXICAL_CORROBORATION
    assert decision.matched_reference == "مكبس"


def test_hatha_almadkhal_explicit_reference_allowed():
    decision = evaluate_identifying_context(
        comment_text="في هذا المدخل خطأ في المثال", identifying_metadata=IdentifyingMetadata("اشتد", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_EXPLICIT_ENTRY_REFERENCE
    assert decision.matched_reference == "هذا المدخل"


def test_almithal_huna_explicit_reference_allowed():
    decision = evaluate_identifying_context(
        comment_text="المثال هنا منسوب إلى القرآن", identifying_metadata=IdentifyingMetadata("اشتد", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_EXPLICIT_ENTRY_REFERENCE
    assert decision.matched_reference == "المثال هنا"


def test_al_maana_muhimm_alone_not_allowed():
    """Negative example explicitly required: a bare domain noun +
    predicate adjective must NOT qualify by itself."""
    decision = evaluate_identifying_context(
        comment_text="المعنى مهم", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_alkalima_jameela_alone_not_allowed():
    decision = evaluate_identifying_context(
        comment_text="الكلمة جميلة", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_uhibb_hathihi_almustalahat_plural_demonstrative_not_allowed():
    """Negative example: a demonstrative + PLURAL noun must not match the
    (singular-only) referent-noun set."""
    decision = evaluate_identifying_context(
        comment_text="أحب هذه المصطلحات", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


def test_fi_maana_alkalima_construct_reference_allowed():
    decision = evaluate_identifying_context(
        comment_text="في معنى الكلمة يوجد خطأ", identifying_metadata=IdentifyingMetadata("اشتد", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_EXPLICIT_ENTRY_REFERENCE
    assert decision.matched_reference == "معنى الكلمة"


def test_fi_tareef_alkalima_construct_reference_allowed():
    decision = evaluate_identifying_context(
        comment_text="في تعريف الكلمة مشكلة", identifying_metadata=IdentifyingMetadata("اشتد", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_EXPLICIT_ENTRY_REFERENCE


def test_altareef_almadhkur_allowed():
    decision = evaluate_identifying_context(
        comment_text="التعريف المذكور غير دقيق", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_EXPLICIT_ENTRY_REFERENCE
    assert decision.matched_reference == "التعريف المذكور"


def test_alshahid_almadhkur_allowed():
    decision = evaluate_identifying_context(
        comment_text="الشاهد المذكور غير صحيح", identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_EXPLICIT_ENTRY_REFERENCE
    assert decision.matched_reference == "الشاهد المذكور"


def test_long_iatibar_suggestion_lexical_corroboration_allowed():
    comment = "لا يعمل الرابط الصوتي في صفحة هذه المادة عندي منذ يومين«اعتبار» هل يمكن توضيح الفرق بين هذه الكلمة والكلمة القريبة منها في المعنى"
    decision = evaluate_identifying_context(
        comment_text=comment, identifying_metadata=IdentifyingMetadata("اِعْتِبَار", "6b746072-ca71-483f-8b73-c52da0efbdde")
    )
    assert decision.allowed is True
    assert decision.reason == REASON_LEXICAL_CORROBORATION
    assert decision.matched_reference == "اِعْتِبَار"


def test_substantive_unrelated_comment_not_allowed_merely_because_long():
    """Must NOT allow merely because the comment is substantive/long."""
    long_unrelated = "هذا نص طويل جدًا يتحدث عن مواضيع كثيرة ومتنوعة دون أي علاقة بأي مدخل معجمي محدد. " * 5
    decision = evaluate_identifying_context(
        comment_text=long_unrelated, identifying_metadata=IdentifyingMetadata("قناطر", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE


# --- No identifying metadata at all -------------------------------------


def test_no_metadata_object_not_allowed():
    decision = evaluate_identifying_context(comment_text="أي نص", identifying_metadata=None)
    assert decision.allowed is False
    assert decision.reason == REASON_NO_IDENTIFYING_METADATA


def test_blank_title_not_allowed():
    decision = evaluate_identifying_context(
        comment_text="أي نص", identifying_metadata=IdentifyingMetadata("   ", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_IDENTIFYING_METADATA


def test_none_title_with_entry_id_present_not_allowed():
    """entry_id alone (no title) must resolve to NO_IDENTIFYING_METADATA
    -- entry_id is never treated as the metadata's identity by itself."""
    decision = evaluate_identifying_context(
        comment_text="لا اعلم", identifying_metadata=IdentifyingMetadata(None, "قناطر")
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_IDENTIFYING_METADATA


# --- 2. Arabic diacritic stripping is deterministic -----------------------


def test_diacritic_stripping_makes_diacritized_title_match_plain_comment():
    decision = evaluate_identifying_context(
        comment_text="في معنى اشتد ضربتم مثالا خاطئا", identifying_metadata=IdentifyingMetadata("اِشْتَدَّ", None)
    )
    assert decision.allowed is True
    assert decision.reason == REASON_LEXICAL_CORROBORATION


def test_diacritic_stripping_is_deterministic_across_repeated_calls():
    kwargs = dict(comment_text="مكبس بكسر الميم", identifying_metadata=IdentifyingMetadata("مكبس", None))
    a = evaluate_identifying_context(**kwargs)
    b = evaluate_identifying_context(**kwargs)
    assert a == b


# --- 3 & 4. entry_id is NEVER semantic evidence --------------------------


def test_entry_id_alone_never_satisfies_lexical_corroboration():
    """entry_id happens to be a real Arabic word here (exactly the row
    6281 real-data shape) -- must still never be treated as the
    identity/lexical-corroboration term."""
    decision = evaluate_identifying_context(
        comment_text="لا اعلم", identifying_metadata=IdentifyingMetadata(None, "قناطر")
    )
    assert decision.reason == REASON_NO_IDENTIFYING_METADATA
    assert decision.allowed is False


def test_arabic_looking_entry_id_does_not_leak_into_a_correct_decision():
    """Even when title correctly corroborates and the decision is
    ALLOWED, entry_id being a real Arabic word must not itself factor
    into that outcome -- changing only entry_id must not change the
    decision at all."""
    decision_with_word_entry_id = evaluate_identifying_context(
        comment_text="مكبس بكسر الميم", identifying_metadata=IdentifyingMetadata("مكبس", "قناطر")
    )
    decision_with_uuid_entry_id = evaluate_identifying_context(
        comment_text="مكبس بكسر الميم", identifying_metadata=IdentifyingMetadata("مكبس", "6b746072-uuid")
    )
    decision_with_no_entry_id = evaluate_identifying_context(
        comment_text="مكبس بكسر الميم", identifying_metadata=IdentifyingMetadata("مكبس", None)
    )
    assert decision_with_word_entry_id == decision_with_uuid_entry_id == decision_with_no_entry_id
    assert decision_with_word_entry_id.matched_reference == "مكبس"  # never the entry_id value


# --- 5. Only comment_text + structured metadata participate --------------


def test_evaluate_identifying_context_signature_has_no_other_parameters():
    import inspect

    sig = inspect.signature(evaluate_identifying_context)
    assert set(sig.parameters.keys()) == {"comment_text", "identifying_metadata"}


def test_result_independent_of_intent_topic_sentiment_confidence():
    """There is no way to pass these to the function at all -- this test
    documents that guarantee is structural, not just a convention."""
    import inspect

    source = inspect.getsource(evaluate_identifying_context)
    for banned in ("primary_intent", "topic_subtype", "sentiment", "confidence"):
        assert banned not in source


# --- 6 & 7. Runs exactly once; rendering only after ALLOW -----------------


def test_policy_runs_exactly_once_per_agent_execution(monkeypatch):
    import rcs.response_intelligence.orchestrator as orch_module

    calls = []
    real = orch_module.evaluate_identifying_context

    def _spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(orch_module, "evaluate_identifying_context", _spy)
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("قناطر", None),
    )
    _run(request, client)
    assert len(calls) == 1


def test_build_identifying_grounding_called_only_when_allowed(monkeypatch):
    import rcs.response_intelligence.orchestrator as orch_module

    calls = []
    real = orch_module.build_identifying_grounding

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(orch_module, "build_identifying_grounding", _spy)

    # Denied case: build_identifying_grounding must never be called.
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    denied_request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("قناطر", None),
    )
    _run(denied_request, client)
    assert len(calls) == 0

    # Allowed case: build_identifying_grounding must be called exactly once.
    client2 = FakeGeminiClient([_VALID_DRAFT_JSON])
    allowed_request = GenerationRequest(
        comment_id="y", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", None),
    )
    _run(allowed_request, client2)
    assert len(calls) == 1


# --- 8. Denied context does not appear in the Gemini request/prompt ------


def test_denied_context_absent_from_prompt():
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("قناطر", None),
    )
    result = _run(request, client)
    assert result.metadata.identifying_context_allowed is False
    assert "IDENTIFYING CONTEXT" not in client.prompts[0]
    assert "قناطر" not in client.prompts[0]


# --- 9. Allowed context still reaches the same downstream path -----------


def test_allowed_context_reaches_prompt_via_the_same_grounding_path():
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", None),
    )
    result = _run(request, client)
    assert result.metadata.identifying_context_allowed is True
    assert result.metadata.identifying_context_decision_reason == REASON_LEXICAL_CORROBORATION
    assert result.metadata.grounding_mode == "IDENTIFYING_ONLY"  # unaffected -- depends only on Official Knowledge
    assert "IDENTIFYING CONTEXT" in client.prompts[0]
    assert "مكبس" in client.prompts[0]


# --- 10. Official Knowledge behavior unchanged ---------------------------


def test_official_knowledge_available_unaffected_by_policy_denial():
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
    ok = FakeOfficialKnowledgeRetriever(ok_result)
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="INQUIRY",
        topic_subtype="GENERAL", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("قناطر", None),  # denied
    )
    result = _run(request, client, ok=ok)
    assert result.metadata.identifying_context_allowed is False  # denied, as expected
    assert result.metadata.official_grounding_available is True  # completely unaffected
    assert "حقيقة معتمدة" in client.prompts[0]


def test_official_knowledge_absent_unaffected_by_policy_allow():
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", None),  # allowed
    )
    result = _run(request, client)
    assert result.metadata.identifying_context_allowed is True
    assert result.metadata.official_grounding_available is False  # completely unaffected


# --- 11. Approved Response Memory behavior unchanged ----------------------


def test_approved_response_memory_unaffected_by_policy():
    from rcs.response_intelligence.retrieval.contracts import ApprovedResponseMemoryResult

    arm_result = ApprovedResponseMemoryRetrievalResult(
        outcome=SUCCESS_WITH_RESULTS,
        results=(
            ApprovedResponseMemoryResult(
                review_item_id="r1", final_reply_id="hash-r1", response_text="نمط أسلوبي معتمد.",
                reviewer_action="SEND_AS_IS", approved_at=dt.datetime(2026, 1, 1), retrieval_score=0.9,
            ),
        ),
    )
    arm = FakeApprovedResponseMemoryRetriever(arm_result)
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("قناطر", None),  # denied
    )
    result = _run(request, client, arm=arm)
    assert result.metadata.identifying_context_allowed is False
    assert result.metadata.style_retrieval_outcome == SUCCESS_WITH_RESULTS
    assert result.metadata.style_result_count == 1
    assert "نمط أسلوبي معتمد" in client.prompts[0]


# --- 12. rg-v10 unchanged (by the identifying-context policy) ----------
#
# Two later, separate prompt-content changes legitimately bumped
# settings.RESPONSE_GENERATION_PROMPT_VERSION, currently "rg-v14.2" --
# see rcs.response_generation.prompt's module docstring. The two
# literals below are updated to that current value; this file's own
# point (the identifying-context policy's own changes never touched
# the prompt version) is unaffected and remains true.


def test_rg_v10_unchanged():
    from config import settings

    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"


def test_generation_result_carries_rg_v10_regardless_of_decision():
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("قناطر", None),
    )
    result = _run(request, client)
    assert result.generation_result.prompt_version == "rg-v14.4"


# --- 13. No new Gemini call is introduced ---------------------------------


def test_no_extra_gemini_call_to_evaluate_policy():
    """One real Gemini call for a normal (no-refinement) execution --
    the policy adds zero calls."""
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", None),
    )
    _run(request, client)
    assert client.call_count == 1


def test_policy_module_never_calls_generate():
    import inspect

    from rcs.response_intelligence import identifying_context_policy

    source = inspect.getsource(identifying_context_policy)
    assert ".generate(" not in source
    assert "SupportsGenerate" not in source


# --- Structural: no hidden fallback, no duplicated logic -----------------


def test_decision_dataclass_fields_are_minimal_and_safe():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(IdentifyingContextDecision)}
    assert fields == {"allowed", "reason", "matched_reference"}


def test_metadata_dataclass_fields_are_minimal():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(IdentifyingMetadata)}
    assert fields == {"original_title", "entry_id"}


# --- 14 & 15. Production Excel caller and Inspector go through the same policy ---
# (see also tests/test_excel_review_flow.py's
# test_generation_grounding_a/b/c_* and tests/test_dev_inspect_agent_inspector.py,
# both re-run unchanged and still passing against the real orchestrator.)


def test_excel_review_flow_supplies_structured_metadata_not_legacy_fields():
    """AST-based (not substring) so an explanatory comment mentioning the
    old function's name by way of history can't trip the check -- what
    matters is that the module no longer IMPORTS it."""
    import ast

    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "rcs" / "review" / "excel_review_flow.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "identifying_metadata=IdentifyingMetadata(" in source
    tree = ast.parse(source)
    imported_names = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert "build_identifying_grounding" not in imported_names


def test_inspector_full_run_accepts_structured_title_entry_id_not_prose():
    import inspect

    from rcs.response_intelligence.inspector import run_full_inspection

    params = set(inspect.signature(run_full_inspection).parameters.keys())
    assert {"original_title", "entry_id"}.issubset(params)
    assert "identifying_context" not in params


def test_inspector_api_full_run_no_longer_accepts_prose_identifying_context():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "rcs" / "review" / "inspector_api.py"
    ).read_text(encoding="utf-8")
    # The full-run handler must read structured fields from the request body.
    assert 'original_title=body.get("original_title")' in source
    assert 'entry_id=body.get("entry_id")' in source


# --- 16. Legacy pre-rendered fields cannot bypass the orchestrator policy ---


def test_legacy_trusted_grounding_text_alone_cannot_bypass_policy():
    """A caller that ONLY sets the legacy trusted_grounding_text field
    (pre-M6.2 convention) and never identifying_metadata must resolve to
    NO_IDENTIFYING_METADATA -- no reverse-parsing, no fallback."""
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        trusted_grounding_text='submission title: "قناطر".',  # legacy field only
    )
    result = _run(request, client)
    assert result.metadata.identifying_context_allowed is False
    assert result.metadata.identifying_context_decision_reason == REASON_NO_IDENTIFYING_METADATA
    assert "قناطر" not in client.prompts[0]


def test_legacy_identifying_context_text_field_alone_cannot_bypass_policy():
    """Same proof for the OTHER legacy field name -- pre-rendered prose
    on GenerationRequest.identifying_context, with no identifying_metadata."""
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="لا اعلم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_context='submission title: "قناطر".',  # legacy field only
    )
    result = _run(request, client)
    assert result.metadata.identifying_context_allowed is False
    assert result.metadata.identifying_context_decision_reason == REASON_NO_IDENTIFYING_METADATA
    assert "قناطر" not in client.prompts[0]


def test_response_agent_state_no_longer_has_a_trusted_grounding_text_field():
    import dataclasses

    from rcs.response_intelligence.state import ResponseAgentState

    fields = {f.name for f in dataclasses.fields(ResponseAgentState)}
    assert "trusted_grounding_text" not in fields
    assert "identifying_metadata" in fields


# --- M6.2.1: entry_id must never reach Gemini, even when allowed --------


def test_uuid_entry_id_never_appears_in_prompt_when_allowed():
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", "6b746072-ca71-483f-8b73-c52da0efbdde"),
    )
    result = _run(request, client)
    assert result.metadata.identifying_context_allowed is True
    assert "6b746072-ca71-483f-8b73-c52da0efbdde" not in client.prompts[0]
    assert "مكبس" in client.prompts[0]  # the human-readable title still renders


def test_arabic_looking_entry_id_never_appears_in_prompt_when_allowed():
    """entry_id happens to be a real, specific Arabic word here (exactly
    the row-6281 real-data shape) -- must still never be rendered into
    the prompt, even though it would read as legitimate Arabic text."""
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", "قناطر"),
    )
    result = _run(request, client)
    assert result.metadata.identifying_context_allowed is True
    assert "قناطر" not in client.prompts[0]
    assert "مكبس" in client.prompts[0]


def test_original_title_still_appears_when_allowed_regardless_of_entry_id_shape():
    for entry_id in (None, "uuid-shaped-1234", "قناطر"):
        client = FakeGeminiClient([_VALID_DRAFT_JSON])
        request = GenerationRequest(
            comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
            topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
            identifying_metadata=IdentifyingMetadata("مكبس", entry_id),
        )
        result = _run(request, client)
        assert result.metadata.identifying_context_allowed is True
        assert "مكبس" in client.prompts[0]
        assert "IDENTIFYING CONTEXT" in client.prompts[0]


def test_entry_id_never_rendered_official_knowledge_unaffected():
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
    from scripts.offline_eval.response_agent.fakes import FakeOfficialKnowledgeRetriever as _FakeOK

    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="INQUIRY",
        topic_subtype="GENERAL", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", "6b746072-ca71-483f-8b73-c52da0efbdde"),
    )
    result = _run(request, client, ok=_FakeOK(ok_result))
    assert result.metadata.official_grounding_available is True
    assert "6b746072-ca71-483f-8b73-c52da0efbdde" not in client.prompts[0]
    assert "حقيقة معتمدة" in client.prompts[0]


def test_rg_v10_unchanged_after_entry_id_fix():
    """Value updated to "rg-v14": later, separate prompt-content
    changes legitimately bumped this for production traceability --
    see rcs.response_generation.prompt's module docstring. This test's
    own point (the entry_id fix itself never touched the prompt
    version) is unaffected."""
    from config import settings

    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"


def test_orchestrator_never_passes_entry_id_to_build_identifying_grounding(monkeypatch):
    """Structural proof: build_identifying_grounding's second positional
    argument is always None from the orchestrator, regardless of what
    entry_id the metadata actually holds."""
    import rcs.response_intelligence.orchestrator as orch_module

    calls = []
    real = orch_module.build_identifying_grounding

    def _spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(orch_module, "build_identifying_grounding", _spy)
    client = FakeGeminiClient([_VALID_DRAFT_JSON])
    request = GenerationRequest(
        comment_id="x", comment_text="مكبس بكسر الميم", primary_intent="COMPLAINT",
        topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.4,
        identifying_metadata=IdentifyingMetadata("مكبس", "قناطر"),
    )
    _run(request, client)
    assert len(calls) == 1
    assert calls[0][1] is None  # entry_id positional arg is always None


# --- Row 15 regression: "أحسنتم" + a real specific entry must be denied ---


def test_row_15_shaped_praise_with_real_entry_is_denied():
    """Row 15's real shape: comment='أحسنتم' (well done), title='إِطَاعَة'
    (a real, unrelated word). No lexical overlap, no explicit-reference
    cue -- must resolve to NO_CORROBORATING_REFERENCE, matching the
    approved policy expectation exactly (the earlier final report's
    'row 15 correctly allowed' claim was a report-wording error, not a
    runtime behavior -- see this test and the corrected report)."""
    decision = evaluate_identifying_context(
        comment_text="أحسنتم", identifying_metadata=IdentifyingMetadata("إِطَاعَة", None)
    )
    assert decision.allowed is False
    assert decision.reason == REASON_NO_CORROBORATING_REFERENCE
    assert decision.matched_reference is None
