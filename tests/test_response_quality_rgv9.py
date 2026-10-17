"""
Phase rg-v9 (2026-09) — regression tests for the three issues found in the
first real human review of 10 live visitor comments:

  1. Response quality: detached "evaluator voice" openings.
  2. Insufficient-content gate: single-letter fragments reaching Gemini.
  3. Topic/routing: investigated only, no behavior change (see the
     accompanying report; no test needed here since nothing changed).

Also proves the four "must remain unchanged" regressions (F, G, H below)
did not regress, and (E) that the factual-discipline prompt language
exists for the IDENTIFYING_ONLY case.
"""

from __future__ import annotations

from rcs.response_generation.eligibility import (
    GENERATE_DRAFT,
    NO_RESPONSE_RECOMMENDED,
    EligibilityInput,
    determine_response_eligibility,
)
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS, build_generation_prompt
from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES
from rcs.response_generation.validate import validate_draft

_LONG_LINGUISTIC_SUGGESTION = (
    "قلتم «التراكيب المتلازمة»اقترح إضافة خاصية البحث بحسب الجذر لتسهيل الوصول إلى الكلمات المشتقةcollocation. "
    "غير أنني أتساءل: لماذا لم تعتمدوا «المتلازمات اللفظية»أتمنى إضافة أمثلة تطبيقية توضّح استخدام الكلمة في سياقات مختلفة"
    "عدد كبير من المختصين في هذا المجال، ويبدو لي أن هذه الصيغة أدق وأوفق من الناحية الاصطلاحية."
)


# --- A. Detailed linguistic suggestion -> visitor-facing reply, not evaluator wording


def test_a_evaluator_voice_opening_fails_validation():
    evaluator_voice_draft = (
        "رأي اصطلاحي دقيق ومسبّب ينبه على التفريق بين المصطلحين، ونشكركم على هذا التوضيح."
    )
    ok, why = validate_draft(evaluator_voice_draft)
    assert ok is False
    assert why == "evaluator_voice_opening"


def test_a_direct_reply_addressed_to_visitor_passes_validation():
    direct_reply = (
        "نشكركم على اقتراحكم بشأن استخدام «المتلازمات اللفظية» في مقابل مصطلح (collocation)، "
        "وعلى توضيحكم للفارق الاصطلاحي بينها وبين «التراكيب المتلازمة»."
    )
    ok, why = validate_draft(direct_reply)
    assert ok is True and why is None


def test_a_evaluator_voice_opening_is_refinement_eligible():
    """A detached-evaluator-voice failure is recoverable by rewriting the
    opening -- it must get the one bounded refinement attempt, not fail
    closed outright."""
    assert "evaluator_voice_opening" in REFINEMENT_ELIGIBLE_REASON_CODES


def test_a_evaluator_voice_check_does_not_false_positive_on_natural_addressed_usage():
    """The same root word, used naturally and addressed TO the visitor
    (not as a detached third-person label), must not be blocked -- this
    is a register check on the OPENING, not a banned-word list."""
    natural_use = "نقدر ملاحظتكم بشأن الفرق بين المعنيين في هذا المدخل."
    ok, why = validate_draft(natural_use)
    assert ok is True and why is None


def test_a_prompt_instructs_direct_reply_not_evaluation():
    assert "Direct reply, not evaluation" in SYSTEM_INSTRUCTIONS
    assert "ملاحظة دقيقة" in SYSTEM_INSTRUCTIONS
    assert "detached" in SYSTEM_INSTRUCTIONS.lower()


def test_a_prompt_engages_with_substantive_linguistic_reasoning():
    prompt = build_generation_prompt(
        comment_text=_LONG_LINGUISTIC_SUGGESTION,
        primary_intent="SUGGESTION",
        topic_subtype="DICTIONARY_SUGGESTION",
        sentiment="NEUTRAL",
        reason_code="suggestion_eligible",
    )
    assert _LONG_LINGUISTIC_SUGGESTION in prompt
    assert "respond to that reasoning directly" in SYSTEM_INSTRUCTIONS


# --- B / C. Single-character fragments must not reach Gemini ----------------


def test_b_single_letter_waw_is_not_eligible_for_generation():
    result = determine_response_eligibility(
        EligibilityInput(comment_text="و", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION")
    )
    assert result.routing_decision == NO_RESPONSE_RECOMMENDED
    assert result.rule_id == "empty_or_meaningless"


def test_c_single_letter_meem_is_not_eligible_for_generation():
    result = determine_response_eligibility(
        EligibilityInput(comment_text="م", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION")
    )
    assert result.routing_decision == NO_RESPONSE_RECOMMENDED
    assert result.rule_id == "empty_or_meaningless"


def test_b_single_letter_with_diacritic_still_insufficient():
    """A single letter plus tashkeel (e.g. "وَ") must still be caught --
    diacritics are combining marks, not letters, so the count is still 1."""
    result = determine_response_eligibility(EligibilityInput(comment_text="وَ", primary_intent="SUGGESTION"))
    assert result.routing_decision == NO_RESPONSE_RECOMMENDED
    assert result.rule_id == "empty_or_meaningless"


# --- D. Legitimate one-word lexical submissions must NOT be blocked --------


def test_d_two_letter_word_is_eligible():
    """"لا" (no/not) -- a genuine two-letter Arabic word -- must not be
    treated as insufficient merely for being short."""
    result = determine_response_eligibility(
        EligibilityInput(comment_text="لا", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION")
    )
    assert result.routing_decision == GENERATE_DRAFT


def test_d_real_one_word_lexical_submission_is_eligible():
    """"مسك" (musk) -- a real one-word dictionary-entry submission taken
    from the actual source workbook used in the human review -- must
    reach GENERATE_DRAFT, proving this is a character-count gate, not a
    disguised word-count minimum."""
    result = determine_response_eligibility(
        EligibilityInput(comment_text="مسك", primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION")
    )
    assert result.routing_decision == GENERATE_DRAFT
    assert result.rule_id == "generate_suggestion"


# --- E. IDENTIFYING_ONLY grounding must never license claims of ------------
# --- acceptance / implementation / correction / institutional action ------


def test_e_prompt_forbids_confirming_or_correcting_disputed_claims_from_model_knowledge():
    assert (
        "Do not confirm, validate, or correct a visitor's linguistic, scientific, or factual claim "
        "using your own knowledge" in SYSTEM_INSTRUCTIONS
    )
    assert "OFFICIAL KNOWLEDGE is the only source that may settle whether a claim is correct" in SYSTEM_INSTRUCTIONS


def test_e_prompt_with_identifying_context_only_still_forbids_action_claims():
    """When only identifying context is supplied (no official grounding
    text), the OFFICIAL KNOWLEDGE block stays empty and the Hard Rules
    banning acceptance/implementation/action claims are still present
    verbatim in the prompt actually sent to Gemini."""
    prompt = build_generation_prompt(
        comment_text=_LONG_LINGUISTIC_SUGGESTION,
        primary_intent="SUGGESTION",
        topic_subtype="DICTIONARY_SUGGESTION",
        sentiment="NEUTRAL",
        reason_code="suggestion_eligible",
        identifying_context="This identifies WHICH dictionary entry... submission title: \"تركيب\".",
        trusted_grounding_text=None,
    )
    assert "(none — do not invent facts)" in prompt
    assert "IDENTIFYING CONTEXT" in prompt
    assert "NOT evidence for a factual or operational claim" in prompt
    assert "تم تدوين" in prompt  # forbidden-action list still present verbatim
    assert "Do not confirm, validate, or correct" in prompt


# --- F. Existing spam routing unchanged -------------------------------------


def test_f_spam_routing_unchanged():
    result = determine_response_eligibility(
        EligibilityInput(comment_text="اشترِ منتجاتنا الآن رابط في البايو", primary_intent="SPAM")
    )
    # A SPAM label alone is uncorroborated: human review (REVIEW_ONLY) instead of "no response".
    assert result.routing_decision == "REVIEW_ONLY"
    assert result.reason_code == "uncertain_exclusion_requires_review"
    corroborated = determine_response_eligibility(
        EligibilityInput(comment_text="اشترِ منتجاتنا الآن رابط في البايو", primary_intent="SPAM", topic_subtype="OFF_TOPIC_PERSONAL")
    )
    assert corroborated.routing_decision == NO_RESPONSE_RECOMMENDED
    assert corroborated.reason_code == "spam"
    assert corroborated.rule_id == "spam_or_operational_exclude"


# --- G. Existing PII redaction unchanged ------------------------------------


def test_g_pii_redaction_unchanged():
    ok, why = validate_draft("شكرًا لتواصلكم على الرقم 0501234567 مع المعجم.")
    assert ok is False
    assert why == "pii_in_draft"


# --- H. Existing one-refinement maximum unchanged ---------------------------


def test_h_refinement_allowlist_still_excludes_pii_and_internal_leak():
    assert "pii_in_draft" not in REFINEMENT_ELIGIBLE_REASON_CODES
    assert "internal_leak" not in REFINEMENT_ELIGIBLE_REASON_CODES
