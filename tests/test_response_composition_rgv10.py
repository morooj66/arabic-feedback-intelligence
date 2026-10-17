"""
Phase rg-v10 (2026-09) — regression tests for the SECOND real human
review finding: rg-v9 eliminated the specific banned evaluator-noun
openings, but the model settled on a differently-worded, still-detached
"topic-preamble + restatement + thanks" summary shape instead.

These tests deliberately check PROMPT CONTENT/INSTRUCTIONS, never one
exact generated sentence -- Gemini's actual wording is not deterministic
and must not be pinned to a fixture string (per instruction).
"""

from __future__ import annotations

from dataclasses import fields

from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS, build_generation_prompt
from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES
from rcs.response_generation.validate import validate_draft

_LONG_LINGUISTIC_SUGGESTION = (
    "قلتم «التراكيب المتلازمة»اقترح إضافة خاصية البحث بحسب الجذر لتسهيل الوصول إلى الكلمات المشتقةcollocation. "
    "غير أنني أتساءل: لماذا لم تعتمدوا «المتلازمات اللفظية»."
)


# --- 1. Direct, visitor-facing instructions (still present from rg-v9) -----


def test_prompt_still_instructs_direct_reply_not_evaluation():
    assert "Direct reply, not evaluation" in SYSTEM_INSTRUCTIONS
    assert "writing directly TO the visitor" in SYSTEM_INSTRUCTIONS


# --- 2. rg-v10: the new topic-preamble/restatement shape is explicitly ------
# --- named and forbidden, not just the old literal nouns --------------------


def test_prompt_names_and_forbids_the_topic_preamble_restatement_shape():
    assert "Response composition (rg-v10)" in SYSTEM_INSTRUCTIONS
    assert "حول/بخصوص ما طرحتموه" in SYSTEM_INSTRUCTIONS
    assert "compressed restatement" in SYSTEM_INSTRUCTIONS


def test_prompt_forbids_third_person_reportage_framing():
    """"ذكر الزائر..." / "أشار إلى..." -- narrating that the visitor made
    a point, instead of reacting to it -- is a second surface form of
    the same detached-voice problem and must be named explicitly."""
    assert "ذكر الزائر" in SYSTEM_INSTRUCTIONS
    assert "third-person reportage" in SYSTEM_INSTRUCTIONS


def test_prompt_no_longer_offers_the_pattern_that_became_a_template():
    """rg-v9 itself suggested "بخصوص ما تفضلتم به حول..." as a GOOD
    example opening -- the second human review showed Gemini turned
    exactly that suggestion into its new default template. rg-v10 must
    not keep repeating it as a positive example."""
    assert 'بخصوص ما تفضلتم به حول' not in SYSTEM_INSTRUCTIONS


def test_prompt_instructs_composing_around_one_central_point():
    assert "ONE point most deserves a reaction" in SYSTEM_INSTRUCTIONS
    assert "write the reply AROUND that one point" in SYSTEM_INSTRUCTIONS


# --- 3. Adaptive structure -- no required fixed template --------------------


def test_prompt_states_no_required_stage_sequence():
    assert "no required sequence of stages" in SYSTEM_INSTRUCTIONS
    assert "no fixed order" in SYSTEM_INSTRUCTIONS


def test_prompt_still_allows_short_single_sentence_replies():
    """Short valid word submissions (e.g. a one-word lexical submission)
    must not be forced into a longer, multi-stage reply -- the adaptive
    length guidance from earlier versions must still be present verbatim
    alongside the new composition instructions."""
    assert "usually 1 short sentence, maximum 2" in SYSTEM_INSTRUCTIONS
    assert "Do not write a long response merely because the visitor wrote a long message" in SYSTEM_INSTRUCTIONS
    assert "one-sentence reaction to the actual point is often better" in SYSTEM_INSTRUCTIONS


# --- 4. Factual authority / no fabricated acceptance-action (unchanged) ----


def test_official_knowledge_remains_the_only_factual_authority():
    assert "OFFICIAL KNOWLEDGE is the only source that may settle whether a claim is correct" in SYSTEM_INSTRUCTIONS


def test_no_fabricated_acceptance_or_action_claims():
    assert "تم تدوين" in SYSTEM_INSTRUCTIONS
    assert "never claim acceptance or implementation" in SYSTEM_INSTRUCTIONS


def test_detailed_linguistic_suggestion_can_be_acknowledged_without_confirming_correctness():
    prompt = build_generation_prompt(
        comment_text=_LONG_LINGUISTIC_SUGGESTION,
        primary_intent="SUGGESTION",
        topic_subtype="ENTRY_CORRECTION",
        sentiment="NEUTRAL",
        reason_code="suggestion_eligible",
        identifying_context=None,
        trusted_grounding_text=None,
    )
    assert _LONG_LINGUISTIC_SUGGESTION in prompt
    assert "(none — do not invent facts)" in prompt
    assert "never confirm or correct their linguistic reasoning as fact" in SYSTEM_INSTRUCTIONS


# --- 5. No new phrase blacklist was grown; existing narrow gate untouched --


def test_evaluator_voice_gate_was_not_expanded_with_new_literal_phrases():
    """rg-v10 is a prompt-only fix -- the validator's six-literal gate
    from rg-v9 must remain exactly as narrow as before, not grown to
    include "حول ما طرحتموه"/"بخصوص ما عرضتموه" or similar."""
    from rcs.response_generation.validate import _EVALUATOR_VOICE_OPENINGS

    assert len(_EVALUATOR_VOICE_OPENINGS) == 6
    for banned_growth in ("حول ما طرحتموه", "بخصوص ما عرضتموه", "حول/بخصوص"):
        assert banned_growth not in _EVALUATOR_VOICE_OPENINGS


def test_the_new_preamble_pattern_still_passes_validation_by_design():
    """The topic-preamble/restatement shape is a COMPOSITION problem the
    prompt now addresses -- it is deliberately NOT a validator failure,
    since a phrase-based gate here would risk false positives on
    legitimate uses of "بخصوص"/"حول" (per the task's explicit
    instruction not to blacklist these). This test documents that
    choice: such a draft still passes validate_draft."""
    draft = "بخصوص ما عرضتموه بشأن المصطلح المقترح، نشكركم على هذا التوضيح."
    ok, why = validate_draft(draft)
    assert ok is True and why is None


# --- 6. Existing gates / refinement invariant remain intact -----------------


def test_pii_gate_unchanged():
    ok, why = validate_draft("شكرًا لتواصلكم على الرقم 0501234567 مع المعجم.")
    assert ok is False and why == "pii_in_draft"


def test_internal_leak_gate_unchanged():
    ok, why = validate_draft("شكرًا، نموذج MARBERTv2 واثق بنسبة confidence عالية.")
    assert ok is False and why == "internal_leak"


def test_unsupported_claim_gate_unchanged():
    ok, why = validate_draft("تم تحويل ملاحظتك إلى فريق المحتوى.")
    assert ok is False and why == "unsupported_operational_claim"


def test_refinement_allowlist_unchanged_by_this_prompt_only_change():
    """Pinned as of rg-v10's own (prompt-only) change. The Ambiguous/
    Insufficient-Context Response Handling milestone later added
    "unsupported_visitor_intent_claim" to this allowlist (a genuine,
    separately-justified refinement-eligibility extension, not a change
    to rg-v10 itself) -- this assertion is updated to include it rather
    than pinning a now-stale set."""
    # The reply-safety milestone later ADDED refinable codes (see tests/test_reply_grounding_and_refinement.py for the exact set); this milestone's own
    # set must still be contained in it.
    assert frozenset(
        {
            "unsupported_operational_claim",
            "generic_boilerplate",
            "empty_output",
            "too_short",
            "excessive_length",
            "missing_arabic",
            "evaluator_voice_opening",
            "unsupported_visitor_intent_claim",
        }
    ) <= REFINEMENT_ELIGIBLE_REASON_CODES
