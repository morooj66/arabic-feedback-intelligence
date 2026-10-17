"""Focused rg-v14.2 regressions; deterministic and network-free."""

from __future__ import annotations

import pytest

from config import settings
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS
from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES
from rcs.response_generation.validate import validate_draft


def test_rg_v14_2_is_the_current_traceable_prompt_version():
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    assert "Evidence-backed response choices (rg-v14.2)" in SYSTEM_INSTRUCTIONS


def test_bare_fragment_requires_neutral_clarification_without_invented_motive():
    assert "ask one short neutral clarification question" in SYSTEM_INSTRUCTIONS
    assert "invent motives such as interest in lexical enrichment" in SYSTEM_INSTRUCTIONS

    ok, reason = validate_draft(
        "نرحب بمشاركتكم لمفردة «السَّديم»، ونشكر اهتمامكم بالإثراء المعجمي.",
        comment_text="السَّديم",
    )
    assert ok is False
    assert reason == "generic_boilerplate"


def test_invented_lexical_enrichment_motive_fails_without_stacked_courtesy():
    ok, reason = validate_draft(
        "نشكر اهتمامكم بالإثراء المعجمي.",
        comment_text="السَّديم",
    )
    assert ok is False
    assert reason == "unsupported_visitor_intent_claim"


def test_explicit_lexical_enrichment_evidence_still_passes_attribution_gate():
    ok, reason = validate_draft(
        "نشكر اهتمامكم بالإثراء المعجمي.",
        comment_text="أهتم بالإثراء المعجمي وأرغب في المساهمة بهذه الكلمة.",
    )
    assert ok is True
    assert reason is None


@pytest.mark.parametrize(
    "draft",
    (
        "يؤسفنا أن الشرح لم يكن واضحًا، ونقدّر تنبيهكم.",
        "نرحب باستفساركم، ونشكر اهتمامكم.",
        "نقدّر اقتراحكم، ويسرنا اهتمامكم بالمنصة.",
    ),
)
def test_stacked_courtesy_fails_validation_and_can_refine_once(draft):
    ok, reason = validate_draft(draft, comment_text="أتمنى إضافة هذه الخاصية.")
    assert ok is False
    assert reason == "generic_boilerplate"
    assert reason in REFINEMENT_ELIGIBLE_REASON_CODES


def test_one_brief_apology_is_not_rejected():
    ok, reason = validate_draft(
        "نأسف لأن الشرح الحالي لم يوضح المعنى المطلوب.",
        comment_text="قرأت الشرح ولم أفهم المعنى.",
    )
    assert ok is True
    assert reason is None


def test_ungrounded_already_searched_inquiry_requires_plain_limitation():
    assert "an accurate simplification cannot be supplied from the trusted information available" in SYSTEM_INSTRUCTIONS
    assert "Do not convert the unmet request into praise, gratitude" in SYSTEM_INSTRUCTIONS


def test_mixed_suggestion_must_engage_value_not_repeat_and_thank():
    assert "engage the practical benefit or problem the explicit suggestion addresses" in SYSTEM_INSTRUCTIONS
    assert 'Do not merely repeat the suggestion and attach "نقدّر/نشكر"' in SYSTEM_INSTRUCTIONS


def test_human_review_and_grounding_boundaries_remain_explicit():
    assert "This draft is NEVER sent automatically" in SYSTEM_INSTRUCTIONS
    assert "A human chooses SEND_AS_IS, EDIT_AND_SEND, or NO_RESPONSE_NEEDED" in SYSTEM_INSTRUCTIONS
    assert "never supply or paraphrase a definition from your own knowledge" in SYSTEM_INSTRUCTIONS
    assert "never say or imply it has been done, is being done, or will be done" in SYSTEM_INSTRUCTIONS
