"""Focused rg-v14.3 regressions; deterministic and network-free."""
from config import settings
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS as S
from rcs.response_generation.validate import validate_draft


def test_version_and_block_present():
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    assert "Reply integrity (rg-v14.3, opener guidance rg-v14.4)" in S


def test_word_plus_meaning_is_acknowledged_neutrally():
    assert "Do not call it a suggestion, proposal, question or request" in S
    assert "do not restate, endorse or evaluate the meaning they gave" in S


def test_reported_errors_are_not_conceded():
    assert "Never state or imply the error exists" in S
    assert "never apologise for or admit an error, inaccuracy or technical failure" in S
    assert "never concede that the entry is wrong" in S
    assert "apologize only for their inconvenience, never admit a fault" in S


def test_labels_are_routing_hints_only():
    assert "are routing hints only; the visitor's own words decide the reply" in S
    assert "A predicted COMPLAINT is never a reason to apologise or concede a fault" in S


def test_arabic_quality_guidance():
    assert 'never "على مشاركتنا"' in S
    assert "Never open every reply with «نشكركم على»" in S


def test_existing_safety_and_grounding_rules_untouched():
    for needle in ("never supply or paraphrase a definition from your own knowledge",
                   "Do not confirm, validate, or correct a visitor's linguistic, scientific, or factual claim",
                   "OFFICIAL KNOWLEDGE is the only source that may settle",
                   "The sole fixed navigation exception approved for this prompt",
                   "Do not fabricate facts, policies, promises, timelines, or completed actions"):
        assert needle in S


def test_validators_still_reject_what_the_prompt_forbids():
    assert validate_draft("نعتذر عن الخطأ الوارد في التشكيل.", comment_text="التشكيل خاطئ")[1] == "unsupported_correctness_claim"
    assert validate_draft("نشكركم على اقتراحكم.", comment_text="خرمدة: ضجيج وفوضى في المكان")[1] == "unsupported_visitor_intent_claim"


# ---- final consolidation (rg-v14.4) -----------------------------------------------------------------------------------------------------------
import pytest
from rcs.response_generation.prompt import _build_refinement_instruction_block
from rcs.response_generation.refinement import REFINEMENT_GUIDANCE, RefinementContext, is_refinement_eligible


@pytest.mark.parametrize("draft", ["نشكركم على مشاركتنا مقترحاتكم وملاحظاتكم.", "شكرًا على اقتراحنا.", "نشكركم على تنبيهنا."])
def test_wrong_possessive_is_rejected(draft):
    assert validate_draft(draft, comment_text="أقترح إضافة أمثلة")[1] == "wrong_pronoun_construction"


@pytest.mark.parametrize("draft", ["نشكركم على مشاركتكم مقترحاتكم.", "نشكركم على تزويدنا بالتفاصيل.", "نشكركم على تواصلكم معنا.", "نقدّر مشاركتنا للمعرفة."])
def test_valid_constructions_are_not_banned(draft):
    assert validate_draft(draft, comment_text="أقترح إضافة أمثلة")[1] != "wrong_pronoun_construction"


def test_pronoun_failure_is_refinable_with_guidance():
    assert is_refinement_eligible("FAILED_REVIEW_ONLY", "wrong_pronoun_construction")
    assert "على مشاركتكم" in REFINEMENT_GUIDANCE["wrong_pronoun_construction"]


def test_repair_block_keeps_the_specific_topic_and_stays_generic_for_other_codes():
    blk = _build_refinement_instruction_block(RefinementContext(failure_reason_code="unsupported_assertion", previous_draft="x"))
    assert "keep the specific term, correction or point they raised" in blk and "do not fall back to a generic thank-you" in blk


def test_opener_guidance_is_situation_matched_not_banned():
    assert "Never open every reply with «نشكركم على»" in S and "«شكرًا لمشاركتكم…»" in S
