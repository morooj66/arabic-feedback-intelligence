"""Focused rg-v14.1 regressions; deterministic and network-free."""

from __future__ import annotations

import pytest

from config import settings
from rcs.response_generation.prompt import REFERENCE_HOME_URL, SYSTEM_INSTRUCTIONS
from rcs.response_generation.validate import validate_draft


def test_rg_v14_1_is_the_current_traceable_prompt_version():
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"
    assert "Natural concise composition (rg-v14.1)" in SYSTEM_INSTRUCTIONS


def test_rg_v14_1_requires_one_natural_acknowledgment_at_most():
    assert "Use at most one acknowledgment or courtesy clause" in SYSTEM_INSTRUCTIONS
    assert 'Never stack two acknowledgments such as "نقدّر... ونشكر..."' in SYSTEM_INSTRUCTIONS
    assert 'Do not default to "نقدّر", "نعتز", "يسرنا", or "نشكر"' in SYSTEM_INSTRUCTIONS
    assert "Do not repeat or paraphrase the visitor's comment" in SYSTEM_INSTRUCTIONS
    assert "plain, contemporary professional Arabic" in SYSTEM_INSTRUCTIONS


def test_rg_v14_1_navigation_requires_explicit_visitor_intent():
    assert REFERENCE_HOME_URL == "https://dictionary-platform.invalid/home"
    assert "A bare word, phrase, or ambiguous fragment is not a meaning-lookup or navigation request" in SYSTEM_INSTRUCTIONS
    assert "only when the visitor explicitly requests navigation or a meaning lookup" in SYSTEM_INSTRUCTIONS
    assert "do not include the homepage URL" in SYSTEM_INSTRUCTIONS


def test_rg_v14_1_does_not_authorize_invented_ui_or_procedures():
    assert "Do not name UI controls, menus, buttons, search bars, screens, or navigation procedures" in SYSTEM_INSTRUCTIONS
    assert "approval of the homepage URL authorizes only the URL itself" in SYSTEM_INSTRUCTIONS
    assert "Do not mention official channels, contact methods, departments, policies, or procedures" in SYSTEM_INSTRUCTIONS


def test_rg_v14_1_preserves_ungrounded_lexical_and_review_safety():
    assert "never supply or paraphrase a definition from your own knowledge" in SYSTEM_INSTRUCTIONS
    assert "never invent the answer, promise escalation or resolution" in SYSTEM_INSTRUCTIONS
    assert "do not pretend the question was answered" in SYSTEM_INSTRUCTIONS
    assert "This draft is NEVER sent automatically" in SYSTEM_INSTRUCTIONS
    assert "A human chooses SEND_AS_IS, EDIT_AND_SEND, or NO_RESPONSE_NEEDED" in SYSTEM_INSTRUCTIONS


@pytest.mark.parametrize(
    "comment_text",
    (
        "أعجبني وضوح المنصة، وأتمنى إضافة خاصية مقارنة معاني الكلمة.",
        "من المفيد عرض الأمثلة في موضع أوضح.",
        "أتمنى توفير خيار لتكبير خط الشرح.",
    ),
)
def test_explicit_suggestion_phrases_support_suggestion_attribution(comment_text):
    ok, reason = validate_draft(
        "شكرًا لاقتراحكم بشأن تسهيل استخدام المنصة.",
        comment_text=comment_text,
    )
    assert ok is True
    assert reason is None


@pytest.mark.parametrize(
    "comment_text",
    (
        "السَّديم",
        "أنشطة لا صفية",
        "الخط في الصفحة",
    ),
)
def test_ambiguous_fragments_still_fail_unsupported_suggestion_attribution(comment_text):
    ok, reason = validate_draft(
        "شكرًا لاقتراحكم بشأن تسهيل استخدام المنصة.",
        comment_text=comment_text,
    )
    assert ok is False
    assert reason == "unsupported_visitor_intent_claim"

