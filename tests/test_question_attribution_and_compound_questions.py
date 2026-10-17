"""P1 compound questions, P2 repair guidance, P3 question/inquiry attribution (synthetic strings only)."""
import pytest

from rcs.response_generation import claim_gates as C
from rcs.response_generation.refinement import REFINEMENT_GUIDANCE
from rcs.response_generation.validate import comment_inquiry_evidence, validate_draft

WORD = "خرمدة: ضجيج وفوضى في المكان"


def _code(draft, comment):
    return validate_draft(draft, comment_text=comment)[1]


# ---- P1 ----------------------------------------------------------------------------------
@pytest.mark.parametrize("draft", [
    "هل تقصدون الكلمة، أم تريدون معناها؟",
    "هل تقصدون الكلمة، أو تريدون شرحها؟",
    "ما الكلمة التي تقصدونها، وفي أي سياق وردت؟",
])
def test_compound_questions_are_not_split_into_unsupported_statements(draft):
    assert not C.unsupported_assertion(draft, None, None)


@pytest.mark.parametrize("draft", [
    "هل تقصدون الكلمة، علماً أن المعجم يعتمد هذا المعنى؟",
    "نشكركم، أم يعتمد المعجم هذا المعنى.",
    "الموقع ما يفتح",
])
def test_declarative_clauses_stay_flagged(draft):
    assert C.unsupported_assertion(draft, None, None)


# ---- P2 ----------------------------------------------------------------------------------
def test_repair_guidance_for_visitor_intent_claim():
    g = REFINEMENT_GUIDANCE["unsupported_visitor_intent_claim"]
    assert "suggested" in g and "requested" in g and "asked" in g and "do not ask what they meant" in g


# ---- P3 ----------------------------------------------------------------------------------
@pytest.mark.parametrize("comment,expected", [
    ("ما معنى كلمة سوار", "strong"),
    ("هل يوجد شرح؟", "strong"),
    ("أرجو توضيح معنى هذه الكلمة", "strong"),
    ("اشرحوا لي هذه الكلمة", "strong"),
    ("لا أعرف معنى هذه الكلمة", "strong"),
    ("لدي سؤال عن الكلمة", "strong"),
    ("سوار؟", "weak"),
    (WORD, None),
    ("لا أسأل عن شيء", None),
    ("شكرا لكم", None),
    ("أقترح إضافة كلمة", None),
])
def test_inquiry_evidence(comment, expected):
    assert comment_inquiry_evidence(comment) == expected


@pytest.mark.parametrize("draft", ["نشكركم على سؤالكم عن «خرمدة».", "نشكركم على استفساركم.", "نشكركم على تساؤلكم."])
def test_invented_question_is_rejected(draft):
    assert _code(draft, WORD) == "unsupported_visitor_intent_claim"


def test_neutral_acknowledgment_passes():
    assert _code("نشكركم على مشاركة كلمة «خرمدة» ومعناها.", WORD) is None


def test_genuine_question_supports_attribution():
    assert _code("نشكركم على استفساركم عن معنى «سوار».", "ما معنى سوار") is None
    assert _code("نشكركم على استفساركم.", "أرجو توضيح معنى الكلمة") is None


def test_question_mark_alone_supports_only_generic_attribution():
    assert _code("نشكركم على استفساركم.", "سوار؟") is None
    assert _code("نشكركم على استفساركم عن الكلمة.", "سوار؟") == "unsupported_visitor_intent_claim"


def test_quoted_term_must_appear_in_comment():
    assert _code("نشكركم على استفساركم عن «قلادة».", "ما معنى سوار") == "unsupported_visitor_intent_claim"
    assert _code("نشكركم على استفساركم عن «الخرمدة».", "أرجو توضيح خرمدة") is None


def test_existing_suggestion_attribution_unchanged():
    assert _code("نشكركم على اقتراحكم.", "الموقع جميل") == "unsupported_visitor_intent_claim"
    assert _code("نشكركم على اقتراحكم.", "أتمنى لو يضاف بحث بالجذر") is None


def test_visitor_naming_their_own_inquiry_is_strong_evidence():
    assert comment_inquiry_evidence("استفسار بشأن الكلمة") == "strong"
