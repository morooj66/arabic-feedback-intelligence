"""The attribution rule ("اقتراحكم/مقترحكم" needs a suggestion in the visitor's own text) recognises suggestions by expression in context and stays closed to invented ones.
Regression for the live rg-v14.2 pilot cases #8 and #12 (valid thanks rejected because the suggestion was a wish / a polite request)."""
import pytest

from rcs.response_generation.validate import comment_expresses_suggestion as expresses, validate_draft

C8 = "السلام عليكم ورحمة الله، شكرا لكم على هذا العمل المفيد في خدمة اللغة العربية، وأتمنى أن تعتمد الشروح على أمثلة موثقة وموضوعية من مصادر معروفة، وأن يذكر مصدر كل مثال حتى يطمئن القارئ إلى صحته ويستفيد منه."
C12 = "جزاكم الله خيرا على جهدكم، ممكن ان يضاف مثال تطبيقي لكل كلمة"


@pytest.mark.parametrize("text", [
    "أقترح إضافة خاصية البحث بالجذر", "نقترح تعديل تعريف الكلمة", "هذا اقتراح لتحسين الأمثلة",                    # explicit
    "أتمنى لو كان المثال حقيقيا", "أتمنى أن يضاف مثال", "اتمنى إضافة خاصية جديدة", "ياليت تضيفون خاصية البحث", C8,  # wishes
    "ممكن ان يضاف مثال تطبيقي", "هل يمكن إضافة كلمة جديدة؟", "من الممكن توفير خيار للتحميل", "أرجو إضافة المعنى الثاني", C12,  # polite requests
    "يفضل أن يضاف مثال", "ينبغي تصحيح هذا المثال", "من المفيد توفير خيار للبحث", "أضيفوا هذه الكلمة", "حسّنوا الأمثلة",       # recommendations / imperatives
])
def test_genuine_suggestions_are_recognised(text):
    assert expresses(text) is True


@pytest.mark.parametrize("text", [
    "لا أقترح إضافة هذه الكلمة", "ما أقترح حذفها", "لا أتمنى أن يضاف مثال", "غير ممكن إضافة هذه الكلمة", "ليس من الممكن تعديل التعريف",     # negated
    "أتمنى لكم التوفيق", "أتمنى أن تكونوا بخير", "نتمنى لكم النجاح",                                                         # courtesy wishes, not suggestions
    "ما معنى كلمة سوار؟", "كيف أضيف كلمة جديدة؟", "هل يمكن البحث بدون تشكيل؟", "لماذا حذفت الكلمة؟", "كيف يمكنني إضافة كلمة؟",  # ordinary questions
    "الموقع ما يفتح عندي", "الصفحة الرئيسية: الخط صغير جدا", "شكرا لكم على هذا الجهد", "الترجمة خاطئة", "",               # no suggestion at all
])
def test_non_suggestions_are_not_recognised(text):
    assert expresses(text) is False


def test_exact_rejected_reply_8_now_passes_with_its_comment():
    assert validate_draft("نشكركم على مقترحكم المتعلق بالاعتماد على الأمثلة الموثقة والموضوعية في الشروح.", comment_text=C8) == (True, None)


def test_exact_rejected_reply_12_now_passes_with_its_comment():
    assert validate_draft("جزاكم الله خيرًا على دعواتكم الطيبة، وعلى اقتراحكم بشأن إضافة الأمثلة التطبيقية.", comment_text=C12) == (True, None)


@pytest.mark.parametrize("comment", [
    "خرمدة: ضجيج وفوضى في المكان",            # a bare word submission (synthetic): still no stated suggestion
    "أنشطة لا صفية",                                 # the original audited fabrication case
    "الموقع ما يفتح عندي", "ما معنى كلمة سوار؟", "لا أقترح إضافة هذه الكلمة", "أتمنى لكم التوفيق", "شكرا لكم",
])
def test_invented_suggestions_are_still_rejected(comment):
    ok, why = validate_draft("نشكركم على اقتراحكم القيّم بشأن الكلمة.", comment_text=comment)
    assert (ok, why) == (False, "unsupported_visitor_intent_claim")


def test_other_attribution_families_are_unchanged():
    assert validate_draft("نشكركم على طلبكم الكريم بشأن الكلمة.", comment_text="الموقع جميل")[1] == "unsupported_visitor_intent_claim"
    assert validate_draft("نشكركم على تصحيحكم القيّم للكلمة.", comment_text="الموقع جميل")[1] == "unsupported_visitor_intent_claim"
    assert validate_draft("نشكركم على طلبكم الكريم بشأن الكلمة.", comment_text="أرجو توضيح الكلمة")[1] is None


def test_without_a_comment_the_gate_is_a_no_op_as_before():
    assert validate_draft("نشكركم على اقتراحكم القيّم بشأن الكلمة.") == (True, None)
