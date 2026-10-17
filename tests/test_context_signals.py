"""Contextual classification safeguards: signals + conflict flags only (synthetic strings; no model, no network)."""
import pytest

from config import settings
from rcs.classification import context_signals as C


def codes(text, intent=None, topic=None, sentiment=None):
    return {c.code for c in C.detect_conflicts(C.analyze(text), intent, topic, sentiment)}


# ---- governance -------------------------------------------------------------------------------------------------------
def test_default_off_and_no_prediction_is_ever_returned():
    assert settings.CONTEXT_SIGNALS_ENABLED is False
    out = C.detect_conflicts(C.analyze("الموقع لا يعمل أبدا"), "SUGGESTION", "GENERAL", "NEUTRAL")
    assert all(isinstance(c, C.Conflict) for c in out)               # flags only: no label, confidence or routing value comes back


def test_taxonomy_is_untouched():
    assert set(settings.PRIMARY_INTENTS) >= {"THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM"}
    assert "PLATFORM_FEATURE" not in settings.TOPIC_SUBTYPES and "word_addition" not in settings.TOPIC_SUBTYPES


def test_a_single_cue_word_is_not_a_conflict():
    only = lambda text, i, tp, s: {c.code for c in C.detect_conflicts(C.analyze(text), i, tp, s) if c.dimension in ("intent", "topic")}
    assert only("سيء", "SUGGESTION", "WORD_SUBMISSION", "NEUTRAL") == set()        # affect without a dictionary / platform context
    assert codes("خطأ", "SPAM", "GENERAL", "NEUTRAL") == set()                      # error word without an entry referent
    assert codes("لا يعمل", "SPAM", "GENERAL", "NEUTRAL") == set()                  # failure predicate without a platform object


# ---- intent ----------------------------------------------------------------------------------------------------------
def test_calm_correction_is_not_a_complaint():
    assert "calm_correction_predicted_complaint" in codes("التعريف في مادة سوار غير صحيح والصواب أنه حلية", "COMPLAINT", "ENTRY_CORRECTION", "NEUTRAL")


def test_genuinely_dissatisfied_correction_is_not_flagged_as_calm():
    t = "للأسف التعريف في مادة سوار غير صحيح وهذا إهمال"
    assert "calm_correction_predicted_complaint" not in codes(t, "COMPLAINT", "ENTRY_CORRECTION", "NEGATIVE")
    assert "dissatisfaction_not_predicted_complaint" in codes(t, "SUGGESTION", "ENTRY_CORRECTION", "NEUTRAL")


def test_question_predicted_as_something_else():
    assert "question_not_predicted_inquiry" in codes("هل يوجد معنى آخر لكلمة سوار؟", "SPAM", "OFF_TOPIC_PERSONAL", "NEUTRAL")
    assert "question_not_predicted_inquiry" not in codes("هل يوجد معنى آخر لكلمة سوار؟", "INQUIRY", "GENERAL", "NEUTRAL")


def test_interrogative_inside_a_long_narrative_is_not_a_question():
    long = "كان أبي يقول لنا دائما في المساء عندما نجتمع حول النار ونتحدث عن أيام زمان وعن الناس الذين عرفناهم وأين ذهبوا وكيف تغير كل شيء من حولنا حتى صارت كلمات اللغة غريبة"
    assert C.analyze(long).question is None


def test_ls_in_word_lists_do_not_make_a_question():
    assert C.analyze("لكم عمل رائع").question is None                  # «لكم» is not the question word «كم»
    assert C.analyze("وش").question is None                             # dialect interrogative alone: a word, not a question
    assert C.analyze("وش المقصود بهذه الكلمة؟").question == "strong"


def test_thanks_with_a_substantive_ask_is_flagged_for_review():
    assert "thanks_with_substantive_content" in codes("شكرا لكم أتمنى إضافة ميزة البحث بالجذر", "THANKS_PRAISE", "GENERAL", "POSITIVE")
    assert "thanks_with_substantive_content" not in codes("شكرا لكم على هذا الجهد", "THANKS_PRAISE", "GENERAL", "POSITIVE")


def test_praise_predicted_as_spam():
    assert "praise_not_predicted_thanks" in codes("شكرا جزيلا لكم على هذا العمل الرائع والمفيد", "SPAM", "GENERAL", "POSITIVE")


# ---- suggestion subtypes are descriptive metadata ---------------------------------------------------------------------
@pytest.mark.parametrize("text,subtype", [
    ("أتمنى إضافة ميزة المفضلة في التطبيق", "platform_feature"),
    ("أقترح إضافة أمثلة وشواهد للمداخل", "entry_content"),
    ("أقترح إضافة كلمة جديدة إلى المعجم", "word_addition"),
])
def test_suggestion_subtypes(text, subtype):
    s = C.analyze(text)
    assert s.suggestion_request and subtype in s.suggestion_subtypes and s.suggestion_form == "explicit_request"


def test_bare_word_with_meaning_is_a_bare_submission_not_a_request():
    s = C.analyze("خرمدة: ضجيج وفوضى في المكان")
    assert s.suggestion_form == "bare_submission" and not s.suggestion_request and s.suggestion_subtypes == ()


# ---- topic -----------------------------------------------------------------------------------------------------------
def test_platform_failure_is_a_technical_signal_and_personal_stories_are_not():
    assert "platform_failure_not_technical_topic" in codes("الموقع لا يفتح عندي من أمس", "COMPLAINT", "GENERAL", "NEGATIVE")
    assert C.analyze("ما يشتغل معي الا بالصبر").technical_failure is False


def test_entry_error_report_predicted_as_word_submission():
    assert "entry_error_report_not_entry_correction" in codes("التشكيل في عنوان البطاقة خطأ", "COMPLAINT", "WORD_SUBMISSION", "NEUTRAL")
    assert "entry_error_report_not_entry_correction" not in codes("التشكيل في عنوان البطاقة خطأ", "COMPLAINT", "ENTRY_CORRECTION", "NEUTRAL")


def test_platform_feature_idea_is_not_a_word_submission_or_entry_correction():
    assert "platform_feature_idea_not_dictionary_suggestion" in codes("أتمنى إضافة ميزة المفضلة في التطبيق", "SUGGESTION", "GENERAL", "NEUTRAL")
    assert "platform_feature_idea_not_dictionary_suggestion" not in codes("أتمنى إضافة ميزة المفضلة في التطبيق", "SUGGESTION", "DICTIONARY_SUGGESTION", "NEUTRAL")


def test_dictionary_request_predicted_off_topic_is_flagged_because_that_label_excludes():
    assert "dictionary_request_predicted_off_topic" in codes("أقترح إضافة كلمة جديدة إلى المعجم", "SPAM", "OFF_TOPIC_PERSONAL", "NEUTRAL")
    assert "dictionary_request_predicted_off_topic" not in codes("الله يسعد صباحكم يا أحبة", "SPAM", "OFF_TOPIC_PERSONAL", "NEUTRAL")


def test_no_conflict_for_a_missing_signal():
    assert codes("كلمة", "SUGGESTION", "ENTRY_CORRECTION", "NEUTRAL") == set()
    assert codes("كلمة", "COMPLAINT", "TECHNICAL_ISSUE", "NEUTRAL") == set()


# ---- sentiment -------------------------------------------------------------------------------------------------------
def test_negation_flips_polarity():
    assert C.analyze("هذا الموقع ليس سيئا").polarity == "positive"
    assert C.analyze("ما أعجبني التصميم").polarity == "negative"
    assert C.analyze("الموقع سيء جدا").polarity == "negative"


def test_calm_correction_predicted_negative_and_negative_experience_predicted_neutral():
    assert "calm_content_predicted_negative" in codes("التعريف في مادة سوار غير صحيح والصواب أنه حلية", "SUGGESTION", "ENTRY_CORRECTION", "NEGATIVE")
    assert "negative_experience_not_negative" in codes("الموقع سيء والبحث فيه مزعج", "COMPLAINT", "TECHNICAL_ISSUE", "NEUTRAL")


def test_praise_mixed_with_suggestion_is_not_negative():
    s = C.analyze("شكرا لكم جهد رائع وأتمنى إضافة أمثلة")
    assert s.mixed_praise_suggestion
    assert "praise_with_suggestion_predicted_negative" in codes("شكرا لكم جهد رائع وأتمنى إضافة أمثلة", "SUGGESTION", "DICTIONARY_SUGGESTION", "NEGATIVE")


# ---- the only effect on the workflow -------------------------------------------------------------------------------
def test_conflict_can_only_add_a_review_flag_or_turn_exclude_into_review():
    c = (C.Conflict("topic", "x"),)
    assert C.apply_to_decision((), "EXCLUDE", None, False) == ("EXCLUDE", None, False)
    assert C.apply_to_decision(c, "EXCLUDE", "ai_intent_spam", False) == ("REVIEW_ONLY", "context_signal_conflict", True)
    assert C.apply_to_decision(c, "ACKNOWLEDGE_THEN_REVIEW", None, False) == ("ACKNOWLEDGE_THEN_REVIEW", None, True)
    assert C.apply_to_decision(c, "AUTO_REPLY_CANDIDATE", None, False)[2] is True        # never more automatic


def test_module_has_no_model_or_network_dependency():
    import inspect
    src = inspect.getsource(C)
    for banned in ("requests", "httpx", "urllib", "torch", "joblib", "sklearn", "genai", "sqlite3", "open("):
        assert banned not in src
