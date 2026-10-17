"""CANDIDATE v2 Jev instructions (prepared, NOT yet evaluated; no paid call has been made with them).

Derived from the observed development errors of the v1 prompt (see experiments_report.md) and constrained to
what LABELING_GUIDE.md already says -- nothing here changes the taxonomy or relabels anything:

  * Topic WORD_SUBMISSION vs ENTRY_CORRECTION: v1 sent 19 of 24 gold WORD_SUBMISSION rows to ENTRY_CORRECTION
    (visitors who type a word plus its meaning/example/etymology). v2 states that supplying lexical content for a
    word is a submission, and reserves ENTRY_CORRECTION for an explicit statement that an existing entry is wrong
    or deficient.
  * Intent COMPLAINT: v1 flagged 13 rows (4 correct); 9 false flags were polite or neutral notes that an entry is
    wrong. The guide says COMPLAINT needs frustration/blame in the tone and prefers SUGGESTION when 50/50; v2 says so.

Deliberately NOT encoded (label-policy questions for the dataset owners, not prompt problems):
  * bare "word + meaning" rows labelled INQUIRY in Gold (contradicts the guide's "genuine question" definition);
  * NEGATIVE sentiment on test/gibberish/short-error messages (the guide ties NEGATIVE to an irritated/angry tone).

Any evaluation of v2 must be run on data NOT used to write it (the 94 validation rows motivated these edits).
"""

from __future__ import annotations

from rcs.classification.jev_criteria import INTENT_CRITERIA, QUESTIONS, SENTIMENT_CRITERIA, TOPIC_CRITERIA

V2_VERSION = "jev-criteria-v2-candidate"

_INTENT_V2 = dict(INTENT_CRITERIA)
_INTENT_V2["COMPLAINT"] = (
    "إحباط أو لوم أو استياء ظاهر في النبرة (مثل «الخدمة سيئة»، «لا يعمل ولا أحد يهتم»). "
    "مجرد الإشارة الهادئة إلى خطأ في مدخل أو اقتراح تصويبه ليس شكوى بل اقتراح، "
    "ولا يُعدّ شكوى ما لم تظهر نبرة غضب أو لوم؛ وعند التردد بين الشكوى والاقتراح يُفضَّل الاقتراح."
)

_TOPIC_V2 = dict(TOPIC_CRITERIA)
_TOPIC_V2["WORD_SUBMISSION"] = (
    "تقديم كلمة أو عبارة أو مدخل جديد للمعجم، بما في ذلك كتابة الكلمة مع معناها أو مثالها أو أصلها أو شرحها "
    "كمحتوى معجمي مقترح، حتى لو بدا النص كمدخل معجمي مكتمل. إن قال الزائر صراحةً إن مدخلًا موجودًا خاطئ أو ناقص فهو تصحيح مدخل."
)
_TOPIC_V2["ENTRY_CORRECTION"] = (
    "قول صريح بأن مدخلًا موجودًا في المعجم خاطئ أو ناقص أو مكرر أو غير متسق (معنى أو إملاء أو قواعد أو مثال أو مضاد)، "
    "مع الإشارة إلى كلمة أو تعريف محدد. مجرد كتابة معنى أو شرح لكلمة دون ادعاء وجود خطأ هو تقديم كلمة."
)

CANDIDATE_QUESTIONS = {
    "intent": (QUESTIONS["intent"][0], _INTENT_V2, QUESTIONS["intent"][2]),
    "topic": (QUESTIONS["topic"][0], _TOPIC_V2, QUESTIONS["topic"][2]),
    "sentiment": (QUESTIONS["sentiment"][0], dict(SENTIMENT_CRITERIA), QUESTIONS["sentiment"][2]),
}

for _name, (_i, _c, _labels) in CANDIDATE_QUESTIONS.items():
    if set(_c) != set(_labels):
        raise RuntimeError(f"v2 {_name} criteria drifted from the frozen taxonomy")

CHANGED = {"intent": ("COMPLAINT",), "topic": ("WORD_SUBMISSION", "ENTRY_CORRECTION"), "sentiment": ()}
