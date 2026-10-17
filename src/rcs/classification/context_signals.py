"""Contextual classification safeguards: descriptive signals + CONFLICT FLAGS for human review. OPT-IN, DEFAULT OFF.

What this is
    A small, deterministic, model-free layer that reads a comment's STRUCTURE (a question, a request, a report that something is wrong, dissatisfaction,
    a failure of the platform, praise, a "word: meaning" form, negation of a polar word) and compares the resulting reading with the labels the
    classifiers produced. Where the two disagree on firm evidence, it records a named conflict so a reviewer looks at the item.

What this is NOT
    * It never changes a prediction, a confidence, the taxonomy or a model. `detect_conflicts` returns flags only.
    * It is not a keyword override. Every signal is built from at least two independent evidence families (a cue AND its context: a failure verb AND a
      platform object, an error word AND an entry referent, a polar word AND no negation, ...) and a conflict is raised only when the strong reading and
      the prediction disagree. A single cue word never produces an intent or topic conflict by itself; a sentiment conflict needs an unnegated polar word
      (tone is carried by the wording itself) and absence of evidence is never a conflict.
    * Suggestion subtypes (`suggestion_subtypes`) are DESCRIPTIVE metadata ("word_addition", "entry_content", "platform_feature", "service_process").
      They are not labels and nothing reads them to route.
    * No external call, no I/O, no state. Text is never logged here; conflicts carry codes only.

Reading rules follow LABELING_GUIDE.md: COMPLAINT needs a dissatisfied tone (a calm correction is closer to SUGGESTION); a platform feature idea is
DICTIONARY_SUGGESTION; sentiment is TONE, judged independently from intent (a calm correction is NEUTRAL; praise mixed with a suggestion is not negative).
Enabled only by RCS_CONTEXT_SIGNALS=1 (config.settings.CONTEXT_SIGNALS_ENABLED); see rcs.review.excel_review_flow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import unicodedata

_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed]")
_ALEF = str.maketrans("أإآٱ", "اااا")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ELONG = re.compile(r"(.)\1{2,}")


def normalize_arabic(text) -> str:
    """Same normalisation as the offline experimental helper (diacritics, tatweel, alef variants, digits, «?»→«؟», letter elongation); kept local so this runtime
    module does not depend on the offline-only experimental family."""
    s = unicodedata.normalize("NFKC", str(text if text is not None else "")).replace("ـ", "")
    s = _DIACRITICS.sub("", s).translate(_ALEF).translate(_DIGITS).replace("?", "؟")
    return " ".join(_ELONG.sub(r"\1\1", s).lower().split())

LAYER_NAME = "context_signals_v1"

_AR = "ء-ي"
_PREFIX = r"(?:و|ف|ب|ل|ك|ال|وال|بال|فال|لل|كال)?"
_NEGATORS = {"لا", "ما", "لم", "لن", "ليس", "ليست", "غير", "بدون", "عدم", "مو", "مب", "مش"}


def _fold(p: str) -> str:
    """Normalise a pattern like the text (alef/diacritics), then make ة/ه and ى/ي interchangeable."""
    p = normalize_arabic(p)
    return p.replace("ة", "[ةه]").replace("ى", "[ىي]")


def _words(*words: str) -> "re.Pattern[str]":
    """Single-word cues: whole-token match, optional clitic prefix."""
    return re.compile(rf"(?<![{_AR}]){_PREFIX}(?:" + "|".join(_fold(w) for w in words) + rf")(?![{_AR}])")


def _words_np(*words: str, prefix: str = "(?:و|ف)?") -> "re.Pattern[str]":
    """Whole-token cues that must NOT take the ل/ب/ك/ال clitics (e.g. «لكم» is not the question word «كم»)."""
    return re.compile(rf"(?<![{_AR}]){prefix}(?:" + "|".join(_fold(w) for w in words) + rf")(?![{_AR}])")


def _phrases(*phrases: str) -> "re.Pattern[str]":
    return re.compile(rf"(?<![{_AR}])(?:" + "|".join(_fold(p).replace(" ", r"\s+") for p in phrases) + rf")(?![{_AR}])")


# --- Evidence families -----------------------------------------------------------------------------------------------------------------------------
_Q_WORD = _words_np("هل", "كيف", "لماذا", "متى", "اين", "كم", "ماذا")
_Q_DIALECT = _words_np("ليش", "ليه", "وين", "فين", "شنو", "وش", "ايش", "مين", "منو")      # also ordinary words in slang lists: a question only together with «؟»
_Q_PHRASE = _phrases("ما معنى", "ما هو", "ما هي", "ما الفرق", "ما المقصود", "ما المراد", "ما حكم", "ما اصل", "ما الصواب", "ما سبب", "ما الدليل", "ما تعريف")
_Q_INDIRECT = re.compile(
    rf"(?<![{_AR}])(?:ارجو|اريد|اود|ابغى|ابي|احتاج|ارغب|نرجو|يرجى|الرجاء)\s+(?:\S+\s+){{0,2}}?"
    rf"(?:معرفة|توضيح|شرح|بيان|تفسير|ايضاح|افادة|اجابة|معنى|معلومات|الاستفسار|السؤال)"
    rf"|(?<![{_AR}])(?:اشرحوا|وضحوا|فسروا|بينوا|اخبروني|افيدوني|عرفوني|اشرح|وضح|فسر)(?![{_AR}])"
    rf"|(?<![{_AR}])(?:اسال|استفسر|اتساءل|نستفسر)(?![{_AR}])|(?:لدي|عندي)\s+(?:سؤال|استفسار|تساؤل)"
    rf"|(?<![{_AR}])(?:لا|لم|ما)\s+(?:اعرف|افهم|فهمت|ادري)\s+(?:معنى|ما|كيف|لماذا|اي)"
    rf"|(?:ابحث|بحثت)\s+عن\s+(?:معنى|تعريف|شرح)"
)

_REQUEST = re.compile(
    rf"(?<![{_AR}])(?:اقترح|اقتراح|اقتراحي|مقترح|اتمنى|نتمنى|ياليت|يا\s+ليت|ليت|يفضل|ينبغي|يجب|لابد|لا\s+بد|رجاء|ارجو|نرجو|يرجى|الرجاء)(?![{_AR}])"
    rf"|(?<![{_AR}])لو\s+(?:ان|انكم|تضيفون|يضاف|تتم|تكون|تقومون)"
    rf"|(?<![{_AR}])ممكن\s+(?:ان\s+)?(?:\S+\s+)?(?:تضيفون|يضاف|اضافة|تضاف|تكون|تعدلون|تغيرون|تضعون|توفرون|تفعلون|تضيفوا)"
    rf"|(?<![{_AR}])(?:اضيفوا|اضيفو|اضف|ضيفوا|ادرجوا|اعتمدوا|وفروا|فعلوا|حسنوا|عدلوا|صححوا|غيروا|اجعلوا|ضعوا|اضافة|ادراج|توفير|تفعيل|تحسين)(?![{_AR}])"
)

_ERROR_CUE = _words("خطا", "غلط", "خاطئ", "خاطئه", "خاطيء", "مغلوط", "الصواب", "صوابها", "صحيحها", "الاصح", "تصويب", "مكرر", "تكرار", "ناقص",
                     "نقص", "اقتطاع", "اقتطاعها", "تصحيف", "تحريف", "التباس", "خلط", "سقط")
_ERROR_PHRASE = _phrases("غير صحيح", "غير صحيحه", "غير دقيق", "غير دقيقه", "لا يتناسب", "لا معنى", "لا يوافق", "غير مناسب", "تصحيح المعنى", "صحيح هو")
_ENTRY_REFERENT = _words("المدخل", "المادة", "الماده", "الكلمة", "الكلمه", "التعريف", "المعنى", "الشاهد", "الايه", "الاية", "الترجمة", "الترجمه", "المرادف",
                         "التشكيل", "الاملاء", "العنوان", "الجذر", "البطاقة", "البطاقه", "الشرح", "المقابل", "الحرف", "اللفظ", "المصطلح", "المفردة",
                         "المفردات", "الحديث", "البيت", "المثال", "الامثلة", "معنى", "تعريف", "كلمة", "كلمه", "مادة", "ماده", "شاهد", "ترجمة", "ترجمه")

_AFFECT_WORD = _words("سيء", "سيئ", "سيئه", "سيئة", "سيئا", "سيئين", "رديء", "رديئه", "رديئا", "مزعج", "مزعجه", "مقرف", "فاشل", "فاشله", "فشل", "متدني", "محبط", "احباط",
                       "مؤسف", "للاسف", "عيب", "عار", "اهمال", "مهمل", "مهملين", "تقصير", "مقصرين", "تافه", "قرف", "كارثه", "كارثة", "فضيحه", "فضيحة",
                       "اسوا", "اسوأ", "طفشت", "زهقت", "مللنا", "تعبنا", "مستحيل", "استحي")
_AFFECT_PHRASE = _phrases("غير مقبول", "غير معقول", "لا يعقل", "لا يليق", "ما حد يهتم", "ما احد يهتم", "لا احد يهتم", "ما يصير", "حرام عليكم", "مستوى ضعيف",
                          "مستوى متدني", "لا مبالاة", "اين الاهتمام", "اكثر من مره", "كل مره", "مرارا وتكرارا")
_HEAVY_PUNCT = re.compile(r"!{2,}|[!؟?]{3,}|؟!|!؟")

_TECH_FAILURE = _phrases(
    "لا يعمل", "ما يعمل", "لا تعمل", "ما تعمل", "لا يفتح", "ما يفتح", "لا تفتح", "ما تفتح", "ما يشتغل", "لا يشتغل", "ما تشتغل", "لا تشتغل",
    "لا يحمل", "ما يحمل", "ما تحمل", "لا تحمل", "لا تظهر", "ما تظهر", "لا يظهر", "ما يظهر", "لا يستجيب", "ما يستجيب", "لا يتم تحميل", "لا استطيع الدخول",
    "ما اقدر ادخل", "ما قدرت ادخل", "تعذر الدخول", "تعذر التحميل", "تعذر فتح", "الصفحه بيضاء",
)
_TECH_FAILURE_WORD = _words("متوقف", "توقف", "تعطل", "معطل", "معطله", "يتعطل", "يعلق", "تعليق", "بطيء", "بطئ", "بطيئه", "ثقيل", "ثقيله", "خطأ")
_PLATFORM_OBJECT = _words("الموقع", "التطبيق", "الصفحة", "الصفحه", "الصفحات", "البحث", "الصوت", "الصوتيات", "النطق", "الرابط", "الروابط", "الدخول", "التحميل",
                           "المنصة", "المنصه", "الزر", "الازرار", "الصور", "الشاشة", "الشاشه", "الاتصال", "الايفون", "الاندرويد", "موقعكم", "تطبيقكم")

_PRAISE = re.compile(
    rf"(?<![{_AR}])(?:شكرا|شكراً|اشكركم|نشكركم|شاكر|شاكرين|مشكورين|مشكور|جزاكم|جزاك|بارك|يعطيكم\s+العافيه|الله\s+يعطيكم|تسلمون|احسنتم|ماشاء\s+الله|ما\s+شاء\s+الله|"
    rf"ممتاز|ممتازه|رائع|رائعه|مبدع|مبدعين|ابداع|متميز|متميزه|عظيم|جهد\s+مشكور|جهود\s+مشكوره|عمل\s+جبار|موفقين|يوفقكم|الف\s+شكر)(?![{_AR}])"
)
_POSITIVE_WORD = _words("ممتاز", "ممتازه", "رائع", "رائعه", "رائعا", "مبدع", "جميل", "جميله", "جميلا", "مفيد", "مفيده", "مفيدا", "متميز", "عظيم", "راقي", "اعجبني", "استفدت", "احسنتم", "شكرا")
_NEG_POLARITY_WORD = _words("سيء", "سيئ", "سيئا", "رديء", "رديئا", "مزعج", "مزعجا", "مقرف", "فاشل", "محبط", "تافه", "مخيب", "ضعيف", "ضعيفه", "مؤسف")

_DICT_REFERENT = _words("كلمة", "كلمه", "كلمات", "مفردة", "مفردات", "معجم", "المعجم", "قاموس", "مدخل", "مادة", "ماده", "تعريف", "معنى", "شاهد", "اشتقاق", "جذر",
                         "مصطلح", "لفظ", "ترجمة", "ترجمه", "منصة", "منصه", "موقع", "تطبيق", "سوار")

_GLOSS = re.compile(r"^[^\n:：\-–—=]{1,40}\s*[:：\-–—=]\s*\S")

# Suggestion subtypes: descriptive metadata only.
_SUB_WORD = _phrases("اضافة كلمة", "اضافة مفردة", "اضافة مصطلح", "اضافة لفظ", "ادراج كلمة", "ادراج مفردة", "اقتراح كلمة", "اقتراح مفردة", "اقتراح مصطلح",
                     "اضيفوا كلمة", "اضيفوا مفردة", "تضيفون كلمة", "تضيفون مفردة", "كلمة جديدة", "كلمات جديدة", "مفردة جديدة", "يضاف مصطلح", "تضاف كلمة")
_SUB_CONTENT = _words("امثلة", "مثال", "شواهد", "شاهد", "توثيق", "مصادر", "معاني", "تعاريف", "جذر", "اشتقاق", "نطق", "تشكيل", "مرادفات", "ترجمات", "تفصيل", "توضيح")
_SUB_PLATFORM = _words("ميزة", "ميزه", "خاصية", "خاصيه", "ازرار", "زر", "تطبيق", "فلتر", "تصفية", "تصنيف", "مفضلة", "مفضله", "اشعارات", "واجهة", "واجهه",
                        "تصميم", "اندرويد", "ايفون", "قائمة", "قائمه", "البحث", "بحث", "الوضع", "تحميل", "مشاركة", "حفظ", "طباعة", "نسخة", "نسخه", "لغات", "اللغات")
_SUB_SERVICE = _words("الرد", "الطلبات", "التواصل", "المراجعة", "المراجعه", "الاستجابة", "الاستجابه", "سرعة", "الشكاوى")


_SENT_SPLIT = re.compile(r"(?<=[.!؟\n؛])|(?<=\?)")
_MAX_QUESTION_SENTENCE = 14


def _question_sentence(norm: str) -> bool:
    """An interrogative is evidence of a QUESTION TO THE INSTITUTION only inside a short sentence that is built as a question: the interrogative opens the sentence
    (first two tokens) or the sentence ends with «؟». The same word inside a long narrative, a verse or a word list is not (measured on the development data)."""
    for sent in _SENT_SPLIT.split(norm):
        toks = _tokens(sent)
        if not toks or len(toks) > _MAX_QUESTION_SENTENCE:
            continue
        marked = sent.rstrip().endswith("؟")
        head = " ".join(toks[:2])
        std = _Q_WORD.search(sent)
        if std and (marked or _Q_WORD.search(head)):
            return True
        if marked and _Q_DIALECT.search(sent):
            return True
        if _Q_DIALECT.search(head) and marked:
            return True
    return False


def _tokens(norm: str) -> list[str]:
    return re.findall(rf"[{_AR}]+", norm)


def _negated(tokens: list[str], idx: int, window: int = 2) -> bool:
    return any(t in _NEGATORS for t in tokens[max(0, idx - window): idx])


def _cue_hits(rx: "re.Pattern[str]", norm: str):
    return list(rx.finditer(norm))


def _word_index(norm: str, pos: int) -> int:
    return len(_tokens(norm[:pos]))


@dataclass(frozen=True)
class ContextSignals:
    question: Optional[str]                  # "strong" (interrogative word/phrase or indirect request for information) | "weak" (question mark only) | None
    suggestion_request: bool                 # an unnegated proposal / wish / request-to-change
    suggestion_form: Optional[str]           # "explicit_request" | "bare_submission" (a word or a word:meaning form with no request) | None
    suggestion_subtypes: tuple[str, ...]     # DESCRIPTIVE only: word_addition | entry_content | platform_feature | service_process
    error_report: bool                       # an error cue about a named entry part (cue + referent)
    affect: bool                             # dissatisfaction / blame in the wording (unnegated)
    technical_failure: bool                  # a failure predicate about a platform object
    praise: bool
    polarity: Optional[str]                  # "positive" | "negative" | None, negation-aware
    negation_flipped: bool                   # at least one polar word was negated
    dictionary_referent: bool
    mixed_praise_suggestion: bool            # descriptive


def analyze(text) -> ContextSignals:
    norm = normalize_arabic(text)
    toks = _tokens(norm)

    # -- question
    has_mark = "؟" in norm
    q_strong = _question_sentence(norm) or bool(_Q_PHRASE.search(norm) or _Q_INDIRECT.search(norm))
    question = "strong" if q_strong else ("weak" if has_mark else None)

    # -- request / suggestion (unnegated)
    request = any(not _negated(toks, _word_index(norm, m.start())) for m in _cue_hits(_REQUEST, norm))

    # -- error report: an error cue AND an entry referent
    err_cue = bool(_ERROR_CUE.search(norm) or _ERROR_PHRASE.search(norm))
    err_cue = err_cue and any(not _negated(toks, _word_index(norm, m.start())) for m in _cue_hits(_ERROR_CUE, norm) + _cue_hits(_ERROR_PHRASE, norm))
    error_report = err_cue and bool(_ENTRY_REFERENT.search(norm))

    # -- polarity with negation
    pos_hits = [(m, _negated(toks, _word_index(norm, m.start()))) for m in _cue_hits(_POSITIVE_WORD, norm)]
    neg_hits = [(m, _negated(toks, _word_index(norm, m.start()))) for m in _cue_hits(_NEG_POLARITY_WORD, norm) + _cue_hits(_AFFECT_WORD, norm)]
    flipped = any(n for _, n in pos_hits) or any(n for _, n in neg_hits)
    neg_score = sum(1 for _, n in neg_hits if not n) + sum(1 for _, n in pos_hits if n)
    pos_score = sum(1 for _, n in pos_hits if not n) + sum(1 for _, n in neg_hits if n)
    if _PRAISE.search(norm) and not any(n for _, n in pos_hits):
        pos_score += 1
    polarity = "negative" if neg_score > pos_score else ("positive" if pos_score > neg_score else None)

    # -- affect (dissatisfaction): a lexical cue that is not negated, or a stock complaint phrase, or heavy punctuation WITH a negative cue
    lexical_affect = any(not n for m, n in [(m, _negated(toks, _word_index(norm, m.start()))) for m in _cue_hits(_AFFECT_WORD, norm)])
    affect = lexical_affect or bool(_AFFECT_PHRASE.search(norm)) or (bool(_HEAVY_PUNCT.search(str(text or ""))) and neg_score > 0)

    # -- technical failure: a failure predicate AND a platform object
    failure = bool(_TECH_FAILURE.search(norm)) or any(not _negated(toks, _word_index(norm, m.start()), 1) for m in _cue_hits(_TECH_FAILURE_WORD, norm))
    technical_failure = failure and bool(_PLATFORM_OBJECT.search(norm))

    praise = bool(_PRAISE.search(norm)) and polarity != "negative"
    dictionary_referent = bool(_DICT_REFERENT.search(norm))

    subtypes: list[str] = []
    if request:
        if _SUB_WORD.search(norm):
            subtypes.append("word_addition")
        if _SUB_PLATFORM.search(norm):
            subtypes.append("platform_feature")
        if _SUB_CONTENT.search(norm):
            subtypes.append("entry_content")
        if _SUB_SERVICE.search(norm):
            subtypes.append("service_process")

    gloss = bool(_GLOSS.match(str(text or "").strip())) and len(toks) <= 30
    if request:
        form = "explicit_request"
    elif gloss or (0 < len(toks) <= 4 and question is None and not praise and not affect):
        form = "bare_submission"
    else:
        form = None

    return ContextSignals(question=question, suggestion_request=request, suggestion_form=form, suggestion_subtypes=tuple(subtypes), error_report=error_report,
                          affect=affect, technical_failure=technical_failure, praise=praise, polarity=polarity, negation_flipped=flipped,
                          dictionary_referent=dictionary_referent, mixed_praise_suggestion=praise and request)


@dataclass(frozen=True)
class Conflict:
    dimension: str      # "intent" | "topic" | "sentiment"
    code: str


def detect_conflicts(signals: ContextSignals, intent: Optional[str], topic: Optional[str], sentiment: Optional[str]) -> tuple[Conflict, ...]:
    """Flags only. A conflict needs a firm structural reading that disagrees with the prediction; absence of evidence is never a conflict."""
    s, out = signals, []
    calm_correction = s.error_report and not s.affect and not s.technical_failure
    substantive_ask = s.suggestion_request or s.question == "strong" or s.error_report

    # ---------------- intent
    if intent == "COMPLAINT" and calm_correction and s.polarity != "negative":
        out.append(Conflict("intent", "calm_correction_predicted_complaint"))
    if intent in ("SUGGESTION", "SPAM", "THANKS_PRAISE", "INQUIRY") and s.affect and (s.technical_failure or s.error_report or s.dictionary_referent):
        out.append(Conflict("intent", "dissatisfaction_not_predicted_complaint"))
    # (no "INQUIRY without a question cue" conflict: 80 of 119 labelled INQUIRY comments in the development data have none - they ask by naming a word)
    if intent in ("SPAM", "THANKS_PRAISE", "SUGGESTION", "COMPLAINT") and s.question == "strong" and not s.affect and s.dictionary_referent:
        out.append(Conflict("intent", "question_not_predicted_inquiry"))
    if intent in ("SPAM", "COMPLAINT", "INQUIRY") and s.praise and not substantive_ask and not s.affect:
        out.append(Conflict("intent", "praise_not_predicted_thanks"))
    if intent == "THANKS_PRAISE" and (s.question == "strong" or s.error_report or s.technical_failure or s.suggestion_request):
        out.append(Conflict("intent", "thanks_with_substantive_content"))

    # ---------------- topic
    if topic != "TECHNICAL_ISSUE" and s.technical_failure:
        out.append(Conflict("topic", "platform_failure_not_technical_topic"))
    if topic in ("WORD_SUBMISSION", "GENERAL", "OFF_TOPIC_PERSONAL") and s.error_report:
        out.append(Conflict("topic", "entry_error_report_not_entry_correction"))
    # (no topic conflict for a MISSING signal: most labelled ENTRY_CORRECTION / TECHNICAL_ISSUE comments carry no explicit error or failure wording)
    if topic not in ("DICTIONARY_SUGGESTION", "TECHNICAL_ISSUE") and s.suggestion_request and "platform_feature" in s.suggestion_subtypes and "word_addition" not in s.suggestion_subtypes \
            and not s.error_report:
        out.append(Conflict("topic", "platform_feature_idea_not_dictionary_suggestion"))
    if topic == "OFF_TOPIC_PERSONAL" and s.dictionary_referent and (s.suggestion_request or s.error_report or s.question == "strong"):
        out.append(Conflict("topic", "dictionary_request_predicted_off_topic"))      # OFF_TOPIC_PERSONAL routes to EXCLUDE, so a wrong label here hides a real request

    # ---------------- sentiment
    if sentiment != "NEGATIVE" and s.affect and s.polarity == "negative":
        out.append(Conflict("sentiment", "negative_experience_not_negative"))
    if sentiment == "NEGATIVE" and not s.affect and s.polarity != "negative" and (calm_correction or (s.question == "strong" and s.dictionary_referent) or s.suggestion_request or s.praise):
        out.append(Conflict("sentiment", "calm_content_predicted_negative"))
    if sentiment == "NEGATIVE" and s.polarity == "positive" and s.negation_flipped and not s.affect:
        out.append(Conflict("sentiment", "negated_negative_predicted_negative"))
    if sentiment == "POSITIVE" and s.polarity == "negative" and s.negation_flipped:
        out.append(Conflict("sentiment", "negated_positive_predicted_positive"))
    if sentiment == "NEGATIVE" and s.mixed_praise_suggestion and not s.affect:
        out.append(Conflict("sentiment", "praise_with_suggestion_predicted_negative"))
    return tuple(out)


def apply_to_decision(conflicts: tuple[Conflict, ...], routing_decision: str, routing_reason: Optional[str], needs_review: bool) -> tuple[str, Optional[str], bool]:
    """The ONLY effect of a conflict: it can add a review flag and turn an EXCLUDE into REVIEW_ONLY. It never makes anything more automatic and never edits a label."""
    if not conflicts:
        return routing_decision, routing_reason, needs_review
    if routing_decision == "EXCLUDE":
        return "REVIEW_ONLY", "context_signal_conflict", True
    return routing_decision, routing_reason, True


def summarize(text, intent, topic, sentiment) -> tuple[ContextSignals, tuple[Conflict, ...]]:
    sig = analyze(text)
    return sig, detect_conflicts(sig, intent, topic, sentiment)


def conflict_codes(conflicts: tuple[Conflict, ...]) -> list[str]:
    return [f"{c.dimension}:{c.code}" for c in conflicts]
