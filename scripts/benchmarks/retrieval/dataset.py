"""
Phase M5B.2 — synthetic Arabic retrieval benchmark dataset.

EVALUATION-ONLY. Every string in this file is hand-written for this
benchmark. None of it is a real visitor comment, a real reviewer reply,
or derived from any production database (PostgreSQL, MongoDB, or any
organization system). No visitor_name/visitor_email/comment_id/
source_file field exists anywhere here -- this is intentionally NOT
shaped like a `ReviewItem` row, so it can never be mistaken for one or
accidentally merged into production data.

Grounded in the CURRENT frozen taxonomy only (config.settings
PRIMARY_INTENTS / TOPIC_SUBTYPES) -- no label invented. Ground truth
(`relevant_candidate_ids`) is authored independently of any embedding
model: it reflects the human judgment "this query and this candidate
describe the same underlying case," decided before any candidate model
was run.

CANDIDATES = the corpus being retrieved from (~26 short case
descriptions, each representing one distinct visitor-comment scenario).
QUERIES = the retrieval inputs (~28 paraphrases/variations of a subset
of candidates, in Modern Standard Arabic, Saudi/Gulf conversational
wording, with spelling variation and light/no punctuation) plus a few
candidates with NO query pointing at them, acting as distractors that
make retrieval non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str
    intent: str
    topic: str


@dataclass(frozen=True)
class Query:
    query_id: str
    query_text: str
    relevant_candidate_ids: tuple[str, ...]
    variation_type: str
    intent: str | None = None
    topic: str | None = None
    difficulty: str = "normal"


# --- Candidate corpus (26 items) --------------------------------------
# Every intent/topic value below is taken verbatim from
# config.settings.PRIMARY_INTENTS / TOPIC_SUBTYPES.

CANDIDATES: tuple[Candidate, ...] = (
    Candidate("C01", "نشكركم على جهودكم الرائعة في تطوير المعجم.", "THANKS_PRAISE", "GENERAL"),
    Candidate("C02", "شكرًا لقبول اقتراحي بإضافة كلمة جديدة للمعجم.", "THANKS_PRAISE", "WORD_SUBMISSION"),
    Candidate("C03", "الموقع لا يفتح عندي منذ يومين ولا أستطيع الدخول إليه.", "COMPLAINT", "TECHNICAL_ISSUE"),
    Candidate("C04", "الصوت الخاص بنطق الكلمة لا يعمل عند الضغط عليه.", "COMPLAINT", "TECHNICAL_ISSUE"),
    Candidate("C05", "المعنى المكتوب في هذا المدخل غير صحيح تمامًا.", "COMPLAINT", "ENTRY_CORRECTION"),
    Candidate("C06", "أقترح تصحيح الخطأ الإملائي الموجود في كتابة هذه الكلمة.", "SUGGESTION", "ENTRY_CORRECTION"),
    Candidate("C07", "أقترح إضافة أمثلة توضيحية أكثر لهذا المدخل.", "SUGGESTION", "DICTIONARY_SUGGESTION"),
    Candidate("C08", "أرغب في ترشيح كلمة جديدة لم تُدرج في المعجم بعد.", "SUGGESTION", "WORD_SUBMISSION"),
    Candidate("C09", "ما معنى هذه الكلمة بالضبط؟", "INQUIRY", "GENERAL"),
    Candidate("C10", "كيف يمكنني إضافة كلمة جديدة إلى المعجم؟", "INQUIRY", "WORD_SUBMISSION"),
    Candidate("C11", "هل هذه الكلمة مؤنثة أم مذكرة؟", "INQUIRY", "GENERAL"),
    Candidate("C12", "اربح الآن جوائز مجانية فورية عبر هذا الرابط الحصري!!!", "SPAM", "OFF_TOPIC_PERSONAL"),
    Candidate("C13", "لدي مشكلة شخصية مع جاري ولا علاقة لها بالمعجم إطلاقًا.", "COMPLAINT", "OFF_TOPIC_PERSONAL"),
    Candidate("C14", "أقترح تحسين تصميم الموقع ليكون أسهل في الاستخدام.", "SUGGESTION", "GENERAL"),
    Candidate("C15", "خاصية البحث لا تعرض أي نتائج مهما كتبت فيها.", "COMPLAINT", "TECHNICAL_ISSUE"),
    Candidate("C16", "فريق العمل متميز جدًا وأشكركم على هذا المجهود الكبير.", "THANKS_PRAISE", "GENERAL"),
    Candidate("C17", "كيف يمكنني الإبلاغ عن خطأ تقني في الموقع؟", "INQUIRY", "TECHNICAL_ISSUE"),
    Candidate("C18", "أرشح كلمة (تمكين) لتكون كلمة العام هذا العام.", "SUGGESTION", "WORD_OF_YEAR_NOMINATION"),
    Candidate("C19", "طريقة نطق الكلمة في التسجيل الصوتي المرفق خاطئة.", "COMPLAINT", "ENTRY_CORRECTION"),
    Candidate("C20", "أقترح إضافة مرادفات لهذه الكلمة لتوضيح معناها أكثر.", "SUGGESTION", "DICTIONARY_SUGGESTION"),
    Candidate("C21", "هل تم تعديل الخطأ الذي أبلغت عنه في مدخل سابق؟", "INQUIRY", "ENTRY_CORRECTION"),
    Candidate("C22", "الموقع بطيء جدًا عند التصفح ويستغرق وقتًا طويلاً ليحمّل.", "COMPLAINT", "TECHNICAL_ISSUE"),
    Candidate("C23", "أتمنى إضافة وضع ليلي (دارك مود) لواجهة الموقع.", "SUGGESTION", "GENERAL"),
    Candidate("C24", "المعجم مفيد جدًا ويسعدني استخدامه بشكل يومي.", "THANKS_PRAISE", "GENERAL"),
    Candidate("C25", "من أين جاء أصل هذه الكلمة تاريخيًا؟", "INQUIRY", "GENERAL"),
    Candidate("C26", "أواجه مشكلة في تسجيل الدخول إلى حسابي في خدمة أخرى غير المعجم.", "COMPLAINT", "OFF_TOPIC_PERSONAL"),
)

_CANDIDATE_IDS = frozenset(c.candidate_id for c in CANDIDATES)


# --- Queries (28 items) -------------------------------------------------
# variation_type values: msa, gulf_colloquial, spelling_variation,
# no_punctuation, short, paraphrase (a query may combine more than one
# real-world trait; one label is picked for the DOMINANT trait being
# tested).

QUERIES: tuple[Query, ...] = (
    # -- technical issue / site not opening (paraphrases of C03) --
    Query("Q01", "الموقع ما يفتح عندي", ("C03",), "gulf_colloquial", "COMPLAINT", "TECHNICAL_ISSUE"),
    Query("Q02", "أواجه مشكلة في الوصول إلى الموقع من يومين", ("C03",), "paraphrase", "COMPLAINT", "TECHNICAL_ISSUE"),
    Query("Q03", "لا استطيع الدخول للموقع", ("C03",), "spelling_variation", "COMPLAINT", "TECHNICAL_ISSUE"),
    # -- audio pronunciation not working (paraphrases of C04) --
    Query("Q04", "زر الاستماع للنطق ما يشتغل", ("C04",), "gulf_colloquial", "COMPLAINT", "TECHNICAL_ISSUE"),
    Query("Q05", "الصوت ماله صوت لما اضغط عليه", ("C04",), "gulf_colloquial", "COMPLAINT", "TECHNICAL_ISSUE"),
    # -- entry meaning is wrong (paraphrases of C05) --
    Query("Q06", "المعنى المكتوب خطأ", ("C05",), "short", "COMPLAINT", "ENTRY_CORRECTION"),
    Query("Q07", "التعريف الموجود لهذه الكلمة غير دقيق أبدًا", ("C05",), "paraphrase", "COMPLAINT", "ENTRY_CORRECTION"),
    # -- spelling correction suggestion (paraphrases of C06) --
    Query("Q08", "كتبت الكلمة غلط وابغى اعدلها", ("C06",), "gulf_colloquial", "SUGGESTION", "ENTRY_CORRECTION"),
    Query("Q09", "في خطا املائي في المدخل ياليت يتصحح", ("C06",), "spelling_variation", "SUGGESTION", "ENTRY_CORRECTION"),
    Query("Q10", "ملاحظة تتعلق بخطأ في كتابة المدخل", ("C06",), "paraphrase", "SUGGESTION", "ENTRY_CORRECTION"),
    # -- add more examples request (paraphrases of C07) --
    Query("Q11", "ياليت تزيدون أمثلة توضح استخدام الكلمة", ("C07",), "gulf_colloquial", "SUGGESTION", "DICTIONARY_SUGGESTION"),
    # -- nominate a new word (paraphrases of C08) --
    Query("Q12", "عندي كلمة اقترحها ما هي موجودة عندكم", ("C08",), "gulf_colloquial", "SUGGESTION", "WORD_SUBMISSION"),
    Query("Q13", "أود ترشيح كلمة غير مدرجة حاليًا", ("C08",), "paraphrase", "SUGGESTION", "WORD_SUBMISSION"),
    # -- meaning inquiry (paraphrases of C09) --
    Query("Q14", "وش معنى هالكلمة؟", ("C09",), "gulf_colloquial", "INQUIRY", "GENERAL"),
    Query("Q15", "ايش معناها بالضبط", ("C09",), "gulf_colloquial", "INQUIRY", "GENERAL"),
    Query("Q16", "ما تفسير هذه الكلمة", ("C09",), "no_punctuation", "INQUIRY", "GENERAL"),
    # -- how to submit a word (paraphrases of C10) --
    Query("Q17", "كيف اضيف كلمة جديدة للمعجم", ("C10",), "no_punctuation", "INQUIRY", "WORD_SUBMISSION"),
    Query("Q18", "وش الطريقة عشان اقترح كلمة", ("C10",), "gulf_colloquial", "INQUIRY", "WORD_SUBMISSION"),
    # -- grammatical gender inquiry (paraphrases of C11) --
    Query("Q19", "هذي الكلمة مؤنث ولا مذكر", ("C11",), "gulf_colloquial", "INQUIRY", "GENERAL"),
    # -- spam (paraphrase of C12) --
    Query("Q20", "اضغط هنا واربح جوائز مجانية الان", ("C12",), "no_punctuation", "SPAM", "OFF_TOPIC_PERSONAL"),
    # -- UX/design suggestion (paraphrases of C14) --
    Query("Q21", "الموقع صعب الاستخدام ياليت يصير اسهل", ("C14",), "gulf_colloquial", "SUGGESTION", "GENERAL"),
    # -- search not working (paraphrases of C15) --
    Query("Q22", "خانة البحث ما ترجع لي أي نتيجة", ("C15",), "gulf_colloquial", "COMPLAINT", "TECHNICAL_ISSUE"),
    Query("Q23", "البحث لا يعمل مهما كتبت", ("C15",), "spelling_variation", "COMPLAINT", "TECHNICAL_ISSUE"),
    # -- how to report a bug (paraphrases of C17) --
    Query("Q24", "كيف ابلغكم عن مشكلة تقنية بالموقع", ("C17",), "gulf_colloquial", "INQUIRY", "TECHNICAL_ISSUE"),
    # -- pronunciation recording wrong (paraphrases of C19) --
    Query("Q25", "طريقة نطق الكلمة بالتسجيل خطأ", ("C19",), "no_punctuation", "COMPLAINT", "ENTRY_CORRECTION"),
    # -- slow site (paraphrases of C22, deliberately similar to C03/C15 as a hard case) --
    Query("Q26", "الموقع بطيء مره ويطول علي التحميل", ("C22",), "gulf_colloquial", "COMPLAINT", "TECHNICAL_ISSUE"),
    # -- word etymology inquiry (paraphrases of C25) --
    Query("Q27", "اصل هذي الكلمة من وين", ("C25",), "gulf_colloquial", "INQUIRY", "GENERAL"),
    # -- multi-relevant case: a combined meaning+gender question genuinely
    #    matches BOTH C09 (meaning) and C11 (grammatical gender) -- included
    #    specifically to exercise multi-relevant-item metric handling, not
    #    an artificial label.
    Query("Q28", "عندي استفسار عن معنى هذي الكلمة ونوعها مؤنث ولا مذكر", ("C09", "C11"), "gulf_colloquial", "INQUIRY", "GENERAL"),
)
