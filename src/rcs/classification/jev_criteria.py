"""Jev question definitions for Intent, Topic and Sentiment.

Wording follows LABELING_GUIDE.md (the human annotation rules the Gold labels
were produced under). The label sets are the FROZEN taxonomy in
config.settings; this module only describes them to the provider and a drift
guard fails import if the two ever diverge. Pure data: no network, no I/O.
"""

from __future__ import annotations

from config import settings

INTENT_INSTRUCTIONS = (
    "اختر النية التواصلية الوحيدة لتعليق الزائر (ماذا يريد من رسالته) بصرف النظر عن موضوعه، "
    "ولا تستنتج النية من الموضوع."
)
TOPIC_INSTRUCTIONS = (
    "اختر الموضوع الأساسي الوحيد لتعليق الزائر (عمّ يدور) بصرف النظر عن نيته ونبرته، "
    "ولا تستنتج الموضوع من النية."
)
SENTIMENT_INSTRUCTIONS = (
    "اختر نبرة التعليق فقط، مستقلة عن نيته وموضوعه: الشكوى الهادئة قد تكون محايدة، "
    "والاقتراح ليس إيجابيًا تلقائيًا."
)

INTENT_CRITERIA = {
    "THANKS_PRAISE": "شكر أو ثناء أو دعاء وبركة موجّه للمنصة أو الفريق دون طلب أو شكوى جوهرية "
    "(مثل «شكرًا لكم»، «جزاكم الله خيرًا»، «بارك الله فيكم»).",
    "COMPLAINT": "إحباط أو نقد أو بلاغ سلبي تحمل نبرته عدم رضا أو لوم (مثل «لا يعمل»، «الخدمة سيئة»). "
    "إذا كان العرض هادئًا بصيغة اقتراح فهو اقتراح؛ وعند التردد بين الشكوى والاقتراح يُفضَّل الاقتراح.",
    "SUGGESTION": "اقتراح أو فكرة، أو تقديم هادئ لكلمة أو تصحيح أو ميزة دون نبرة استياء (مثل «أتمنى»، «أقترح»).",
    "INQUIRY": "سؤال حقيقي يطلب معلومة لا إجراءً (مثل «كيف»، «هل يمكن»، «أين أجد»).",
    "SPAM": "ليس تقديمًا معجميًا حقيقيًا: رموز عشوائية أو إعلان أو إساءة أو محتوى خارج المنصة، "
    "أو قوائم عامية وحكايات شخصية بلا غرض معجمي حتى لو كانت الكلمات عربية سليمة. القِصَر وحده ليس سببًا.",
}

TOPIC_CRITERIA = {
    "WORD_SUBMISSION": "اقتراح إضافة كلمة أو مدخل جديد غير موجود في المعجم (تسمية كلمة محددة لإضافتها). "
    "إن كانت الكلمة موجودة وفيها خطأ فهو تصحيح مدخل لا إضافة.",
    "ENTRY_CORRECTION": "الإبلاغ عن خطأ أو نقص في مدخل موجود أصلًا (معنى أو إملاء أو قواعد أو مثال) "
    "مع الإشارة إلى كلمة أو تعريف محدد.",
    "DICTIONARY_SUGGESTION": "فكرة لميزة أو عملية تخص منصة المعجم نفسها (مثل خاصية بحث أو تصنيف أو تحسين خدمة)، "
    "وليست كلمة بعينها وليست تصحيح مدخل.",
    "TECHNICAL_ISSUE": "عطل أو مشكلة تقنية في الموقع أو التطبيق (لا يفتح، لا يعمل، روابط أو ميزات معطلة).",
    "WORD_OF_YEAR_NOMINATION": "ترشيح صريح لكلمة ضمن حملة «كلمة العام» (يذكر الحملة أو يستخدم لغة ترشيح واضحة). "
    "ذكر «كلمة العام» عرضًا أو اقتراح كلمة عامة لا يكفي.",
    "OFF_TOPIC_PERSONAL": "محتوى لا علاقة له بالمعجم: قصص شخصية أو عامية أو أمثال أو قوائم كلمات بلا طلب معجمي، "
    "أو طلبات غير مرتبطة. إذا وُجد مضمون شخصي أو خارج المنصة فهذا أولى من «عام».",
    "GENERAL": "شكر أو دعاء أو ملاحظة موجزة دون موضوع معجمي محدد ودون محتوى شخصي "
    "(مثل «شكرًا لكم» أو «جزاكم الله خيرًا» وحدها).",
}

SENTIMENT_CRITERIA = {
    "POSITIVE": "نبرة ودودة أو ممتنة أو متحمسة (ثناء، امتنان بنبرة إيجابية).",
    "NEUTRAL": "نبرة واقعية أو هادئة أو محايدة عاطفيًا (معظم الطلبات والتصحيحات الهادئة والأسئلة).",
    "NEGATIVE": "نبرة منزعجة أو غاضبة أو غير راضية بوضوح (نقد حاد، إحباط، عدوانية).",
}

QUESTIONS = {
    "intent": (INTENT_INSTRUCTIONS, INTENT_CRITERIA, settings.PRIMARY_INTENTS),
    "topic": (TOPIC_INSTRUCTIONS, TOPIC_CRITERIA, settings.TOPIC_SUBTYPES),
    "sentiment": (SENTIMENT_INSTRUCTIONS, SENTIMENT_CRITERIA, settings.SENTIMENTS),
}

for _name, (_instr, _criteria, _labels) in QUESTIONS.items():
    if set(_criteria) != set(_labels):
        raise RuntimeError(f"Jev {_name} criteria drifted from the frozen taxonomy")
