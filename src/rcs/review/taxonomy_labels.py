"""Canonical Arabic labels for the configured 15-class taxonomy.

Do not add legacy aliases here. Unknown stored codes stay unmapped.
"""

from __future__ import annotations

from config import settings

INTENT_AR = {
    "THANKS_PRAISE": "شكر وثناء",
    "COMPLAINT": "شكوى",
    "SUGGESTION": "اقتراح",
    "INQUIRY": "استفسار",
    "SPAM": "غير ذي صلة",
}

TOPIC_AR = {
    "WORD_SUBMISSION": "اقتراح مفردة",
    "ENTRY_CORRECTION": "تصحيح مدخل",
    "DICTIONARY_SUGGESTION": "اقتراح معجمي",
    "TECHNICAL_ISSUE": "مشكلة تقنية",
    "WORD_OF_YEAR_NOMINATION": "ترشيح كلمة العام",
    "OFF_TOPIC_PERSONAL": "خارج الموضوع",
    "GENERAL": "عام",
}

SENTIMENT_AR = {
    "POSITIVE": "إيجابي",
    "NEGATIVE": "سلبي",
    "NEUTRAL": "محايد",
}


def assert_canonical_coverage() -> None:
    assert tuple(INTENT_AR) == settings.PRIMARY_INTENTS
    assert tuple(TOPIC_AR) == settings.TOPIC_SUBTYPES
    assert tuple(SENTIMENT_AR) == settings.SENTIMENTS
