"""Deterministic keyword classifier used ONLY in demo mode, in place of the frozen models (which are not deployed with a demo). It returns the same shapes as
rcs.classification.runtime and is clearly labelled in the stored provenance ("DEMO_RULES"), so a demo row can never be mistaken for a frozen-model prediction.
It is intentionally simple: it exists to drive the complete workflow on synthetic comments, not to classify well."""

from __future__ import annotations

import re

_SPAM = ("اشتروا", "عروض حصرية", "أسعار مخفضة", "خصم", "اربح", "وظيفة", "استضافة", "للتواصل واتساب")
_THANKS = ("شكر", "جزاكم", "بارك", "رائع", "ممتاز", "مفيد")
_COMPLAINT = ("لا يعمل", "لا تفتح", "لا يفتح", "خطأ", "بطيء", "غير دقيق", "مشكلة", "ناقص", "يتوقف", "تظهر رسالة")
_SUGGEST = ("أقترح", "اقترح", "أتمنى", "أرجو", "يرجى", "ترشيح", "أرشح", "إضافة", "هل يمكن إضافة")
_INQUIRY_START = ("ما ", "هل ", "كيف ", "لماذا ", "متى ", "أين ", "من ")
_TECH = ("لا تفتح", "لا يفتح", "لا يعمل", "بطيء", "رسالة خطأ", "الصفحة", "الموقع", "رابط", "الصوت", "يتوقف")
_CORRECTION = ("غير دقيق", "خطأ إملائي", "ناقص", "المعنى المذكور", "تصحيح")
_WORD_OF_YEAR = ("كلمة العام", "أرشح كلمة")
_WORD_SUBMISSION = ("إضافة كلمة", "اقترح كلمة", "أقترح كلمة", "إضافة الكلمة")
_DICT_SUGGEST = ("خاصية", "البحث", "أمثلة", "قائمة", "الجذر")
_POSITIVE = _THANKS + ("أرشح",)


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


def intent_of(text: str) -> tuple[str, float]:
    t = " ".join((text or "").split())
    if _has(t, _SPAM):
        return "SPAM", 0.9
    if _has(t, _THANKS) and not _has(t, _SUGGEST):
        return "THANKS_PRAISE", 0.85
    if _has(t, _COMPLAINT):
        return "COMPLAINT", 0.8
    if _has(t, _SUGGEST):
        return "SUGGESTION", 0.8
    if "؟" in t or t.startswith(_INQUIRY_START):
        return "INQUIRY", 0.8
    return "INQUIRY", 0.55


def topic_of(text: str, intent: str) -> tuple[str, float]:
    t = " ".join((text or "").split())
    if intent == "SPAM":
        return "OFF_TOPIC_PERSONAL", 0.85
    if _has(t, _WORD_OF_YEAR):
        return "WORD_OF_YEAR_NOMINATION", 0.85
    if _has(t, _CORRECTION):
        return "ENTRY_CORRECTION", 0.8
    if _has(t, _TECH):
        return "TECHNICAL_ISSUE", 0.8
    if _has(t, _WORD_SUBMISSION):
        return "WORD_SUBMISSION", 0.8
    if _has(t, _DICT_SUGGEST):
        return "DICTIONARY_SUGGESTION", 0.75
    return "GENERAL", 0.6


def sentiment_of(text: str, intent: str) -> tuple[str, float]:
    if intent == "COMPLAINT":
        return "NEGATIVE", 0.8
    if intent == "THANKS_PRAISE" or _has(text or "", _POSITIVE):
        return "POSITIVE", 0.8
    return "NEUTRAL", 0.7


def predict_intent(texts):
    return [intent_of(t) for t in texts]


def predict_topic(comment_ids, texts):
    return {cid: topic_of(t, intent_of(t)[0]) for cid, t in zip(comment_ids, texts)}


def predict_sentiment(comment_ids, texts):
    return {cid: sentiment_of(t, intent_of(t)[0]) for cid, t in zip(comment_ids, texts)}


def provenance() -> dict[str, str]:
    return {"reason": "demo_rule_classifier", "source": "DEMO_RULES_SYNTHETIC_ONLY", "model_name": "demo-keyword-rules-v1", "prompt_version": "demo-rules-v1"}


def apply_optional_topic_layer(topic_preds, ids, texts, lemmas=None):
    return topic_preds, False


def with_layer_provenance(prov, layer_applied):
    return prov
