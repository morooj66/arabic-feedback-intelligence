"""Reviewer-facing text for the blind evaluation workbook. Pure data; the codes come from the frozen taxonomy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from rcs.review.taxonomy_labels import INTENT_AR, SENTIMENT_AR, TOPIC_AR, assert_canonical_coverage  # noqa: E402

assert_canonical_coverage()
INTENTS = list(settings.PRIMARY_INTENTS)
TOPICS = list(settings.TOPIC_SUBTYPES)
SENTIMENTS = list(settings.SENTIMENTS)

README_LINES = [
    "Blind evaluation of visitor-comment classification - reviewer instructions",
    "",
    "WHAT THIS IS. 75 visitor comments to label independently with three fields: Intent, Topic, Sentiment. Your labels will be used as the "
    "reference answers to measure how accurate the classification system is. Nobody has pre-labelled these records.",
    "",
    "HOW TO WORK.",
    "1. Open the 'Review' sheet. Read each comment_text and fill the three columns by choosing from the drop-down lists (exact codes only).",
    "2. Judge Intent, Topic and Sentiment INDEPENDENTLY. Do not infer one from another.",
    "3. If you genuinely cannot judge a comment (a single word, a fragment, unreadable), choose INSUFFICIENT_CONTEXT in the "
    "'insufficient_context' column and leave Intent, Topic and Sentiment EMPTY for that row. Do not guess.",
    "4. 'reviewer_notes' is optional - use it for anything ambiguous, for a second plausible label, or to flag a problem.",
    "5. Use the 'Guidelines' sheet for the rules and the 'Codes' sheet for the Arabic names of the codes.",
    "",
    "PLEASE DO NOT.",
    "- Do not use AI tools, translation or classification tools, search engines, or the source system to help label.",
    "- Do not discuss records with other reviewers until both of you have finished (agreement is measured afterwards).",
    "- Do not copy comments out of this file, and do not try to identify any person.",
    "",
    "PRIVACY. Names that identify a visitor or a private person, phone numbers, e-mails and ID numbers were replaced by [NAME], [ID], [EMAIL], [URL]. "
    "Treat placeholders as ordinary text. If you still see personal data that should have been masked, write PRIVACY in reviewer_notes for that row and "
    "tell the coordinator; do not repeat it anywhere.",
    "",
    "RETURN THE WORKBOOK AS IT IS (do not rename sheets or columns, do not reorder rows). Save a copy before you start.",
    "",
    "Rules version: clarified classification policy v1 (see 'Guidelines').",
]

GUIDELINE_ROWS = [
    ("h", "FIELD / CODE", "RULE"),
    ("h", "INTENT - what the visitor is doing", ""),
    ("r", "INQUIRY (استفسار)", "A genuine request for information (typically a question: كيف، هل، أين، ما معنى …). A bare word followed by its meaning, with no question, is NOT an inquiry."),
    ("r", "SUGGESTION (اقتراح)", "A proposed addition or improvement: a word (including a word with its meaning or example), a correction, a feature or a change. Offered calmly."),
    ("r", "COMPLAINT (شكوى)", "EXPLICIT dissatisfaction or blame in the wording itself (e.g. «الخدمة سيئة», «لا يعمل ولا أحد يهتم»). A calm statement that an entry is wrong is not a complaint. If it is truly 50/50 between COMPLAINT and SUGGESTION choose SUGGESTION."),
    ("r", "THANKS_PRAISE (شكر وثناء)", "Gratitude or praise, including a blessing used as thanks («جزاكم الله خيرا»), with no substantive request or complaint."),
    ("r", "SPAM (غير ذي صلة)", "Not a genuine dictionary communication: test messages, gibberish, adverts, abuse, unrelated chatter or personal stories with no dictionary purpose. Short is not the same as spam."),
    ("h", "TOPIC - what the message is about", ""),
    ("r", "WORD_SUBMISSION (اقتراح مفردة)", "A word proposed for addition, INCLUDING a word with its meaning, example or origin written out as content, even if it looks like a finished dictionary entry."),
    ("r", "ENTRY_CORRECTION (تصحيح مدخل)", "An EXPLICIT correction or error report about an EXISTING entry: it says something is wrong, missing, duplicated or inconsistent (meaning, spelling, grammar, example)."),
    ("r", "TECHNICAL_ISSUE (مشكلة تقنية)", "An access, display or functionality problem with the website or app (page does not open, text does not appear, login or search fails). A mistake in the CONTENT of an entry is ENTRY_CORRECTION instead."),
    ("r", "DICTIONARY_SUGGESTION (اقتراح معجمي)", "A suggestion about dictionary content or functionality that is not a direct word submission or correction: a new feature or kind of content, a proposal to contribute or link a dictionary or resource, a request to improve the service."),
    ("r", "WORD_OF_YEAR_NOMINATION (ترشيح كلمة العام)", "An explicit nomination for the word-of-the-year campaign. Mentioning the phrase in passing is not enough."),
    ("r", "OFF_TOPIC_PERSONAL (خارج الموضوع)", "Not about the dictionary at all: personal stories, unrelated requests, colloquial lists with no lexicographic ask."),
    ("r", "GENERAL (عام)", "Plain thanks, a blessing or a brief remark with no specific topic and no personal or off-platform content."),
    ("h", "SENTIMENT - emotional tone only", ""),
    ("r", "POSITIVE (إيجابي)", "Warmth, praise, enthusiasm in the wording."),
    ("r", "NEUTRAL (محايد)", "Factual, calm or emotionally flat. Most submissions, questions and calm corrections."),
    ("r", "NEGATIVE (سلبي)", "Irritation, anger, hostility or clear dissatisfaction in the wording."),
    ("r", "Sentiment rules", "Judge the TONE of the words, not the intent or topic. Courtesy alone (السلام عليكم، تحية طيبة، شكرا as an opening) is NOT positive. Test messages and gibberish are NOT automatically negative. A calm correction is neutral; blunt or blaming wording is negative. Praise inside the word being defined (e.g. 'beautiful') is not the visitor's sentiment."),
    ("h", "GENERAL RULES", ""),
    ("r", "Independence", "Intent, topic and sentiment are three separate judgments. A COMPLAINT can be NEUTRAL in tone; a SUGGESTION is not automatically POSITIVE; a TECHNICAL_ISSUE can be any intent."),
    ("r", "INSUFFICIENT_CONTEXT", "A single word or fragment with no way to tell what the visitor wants, or unreadable text. Leave the three label columns empty."),
    ("r", "One label per field", "Choose the single dominant label. Use reviewer_notes for a second plausible label."),
    ("h", "SYNTHETIC EXAMPLES (not from the data)", ""),
    ("r", "«ما معنى كلمة «قارورة»؟»", "Intent INQUIRY; Topic GENERAL (no dictionary topic beyond the question) - Sentiment NEUTRAL"),
    ("r", "«أقترح إضافة كلمة «تلفزة» ومعناها جهاز عرض»", "Intent SUGGESTION; Topic WORD_SUBMISSION; Sentiment NEUTRAL"),
    ("r", "«تعريف كلمة «مرآب» في المعجم غير صحيح، الصواب …»", "Intent SUGGESTION (calm); Topic ENTRY_CORRECTION; Sentiment NEUTRAL"),
    ("r", "«الموقع لا يفتح عندي منذ أمس وهذا غير مقبول»", "Intent COMPLAINT; Topic TECHNICAL_ISSUE; Sentiment NEGATIVE"),
    ("r", "«السلام عليكم، أتمنى إضافة خاصية البحث بالجذر»", "Intent SUGGESTION; Topic DICTIONARY_SUGGESTION; Sentiment NEUTRAL (courtesy opening is not positive)"),
]

CODE_TABLE = [("CODE", "Arabic", "Meaning")]
for k in INTENTS:
    CODE_TABLE.append((k, INTENT_AR[k], "Intent"))
for k in TOPICS:
    CODE_TABLE.append((k, TOPIC_AR[k], "Topic"))
for k in SENTIMENTS:
    CODE_TABLE.append((k, SENTIMENT_AR[k], "Sentiment"))
CODE_TABLE.append(("INSUFFICIENT_CONTEXT", "سياق غير كاف", "Use instead of labels when no label can be judged"))
