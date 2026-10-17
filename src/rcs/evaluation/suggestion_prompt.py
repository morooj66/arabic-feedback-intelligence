"""
Phase B.2 — builds the prompt for one ai_assisted_train row's AI-suggested
label. Distinct from src/rcs/classification/prompt.py (Phase 4's live
classification prompt): this one (1) includes a fixed few-shot block, (2)
asks for `sentiment` as a third, independent signal, (3) never asks for
is_content_meaningful or a reply draft — a suggestion row already passed
Phase 3's junk filter and Phase B's gold-sampling eligibility filter, and
this call NEVER produces anything sent to a visitor or auto-published.

Only comment text is included (no original_type/age_group/platform
metadata) — the gold dataset CSV that ai_assisted_train rows are read from
does not carry that metadata forward (data/gold/gold_dataset_round1.csv
only has comment_id/comment_text/gold_*/split/annotation_stage/...), and
re-querying review.sqlite per suggestion row is unnecessary complexity for
what this phase needs.
"""

from __future__ import annotations

from config import settings
from ..classification.redact import redact_phone_numbers

_INTENT_LINES = "\n".join(f"  - {c}" for c in settings.PRIMARY_INTENTS)
_TOPIC_LINES = "\n".join(f"  - {c}" for c in settings.TOPIC_SUBTYPES)
_SUBCATEGORY_LINES = "\n".join(f"  - {s}" for s in settings.SUBCATEGORIES)
_SENTIMENT_LINES = "\n".join(f"  - {s}" for s in settings.SENTIMENTS)

_DISAMBIGUATION = """
LABELING DISAMBIGUATION (from the project's gold-labeling guide — apply from the comment \
TEXT only; do not infer intent from topic or topic from intent):

WORD_SUBMISSION vs ENTRY_CORRECTION:
  - WORD_SUBMISSION: proposing a NEW word/entry not yet in the dictionary (naming a word to add).
  - ENTRY_CORRECTION: reporting something WRONG with an EXISTING entry (wrong meaning, spelling, \
grammar, incomplete definition). A calmly-worded correction is still ENTRY_CORRECTION for topic; \
its intent may be SUGGESTION or COMPLAINT depending on tone (see below).

COMPLAINT vs SUGGESTION (intent):
  - COMPLAINT: frustration, criticism, or dissatisfaction in the wording ("الخدمة سيئة", "خطأ", \
"لا يعمل", irritated tone even if brief).
  - SUGGESTION: calm proposal or neutral-to-positive submission with no dissatisfaction ("أتمنى", \
"أقترح", politely proposing a word/correction/feature). When truly 50/50 on tone, prefer SUGGESTION.
  - Do NOT label every word-submission channel comment as SUGGESTION — read whether the visitor is \
complaining about something wrong vs calmly proposing something new.

INQUIRY (intent):
  - A genuine question seeking information, not action: look for "كيف", "هل", "أين", "ما معنى", \
"هل كلمة ... فصيحة", requests for synonym/meaning/explanation. Topic may be GENERAL or \
WORD_SUBMISSION depending on subject matter.

SPAM (intent):
  - Rare here (most junk was pre-filtered). Use only for meaningless, promotional, abusive, or \
clearly off-platform content that is still not a genuine dictionary submission — gibberish, ads, \
test noise, unrelated personal requests with no dictionary purpose. Do not use SPAM merely because \
a comment is short.

DUA_BLESSING vs THANKS_PRAISE:
  - DUA_BLESSING: religious supplication/blessing ("جزاكم الله خيرًا", "بارك الله فيكم", "الله \
يعطيكم العافية") — even if it also reads as thanks.
  - THANKS_PRAISE: secular thanks/praise ("شكرًا لكم", "الموقع رائع") with no religious-invocation \
language.

Pure thanks/blessings with no dictionary topic → topic_subtype GENERAL.

OFF_TOPIC_PERSONAL vs GENERAL (topic):
  - GENERAL: pure thanks/blessings/remarks with no personal or off-platform subject \
("شكرًا لكم" alone).
  - OFF_TOPIC_PERSONAL: identifiable content not about the dictionary — personal stories, \
slang lists, proverbs, unrelated requests, colloquial words with no lexicographic ask.

INSUFFICIENT_CONTEXT (human reviewers only):
  - Not a Gemini output. Humans mark this when text is too short, fragmented, or ambiguous \
to classify reliably (bare words, decontextualized phrases). Leave all gold label fields \
blank. Do not use SPAM as a fallback for illegible text — SPAM requires readable content \
that is clearly not a genuine dictionary submission.
"""

_INSTRUCTIONS = f"""You are suggesting a gold-standard label for one HISTORICAL visitor comment \
to an Arabic dictionary platform, a crowdsourced Arabic lexicon platform. This is NOT a live \
comment and your output will NEVER be sent to anyone or published automatically — it is a \
DRAFT SUGGESTION that a human reviewer will confirm, edit, or reject before it becomes part of \
a training/evaluation dataset.

Classify along THREE INDEPENDENT DIMENSIONS. Do not infer one from another — read the actual \
text and judge each on its own:

1. primary_intent — the visitor's COMMUNICATIVE INTENT. Choose exactly one from:
{_INTENT_LINES}

2. topic_subtype — what the message is ABOUT. Choose exactly one from:
{_TOPIC_LINES}

3. sentiment — the comment's overall TONE, independent of intent and topic. Choose exactly one \
from:
{_SENTIMENT_LINES}
A COMPLAINT is not automatically NEGATIVE sentiment (a calmly-worded complaint can be NEUTRAL \
in tone), and a SUGGESTION is not automatically POSITIVE — judge tone from the actual wording.
{_DISAMBIGUATION}
If topic_subtype is ENTRY_CORRECTION or TECHNICAL_ISSUE, also choose one subcategory from:
{_SUBCATEGORY_LINES}
Otherwise leave subcategory null.

If — and only if — the comment genuinely contains a second, distinct INTENT, set \
secondary_intent to one of the intents above. It must not equal primary_intent. Leave it null \
for ordinary single-intent comments.

Below are labeled examples for calibration only — they are NOT the comment you are classifying \
and must not be echoed back.
"""


def _format_example(ex: dict) -> str:
    parts = [f'  comment: {ex["comment_text"]!r}']
    parts.append(f'  -> primary_intent: {ex["gold_primary_intent"]}')
    parts.append(f'  -> topic_subtype: {ex["gold_topic_subtype"]}')
    if ex.get("gold_secondary_intent"):
        parts.append(f'  -> secondary_intent: {ex["gold_secondary_intent"]}')
    if ex.get("gold_subcategory"):
        parts.append(f'  -> subcategory: {ex["gold_subcategory"]}')
    parts.append(f'  -> sentiment: {ex["gold_sentiment"]}')
    return "\n".join(parts)


def build_suggestion_prompt(row: dict, few_shot_examples: list[dict]) -> str:
    """row must contain: comment_text."""
    redacted = redact_phone_numbers(row.get("comment_text"))

    examples_block = "\n\n".join(
        f"--- EXAMPLE {i} ---\n{_format_example(ex)}"
        for i, ex in enumerate(few_shot_examples, start=1)
    )

    target_block = f"""
--- COMMENT TO CLASSIFY ---
comment: {redacted!r}
--- END COMMENT ---

Respond with a single JSON object matching the required schema. No prose outside the JSON.
"""
    return _INSTRUCTIONS + "\n" + examples_block + "\n" + target_block
