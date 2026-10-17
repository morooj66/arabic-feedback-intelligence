"""
Builds the single prompt sent to Gemini for one comment.

What goes in: redacted comment text, its title (may carry an existing
MeaningFault/TechFault/typo/GrammerFault/Other hint), its raw `النوع`
value (explicitly labeled as unverified metadata), age_group, platform,
and whether it's linked to a specific dictionary entry.

What never goes in: visitor_name, visitor_email — not needed for
classification and excluded per DESIGN_SPEC.md Section 7.

Post-migration (MIGRATION_PLAN.md): this prompt asks for classification
ONLY — no reply drafting instructions. It also explains PRIMARY_INTENT and
TOPIC_SUBTYPE as two independent dimensions and explicitly warns against
inferring one from the other, per the product requirement that Gemini must
not assume every ENTRY_CORRECTION is a COMPLAINT or every WORD_SUBMISSION
is a SUGGESTION just from metadata.
"""

from __future__ import annotations

from config import settings
from .redact import redact_phone_numbers

_INTENT_LINES = "\n".join(f"  - {c}" for c in settings.PRIMARY_INTENTS)
_TOPIC_LINES = "\n".join(f"  - {c}" for c in settings.TOPIC_SUBTYPES)
_SUBCATEGORY_LINES = "\n".join(f"  - {s}" for s in settings.SUBCATEGORIES)

_INSTRUCTIONS = f"""You are classifying a single visitor submission to an Arabic dictionary platform, \
a crowdsourced Arabic lexicon platform. You are classifying the VISITOR'S COMMENT ONLY — \
you are not writing a response to be sent to anyone, and nothing you output will be sent \
to the visitor automatically. A human reviewer makes all final decisions.

You are classifying along TWO INDEPENDENT DIMENSIONS. Do not infer one from the other — \
read the actual text and judge each dimension on its own:

1. primary_intent — the visitor's COMMUNICATIVE INTENT (what kind of message this is, \
regardless of subject matter). Choose exactly one from:
{_INTENT_LINES}

2. topic_subtype — what the message is ABOUT (regardless of tone/intent). Choose exactly \
one from:
{_TOPIC_LINES}

These are independent. Do NOT automatically assume every ENTRY_CORRECTION topic implies a \
COMPLAINT intent, and do NOT automatically assume every WORD_SUBMISSION topic implies a \
SUGGESTION intent — a visitor can report an incorrect entry as a neutral SUGGESTION or as an \
angry COMPLAINT; judge the intent from the actual tone and wording of the comment, and the \
topic from what it's actually about. Examples: proposing a new word to add is \
SUGGESTION + WORD_SUBMISSION. Reporting a broken website feature is COMPLAINT + \
TECHNICAL_ISSUE. Asking how to submit a word is INQUIRY + WORD_SUBMISSION. "جزاكم الله خيرًا" \
is DUA_BLESSING + GENERAL. "شكرًا لكم، عمل رائع" is THANKS_PRAISE + GENERAL. A message with no \
specific dictionary topic (thanks, a blessing, a general remark) gets topic_subtype GENERAL.

DUA_BLESSING vs THANKS_PRAISE: DUA_BLESSING is a religious supplication/blessing phrase \
directed at the reader (e.g. "جزاكم الله خيرًا", "بارك الله فيكم", "الله يعطيكم العافية") — the \
visitor is invoking God's favor on the team, not directly praising the work. THANKS_PRAISE is \
explicit secular thanks or praise of the site/dictionary itself (e.g. "شكرًا لكم", "الموقع \
رائع", "عمل ممتاز") with no religious-invocation language. A comment can plausibly be read as \
either only when both a blessing phrase AND explicit praise are present — in that case use \
secondary_intent to record the second one.

If topic_subtype is ENTRY_CORRECTION or TECHNICAL_ISSUE, also choose one subcategory from:
{_SUBCATEGORY_LINES}
Otherwise leave subcategory null.

If — and only if — the comment genuinely contains a second, distinct INTENT (for example, \
thanks combined with a complaint), set secondary_intent to one of the intents above. It must \
not equal primary_intent. Leave it null for ordinary single-intent comments — do not force a \
secondary intent that isn't really there. There is no secondary topic field — classify only \
the single dominant topic.

Set is_content_meaningful to false if the text is empty, gibberish, or an unfilled form \
placeholder, even though most of that is normally caught before it reaches you.

Do not draft or suggest any reply text. You are classifying only — a separate step handles \
acknowledgement and response drafting.

The `original_type` and `original_title` fields below come from the platform's own submission \
form. Treat them as unverified contextual metadata, not ground truth about what the comment \
actually is — classify based on the comment text itself.
"""


def build_prompt(row: dict) -> str:
    """
    row must contain: original_comment, original_title, original_type,
    age_group, platform, entry_id, lemma (entry_id/lemma may be None).
    """
    redacted_comment = redact_phone_numbers(row.get("original_comment"))
    has_linked_entry = bool(row.get("entry_id") or row.get("lemma"))

    context = f"""
--- SUBMISSION ---
comment: {redacted_comment!r}
original_title (unverified metadata): {row.get("original_title")!r}
original_type (unverified metadata): {row.get("original_type")!r}
age_group: {row.get("age_group")!r}
platform: {row.get("platform")!r}
linked_to_specific_dictionary_entry: {has_linked_entry}
--- END SUBMISSION ---

Respond with a single JSON object matching the required schema. No prose outside the JSON.
"""
    return _INSTRUCTIONS + context
