"""
Deterministic response-eligibility rules.

Separate from `rcs.routing.engine` (classification operational routing:
AUTO_REPLY_CANDIDATE / ACKNOWLEDGE_THEN_REVIEW / REVIEW_ONLY / EXCLUDE).

Reply-safety / classification-safety policy (this revision):
  * A model label alone (SPAM intent, OFF_TOPIC_PERSONAL topic, or the operational EXCLUDE derived from them) is NOT enough to recommend
    "no response". Evidence on the blind set: single-label exclusions were wrong for 11 of 31 records; records where BOTH the intent model
    (SPAM) and the topic model (OFF_TOPIC_PERSONAL) agreed were wrong for 0 of 10. An uncorroborated exclusion is routed to a human
    (REVIEW_ONLY, no draft, original predictions preserved); only a corroborated one (or a caller-supplied deterministic corroboration such
    as the prefilter) stays NO_RESPONSE_RECOMMENDED, which a reviewer can still override.
  * A comment that asks for facts (an INQUIRY, or any question) is never answered by a generated draft while no approved OFFICIAL KNOWLEDGE is
    available: it goes to a human (REVIEW_ONLY). Detection uses the text itself, not only the (weak) Intent label.

GENERATE_DRAFT — meaningful comments that deserve a reply or reviewer
assistance (thanks, suggestion, complaint, inquiry, technical issue, and
other relevant dictionary/service comments). Uncertain classifiers do not
suppress these. Inquiry/technical drafts stay constrained; Gemini must not
invent facts.

NO_RESPONSE_RECOMMENDED — no Gemini call for clearly non-actionable
content (spam, off-topic personal, empty/invalid, operational EXCLUDE).

The reviewer still chooses SEND_AS_IS, EDIT_AND_SEND, or NO_RESPONSE_NEEDED
on any draft. Gemini never sends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import settings

GENERATE_DRAFT = "GENERATE_DRAFT"
REVIEW_ONLY = "REVIEW_ONLY"
NO_RESPONSE_RECOMMENDED = "NO_RESPONSE_RECOMMENDED"

# Auditable evaluation order. First matching rule wins.
ELIGIBILITY_RULE_ORDER = (
    "empty_or_meaningless",
    "spam_or_operational_exclude",
    "uncertain_exclusion_review",
    "insufficient_context",
    "word_of_year",
    "technical_issue",
    "factual_question_without_grounding",
    "inquiry_without_trusted_grounding",
    "generate_thanks",
    "generate_suggestion",
    "generate_complaint",
    "generate_inquiry_with_grounding",
    "default_generate",
)


@dataclass(frozen=True)
class EligibilityInput:
    comment_text: str
    primary_intent: Optional[str] = None
    topic_subtype: Optional[str] = None
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    needs_review: bool = False
    is_content_meaningful: bool = True
    insufficient_context: bool = False
    trusted_grounding_available: bool = False
    classification_routing_decision: Optional[str] = None
    # Deterministic, model-independent evidence that the comment really is junk (e.g. the prefilter). Never set from a model label.
    exclusion_corroborated: bool = False


@dataclass(frozen=True)
class EligibilityResult:
    routing_decision: str
    reason_code: str
    rule_id: str


def _blank(text: str | None) -> bool:
    return text is None or not str(text).strip()


_MIN_MEANINGFUL_LETTERS = 2


def _lacks_minimum_meaningful_content(text: str | None) -> bool:
    """True when the text has fewer than `_MIN_MEANINGFUL_LETTERS` letter
    characters -- e.g. a single stray letter such as "و" or "م" (both
    real, common one-letter Arabic words/prefixes on their own, but not
    a submission with any content to respond to).

    Deliberately a LETTER-COUNT threshold, not a word-count threshold:
    a genuine one-word lexical submission (e.g. "لا", "مسك", "كرسي")
    still has two or more letters and is never suppressed by this
    check -- only single-character fragments are.

    Diacritics (tashkeel) are Unicode combining marks (category Mn),
    not letters, so str.isalpha() already excludes them: a single
    letter plus diacritics (e.g. "وَ") still counts as exactly one
    letter and is correctly still caught.
    """
    if text is None:
        return True
    letters = sum(1 for ch in str(text) if ch.isalpha())
    return letters < _MIN_MEANINGFUL_LETTERS


_QUESTION_TOKENS = frozenset(
    {"هل", "كيف", "ماذا", "لماذا", "متى", "اين", "أين", "كم", "ايش", "وش", "ليش", "شنو", "ليه", "لما", "علاش"}
)
_QUESTION_PHRASES = ("ما معنى", "ما هو", "ما هي", "ما الفرق", "ما حكم", "ما الصواب", "ما المقصود", "ما اصل", "ما أصل", "ماهو", "ماهي", "ما تعريف")


def asks_for_facts(text: str | None) -> bool:
    """Deterministic: True when the comment poses a question (question mark, interrogative word or question phrase). Intentionally
    over-inclusive: the cost of a false positive is a human reply, the cost of a false negative is a fabricated one."""
    t = str(text or "")
    if "؟" in t or "?" in t:
        return True
    norm = " ".join(t.replace("،", " ").replace(".", " ").split())
    toks = set(norm.split())
    if toks & _QUESTION_TOKENS:
        return True
    return any(p in norm for p in _QUESTION_PHRASES)


def _model_exclusion_corroborated(inp: "EligibilityInput") -> bool:
    return bool(inp.exclusion_corroborated) or (inp.primary_intent == "SPAM" and inp.topic_subtype == "OFF_TOPIC_PERSONAL")


def determine_response_eligibility(inp: EligibilityInput) -> EligibilityResult:
    """Pure function. No network. No Gemini."""
    intent = inp.primary_intent
    topic = inp.topic_subtype
    class_route = inp.classification_routing_decision

    if _blank(inp.comment_text) or not inp.is_content_meaningful or _lacks_minimum_meaningful_content(inp.comment_text):
        return EligibilityResult(
            NO_RESPONSE_RECOMMENDED,
            "empty_or_meaningless",
            "empty_or_meaningless",
        )

    if (
        intent == "SPAM"
        or class_route == "EXCLUDE"
        or topic == "OFF_TOPIC_PERSONAL"
    ):
        if settings.RESPONSE_REQUIRE_CORROBORATION_FOR_EXCLUSION and not _model_exclusion_corroborated(inp):
            return EligibilityResult(
                REVIEW_ONLY,
                "uncertain_exclusion_requires_review",
                "uncertain_exclusion_review",
            )
        if intent == "SPAM":
            reason = "spam"
        elif topic == "OFF_TOPIC_PERSONAL":
            reason = "off_topic_personal"
        else:
            reason = f"operational_exclude_{class_route or 'unknown'}"
        return EligibilityResult(
            NO_RESPONSE_RECOMMENDED,
            reason,
            "spam_or_operational_exclude",
        )

    if inp.insufficient_context:
        return EligibilityResult(
            NO_RESPONSE_RECOMMENDED,
            "insufficient_context",
            "insufficient_context",
        )

    if topic == "WORD_OF_YEAR_NOMINATION":
        return EligibilityResult(
            GENERATE_DRAFT,
            "word_of_year_safe_acknowledgement",
            "word_of_year",
        )

    if topic == "TECHNICAL_ISSUE":
        return EligibilityResult(
            GENERATE_DRAFT,
            "technical_issue_safe_acknowledgement",
            "technical_issue",
        )

    if (
        settings.RESPONSE_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS
        and not inp.trusted_grounding_available
        and (intent == "INQUIRY" or asks_for_facts(inp.comment_text))
    ):
        # Fail closed: no approved OFFICIAL KNOWLEDGE, so any definition / policy / status answer would be invented. A human answers.
        return EligibilityResult(
            REVIEW_ONLY,
            "factual_question_without_grounding",
            "factual_question_without_grounding",
        )

    if intent == "INQUIRY" and not inp.trusted_grounding_available:
        # Legacy path: reachable ONLY when RESPONSE_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS is switched off.
        return EligibilityResult(
            GENERATE_DRAFT,
            "inquiry_without_trusted_grounding",
            "inquiry_without_trusted_grounding",
        )

    if intent in ("THANKS_PRAISE", "DUA_BLESSING"):
        return EligibilityResult(GENERATE_DRAFT, "thanks_praise_eligible", "generate_thanks")

    if intent == "SUGGESTION":
        return EligibilityResult(GENERATE_DRAFT, "suggestion_eligible", "generate_suggestion")

    if intent == "COMPLAINT":
        return EligibilityResult(GENERATE_DRAFT, "complaint_safe_acknowledgement", "generate_complaint")

    if intent == "INQUIRY" and inp.trusted_grounding_available:
        return EligibilityResult(
            GENERATE_DRAFT,
            "inquiry_with_trusted_grounding",
            "generate_inquiry_with_grounding",
        )

    # Uncertain classifiers (low confidence, needs_review, classification
    # REVIEW_ONLY) do not suppress a draft for remaining relevant comments.
    return EligibilityResult(GENERATE_DRAFT, "valid_comment_draft", "default_generate")
