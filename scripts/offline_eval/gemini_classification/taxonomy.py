"""The approved, frozen taxonomy -- read from config.settings and pinned here so any drift is detected, never silently followed."""
from __future__ import annotations

from config import settings

EXPECTED_INTENTS = ("THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM")
EXPECTED_TOPICS = ("WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION", "TECHNICAL_ISSUE", "WORD_OF_YEAR_NOMINATION", "OFF_TOPIC_PERSONAL", "GENERAL")
EXPECTED_SENTIMENTS = ("POSITIVE", "NEGATIVE", "NEUTRAL")
CONFIDENCE_CATEGORIES = ("high", "medium", "low")

INTENTS = tuple(settings.PRIMARY_INTENTS)
TOPICS = tuple(settings.TOPIC_SUBTYPES)
SENTIMENTS = tuple(settings.SENTIMENTS)

TASKS = ("intent", "topic", "sentiment")
LABELS = {"intent": INTENTS, "topic": TOPICS, "sentiment": SENTIMENTS}


class TaxonomyError(ValueError):
    pass


def assert_taxonomy_unchanged() -> None:
    """Raises if config.settings no longer matches the taxonomy this evaluation was designed for (5 intents / 7 topics / 3 sentiments)."""
    if set(INTENTS) != set(EXPECTED_INTENTS) or set(TOPICS) != set(EXPECTED_TOPICS) or set(SENTIMENTS) != set(EXPECTED_SENTIMENTS):
        raise TaxonomyError("config.settings taxonomy differs from the approved 5/7/3 taxonomy; refusing to evaluate")
    if (len(INTENTS), len(TOPICS), len(SENTIMENTS)) != (5, 7, 3):
        raise TaxonomyError("unexpected number of classes")


def validate_labels(intent: str, topic: str, sentiment: str) -> None:
    for task, value in (("intent", intent), ("topic", topic), ("sentiment", sentiment)):
        if value not in LABELS[task]:
            raise TaxonomyError(f"{task}={value!r} is not in the approved taxonomy")


assert_taxonomy_unchanged()
