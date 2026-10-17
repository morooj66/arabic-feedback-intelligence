"""The predefined systems compared on real data (fixed in advance; no others are tried).

frozen                      all three tasks from the frozen classifiers
gemini                      all three tasks from Gemini
hybrid_topic_gemini         topic from Gemini; intent and sentiment frozen
hybrid_intent_topic_gemini  intent and topic from Gemini; sentiment frozen (the frozen sentiment model led the earlier exploratory comparison)
Per-task winners are chosen by selection.py; the hybrids exist so that JOINT effects (routing depends on intent and topic together) are measured.
"""
from __future__ import annotations

from typing import Mapping

from .evaluate import Prediction

SYSTEMS: dict[str, dict[str, str]] = {
    "frozen": {"intent": "frozen", "topic": "frozen", "sentiment": "frozen"},
    "gemini": {"intent": "gemini", "topic": "gemini", "sentiment": "gemini"},
    "hybrid_topic_gemini": {"intent": "frozen", "topic": "gemini", "sentiment": "frozen"},
    "hybrid_intent_topic_gemini": {"intent": "gemini", "topic": "gemini", "sentiment": "frozen"},
}


def compose(sources: Mapping[str, str], frozen: Mapping[str, Prediction], gemini: Mapping[str, Prediction]) -> dict[str, Prediction]:
    """Rows are kept only where every needed source has a prediction (so comparisons stay on identical rows)."""
    pools = {"frozen": frozen, "gemini": gemini}
    needed = {sources[t] for t in ("intent", "topic", "sentiment")}
    ids = set.intersection(*(set(pools[n]) for n in needed))
    out = {}
    for i in ids:
        amb = gemini[i].ambiguous if i in gemini else False
        out[i] = Prediction(getattr(pools[sources["intent"]][i], "intent"), getattr(pools[sources["topic"]][i], "topic"),
                            getattr(pools[sources["sentiment"]][i], "sentiment"), amb, "")
    return out


def all_systems(frozen: Mapping[str, Prediction], gemini: Mapping[str, Prediction]) -> dict[str, dict[str, Prediction]]:
    return {name: compose(src, frozen, gemini) for name, src in SYSTEMS.items()}


def composed_from_winners(winners: Mapping[str, str]) -> dict[str, str]:
    """winners: task -> 'frozen' | 'gemini'. Returns the source map of the selected architecture."""
    return {t: ("gemini" if winners.get(t) == "gemini" else "frozen") for t in ("intent", "topic", "sentiment")}
