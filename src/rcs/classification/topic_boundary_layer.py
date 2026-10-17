"""OPT-IN post-classification layer for the WORD_SUBMISSION / ENTRY_CORRECTION boundary. DEFAULT OFF.

Spec (frozen, zero learned parameters; sha256 recorded in scripts/offline_eval/jev_dev_validation/candidate_ws_ec_layer.py):

    if base_topic in {WORD_SUBMISSION, ENTRY_CORRECTION}:
        topic = ENTRY_CORRECTION  if (lemma_present or error_claim_cue(text))  else  WORD_SUBMISSION
    else:
        topic unchanged

`lemma_present` is the dictionary-entry metadata already stored on the review item (not visitor data). Intent, sentiment, confidences,
routing inputs other than the topic label, and every other topic class are left exactly as the base classifier produced them.
Enabled only by RCS_TOPIC_BOUNDARY_LAYER=1 (config.settings.TOPIC_BOUNDARY_LAYER_ENABLED). Development evidence only; it has not
been evaluated on independent data, so it must not be enabled in production before that evaluation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

WORD_SUBMISSION = "WORD_SUBMISSION"
ENTRY_CORRECTION = "ENTRY_CORRECTION"
LAYER_NAME = "ws_ec_boundary_layer_v1"
PROVENANCE_SUFFIX = "+WS_EC_BOUNDARY_LAYER_V1"

# Explicit error-claim wording (taken from the signal column of LABELING_GUIDE.md plus plain spelling variants).
ERROR_CLAIM_CUES = (
    "خطأ", "خطا", "خاطئ", "خاطئة", "غلط", "غير صحيح", "غير صحيحة", "تصويب", "تصحيح", "الصواب", "مكرر", "تكرار",
    "ناقص", "لا يتناسب", "لا معنى", "لا يمت", "اضطرب", "مضطرب",
)


def lemma_present(value) -> bool:
    """True when a source lemma exists (works for None, NaN, 'nan', 'None', '' and real strings)."""
    if value is None:
        return False
    s = str(value).strip()
    return s not in ("", "nan", "None", "NaN", "null")


def has_error_claim(text: str | None) -> bool:
    t = " ".join(str(text or "").split())
    return any(cue in t for cue in ERROR_CLAIM_CUES)


def boundary_topic(base_topic: str, text: str | None, has_lemma: bool) -> str:
    if base_topic not in (WORD_SUBMISSION, ENTRY_CORRECTION):
        return base_topic
    return ENTRY_CORRECTION if (has_lemma or has_error_claim(text)) else WORD_SUBMISSION


def apply_boundary_layer(
    topic_preds: Mapping[str, tuple[str, float | None]],
    comment_ids: Sequence[str],
    texts: Sequence[str],
    lemmas: Sequence | None,
) -> dict[str, tuple[str, float | None]]:
    """Returns a NEW dict {comment_id: (topic, confidence)}; confidences are passed through unchanged."""
    out: dict[str, tuple[str, float | None]] = {}
    for i, cid in enumerate(comment_ids):
        label, conf = topic_preds[cid]
        has_lemma = lemma_present(lemmas[i]) if lemmas is not None else False
        out[cid] = (boundary_topic(label, texts[i], has_lemma), conf)
    return out
