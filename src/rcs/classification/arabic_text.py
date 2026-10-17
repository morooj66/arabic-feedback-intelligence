"""Arabic text preparation shared by the OFFLINE classification candidates (not imported by any runtime path).

Fixes the frozen vectorizers' defects: the default `\\b\\w\\w+\\b` tokenizer returns no / broken tokens for diacritised words, and there is no
spelling normalisation. `normalize_arabic` removes diacritics and tatweel (so a diacritised word is never dropped), unifies alef variants and digits,
unifies the question mark and collapses letter elongation. It keeps ta-marbuta, alef-maksura and hamza-on-waw/ya (meaningful distinctions).
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_ALEF = str.maketrans("أإآٱ", "اااا")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ELONG = re.compile(r"(.)\1{2,}")


def normalize_arabic(text) -> str:
    s = unicodedata.normalize("NFKC", str(text if text is not None else ""))
    s = s.replace("ـ", "")
    s = _DIACRITICS.sub("", s)
    s = s.translate(_ALEF).translate(_DIGITS).replace("?", "؟")
    s = _ELONG.sub(r"\1\1", s)
    return " ".join(s.lower().split())


def has_question(text) -> bool:
    """Deterministic question cue (question mark or an interrogative word) - a model INPUT feature, never a decision rule."""
    n = normalize_arabic(text)
    if "؟" in n:
        return True
    toks = set(re.sub(r"[،,.;:!()\"'«»]", " ", n).split())
    return bool(toks & {"هل", "كيف", "ماذا", "لماذا", "متى", "اين", "كم", "ايش", "وش", "ليش", "شنو", "ليه"}) or any(
        p in n for p in ("ما معنى", "ما هو", "ما هي", "ما الفرق", "ما حكم", "ما المقصود")
    )
