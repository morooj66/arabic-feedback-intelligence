"""Shared helpers for the base-model error analysis. LOCAL ONLY; development data + aggregate counts on the blind set.

RULES (hard): the 75 blind labels are used ONLY for aggregate diagnostics (counts of confusions / expression hits). Nothing here fits,
tunes, selects a threshold, or derives an example from them. Comment text is never printed or written to a report.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "data" / "evaluation" / "error_analysis"
GOLD = ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
RAW_XLSX = ROOT / "data" / "raw" / "historical-export.xlsx"
BLIND_LABELS = ROOT / "data" / "evaluation" / "blind_eval_v1" / "returned" / "labels_verbatim_from_numbers.json"
BLIND_MAP = ROOT / "data" / "evaluation" / "blind_eval_v1" / "coordinator_only" / "blind_eval_mapping.local.csv"

_DIAC = re.compile(r"[ً-ٰٟـ]")
_MAP = str.maketrans("أإآٱىة", "اااايه")


def norm_ar(t: str) -> str:
    """Remove diacritics/tatweel, unify alef / ya / ta-marbuta, collapse whitespace (used for matching only)."""
    return " ".join(unicodedata.normalize("NFKC", _DIAC.sub("", str(t or ""))).translate(_MAP).split())


# Expression families, declared BEFORE counting. The first list per family is exactly what was asked for; the rest are plain variants.
FAMILIES: dict[str, list[str]] = {
    "not_opening_or_working": ["ما يفتح", "ما تفتح", "ما فتح", "لا يفتح", "لا تفتح", "ما يشتغل", "ما اشتغل", "ما تشتغل", "مو شغال", "مب شغال",
                               "ما يعمل", "لا يعمل", "لا تعمل", "مو شغاله"],
    "not_displaying": ["ما ظهر", "ما يظهر", "ما تظهر", "ما يطلع", "ما طلع", "ما تطلع", "لا يظهر", "لا تظهر"],
    "hang_or_crash": ["يعلق", "علق", "معلق", "يتعطل", "تعطل", "بطيء"],
    "unsuccessful_attempt": ["ما قدرت", "ما اقدر", "ما استطعت", "حاولت وما", "حاولت مرارا", "ما ضبط", "ما ضبطت"],
    "repeated_failure": ["كل ما ابحث", "كل ما ادخل", "كل مره"],
    "problem_statement": ["فيه مشكله", "في مشكله", "عندي مشكله", "مشكله في", "مشكله"],
    "wrong_result": ["النتيجه غلط", "النتيجه خطا", "نتيجه غلط"],
}
# Contrast cues (the intent distinctions the task asks us to keep apart)
CONTRAST = {
    "howto_question": ["كيف", "طريقه", "وين اجد", "وين الاقي", "هل يمكن", "هل اقدر", "؟", "?"],
    "feature_request": ["اتمنى", "ياليت", "اقترح", "ممكن تضيفون", "تضيفون", "ارجو اضافه", "خاصيه", "ميزه"],
    "lexical_error_claim": ["خطا", "غير صحيح", "غير صحيحه", "تصويب", "الصواب", "مكرر", "غلط"],
}


def hits(text: str, phrases: list[str]) -> bool:
    n = norm_ar(text)
    return any(norm_ar(p) in n for p in phrases)


def any_failure(text: str) -> bool:
    return any(hits(text, v) for v in FAMILIES.values())


def load_dev() -> list[dict]:
    """Labeled development rows (train + val), raw comment text kept local. Sealed human_test rows are dropped on read."""
    rows = []
    with open(GOLD, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in ("train", "val") and r["gold_annotation_status"] == "LABELED" and r["gold_primary_intent"]:
                rows.append(r)
    return rows


def load_blind() -> list[dict]:
    lab = json.loads(BLIND_LABELS.read_text(encoding="utf-8"))
    comp = {r["record_id"]: r["component"] for r in csv.DictReader(open(BLIND_MAP, encoding="utf-8"))}
    return [dict(r, component=comp[r["record_id"]]) for r in lab]
