"""EXPERIMENTAL Intent classifier candidate (v1). OFFLINE ONLY. NOT imported by any runtime path, NOT activated, NOT a production model.

What it fixes relative to the frozen Intent model (word TF-IDF + balanced LogisticRegression on raw text):
  * Arabic text preprocessing. The frozen vectorizer's default tokenizer returns NO tokens for diacritised words
    ("يَفْتَح" -> []) and does no spelling normalisation. Here diacritics / tatweel are removed, alef forms and digits are unified,
    the question mark is unified and letter elongation is collapsed BEFORE tokenisation, so diacritised words are never dropped.
    Two levels, to preserve meaningful distinctions: a MILD form feeds the TF-IDF text features (ta-marbuta, alef-maksura, hamza on
    waw/ya are kept), and a MATCH form (additionally ta-marbuta->ha, alef-maksura->ya) is used only to evaluate dialect-spelling-robust
    lexicon features.
  * Word AND character n-gram TF-IDF (dialect morphology / spelling variants).
  * Learned contextual features (dense block): families of failure-report wording, contrast cues (how-to question, feature request,
    lexical-error claim), thanks/praise, courtesy opening, and STRUCTURAL context (negation + predicate, attempt-and-fail markers,
    question mark, definition-like colon form, length, diacritic presence). They enter a LEARNED model as input columns; there is no
    keyword decision rule anywhere.
  * Class-imbalance handling (balanced class weights) and optional down-weighted synthetic contrast examples, which are always
    tagged and never mixed into real-data evaluation.

The fitted object stores everything needed to reproduce it. Training / evaluation lives in scripts/offline_eval/intent_candidate/.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CANDIDATE_VERSION = "intent_candidate_v1"
CLASSES = ("THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM")  # frozen taxonomy, frozen label order

DEFAULT_PARAMS = {
    "word_ngram_range": (1, 2), "word_max_features": 50000,
    "char_ngram_range": (2, 5), "char_min_df": 2, "char_max_features": 60000,
    "C": 1.0, "class_weight": "balanced", "max_iter": 3000, "dense_feature_scale": 2.0,
    "synthetic_weight": 1.0, "random_state": 42,
}

# --------------------------------------------------------------------------------------------- preprocessing

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_TATWEEL = "ـ"
_ALEF = str.maketrans("أإآٱ", "اااا")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ELONG = re.compile(r"(.)\1{2,}")


def normalize_arabic(text) -> str:
    """MILD normalisation for TF-IDF text features. Never returns fewer tokens than the input has words (diacritics are removed,
    not used as separators). Keeps ta-marbuta, alef-maksura and hamza-on-waw/ya (meaningful distinctions)."""
    s = unicodedata.normalize("NFKC", str(text if text is not None else ""))
    s = s.replace(_TATWEEL, "")
    s = _DIACRITICS.sub("", s)
    s = s.translate(_ALEF).translate(_DIGITS).replace("?", "؟")
    s = _ELONG.sub(r"\1\1", s)
    return " ".join(s.lower().split())


def match_form(text) -> str:
    """MATCH form: MILD + ta-marbuta->ha, alef-maksura->ya. Used only for dialect-spelling-robust lexicon matching."""
    return normalize_arabic(text).translate(str.maketrans("ةى", "هي"))


# ------------------------------------------------------------------------------------------ learned features

LEXICON = {
    # reports of an actual failure / unsuccessful experience (colloquial + MSA); candidates for COMPLAINT (+ TECHNICAL_ISSUE topic)
    "fam_not_opening_or_working": ["ما يفتح", "ما تفتح", "ما فتح", "لا يفتح", "لا تفتح", "ما يشتغل", "ما اشتغل", "ما تشتغل", "مو شغال", "مب شغال",
                                   "ما يعمل", "لا يعمل", "لا تعمل", "مو شغاله"],
    "fam_not_displaying": ["ما ظهر", "ما يظهر", "ما تظهر", "ما يطلع", "ما طلع", "ما تطلع", "لا يظهر", "لا تظهر"],
    "fam_hang_or_crash": ["يعلق", "علق", "معلق", "يتعطل", "تعطل", "بطيء"],
    "fam_unsuccessful_attempt": ["ما قدرت", "ما اقدر", "ما استطعت", "حاولت وما", "حاولت مرارا", "ما ضبط", "ما ضبطت"],
    "fam_repeated_failure": ["كل ما ابحث", "كل ما ادخل", "كل مره"],
    "fam_problem_statement": ["فيه مشكله", "في مشكله", "عندي مشكله", "مشكله في", "مشكله"],
    "fam_wrong_result": ["النتيجه غلط", "النتيجه خطا", "نتيجه غلط"],
    # contrast cues: the same vocabulary used to ASK, to PROPOSE, or to correct an entry
    "ctr_howto_question": ["كيف", "طريقه", "وين اجد", "وين الاقي", "هل يمكن", "هل اقدر", "هل", "؟"],
    "ctr_feature_request": ["اتمنى", "ياليت", "اقترح", "ممكن تضيفون", "تضيفون", "ارجو اضافه", "خاصيه", "ميزه"],
    "ctr_lexical_error_claim": ["خطا", "غير صحيح", "غير صحيحه", "تصويب", "الصواب", "مكرر", "غلط"],
    "thanks_praise": ["شكرا", "جزاكم الله", "بارك الله", "ممتاز", "رائع", "مشكورين", "احسنتم", "تسلمون", "ما قصرتم", "الله يعطيكم"],
    "courtesy_opening": ["السلام عليكم", "تحيه طيبه", "بعد التحيه", "مساء الخير", "صباح الخير"],
    "definition_marker": ["معناها", "معنى", "يعني", "تعني", ":"],
}
_NEG_PRED = re.compile(r"(?:^|\s)(?:ما|مو|مب|لا|لم|لن|ماعاد)\s+[^\s]{2,}")
_ATTEMPT = ("حاولت", "جربت", "كل ما", "كلما", "اكثر من مره", "مره بعد مره", "من امس", "من الصبح")
_HAS_DIACRITIC_SRC = re.compile(r"[ً-ٰٟـ]")

FEATURE_NAMES = (list(LEXICON) + ["struct_negation_predicate", "struct_attempt_or_duration_marker", "struct_attempt_or_duration_then_negation", "struct_question_mark",
                                   "struct_log_length", "struct_log_tokens", "struct_has_diacritics", "struct_latin_share"])


def _count(m: str, phrase: str) -> int:
    """Occurrences of `phrase` in the MATCH-form text. Punctuation cues match anywhere; word cues match on token boundaries so that
    e.g. 'هل' does not fire inside 'اهل'."""
    phrase = phrase.strip()
    if phrase in ("؟", ":"):
        return m.count(phrase)
    return (" " + m + " ").count(" " + phrase + " ")


def dense_features(texts: Sequence[str]) -> np.ndarray:
    rows = []
    for t in texts:
        m = match_form(t)
        mp = re.sub(r"[،,.;!؟:()\"'«»\[\]]", " ", m)       # punctuation -> space for token-boundary matching (the ':' and '؟' cues use `m`)
        lex = [min(3.0, float(sum(_count(m if phrase.strip() in ("؟", ":") else " ".join(mp.split()), phrase) for phrase in phrases)))
               for phrases in LEXICON.values()]
        neg = min(3.0, float(len(_NEG_PRED.findall(m))))
        att = float(any(a in m for a in _ATTEMPT))
        att_neg = float(att and neg > 0)
        q = float("؟" in m)
        letters = [c for c in str(t) if c.isalpha()]
        latin = sum(1 for c in letters if "a" <= c.lower() <= "z") / max(1, len(letters))
        rows.append(lex + [neg, att, att_neg, q, np.log1p(len(m)) / 6.0, np.log1p(len(m.split())) / 4.0,
                           float(bool(_HAS_DIACRITIC_SRC.search(str(t)))), latin])
    return np.asarray(rows, dtype=float)


# --------------------------------------------------------------------------------------------------- model


class IntentCandidate:
    """Word + character TF-IDF (fitted on the TRAINING texts only) + dense learned features -> balanced LogisticRegression."""

    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}
        self.version = CANDIDATE_VERSION

    def _matrix(self, texts: Sequence[str], fit: bool):
        p = self.params
        if fit:
            self.word_ = TfidfVectorizer(preprocessor=normalize_arabic, ngram_range=tuple(p["word_ngram_range"]), min_df=1,
                                         sublinear_tf=True, max_features=p["word_max_features"])
            self.char_ = TfidfVectorizer(preprocessor=normalize_arabic, analyzer="char_wb", ngram_range=tuple(p["char_ngram_range"]),
                                         min_df=p["char_min_df"], sublinear_tf=True, max_features=p["char_max_features"])
            parts = [self.word_.fit_transform(texts), self.char_.fit_transform(texts)]
        else:
            parts = [self.word_.transform(texts), self.char_.transform(texts)]
        parts.append(csr_matrix(dense_features(texts) * p["dense_feature_scale"]))
        return hstack(parts).tocsr()

    def fit(self, texts: Sequence[str], y: Sequence[str], sample_weight: Sequence[float] | None = None):
        X = self._matrix(list(texts), fit=True)
        self.clf_ = LogisticRegression(C=self.params["C"], class_weight=self.params["class_weight"], max_iter=self.params["max_iter"],
                                       random_state=self.params["random_state"])
        self.clf_.fit(X, list(y), sample_weight=None if sample_weight is None else np.asarray(sample_weight, dtype=float))
        order = [list(self.clf_.classes_).index(c) for c in CLASSES]
        self._order = order
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """(n, 5) probabilities in the frozen class order CLASSES."""
        return self.clf_.predict_proba(self._matrix(list(texts), fit=False))[:, self._order]

    def predict(self, texts: Sequence[str]) -> list[str]:
        P = self.predict_proba(texts)
        return [CLASSES[int(i)] for i in P.argmax(axis=1)]

    def describe(self) -> dict:
        return {"version": self.version, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params.items()},
                "classes": list(CLASSES), "n_word_features": len(self.word_.vocabulary_), "n_char_features": len(self.char_.vocabulary_),
                "dense_features": FEATURE_NAMES}
