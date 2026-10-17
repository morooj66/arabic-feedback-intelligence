"""OFFLINE lightweight Arabic classifier (word + character TF-IDF -> balanced Logistic Regression) used by the classification v2 study for
Intent, Topic and Sentiment. EXPERIMENTAL: not imported by any runtime path, never activated by this module.

Vectorizers are fitted on the TRAINING texts only (inside every fold). Preprocessing is `arabic_text.normalize_arabic` (diacritics never drop a
word). Optional dense inputs enter the learned model as extra columns - there is no decision rule anywhere:
  * text features: question cue, log length, has-diacritics flag
  * extra: caller-supplied numeric columns (e.g. log-probabilities of the pretrained sentiment model)
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rcs.classification.arabic_text import has_question, normalize_arabic

DEFAULTS = dict(word_ngram_range=(1, 2), char_ngram_range=(2, 5), char_min_df=2, C=1.0, class_weight="balanced", max_iter=3000,
                text_features=False, dense_scale=1.0, random_state=42, use_char=True, use_tfidf=True)
_DIAC = re.compile(r"[ً-ٰٟـ]")


def text_feature_block(texts: Sequence[str]) -> np.ndarray:
    rows = []
    for t in texts:
        t = str(t or "")
        n = normalize_arabic(t)
        rows.append([float(has_question(t)), np.log1p(len(n)) / 6.0, float(bool(_DIAC.search(t)))])
    return np.asarray(rows, dtype=float)


class LexicalClassifier:
    def __init__(self, labels: Sequence[str], **params):
        self.labels = list(labels)
        self.params = {**DEFAULTS, **params}

    def _dense(self, texts, extra):
        parts = []
        if self.params["text_features"]:
            parts.append(text_feature_block(texts))
        if extra is not None:
            parts.append(np.asarray(extra, dtype=float))
        return np.hstack(parts) * self.params["dense_scale"] if parts else None

    def _matrix(self, texts, extra, fit):
        p = self.params
        if not p["use_tfidf"]:
            d = self._dense(texts, extra)
            return csr_matrix(d)
        if fit:
            self.word_ = TfidfVectorizer(preprocessor=normalize_arabic, ngram_range=tuple(p["word_ngram_range"]), min_df=1, sublinear_tf=True, max_features=50000)
            blocks = [self.word_.fit_transform(texts)]
            if p["use_char"]:
                self.char_ = TfidfVectorizer(preprocessor=normalize_arabic, analyzer="char_wb", ngram_range=tuple(p["char_ngram_range"]),
                                             min_df=p["char_min_df"], sublinear_tf=True, max_features=60000)
                blocks.append(self.char_.fit_transform(texts))
        else:
            blocks = [self.word_.transform(texts)]
            if p["use_char"]:
                blocks.append(self.char_.transform(texts))
        d = self._dense(texts, extra)
        if d is not None:
            blocks.append(csr_matrix(d))
        return hstack(blocks).tocsr()

    def fit(self, texts, y, *, extra=None, sample_weight=None):
        X = self._matrix(list(texts), extra, fit=True)
        self.clf_ = LogisticRegression(C=self.params["C"], class_weight=self.params["class_weight"], max_iter=self.params["max_iter"], random_state=self.params["random_state"])
        self.clf_.fit(X, list(y), sample_weight=None if sample_weight is None else np.asarray(sample_weight, dtype=float))
        self._order = [list(self.clf_.classes_).index(c) if c in self.clf_.classes_ else None for c in self.labels]
        return self

    def predict_proba(self, texts, *, extra=None) -> np.ndarray:
        raw = self.clf_.predict_proba(self._matrix(list(texts), extra, fit=False))
        out = np.zeros((raw.shape[0], len(self.labels)))
        for j, idx in enumerate(self._order):
            if idx is not None:
                out[:, j] = raw[:, idx]
        return out

    def predict(self, texts, *, extra=None) -> list[str]:
        return [self.labels[i] for i in self.predict_proba(texts, extra=extra).argmax(1)]

    def describe(self) -> dict:
        return {"labels": self.labels, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params.items()},
                "n_word_features": len(self.word_.vocabulary_) if self.params["use_tfidf"] else 0, "n_char_features": len(self.char_.vocabulary_) if self.params["use_tfidf"] and self.params["use_char"] else 0}
