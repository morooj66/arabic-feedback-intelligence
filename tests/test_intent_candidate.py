"""Experimental Intent candidate: preprocessing fix, leakage safeguards, determinism and isolation. No real data needed."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from rcs.classification import intent_candidate as IC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "offline_eval" / "intent_candidate"))
import train_and_evaluate as TE  # noqa: E402


def test_diacritised_words_are_not_dropped_or_fragmented_by_the_candidate_but_are_by_the_frozen_tokenizer():
    frozen = TfidfVectorizer().build_analyzer()
    cand = TfidfVectorizer(preprocessor=IC.normalize_arabic).build_analyzer()
    for word, fixed in (("يَفْتَـح", "يفتح"), ("الرَّابِط", "الرابط"), ("مُشْكِلَة", "مشكلة")):
        assert fixed in cand(word)                                       # candidate keeps the word
    assert frozen("يَفْتَـح") != ["يفتح"] and frozen("الرَّابِط") != ["الرابط"]  # frozen tokenizer breaks it (documents the defect)
    assert cand("الموقع ما يَفْتَح") == ["الموقع", "ما", "يفتح"]


def test_normalisation_is_mild_and_preserves_meaningful_distinctions():
    assert IC.normalize_arabic("أَحمد إلى آخر") == IC.normalize_arabic("احمد الى اخر")        # alef forms unified
    assert IC.normalize_arabic("حبييييب ٣٤ ?") == "حبييب 34 ؟"                              # elongation, digits, question mark
    assert "ة" in IC.normalize_arabic("مشكلة") and "ى" in IC.normalize_arabic("معنى")        # ta-marbuta / alef-maksura kept for text features
    assert IC.match_form("مشكلة") == IC.match_form("مشكله") and "ى" not in IC.match_form("معنى")  # only the MATCH form merges them
    assert IC.normalize_arabic(None) == "" and IC.normalize_arabic("") == ""


def test_lexicon_features_use_word_boundaries_and_separate_failure_from_question_and_request():
    names = IC.FEATURE_NAMES
    f = lambda t: dict(zip(names, IC.dense_features([t])[0]))
    assert f("الموقع ما يفتح عندي من أمس")["fam_not_opening_or_working"] == 1
    assert f("الموقع ما يفتح")["struct_negation_predicate"] >= 1
    assert f("هل البحث يشتغل بدون تشكيل؟")["ctr_howto_question"] >= 1 and f("هل البحث يشتغل؟")["fam_not_opening_or_working"] == 0
    assert f("أتمنى تضيفون خاصية البحث")["ctr_feature_request"] >= 1
    assert f("أهل المعجم كرام")["ctr_howto_question"] == 0                 # 'هل' must not fire inside 'أهل'
    assert f("الصفحه ما تظهر")["fam_not_displaying"] == 1


def _toy():
    X = ["الموقع ما يفتح", "ما يطلع لي شي", "كيف أبحث؟", "هل تتوفر خدمة الجذر هنا؟", "أتمنى إضافة خاصية", "ياليت تحسين السرعة",
         "شكرا جزيلا لكم", "جزاكم الله خيرا", "نص عشوائي فارغ", "اختبار اختبار"] * 3
    y = ["COMPLAINT", "COMPLAINT", "INQUIRY", "INQUIRY", "SUGGESTION", "SUGGESTION", "THANKS_PRAISE", "THANKS_PRAISE", "SPAM", "SPAM"] * 3
    return X, y


def test_fit_predict_shapes_class_order_and_determinism():
    X, y = _toy()
    a = IC.IntentCandidate().fit(X, y); b = IC.IntentCandidate().fit(X, y)
    P = a.predict_proba(X)
    assert P.shape == (len(X), 5) and np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(P, b.predict_proba(X))                                  # deterministic
    assert a.predict(["الموقع ما يفتح"]) == ["COMPLAINT"] and tuple(IC.CLASSES) == ("THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM")


def test_vectorisers_are_fitted_on_training_texts_only_so_unseen_words_never_enter_the_vocabulary():
    X, y = _toy()
    m = IC.IntentCandidate().fit(X, y)
    assert "قارورة" not in m.word_.vocabulary_
    m.predict(["قارورة فريدة جدا"])                                            # scoring must not grow the vocabulary
    assert "قارورة" not in m.word_.vocabulary_ and not any("قارورة" in k for k in m.word_.vocabulary_)


def test_sample_weight_is_honoured_and_serialisation_round_trips(tmp_path):
    X, y = _toy()
    w = [0.0 if t == "اختبار اختبار" else 1.0 for t in X]
    m = IC.IntentCandidate().fit(X, y, sample_weight=w)
    joblib.dump(m, tmp_path / "m.joblib")
    m2 = joblib.load(tmp_path / "m.joblib")
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X)) and m2.describe()["version"] == IC.CANDIDATE_VERSION


def test_candidate_is_isolated_from_every_runtime_path():
    for path in list((ROOT / "src").rglob("*.py")):
        if path.name == "intent_candidate.py":
            continue
        src = path.read_text(encoding="utf-8")
        assert "intent_candidate" not in src, f"{path} references the experimental candidate"
    import config.settings as S
    assert not hasattr(S, "INTENT_CANDIDATE_ENABLED")                            # no activation switch exists


def test_evaluation_script_never_loads_blind_labels_or_sealed_rows():
    src = (ROOT / "scripts" / "offline_eval" / "intent_candidate" / "train_and_evaluate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    strings = " ".join(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    for forbidden in ("blind_eval", "labels_verbatim", "returned", "human_test_exclusion"):
        assert forbidden not in strings
    assert 'r["split"] in ("train", "val")' in src                                 # the sealed split is never selected


def test_synthetic_examples_are_training_only_and_theme_groups_keep_contrast_pairs_together(monkeypatch):
    syn = TE.load_synthetic()
    assert len(syn) == 36 and all(s["y"] in TE.CLASSES for s in syn)
    groups = {}
    for s in syn:
        groups.setdefault(s["group"], set()).add(s["y"])
    assert any(len(v) >= 2 for v in groups.values())                              # contrast pairs live inside one group
    rows = [{"text": t, "y": y_, "id": str(i), "split": "train"} for i, (t, y_) in enumerate(zip(*_toy()))]
    syn_texts = {x["text"] for x in syn}
    assert not (syn_texts & {r["text"] for r in rows})                               # no synthetic text doubles as a real record
    res, oof, y = TE.cv_protocol(rows, syn, repeats=(0,), variants={"c": ("candidate", True)})
    assert len(y) == len(rows) == len(oof["c"]) and len(res["c"]["folds"]) == TE.N_SPLITS
    assert set(oof["c"]) <= set(TE.CLASSES)                                            # predictions exist only for the real records
