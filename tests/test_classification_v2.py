"""Classification v2 (offline study + experimental artifacts): preprocessing, leakage guards, isolation, artifact integrity."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from rcs.classification.arabic_text import has_question, normalize_arabic
from rcs.classification.lexical_classifier import LexicalClassifier

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data/models/experimental/classification_v2"
OUT = ROOT / "data/evaluation/classification_v2"
needs_artifacts = pytest.mark.skipif(not (ART / "intent" / "model.joblib").exists(), reason="local experimental artifacts not built")


def test_diacritised_words_are_never_dropped():
    assert normalize_arabic("الموقع لا يَفْتَحُ") == "الموقع لا يفتح"
    assert normalize_arabic("الرَّابِط") == "الرابط"
    assert normalize_arabic("كلمـــــة") == "كلمة"                      # tatweel + elongation
    assert normalize_arabic("أب إب آب") == "اب اب اب" and normalize_arabic("١٢٣") == "123"
    assert "ة" in normalize_arabic("مدرسة") and "ى" in normalize_arabic("معنى")        # meaningful distinctions kept


def test_question_cue_feature():
    assert has_question("ما معنى كلمة سوار") and has_question("هل يعمل؟") and has_question("كيف أضيف كلمة")
    assert not has_question("الموقع ما يفتح عندي")


def test_vectorizers_are_fitted_on_training_texts_only_and_diacritics_do_not_break_features():
    train = ["الموقع ما يفتح", "شكرا لكم", "هل يمكن الاضافة؟", "كلمة جميلة"] * 3
    m = LexicalClassifier(["a", "b"]).fit(train, ["a", "b", "b", "a"] * 3)
    assert "جديدة" not in m.word_.vocabulary_
    assert m.predict(["الموقع ما يَفْتَحُ"]) == m.predict(["الموقع ما يفتح"])           # diacritics change nothing
    assert np.allclose(m.predict_proba(["شكرا"]), m.predict_proba(["شكرا"]))


def test_dense_only_and_extra_columns_work_and_are_deterministic():
    X = ["كلمة جميلة", "الموقع ما يفتح", "شكرا لكم", "هل يمكن؟"] * 4
    y = ["a", "b", "a", "b"] * 4
    extra = np.random.RandomState(0).normal(size=(len(X), 5))
    a = LexicalClassifier(["a", "b"], use_tfidf=False, text_features=True).fit(X, y, extra=extra).predict_proba(X, extra=extra)
    b = LexicalClassifier(["a", "b"], use_tfidf=False, text_features=True).fit(X, y, extra=extra).predict_proba(X, extra=extra)
    assert np.allclose(a, b) and a.shape == (len(X), 2)


def test_experimental_modules_are_only_reachable_through_the_opt_in_pilot_mode():
    names = ("arabic_text", "lexical_classifier", "candidate_v2")
    allowed = {"arabic_text", "lexical_classifier", "candidate_v2", "v2_stack"}          # the experimental family itself
    for p in (ROOT / "src").rglob("*.py"):
        if p.stem in allowed:
            continue
        text = p.read_text(encoding="utf-8")
        assert not any(re.search(rf"\b{n}\b", text) for n in names), p
        if p.stem != "runtime":
            assert "v2_stack" not in text, p
    runtime_text = (ROOT / "src/rcs/classification/runtime.py").read_text()
    assert not re.search(r"^(from|import) .*v2_stack", runtime_text, re.M)                  # only lazy, inside the pilot branch
    assert 'CLASSIFICATION_MODE", "legacy_rcs"' in (ROOT / "config/settings.py").read_text()   # default stays the frozen stack


def test_study_scripts_never_touch_blind_or_sealed_data():
    for p in (ROOT / "scripts/offline_eval/classification_v2").glob("*.py"):
        if p.name == "final_validation.py":           # the final validation stage deliberately reads the exposed blind labels as a secondary, never-selecting check
            continue
        text = p.read_text(encoding="utf-8")
        assert "blind_eval" not in text and "labels_verbatim" not in text, p
        assert "HUMAN_TEST" not in text and "final_test/human" not in text and "test_preds" not in text, p
        assert 'split"] in ("train", "val")' in text or p.name != "common.py"


@needs_artifacts
def test_study_provenance_and_exclusions():
    res = json.loads((OUT / "study_results.json").read_text())
    assert res["meta"]["blind_75_used"] is False and res["meta"]["sealed_human_test_read"] is False and res["meta"]["synthetic_in_evaluation"] is False
    assert res["meta"]["gold_sha256"].startswith("ac05bb8dc47a9787")                       # Gold file unchanged
    import csv

    excl = list(csv.DictReader(open(OUT / "training_exclusions.csv", encoding="utf-8")))
    assert len(excl) == res["meta"]["policy_conflict_rows_excluded_from_training"] > 0
    assert all(r["rule"].startswith("R-INTENT-DEF") and r["original_gold_intent"] in ("INQUIRY", "SPAM") for r in excl)   # recorded with original gold, never relabelled


@needs_artifacts
@pytest.mark.parametrize("task", ["intent", "topic", "sentiment"])
def test_artifacts_are_experimental_verifiable_and_clean(task):
    from rcs.classification.candidate_v2 import load_candidate, verify_artifact

    d = ART / task
    verify_artifact(d)
    model, cfg = load_candidate(d)
    assert cfg["status"] == "EXPERIMENTAL - NOT ACTIVATED"
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["blind_75_used"] is False and prov["sealed_human_test_read"] is False and prov["frozen_models_modified"] is False
    (d / "tamper.tmp").write_text("x")
    (d / "tamper.tmp").unlink()
    sums = (d / "SHA256SUMS").read_text()
    assert hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest() in sums
    assert model.labels == cfg["labels"]


@needs_artifacts
def test_a_modified_artifact_is_refused(tmp_path):
    import shutil

    from rcs.classification.candidate_v2 import verify_artifact

    d = tmp_path / "x"
    shutil.copytree(ART / "sentiment", d)
    (d / "config.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError):
        verify_artifact(d)


@needs_artifacts
def test_saved_models_reproduce_from_cached_features():
    from rcs.classification.candidate_v2 import load_candidate
    from scripts.offline_eval.classification_v2.common import load_dev

    rows = load_dev()[:40]
    E5 = np.load(OUT / "e5_small_dev563.npy")[:40]
    cam = np.log(np.clip(np.array(json.load(open(OUT / "camelbert_probs_dev563.json"))["probabilities"])[:40], 1e-6, 1))
    texts = [r["comment_text"] for r in rows]
    for task, extra in (("intent", np.hstack([E5, cam])), ("sentiment", np.hstack([E5, cam])), ("topic", E5)):
        m, cfg = load_candidate(ART / task)
        assert tuple(cfg["extra_features"]) == (("e5", "camel") if task != "topic" else ("e5",))
        P1, P2 = m.predict_proba(texts, extra=extra), m.predict_proba(texts, extra=extra)
        assert np.allclose(P1, P2) and np.allclose(P1.sum(1), 1.0)


def test_frozen_production_model_is_unchanged():
    p = ROOT / "data/models/final_test/intent_tfidf_logistic.joblib"
    if p.exists():
        assert hashlib.sha256(p.read_bytes()).hexdigest().startswith("234ed38f9a226bbcb2542b59")
