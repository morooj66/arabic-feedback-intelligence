"""Controlled-local-pilot integration of the classification_v2 stack (Intent I10, frozen MARBERTv2 topic, Sentiment S5).
Uses the real cached local models (no network). Skipped when the experimental artifacts or cached weights are absent."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from config import settings
from rcs.classification import frozen_inference, runtime, v2_stack
from rcs.classification.lexical_classifier import LexicalClassifier
from rcs.classification.v2_stack import V2Stack, V2StackError

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data/models/experimental/classification_v2"
HUB = Path.home() / ".cache/huggingface/hub"
needs_models = pytest.mark.skipif(
    not ((ART / "intent/model.joblib").exists() and (HUB / "models--intfloat--multilingual-e5-small").exists()
         and (HUB / "models--CAMeL-Lab--bert-base-arabic-camelbert-da-sentiment").exists()),
    reason="experimental artifacts or cached local weights not available",
)


@pytest.fixture(scope="module")
def stack():
    s = V2Stack()
    s.load()
    return s


@pytest.fixture
def pilot(monkeypatch, stack):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "local_v2_pilot")
    v2_stack.reset_for_testing(stack)
    yield stack
    v2_stack.reset_for_testing(None)


def _ids(n):
    return [f"c{i}" for i in range(n)]


# ----------------------------------------------------------------------------------------------- taxonomy / schema
@needs_models
def test_artifact_labels_are_exactly_the_frozen_taxonomy(stack):
    assert set(stack.models["intent"].labels) == set(settings.PRIMARY_INTENTS) and len(stack.models["intent"].labels) == len(settings.PRIMARY_INTENTS)
    assert set(stack.models["sentiment"].labels) == set(settings.SENTIMENTS) and len(stack.models["sentiment"].labels) == len(settings.SENTIMENTS)


@needs_models
def test_every_taxonomy_code_is_reachable_and_outputs_are_always_in_the_taxonomy(stack):
    from scripts.offline_eval.classification_v2.common import load_dev

    texts = [r["comment_text"] for r in load_dev()]
    intents = stack.predict_intent(texts)
    sents = stack.predict_sentiment(_ids(len(texts)), texts)
    assert {l for l, _ in intents} == set(settings.PRIMARY_INTENTS)                     # all 5 intent codes are produced
    assert {l for l, _ in sents.values()} == set(settings.SENTIMENTS)                   # all 3 sentiment codes are produced
    assert all(l in settings.PRIMARY_INTENTS and 0.0 <= c <= 1.0 for l, c in intents)
    assert all(l in settings.SENTIMENTS and 0.0 <= c <= 1.0 for l, c in sents.values())


@needs_models
def test_prediction_schema_matches_the_frozen_runtime_contract(pilot):
    T = ["شكرا جزيلا لكم", "هل يمكن اضافة خاصية البحث؟"]
    intent = runtime.predict_intent(T)
    assert isinstance(intent, list) and len(intent) == 2 and all(isinstance(p, tuple) and isinstance(p[0], str) and isinstance(p[1], float) for p in intent)
    sent = runtime.predict_sentiment(["a", "b"], T)
    assert isinstance(sent, dict) and set(sent) == {"a", "b"} and all(isinstance(v, tuple) and isinstance(v[1], float) for v in sent.values())
    assert runtime.predict_intent([]) == [] and runtime.predict_sentiment([], []) == {}


def test_provenance_has_the_same_keys_as_the_frozen_stack(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "legacy_rcs")
    legacy = runtime.provenance()
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "local_v2_pilot")
    pilot_p = runtime.provenance()
    assert set(pilot_p) == set(legacy) == {"reason", "source", "model_name", "prompt_version"}
    assert pilot_p["source"] != legacy["source"] and "MARBERTV2" in pilot_p["source"] and "V2_I10" in pilot_p["source"] and "V2_S5" in pilot_p["source"]


# ---------------------------------------------------------------------------------------- frozen stack stays the default
def test_default_mode_is_the_frozen_stack_and_never_touches_v2(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "legacy_rcs")
    monkeypatch.setattr(v2_stack, "get_stack", lambda: (_ for _ in ()).throw(AssertionError("v2 stack must not be used by default")))
    monkeypatch.setattr(frozen_inference, "predict_intent", lambda t: [("SPAM", 0.5)] * len(t))
    monkeypatch.setattr(frozen_inference, "predict_sentiment", lambda ids, t: {i: ("NEUTRAL", 0.5) for i in ids})
    assert runtime.predict_intent(["x"]) == [("SPAM", 0.5)]
    assert runtime.predict_sentiment(["a"], ["x"]) == {"a": ("NEUTRAL", 0.5)}
    assert settings.CLASSIFICATION_MODES[0] == "legacy_rcs"


def test_pilot_keeps_the_frozen_topic_model(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "local_v2_pilot")
    monkeypatch.setattr(frozen_inference, "predict_topic", lambda ids, t: {i: ("GENERAL", 0.4) for i in ids})
    assert runtime.predict_topic(["a"], ["x"]) == {"a": ("GENERAL", 0.4)}


def test_pilot_is_refused_in_production_and_invalid_modes_still_fail_closed(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "local_v2_pilot")
    monkeypatch.setenv("RCS_ENV", "production")
    with pytest.raises(runtime.ClassificationModeError):
        runtime.predict_intent(["x"])
    with pytest.raises(runtime.ClassificationModeError):
        runtime.provenance()
    monkeypatch.delenv("RCS_ENV")
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "unexpected")
    with pytest.raises(runtime.ClassificationModeError):
        runtime.provenance()


# ------------------------------------------------------------------------------------------- Arabic input regression
@needs_models
def test_diacritised_text_is_handled_and_stays_valid(stack):
    plain = "الموقع ما يفتح عندي"
    dia = "الْمَوْقِعُ مَا يَفْتَحُ عِنْدِي"
    out = stack.predict_intent([plain, dia]); s = stack.predict_sentiment(["a", "b"], [plain, dia])
    assert all(l in settings.PRIMARY_INTENTS and np.isfinite(c) for l, c in out) and all(l in settings.SENTIMENTS for l, _ in s.values())


@needs_models
def test_saudi_colloquial_gratitude_and_questions(stack):
    expect = {
        "الله يعطيكم العافية ما قصرتوا": ("THANKS_PRAISE", "POSITIVE"),
        "يعطيكم العافية": ("THANKS_PRAISE", "POSITIVE"),
        "شكرا جزيلا لكم": ("THANKS_PRAISE", "POSITIVE"),
    }
    T = list(expect)
    intents = stack.predict_intent(T); sents = stack.predict_sentiment(_ids(len(T)), T)
    for t, (i, s) in zip(T, zip(intents, sents.values())):
        assert (i[0], s[0]) == expect[t], t              # snapshot of the selected artifacts' behaviour on clear colloquial thanks
    assert stack.predict_intent(["هل يمكن اضافة خاصية البحث؟"])[0][0] == "INQUIRY"


@needs_models
def test_empty_and_malformed_comments_never_crash_and_keep_alignment(stack):
    bad = ["", "   ", None, 123, "؟؟؟", "😀😀", "hello world", "a" * 5000, "\n\n", "مرحبا"]
    out = stack.predict_intent(bad)
    s = stack.predict_sentiment(_ids(len(bad)), bad)
    assert len(out) == len(bad) and len(s) == len(bad)
    assert all(l in settings.PRIMARY_INTENTS and 0.0 <= c <= 1.0 for l, c in out)
    assert all(l in settings.SENTIMENTS for l, _ in s.values())
    assert len({o[0] for o in (out[0], out[1], out[2], out[8])}) == 1                       # blank-like inputs are treated identically (None -> '', whitespace)
    assert max(abs(out[0][1] - out[k][1]) for k in (1, 2, 8)) < 1e-5                      # (batch padding changes features only at ~1e-8)


@needs_models
def test_duplicate_and_repeated_texts_are_consistent(stack):
    T = ["شكرا لكم", "شكرا لكم", "كيف أضيف كلمة؟"]
    a = stack.predict_intent(T)
    b = stack.predict_intent(T)
    assert a[0] == a[1] and [x[0] for x in a] == [x[0] for x in b] and all(abs(x[1] - y[1]) < 1e-5 for x, y in zip(a, b))


def test_blank_rows_are_stopped_upstream_by_the_prefilter():
    from rcs.ingest.cleaning import normalize_text
    from rcs.prefilter.spam_rules import detect_null_or_near_empty

    for blank in (None, "", "   ", "\n\n", "؟"):
        assert detect_null_or_near_empty(normalize_text(blank)) is not None, repr(blank)    # the pipeline normalises BEFORE the prefilter


# ----------------------------------------------------------------------------------------------- shared computation
@needs_models
def test_embeddings_and_camel_outputs_are_shared_between_intent_and_sentiment():
    s = V2Stack(); s.load()
    T = ["شكرا جزيلا لكم", "هل يمكن اضافة خاصية البحث؟", "الموقع ما يفتح"]
    s.predict_intent(T)
    after_intent = dict(s.features.stats)
    s.predict_sentiment(_ids(3), T)
    assert s.features.stats["misses"] == after_intent["misses"] and s.features.stats["e5_batches"] == after_intent["e5_batches"]
    assert s.features.stats["camel_batches"] == after_intent["camel_batches"]
    assert s.features.stats["hits"] >= 3


@needs_models
def test_feature_cache_is_bounded_and_does_not_change_predictions(monkeypatch):
    s = V2Stack(); s.load()
    monkeypatch.setattr(v2_stack, "CACHE_MAX", 2)
    T = ["شكرا لكم", "كيف أضيف كلمة؟", "الموقع ما يفتح", "ممتاز"]      # more distinct texts than the cache holds, in ONE batch
    first = s.predict_intent(T)
    assert len(s.features._cache) <= 2 and len(first) == 4
    again = s.predict_intent(T)
    assert [a[0] for a in again] == [a[0] for a in first] and all(abs(a[1] - b[1]) < 1e-5 for a, b in zip(again, first))


# ----------------------------------------------------------------------------------------------- loading failures
def _copy_artifacts(tmp_path):
    d = tmp_path / "art"
    shutil.copytree(ART, d)
    return d


def _rewrite_manifest(task_dir: Path):
    names = ("config.json", "metrics.json", "model.joblib", "provenance.json", "reproducibility.json")
    (task_dir / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((task_dir / n).read_bytes()).hexdigest()}  {n}\n" for n in names))


@needs_models
def test_missing_artifact_directory_is_an_explicit_error(tmp_path):
    with pytest.raises(V2StackError):
        V2Stack(tmp_path / "nowhere").load_artifacts()


@needs_models
def test_tampered_artifact_is_refused(tmp_path):
    d = _copy_artifacts(tmp_path)
    (d / "intent/config.json").write_text('{"status": "ACTIVE"}')
    with pytest.raises(V2StackError):
        V2Stack(d).load_artifacts()


@needs_models
def test_activated_status_wrong_feature_layout_or_wrong_taxonomy_are_refused(tmp_path):
    d = _copy_artifacts(tmp_path)
    cfg = json.loads((d / "sentiment/config.json").read_text())
    cfg["status"] = "ACTIVE"
    (d / "sentiment/config.json").write_text(json.dumps(cfg)); _rewrite_manifest(d / "sentiment")
    with pytest.raises(V2StackError, match="status"):
        V2Stack(d).load_artifacts()
    d2 = _copy_artifacts(tmp_path / "b")
    cfg = json.loads((d2 / "intent/config.json").read_text()); cfg["extra_features"] = ["e5"]
    (d2 / "intent/config.json").write_text(json.dumps(cfg)); _rewrite_manifest(d2 / "intent")
    with pytest.raises(V2StackError, match="feature layout"):
        V2Stack(d2).load_artifacts()
    d3 = _copy_artifacts(tmp_path / "c")
    wrong = LexicalClassifier(["X", "Y"]).fit(["اختبار", "تجربة"] * 6, ["X", "Y"] * 6)
    joblib.dump(wrong, d3 / "intent/model.joblib"); _rewrite_manifest(d3 / "intent")
    with pytest.raises(V2StackError, match="taxonomy"):
        V2Stack(d3).load_artifacts()


@needs_models
def test_missing_feature_weights_fail_loudly(monkeypatch):
    from rcs.response_intelligence.retrieval import embeddings

    def boom(self):
        raise embeddings.EmbeddingModelUnavailable("x")

    monkeypatch.setattr(embeddings.E5EmbeddingAdapter, "load", boom)
    s = V2Stack()
    with pytest.raises(V2StackError):
        s.predict_intent(["شكرا"])


@needs_models
def test_no_silent_fallback_to_the_frozen_models_when_v2_fails(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "local_v2_pilot")
    broken = V2Stack()
    monkeypatch.setattr(broken.features, "get", lambda texts: (_ for _ in ()).throw(V2StackError("feature extraction failed")))
    v2_stack.reset_for_testing(broken)
    never = lambda *a, **k: (_ for _ in ()).throw(AssertionError("frozen model used as a fallback"))
    monkeypatch.setattr(frozen_inference, "predict_intent", never)
    monkeypatch.setattr(frozen_inference, "predict_sentiment", never)
    try:
        with pytest.raises(V2StackError):
            runtime.predict_intent(["شكرا"])
        with pytest.raises(V2StackError):
            runtime.predict_sentiment(["a"], ["شكرا"])
    finally:
        v2_stack.reset_for_testing(None)


@needs_models
def test_length_mismatch_between_ids_and_texts_is_an_error(stack):
    with pytest.raises(V2StackError):
        stack.predict_sentiment(["a"], ["x", "y"])


# ------------------------------------------------------------------------- existing classification API compatibility
class _Repo:
    def __init__(self, texts):
        self.texts, self.results, self.metadata = texts, None, None

    def load_for_classification(self):
        return pd.DataFrame([{"comment_id": f"c{i}", "original_comment": t} for i, t in enumerate(self.texts)])

    def apply_classification_batch(self, results, **metadata):
        self.results, self.metadata = results, metadata
        return {"classified": len(results), "failed": 0, "skipped_already_classified": 0}


@needs_models
def test_review_flow_runs_on_the_pilot_stack_with_the_unchanged_result_schema(pilot, monkeypatch):
    from rcs.review import excel_review_flow

    texts = ["شكرا جزيلا لكم", "هل يمكن اضافة خاصية البحث؟", "أقترح إضافة كلمة جديدة إلى المعجم"]
    repo = _Repo(texts)
    monkeypatch.setattr(excel_review_flow, "resolve_bundle", lambda _e: SimpleNamespace(review_items=repo))
    monkeypatch.setattr(excel_review_flow, "predict_topic", lambda ids, t: {i: ("GENERAL", 0.5) for i in ids})
    out = excel_review_flow.run_frozen_classification(object(), ["c0", "c1", "c2"])
    assert out["classified"] == 3 and set(out["sentiments"]) == {"c0", "c1", "c2"}
    assert set(repo.results[0]) == {"comment_id", "success", "ai_primary_intent", "ai_topic_subtype", "ai_subcategory", "ai_secondary_intent",
                                    "ai_routing_decision", "ai_classification_reason", "ai_confidence", "ai_classification_source", "ai_needs_review",
                                    "routing_reason_code", "ai_sentiment"}
    for r in repo.results:
        assert r["ai_primary_intent"] in settings.PRIMARY_INTENTS and r["ai_sentiment"] in settings.SENTIMENTS
        assert isinstance(r["ai_confidence"], float) and 0.0 <= r["ai_confidence"] <= 1.0
        assert r["ai_needs_review"] == (r["ai_confidence"] < settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD)       # confidence handling unchanged
        assert r["ai_classification_source"] == "V2_I10_INTENT+FROZEN_MARBERTV2_TOPIC+V2_S5_SENTIMENT"
    assert repo.metadata["prompt_version"] == "classification-v2-local-pilot"
    assert repo.results[0]["ai_primary_intent"] == "THANKS_PRAISE" and repo.results[0]["ai_routing_decision"] == "AUTO_REPLY_CANDIDATE"


@needs_models
def test_review_flow_persists_nothing_when_the_v2_stack_fails(monkeypatch):
    from rcs.review import excel_review_flow

    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "local_v2_pilot")
    broken = V2Stack(Path("/nonexistent"))
    v2_stack.reset_for_testing(broken)
    repo = _Repo(["شكرا"])
    monkeypatch.setattr(excel_review_flow, "resolve_bundle", lambda _e: SimpleNamespace(review_items=repo))
    monkeypatch.setattr(excel_review_flow, "predict_topic", lambda ids, t: {i: ("GENERAL", 0.5) for i in ids})
    try:
        with pytest.raises(V2StackError):
            excel_review_flow.run_frozen_classification(object(), ["c0"])
        assert repo.results is None                                   # no partial / mixed-stack classification was written
    finally:
        v2_stack.reset_for_testing(None)


@needs_models
def test_playground_and_inspector_use_the_same_runtime_functions(pilot):
    from rcs.response_intelligence import inspector

    assert inspector.classification_runtime is runtime
