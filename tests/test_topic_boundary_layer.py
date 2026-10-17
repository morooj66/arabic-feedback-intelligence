"""Opt-in WS/EC boundary layer: pure behaviour, default-OFF guarantee, and the pipeline integration."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from config import settings
from rcs.classification import runtime, topic_boundary_layer as T
from rcs.review import excel_review_flow


def test_layer_is_off_by_default_in_settings():
    # the project default (no env var) must be disabled; a developer's shell must not silently change CI behaviour
    if "RCS_TOPIC_BOUNDARY_LAYER" in os.environ:
        pytest.skip("RCS_TOPIC_BOUNDARY_LAYER is set in this shell")
    assert settings.TOPIC_BOUNDARY_LAYER_ENABLED is False


@pytest.mark.parametrize("base,text,lemma,expected", [
    ("WORD_SUBMISSION", "كلمة ومعناها شرح", False, "WORD_SUBMISSION"),
    ("ENTRY_CORRECTION", "كلمة ومعناها شرح", False, "WORD_SUBMISSION"),
    ("WORD_SUBMISSION", "التعريف غير صحيح", False, "ENTRY_CORRECTION"),
    ("WORD_SUBMISSION", "كلمة ومعناها", True, "ENTRY_CORRECTION"),
    ("GENERAL", "هذا خطأ", True, "GENERAL"),
    ("TECHNICAL_ISSUE", "الموقع لا يعمل خطأ", False, "TECHNICAL_ISSUE"),
    ("DICTIONARY_SUGGESTION", "أتمنى خاصية", True, "DICTIONARY_SUGGESTION"),
])
def test_boundary_topic_follows_the_frozen_spec(base, text, lemma, expected):
    assert T.boundary_topic(base, text, lemma) == expected


@pytest.mark.parametrize("value,expected", [(None, False), ("", False), ("nan", False), (float("nan"), False), ("None", False),
                                            ("  ", False), ("جَنَى", True)])
def test_lemma_present_handles_every_missing_representation(value, expected):
    assert T.lemma_present(value) is expected


def test_apply_boundary_layer_keeps_confidences_and_other_classes():
    preds = {"a": ("WORD_SUBMISSION", 0.7), "b": ("ENTRY_CORRECTION", 0.6), "c": ("GENERAL", 0.9)}
    out = T.apply_boundary_layer(preds, ["a", "b", "c"], ["كلمة", "كلمة", "هذا خطأ"], [True, None, True])
    assert out == {"a": ("ENTRY_CORRECTION", 0.7), "b": ("WORD_SUBMISSION", 0.6), "c": ("GENERAL", 0.9)}
    assert preds["a"] == ("WORD_SUBMISSION", 0.7)             # the input mapping is not mutated


def test_runtime_returns_the_same_object_when_disabled_and_changes_only_ws_ec_when_enabled(monkeypatch):
    preds = {"a": ("WORD_SUBMISSION", 0.7), "c": ("GENERAL", 0.9)}
    monkeypatch.setattr(settings, "TOPIC_BOUNDARY_LAYER_ENABLED", False)
    same, applied = runtime.apply_optional_topic_layer(preds, ["a", "c"], ["كلمة", "x"], [True, True])
    assert same is preds and applied is False
    monkeypatch.setattr(settings, "TOPIC_BOUNDARY_LAYER_ENABLED", True)
    out, applied = runtime.apply_optional_topic_layer(preds, ["a", "c"], ["كلمة", "x"], [True, True])
    assert applied is True and out == {"a": ("ENTRY_CORRECTION", 0.7), "c": ("GENERAL", 0.9)}
    prov = runtime.with_layer_provenance(runtime.provenance(), True)
    assert prov["source"].endswith(T.PROVENANCE_SUFFIX) and runtime.with_layer_provenance(runtime.provenance(), False) == runtime.provenance()


def _run_pipeline(monkeypatch, enabled, with_lemma_column=True):
    class Repo:
        def __init__(self):
            self.results, self.metadata = None, None

        def load_for_classification(self):
            row = {"comment_id": "c1", "original_comment": "كلمة ومعناها شرح"}
            if with_lemma_column:
                row["lemma"] = "جَنَى"
            return pd.DataFrame([row])

        def apply_classification_batch(self, results, **metadata):
            self.results, self.metadata = results, metadata
            return {"classified": 1, "failed": 0, "skipped_already_classified": 0}

    repo = Repo()
    monkeypatch.setattr(settings, "TOPIC_BOUNDARY_LAYER_ENABLED", enabled)
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "legacy_rcs")
    monkeypatch.setattr(excel_review_flow, "resolve_bundle", lambda _e: SimpleNamespace(review_items=repo))
    monkeypatch.setattr(excel_review_flow, "predict_intent", lambda texts: [("SUGGESTION", 0.8)])
    monkeypatch.setattr(excel_review_flow, "predict_topic", lambda ids, texts: {ids[0]: ("WORD_SUBMISSION", 0.6)})
    monkeypatch.setattr(excel_review_flow, "predict_sentiment", lambda ids, texts: {ids[0]: ("NEUTRAL", 0.7)})
    excel_review_flow.run_frozen_classification(object(), ["c1"])
    return repo


def test_pipeline_default_output_is_unchanged_when_the_flag_is_off(monkeypatch):
    repo = _run_pipeline(monkeypatch, enabled=False)
    assert repo.results[0]["ai_topic_subtype"] == "WORD_SUBMISSION"
    assert T.PROVENANCE_SUFFIX not in repo.results[0]["ai_classification_source"]
    assert T.PROVENANCE_SUFFIX not in repo.metadata["model_name"]


def test_pipeline_with_the_flag_on_changes_only_the_topic_and_records_provenance(monkeypatch):
    off = _run_pipeline(monkeypatch, enabled=False)
    on = _run_pipeline(monkeypatch, enabled=True)
    assert on.results[0]["ai_topic_subtype"] == "ENTRY_CORRECTION"          # lemma present -> ENTRY_CORRECTION
    assert T.PROVENANCE_SUFFIX in on.results[0]["ai_classification_source"] and T.PROVENANCE_SUFFIX in on.metadata["model_name"]
    for k in ("ai_primary_intent", "ai_confidence", "ai_needs_review"):
        assert on.results[0][k] == off.results[0][k]
    assert on.results[0]["ai_sentiment"] == off.results[0]["ai_sentiment"]


def test_pipeline_tolerates_a_missing_lemma_column(monkeypatch):
    repo = _run_pipeline(monkeypatch, enabled=True, with_lemma_column=False)
    assert repo.results[0]["ai_topic_subtype"] == "WORD_SUBMISSION"          # no lemma, no error claim
