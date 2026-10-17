"""Production path controls for the unchanged frozen classification stack."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rcs.classification import frozen_inference
from rcs.training import hf_loading


def test_default_frozen_paths_remain_the_existing_project_artifacts(monkeypatch):
    for name in (
        "RCS_FROZEN_SELECTION_PATH",
        "RCS_FROZEN_INTENT_PATH",
        "RCS_MARBERTV2_TOPIC_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert frozen_inference.frozen_selection_path().name == "FROZEN_MODEL_SELECTION.json"
    assert frozen_inference.frozen_intent_path().name == "intent_tfidf_logistic.joblib"
    assert frozen_inference.frozen_topic_path().name == "best"


def test_all_private_frozen_paths_are_independently_configurable(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    intent = tmp_path / "intent.joblib"
    topic = tmp_path / "topic"
    monkeypatch.setenv("RCS_FROZEN_SELECTION_PATH", str(selection))
    monkeypatch.setenv("RCS_FROZEN_INTENT_PATH", str(intent))
    monkeypatch.setenv("RCS_MARBERTV2_TOPIC_CHECKPOINT", str(topic))
    assert frozen_inference.frozen_selection_path() == selection
    assert frozen_inference.frozen_intent_path() == intent
    assert frozen_inference.frozen_topic_path() == topic


@pytest.mark.parametrize(
    "name",
    ["RCS_FROZEN_SELECTION_PATH", "RCS_FROZEN_INTENT_PATH", "RCS_MARBERTV2_TOPIC_CHECKPOINT"],
)
def test_relative_private_override_is_rejected_in_production(name, monkeypatch):
    monkeypatch.setenv("RCS_ENV", "production")
    monkeypatch.setenv(name, "relative/path")
    accessor = {
        "RCS_FROZEN_SELECTION_PATH": frozen_inference.frozen_selection_path,
        "RCS_FROZEN_INTENT_PATH": frozen_inference.frozen_intent_path,
        "RCS_MARBERTV2_TOPIC_CHECKPOINT": frozen_inference.frozen_topic_path,
    }[name]
    with pytest.raises(ValueError, match="absolute path"):
        accessor()


def test_selection_override_is_actually_read(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"sentiment": {"model_id": "synthetic"}}))
    monkeypatch.setenv("RCS_FROZEN_SELECTION_PATH", str(selection))
    monkeypatch.setattr(frozen_inference, "_freeze", None)
    monkeypatch.setattr(frozen_inference, "_freeze_source", None)
    assert frozen_inference._load_freeze()["sentiment"]["model_id"] == "synthetic"


def test_camelbert_override_preserves_manifest_validation(tmp_path, monkeypatch):
    entry = {
        "repo_id": "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
        "revision": "a" * 40,
        "local_dir": "unused",
        "files": [{"name": "config.json", "size_bytes": 2}],
    }
    target = tmp_path / "camel"
    target.mkdir()
    (target / ".revision").write_text("a" * 40)
    (target / "config.json").write_text("{}")
    monkeypatch.setenv("RCS_ENV", "production")
    monkeypatch.setenv("RCS_CAMELBERT_SENTIMENT_PATH", str(target))
    monkeypatch.setattr(hf_loading, "_pinned_entry", lambda _model_id: entry)
    assert hf_loading.resolve_model_source(entry["repo_id"]) == (str(target), {})
    (target / ".revision").write_text("b" * 40)
    with pytest.raises(hf_loading.PinnedModelUnavailable):
        hf_loading.resolve_model_source(entry["repo_id"])


def test_relative_camelbert_override_is_rejected_in_production(monkeypatch):
    monkeypatch.setenv("RCS_ENV", "production")
    monkeypatch.setenv("RCS_CAMELBERT_SENTIMENT_PATH", "relative/camel")
    entry = {
        "repo_id": "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
        "revision": "a" * 40,
        "local_dir": "unused",
        "files": [],
    }
    monkeypatch.setattr(hf_loading, "_pinned_entry", lambda _model_id: entry)
    with pytest.raises(hf_loading.PinnedModelUnavailable, match="absolute path"):
        hf_loading.resolve_model_source(entry["repo_id"])


def test_large_transformer_inference_is_serialized_across_concurrent_requests(tmp_path, monkeypatch):
    checkpoint = tmp_path / "topic"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"synthetic")
    monkeypatch.setenv("RCS_MARBERTV2_TOPIC_CHECKPOINT", str(checkpoint))

    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    class FakeTopicModule:
        @staticmethod
        def evaluate_frozen_topic(examples, _checkpoint):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1
            return {
                "comment_ids": [ex.comment_id for ex in examples],
                "y_pred": ["GENERAL" for _ in examples],
                "probabilities": [[1.0] for _ in examples],
            }

    monkeypatch.setattr(frozen_inference, "_load_human_test_module", lambda: FakeTopicModule())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: frozen_inference.predict_topic([f"c{i}"], ["تعليق اصطناعي"]), range(2)))
    assert state["max_active"] == 1
    assert all(next(iter(result.values()))[0] == "GENERAL" for result in results)
