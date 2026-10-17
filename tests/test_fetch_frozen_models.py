"""Phase 7 — scripts/fetch_frozen_models.py: reconstructs the frozen-model
deployment layout from a private artifact source, verifying every file
against config/frozen_models_manifest.json's checksums before placing it.
Uses small fake files and a fake manifest throughout — never touches the
real 621MB checkpoint or the real manifest."""

from __future__ import annotations

import hashlib
import importlib
import json

import pytest

fetch_frozen_models = importlib.import_module("scripts.fetch_frozen_models")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    """Points the module at an isolated project root + manifest, so placed
    files land under tmp_path instead of the real repo."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    manifest_path = project_root / "config" / "frozen_models_manifest.json"
    manifest_path.parent.mkdir(parents=True)

    payload_a = b"fake topic checkpoint bytes"
    payload_b = b"fake intent joblib bytes"
    manifest = {
        "default_revision": None,
        "files": [
            {
                "repo_path": "topic_marbertv2/config.json",
                "local_dest": "data/models/final_test/checkpoints/marbertv2_topic/best/config.json",
                "sha256": _sha256_bytes(payload_a),
                "size_bytes": len(payload_a),
            },
            {
                "repo_path": "intent_tfidf_logistic.joblib",
                "local_dest": "data/models/final_test/intent_tfidf_logistic.joblib",
                "sha256": _sha256_bytes(payload_b),
                "size_bytes": len(payload_b),
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(fetch_frozen_models, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(fetch_frozen_models, "_MANIFEST_PATH", manifest_path)

    return {
        "root": project_root,
        "manifest": manifest,
        "payload_a": payload_a,
        "payload_b": payload_b,
    }


def _write_source(tmp_path, manifest, overrides: dict[str, bytes] | None = None, omit: set[str] | None = None):
    overrides = overrides or {}
    omit = omit or set()
    source_dir = tmp_path / "source"
    for entry in manifest["files"]:
        repo_path = entry["repo_path"]
        if repo_path in omit:
            continue
        dest = source_dir / repo_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = overrides.get(repo_path)
        if content is None:
            # Reconstruct the correct bytes from the fixture by matching size — the
            # fixture always supplies the real payload via `overrides` when needed;
            # tests that want valid content pass it explicitly.
            raise KeyError(f"no content provided for {repo_path}")
        dest.write_bytes(content)
    return source_dir


def test_offline_fetch_places_files_and_verifies_checksums(fake_project, tmp_path, monkeypatch):
    manifest = fake_project["manifest"]
    source_dir = _write_source(
        tmp_path,
        manifest,
        overrides={
            "topic_marbertv2/config.json": fake_project["payload_a"],
            "intent_tfidf_logistic.joblib": fake_project["payload_b"],
        },
    )
    monkeypatch.setenv("RCS_FROZEN_MODELS_SOURCE_DIR", str(source_dir))

    fetch_frozen_models.fetch_frozen_models()

    root = fake_project["root"]
    placed_a = root / "data/models/final_test/checkpoints/marbertv2_topic/best/config.json"
    placed_b = root / "data/models/final_test/intent_tfidf_logistic.joblib"
    assert placed_a.read_bytes() == fake_project["payload_a"]
    assert placed_b.read_bytes() == fake_project["payload_b"]


def test_second_run_is_a_verified_noop_and_does_not_touch_the_source(fake_project, tmp_path, monkeypatch):
    manifest = fake_project["manifest"]
    source_dir = _write_source(
        tmp_path,
        manifest,
        overrides={
            "topic_marbertv2/config.json": fake_project["payload_a"],
            "intent_tfidf_logistic.joblib": fake_project["payload_b"],
        },
    )
    monkeypatch.setenv("RCS_FROZEN_MODELS_SOURCE_DIR", str(source_dir))
    fetch_frozen_models.fetch_frozen_models()

    # Delete the source entirely — a second run must not need it, because
    # _already_satisfied() should short-circuit before ever reading it.
    for f in source_dir.rglob("*"):
        if f.is_file():
            f.unlink()

    fetch_frozen_models.fetch_frozen_models()  # must not raise


def test_checksum_mismatch_is_rejected_and_never_placed(fake_project, tmp_path, monkeypatch):
    manifest = fake_project["manifest"]
    source_dir = _write_source(
        tmp_path,
        manifest,
        overrides={
            "topic_marbertv2/config.json": b"CORRUPTED, WRONG BYTES",
            "intent_tfidf_logistic.joblib": fake_project["payload_b"],
        },
    )
    monkeypatch.setenv("RCS_FROZEN_MODELS_SOURCE_DIR", str(source_dir))

    with pytest.raises(fetch_frozen_models.FrozenModelArtifactError, match="[Ss]ize mismatch"):
        fetch_frozen_models.fetch_frozen_models()

    placed_a = fake_project["root"] / "data/models/final_test/checkpoints/marbertv2_topic/best/config.json"
    assert not placed_a.exists()


def test_missing_file_in_source_is_rejected_with_a_clear_message(fake_project, tmp_path, monkeypatch):
    manifest = fake_project["manifest"]
    source_dir = _write_source(
        tmp_path,
        manifest,
        overrides={"topic_marbertv2/config.json": fake_project["payload_a"]},
        omit={"intent_tfidf_logistic.joblib"},
    )
    monkeypatch.setenv("RCS_FROZEN_MODELS_SOURCE_DIR", str(source_dir))

    with pytest.raises(fetch_frozen_models.FrozenModelArtifactError, match="intent_tfidf_logistic.joblib"):
        fetch_frozen_models.fetch_frozen_models()


def test_online_mode_without_repo_id_fails_clearly_before_any_network_call(fake_project, monkeypatch):
    monkeypatch.delenv("RCS_FROZEN_MODELS_SOURCE_DIR", raising=False)
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(fetch_frozen_models.FrozenModelArtifactError, match="HF_MODEL_REPO_ID"):
        fetch_frozen_models.fetch_frozen_models()


def test_online_mode_without_token_fails_clearly(fake_project, monkeypatch):
    monkeypatch.delenv("RCS_FROZEN_MODELS_SOURCE_DIR", raising=False)
    monkeypatch.setenv("HF_MODEL_REPO_ID", "someuser/rcs-frozen-models")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(fetch_frozen_models.FrozenModelArtifactError, match="HF_TOKEN"):
        fetch_frozen_models.fetch_frozen_models()


def test_online_mode_without_a_pinned_revision_refuses_to_float_on_main(fake_project, monkeypatch):
    monkeypatch.delenv("RCS_FROZEN_MODELS_SOURCE_DIR", raising=False)
    monkeypatch.setenv("HF_MODEL_REPO_ID", "someuser/rcs-frozen-models")
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test-only")
    monkeypatch.delenv("HF_MODEL_REVISION", raising=False)
    assert fake_project["manifest"]["default_revision"] is None

    with pytest.raises(fetch_frozen_models.FrozenModelArtifactError, match="pinned revision"):
        fetch_frozen_models.fetch_frozen_models()


@pytest.mark.parametrize("revision", ["main", "v1.0", "abc123", "A" * 40, "g" * 40])
def test_private_model_revision_must_be_a_full_lowercase_commit_sha(fake_project, monkeypatch, revision):
    monkeypatch.delenv("RCS_FROZEN_MODELS_SOURCE_DIR", raising=False)
    monkeypatch.setenv("HF_MODEL_REPO_ID", "someuser/rcs-frozen-models")
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test-only")
    monkeypatch.setenv("HF_MODEL_REVISION", revision)

    with pytest.raises(fetch_frozen_models.FrozenModelArtifactError, match="40-character lowercase commit SHA"):
        fetch_frozen_models.fetch_frozen_models()
