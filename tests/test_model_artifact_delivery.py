"""Reproducible delivery of the pinned public sentiment model (CAMeLBERT): manifest integrity, fetch script behaviour, and the loader's fail-closed rules.
Everything runs on small fake files - no model is downloaded and no network is used."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

from rcs.training import hf_loading

fetch = importlib.import_module("scripts.fetch_frozen_models")
ROOT = Path(__file__).resolve().parents[1]
REPO = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
TESTED_REVISION = "b0df630d23e5aa5f8a326017605337f6d0863ecd"            # refs/main of the cache of the system that was actually tested
TESTED_WEIGHTS_SHA = "3d084c6021a61a2e8dfe496ea4e0c60580b3f02ffbea214039dd17676cb2dd32"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- the real manifest
def _real_entry():
    manifest = json.loads((ROOT / "config/frozen_models_manifest.json").read_text())
    return next(m for m in manifest["public_models"] if m["repo_id"] == REPO), manifest


def test_manifest_pins_the_tested_camelbert_revision_and_every_file():
    entry, manifest = _real_entry()
    assert entry["revision"] == TESTED_REVISION and re.fullmatch(r"[0-9a-f]{40}", entry["revision"])
    names = {f["name"]: f for f in entry["files"]}
    assert set(names) == {"config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"}
    assert names["model.safetensors"]["sha256"] == TESTED_WEIGHTS_SHA
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size_bytes"] > 0 for f in entry["files"])
    assert entry["local_dir"].startswith("data/models/public/")
    # the pre-existing private-artifact entries are untouched
    assert manifest["default_revision"] == "43f6b1a8992f96b49c8775b60cfe4be0bde06dc5" and len(manifest["files"]) == 7


def test_the_real_manifest_passes_the_scripts_own_validation():
    assert fetch._public_models(json.loads((ROOT / "config/frozen_models_manifest.json").read_text()))


# ---------------------------------------------------------------- fetch script
FILES = {"config.json": b'{"id2label": {"0": "positive"}}', "model.safetensors": b"fake-weights" * 10, "vocab.txt": b"a\nb\n"}


@pytest.fixture()
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    entry = {"repo_id": "Org/Model", "revision": "a" * 40, "local_dir": "data/models/public/model",
             "files": [{"name": n, "size_bytes": len(b), "sha256": _sha(b)} for n, b in FILES.items()]}
    manifest = {"default_revision": None, "files": [], "public_models": [entry]}
    (root / "config" / "frozen_models_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(fetch, "_PROJECT_ROOT", root)
    monkeypatch.setattr(fetch, "_MANIFEST_PATH", root / "config" / "frozen_models_manifest.json")
    src = tmp_path / "source" / "public" / "Org__Model"
    src.mkdir(parents=True)
    for n, b in FILES.items():
        (src / n).write_bytes(b)
    monkeypatch.setenv("RCS_FROZEN_MODELS_SOURCE_DIR", str(tmp_path / "source"))
    return {"root": root, "entry": entry, "src": src, "dest": root / entry["local_dir"], "manifest_path": root / "config" / "frozen_models_manifest.json"}


def test_offline_fetch_places_verified_files_and_records_the_revision(project):
    fetch.fetch_frozen_models()
    for n, b in FILES.items():
        assert (project["dest"] / n).read_bytes() == b
    assert (project["dest"] / ".revision").read_text().strip() == "a" * 40
    assert not project["dest"].with_name("model.partial").exists()


def test_second_run_is_a_verified_noop(project, capsys, monkeypatch):
    fetch.fetch_frozen_models(); capsys.readouterr()
    monkeypatch.delenv("RCS_FROZEN_MODELS_SOURCE_DIR")                                   # a second run must not need (or contact) any source
    fetch.fetch_frozen_models()
    assert "nothing to do" in capsys.readouterr().out


def test_a_modified_file_is_detected_and_refetched(project):
    fetch.fetch_frozen_models()
    (project["dest"] / "vocab.txt").write_bytes(b"tampered")
    assert fetch._public_satisfied(project["entry"]) is False
    fetch.fetch_frozen_models()
    assert (project["dest"] / "vocab.txt").read_bytes() == FILES["vocab.txt"]


def test_checksum_mismatch_is_rejected_and_nothing_is_placed(project):
    (project["src"] / "model.safetensors").write_bytes(b"X" * len(FILES["model.safetensors"]))      # right size, wrong bytes
    with pytest.raises(fetch.FrozenModelArtifactError, match="Checksum mismatch"):
        fetch.fetch_frozen_models()
    assert not project["dest"].exists()


def test_size_mismatch_and_missing_file_are_rejected(project):
    (project["src"] / "vocab.txt").write_bytes(b"too long to match")
    with pytest.raises(fetch.FrozenModelArtifactError, match="Size mismatch"):
        fetch.fetch_frozen_models()
    (project["src"] / "vocab.txt").unlink()
    with pytest.raises(fetch.FrozenModelArtifactError, match="not found"):
        fetch.fetch_frozen_models()
    assert not project["dest"].exists()


@pytest.mark.parametrize("bad", ["main", "v1.0", "abc123", "A" * 40, "g" * 40, "", None])
def test_a_floating_or_malformed_revision_is_refused(project, bad):
    m = json.loads(project["manifest_path"].read_text()); m["public_models"][0]["revision"] = bad
    project["manifest_path"].write_text(json.dumps(m))
    with pytest.raises(fetch.FrozenModelArtifactError, match="40-character commit SHA"):
        fetch.fetch_frozen_models()


def test_a_stale_partial_directory_never_reaches_the_loader(project):
    stale = project["dest"].with_name("model.partial"); stale.mkdir(parents=True); (stale / "junk").write_text("x")
    fetch.fetch_frozen_models()
    assert not stale.exists() and (project["dest"] / "config.json").exists()


def test_online_mode_requests_exactly_the_pinned_revision_and_files_without_a_token(project, monkeypatch):
    monkeypatch.delenv("RCS_FROZEN_MODELS_SOURCE_DIR")
    monkeypatch.setenv("HF_TOKEN", "must-not-be-sent-to-a-public-repo")
    calls = []

    def fake_snapshot_download(**kw):
        calls.append(kw)
        for n, b in FILES.items():
            (Path(kw["local_dir"]) / n).write_bytes(b)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    fetch.fetch_frozen_models()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "Org/Model" and calls[0]["revision"] == "a" * 40
    assert sorted(calls[0]["allow_patterns"]) == sorted(FILES)
    assert "token" not in calls[0]                                                       # a public model never needs (or leaks) the private-repo token
    assert (project["dest"] / ".revision").exists()


def test_online_download_that_does_not_match_is_rejected(project, monkeypatch):
    monkeypatch.delenv("RCS_FROZEN_MODELS_SOURCE_DIR")

    def corrupt(**kw):
        for n, b in FILES.items():
            (Path(kw["local_dir"]) / n).write_bytes(b[::-1])

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", corrupt)
    with pytest.raises(fetch.FrozenModelArtifactError, match="(Checksum|Size) mismatch|mismatch"):
        fetch.fetch_frozen_models()
    assert not project["dest"].exists()


# ---------------------------------------------------------------- loader
@pytest.fixture()
def loader(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    entry = {"repo_id": "Org/Model", "revision": "b" * 40, "local_dir": "data/models/public/model",
             "files": [{"name": n, "size_bytes": len(b), "sha256": _sha(b)} for n, b in FILES.items()]}
    (root / "config" / "frozen_models_manifest.json").write_text(json.dumps({"files": [], "public_models": [entry]}))
    monkeypatch.setattr(hf_loading, "_PROJECT_ROOT", root)
    monkeypatch.setattr(hf_loading, "_MANIFEST_PATH", root / "config" / "frozen_models_manifest.json")
    monkeypatch.delenv("RCS_ENV", raising=False)
    local = root / entry["local_dir"]

    def materialise(revision="b" * 40, drop=None):
        local.mkdir(parents=True, exist_ok=True)
        for n, b in FILES.items():
            if n != drop:
                (local / n).write_bytes(b)
        (local / ".revision").write_text(revision + "\n")
    return types.SimpleNamespace(local=local, materialise=materialise, root=root)


def test_complete_local_copy_is_loaded_from_disk_without_touching_the_hub(loader):
    loader.materialise()
    assert hf_loading.resolve_model_source("Org/Model") == (str(loader.local), {})


def test_dev_without_a_local_copy_uses_the_hub_at_the_pinned_revision_only(loader):
    assert hf_loading.resolve_model_source("Org/Model") == ("Org/Model", {"revision": "b" * 40})


@pytest.mark.parametrize("variant", ["missing", "wrong_revision", "missing_file"])
def test_production_fails_closed_when_the_pinned_model_is_not_properly_present(loader, monkeypatch, variant):
    monkeypatch.setenv("RCS_ENV", "production")
    if variant == "wrong_revision":
        loader.materialise(revision="c" * 40)
    elif variant == "missing_file":
        loader.materialise(drop="vocab.txt")
    with pytest.raises(hf_loading.PinnedModelUnavailable, match="fetch_frozen_models"):
        hf_loading.resolve_model_source("Org/Model")


def test_production_with_a_complete_local_copy_works(loader, monkeypatch):
    monkeypatch.setenv("RCS_ENV", "production")
    loader.materialise()
    assert hf_loading.resolve_model_source("Org/Model")[0] == str(loader.local)


def test_a_different_model_is_never_substituted(loader, monkeypatch):
    for env in (None, "production"):
        if env:
            monkeypatch.setenv("RCS_ENV", env)
        loader.materialise()
        source, _extra = hf_loading.resolve_model_source("Org/Model")
        assert source in (str(loader.local), "Org/Model")
    assert hf_loading.resolve_model_source("Someone/Else") == ("Someone/Else", {})        # models the manifest does not pin behave exactly as before


def test_loaders_pass_the_resolved_source_and_let_callers_override_kwargs(loader, monkeypatch):
    seen = {}

    class Fake:
        @staticmethod
        def from_pretrained(source, **kw):
            seen["source"], seen["kw"] = source, kw
            return object()

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", Fake)
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", Fake)
    hf_loading.load_tokenizer("Org/Model")
    assert seen == {"source": "Org/Model", "kw": {"revision": "b" * 40}}
    hf_loading.load_sequence_classifier("Org/Model")
    assert seen["kw"] == {"revision": "b" * 40, "use_safetensors": True}
    loader.materialise()
    hf_loading.load_sequence_classifier("Org/Model")
    assert seen["source"] == str(loader.local) and seen["kw"] == {"use_safetensors": True}


def test_the_real_dev_path_resolves_the_tested_revision(monkeypatch):
    monkeypatch.delenv("RCS_ENV", raising=False)
    source, extra = hf_loading.resolve_model_source(REPO)
    assert source in (REPO, str(ROOT / "data/models/public/camelbert-da-sentiment")) and extra in ({"revision": TESTED_REVISION}, {})


def test_every_consumer_of_the_sentiment_model_goes_through_the_pinned_loader():
    for rel in ("src/rcs/training/sentiment_pretrained.py", "src/rcs/classification/v2_stack.py"):
        text = (ROOT / rel).read_text()
        assert "from_pretrained" not in text and "hf_loading" in text, rel
