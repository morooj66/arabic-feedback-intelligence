"""The Render blueprint and the settings it relies on: required variables present, secrets never carry a value, uploads on the persistent disk, no demo flags."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVICE = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
ENV = {e["key"]: e for e in SERVICE["envVars"]}
SECRETS = ("DATABASE_URL", "GEMINI_API_KEY", "HF_TOKEN", "ALLOWED_ORIGINS", "RCS_TRUSTED_PROXIES")


def test_required_production_variables_are_declared():
    for key in ("RCS_ENV", "DATABASE_URL", "PERSISTENCE_BACKEND", "GEMINI_API_KEY", "ALLOWED_ORIGINS", "HF_MODEL_REPO_ID", "HF_MODEL_REVISION", "HF_TOKEN", "RCS_UPLOAD_DIR"):
        assert key in ENV, key
    assert ENV["RCS_ENV"]["value"] == "production" and ENV["PERSISTENCE_BACKEND"]["value"] == "postgres"


def test_secrets_are_prompted_for_never_stored_in_the_blueprint():
    for key in SECRETS:
        assert ENV[key].get("sync") is False and "value" not in ENV[key], key
    text = (ROOT / "render.yaml").read_text()
    import re

    assert not re.search(r"://[^\s/@:]+:[^\s@]+@", text)              # no connection string with embedded credentials
    assert "mongodb" not in text.lower()


def test_demo_mode_cannot_be_enabled_from_the_production_blueprint():
    assert not [k for k in ENV if k.startswith("RCS_DEMO_")]


def test_uploads_live_on_the_persistent_disk_and_the_model_revision_is_pinned():
    disk = SERVICE["disk"]
    assert ENV["RCS_UPLOAD_DIR"]["value"].startswith(disk["mountPath"].rstrip("/") + "/")
    revision = ENV["HF_MODEL_REVISION"]["value"]
    assert revision != "main" and (len(revision) == 40 or revision == "<pinned-commit-sha>")      # the public copy of the blueprint carries a documented placeholder
    assert SERVICE["autoDeploy"] is False and "fetch_frozen_models.py" in SERVICE["buildCommand"]


def test_the_upload_directory_setting_honours_the_environment():
    code = "import sys; sys.path[:0]=['src','.']; from config import settings; print(settings.EXCEL_UPLOAD_DIR)"
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "RCS_UPLOAD_DIR": "/var/data/uploads"}
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip() == "/var/data/uploads"
    env.pop("RCS_UPLOAD_DIR")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip().endswith("data/uploads")
