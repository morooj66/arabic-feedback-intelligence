"""Fetch the frozen, fine-tuned MARBERTv2 topic checkpoint, the fitted
intent classifier, and the frozen model-selection metadata from a private
Hugging Face Hub model repo, verify their integrity, and place them at the
exact local paths rcs.classification.frozen_inference already expects.

This is a deployment build step, not part of the application. It never
retrains, quantizes, or otherwise modifies any model — it only moves
already-frozen bytes from private storage onto the local filesystem and
checks them against known-good checksums recorded in
config/frozen_models_manifest.json (a non-secret file: paths and sha256
hashes only, no visitor data, no credentials).

Configuration (environment variables):
  HF_MODEL_REPO_ID    Required (unless RCS_FROZEN_MODELS_SOURCE_DIR is set).
                       The private HF model repo, e.g. "yourname/rcs-frozen-models".
  HF_MODEL_REVISION    A pinned commit SHA (or tag) to fetch — never "main".
                       Falls back to manifest["default_revision"] if unset;
                       fails if neither is present, so a deploy can never
                       silently float onto whatever the repo's HEAD is.
  HF_TOKEN             Read-only token with access to the private repo.
                       Never printed, never logged, never committed.
  RCS_FROZEN_MODELS_SOURCE_DIR
                       Local/offline test mode. When set, files are copied
                       from this directory (which must mirror the repo's
                       internal repo_path layout) instead of contacting the
                       Hugging Face Hub at all. Used by
                       tests/test_fetch_frozen_models.py and by the manual
                       verification run in Phase 7 to prove this script
                       reconstructs the deployment layout without needing
                       real Hub access.

Exit code is non-zero with a clear, specific message on any failure
(missing token/repo/revision, a file absent from the snapshot, or a
checksum mismatch indicating a corrupt or unexpected download) — this
script is meant to run during the Render build step, so a failure here
must fail the build rather than let a broken deployment start.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _PROJECT_ROOT / "config" / "frozen_models_manifest.json"


class FrozenModelArtifactError(RuntimeError):
    """Raised for any condition that must fail the build, not be papered over."""


def _load_manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        raise FrozenModelArtifactError(f"Manifest not found at {_MANIFEST_PATH}.")
    return json.loads(_MANIFEST_PATH.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _already_satisfied(manifest: dict) -> bool:
    """True if every local_dest already exists and matches its checksum —
    lets local re-runs and repeated `render build` retries skip work."""
    for entry in manifest["files"]:
        dest = _PROJECT_ROOT / entry["local_dest"]
        if not dest.exists() or dest.stat().st_size != entry["size_bytes"]:
            return False
        if _sha256(dest) != entry["sha256"]:
            return False
    return True


def _resolve_revision(manifest: dict) -> str:
    revision = os.environ.get("HF_MODEL_REVISION") or manifest.get("default_revision")
    if not revision:
        raise FrozenModelArtifactError(
            "No pinned revision available: set HF_MODEL_REVISION, or record a "
            "default_revision in config/frozen_models_manifest.json. Refusing to "
            "fetch from a floating branch (e.g. 'main') for a production build."
        )
    if not __import__("re").fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise FrozenModelArtifactError(
            "HF_MODEL_REVISION/default_revision must be a full 40-character lowercase commit SHA; "
            "refusing a floating branch, tag, or abbreviated revision."
        )
    return revision


def _download_snapshot(manifest: dict) -> Path:
    """Downloads only the manifest's required files, at the pinned revision,
    into the local HF cache. Returns the local snapshot directory."""
    repo_id = os.environ.get("HF_MODEL_REPO_ID")
    if not repo_id:
        raise FrozenModelArtifactError(
            "HF_MODEL_REPO_ID is not set. It must name the private Hugging Face "
            "model repo holding the frozen artifacts (e.g. 'yourname/rcs-frozen-models')."
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise FrozenModelArtifactError(
            "HF_TOKEN is not set. A read-only token with access to the private "
            f"repo '{repo_id}' is required — set it as a Render environment "
            "variable (or in your local shell for a manual run), never in a file."
        )
    revision = _resolve_revision(manifest)
    print(f"fetch_frozen_models: downloading from {repo_id} @ {revision}")
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    allow_patterns = [entry["repo_path"] for entry in manifest["files"]]
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            repo_type="model",
            allow_patterns=allow_patterns,
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            raise FrozenModelArtifactError(
                f"Access denied fetching '{repo_id}' at revision '{revision}'. "
                "Check that HF_TOKEN is valid and has read access to this "
                "private repo."
            ) from exc
        if status == 404:
            raise FrozenModelArtifactError(
                f"Repo or revision not found: '{repo_id}' @ '{revision}'. Check "
                "HF_MODEL_REPO_ID and HF_MODEL_REVISION are correct."
            ) from exc
        raise FrozenModelArtifactError(
            f"Failed to fetch '{repo_id}' @ '{revision}' from Hugging Face Hub: {exc}"
        ) from exc
    return Path(snapshot_dir)


def _place_files(manifest: dict, source_dir: Path, *, offline: bool) -> None:
    for entry in manifest["files"]:
        source = source_dir / entry["repo_path"]
        if not source.exists():
            origin = "local source directory" if offline else "downloaded snapshot"
            raise FrozenModelArtifactError(
                f"Expected file '{entry['repo_path']}' not found in the {origin} "
                f"({source}). The repo/revision may be wrong, incomplete, or the "
                "manifest is stale relative to what was actually uploaded."
            )
        actual_size = source.stat().st_size
        if actual_size != entry["size_bytes"]:
            raise FrozenModelArtifactError(
                f"Size mismatch for '{entry['repo_path']}': expected "
                f"{entry['size_bytes']} bytes, got {actual_size}. Treating this "
                "as a corrupt or unexpected artifact — refusing to deploy it."
            )
        actual_hash = _sha256(source)
        if actual_hash != entry["sha256"]:
            raise FrozenModelArtifactError(
                f"Checksum mismatch for '{entry['repo_path']}': expected "
                f"{entry['sha256']}, got {actual_hash}. The downloaded artifact "
                "does not match the known-good frozen model — refusing to place "
                "it. This must never be silently ignored."
            )

        dest = _PROJECT_ROOT / entry["local_dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_suffix(dest.suffix + ".partial")
        shutil.copyfile(source, tmp_dest)
        os.replace(tmp_dest, dest)  # atomic — never leaves a half-written file at `dest`
        print(f"  placed {entry['local_dest']} ({actual_size} bytes, sha256 verified)")


_REVISION_RE = __import__("re").compile(r"^[0-9a-f]{40}$")
_REVISION_MARKER = ".revision"


def _public_models(manifest: dict) -> list[dict]:
    models = manifest.get("public_models") or []
    for entry in models:
        if not _REVISION_RE.match(str(entry.get("revision", ""))):
            raise FrozenModelArtifactError(
                f"Public model '{entry.get('repo_id')}' must be pinned to a full 40-character commit SHA in the manifest; refusing to follow a floating branch or tag."
            )
    return models


def _public_satisfied(entry: dict) -> bool:
    """True if the model's local_dir already holds every file at the pinned revision with matching size and checksum."""
    dest = _PROJECT_ROOT / entry["local_dir"]
    marker = dest / _REVISION_MARKER
    if not marker.exists() or marker.read_text().strip() != entry["revision"]:
        return False
    for f in entry["files"]:
        p = dest / f["name"]
        if not p.exists() or p.stat().st_size != f["size_bytes"] or _sha256(p) != f["sha256"]:
            return False
    return True


def _verify_public_file(entry: dict, f: dict, path: Path) -> None:
    if not path.exists():
        raise FrozenModelArtifactError(f"Expected file '{f['name']}' of public model '{entry['repo_id']}' @ {entry['revision']} was not found ({path}).")
    if path.stat().st_size != f["size_bytes"]:
        raise FrozenModelArtifactError(
            f"Size mismatch for '{entry['repo_id']}/{f['name']}': expected {f['size_bytes']} bytes, got {path.stat().st_size}. Refusing to use a different artifact."
        )
    if _sha256(path) != f["sha256"]:
        raise FrozenModelArtifactError(
            f"Checksum mismatch for '{entry['repo_id']}/{f['name']}' @ {entry['revision']}. The download does not match the model that was tested; refusing to place it."
        )


def _fetch_public_model(entry: dict) -> None:
    """Place one public model's pinned files under entry['local_dir'], atomically (a half-downloaded model is never left where the loader would find it)."""
    dest = _PROJECT_ROOT / entry["local_dir"]
    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    source_override = os.environ.get("RCS_FROZEN_MODELS_SOURCE_DIR")
    if source_override:
        source_dir = Path(source_override) / "public" / entry["repo_id"].replace("/", "__")
        print(f"fetch_frozen_models: offline mode — public model {entry['repo_id']} from {source_dir}")
        for f in entry["files"]:
            _verify_public_file(entry, f, source_dir / f["name"])
            shutil.copyfile(source_dir / f["name"], staging / f["name"])
    else:
        print(f"fetch_frozen_models: downloading public model {entry['repo_id']} @ {entry['revision']}")
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError

        try:
            snapshot_download(
                repo_id=entry["repo_id"], revision=entry["revision"], repo_type="model",
                allow_patterns=[f["name"] for f in entry["files"]], local_dir=str(staging),
            )
        except HfHubHTTPError as exc:
            raise FrozenModelArtifactError(f"Failed to fetch public model '{entry['repo_id']}' @ {entry['revision']}: {exc}") from exc
        for f in entry["files"]:
            _verify_public_file(entry, f, staging / f["name"])
    (staging / _REVISION_MARKER).write_text(entry["revision"] + "\n")
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(staging, dest)
    print(f"  placed {entry['local_dir']} ({len(entry['files'])} files, sha256 verified, revision {entry['revision'][:12]})")


def fetch_frozen_models() -> None:
    manifest = _load_manifest()
    public = _public_models(manifest)
    public_todo = [m for m in public if not _public_satisfied(m)]
    private_ok = _already_satisfied(manifest)

    if private_ok and not public_todo:
        print("fetch_frozen_models: all required files already present and verified — nothing to do.")
        return

    if not private_ok:
        source_override = os.environ.get("RCS_FROZEN_MODELS_SOURCE_DIR")
        if source_override:
            print(f"fetch_frozen_models: offline mode — sourcing from {source_override}")
            _place_files(manifest, Path(source_override), offline=True)
        else:
            snapshot_dir = _download_snapshot(manifest)
            _place_files(manifest, snapshot_dir, offline=False)
        total_bytes = sum(e["size_bytes"] for e in manifest["files"])
        print(f"fetch_frozen_models: private artifacts done — {len(manifest['files'])} files, {total_bytes / 1024 / 1024:.1f} MB total.")

    for entry in public_todo:
        _fetch_public_model(entry)
    print(f"fetch_frozen_models: done — {len(public)} public model(s) verified.")


if __name__ == "__main__":
    try:
        fetch_frozen_models()
    except FrozenModelArtifactError as exc:
        print(f"fetch_frozen_models: FAILED — {exc}", file=sys.stderr)
        sys.exit(1)
