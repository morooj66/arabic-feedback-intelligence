"""Import batch identity for Excel uploads.

A comment's id is sha256("<source_file>|<row index>") (rcs.ingest.loader._comment_id), so the batch name (``source_file``) IS the identity of every row.
Before this module, a different workbook uploaded under an existing filename produced the SAME ids as the earlier batch: the new rows were silently
skipped (processed rows are locked) or, worse, overwrote unprocessed rows, and the job still reported success.

Rules (backward compatible; no schema change, no historical id is touched):
* the first upload under a name keeps that name, exactly as before;
* a later upload whose comment content is COMPATIBLE with the rows already stored under that name (every stored row matches the file at the same row
  index) is the same batch -> same name -> idempotent re-import / resume;
* otherwise it is a DIFFERENT batch and receives the content-addressed name ``<stem>__<sha8><suffix>`` (also idempotent: the same file always maps to
  the same name);
* if even that name is occupied by incompatible content, the import fails with ImportIdentityCollision -- never a silent skip.
``assert_no_collision`` is the same check as a last line of defence in front of the ingest step.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from config import settings
from rcs.ingest.cleaning import normalize_text
from rcs.ingest.loader import load_raw
from rcs.review.persistence import resolve_bundle

_CHUNK = 500
_claims: dict[str, str] = {}          # in-process: batch name -> content sha, closes the window between "session created" and "first rows stored"
_lock = threading.Lock()


class ImportIdentityCollision(RuntimeError):
    """Stored rows under this batch name belong to different content, so importing would mix or overwrite batches. Nothing was changed."""


@dataclass(frozen=True)
class IdentityCheck:
    state: str                 # "free" | "compatible" | "conflict"
    stored_rows: int = 0
    mismatched_rows: int = 0


def fingerprint_of(texts: Mapping[int, str]) -> str:
    return hashlib.sha256("\n".join(f"{i}\t{texts[i]}" for i in sorted(texts)).encode("utf-8")).hexdigest()


def content_fingerprint(path: Path) -> str:
    """Hash of the comment text per row (not of the file bytes): re-saving the same comments with new file metadata is still the same batch."""
    return fingerprint_of(_file_texts(Path(path)))


def unique_name(name: str, sha256: str) -> str:
    p = Path(name)
    return f"{p.stem}__{sha256[:8]}{p.suffix}"


def _file_texts(path: Path) -> dict[int, str]:
    raw = load_raw(Path(path), settings.SOURCE_COLUMNS)
    return {int(i): (normalize_text(t) or "") for i, t in zip(raw["source_row_index"], raw["original_comment"])}


def check_identity(engine, path: Path, source_file: str, *, content_sha: str | None = None, texts: Mapping[int, str] | None = None) -> IdentityCheck:
    """Compare the rows stored under ``source_file`` with the workbook at ``path`` (comment text at the same row index)."""
    file_texts = texts if texts is not None else _file_texts(path)
    sha = content_sha or fingerprint_of(file_texts)
    repo = resolve_bundle(engine).review_items
    ids = repo.comment_ids_for_source_file(source_file)
    if not ids:
        # nothing stored yet: only the in-process claim of another, not yet imported upload can occupy the name
        with _lock:
            claimed = _claims.get(source_file)
        return IdentityCheck("conflict") if claimed not in (None, sha) else IdentityCheck("free")
    mismatched = 0
    stored = 0
    for k in range(0, len(ids), _CHUNK):
        for row in repo.get_dedup_candidates(ids[k:k + _CHUNK]):
            stored += 1
            if file_texts.get(int(row.source_row_index)) != (normalize_text(row.original_comment) or ""):
                mismatched += 1
    return IdentityCheck("conflict" if mismatched else "compatible", stored, mismatched)


def resolve_source_file(engine, path: Path) -> Path:
    """Return the (possibly renamed) path whose file name is the batch identity to use. The file is only renamed inside its own upload directory."""
    path = Path(path)
    texts = _file_texts(path)
    sha = fingerprint_of(texts)
    plain = path.name
    for name in (plain, unique_name(plain, sha)):
        check = check_identity(engine, path, name, content_sha=sha, texts=texts)
        if check.state in ("free", "compatible"):
            with _lock:
                _claims[name] = sha
            return path if name == plain else path.rename(path.with_name(name))
    raise ImportIdentityCollision(f"batch name {plain!r} and its content-addressed alternative are both occupied by different content; import refused")


def assert_no_collision(engine, path: Path) -> None:
    """Fail clearly (before any write) if rows already stored under this file's name belong to different content."""
    check = check_identity(engine, Path(path), Path(path).name)
    if check.state == "conflict":
        raise ImportIdentityCollision(
            f"{check.mismatched_rows} of {check.stored_rows} stored rows under {Path(path).name!r} differ from this workbook; import refused, nothing was changed")


