"""
Append-only JSONL store for explicitly-saved Playground cases.

  * Nothing is written unless a caller explicitly saves a case (no autosave,
    no logging of run inputs).
  * The directory must NOT be (or sit under) any gold / production data
    directory; the guard below refuses such locations.
  * Never touches a database. Files are created owner-only where the OS allows.
  * Each save appends a new version; reads collapse to the latest version per
    case_id; delete appends a tombstone (history stays auditable on disk).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

_FORBIDDEN_DIR_PARTS = frozenset({"gold", "db", "uploads", "raw", "models", "selection", "final_test", "feedback_research", "fixtures", "processed"})
CASES_FILENAME = "cases.jsonl"


class UnsafeStorageLocation(ValueError):
    pass


def assert_safe_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    bad = [p for p in resolved.parts if p.lower() in _FORBIDDEN_DIR_PARTS]
    if bad:
        raise UnsafeStorageLocation(f"playground results must not be stored under gold/production data directories (found {bad})")
    return resolved


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


class PlaygroundStore:
    def __init__(self, directory: Path):
        self.dir = assert_safe_dir(directory)
        self.path = self.dir / CASES_FILENAME
        self._lock = threading.Lock()

    # -- internals ---------------------------------------------------------
    def _append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.dir, 0o700)
            except OSError:
                pass
            new_file = not self.path.exists()
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if new_file:
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with self._lock, open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # a torn line must never break reads
        return out

    # -- API ---------------------------------------------------------------
    def save_case(self, record: dict[str, Any], *, saved_by: str | None) -> dict[str, Any]:
        case_id = record.get("case_id") or f"pg-{uuid.uuid4().hex[:12]}"
        latest = self.get_case(case_id)
        version = (latest["version"] + 1) if latest else 1
        full = {**record, "record_type": "case", "case_id": case_id, "version": version, "saved_at": _now(), "saved_by": saved_by}
        self._append(full)
        return full

    def list_cases(self) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for rec in self._read_all():
            cid = rec.get("case_id")
            if not cid:
                continue
            if rec.get("record_type") == "deleted":
                latest.pop(cid, None)
            elif rec.get("record_type") == "case":
                if cid not in latest or rec.get("version", 0) >= latest[cid].get("version", 0):
                    latest[cid] = rec
        return sorted(latest.values(), key=lambda r: r.get("saved_at", ""))

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        for rec in self.list_cases():
            if rec["case_id"] == case_id:
                return rec
        return None

    def delete_case(self, case_id: str, *, deleted_by: str | None) -> bool:
        if self.get_case(case_id) is None:
            return False
        self._append({"record_type": "deleted", "case_id": case_id, "deleted_at": _now(), "deleted_by": deleted_by})
        return True
