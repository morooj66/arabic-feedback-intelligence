"""Hash-approved sanitised inputs: the ONLY text that may ever leave the machine for the real-data comparison.

    build(records, out_dir)  runs the preflight (production sanitiser + residual-risk screen) and writes, per dataset,
        <dataset>_sanitized_inputs.jsonl   {"id", "sanitized_text"} for rows that passed
        <dataset>_INSPECTION.md            table for the independent human reader (owner-only permissions; never copy, paste or upload it)
        <dataset>_quarantined.json         ids and reason codes only
    and returns the SHA-256 of the jsonl. The owner records that hash in the approval file after a person has read the INSPECTION file;
    compare_real.py refuses to send anything whose hash differs. Automation reduces risk but does not guarantee anonymisation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from .privacy import preflight
from .runner import Record

DEFAULT_DIR = Path(__file__).with_name("results") / "inputs"


def jsonl_path(dataset: str, directory: Path = DEFAULT_DIR) -> Path:
    return directory / f"{dataset}_sanitized_inputs.jsonl"


def build(records: Iterable[Record], dataset: str, out_dir: Path = DEFAULT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, quarantined = [], {}
    for r in records:
        p = preflight(r.text)
        if p.eligible:
            kept.append((r.record_id, p.text, sum(v for v in p.actions.values() if isinstance(v, int))))
        else:
            quarantined[r.record_id] = p.blocked_reason
    kept.sort(key=lambda x: x[0])
    body = "".join(json.dumps({"id": i, "sanitized_text": t}, ensure_ascii=False) + "\n" for i, t, _ in kept)
    path = jsonl_path(dataset, out_dir)
    path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    insp = out_dir / f"{dataset}_INSPECTION.md"
    lines = [f"# {dataset}: sanitised inputs for independent privacy review", "", f"sha256 of {path.name}: `{digest}`", "",
             "Quarantine (do not approve) any record with a personal name, nickname, kunya, relative's name, signature, handle, contact detail, workplace or sensitive circumstance.", "",
             "| id | masks applied | sanitised text |", "|---|---|---|"]
    lines += [f"| {i} | {n} | {t.replace('|', '/')} |" for i, t, n in kept]
    insp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for f in (path, insp):
        os.chmod(f, 0o600)
    q = out_dir / f"{dataset}_quarantined.json"
    q.write_text(json.dumps(quarantined, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"dataset": dataset, "sent_candidates": len(kept), "quarantined": len(quarantined), "sha256": digest, "files": [path.name, insp.name, q.name]}


class InputHashMismatch(RuntimeError):
    pass


def load_verified(dataset: str, expected_sha256: str, directory: Path = DEFAULT_DIR) -> Mapping[str, str]:
    path = jsonl_path(dataset, directory)
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise InputHashMismatch(f"{dataset}: sanitised inputs differ from the hash the owner approved; nothing will be sent")
    return {d["id"]: d["sanitized_text"] for d in (json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip())}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - reads real text locally; not run during preparation
    import argparse

    from . import real_data as rd

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("human_test", "blind_natural"), required=True)
    args = ap.parse_args(argv)
    s = rd.load_human_test({}).records if args.dataset == "human_test" else rd.load_blind_natural().records
    print(json.dumps(build(s, args.dataset), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
