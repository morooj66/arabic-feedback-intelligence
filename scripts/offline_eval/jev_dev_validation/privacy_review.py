"""Privacy tooling for the Jev evaluation inputs. LOCAL ONLY: no network, no external call, ever.

  python privacy_review.py ledger              build the transmission audit record (ids + hashes, NO text) and lock it down
  python privacy_review.py verify              run every automated privacy check; prints counts and eval ids only
  python privacy_review.py quarantine tr-0007  add records to the manual quarantine list and regenerate the inputs + hash

IMPORTANT LIMIT: these checks reduce risk, they do NOT prove anonymisation. Bare Arabic personal names, nicknames and
personal circumstances cannot be detected reliably by rules, so a human read of INSPECTION.md remains the control that
matters. Nothing in this module prints, logs or writes comment text outside the locked audit directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_common  # noqa: E402,F401
from jev_common import OUT_DIR, ROOT, V2_HOLDOUT_DIR  # noqa: E402
from jev_redaction import Quarantined, redact_strict  # noqa: E402

AUDIT_DIR = ROOT / "data" / "evaluation" / "privacy_audit"

# Eval ids a reviewer confirmed contain a personal name in text that WAS transmitted (ids only; no names recorded here).
KNOWN_EXPOSURE = {
    "val_run_2026-10-14": ["dev-0008", "dev-0012", "dev-0015", "dev-0045", "dev-0065", "dev-0100"],
    "holdout_canary_2026-10-15": ["tr-0001", "tr-0006"],
}

_DIAC = re.compile(r"[ً-ٰٟـ]")
_TOKEN = re.compile(r"[؀-ۿ]{3,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL = re.compile(r"(?i)https?://|www\.")
_LONG_DIGITS = re.compile(r"(?:\d[\s().-]*){7,}")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_token(t: str) -> str:
    t = unicodedata.normalize("NFKC", _DIAC.sub("", t))
    return t.translate(str.maketrans("أإآٱىة", "اااايه"))


def tokens(text: str) -> set[str]:
    return {norm_token(m) for m in _TOKEN.findall(text)}


RAW_XLSX = ROOT / "data" / "raw" / "historical-export.xlsx"
RARE_IN_CORPUS_MAX_DF = 3


def corpus_document_frequency() -> Counter:
    """Document frequency of every Arabic token over the full local source corpus (8k+ comments). Used only to decide whether
    a token is idiosyncratic (a name-like or very rare word) or an ordinary word; the corpus itself is never printed or copied."""
    import pandas as pd
    from config import settings
    from rcs.ingest.loader import load_raw

    raw = load_raw(RAW_XLSX, settings.SOURCE_COLUMNS)
    df = Counter()
    for t in raw["original_comment"].fillna("").astype(str):
        df.update(tokens(t))
    return df


def lock(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------------------- ledger


def build_ledger(audit_dir: Path = AUDIT_DIR, val_dir: Path = OUT_DIR, hold_dir: Path = V2_HOLDOUT_DIR) -> dict:
    """Audit record of everything transmitted to the provider. The ledger holds ids, hashes, counts and provider
    request ids only; the exact transmitted texts go to a separate owner-only (0400) file in a 0700 directory."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    lock(audit_dir, 0o700)
    entries, texts = [], []

    def add(run, criteria, inputs, preds):
        by_id = {r["eval_id"]: r["text"] for r in inputs}
        for p in preds:
            i = p["eval_id"]
            meta = p.get("provider_meta", {})
            entries.append({"run": run, "criteria": criteria, "eval_id": i, "text_sha256": sha256_text(by_id[i]),
                            "chars": len(by_id[i]), "status": "success", "provider_request_id": meta.get("id"),
                            "usd": (meta.get("usage") or {}).get("cost")})
            texts.append({"run": run, "eval_id": i, "text": by_id[i]})

    val_inputs = _rows(val_dir / "sanitized_inputs.jsonl")
    add("val_canary_2026-10-14", "v1", val_inputs, _rows(val_dir / "jev_canary_predictions.jsonl"))
    add("val_rest_2026-10-14", "v1", val_inputs, _rows(val_dir / "jev_rest_predictions.jsonl"))
    old = hold_dir / "sanitized_inputs.v0_approved_7047d9b0.jsonl"
    if not old.exists():
        old = audit_dir / "holdout_inputs_at_canary_7047d9b0.jsonl"
    hold_inputs = _rows(old)
    add("holdout_canary_2026-10-15", "v1", hold_inputs, _rows(hold_dir / "jev_v1_canary.jsonl"))
    v2 = _rows(hold_dir / "jev_v2_canary.jsonl")
    add("holdout_canary_2026-10-15", "v2", hold_inputs, v2)
    sent_v2 = {p["eval_id"] for p in v2}
    by_id = {r["eval_id"]: r["text"] for r in hold_inputs}
    # the 5th v2 request was sent and answered HTTP 403 (no response body was kept, no cost was reported)
    for i in [r["eval_id"] for r in hold_inputs][:5]:
        if i not in sent_v2:
            entries.append({"run": "holdout_canary_2026-10-15", "criteria": "v2", "eval_id": i,
                            "text_sha256": sha256_text(by_id[i]), "chars": len(by_id[i]),
                            "status": "sent_provider_http_403", "provider_request_id": None, "usd": None})
            texts.append({"run": "holdout_canary_2026-10-15", "eval_id": i, "text": by_id[i]})
    ledger = {
        "purpose": "audit of data transmitted to the external classification provider (OpenRouter / Vendor Jev)",
        "contains_text": False,
        "requests_total": len(entries),
        "requests_successful": sum(e["status"] == "success" for e in entries),
        "usd_total_reported": round(sum(e["usd"] or 0 for e in entries), 6),
        "by_run": {f"{k[0]}|{k[1]}": v for k, v in Counter((e["run"], e["criteria"]) for e in entries).items()},
        "input_hashes": {"val_inputs": sha256_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_inputs)),
                         "holdout_inputs_at_canary": sha256_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hold_inputs))},
        "known_exposure_confirmed_by_reviewer_ids_only": KNOWN_EXPOSURE,
        "known_exposure_note": "Not exhaustive for the validation run: a reviewer identified these while reading; others may exist.",
        "entries": entries,
    }
    lp = audit_dir / "transmission_ledger.json"
    tp = audit_dir / "transmitted_payload_texts.jsonl"
    for p in (lp, tp):
        if p.exists():
            lock(p, 0o600)
    lp.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    tp.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
    lock(lp, 0o400)
    lock(tp, 0o400)
    if old.exists() and old.parent != audit_dir:  # move the full archived input out of the working folder
        dest = audit_dir / "holdout_inputs_at_canary_7047d9b0.jsonl"
        if dest.exists():
            lock(dest, 0o600)
        old.replace(dest)
        lock(dest, 0o400)
    return {k: v for k, v in ledger.items() if k != "entries"}


# ----------------------------------------------------------------------- verification


def verify(hold_dir: Path = V2_HOLDOUT_DIR, val_dir: Path = OUT_DIR, audit_dir: Path = AUDIT_DIR) -> dict:
    res: dict = {"checks": {}, "flagged_ids": {}}
    ok = res["checks"]
    inputs = _rows(hold_dir / "sanitized_inputs.jsonl")
    ids = [r["eval_id"] for r in inputs]
    plan = json.loads((hold_dir / "request_plan.json").read_text())
    quarantined = json.loads((hold_dir / "quarantined.json").read_text())
    qids = {q["eval_id"] for q in quarantined}
    manual = set(json.loads((hold_dir / "manual_quarantine.local.json").read_text())) if (hold_dir / "manual_quarantine.local.json").exists() else set()
    digest = hashlib.sha256((hold_dir / "sanitized_inputs.jsonl").read_bytes()).hexdigest()
    res["counts"] = {"ready": len(inputs), "quarantined": len(quarantined), "manual_quarantined": len(manual)}
    res["input_sha256"] = digest
    ok["hash_matches_request_plan"] = digest == plan["sanitized_inputs_sha256"]
    ok["no_quarantined_id_in_inputs"] = not (qids & set(ids))
    ok["manual_list_subset_of_quarantined"] = manual <= qids
    exposed = set(KNOWN_EXPOSURE["holdout_canary_2026-10-15"])
    ok["known_exposed_holdout_ids_excluded"] = not (exposed & set(ids))
    ok["eval_ids_unique"] = len(ids) == len(set(ids))

    fixed_point_fail, struct = [], {"email": [], "url": [], "digits7": [], "at_sign": [], "id_adjacent": []}
    for r in inputs:
        t = r["text"]
        try:
            if redact_strict(t).text != t:
                fixed_point_fail.append(r["eval_id"])
        except Quarantined:
            fixed_point_fail.append(r["eval_id"])
        if _EMAIL.search(t): struct["email"].append(r["eval_id"])
        if _URL.search(t): struct["url"].append(r["eval_id"])
        if _LONG_DIGITS.search(t): struct["digits7"].append(r["eval_id"])
        if "@" in t: struct["at_sign"].append(r["eval_id"])
        if re.search(r"\[ID\]\s*[؀-ۿ]", t): struct["id_adjacent"].append(r["eval_id"])
    ok["every_text_is_a_redaction_fixed_point"] = not fixed_point_fail
    ok["no_structural_identifier_patterns"] = not any(struct.values())
    res["flagged_ids"]["not_fixed_point"] = fixed_point_fail
    res["flagged_ids"].update({k: v for k, v in struct.items() if v})

    # (a) GATE: tokens of the records a reviewer CONFIRMED contain a personal name (never printed). An ordinary word is
    #     shared by many records, so only tokens that are rare in the WHOLE source corpus (df <= 3) are meaningful.
    # (b) INFORMATIONAL: the same test against all manually quarantined records, which also include long letters and
    #     personal stories whose ordinary words would otherwise create noise.
    source = audit_dir / "holdout_inputs_at_canary_7047d9b0.jsonl"
    if not source.exists():
        source = hold_dir / "sanitized_inputs.v0_approved_7047d9b0.jsonl"
    confirmed_tokens: set[str] = set()
    manual_tokens: set[str] = set()
    if source.exists():
        for r in _rows(source):
            if r["eval_id"] in exposed:
                confirmed_tokens |= tokens(r["text"])
            if r["eval_id"] in manual:
                manual_tokens |= tokens(r["text"])
    val_inputs = {r["eval_id"]: r["text"] for r in _rows(val_dir / "sanitized_inputs.jsonl")}
    for i in KNOWN_EXPOSURE["val_run_2026-10-14"]:
        if i in val_inputs:
            confirmed_tokens |= tokens(val_inputs[i])
    df = Counter()
    for r in inputs:
        df.update(tokens(r["text"]))
    corpus_df = corpus_document_frequency() if RAW_XLSX.exists() else df
    rare_confirmed = {t for t in confirmed_tokens if corpus_df[t] <= RARE_IN_CORPUS_MAX_DF}
    hit_confirmed = sorted({r["eval_id"] for r in inputs if tokens(r["text"]) & rare_confirmed})
    rare_manual = {t for t in manual_tokens if corpus_df[t] <= RARE_IN_CORPUS_MAX_DF}
    hit_manual = sorted({r["eval_id"] for r in inputs if tokens(r["text"]) & rare_manual})
    res["flagged_ids"]["shares_rare_token_with_a_CONFIRMED_exposure_record"] = hit_confirmed
    res["counts"]["rare_confirmed_tokens_checked"] = len(rare_confirmed)
    res["counts"]["informational_records_sharing_rare_token_with_manually_quarantined"] = len(hit_manual)
    res["flagged_ids"]["PRIORITY_FOR_HUMAN_REVIEW_share_rare_token_with_manually_quarantined"] = hit_manual
    ok["no_record_shares_a_rare_token_with_a_confirmed_exposure_record"] = not hit_confirmed

    ok["audit_dir_owner_only"] = audit_dir.exists() and not (audit_dir.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO))
    ok["ledger_present"] = (audit_dir / "transmission_ledger.json").exists()
    res["all_automated_checks_passed"] = all(ok.values())
    res["disclaimer"] = ("Automated checks cannot guarantee anonymisation; an independent human review of INSPECTION.md "
                         "is required before any further transmission.")
    return res


# ------------------------------------------------------------------------ quarantine


def add_quarantine(new_ids: list[str], hold_dir: Path = V2_HOLDOUT_DIR) -> dict:
    mp = hold_dir / "manual_quarantine.local.json"
    cur = set(json.loads(mp.read_text())) if mp.exists() else set()
    known = {r["eval_id"] for r in _rows(hold_dir / "sanitized_inputs.jsonl")} | {q["eval_id"] for q in json.loads((hold_dir / "quarantined.json").read_text())}
    bad = [i for i in new_ids if i not in known]
    if bad:
        raise SystemExit(f"unknown eval id(s): {bad}")
    mp.write_text(json.dumps(sorted(cur | set(new_ids))))
    script = Path(__file__).resolve().parent / "prepare_inputs.py"
    subprocess.run([sys.executable, str(script), "--split", "train", "--manual-quarantine", str(mp)],
                   check=True, stdout=subprocess.DEVNULL)
    for f in ("sanitized_inputs.jsonl", "INSPECTION.md", "id_map.local.json", "gold_labels.local.json"):
        lock(hold_dir / f, 0o600)
    plan = json.loads((hold_dir / "request_plan.json").read_text())
    return {"added": sorted(set(new_ids) - cur), "manual_total": len(cur | set(new_ids)),
            "NEW_input_sha256_requires_reapproval": plan["sanitized_inputs_sha256"]}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "ledger":
        print(json.dumps(build_ledger(), indent=1))
    elif cmd == "verify":
        r = verify()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["all_automated_checks_passed"] else 1)
    elif cmd == "quarantine":
        print(json.dumps(add_quarantine(sys.argv[2:]), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
