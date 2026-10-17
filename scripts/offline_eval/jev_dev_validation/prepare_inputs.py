"""Build sanitized Jev evaluation inputs from the DEVELOPMENT validation split. LOCAL ONLY.

Makes no network call and writes nothing outside data/evaluation/jev_dev_validation/.
Selects ONLY rows with split == 'val' (previously exposed development data),
asserts none of them is a sealed human_test id, redacts every text with
fail-closed rules, quarantines anything uncertain, and writes the plan for
human inspection. Nothing is sent anywhere by this script.

Usage:  python scripts/offline_eval/jev_dev_validation/prepare_inputs.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from jev_common import (  # noqa: E402
    ALLOWED_SPLITS, DEFAULT_SPLITS, GOLD_CSV, HUMAN_TEST_EXCLUSION, OBSERVED_USD_PER_REQUEST, OUT_DIR, V2_HOLDOUT_DIR,
)
from config import settings  # noqa: E402
from jev_protocol import build_request, estimate_cost  # noqa: E402
from jev_redaction import Quarantined, redact_strict  # noqa: E402


def _human_test_ids(path: Path) -> set[str]:
    ids: set[str] = set()

    def walk(o):
        if isinstance(o, str) and len(o) == 64:
            ids.add(o)
        elif isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(json.loads(path.read_text()))
    return ids


def load_dev_rows(path: Path = GOLD_CSV, splits: tuple[str, ...] = DEFAULT_SPLITS) -> pd.DataFrame:
    """Stream the gold CSV and keep ONLY allowed-split rows.

    The file also contains the sealed human_test rows (split='test'); this
    reader drops every other row the moment its split cell is read, so their
    text and labels are never retained, parsed into a DataFrame, or inspected."""
    if not set(splits) <= set(ALLOWED_SPLITS):
        raise SystemExit(f"split(s) {sorted(set(splits) - set(ALLOWED_SPLITS))} are not selectable (sealed or unknown)")
    kept = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") in splits:
                kept.append(row)
    df = pd.DataFrame(kept)
    if "is_few_shot_example" in df.columns:
        df["is_few_shot_example"] = df["is_few_shot_example"].astype(str).str.lower().isin(("true", "1", "yes"))
    for col in ("gold_primary_intent", "gold_topic_subtype", "gold_sentiment"):
        df[col] = df[col].replace("", pd.NA)
    return df


def select_rows(df: pd.DataFrame, sealed_ids: set[str], splits: tuple[str, ...] = DEFAULT_SPLITS) -> pd.DataFrame:
    """Development validation rows with a complete, human-confirmed gold label set.

    The split filter runs BEFORE anything else touches row content, and a
    sealed-id overlap is a hard error, never a silent skip."""
    dev = df[df["split"].isin(splits)].copy()
    overlap = set(dev["comment_id"]) & sealed_ids
    if overlap:
        raise SystemExit(f"refusing to continue: {len(overlap)} selected rows are sealed human_test ids")
    dev = dev[dev["gold_annotation_status"] == "LABELED"]
    for col in ("gold_primary_intent", "gold_topic_subtype", "gold_sentiment"):
        dev = dev[dev[col].notna()]
    if "is_few_shot_example" in dev.columns:
        dev = dev[~dev["is_few_shot_example"].fillna(False).astype(bool)]
    return dev.reset_index(drop=True)


def build(df: pd.DataFrame, sealed_ids: set[str], released: frozenset[str] = frozenset(),
          splits: tuple[str, ...] = DEFAULT_SPLITS, prefix: str = "dev", exclude_texts: frozenset[str] = frozenset(),
          manual_quarantine: frozenset[str] = frozenset()) -> dict:
    """`released`: eval_ids a human reviewed locally (INSPECTION of quarantined.json + id_map) and cleared
    for Latin-name replacement. Empty by default, so the default behaviour is pure quarantine."""
    rows = select_rows(df, sealed_ids, splits)
    kept, quarantined, gold, idmap, report_rows = [], [], {}, {}, []
    seen_norm: set[str] = set()
    dup_excluded = Counter()
    rep_total: Counter = Counter()
    q_support: Counter = Counter()
    reasons: Counter = Counter()
    for i, r in rows.iterrows():
        eval_id = f"{prefix}-{i + 1:04d}"
        if eval_id in manual_quarantine:  # a human/reviewer read the sanitized text and found a possible name or personal circumstance
            quarantined.append({"eval_id": eval_id, "reason": "manual_review_possible_personal_data"})
            reasons["manual_review_possible_personal_data"] += 1
            q_support.update([("intent", r["gold_primary_intent"]), ("topic", r["gold_topic_subtype"]),
                              ("sentiment", r["gold_sentiment"])])
            continue
        try:
            red = redact_strict(str(r["comment_text"]), release_latin_names=eval_id in released)
        except Quarantined as q:
            quarantined.append({"eval_id": eval_id, "reason": q.reason})
            q_support.update([("intent", r["gold_primary_intent"]), ("topic", r["gold_topic_subtype"]),
                              ("sentiment", r["gold_sentiment"])])
            reasons[q.reason] += 1
            continue
        norm = " ".join(red.text.split()).lower()
        if norm in exclude_texts:
            dup_excluded["identical_to_excluded_set"] += 1  # e.g. a validation text that motivated v2
            continue
        if norm in seen_norm:
            dup_excluded["repeated_within_set"] += 1
            continue
        seen_norm.add(norm)
        kept.append({"eval_id": eval_id, "text": red.text})
        gold[eval_id] = {"intent": r["gold_primary_intent"], "topic": r["gold_topic_subtype"],
                         "sentiment": r["gold_sentiment"]}
        idmap[eval_id] = r["comment_id"]
        rep_total.update(red.replacements)
        report_rows.append({"eval_id": eval_id, "replacements": red.replacements, "text": red.text})
    return dict(kept=kept, quarantined=quarantined, gold=gold, idmap=idmap, report_rows=report_rows,
                rep_total=rep_total, reasons=reasons, selected=len(rows), q_support=q_support,
                dup_excluded=dup_excluded)


def write_outputs(result: dict, out_dir: Path = OUT_DIR, versions: tuple[str, ...] = ("v1",),
                  splits: tuple[str, ...] = DEFAULT_SPLITS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = out_dir / "sanitized_inputs.jsonl"
    inputs_path.write_text(
        "".join(json.dumps(k, ensure_ascii=False) + "\n" for k in result["kept"]), encoding="utf-8")
    digest = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    payloads = [build_request(k["text"]) for k in result["kept"]]
    cost = estimate_cost(payloads)
    per_version = {v: estimate_cost([build_request(k["text"], v) for k in result["kept"]]) for v in versions}
    total_req = sum(c["requests"] for c in per_version.values())
    (out_dir / "gold_labels.local.json").write_text(json.dumps(result["gold"], ensure_ascii=False, indent=1))
    (out_dir / "id_map.local.json").write_text(json.dumps(result["idmap"], indent=1))
    (out_dir / "quarantined.json").write_text(json.dumps(result["quarantined"], indent=1))
    report = {
        "split_used": list(splits),
        "rows_selected_labeled": result["selected"],
        "rows_sanitized_ready": len(result["kept"]),
        "rows_quarantined": len(result["quarantined"]),
        "rows_excluded_as_duplicates": dict(result["dup_excluded"]),
        "quarantine_reasons": dict(result["reasons"]),
        "rows_with_any_replacement": sum(1 for r in result["report_rows"] if r["replacements"]),
        "replacements_by_detector": dict(result["rep_total"]),
        "quarantined_label_support": {
            q: {lab: n for (task, lab), n in result["q_support"].items() if task == q}
            for q in ("intent", "topic", "sentiment")
        },
        "label_support": {
            q: dict(Counter(g[q] for g in result["gold"].values()))
            for q in ("intent", "topic", "sentiment")
        },
        "sanitized_inputs_sha256": digest,
    }
    (out_dir / "redaction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    plan = {
        **cost,
        "model": settings.JEV_TOPIC_MODEL,
        "expected_snapshot": settings.JEV_TOPIC_EXPECTED_SNAPSHOT,
        "calls_executed": 0,
        "versions": per_version,
        "total_requests_all_versions": total_req,
        "observed_rate_estimate_usd": round(OBSERVED_USD_PER_REQUEST * total_req, 4),
        "sanitized_inputs_sha256": digest,
        "execute_requires": ["--execute", f"--approve-sha256 {digest}", "--max-requests N", "--max-usd X",
                             "OPENROUTER_API_KEY in the environment"],
    }
    (out_dir / "request_plan.json").write_text(json.dumps(plan, indent=1))
    lines = ["# Sanitized inputs for inspection (sanitized text only; originals are not written here)", "",
             f"sha256 of sanitized_inputs.jsonl: `{digest}`", "",
             "| eval_id | replacements | sanitized text |", "|---|---|---|"]
    for r in result["report_rows"]:
        lines.append(f"| {r['eval_id']} | {r['replacements'] or ''} | {r['text'].replace('|', '/').replace(chr(10), ' ')} |")
    (out_dir / "INSPECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": report, "plan": plan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-file", help="JSON list of quarantined eval_ids a human reviewed and cleared")
    ap.add_argument("--split", choices=ALLOWED_SPLITS, default="val",
                    help="val = original evaluation set; train = unseen holdout for the v1-vs-v2 Jev comparison")
    ap.add_argument("--manual-quarantine", help="JSON list of eval_ids to quarantine after a manual read of the sanitized text")
    ap.add_argument("--exclude-texts-from", help="sanitized_inputs.jsonl whose texts must not reappear (default for --split train: the val inputs)")
    args = ap.parse_args()
    splits = (args.split,)
    out_dir = OUT_DIR if args.split == "val" else V2_HOLDOUT_DIR
    versions = ("v1",) if args.split == "val" else ("v1", "v2")
    prefix = "dev" if args.split == "val" else "tr"
    released = frozenset(json.loads(Path(args.release_file).read_text())) if args.release_file else frozenset()
    excl_path = Path(args.exclude_texts_from) if args.exclude_texts_from else (OUT_DIR / "sanitized_inputs.jsonl" if args.split == "train" else None)
    exclude = frozenset(" ".join(json.loads(l)["text"].split()).lower() for l in excl_path.read_text(encoding="utf-8").splitlines() if l.strip()) if excl_path and excl_path.exists() else frozenset()
    manual = frozenset(json.loads(Path(args.manual_quarantine).read_text())) if args.manual_quarantine else frozenset()
    summary = write_outputs(
        build(load_dev_rows(splits=splits), _human_test_ids(HUMAN_TEST_EXCLUSION), released, splits, prefix, exclude, manual),
        out_dir, versions, splits)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nInspect {out_dir / 'INSPECTION.md'} before approving any external call. NO call was made.")


if __name__ == "__main__":
    main()
