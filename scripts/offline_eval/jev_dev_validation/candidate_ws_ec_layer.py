"""CANDIDATE: WORD_SUBMISSION / ENTRY_CORRECTION boundary layer. LOCAL ONLY. NOT wired into any production path.

Why this candidate (strongest evidence so far):
  * the boundary is the largest, most consistent topic error of BOTH baselines (RCS sends ENTRY_CORRECTION to WORD_SUBMISSION,
    Jev v1 sends WORD_SUBMISSION to ENTRY_CORRECTION);
  * the guide's deciding question ("does the entry already exist, and is something said to be wrong?") has two observable proxies:
    the source `lemma` metadata (the submission is attached to an existing entry) and explicit error-claim wording;
  * a rule with ZERO learned parameters, so there is nothing to overfit and it can be tested on every development source.

SPEC (frozen; the sha256 below identifies it before any new data is touched):
    if base_topic in {WORD_SUBMISSION, ENTRY_CORRECTION}:
        topic = ENTRY_CORRECTION  if (lemma_present or error_claim_cue(text))  else  WORD_SUBMISSION
    else: topic unchanged
  `lemma_present` is operational dictionary metadata, not visitor data, and is used LOCALLY only (never sent to any provider).
  Every other class and every other task is left exactly as the base model predicted.

Run:  python scripts/offline_eval/jev_dev_validation/candidate_ws_ec_layer.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_common  # noqa: E402,F401
import jev_metrics as M  # noqa: E402
from config import settings  # noqa: E402
from jev_common import GOLD_CSV, OUT_DIR, V2_HOLDOUT_DIR  # noqa: E402
from local_feature_experiment import LEX, RAW_XLSX  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402

WS, EC = "WORD_SUBMISSION", "ENTRY_CORRECTION"
from rcs.classification.topic_boundary_layer import ERROR_CLAIM_CUES, boundary_topic, has_error_claim  # noqa: E402

assert tuple(LEX["error_claim"]) == ERROR_CLAIM_CUES, "experiment lexicon drifted from the shipped layer"
SPEC = {"name": "ws_ec_boundary_layer_v1", "applies_to": [WS, EC], "entry_correction_if": ["lemma_present", "error_claim_cue"],
        "error_claim_cues": list(ERROR_CLAIM_CUES), "learned_parameters": 0,
        "inputs": ["base topic prediction", "comment text (local)", "lemma present flag (local metadata)"]}
SPEC_SHA256 = hashlib.sha256(json.dumps(SPEC, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def layer(base_topic: str, text: str, lemma_present: bool) -> str:
    return boundary_topic(base_topic, text, lemma_present)


def _paired_boot(gold, a, b, labels, n=2000, seed=20261016):
    rng = random.Random(seed)
    da, df = [], []
    for _ in range(n):
        idx = [rng.randrange(len(gold)) for _ in range(len(gold))]
        g = [gold[i] for i in idx]; pa = [a[i] for i in idx]; pb = [b[i] for i in idx]
        da.append(M.accuracy(g, pb) - M.accuracy(g, pa))
        df.append(M.macro_f1(M.per_class(g, pb, labels), only_supported=True) - M.macro_f1(M.per_class(g, pa, labels), only_supported=True))
    da.sort(); df.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    return (da[lo], da[hi]), (df[lo], df[hi])


def two_class(rows):
    """rows: [(gold_topic, text, lemma)] restricted to gold WS/EC; the rule alone as a WS-vs-EC resolver."""
    sub = [r for r in rows if r[0] in (WS, EC)]
    if not sub:
        return None
    ec = [r for r in sub if r[0] == EC]; ws = [r for r in sub if r[0] == WS]
    ec_rec = sum(layer(WS, t, l) == EC for _, t, l in ec) / len(ec) if ec else float("nan")
    ws_rec = sum(layer(WS, t, l) == WS for _, t, l in ws) / len(ws) if ws else float("nan")
    return {"n": len(sub), "n_ws": len(ws), "n_ec": len(ec), "accuracy": sum(layer(WS, t, l) == g for g, t, l in sub) / len(sub),
            "ec_recall": ec_rec, "ws_recall": ws_rec, "balanced_accuracy": (ec_rec + ws_rec) / 2,
            "majority_baseline": max(len(ws), len(ec)) / len(sub)}


def main() -> None:
    raw = load_raw(RAW_XLSX, settings.SOURCE_COLUMNS)
    lem = {r.comment_id: (str(r.lemma).strip() not in ("", "nan", "None")) for r in raw.itertuples()}
    # ---- gold rows by LABEL SOURCE (annotation_stage), raw text kept local
    by_stage: dict[str, list] = {}
    with open(GOLD_CSV, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in ("train", "val") and r["gold_annotation_status"] == "LABELED" and r["gold_topic_subtype"]:
                by_stage.setdefault(r["annotation_stage"], []).append((r["gold_topic_subtype"], r["comment_text"], lem.get(r["comment_id"], False)))
    hold_gold = json.loads((V2_HOLDOUT_DIR / "gold_labels.local.json").read_text())
    hold_map = json.loads((V2_HOLDOUT_DIR / "id_map.local.json").read_text())
    hold_txt = {json.loads(l)["eval_id"]: json.loads(l)["text"] for l in (V2_HOLDOUT_DIR / "sanitized_inputs.jsonl").read_text(encoding="utf-8").splitlines()}
    hold_rows = [(hold_gold[i]["topic"], hold_txt[i], lem.get(hold_map[i], False)) for i in hold_gold]

    L = ["# Candidate: WORD_SUBMISSION / ENTRY_CORRECTION boundary layer (development evidence only)", "",
         f"Spec sha256 (frozen): `{SPEC_SHA256}`. Zero learned parameters; `lemma` is local source metadata; nothing is sent anywhere.", "",
         "## 1. The rule alone as a WS-vs-EC resolver, on every development label source", "",
         "| label source | WS/EC rows | accuracy | majority baseline | EC recall | WS recall | balanced accuracy |", "|---|---|---|---|---|---|---|"]
    out = {"spec_sha256": SPEC_SHA256, "two_class": {}}
    sources = {"human_val (val)": by_stage.get("human_val", []), "human_train_seed": by_stage.get("human_train_seed", []),
               "ai_assisted_train": by_stage.get("ai_assisted_train", []), "holdout (ai_assisted, sanitized, ready)": hold_rows}
    for name, rows in sources.items():
        s = two_class(rows)
        out["two_class"][name] = s
        L.append(f"| {name} | {s['n']} ({s['n_ws']} WS / {s['n_ec']} EC) | {s['accuracy']:.2f} | {s['majority_baseline']:.2f} | "
                 f"{s['ec_recall']:.2f} | {s['ws_recall']:.2f} | {s['balanced_accuracy']:.2f} |")
    L.append("")

    # ---- before / after on the 94 validation records with real model outputs
    gold = json.loads((OUT_DIR / "gold_labels.local.json").read_text())
    idmap = json.loads((OUT_DIR / "id_map.local.json").read_text())
    txt = {json.loads(l)["eval_id"]: json.loads(l)["text"] for l in (OUT_DIR / "sanitized_inputs.jsonl").read_text(encoding="utf-8").splitlines()}
    rcs = json.loads((OUT_DIR / "rcs_predictions.json").read_text())
    jev = {json.loads(l)["eval_id"]: json.loads(l) for l in (OUT_DIR / "jev_predictions.jsonl").read_text().splitlines()}
    ids = list(gold)
    g = [gold[i]["topic"] for i in ids]
    L += ["## 2. Before / after on the 94 development-validation records (human labels; the same records for every row)", ""]
    labels = M.TASK_LABELS["topic"]
    results = {}
    for base_name, P in (("Frozen RCS", rcs), ("Jev v1", jev)):
        base = [P[i]["topic"] for i in ids]
        new = [layer(P[i]["topic"], txt[i], lem.get(idmap[i], False)) for i in ids]
        changed = [i for i, a, b in zip(ids, base, new) if a != b]
        assert all(P[i]["topic"] in (WS, EC) for i in changed), "layer touched a class outside WS/EC"
        (alo, ahi), (flo, fhi) = _paired_boot(g, base, new, labels)
        sb, sn = M.score_task("topic", g, base), M.score_task("topic", g, new)
        regress = [i for i, x, a, b in zip(ids, g, base, new) if a == x and b != x]
        gain = [i for i, x, a, b in zip(ids, g, base, new) if a != x and b == x]
        results[base_name] = {"base": sb, "new": sn, "regressions": regress, "gains": gain, "changed": len(changed),
                              "d_acc_ci": (alo, ahi), "d_f1_ci": (flo, fhi)}
        L += [f"### {base_name} -> {base_name} + layer", "",
              f"accuracy {sb['accuracy']:.3f} -> {sn['accuracy']:.3f} (paired 95% CI of the change {alo:+.2f}..{ahi:+.2f}); "
              f"macro-F1 {sb['macro_f1_supported_classes']:.3f} -> {sn['macro_f1_supported_classes']:.3f} ({flo:+.2f}..{fhi:+.2f}); "
              f"predictions changed on {len(changed)} of {len(ids)} records; gains {len(gain)}, regressions {len(regress)} "
              f"(only WS/EC predictions can change).", "",
              "| class | P before -> after | R before -> after | F1 before -> after | support |", "|---|---|---|---|---|"]
        for c in labels:
            a, b = sb["per_class"][c], sn["per_class"][c]
            if a["support"] or a["predicted"] or b["predicted"]:
                L.append(f"| {c} | {a['precision']:.2f} -> {b['precision']:.2f} | {a['recall']:.2f} -> {b['recall']:.2f} | {a['f1']:.2f} -> {b['f1']:.2f} | {a['support']} |")
        for nm, pred in (("before", base), ("after", new)):
            cm = M.confusion(g, pred, labels, labels)
            cols = list(labels) + ["OTHER"]
            L += ["", f"confusion {nm} (rows gold, cols pred: " + ", ".join(c[:6] for c in cols) + ")", "```"]
            for gl in labels:
                if any(cm[gl].values()):
                    L.append(f"{gl[:22]:22} " + " ".join(f"{cm[gl][c]:3d}" for c in cols))
            L.append("```")
        fpws = sum(1 for x, p in zip(g, new) if x != WS and p == WS); fnws = sum(1 for x, p in zip(g, new) if x == WS and p != WS)
        fpec = sum(1 for x, p in zip(g, new) if x != EC and p == EC); fnec = sum(1 for x, p in zip(g, new) if x == EC and p != EC)
        L += ["", f"after layer: WORD_SUBMISSION FP {fpws} / FN {fnws}; ENTRY_CORRECTION FP {fpec} / FN {fnec}.",
              f"regression ids (base right -> layer wrong): {', '.join(regress) if regress else 'none'}", ""]
    L += ["## 3. How to read this", "",
          "* Validation (94) was already used for earlier decisions and the rule's `lemma` signal was first noticed in counts over train+val, so the "
          "validation gain is development evidence, not independent accuracy. The only independent test is Gold Test v2.",
          "* The rule is shown against three label sources with different conventions (human val, human seed, AI-assisted); similar balanced accuracy "
          "across them is the generalisation evidence, and it is what to re-check first on Gold Test v2.",
          "* Frozen RCS is scored only on validation: it was trained on the other splits.", ""]
    (OUT_DIR / "candidate_ws_ec_layer_report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT_DIR / "candidate_ws_ec_layer_spec.json").write_text(json.dumps({"spec": SPEC, "sha256": SPEC_SHA256}, ensure_ascii=False, indent=1))
    out["validation"] = {k: {"acc_before": v["base"]["accuracy"], "acc_after": v["new"]["accuracy"], "f1_before": v["base"]["macro_f1_supported_classes"],
                             "f1_after": v["new"]["macro_f1_supported_classes"], "changed": v["changed"], "gains": len(v["gains"]),
                             "regressions": len(v["regressions"]), "d_acc_ci": v["d_acc_ci"], "d_f1_ci": v["d_f1_ci"]} for k, v in results.items()}
    (OUT_DIR / "candidate_ws_ec_layer.json").write_text(json.dumps(out, indent=1, default=str))
    for f in ("candidate_ws_ec_layer_report.md", "candidate_ws_ec_layer.json", "candidate_ws_ec_layer_spec.json"):
        (OUT_DIR / f).chmod(0o600)
    print(json.dumps({"spec_sha256": SPEC_SHA256, "two_class": out["two_class"], "validation": out["validation"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
