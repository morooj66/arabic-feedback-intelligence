"""One-pass real-data comparison: frozen classifiers vs Gemini vs two predefined hybrids on the two approved held-out sets (reported separately).

    python -m scripts.offline_eval.gemini_classification.compare_real                       # DRY RUN: counts and checks only, reads no comment text
    python -m scripts.offline_eval.gemini_classification.sanitized_inputs --dataset human_test   # (and blind_natural) build the inputs a person must read
    python -m scripts.offline_eval.gemini_classification.compare_real --approval FILE.json  # live run; FILE is the owner's written LiveApproval incl. the reviewed input hashes

The live run is refused unless the approval JSON names both datasets and positive call/cost caps, and unless the prompt still matches prompt_freeze.json.
Reports contain ids, metrics and counts only -- never comment text.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import real_data as rd
from .client import ClassifierClient, LiveGeminiClient
from .config import GenerationSettings
from .evaluate import GoldRow, evaluate_systems, predictions_from_outcomes, run_quality, stability
from .hybrids import all_systems
from .pricing import PricingTable
from .privacy import LiveApproval, RealDataNotApproved
from .prompt import PROMPT_VERSION, prompt_sha256, verify_frozen_prompt
from .runner import PRIVACY_BLOCKED, Budget, ClassificationRunner
from .sanitized_inputs import DEFAULT_DIR, load_verified
from .selection import select

OUT = Path(__file__).with_name("results")
REQUIRED_DATASETS = ("human_test", "blind_natural")
STABILITY_ROWS = 30
FROZEN_TOPIC_MACRO_F1_REPORTED = 0.16071428571428573      # FINAL_HUMAN_TEST_REPORT.md section C; local re-inference must reproduce it


def load_approval(path: Path) -> LiveApproval:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    a = LiveApproval(d["approver"], d["approved_on"], d["scope"], int(d["max_calls"]), float(d["max_usd"]), tuple(d["datasets"]),
                     tuple(sorted(dict(d.get("input_sha256", {})).items())))
    a.validate()
    if not str(d.get("legal_basis", "")).strip() or d.get("paid_tier_confirmed") is not True or d.get("independent_reviewer_read_inputs") is not True:
        raise RealDataNotApproved("approval must state the legal basis, confirm the Gemini project is on Paid Services, and confirm an independent person read the sanitised inputs")
    missing = [x for x in REQUIRED_DATASETS if x not in a.datasets or x not in dict(a.input_sha256)]
    if missing:
        raise RealDataNotApproved(f"approval does not cover: {missing}")
    return a


def dry_run() -> dict:
    return {"prompt_sha256": verify_frozen_prompt()[:16], "prompt_version": PROMPT_VERSION,
            "human_test": rd.counts_only("human_test"), "blind_natural": rd.counts_only("blind_natural"),
            "frozen_prediction_files_present": {
                "human_test_intent": (rd.FINAL_TEST / "intent_tfidf_logistic_test_preds.json").exists(),
                "human_test_sentiment": (rd.FINAL_TEST / "sentiment_camelbert_test_preds.json").exists(),
                "blind_all_tasks": (rd.BLIND / "results" / "rcs_predictions.json").exists()},
            "live_calls_planned": {"final": 113, "stability": STABILITY_ROWS}, "note": "dry run: no text read, no network"}


def frozen_topic_for(records) -> dict[str, str]:
    """Local inference with the frozen MARBERTv2 checkpoint (no network); verified against the reported sealed-test macro-F1 before use."""
    from rcs.classification.frozen_inference import predict_topic

    preds = predict_topic([r.record_id for r in records], [r.text for r in records])
    return {i: label for i, (label, _conf) in preds.items()}


def check_frozen_topic_reproduces(ht: "rd.EvalSet", tolerance: float = 0.02) -> float:
    """Local re-inference must reproduce the reported sealed-test topic macro-F1; otherwise the 'frozen' column is not the frozen system."""
    from . import metrics as m
    from . import taxonomy as tx

    gold = {g.record_id: g.topic for g in ht.gold}
    ids = [i for i in gold if i in ht.frozen]
    got = m.task_report(tx.TOPICS, [gold[i] for i in ids], [ht.frozen[i].topic for i in ids])["macro_f1_all_classes"]
    if abs(got - FROZEN_TOPIC_MACRO_F1_REPORTED) > tolerance:
        raise RuntimeError(f"frozen topic re-inference does not reproduce the reported macro-F1 ({got:.4f}); aborting before any Gemini call")
    return got


def run_live(approval: LiveApproval, *, client: ClassifierClient | None = None, sets: list[rd.EvalSet] | None = None,
             out_dir: Path = OUT, inputs_dir: Path = DEFAULT_DIR, pricing: PricingTable | None = None, sleep: Callable[[float], None] | None = None) -> dict:
    prompt_hash = verify_frozen_prompt()
    pricing = pricing or PricingTable()
    client = client or LiveGeminiClient(approval)
    if sets is None:
        probe = rd.load_human_test({}).records
        topic = frozen_topic_for(probe)
        ht = rd.load_human_test(topic)
        check_frozen_topic_reproduces(ht)
        sets = [ht, rd.load_blind_natural()]
    pre_blocked: dict[str, set[str]] = {}
    for i, s in enumerate(sets):
        if any(r.provenance == "real" for r in s.records):       # real text is replaced by the human-approved, hash-verified sanitised text; anything else is withheld
            expected = dict(approval.input_sha256).get(s.name)
            if not expected:
                raise RealDataNotApproved(f"no approved sanitised-input hash for {s.name!r}")
            texts = load_verified(s.name, expected, inputs_dir)
            pre_blocked[s.name] = {r.record_id for r in s.records if r.record_id not in texts}
            sets[i] = rd.EvalSet(s.name, [replace(r, text=texts[r.record_id]) for r in s.records if r.record_id in texts], s.gold, s.frozen)
    settings = GenerationSettings()
    budget = Budget(approval.max_calls, approval.max_usd)
    kw = {"sleep": sleep} if sleep else {}
    reports, qualities, gem_runs, per_set = {}, {}, {}, {}
    for s in sets:
        runner = ClassificationRunner(client, settings, budget=Budget(max(0, budget.max_calls - sum(q["_calls"] for q in qualities.values())),
                                                                          max(0.0, budget.max_usd - sum(q["total_cost_usd"] for q in qualities.values()))),
                                      pricing=pricing, approval=approval, **kw)
        run = runner.run(s.records)
        q = run_quality(run.outcomes)
        q["_calls"] = run.calls_made
        qualities[s.name] = q
        gem = predictions_from_outcomes(run.outcomes)
        gem_runs[s.name] = (runner, run, gem)
        blocked = {o.record_id for o in run.outcomes if o.status == PRIVACY_BLOCKED} | pre_blocked.get(s.name, set())
        gold = [replace(g, privacy_eligible=g.record_id not in blocked) for g in s.gold]
        systems = all_systems(s.frozen, gem)
        reports[s.name] = evaluate_systems(gold, systems, baseline="frozen", seed=11)
        per_set[s.name] = {"rows": len(s.records), "gemini_ok": len(gem), "privacy_blocked": len(blocked)}
    # stability: repeat a fixed slice of already-classified rows with the same prompt and seed
    stab = {}
    remaining = budget.max_calls - sum(r.calls_made for r, _, _ in gem_runs.values())
    n_rep = min(STABILITY_ROWS, remaining)
    if n_rep > 0:
        pool = [(s, r) for s in sets for r in s.records if r.record_id in gem_runs[s.name][2]]
        pool.sort(key=lambda x: x[1].record_id)
        chosen = pool[:n_rep]
        rep_runner = ClassificationRunner(client, settings, budget=Budget(n_rep + 5, approval.max_usd), pricing=pricing, approval=approval, **kw)
        again = predictions_from_outcomes(rep_runner.run([r for _, r in chosen]).outcomes)
        first = {r.record_id: gem_runs[s.name][2][r.record_id] for s, r in chosen}
        stab = stability(first, again)
        stab["calls"] = rep_runner.calls_made
    decision = select({k: v for k, v in reports.items()}, {k: {kk: vv for kk, vv in q.items() if kk != "_calls"} for k, q in qualities.items()})
    result = {"prompt_sha256": prompt_hash, "model": settings.model_id, "seed": settings.seed, "sets": per_set, "reports": reports,
              "run_quality": {k: {kk: vv for kk, vv in q.items() if kk != "_calls"} for k, q in qualities.items()}, "stability": stab, "decision": decision,
              "cost_per_1000_comments_usd": decision["reliability"]["cost_per_1000_usd"]}
    out_dir.mkdir(exist_ok=True)
    (out_dir / "real_comparison_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approval", type=Path)
    args = ap.parse_args(argv)
    if args.approval is None:
        print(json.dumps(dry_run(), ensure_ascii=False, indent=1))
        return 0
    res = run_live(load_approval(args.approval))
    print(json.dumps({"architecture": res["decision"]["architecture"], "tasks": {t: d["decision"] for t, d in res["decision"]["tasks"].items()},
                      "cost_per_1000_usd": res["cost_per_1000_comments_usd"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
