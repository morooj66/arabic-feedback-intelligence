"""Pre-registered decision rule (fixed before any real-data run). Gemini replaces the frozen classifier for a task only if ALL hold:
  1. reliability: failure rate <= 2%, p95 latency <= 15 s, cost <= 5 USD per 1,000 comments;
  2. on EVERY held-out set the macro-F1 (supported classes) difference Gemini - frozen is >= 0 and > 0 on at least one;
  3. on at least one held-out set the paired-bootstrap 95% interval for that difference lies above 0;
  4. no class with support >= 5 loses more than 0.25 F1.
If (1)-(2) hold and (4) holds but (3) does not -> INCONCLUSIVE (recommend the smallest next action, not a new broad evaluation); otherwise FROZEN.
The architecture is then composed per task, and rejected for the intent+topic pair if it adds false exclusions on any held-out set."""
from __future__ import annotations

from typing import Mapping

from . import taxonomy as tx

THRESHOLDS = {"max_failure_rate": 0.02, "max_p95_latency_s": 15.0, "max_cost_per_1000_usd": 5.0, "min_class_support": 5, "max_class_f1_drop": 0.25}
KEY = "macro_f1_supported_classes"


def reliability(run_quality: Mapping[str, dict]) -> dict:
    worst_fail = max((q["schema_failure_rate"] + q["api_failure_rate"]) for q in run_quality.values())
    worst_p95 = max(q["latency_seconds"]["p95"] for q in run_quality.values())
    calls = sum(q["records"] for q in run_quality.values()) or 1
    cost_1000 = sum(q["total_cost_usd"] for q in run_quality.values()) / calls * 1000
    ok = worst_fail <= THRESHOLDS["max_failure_rate"] and worst_p95 <= THRESHOLDS["max_p95_latency_s"] and cost_1000 <= THRESHOLDS["max_cost_per_1000_usd"]
    return {"ok": ok, "failure_rate": worst_fail, "p95_latency_s": worst_p95, "cost_per_1000_usd": cost_1000}


def decide_task(task: str, reports: Mapping[str, dict], rel: dict, frozen: str = "frozen", gemini: str = "gemini") -> dict:
    """reports: held-out set name -> evaluate_systems() report (containing systems `frozen` and `gemini`)."""
    if not rel["ok"]:
        return {"task": task, "decision": "FROZEN", "reason": "gemini reliability/latency/cost thresholds not met"}
    diffs, significant, regress = {}, False, []
    for name, rep in reports.items():
        d = rep["systems"][gemini][task][KEY] - rep["systems"][frozen][task][KEY]
        diffs[name] = d
        ci = rep["paired_vs_baseline"][gemini][task][KEY]["ci95"]
        significant = significant or ci[0] > 0
        fs = {c["label"]: c for c in rep["systems"][frozen][task]["per_class"]}
        for c in rep["systems"][gemini][task]["per_class"]:
            f = fs[c["label"]]
            if f["support"] >= THRESHOLDS["min_class_support"] and f["f1"] - c["f1"] > THRESHOLDS["max_class_f1_drop"]:
                regress.append((name, c["label"]))
    if any(d < 0 for d in diffs.values()):
        return {"task": task, "decision": "FROZEN", "reason": "gemini is worse on at least one held-out set", "macro_f1_diffs": diffs}
    if all(d == 0 for d in diffs.values()):
        return {"task": task, "decision": "FROZEN", "reason": "no measured gain", "macro_f1_diffs": diffs}
    if regress:
        return {"task": task, "decision": "FROZEN", "reason": f"per-class regression {regress}", "macro_f1_diffs": diffs}
    if not significant:
        return {"task": task, "decision": "INCONCLUSIVE", "reason": "gemini is not worse anywhere but no paired interval excludes zero", "macro_f1_diffs": diffs}
    return {"task": task, "decision": "GEMINI", "reason": "meets all pre-registered criteria", "macro_f1_diffs": diffs}


def routing_guard(reports: Mapping[str, dict], candidate: str, frozen: str = "frozen") -> dict:
    extra = {n: r["systems"][candidate]["routing"]["false_exclusion"] - r["systems"][frozen]["routing"]["false_exclusion"] for n, r in reports.items()}
    return {"ok": all(v <= 0 for v in extra.values()), "extra_false_exclusions_vs_frozen": extra}


_PAIR_SYSTEM = {("frozen", "frozen"): "frozen", ("gemini", "gemini"): "hybrid_intent_topic_gemini", ("frozen", "gemini"): "hybrid_topic_gemini"}


def select(reports: Mapping[str, dict], run_quality: Mapping[str, dict]) -> dict:
    from .hybrids import SYSTEMS

    rel = reliability(run_quality)
    tasks = {t: decide_task(t, reports, rel) for t in tx.TASKS}
    winners = {t: ("gemini" if d["decision"] == "GEMINI" else "frozen") for t, d in tasks.items()}
    notes = []
    if (winners["intent"], winners["topic"]) == ("gemini", "frozen"):       # routing for this pair was not measured jointly
        winners["intent"] = "frozen"
        notes.append("gemini intent with frozen topic was not evaluated jointly; intent kept frozen")
    pair_system = _PAIR_SYSTEM[(winners["intent"], winners["topic"])]
    guard = routing_guard(reports, pair_system) if pair_system != "frozen" else {"ok": True, "extra_false_exclusions_vs_frozen": {}}
    if not guard["ok"]:
        winners["intent"] = winners["topic"] = "frozen"
        notes.append("intent/topic reverted to frozen: the Gemini pair added false exclusions")
    sources = {t: winners[t] for t in tx.TASKS}
    arch = next((n for n, s in SYSTEMS.items() if s == sources), "custom:" + "+".join(f"{t}={sources[t]}" for t in tx.TASKS))
    return {"reliability": rel, "tasks": tasks, "winners": winners, "architecture": arch, "routing_guard": guard, "notes": notes,
            "inconclusive_tasks": [t for t, d in tasks.items() if d["decision"] == "INCONCLUSIVE"]}
