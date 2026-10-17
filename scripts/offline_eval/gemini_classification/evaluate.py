"""Fair multi-system comparison. Every system is scored on the IDENTICAL set of eligible rows:
    eligible = (not blocked by the privacy preflight) AND (every compared system produced a valid prediction for it).
A failure-inclusive ("strict") view is reported as well, in which a missing prediction counts as wrong, so a system cannot look better by failing on hard rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rcs.routing.engine import determine_routing

from . import metrics as m
from . import taxonomy as tx
from .runner import OK, CallOutcome


@dataclass(frozen=True)
class GoldRow:
    record_id: str
    intent: str
    topic: str
    sentiment: str
    ambiguous: bool = False         # gold "insufficient context" style flag (used only for the ambiguity report)
    privacy_eligible: bool = True


@dataclass(frozen=True)
class Prediction:
    intent: str
    topic: str
    sentiment: str
    ambiguous: bool = False
    confidence: str = ""


def routing_for(intent: str, topic: str) -> str:
    """Exactly how the production Excel pipeline derives routing from labels (run_frozen_classification)."""
    return determine_routing(primary_intent=intent, topic_subtype=topic, subcategory=None, secondary_intent=None, is_content_meaningful=True).routing_decision


def eligible_ids(gold: Sequence[GoldRow], systems: Mapping[str, Mapping[str, Prediction]]) -> list[str]:
    ids = []
    for g in gold:
        if g.privacy_eligible and all(g.record_id in preds for preds in systems.values()):
            ids.append(g.record_id)
    return ids


def evaluate_systems(gold: Sequence[GoldRow], systems: Mapping[str, Mapping[str, Prediction]], *, baseline: str, n_resamples: int = 2000, seed: int = 0) -> dict:
    if baseline not in systems:
        raise ValueError("baseline system missing")
    by_id = {g.record_id: g for g in gold}
    ids = eligible_ids(gold, systems)
    report: dict = {"eligible_rows": len(ids), "total_gold_rows": len(gold), "excluded_rows": len(gold) - len(ids), "systems": {}, "paired_vs_baseline": {}}

    def column(task: str, name: str, rows: Sequence[str]):
        return [getattr(systems[name][i], task) for i in rows]

    gold_cols = {t: [getattr(by_id[i], t) for i in ids] for t in tx.TASKS}
    gold_routing = [routing_for(by_id[i].intent, by_id[i].topic) for i in ids]
    for name in systems:
        rep = {t: m.task_report(tx.LABELS[t], gold_cols[t], column(t, name, ids)) for t in tx.TASKS}
        pred_routing = [routing_for(systems[name][i].intent, systems[name][i].topic) for i in ids]
        rep["routing"] = m.routing_report(gold_routing, pred_routing)
        rep["ambiguity"] = m.ambiguity_report([by_id[i].ambiguous for i in ids], [systems[name][i].ambiguous for i in ids])
        report["systems"][name] = rep
    for name in systems:
        if name == baseline:
            continue
        report["paired_vs_baseline"][name] = {
            t: m.paired_bootstrap(tx.LABELS[t], gold_cols[t], column(t, baseline, ids), column(t, name, ids), n_resamples=n_resamples, seed=seed)
            for t in tx.TASKS
        }
    # failure-inclusive view: a missing prediction counts as wrong
    strict = {}
    usable = [g for g in gold if g.privacy_eligible]
    for name, preds in systems.items():
        strict[name] = {t: sum(1 for g in usable if g.record_id in preds and getattr(preds[g.record_id], t) == getattr(g, t)) / max(1, len(usable)) for t in tx.TASKS}
        strict[name]["missing_predictions"] = sum(1 for g in usable if g.record_id not in preds)
    report["failure_inclusive_accuracy"] = strict
    return report


def run_quality(outcomes: Sequence[CallOutcome]) -> dict:
    """Reliability, latency and cost of one Gemini run (the numbers the recommendation needs next to accuracy)."""
    n = len(outcomes)
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    lat = sorted(o.latency_seconds for o in outcomes if o.status == OK) or [0.0]

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(p * len(lat)))]

    attempted = [o for o in outcomes if o.attempts]
    return {
        "records": n, "status_counts": counts,
        "schema_failure_rate": counts.get("SCHEMA_FAILURE", 0) / max(1, len(attempted)),
        "api_failure_rate": counts.get("API_FAILURE", 0) / max(1, len(attempted)),
        "privacy_blocked": counts.get("PRIVACY_BLOCKED", 0),
        "latency_seconds": {"p50": pct(0.5), "p95": pct(0.95), "max": lat[-1], "mean": sum(lat) / len(lat)},
        "input_tokens": sum(o.usage.input_tokens for o in outcomes), "output_tokens": sum(o.usage.output_tokens for o in outcomes),
        "thought_tokens": sum(o.usage.thought_tokens for o in outcomes), "total_cost_usd": sum(o.cost_usd for o in outcomes),
        "mean_attempts": sum(o.attempts for o in attempted) / max(1, len(attempted)),
    }


def stability(run_a: Mapping[str, Prediction], run_b: Mapping[str, Prediction]) -> dict:
    """Agreement between two repeat runs of the SAME prompt/seed on the same rows (the 'stability' stage)."""
    ids = sorted(set(run_a) & set(run_b))
    if not ids:
        return {"rows": 0}
    out = {"rows": len(ids)}
    for t in tx.TASKS:
        out[f"{t}_agreement"] = sum(getattr(run_a[i], t) == getattr(run_b[i], t) for i in ids) / len(ids)
    out["all_three_agreement"] = sum(all(getattr(run_a[i], t) == getattr(run_b[i], t) for t in tx.TASKS) for i in ids) / len(ids)
    return out


def predictions_from_outcomes(outcomes: Sequence[CallOutcome]) -> dict[str, Prediction]:
    return {o.record_id: Prediction(o.prediction.intent, o.prediction.topic, o.prediction.sentiment, o.prediction.ambiguous, o.prediction.confidence)
            for o in outcomes if o.status == OK and o.prediction is not None}
