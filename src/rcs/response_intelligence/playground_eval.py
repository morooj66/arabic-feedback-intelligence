"""
Playground evaluation rules: expected-vs-actual comparison, PASS/FAIL, and
metrics. Pure functions, no I/O.

Principles:
  * An expected label is ground truth ONLY when its case status is VERIFIED
    (an explicit, separate action). UNVERIFIED cases are never scored.
  * NEEDS_HUMAN_REVIEW cases are ambiguous by definition and never scored.
  * PASS/FAIL exists only for VERIFIED cases, and only over the fields that
    have an expected label; unset fields are "not evaluated".
  * Aggregate metrics are withheld when there are too few verified labels,
    and sample sizes are always reported. Playground cases are hand-picked,
    so metrics describe THOSE cases, not production accuracy.
"""

from __future__ import annotations

from typing import Any

from config import settings

STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_VERIFIED = "VERIFIED"
STATUS_NEEDS_REVIEW = "NEEDS_HUMAN_REVIEW"
CASE_STATUSES = (STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_NEEDS_REVIEW)

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_NOT_EVALUATED = "NOT_EVALUATED"
VERDICT_NEEDS_REVIEW = "NEEDS_HUMAN_REVIEW"
VERDICT_ERROR = "ERROR"

MIN_LABELED_FOR_AGGREGATES = 10
LOW_SUPPORT = 5

TASKS = ("intent", "topic", "sentiment")
EXPECTED_FIELDS = ("intent", "topic", "sentiment", "routing", "response_decision")


def _allowed(field: str) -> tuple[str, ...]:
    return {
        "intent": settings.PRIMARY_INTENTS,
        "topic": settings.TOPIC_SUBTYPES,
        "sentiment": settings.SENTIMENTS,
        "routing": settings.ROUTING_DECISIONS,
        "response_decision": settings.RESPONSE_ELIGIBILITY_DECISIONS,
    }[field]


def validate_expected(expected: dict[str, Any] | None) -> dict[str, str]:
    expected = expected or {}
    unknown = set(expected) - set(EXPECTED_FIELDS)
    if unknown:
        raise ValueError(f"unknown expected fields: {sorted(unknown)}")
    out = {}
    for field in EXPECTED_FIELDS:
        v = expected.get(field)
        if v in (None, ""):
            continue
        if v not in _allowed(field):
            raise ValueError(f"invalid expected {field} {v!r}; allowed: {list(_allowed(field))}")
        out[field] = v
    return out


def validate_status(status: str | None) -> str:
    status = status or STATUS_UNVERIFIED
    if status not in CASE_STATUSES:
        raise ValueError(f"invalid status {status!r}; allowed: {list(CASE_STATUSES)}")
    return status


def actual_values(variant: dict[str, Any]) -> dict[str, Any]:
    """Extract the actual model/agent outputs from one variant result.
    response_decision comes from the Agent run that used the classifier's own
    predictions (the fixed-label run isolates the Agent and is never used)."""
    cls = variant.get("classification") or {}
    out: dict[str, Any] = {}
    for task in TASKS:
        item = cls.get(task)
        out[task] = item["label"] if item else None
    routing = variant.get("routing")
    out["routing"] = routing["decision"] if routing else None
    run = (variant.get("agent_runs") or {}).get("predicted")
    decision = (run or {}).get("decision")
    out["response_decision"] = decision["routing_decision"] if decision else None
    return out


def compare(expected: dict[str, str], status: str, variant: dict[str, Any]) -> dict[str, Any]:
    expected = validate_expected(expected)
    status = validate_status(status)
    actual = actual_values(variant)
    if status == STATUS_NEEDS_REVIEW:
        return {"verdict": VERDICT_NEEDS_REVIEW, "status": status, "fields": {}, "reason": "ambiguous example: never scored"}
    if status != STATUS_VERIFIED:
        return {"verdict": VERDICT_NOT_EVALUATED, "status": status, "fields": {}, "reason": "expected labels are not verified"}
    if not expected:
        return {"verdict": VERDICT_NOT_EVALUATED, "status": status, "fields": {}, "reason": "no expected labels entered"}
    fields = {}
    evaluated = matched = 0
    errored = False
    for field, exp in expected.items():
        act = actual.get(field)
        if act is None:
            fields[field] = {"expected": exp, "actual": None, "evaluated": False, "match": None, "reason": "no actual value (error, skipped, or mode does not produce it)"}
            errored = True
            continue
        ok = act == exp
        fields[field] = {"expected": exp, "actual": act, "evaluated": True, "match": ok}
        evaluated += 1
        matched += int(ok)
    if evaluated == 0:
        verdict = VERDICT_ERROR
    elif evaluated and matched < evaluated:
        verdict = VERDICT_FAIL
    elif errored:
        verdict = VERDICT_ERROR  # something expected could not be checked: never claim PASS
    else:
        verdict = VERDICT_PASS
    return {"verdict": verdict, "status": status, "fields": fields, "evaluated_fields": evaluated, "matched_fields": matched}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(pairs: list[tuple[str, str]], task: str) -> dict[str, Any]:
    """pairs: (expected, predicted) over VERIFIED cases that have both."""
    labels = list(_allowed(task))
    n = len(pairs)
    idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for exp, act in pairs:
        cm[idx[exp]][idx[act]] += 1
    support = [sum(r) for r in cm]
    predicted = [sum(cm[r][c] for r in range(len(labels))) for c in range(len(labels))]
    per_class = []
    for i, l in enumerate(labels):
        tp = cm[i][i]
        p = tp / predicted[i] if predicted[i] else None
        r = tp / support[i] if support[i] else None
        f1 = None
        if support[i]:
            f1 = 0.0 if (p is None or (p + r) == 0) else 2 * p * r / (p + r)
        per_class.append(
            {"label": l, "support": support[i], "predicted": predicted[i], "precision": p, "recall": r, "f1": f1, "low_support": 0 < support[i] < LOW_SUPPORT}
        )
    supported = [c for c in per_class if c["support"] > 0]
    result: dict[str, Any] = {
        "task": task,
        "n": n,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm, "rows": "expected (verified)", "columns": "predicted (model)"},
        "supported_classes": len(supported),
        "sufficient": False,
        "accuracy": None,
        "macro_f1_supported_classes": None,
        "weighted_f1": None,
        "withheld_reason": None,
        "caveat": "Hand-picked playground cases: describes these cases only, not production accuracy.",
    }
    if n < MIN_LABELED_FOR_AGGREGATES:
        result["withheld_reason"] = f"only {n} verified labeled case(s); need at least {MIN_LABELED_FOR_AGGREGATES} before any aggregate score is shown"
        return result
    if len(supported) < 2:
        result["withheld_reason"] = "verified cases cover a single class; macro/weighted scores are not meaningful"
        return result
    result["sufficient"] = True
    result["accuracy"] = sum(cm[i][i] for i in range(len(labels))) / n
    result["macro_f1_supported_classes"] = sum(c["f1"] for c in supported) / len(supported)
    result["weighted_f1"] = sum(c["f1"] * c["support"] for c in supported) / n
    return result


def compute_evaluation(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """cases: latest saved version of each case (see playground_store)."""
    verified = [c for c in cases if c.get("status") == STATUS_VERIFIED]
    out: dict[str, Any] = {
        "case_counts": {
            "total": len(cases),
            "verified": len(verified),
            "unverified": sum(c.get("status") == STATUS_UNVERIFIED for c in cases),
            "needs_human_review": sum(c.get("status") == STATUS_NEEDS_REVIEW for c in cases),
        },
        "classification": {},
        "routing": None,
        "agent": {},
        "case_verdicts": [],
        "notes": [
            "Only VERIFIED cases count. UNVERIFIED and NEEDS_HUMAN_REVIEW cases are excluded from every metric.",
            "Mock Agent results are placeholders; they validate plumbing and routing only, never response quality.",
        ],
    }
    for task in TASKS:
        pairs = []
        for c in verified:
            exp = (c.get("expected") or {}).get(task)
            act = (c.get("result") or {}).get("actual", {}).get(task)
            if exp and act:
                pairs.append((exp, act))
        out["classification"][task] = classification_metrics(pairs, task)
    # classification routing accuracy (simple count; no per-class claims)
    rp = [((c["expected"]["routing"]), c["result"]["actual"].get("routing")) for c in verified if (c.get("expected") or {}).get("routing") and (c.get("result") or {}).get("actual", {}).get("routing")]
    out["routing"] = {"n": len(rp), "correct": sum(e == a for e, a in rp), "accuracy": (sum(e == a for e, a in rp) / len(rp)) if len(rp) >= MIN_LABELED_FOR_AGGREGATES else None,
                      "withheld_reason": None if len(rp) >= MIN_LABELED_FOR_AGGREGATES else f"only {len(rp)} verified routing case(s); need {MIN_LABELED_FOR_AGGREGATES}"}
    # Agent: response-decision agreement, kept strictly separate for mock and live
    for mode in ("mock", "live"):
        rows = [c for c in verified if (c.get("expected") or {}).get("response_decision") and (c.get("result") or {}).get("agent_mode") == mode and c["result"]["actual"].get("response_decision")]
        out["agent"][mode] = {
            "n": len(rows),
            "correct": sum(c["expected"]["response_decision"] == c["result"]["actual"]["response_decision"] for c in rows),
            "note": "plumbing/routing check only; says nothing about generated response quality" if mode == "mock" else "decision agreement only; response quality still needs human review",
        }
    for c in cases:
        out["case_verdicts"].append({"case_id": c["case_id"], "status": c.get("status"), "verdict": (c.get("verdict") or {}).get("verdict")})
    return out
