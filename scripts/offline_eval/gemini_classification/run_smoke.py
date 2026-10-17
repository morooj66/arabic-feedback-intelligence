"""Live smoke test: at most 10 Gemini calls on SYNTHETIC comments only (authorised 2026-10-16). Real visitor text is never loaded here.

    python -m scripts.offline_eval.gemini_classification.run_smoke

Reads only GEMINI_API_KEY from the project's .env (never printed), refuses anything but synthetic records, enforces a hard cap of 10 calls
(retries included) and 0.25 USD, stops after two consecutive failures, and writes a text-free report to results/smoke_report.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import synthetic_cases as sc
from .client import LiveGeminiClient
from .config import GenerationSettings

from .pricing import PricingTable
from .privacy import LiveApproval
from .prompt import PROMPT_VERSION
from .runner import OK, Budget, ClassificationRunner

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("results")
SMOKE_IDS = ("s001", "s006", "s012", "s016", "s020", "s026", "s031", "s037", "s042", "s050")   # one or more per class, a bare word and an injection attempt
MAX_CALLS, MAX_USD = 10, 0.25


def read_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "GEMINI_API_KEY":
            return v.strip().strip("'\"")
    raise SystemExit("GEMINI_API_KEY not found in .env")


def main() -> int:
    records = [r for r in sc.records() if r.record_id in SMOKE_IDS]
    assert len(records) == 10 and all(r.provenance == "synthetic" for r in records)
    approval = LiveApproval("project owner (written instruction in session, 2026-10-16)", "2026-10-16", "synthetic smoke test only", MAX_CALLS, MAX_USD, ("synthetic_v1",))
    client = LiveGeminiClient(approval, api_key=read_key(), timeout_seconds=60.0)
    runner = ClassificationRunner(client, GenerationSettings(), budget=Budget(MAX_CALLS, MAX_USD), pricing=PricingTable(), approval=approval)
    gold = {g.record_id: g for g in sc.gold_rows()}
    rows, consecutive_failures = [], 0
    for rec in records:
        out = runner.classify_one(rec)
        consecutive_failures = 0 if out.status == OK else consecutive_failures + 1
        g = gold[rec.record_id]
        match = None if out.prediction is None else {t: getattr(out.prediction, t) == getattr(g, t) for t in ("intent", "topic", "sentiment")}
        rows.append({"id": rec.record_id, "status": out.status, "failure_code": out.failure_code, "attempts": out.attempts, "latency_s": round(out.latency_seconds, 2),
                     "input_tokens": out.usage.input_tokens, "output_tokens": out.usage.output_tokens, "thought_tokens": out.usage.thought_tokens,
                     "cost_usd": round(out.cost_usd, 6), "label_matches_synthetic_gold": match,
                     "ambiguous": getattr(out.prediction, "ambiguous", None), "confidence": getattr(out.prediction, "confidence", None)})
        if consecutive_failures >= 2 or runner.calls_made >= MAX_CALLS:
            break
    report = {"prompt_version": PROMPT_VERSION, "model": GenerationSettings().model_id, "calls_made": runner.calls_made, "total_cost_usd": round(runner.total_cost, 6),
              "rows": rows, "ok": sum(r["status"] == OK for r in rows), "attempted": len(rows)}
    OUT.mkdir(exist_ok=True)
    (OUT / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "calls_made", "total_cost_usd", "ok", "attempted")}))
    for r in rows:
        print(r["id"], r["status"], r["failure_code"] or "", f"{r['latency_s']}s", f"in={r['input_tokens']} out={r['output_tokens']} think={r['thought_tokens']}", r["label_matches_synthetic_gold"])
    return 0 if report["ok"] == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
