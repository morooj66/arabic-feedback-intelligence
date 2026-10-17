"""End-to-end offline validation with the MOCK client on SYNTHETIC comments only.

    python -m scripts.offline_eval.gemini_classification.run_offline_validation

Blocks every socket connection during the run (and counts attempts), makes no database write, and writes only
scripts/offline_eval/gemini_classification/results/offline_validation_report.json (git-ignored; no comment text).
There is deliberately no --live option: live evaluation needs a written LiveApproval (see privacy.py) and a separate, explicitly approved script run.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

from . import synthetic_cases as sc
from .client import MockGeminiClient
from .config import GenerationSettings
from .evaluate import evaluate_systems, predictions_from_outcomes, run_quality, stability
from .pricing import PricingTable
from .prompt import PROMPT_VERSION, SYSTEM_INSTRUCTION
from .runner import Budget, ClassificationRunner

OUT = Path(__file__).with_name("results")


class NetworkBlocked(RuntimeError):
    pass


def main() -> int:
    attempts = []
    real_connect = socket.socket.connect

    def blocked(self, address):  # any attempted connection is recorded and refused
        attempts.append(str(address))
        raise NetworkBlocked("network access is not allowed in the offline validation")

    socket.socket.connect = blocked
    try:
        records, gold = sc.records(), sc.gold_rows()
        client = MockGeminiClient()
        runner = ClassificationRunner(client, GenerationSettings(), budget=Budget(max_calls=500, max_usd=5.0), pricing=PricingTable())
        run1 = runner.run(records)
        run2 = ClassificationRunner(MockGeminiClient(), GenerationSettings(), budget=Budget(500, 5.0)).run(records)
        gemini = predictions_from_outcomes(run1.outcomes)
        baseline = sc.degrade(gold, flip_rate=0.35, seed=7)
        report = {
            "prompt_version": PROMPT_VERSION, "prompt_characters": len(SYSTEM_INSTRUCTION),
            "run_quality": run_quality(run1.outcomes), "stability_mock_repeat": stability(gemini, predictions_from_outcomes(run2.outcomes)),
            "comparison": evaluate_systems(gold, {"simulated_baseline": baseline, "mock_gemini": gemini}, baseline="simulated_baseline", n_resamples=500, seed=1),
            "network_connection_attempts": len(attempts), "mock_calls": client.calls,
            "note": "Synthetic comments and a keyword mock: these numbers validate the harness only and say nothing about Gemini or the frozen classifiers.",
        }
    finally:
        socket.socket.connect = real_connect
    OUT.mkdir(exist_ok=True)
    (OUT / "offline_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("prompt_version", "mock_calls", "network_connection_attempts")}, ensure_ascii=False))
    print("run quality:", json.dumps(report["run_quality"]["status_counts"]), "| eligible rows:", report["comparison"]["eligible_rows"], "of", report["comparison"]["total_gold_rows"])
    return 0 if not attempts else 1


if __name__ == "__main__":
    sys.exit(main())
