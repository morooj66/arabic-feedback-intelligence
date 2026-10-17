"""Re-examine the quarantined holdout records under the context-aware name policy. LOCAL ONLY; prints ids, reasons and counts.

Recovery rule (fail closed): a record is RECOVERED when the policy returns text AND
  * it was quarantined only by a blunt heuristic / boilerplate false positive (Latin form labels, nasab-chain or ID-adjacency
    rules, redactor fail-closed) and the policy now handles the cause, OR
  * it was flagged by a human reviewer for a possible name, AND at least one identity mask was actually applied
    (otherwise nothing shows the identity context was handled, so it stays quarantined until a person releases it).
Sensitive personal circumstances and identity-only content are never recovered automatically.
The recovered texts are written to a separate, owner-only candidate file; the APPROVED inputs are not modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_name_policy as N  # noqa: E402
import prepare_inputs as P  # noqa: E402
from jev_common import HUMAN_TEST_EXCLUSION, OBSERVED_USD_PER_REQUEST, V2_HOLDOUT_DIR  # noqa: E402
from jev_protocol import build_request, estimate_cost  # noqa: E402


def decide(prev_reason: str, res: N.Result) -> tuple[str, str]:
    """-> (status, why). status is RECOVERED or STILL_QUARANTINED."""
    if res.text is None:
        return "STILL_QUARANTINED", f"policy:{res.quarantine_reason}"
    identity = [k for k in N.IDENTITY_MASK_ACTIONS if res.actions.get(k)]
    if prev_reason == "manual_review_possible_personal_data" and not identity:
        return "STILL_QUARANTINED", "human_flagged_name_but_no_identity_mask_applied"
    return "RECOVERED", ("identity_masked:" + "+".join(identity)) if identity else "false_positive_of_blunt_rule"


def main() -> None:
    sealed = P._human_test_ids(HUMAN_TEST_EXCLUSION)
    rows = P.select_rows(P.load_dev_rows(splits=("train",)), sealed, ("train",))
    by_eval = {f"tr-{i + 1:04d}": r for i, r in rows.iterrows()}
    q = json.loads((V2_HOLDOUT_DIR / "quarantined.json").read_text())
    recovered, still, gold, idmap = [], [], {}, {}
    why_counter, prev_counter = Counter(), Counter()
    for x in q:
        r = by_eval[x["eval_id"]]
        res = N.sanitize(str(r["comment_text"]))
        status, why = decide(x["reason"], res)
        if status == "RECOVERED":
            recovered.append({"eval_id": x["eval_id"], "text": res.text})
            gold[x["eval_id"]] = {"intent": r["gold_primary_intent"], "topic": r["gold_topic_subtype"], "sentiment": r["gold_sentiment"]}
            idmap[x["eval_id"]] = r["comment_id"]
            prev_counter[x["reason"]] += 1
            why_counter[why] += 1
        else:
            still.append({"eval_id": x["eval_id"], "previous_reason": x["reason"], "why_still_quarantined": why})
    d = V2_HOLDOUT_DIR
    inputs = d / "sanitized_inputs.recovered_candidate.jsonl"
    inputs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recovered), encoding="utf-8")
    (d / "gold_labels.recovered_candidate.local.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1))
    (d / "id_map.recovered_candidate.local.json").write_text(json.dumps(idmap, indent=1))
    (d / "still_quarantined_after_policy.json").write_text(json.dumps(still, indent=1))
    lines = ["# Recovered candidate records (sanitized text only) - REQUIRES independent human privacy review before any transmission", "",
             "| eval_id | sanitized text |", "|---|---|"]
    for r in recovered:
        lines.append(f"| {r['eval_id']} | {r['text'].replace('|', '/').replace(chr(10), ' ')} |")
    (d / "INSPECTION_recovered_candidate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for f in ("sanitized_inputs.recovered_candidate.jsonl", "gold_labels.recovered_candidate.local.json", "id_map.recovered_candidate.local.json",
              "still_quarantined_after_policy.json", "INSPECTION_recovered_candidate.md"):
        (d / f).chmod(0o600)
    digest = hashlib.sha256(inputs.read_bytes()).hexdigest()
    cost = {v: estimate_cost([build_request(r["text"], v) for r in recovered]) for v in ("v1", "v2")}
    summary = {
        "examined": len(q), "recovered": len(recovered), "still_quarantined": len(still),
        "recovered_by_previous_reason": dict(prev_counter), "recovered_because": dict(why_counter),
        "still_quarantined_because": dict(Counter(s["why_still_quarantined"] for s in still)),
        "still_quarantined_ids": [s["eval_id"] for s in still],
        "recovered_support": {t: dict(Counter(g[t] for g in gold.values())) for t in ("intent", "topic", "sentiment")},
        "candidate_inputs_sha256_NOT_APPROVED": digest,
        "if_approved_requests_per_version": len(recovered),
        "observed_rate_estimate_usd_per_version": round(OBSERVED_USD_PER_REQUEST * len(recovered), 4),
        "calls_executed": 0,
    }
    (d / "recovery_summary.json").write_text(json.dumps(summary, indent=1))
    (d / "recovery_summary.json").chmod(0o600)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
