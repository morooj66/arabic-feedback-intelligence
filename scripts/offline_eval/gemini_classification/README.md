# Gemini classification evaluation harness (offline, candidate `gc-v1`)

Purpose: measure, on identical held-out rows, whether a Gemini classifier can beat the frozen classifiers for intent (5), topic (7) and sentiment (3),
**without** changing the taxonomy, the frozen models, the production pipeline or the response-only review workflow. Reviewers do not correct or
confirm classifications.

Status: **offline preparation only. No live call has been made. No real visitor text was opened or sent.**

| File | Role |
|---|---|
| `taxonomy.py` | approved 5/7/3 taxonomy read from `config.settings`, pinned and checked for drift |
| `schema.py` | structured-output schema + strict validation (exact keys, enums, boolean, reason rules) |
| `prompt.py` | prompt `gc-v1`: labeling-guide definitions and decision boundaries, zero-shot, five invented illustrations; classification only |
| `privacy.py` | production outbound privacy boundary (`sanitize_text`, `guard_prompt`) + `LiveApproval` gate for real data |
| `client.py` | `MockGeminiClient`, `ScriptedMockClient` (failure injection), `LiveGeminiClient` (needs approval; never executed) |
| `config.py` | generation settings: model `gemini-3.6-flash`, fixed seed, `thinking_level`, output cap; temperature left at the provider default |
| `pricing.py` / `pricing.json` | verified price sheet and token/cost accounting (thinking tokens billed as output) |
| `runner.py` | preflight -> egress guard -> call -> bounded retries -> strict validation -> accounting; call and cost caps; text-free call log |
| `metrics.py` | accuracy, macro-F1 (all classes and supported classes), per-class P/R/F1, confusion, Wilson, McNemar, paired bootstrap, routing, ambiguity |
| `evaluate.py` | identical-eligible-row comparison, failure-inclusive view, run quality (failure rate, latency, tokens, cost), stability |
| `synthetic_cases.py` | 54 invented comments with gold labels (incl. fictional PII and an injection attempt) |
| `run_offline_validation.py` | mock-only end-to-end run with all sockets blocked |
| `LIVE_EVALUATION_PROPOSAL.md` | staged live plan, verified model and pricing, cost bounds, required approvals |

Run: `python -m scripts.offline_eval.gemini_classification.run_offline_validation` and `python -m pytest tests/test_gemini_classification_eval.py`.

Guarantees (each covered by a test): no database or production import; production code never imports this package; no Google SDK import at
import time; every outbound prompt passes `guard_prompt`; real data is refused without a `LiveApproval`; call logs hold no comment text.
