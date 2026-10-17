# Live evaluation: status after the synthetic smoke test

**2026-10-16 smoke test (10 live calls, synthetic comments only): 10/10 valid structured responses, 0 retries, mean latency 3.4 s, measured about 1,195 input and 70 output tokens per call, 0 thinking tokens (thinking_level=minimal), total cost 0.0115 USD (about 1.15 USD per 1,000 comments).** The prompt was not changed afterwards and is frozen by hash (`prompt_freeze.json`). The remaining stage is ONE real-data pass (`compare_real.py`, 113 + 30 calls, expected cost about 0.2 USD) and needs `APPROVAL_TEMPLATE.json` completed by the owner. The older staged plan below is superseded: no development-set iteration is planned.

---
# (superseded) earlier staged plan

## Verified model and capabilities (official Google documentation, read 2026-10-16)
| Item | Finding | Source |
|---|---|---|
| Model ID | `gemini-3.6-flash`, listed as **Stable** (also the response-generation model already configured in `config/settings.py`). `gemini-3.8-flash` is newer; it is not used because the comparison should use the model the system already runs. | ai.google.dev/gemini-api/docs/models |
| Structured output | Supported. Interactions API `response_format={"type":"text","mime_type":"application/json","schema":...}`; `enum`, `required`, `additionalProperties`, booleans are supported; `maxLength` is not listed, so the 200-character reason limit is enforced locally. Google says to always validate values in the application. | .../structured-output, .../models/gemini-3.6-flash |
| Seed | `seed` (integer) is a documented `generation_config` field and exists in the installed SDK (`google-genai` 2.18.1). Reproducibility is best-effort, so a stability stage is included. | .../api/interactions-api, SDK source |
| Temperature | Google recommends keeping Gemini 3 at the default 1.0 and warns lower values can degrade results; the installed Interactions `GenerationConfig` has no temperature field. It is therefore **not set**. | docs search result; SDK source |
| Thinking | `thinking_level` in `generation_config`; `gemini-3.6-flash` accepts minimal/low/medium/high, default medium. Thinking tokens are **billed as output**. Plan uses `minimal`. | .../thinking |
| Token usage fields | `usage.total_input_tokens`, `total_output_tokens`, `total_thought_tokens` (used by the cost accounting). | .../api/interactions-api, SDK source |
| Limits | 1,048,576 input / 65,536 output tokens. | .../models/gemini-3.6-flash |

## Verified pricing (USD per 1M tokens, standard tier)
| Period | Input | Output (includes thinking) |
|---|---|---|
| through 2026-12-31 | **0.75** | **3.75** |
| from 2027-01-01 (announced) | 1.50 | 7.50 |
Batch API is half price but asynchronous (not used). **Caveat:** the figures were read by a page summarizer from https://ai.google.dev/gemini-api/docs/pricing; re-read the page immediately before approval and update `pricing.json` if it differs.
I could not list models with an API key (that would be an API call), so key access to `gemini-3.6-flash` is confirmed only by the first smoke call.

## Staged plan (261 calls)
| Stage | Calls | Data | Purpose and gate |
|---|---|---|---|
| 1. Smoke | 10 | **synthetic** | Confirm access, schema adherence, usage fields, latency, determinism. Proceed only if 10/10 valid. Needs API approval only. |
| 2. Development | up to 108 | human validation split (real, sanitised) | Refine the prompt. Stop early if schema-failure rate > 5% or budget reached. Needs privacy approval. |
| 3. Final | 113 | human test (68) and natural blind set (45), **reported separately** | Prompt frozen by hash first; one pass. Frozen-classifier predictions are scored on the identical eligible rows. |
| 4. Stability | 30 | 30 rows repeated from stage 3 | Same prompt and seed; measures run-to-run agreement. |
Not in the 261: the 30 rare-class diagnostic calls (worst case +$0.125). **Decision needed:** include them?

## Cost bound (rows x worst-case call)
Assumptions: input 3,000 tokens/call (prompt ~2,030 estimated + comment up to ~1,000; the estimator is conservative), output capped at 512 tokens including thinking.
| Case | USD |
|---|---|
| Worst case, 261 calls, price through 2026-12-31 | **1.09** |
| Worst case if every call needed the maximum 3 attempts | 3.27 |
| Same worst case at the 2027 price | 2.18 |
| Typical expected (2,300 in / 120 out, one attempt) | about 0.57 |
**Proposed hard caps:** 300 calls and **5 USD**, enforced by the runner (it stops and records BUDGET_STOPPED).

## Remaining privacy requirements (all unmet today)
1. Written `LiveApproval` (approver, date, scope, datasets, call and cost caps) for sending sanitised real comments; stage 1 uses synthetic text only.
2. Every prompt passes the production outbound privacy guard; rows it blocks are removed from **all** compared systems.
3. Reviewed privacy checklist for an external processor (same standard as the earlier Gemini pilots); no raw or redacted comment text in logs, reports or chat.
4. The prompt is frozen (hash recorded) before any held-out row is sent; held-out rows are never used to edit it. Development uses only the validation split.
5. API key supplied through the environment only, never printed.

## Approval requested
Approve (a) the model and caps above, (b) stage 1 (10 synthetic calls) now, (c) stages 2-4 after a separate privacy approval naming the datasets, and (d) whether the 30 rare-class diagnostic calls are included.
