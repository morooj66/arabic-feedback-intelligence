# Synthetic-data demo mode (architecture and operation)

A minimal, isolated mode for an **online demonstration with synthetic data only**. It keeps the complete admin/reviewer workflow and the original UI, and replaces only the parts that would otherwise need restricted artifacts or external services.

## What changes (all in `src/rcs/demo/`, applied at start-up)
| Normal behaviour | In demo mode |
|---|---|
| Frozen classifiers (torch / transformers / scikit-learn, ~1.1 GB of weights) | `rcs.demo.classifier`: deterministic keyword rules; provenance is stored as `DEMO_RULES_SYNTHETIC_ONLY` so a demo row can never pass for a model prediction |
| LLM (Gemini) draft generation | `rcs.demo.provider.DemoProvider`: local, deterministic, five short generic Arabic replies; every reply passes the real validation gates (tested) |
| Workbook upload accepts any valid workbook | only workbooks carrying the synthetic marker (document title `RCS-SYNTHETIC-DEMO-WORKBOOK-v1`), ≤ 200 rows, ≤ 256 KB |
| Account requests, password reset, Microsoft sign-in | disabled (invite-only: the operator creates accounts) |
| Admin inspector entries that load the models | answer "not available in demo mode" |
| Network | every Python-level connection to a non-loopback address is refused; torch, transformers, scikit-learn, joblib, huggingface_hub and the Gemini/OpenAI SDKs can no longer be imported |
**Unchanged:** login, sessions, roles, backend authorization, reviewer isolation, CSRF/origin checks, rate limits, assignment, review, messaging, audit, export, and the UI (the only addition is a small "synthetic demo" pill shown when `/api/health` reports `demo: true`).

## Why it cannot be switched on by accident (`rcs.demo.guard`)
Demo mode needs **all** of: `RCS_DEMO_MODE=synthetic-demo` · `RCS_DEMO_ACK=THIS_INSTANCE_HOLDS_SYNTHETIC_DATA_ONLY` · a database whose name (PostgreSQL) or file stem (SQLite) starts with `rcs_demo` · **none** of the production secrets in the environment (Gemini, OpenRouter, MongoDB, SMTP, Entra, Hugging Face token/model repo) · no development database flags.
- A production deployment sets none of these, so the demo package is never loaded (`demo.state.is_active()` stays `False`; a test asserts it).
- A process with *some* `RCS_DEMO_*` variable but an incomplete or unsafe configuration **refuses to start** with the list of reasons; it never falls back silently (tests cover each failing ingredient).
- A demo process never reads `.env`.
- Pointing a production environment (secrets present, database not named `rcs_demo*`) at demo mode is impossible even if someone sets the two variables.

## Operator steps
1. Create an **empty, dedicated** PostgreSQL database named `rcs_demo_…` (no real data, nothing else in it).
2. On your own machine: `DATABASE_URL=postgresql://…/rcs_demo_online python scripts/demo/init_demo_environment.py --admin-email … --reviewer-email … --credentials-out ./demo_credentials.json` — runs the migrations, creates the accounts with strong random passwords written **only** to that file (mode 600; never printed or committed), and imports a synthetic workbook through the real pipeline. `--reset` rebuilds the demo database from scratch.
3. Hand each invitee their password through a private channel. There are no default accounts; the file is the only record.
4. Deploy the backend container (`Dockerfile.demo`, `render.demo.yaml`) and the Vercel frontend — see [DEPLOY_DEMO.md](DEPLOY_DEMO.md).

## Verified
`tests/test_demo_mode.py` (guard matrix, stand-ins through the real pipeline, activation in a subprocess, refusal to start) and `scripts/e2e/verify_demo_mode.py` (real browser: login hardening, upload policy, import, assignment, reviewer isolation and decisions, messaging, completed export), including a run from a virtualenv **without torch** and with the ML/LLM libraries made unimportable.
**Not verified here:** a Docker build of `Dockerfile.demo`, PostgreSQL (no local server; the migrations and the demo database on PostgreSQL have not been run), and any hosted deployment.
