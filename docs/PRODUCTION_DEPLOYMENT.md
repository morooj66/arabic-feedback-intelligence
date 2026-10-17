# Production deployment (Vercel + Render + PostgreSQL)

```
Browser ──https──► Vercel (React build; /api/* rewritten to the backend: same origin, first-party cookie)
                        └──► Render web service (one instance, Python, frozen classifiers in memory)
                                    ├──► PostgreSQL  (DATABASE_URL; the authoritative store)
                                    ├──► Render disk /var/data (uploaded workbooks)
                                    └──► Gemini API (redacted comment text only)
```

## Authoritative data store
`PERSISTENCE_BACKEND=postgres` (default). One repository bundle serves users, sessions, comments, classifications, drafts, assignments, decisions, audit events, messages, account requests, jobs and reset tokens; analytics and exports are computed from it. MongoDB is an alternative backend that exists in code and was never cut over (see `docs/persistence.md`). Production refuses SQLite and refuses to start without a PostgreSQL `DATABASE_URL`. Schema: `alembic upgrade head` (head `d5f1b3c8a2e6`), run from a trusted machine against the target database before first start.

## Backend (Render) — `render.yaml`
| Setting | Value / note |
|---|---|
| Plan | 1 CPU / 2 GB (`1c-2g`); measured peak ≈ 0.5 GB with all three classifiers loaded; 512 MB is not safe |
| Build | CPU-only torch → `requirements-server.txt` → `scripts/fetch_frozen_models.py` (≈ 1.1 GB download, checksum-verified; fails the build on any mismatch) |
| Start | `python scripts/serve_reviewer_api.py` (binds `$PORT`, `0.0.0.0` when `RCS_ENV=production`) · health `/api/health` |
| Disk | 1 GB at `/var/data`, `RCS_UPLOAD_DIR=/var/data/uploads` (uploads and resumable imports survive restarts) |
| Instances | exactly one (in-process login rate limiter; a disk attaches to one instance) |

| Variable | Required | Purpose |
|---|---|---|
| `RCS_ENV=production` | yes | Secure cookies, strict origin checks, no dev flags |
| `DATABASE_URL` | yes | PostgreSQL URL (secret) |
| `GEMINI_API_KEY` | yes for drafts | without it every draft fails closed to "review only" |
| `ALLOWED_ORIGINS` | yes | exact `https://` origin of the Vercel site |
| `HF_MODEL_REPO_ID`, `HF_MODEL_REVISION`, `HF_TOKEN` | yes (build) | frozen model artifacts; the revision must be a full 40-character commit SHA (a branch, tag or abbreviation is refused) |
| `RCS_FROZEN_SELECTION_PATH`, `RCS_FROZEN_INTENT_PATH`, `RCS_MARBERTV2_TOPIC_CHECKPOINT`, `RCS_CAMELBERT_SENTIMENT_PATH` | optional | override where the provisioned artifacts are read from; absolute paths only in production (defaults are the `data/models/...` locations the fetch step fills) |
| `RCS_TRUSTED_PROXIES` | recommended | IPs/CIDRs allowed to set `X-Forwarded-For` |
| `RCS_PASSWORD_RESET_LINK_BASE` + `RCS_SMTP_HOST/PORT/SECURITY/USERNAME/PASSWORD/FROM` | optional | password-reset e-mail (TLS mandatory); off until both are set |
| `ENTRA_TENANT_ID`, `ENTRA_CLIENT_ID`, `ENTRA_CLIENT_SECRET`, `ENTRA_REDIRECT_URI`, `ENTRA_FRONTEND_RETURN_URL` | optional | Microsoft sign-in; button shows "unavailable" until set |
Never set `RCS_DEMO_*` here (the demo mode refuses production secrets).

## Frontend (Vercel)
Root directory `web`, Vite, `npm run build`, output `dist`. After the backend URL exists: `cd web && npm run configure:proxy -- https://<render-host>` and commit the rewritten `web/vercel.json`. Leave `VITE_API_BASE_URL` unset.

## First deployment, in order
1. Obtain a PostgreSQL database and credentials from the database administrator; apply `alembic upgrade head`.
2. Create accounts with `scripts/manage_users.py` (admin first); never ship default passwords.
3. Provision the model repository and a read-only `HF_TOKEN`; create the Render service from the blueprint (manual deploy), set the secrets, deploy.
4. Create the Vercel project, run `configure:proxy`, set `ALLOWED_ORIGINS` to its origin, redeploy both.
5. Smoke test with a synthetic workbook (import → classification → assignment → review → export); then check `/api/health`.

## Cost (approximate list prices, verify before purchase)
Render 2 GB web service ≈ $25/month + disk ≈ $0.25/GB/month · Vercel Pro ≈ $20 per member/month (Hobby is non-commercial) · PostgreSQL per the chosen provider · Gemini usage-based (measure with a small authorised sample).

## Operations notes
Uploaded workbooks contain visitor data and are not deleted automatically: define a retention rule and purge `RCS_UPLOAD_DIR`. Back up PostgreSQL with the provider's tooling. Rollback: redeploy the previous Render deploy; schema changes are additive.
