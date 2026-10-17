# Deploying the synthetic-data demo (preparation only; nothing here has been executed)

```
Browser ──https──► Vercel (static React build; /api/* rewritten to the backend, same origin)
                        └──► backend container (Dockerfile.demo, demo mode, no model weights)
                                    └──► PostgreSQL database "rcs_demo_*" (synthetic data only)
```

## 1. Database
Create a dedicated PostgreSQL database named `rcs_demo_online` (any host: Supabase free tier, Render Postgres, …). It must hold nothing else. Run `scripts/demo/init_demo_environment.py` from your machine (see DEMO_MODE.md) — it applies the migrations and creates the invite-only accounts.

## 2. Backend (Render example; Fly.io/Railway work the same way)
- Blueprint: `render.demo.yaml` (Docker runtime, `Dockerfile.demo`, `autoDeploy: false`). Image contents: Python + `requirements-demo.txt` only (no torch, no models, no LLM SDK, no secrets).
- Set in the dashboard (never in Git): `DATABASE_URL` (the demo database), `ALLOWED_ORIGINS` (the exact `https://` origin of the Vercel site). Already in the blueprint: `RCS_ENV=production` (Secure cookies, strict origin checks), `RCS_DEMO_MODE`, `RCS_DEMO_ACK`, `PERSISTENCE_BACKEND=postgres`.
- Do **not** set any of: `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `MONGODB_URI`, `RCS_SMTP_*`, `ENTRA_*`, `HF_*` — the server refuses to start if any is present.
- Health check: `/api/health` → `"demo": true`. One instance only (rate limiters are in-process). Expected to run in ≈ 256–512 MB (measure after the first start).

## 3. Frontend (Vercel)
1. Import the repository; **Root Directory:** `web`; Framework: Vite; Build: `npm run build`; Output: `dist`. Leave `VITE_API_BASE_URL` unset.
2. Once the backend URL is known: `cd web && npm run configure:proxy -- https://<backend-host>` and commit the rewritten `web/vercel.json` (it proxies `/api/*`; the cookie becomes first-party). Do not commit a placeholder.
3. Recommended: enable Vercel **Deployment Protection** (password / SSO) in addition to the app's own login, so the demo URL is not publicly crawlable.

## 4. Pre-flight checklist
- [ ] the database is empty apart from synthetic data and is named `rcs_demo_*`
- [ ] no production secret exists anywhere in the host's environment
- [ ] credentials file kept private; passwords delivered individually
- [ ] `ALLOWED_ORIGINS` equals the Vercel origin exactly
- [ ] log in as each role from a private window; check the "synthetic demo" pill; try the sign-up URL (must be unavailable)

## 5. What is not covered
Real Gemini calls, real classifiers, SMTP, Microsoft sign-in, real data, high availability, backups, monitoring. Those belong to a later pilot with their own approvals.
