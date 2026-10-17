# Self-service password reset

`POST /api/password-reset/request {email}` and `POST /api/password-reset/confirm {token, password}` (public). Off until an email sender is installed and `RCS_PASSWORD_RESET_LINK_BASE` is set;
until then both answer `503 password_reset_unavailable` (identically for every address) and `/api/health` reports `password_reset: false`, so the login screen keeps its "contact your administrator" message.

* Token: `secrets.token_urlsafe(32)` (256 bits); only its SHA-256 hash is stored (`password_reset_tokens`, SQL table or Mongo collection); lifetime <= 30 minutes (hard cap); single use (atomic consume);
  a new request supersedes older links. The link carries the token in the URL fragment (`/reset-password#token=...`), which browsers never send to a server or a Referer, and the page removes it from the address bar.
* Request: same `200 {"ok":true}` for registered, unregistered and inactive addresses; sending runs off the request path; nothing is created or sent for unknown/inactive accounts.
  Rate limits (every request counts, account or not): 3 per email and 10 per client address per 30 minutes -> 429. Client address = `resolve_client_ip` (trusted proxies only).
* Confirm: password policy (>= 8 chars, <= 72 bytes for bcrypt) is checked BEFORE the token is consumed; unknown / expired / used tokens are one identical `400 invalid_or_expired_token`; 20 failures per address per 30 minutes -> 429.
  Success replaces the hash, revokes EVERY session of that user, supersedes remaining tokens; no session is created (the user logs in again).
* Logging: raw tokens, links, passwords and email addresses are never logged (send failures log only the exception type).
* Sender: `rcs.review.email_sender` (`EmailSender` Protocol; `UnconfiguredEmailSender` default = fail closed; `FakeEmailSender` for tests/local). A real SMTP sender (`SmtpEmailSender`) is included and is selected automatically from `RCS_SMTP_*` (see `.env.example`); TLS is mandatory (STARTTLS or implicit TLS, certificates verified, no plain-text fallback), the body is base64-encoded UTF-8, and nothing about the message or recipient is logged. Tested against a local fake SMTP server only; **never run against a real mail provider in this repository**.
* Schema: one new table (`password_reset_tokens`) - Alembic revision `b3d9f0a1c7e2` (also created by `init_db` on SQLite/Postgres); Mongo: run `scripts/bootstrap_mongo_indexes.py` for the new TTL + user_id indexes.
