# Security & privacy

## Never in this repository
Real visitor comments (or derived excerpts), names, e-mail addresses, phone numbers, uploaded workbooks, databases, model weights, `.env` files, API keys, connection strings, institutional logos/photographs/documents. `.gitignore` and `.dockerignore` enforce this and `tests/test_deployment_hardening.py` asserts it. Run a secret scanner before every push.

## Controls (each covered by automated tests unless marked)
- **Authentication:** bcrypt; server-side sessions; only the SHA-256 of the session token is stored; `HttpOnly` cookies, `Secure`/`SameSite=None` in production; 12-hour lifetime; deactivated accounts are rejected on the next request; login rate limiting per address and e-mail.
- **Authorization:** `ADMIN`/`REVIEWER` read only from the verified session; every route checks the role in the backend; a reviewer cannot read, decide on or message about another reviewer's items.
- **Accounts:** public account *requests* never create a login; approval is admin-only and always yields a reviewer; no account enumeration (same answer and response time for known and unknown addresses); request rate limits and a pending cap.
- **Password reset:** 256-bit random single-use tokens (only the hash stored), ≤ 30 min, token in the URL fragment, constant answers, rate limits, all sessions revoked on success; SMTP sender with **mandatory TLS** and certificate verification.
- **Web:** origin and content-type checks on state-changing requests; sanitised error responses; upload size/type validation; Excel export neutralises formula injection; audit log holds no free text.
- **Privacy boundary:** PII redaction and prompt guard before any external call; production fails closed if pinned model artifacts are missing; no real provider is contacted by any test.

## Not verified / out of scope
Penetration testing; dependency vulnerability audit; live LLM provider behaviour; PostgreSQL row-locking and live MongoDB; Entra sign-in against a real tenant; messaging rate limits.

## Reporting a vulnerability
See [SECURITY.md](../SECURITY.md).
