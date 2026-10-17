# Authentication, accounts and password recovery

## Model
- Two roles, `ADMIN` and `REVIEWER`, stored on the server-side user row and read **only** from the verified session. Nothing the browser sends can set or change a role.
- One login page (`/`): e-mail + password, "forgot password", "new account request", and a Microsoft button.
- Passwords: bcrypt, 8–72 bytes. Sessions: random 256-bit cookie token (HttpOnly; `SameSite=None; Secure` in production, `Lax` locally), 12-hour lifetime; **only the SHA-256 of the token is stored**; an account deactivated mid-session is rejected on its next request.
- Login rate limiting per address and per e-mail; generic failure for unknown / wrong password / inactive account.

## Account creation
1. **Admins/operators** (unchanged): `scripts/manage_users.py create ...` provisions ADMIN or REVIEWER accounts.
2. **Self-service request** (new): `POST /api/account-requests {email, display_name, password}` (public). It stores a *request*, not an account. The requester cannot log in. An ADMIN approves or rejects in **Settings → Users → طلبات الحسابات المعلّقة**; approval creates an **active REVIEWER** with the password hash chosen at request time. The role is never taken from the request; ADMIN accounts can only be created with `manage_users.py`.
   - No account enumeration (an existing address, a pending/decided request and a new address all get the same answer, and the same response time because the password is always hashed).
   - Limits: 5 requests / hour / client address, 3 / hour / e-mail, at most 500 pending requests; invalid input is a clean 400.
   - Not provided: e-mail ownership verification and a notification to the requester about the decision (both need a mail provider and a policy decision).

## Password recovery
See [password_reset.md](password_reset.md). Needs `RCS_PASSWORD_RESET_LINK_BASE` and the `RCS_SMTP_*` variables; until both are set the feature is off and the login screen says to contact the administrator.

## Microsoft sign-in
Implemented (`rcs.review.entra_auth`, authorization-code + PKCE, ID-token validation, only maps to a *pre-existing* active local user). Requires `ENTRA_TENANT_ID`, `ENTRA_CLIENT_ID`, `ENTRA_CLIENT_SECRET`, `ENTRA_REDIRECT_URI`, `ENTRA_FRONTEND_RETURN_URL` from an institutional app registration. **Not verified against a real tenant.** When not configured the login page shows the Microsoft button **disabled** with "غير متاح حاليًا — لم يتم إعداده بعد"; no login is ever simulated.

## Configuration still required for the pilot
| Item | Needed for | Owner |
|---|---|---|
| SMTP provider credentials + `RCS_SMTP_FROM` | password reset e-mail | institution IT |
| `RCS_PASSWORD_RESET_LINK_BASE` (public frontend URL) | reset links | deployment |
| Entra app registration (5 variables) | Microsoft sign-in | institution IT |
| MongoDB: run `scripts/bootstrap_mongo_indexes.py`; PostgreSQL/SQLite: run migrations (`d5f1b3c8a2e6` is the new head) | account requests, messages | deployment |
