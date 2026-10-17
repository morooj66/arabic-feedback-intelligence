import { FormEvent, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, microsoftSignInUrl } from "../api/client";
import type { ApiError } from "../api/types";
import "./LoginPage.css";
import { SystemState } from "../components/SystemState";
import { STATE_COPY } from "../components/stateCopy";
import { AuthShell } from "../components/AuthShell";

// Machine-readable codes the backend redirects back with via ?error= —
// from the Entra callback (see http_api.py::_entra_login_error_url) and,
// for `service_unavailable` only, also from a failed session-restore
// bootstrap (see ReviewerWorkspace.tsx) reusing this exact same
// query-param mechanism instead of inventing a second one. Never raw
// provider/driver error text, tokens, or secrets — just a short code
// mapped to one fixed Arabic message here.
const SSO_ERROR_MESSAGES: Record<string, string> = {
  not_configured: "تسجيل الدخول عبر Microsoft غير متاح حاليًا.",
  sso_cancelled_or_expired: "تم إلغاء تسجيل الدخول عبر Microsoft أو انتهت صلاحية المحاولة. يرجى المحاولة مرة أخرى.",
  sso_failed: "تعذر تسجيل الدخول عبر Microsoft. حاول مرة أخرى أو استخدم البريد الإلكتروني وكلمة المرور.",
  account_not_registered: "هذا الحساب غير مسجَّل في النظام. يرجى التواصل مع مسؤول النظام.",
  service_unavailable: STATE_COPY.serviceUnavailable,
};
const SSO_ERROR_FALLBACK = "تعذر تسجيل الدخول عبر Microsoft. حاول مرة أخرى أو استخدم البريد الإلكتروني وكلمة المرور.";

function MicrosoftMark() {
  // The official four-square Microsoft mark — a plain inline SVG, same
  // pattern as the email/lock icons above, not a new icon-library
  // dependency.
  return (
    <svg width="18" height="18" viewBox="0 0 21 21" aria-hidden="true">
      <rect x="1" y="1" width="9" height="9" fill="#f25022" />
      <rect x="11" y="1" width="9" height="9" fill="#7fba00" />
      <rect x="1" y="11" width="9" height="9" fill="#00a4ef" />
      <rect x="11" y="11" width="9" height="9" fill="#ffb900" />
    </svg>
  );
}

export function LoginPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [microsoftEnabled, setMicrosoftEnabled] = useState(false);
  const [showForgotHelp, setShowForgotHelp] = useState(false);
  const [passwordResetEnabled, setPasswordResetEnabled] = useState(false);
  const [accountRequestsEnabled, setAccountRequestsEnabled] = useState(false);

  useEffect(() => {
    api
      .health()
      .then((status) => {
        setMicrosoftEnabled(status.entra);
        setPasswordResetEnabled(Boolean(status.password_reset));
        setAccountRequestsEnabled(Boolean(status.account_requests));
      })
      .catch(() => {
        setMicrosoftEnabled(false);
        setPasswordResetEnabled(false);
        setAccountRequestsEnabled(false);
      });
  }, []);

  useEffect(() => {
    const code = searchParams.get("error");
    if (!code) return;
    setError(SSO_ERROR_MESSAGES[code] || SSO_ERROR_FALLBACK);
    // Clears the ?error= param from the URL so a later reload doesn't
    // re-show a stale message.
    setSearchParams({}, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.login({ email, password });
      navigate("/review");
    } catch (err) {
      const apiErr = err as ApiError;
      // Deliberately the same message regardless of whether the email is
      // unknown, the password is wrong, or the account is disabled —
      // never reveal which reason applied. A backend/database outage
      // (503 service_unavailable) is a genuinely different situation —
      // the user's credentials were never actually checked — and must
      // never be reported as "wrong email/password".
      setError(
        apiErr.code === "service_unavailable"
          ? SSO_ERROR_MESSAGES.service_unavailable
          : apiErr.code === "invalid_credentials" || apiErr.status === 401
            ? "البريد الإلكتروني أو كلمة المرور غير صحيحة."
            : "تعذر تسجيل الدخول. حاول مرة أخرى.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell>
      <form className="login-card" onSubmit={onSubmit} aria-labelledby="login-title" aria-describedby="login-hint" aria-busy={busy}>
        <h2 id="login-title">تسجيل الدخول</h2>
        <p className="hint" id="login-hint">استخدم حسابك للوصول إلى نظام مراجعة تعليقات الزوار</p>
        {error ? (
          <SystemState kind="error" className="login-error">
            {error}
          </SystemState>
        ) : null}
        <div className="field">
          <label htmlFor="email">البريد الإلكتروني</label>
          <div className="field-input">
            <span className="ico" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                <rect x="3" y="5" width="18" height="14" rx="2" />
                <path d="M3 7l9 6 9-6" />
              </svg>
            </span>
            <input
              id="email"
              type="email"
              placeholder="أدخل البريد الإلكتروني"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
              dir="ltr"
              required
            />
          </div>
        </div>
        <div className="field">
          <label htmlFor="password">كلمة المرور</label>
          <div className="field-input">
            <span className="ico" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                <rect x="5" y="11" width="14" height="10" rx="2" />
                <path d="M8 11V8a4 4 0 018 0v3" />
              </svg>
            </span>
            <input
              id="password"
              type="password"
              placeholder="أدخل كلمة المرور"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </div>
        </div>

        <div className="login-row">
          <button
            type="button"
            className="btn-forgot-inline"
            onClick={() => (passwordResetEnabled ? navigate("/forgot-password") : setShowForgotHelp((v) => !v))}
            aria-expanded={showForgotHelp}
            aria-controls="forgot-help"
          >
            نسيت كلمة المرور؟
          </button>
        </div>
        {showForgotHelp ? (
          <p className="forgot-help" id="forgot-help" role="status">تواصل مع مسؤول النظام لإعادة تعيين كلمة المرور.</p>
        ) : null}

        <button className="btn-primary btn-block" type="submit" disabled={busy}>
          {busy ? "جاري تسجيل الدخول…" : "تسجيل الدخول"}
          {!busy ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M5 12h14M13 6l6 6-6 6" />
            </svg>
          ) : null}
        </button>

        <div className="or-sep">
          <span>أو</span>
        </div>
        {microsoftEnabled ? (
          <a className="btn-ms" href={microsoftSignInUrl}>
            <MicrosoftMark />
            الدخول عبر Microsoft
          </a>
        ) : (
          <>
            <button className="btn-ms" type="button" disabled aria-disabled="true" aria-describedby="ms-unavailable">
              <MicrosoftMark />
              الدخول عبر Microsoft
            </button>
            <p className="forgot-help" id="ms-unavailable" role="status">الدخول عبر Microsoft غير متاح حاليًا — لم يتم إعداده بعد.</p>
          </>
        )}
        {accountRequestsEnabled ? (
          <div className="login-back">
            <button type="button" className="btn-forgot-inline" onClick={() => navigate("/request-account")}>ليس لديك حساب؟ طلب حساب جديد</button>
          </div>
        ) : null}
      </form>
    </AuthShell>
  );
}
