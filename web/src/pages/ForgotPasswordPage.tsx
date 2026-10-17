import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { ApiError } from "../api/types";
import { AuthShell } from "../components/AuthShell";
import { SystemState } from "../components/SystemState";

const SENT_NOTICE =
  "إذا كان هذا البريد الإلكتروني مسجَّلًا لدينا، فسيصلك رابط لإعادة تعيين كلمة المرور. الرابط صالح لمدة 30 دقيقة ويُستخدم مرة واحدة فقط.";
const UNAVAILABLE = "إعادة تعيين كلمة المرور عبر البريد الإلكتروني غير متاحة حاليًا. تواصل مع مسؤول النظام لإعادة تعيين كلمة المرور.";

export function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [available, setAvailable] = useState<boolean | null>(null);
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .health()
      .then((status) => setAvailable(Boolean(status.password_reset)))
      .catch(() => setAvailable(false));
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.requestPasswordReset(email.trim());
      setSent(true);
    } catch (err) {
      const apiErr = err as ApiError;
      // The server answers identically for registered and unregistered addresses, so there is nothing account-specific to show here.
      setError(
        apiErr.code === "password_reset_unavailable"
          ? UNAVAILABLE
          : apiErr.code === "too_many_attempts" || apiErr.status === 429
            ? "تم تجاوز عدد المحاولات المسموح بها. حاول مرة أخرى بعد قليل."
            : "تعذر إرسال الطلب. حاول مرة أخرى.",
      );
    } finally {
      setBusy(false);
    }
  }

  const backToLogin = (
    <div className="login-back">
      <button type="button" className="btn-forgot-inline" onClick={() => navigate("/")}>العودة إلى تسجيل الدخول</button>
    </div>
  );

  if (available === false) {
    return (
      <AuthShell>
        <div className="login-card">
          <h2 id="forgot-title">إعادة تعيين كلمة المرور</h2>
          <SystemState kind="unavailable" className="login-error">{UNAVAILABLE}</SystemState>
          {backToLogin}
        </div>
      </AuthShell>
    );
  }

  return (
    <AuthShell>
      <form className="login-card" onSubmit={onSubmit} aria-labelledby="forgot-title" aria-busy={busy}>
        <h2 id="forgot-title">إعادة تعيين كلمة المرور</h2>
        <p className="hint">أدخل بريدك الإلكتروني المسجَّل وسنرسل إليك رابطًا لتعيين كلمة مرور جديدة</p>
        {sent ? <SystemState kind="success" className="login-notice">{SENT_NOTICE}</SystemState> : null}
        {error ? <SystemState kind="error" className="login-error">{error}</SystemState> : null}
        {!sent ? (
          <>
            <div className="field">
              <label htmlFor="reset-email">البريد الإلكتروني</label>
              <div className="field-input">
                <span className="ico" aria-hidden="true">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                    <rect x="3" y="5" width="18" height="14" rx="2" />
                    <path d="M3 7l9 6 9-6" />
                  </svg>
                </span>
                <input
                  id="reset-email"
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
            <button className="btn-primary btn-block" type="submit" disabled={busy || available === null}>
              {busy ? "جارٍ الإرسال…" : "إرسال رابط إعادة التعيين"}
            </button>
          </>
        ) : null}
        {backToLogin}
      </form>
    </AuthShell>
  );
}
