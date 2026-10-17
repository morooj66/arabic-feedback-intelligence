import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { ApiError } from "../api/types";
import { AuthShell } from "../components/AuthShell";
import { SystemState } from "../components/SystemState";

const SENT_NOTICE = "تم استلام طلبك. سيبقى الحساب غير مفعّل إلى أن يوافق عليه مسؤول النظام، ولا يمكن تسجيل الدخول قبل ذلك.";
const UNAVAILABLE = "طلب حساب جديد غير متاح حاليًا. تواصل مع مسؤول النظام.";

const ERROR_MESSAGES: Record<string, string> = {
  invalid_email: "أدخل بريدًا إلكترونيًا صالحًا.",
  invalid_name: "أدخل الاسم (حتى 100 حرف).",
  invalid_password: "كلمة المرور يجب ألا تقل عن 8 أحرف.",
  too_many_requests: "تم تجاوز عدد المحاولات المسموح بها. حاول مرة أخرى بعد قليل.",
};

export function RequestAccountPage() {
  const navigate = useNavigate();
  const [available, setAvailable] = useState<boolean | null>(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .health()
      .then((status) => setAvailable(Boolean(status.account_requests)))
      .catch(() => setAvailable(false));
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.requestAccount(email.trim(), name.trim(), password);
      setSent(true);
      setPassword("");
    } catch (err) {
      const apiErr = err as ApiError;
      setError(ERROR_MESSAGES[apiErr.code || ""] || (apiErr.status === 429 ? ERROR_MESSAGES.too_many_requests : "تعذر إرسال الطلب. حاول مرة أخرى."));
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
          <h2 id="request-title">طلب حساب جديد</h2>
          <SystemState kind="unavailable" className="login-error">{UNAVAILABLE}</SystemState>
          {backToLogin}
        </div>
      </AuthShell>
    );
  }

  return (
    <AuthShell>
      <form className="login-card" onSubmit={onSubmit} aria-labelledby="request-title" aria-busy={busy}>
        <h2 id="request-title">طلب حساب جديد</h2>
        <p className="hint">يُنشأ الحساب بعد موافقة مسؤول النظام. لا يمكنك اختيار الصلاحيات؛ تُحدَّد عند الموافقة.</p>
        {sent ? <SystemState kind="success" className="login-notice">{SENT_NOTICE}</SystemState> : null}
        {error ? <SystemState kind="error" className="login-error">{error}</SystemState> : null}
        {!sent ? (
          <>
            <div className="field">
              <label htmlFor="req-name">الاسم</label>
              <div className="field-input"><input id="req-name" type="text" placeholder="أدخل اسمك" value={name} onChange={(e) => setName(e.target.value)} maxLength={100} autoComplete="name" required /></div>
            </div>
            <div className="field">
              <label htmlFor="req-email">البريد الإلكتروني</label>
              <div className="field-input"><input id="req-email" type="email" placeholder="أدخل البريد الإلكتروني" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="username" dir="ltr" required /></div>
            </div>
            <div className="field">
              <label htmlFor="req-password">كلمة المرور</label>
              <div className="field-input"><input id="req-password" type="password" placeholder="8 أحرف على الأقل" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="new-password" minLength={8} required /></div>
            </div>
            <button className="btn-primary btn-block" type="submit" disabled={busy || available === null}>{busy ? "جارٍ الإرسال…" : "إرسال الطلب"}</button>
          </>
        ) : null}
        {backToLogin}
      </form>
    </AuthShell>
  );
}
