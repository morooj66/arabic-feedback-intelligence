import { FormEvent, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { ApiError } from "../api/types";
import { AuthShell } from "../components/AuthShell";
import { SystemState } from "../components/SystemState";

const MIN_LENGTH = 8;
const MAX_BYTES = 72;
const INVALID_LINK = "رابط إعادة التعيين غير صالح أو انتهت صلاحيته. اطلب رابطًا جديدًا.";

function tokenFromHash(hash: string): string {
  // The emailed link carries the token in the URL FRAGMENT (never sent to a server, never in a Referer): /reset-password#token=...
  return new URLSearchParams(hash.replace(/^#/, "")).get("token") || "";
}

export function ResetPasswordPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [token, setToken] = useState(() => tokenFromHash(location.hash));
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState(token ? "" : INVALID_LINK);
  const [linkDead, setLinkDead] = useState(!token);

  useEffect(() => {
    // React to EVERY fragment change, not only the first load: a new link opened in the same tab changes only the hash. A fresh token resets the form;
    // the fragment is always removed from the address bar (and history) right after it has been read, so the token never lingers.
    if (!location.hash) return;
    const fresh = tokenFromHash(location.hash);
    if (fresh) {
      setToken(fresh);
      setLinkDead(false);
      setDone(false);
      setError("");
      setPassword("");
      setConfirm("");
    }
    navigate({ pathname: location.pathname, hash: "" }, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.hash]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    if (password.length < MIN_LENGTH) {
      setError("كلمة المرور يجب ألا تقل عن 8 أحرف.");
      return;
    }
    if (new TextEncoder().encode(password).length > MAX_BYTES) {
      setError("كلمة المرور طويلة جدًا.");
      return;
    }
    if (password !== confirm) {
      setError("كلمتا المرور غير متطابقتين.");
      return;
    }
    setBusy(true);
    try {
      await api.confirmPasswordReset({ token, password });
      setDone(true);
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.code === "invalid_or_expired_token") {
        setLinkDead(true);
        setError(INVALID_LINK);
      } else if (apiErr.code === "weak_password") {
        setError("كلمة المرور لا تستوفي الشروط: 8 أحرف على الأقل.");
      } else if (apiErr.code === "too_many_attempts" || apiErr.status === 429) {
        setError("تم تجاوز عدد المحاولات المسموح بها. حاول مرة أخرى بعد قليل.");
      } else if (apiErr.code === "password_reset_unavailable") {
        setError("إعادة تعيين كلمة المرور غير متاحة حاليًا. تواصل مع مسؤول النظام.");
      } else {
        setError("تعذر تعيين كلمة المرور. حاول مرة أخرى.");
      }
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <AuthShell>
        <div className="login-card">
          <h2 id="reset-title">تم تعيين كلمة المرور</h2>
          <SystemState kind="success" className="login-notice">تم تعيين كلمة المرور الجديدة. جميع الجلسات السابقة أُنهيت، يمكنك الآن تسجيل الدخول.</SystemState>
          <button className="btn-primary btn-block" type="button" onClick={() => navigate("/")}>الذهاب إلى تسجيل الدخول</button>
        </div>
      </AuthShell>
    );
  }

  return (
    <AuthShell>
      <form className="login-card" onSubmit={onSubmit} aria-labelledby="reset-title" aria-busy={busy}>
        <h2 id="reset-title">تعيين كلمة مرور جديدة</h2>
        {!linkDead ? <p className="hint">اختر كلمة مرور جديدة لحسابك (8 أحرف على الأقل)</p> : null}
        {error ? <SystemState kind="error" className="login-error">{error}</SystemState> : null}
        {!linkDead ? (
          <>
            <div className="field">
              <label htmlFor="new-password">كلمة المرور الجديدة</label>
              <div className="field-input">
                <input id="new-password" type="password" placeholder="أدخل كلمة المرور الجديدة" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="new-password" required />
              </div>
            </div>
            <div className="field">
              <label htmlFor="confirm-password">تأكيد كلمة المرور</label>
              <div className="field-input">
                <input id="confirm-password" type="password" placeholder="أعد إدخال كلمة المرور" value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="new-password" required />
              </div>
            </div>
            <button className="btn-primary btn-block" type="submit" disabled={busy}>
              {busy ? "جارٍ الحفظ…" : "حفظ كلمة المرور"}
            </button>
          </>
        ) : (
          <button className="btn-primary btn-block" type="button" onClick={() => navigate("/forgot-password")}>طلب رابط جديد</button>
        )}
        <div className="login-back">
          <button type="button" className="btn-forgot-inline" onClick={() => navigate("/")}>العودة إلى تسجيل الدخول</button>
        </div>
      </form>
    </AuthShell>
  );
}
