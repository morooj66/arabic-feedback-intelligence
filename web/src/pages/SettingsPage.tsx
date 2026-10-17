import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AdminUser, PendingAccountRequest } from "../api/types";
import { ROLE_LABEL } from "../labels";

const SORT_KEY = "rcs.defaultQueueSort";

export function readDefaultSort(): string {
  const value = window.localStorage.getItem(SORT_KEY);
  return value === "newest" || value === "oldest" || value === "priority" ? value : "priority";
}

export function writeDefaultSort(value: string) {
  window.localStorage.setItem(SORT_KEY, value);
}

// Official links are deployment configuration (build-time env), never invented here. A row is a link only when the value is a well-formed https:// URL
// (or a mailto: address for support); otherwise it states honestly that no official link has been configured yet.
const HELP_ROWS = [
  { label: "الدعم الفني", env: "VITE_SUPPORT_URL", allowMailto: true },
  { label: "سياسة الخصوصية", env: "VITE_PRIVACY_URL", allowMailto: false },
  { label: "الشروط والأحكام", env: "VITE_TERMS_URL", allowMailto: false },
] as const;

export function configuredHelpUrl(raw: unknown, allowMailto: boolean): string | null {
  const value = typeof raw === "string" ? raw.trim() : "";
  if (!value) return null;
  if (allowMailto && /^mailto:[^\s@]+@[^\s@]+\.[^\s@]+$/i.test(value)) return value;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname ? url.toString() : null;
  } catch {
    return null;
  }
}

function PendingAccountRequests() {
  const [rows, setRows] = useState<PendingAccountRequest[] | null>(null);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState("");

  const load = () =>
    api
      .accountRequests()
      .then((res) => { setRows(res.requests); })
      .catch(() => setError("تعذر تحميل طلبات الحسابات."));

  useEffect(() => { void load(); }, []);

  async function decide(id: string, approve: boolean) {
    setBusyId(id);
    setError("");
    try {
      if (approve) await api.approveAccountRequest(id);
      else await api.rejectAccountRequest(id);
      await load();
    } catch (e) {
      const code = (e as { code?: string }).code;
      setError(code === "already_decided" ? "تم البت في هذا الطلب مسبقًا." : code === "account_exists" ? "يوجد حساب بهذا البريد بالفعل؛ أُغلق الطلب." : "تعذر تنفيذ الإجراء.");
      await load();
    } finally {
      setBusyId("");
    }
  }

  return (
    <div aria-label="طلبات الحسابات المعلّقة">
      <h3>طلبات الحسابات المعلّقة</h3>
      <p className="settings-help">الموافقة تنشئ حساب «مراجع» فقط؛ حسابات «مسؤول النظام» تُنشأ عبر أداة الإدارة على الخادم. لا يُتحقق من ملكية البريد الإلكتروني تلقائيًا.</p>
      {error ? <p className="login-error" role="alert">{error}</p> : null}
      {rows === null && !error ? (
        <p className="settings-help">جاري التحميل…</p>
      ) : rows && rows.length ? (
        <ul className="settings-rows">
          {rows.map((r) => (
            <li key={r.request_id}>
              <span>{r.display_name} — <bdi dir="ltr">{r.email}</bdi></span>
              <span>
                <button type="button" disabled={busyId === r.request_id} onClick={() => decide(r.request_id, true)}>موافقة</button>{" "}
                <button type="button" disabled={busyId === r.request_id} onClick={() => decide(r.request_id, false)}>رفض</button>
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="settings-help">لا توجد طلبات معلّقة.</p>
      )}
    </div>
  );
}

export function SettingsPage(props: {
  sort: string;
  onSort: (value: string) => void;
  isAdmin?: boolean;
  onOpenInspector?: () => void;
  onOpenPlayground?: () => void;
}) {
  return (
    <div className="settings-shell" dir="rtl">
      <header className="settings-head">
        <p>تفضيلات المراجع لهذه الواجهة. ترتيب صفحة المراجعة يمكن تغييره مؤقتًا من القائمة نفسها.</p>
      </header>

      <section className="settings-card">
        <p className="settings-kicker">تفضيلات المراجعة</p>
        <h2>ترتيب قائمة المراجعة الافتراضي</h2>
        <p className="settings-help">حدد الترتيب الذي تظهر به التعليقات عند فتح صفحة المراجعة.</p>
        <label className="settings-field">
          <span>الترتيب الافتراضي</span>
          <select aria-label="الترتيب الافتراضي" value={props.sort} onChange={(e) => props.onSort(e.target.value)}>
            <option value="priority">حسب الأولوية</option>
            <option value="oldest">الأقدم أولًا</option>
            <option value="newest">الأحدث أولًا</option>
          </select>
        </label>
      </section>

      <section className="settings-card" id="settings-help">
        <h2>المساعدة والمعلومات</h2>
        <ul className="settings-rows">
          {HELP_ROWS.map((row) => {
            const href = configuredHelpUrl((import.meta.env as Record<string, unknown>)[row.env], row.allowMailto);
            return (
              <li key={row.label}>
                <span>{row.label}</span>
                {href ? (
                  <a href={href} target="_blank" rel="noopener noreferrer">فتح</a>
                ) : (
                  <span className="settings-soon">غير متاح بعد — لم يُعتمد رابط رسمي</span>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      <section className="settings-card settings-card--about">
        <h2>عن النظام</h2>
        <dl className="settings-meta">
          <div>
            <dt>اسم النظام</dt>
            <dd>نظام مراجعة تعليقات الزوار</dd>
          </div>
        </dl>
      </section>

      {props.isAdmin ? <AdminUsersSection /> : null}

      {props.isAdmin && props.onOpenInspector ? (
        // Deliberately not a primary navigation destination (see the UI
        // reconciliation audit's Agent Inspector finding) — a
        // technical/development diagnostic reached only from here, kept
        // visually quiet and separate from the day-to-day admin
        // workflow. The server-side ADMIN-only check on every
        // /api/admin/inspector/* endpoint is unchanged and unrelated to
        // this placement.
        <section className="settings-card settings-card--technical">
          <p className="settings-kicker">لأغراض تقنية</p>
          <h2>فاحص الوكيل</h2>
          <p className="settings-help">أداة تشخيصية داخلية لفريق التطوير — غير مخصصة للاستخدام التشغيلي اليومي.</p>
          <button type="button" className="btn-quiet" onClick={props.onOpenInspector}>
            فتح فاحص الوكيل
          </button>
        </section>
      ) : null}

      {props.isAdmin && props.onOpenPlayground ? (
        // Internal evaluation tool, disabled by default on the server
        // (RCS_PLAYGROUND_ENABLED). Placed next to the Agent Inspector, not in
        // the primary navigation.
        <section className="settings-card settings-card--technical">
          <p className="settings-kicker">لأغراض تقنية</p>
          <h2>ساحة التقييم التفاعلية</h2>
          <p className="settings-help">اختبار يدوي للمصنِّفات الثلاثة والوكيل بتعليقات مخصصة — للتطوير فقط، ومعطّلة افتراضيًا.</p>
          <button type="button" className="btn-quiet" onClick={props.onOpenPlayground}>
            فتح ساحة التقييم
          </button>
        </section>
      ) : null}
    </div>
  );
}

// Phase 6 — the smallest "administrative visibility" the spec asks for:
// a read-only account list. Creating/editing accounts stays CLI-only
// (scripts/manage_users.py) — no admin CRUD UI in this phase.
function AdminUsersSection() {
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .adminUsers()
      .then((res) => setUsers(res.users))
      .catch(() => setError("تعذر تحميل قائمة المستخدمين."));
  }, []);

  return (
    <section className="settings-card">
      <p className="settings-kicker">إدارة الحسابات</p>
      <h2>المستخدمون</h2>
      <p className="settings-help">
        عرض فقط — إنشاء الحسابات وتعديل الأدوار يتم عبر أداة الإدارة على الخادم.
      </p>
      <PendingAccountRequests />
      {error ? <p className="login-error">{error}</p> : null}
      {users === null && !error ? (
        <p className="settings-help">جاري التحميل…</p>
      ) : users && users.length ? (
        <ul className="settings-rows">
          {users.map((u) => (
            <li key={u.user_id}>
              <span>
                {u.display_name} — {u.email}
              </span>
              <span className="settings-soon">
                {ROLE_LABEL[u.role] || u.role}
                {u.is_active ? "" : " · معطّل"}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
