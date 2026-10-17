import { useEffect, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { AnalyticsSnapshot, ExcelSession, QueuePage, ReviewerSession, ReviewerSummaryRow, SourceFileSummary, WorkflowCounts } from "../api/types";
import { Avatar } from "../components/Avatar";
import { WorkspaceIcon } from "../components/WorkspaceIcon";
import { STATUS_LABEL } from "../labels";
import { formatTrendBucket, formatYmd } from "../time/appTime";
import { formatAbsoluteAr } from "../time/relative";
import "./AdminDashboard.css";

const number = new Intl.NumberFormat("ar-SA");
const percent = new Intl.NumberFormat("ar-SA", { style: "percent", maximumFractionDigits: 1 });
const SESSION_STATUS_LABEL: Record<string, string> = { uploaded: "بانتظار المعالجة", processing: "قيد المعالجة", ready: "جاهز للإسناد", completed: "مكتمل" };
const OUTCOMES = [
  { key: "SEND_AS_IS", label: "اعتماد دون تعديل", rate: "send_as_is_rate" },
  { key: "EDIT_AND_SEND", label: "تعديل ثم اعتماد", rate: "edited_rate" },
  { key: "NO_RESPONSE_NEEDED", label: "بدون رد", rate: "no_reply_rate" },
] as const;

type Resource<T> = { status: "loading" | "error" | "ready"; data: T | null };
function useResource<T>(fetcher: () => Promise<T>, revision: string | number): Resource<T> {
  const [state, setState] = useState<Resource<T>>({ status: "loading", data: null });
  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading", data: null });
    fetcher().then(data => { if (!cancelled) setState({ status: "ready", data }); })
      .catch(() => { if (!cancelled) setState({ status: "error", data: null }); });
    return () => { cancelled = true; };
    // Callers supply a revision covering the request's parameters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [revision]);
  return state;
}

function ResourceContent<T>({ resource, children }: { resource: Resource<T>; children: (data: T) => ReactNode }) {
  if (resource.status === "error") return <p className="admin-state" role="status">تعذر تحميل هذا القسم. يمكنك إعادة المحاولة من تحديث البيانات.</p>;
  if (resource.status !== "ready") return <p className="admin-state" role="status">جاري تحميل البيانات…</p>;
  return <>{children(resource.data as T)}</>;
}

export function AdminDashboard({ session, onImport, onGoAssignment, onGoCompleted }: {
  session: ReviewerSession;
  onImport: () => void;
  onGoAssignment: (sourceFile?: string) => void;
  onGoCompleted: () => void;
}) {
  const [now, setNow] = useState(() => new Date());
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  // Greeting follows the browser's local clock. Analytics follows the existing API's Riyadh date boundaries.
  const greeting = now.getHours() < 12 ? "صباح الخير" : "مساء الخير";
  const end = formatYmd(now);
  const start = formatYmd(new Date(now.getTime() - 13 * 86_400_000));
  const sources = useResource<SourceFileSummary[]>(() => api.adminSourceSummary().then(r => r.sources), revision);
  const reviewers = useResource<ReviewerSummaryRow[]>(() => api.adminReviewers().then(r => r.reviewers), revision);
  const active = useResource<ExcelSession | null>(() => api.activeExcelSession().then(r => r.active), revision);
  const counts = useResource<WorkflowCounts>(() => api.counts(), revision);
  const analytics = useResource<AnalyticsSnapshot>(() => api.analytics({ range: "custom", start, end }), `${revision}:${start}:${end}`);
  const recent = useResource<QueuePage>(() => api.queue({ filter: "pending", sort: "newest", search: "", offset: 0, limit: 5 }), revision);
  const anyError = [sources, reviewers, active, counts, analytics, recent].some(r => r.status === "error");
  const allLoading = [sources, reviewers, active, counts, analytics, recent].every(r => r.status === "loading");
  const unassigned = sources.data?.reduce((sum, row) => sum + row.unassigned, 0);
  const completed = sources.data?.reduce((sum, row) => sum + row.completed, 0);

  return (
    <div className="admin-home" dir="rtl">
      <header className="admin-welcome">
        <div><h2>{greeting}، {session.display_name}</h2><p>نظرة عامة على سير مراجعة تعليقات الزوار</p></div>
        <div className="admin-date"><time dateTime={now.toISOString()}>{new Intl.DateTimeFormat("ar-SA-u-ca-gregory", { weekday: "long", day: "numeric", month: "long", year: "numeric" }).format(now)}</time><button type="button" className="admin-text-button" disabled={allLoading} onClick={() => { setNow(new Date()); setRevision(n => n + 1); }}>تحديث البيانات</button></div>
      </header>
      {anyError ? <p className="admin-alert" role="alert">بعض البيانات غير متاحة حاليًا. الأقسام الأخرى تعرض البيانات التي تم تحميلها.</p> : null}

      <section className="admin-kpis" aria-label="حالة عبء المراجعة">
        <Metric icon="messages" label="إجمالي التعليقات" value={counts.data?.all} state={counts.status} note="جميع التعليقات في النظام" />
        <Metric icon="assignment" label="غير مسندة" value={unassigned} state={sources.status} note="بحسب ملخص الإسناد الحالي" />
        <Metric icon="review" label="قيد المراجعة" value={counts.data ? counts.data.by_status.IN_REVIEW ?? 0 : undefined} state={counts.status} note="تعليقات بدأت مراجعتها بالفعل" />
        <Metric icon="completed" label="مكتملة" value={completed} state={sources.status} note="قرارات المراجعين فقط" />
      </section>

      <section className="admin-panel admin-outcomes" aria-labelledby="admin-outcomes-title">
        <div className="admin-section-head"><div><h3 id="admin-outcomes-title">نتائج المراجعة البشرية</h3><p>كيف تعامل المراجعون مع المسودات خلال آخر 14 يومًا</p></div><span className="admin-period">آخر 14 يومًا</span></div>
        <ResourceContent resource={analytics}>{data => <>
          <div className="admin-outcome-grid">{OUTCOMES.map((outcome, index) => {
            const count = data.outcomes[outcome.key];
            const rate = data.quality[outcome.rate];
            return <article key={outcome.key} className={`admin-outcome admin-outcome-${index}`}><span>{outcome.label}</span><div><strong>{typeof count === "number" ? number.format(count) : "غير متاح"}</strong>{rate != null && data.quality.sample > 0 ? <span className="admin-rate">{percent.format(rate)}</span> : null}</div>{rate != null && data.quality.sample > 0 ? <div className="admin-meter" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(1, rate)) * 100}%` }} /></div> : null}</article>;
          })}</div>
          <p className="admin-footnote">{data.quality.sample > 0 ? `النسب من ${number.format(data.quality.sample)} قرار مراجعة بشرية مسجّل خلال الفترة؛ لا تشمل الإغلاق الآلي دون رد.` : "لا توجد قرارات مراجعة بشرية مسجلة خلال هذه الفترة؛ النسب غير متاحة."}</p>
        </>}</ResourceContent>
      </section>

      <div className="admin-main-grid">
        <section className="admin-panel" aria-labelledby="admin-trend-title">
          <div className="admin-section-head"><div><h3 id="admin-trend-title">سير المراجعة خلال آخر 14 يومًا</h3><p>التعليقات الواردة مقابل ما تمت مراجعته</p></div></div>
          <ResourceContent resource={analytics}>{data => data.availability.trend === "ok" && data.trend.length > 0 ? <ReviewTrend points={data.trend} /> : <p className="admin-state">لا تتوفر حركة مؤرخة في هذه الفترة لعرض المخطط.</p>}</ResourceContent>
        </section>
        <section className="admin-panel" aria-labelledby="admin-reviewers-title">
          <div className="admin-section-head"><div><h3 id="admin-reviewers-title">تقدم المراجعين</h3><p>أحمال العمل الحالية، دون ترتيب أو تقييم للأداء</p></div></div>
          <ResourceContent resource={reviewers}>{rows => rows.length ? <div className="admin-table-scroll admin-reviewer-scroll" role="region" aria-label="جدول تقدم المراجعين" tabIndex={0}><table className="admin-table" aria-label="تقدم المراجعين"><thead><tr><th scope="col">المراجع</th><th scope="col">المسند</th><th scope="col">المكتمل</th><th scope="col">المتبقي</th></tr></thead><tbody>{rows.map(row => <tr key={row.user_id}><th scope="row"><span className="admin-reviewer"><Avatar displayName={row.display_name} size="sm" /><span>{row.display_name}</span></span></th><td>{number.format(row.assigned_total)}</td><td>{number.format(row.completed)}</td><td>{number.format(row.pending)}</td></tr>)}</tbody></table></div> : <p className="admin-state">لا يوجد مراجعون نشطون حاليًا — لا يمكن إسناد أي عمل حتى تتم إضافة مراجع.</p>}</ResourceContent>
        </section>
      </div>

      <div className="admin-bottom-grid">
        <section className="admin-panel" aria-labelledby="admin-recent-title">
          <div className="admin-section-head"><div><h3 id="admin-recent-title">أحدث التعليقات بانتظار المراجعة</h3><p>آخر التعليقات الواردة التي لم تُحسم بعد</p></div></div>
          <ResourceContent resource={recent}>{page => page.items.length ? <ul className="admin-recent-list">{page.items.map(item => <li key={item.comment_id}><div><p>{item.comment_preview || "نص التعليق غير متاح"}</p><span className="admin-status">{STATUS_LABEL[item.review_status] || "حالة غير معروفة"}</span></div>{item.ingested_at ? <time dateTime={item.ingested_at} title={formatAbsoluteAr(item.ingested_at)}>{formatAbsoluteAr(item.ingested_at)}</time> : <span className="admin-footnote">تاريخ الوصول غير متاح</span>}</li>)}</ul> : <p className="admin-state">لا توجد تعليقات بانتظار المراجعة حاليًا.</p>}</ResourceContent>
        </section>
        <section className="admin-panel admin-next" aria-labelledby="admin-next-title">
          <div className="admin-section-head"><h3 id="admin-next-title">متابعة العمل</h3><WorkspaceIcon name="assignment" /></div>
          <div className="admin-quick-actions"><button type="button" className="admin-primary" onClick={() => onGoAssignment()}>إدارة الإسناد</button><button type="button" className="admin-secondary" onClick={onGoCompleted}>عرض المراجعات المكتملة</button></div>
          <div className="admin-import-summary">
            <ResourceContent resource={active}>{file => file ? <><p className="admin-eyebrow">آخر ملف نشط</p><h4>{file.original_filename}</h4><p>{SESSION_STATUS_LABEL[file.status] || "حالة غير معروفة"} · تمت مراجعة {number.format(file.counts.completed)} من {number.format(file.counts.canonical_total)} تعليقًا</p><button type="button" className="admin-text-button" onClick={onImport}>تفاصيل الاستيراد</button>{file.status !== "completed" && sources.data?.some(source => source.source_file === file.source_file && source.unassigned > 0) ? <button type="button" className="admin-text-button" onClick={() => onGoAssignment(file.source_file)}>متابعة إسناد هذا الملف</button> : null}</> : <><h4>{sources.status === "ready" && sources.data?.length === 0 ? "ابدأ باستيراد التعليقات" : "لا يوجد ملف نشط حاليًا"}</h4><p>ارفع ملف التعليقات لبدء عملية المعالجة والإسناد.</p></>}</ResourceContent>
            <button type="button" className="admin-secondary admin-import-button" onClick={onImport}><WorkspaceIcon name="upload" />استيراد ملف جديد</button>
          </div>
        </section>
      </div>
      <footer className="admin-footer">RCS · نظام مراجعة تعليقات الزوار</footer>
    </div>
  );
}

function Metric({ icon, label, value, state, note }: { icon: string; label: string; value: number | undefined; state: Resource<unknown>["status"]; note: string }) {
  return <article className="admin-kpi" aria-busy={state === "loading"}><div className="admin-kpi-top"><span>{label}</span><span className="admin-kpi-icon"><WorkspaceIcon name={icon} /></span></div><strong>{state === "ready" && value != null ? number.format(value) : "—"}</strong><p>{state === "error" ? "غير متاح حاليًا" : state === "loading" ? "جاري التحميل…" : note}</p></article>;
}

function ReviewTrend({ points }: { points: AnalyticsSnapshot["trend"] }) {
  const maximum = Math.max(1, ...points.flatMap(point => [point.incoming, point.reviewed]));
  return <>
    <div className="admin-chart-legend"><span><i className="admin-chart-incoming" />الواردة</span><span><i className="admin-chart-reviewed" />تمت مراجعتها</span></div>
    <div className="admin-chart" dir="ltr"><div className="admin-chart-axis" aria-hidden="true"><span>{number.format(maximum)}</span><span>{number.format(Math.round(maximum / 2))}</span><span>٠</span></div><div className="admin-chart-plot">{points.map((point, index) => <div key={point.bucket} className="admin-chart-day" tabIndex={0} role="img" aria-label={`${formatTrendBucket(point.bucket)}: الواردة ${number.format(point.incoming)}، تمت مراجعتها ${number.format(point.reviewed)}`}><div className="admin-chart-bars" title={`${formatTrendBucket(point.bucket)} — الواردة: ${point.incoming}، تمت مراجعتها: ${point.reviewed}`}><span className="admin-chart-incoming" style={{ height: `${point.incoming / maximum * 100}%` }} /><span className="admin-chart-reviewed" style={{ height: `${point.reviewed / maximum * 100}%` }} /></div><span className={`admin-chart-date ${index % 3 !== 0 && index !== points.length - 1 ? "admin-chart-date-quiet" : ""}`} aria-hidden="true">{number.format(Number(point.bucket.slice(8, 10)))}</span></div>)}</div></div>
    <p className="admin-footnote">الأيام بحسب توقيت الرياض. لا يمثل المخطط تاريخ حالات الإسناد.</p>
    <details className="admin-chart-details"><summary>عرض القيم اليومية</summary><div className="admin-table-scroll"><table className="admin-table"><caption>التعليقات الواردة والمراجعات حسب اليوم</caption><thead><tr><th scope="col">اليوم</th><th scope="col">الواردة</th><th scope="col">تمت مراجعتها</th></tr></thead><tbody>{points.map(point => <tr key={point.bucket}><th scope="row">{formatTrendBucket(point.bucket)}</th><td>{number.format(point.incoming)}</td><td>{number.format(point.reviewed)}</td></tr>)}</tbody></table></div></details>
  </>;
}
