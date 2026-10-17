import type { QueueItem, WorkflowCounts } from "../api/types";
import { ClassificationFilter } from "../components/ClassificationFilter";
import { ExtraFilters, extraFilterLabels, type ExtraFilterValue } from "../components/ExtraFilters";
import { EmptyState } from "../components/EmptyState";
import { INTENT_LABEL, TOPIC_LABEL, PRIORITY_LABEL, STATUS_LABEL } from "../labels";
import { firstName, greetingPrefix } from "../time/appTime";
import { formatAbsoluteAr, formatRelativeAr } from "../time/relative";
import "./ReviewerAssignedPage.css";
import { SystemState } from "../components/SystemState";
import { STATE_COPY } from "../components/stateCopy";

type WorkFilter = "pending" | "all" | "approved" | "no_reply" | "followup";
type LoadState = "loading" | "ready" | "error";
type Props = {
  displayName: string;
  counts: WorkflowCounts | null;
  countsState: LoadState;
  queue: QueueItem[];
  queueState: LoadState;
  total: number;
  queueFilter: WorkFilter;
  filters: { key: WorkFilter; label: string }[];
  search: string;
  sort: string;
  selectedIntents: string[];
  selectedTopics: string[];
  extra: ExtraFilterValue;
  onExtra: (next: ExtraFilterValue) => void;
  filtersActive: boolean;
  zeroAssigned: boolean;
  allCaughtUp: boolean;
  pageIndex: number;
  pages: number;
  onFilter: (filter: WorkFilter) => void;
  onSearch: (search: string) => void;
  onSort: (sort: string) => void;
  onClassification: (intents: string[], topics: string[]) => void;
  onReset: () => void;
  onPage: (page: number) => void;
  onRetryQueue: () => void;
  onRetryCounts: () => void;
  onStart: () => void;
};

/** Presentation only. Data, ownership, filters and review entry remain in ReviewerWorkspace. */
export function ReviewerAssignedPage(p: Props) {
  const activeLabels = [
    ...(p.search ? [`البحث: ${p.search}`] : []),
    ...p.selectedIntents.map(key => `النية: ${INTENT_LABEL[key] || key}`),
    ...p.selectedTopics.map(key => `الموضوع: ${TOPIC_LABEL[key] || key}`),
    ...extraFilterLabels(p.extra),
  ];
  const historyOnly = p.queueFilter === "approved" || p.queueFilter === "no_reply";
  const startDisabled = p.queueState !== "ready" || !p.queue.length;
  return <div className="reviewer-assigned" dir="rtl">
    <header className="assigned-heading"><div><p className="assigned-title">المسند لي</p><p>تعليقات الزوار المسندة إليك. اختر نطاق العمل ثم ابدأ المراجعة.</p></div><p className="dash-greeting">{greetingPrefix()}، {firstName(p.displayName)}</p></header>
    <section className="assigned-summary" aria-label="ملخص عملي">
      {p.countsState === "loading" ? <p role="status">جاري تحميل ملخص العمل…</p> : null}
      {p.countsState === "error" ? <p role="alert">تعذر تحميل ملخص العمل. <button type="button" onClick={p.onRetryCounts}>إعادة تحميل الملخص</button></p> : null}
      {p.countsState === "ready" && p.counts && p.counts.all > 0 ? <div className="work-summary-stats">
        <div><span>إجمالي المسند إليّ</span><strong>{p.counts.all}</strong></div>
        <div><span>بانتظار المراجعة</span><strong>{p.counts.pending}</strong></div>
        <div><span>مكتمل</span><strong>{p.counts.all - p.counts.pending}</strong></div>
        <div><span>نسبة الإنجاز</span><strong>{Math.round((p.counts.all - p.counts.pending) / p.counts.all * 100)}٪</strong></div>
      </div> : null}
      {p.countsState === "ready" && p.counts && p.counts.all > 0 ? <p className="assigned-summary-note">ملخص جميع التعليقات المسندة إليك؛ لا يتغير بتصفية القائمة. المكتمل هو ما لم يعد بانتظار المراجعة.</p> : null}
      {p.countsState === "ready" && p.counts?.all === 0 ? <p>لم تُسند إليك تعليقات بعد.</p> : null}
    </section>

    {p.zeroAssigned ? <section className="assigned-surface"><EmptyState {...STATE_COPY.assignedEmpty} /></section> : <>
      <section className="assigned-surface" aria-label="البحث والتصفية">
        <div className="assigned-filters">
          <label className="assigned-search"><span>البحث في التعليقات</span><input type="search" aria-label="بحث" placeholder="نص التعليق أو المعرّف" value={p.search} onChange={e => p.onSearch(e.target.value)} /></label>
          <div><span>النية والموضوع</span><ClassificationFilter selectedIntents={p.selectedIntents} selectedTopics={p.selectedTopics} onChange={p.onClassification} /></div>
          <label><span>ترتيب العرض</span><select aria-label="ترتيب" value={p.sort} onChange={e => p.onSort(e.target.value)}><option value="priority">حسب الأولوية</option><option value="oldest">الأقدم أولًا</option><option value="newest">الأحدث أولًا</option><option value="comment_newest">الأحدث تعليقًا</option><option value="comment_oldest">الأقدم تعليقًا</option></select></label>
          <ExtraFilters value={p.extra} onChange={p.onExtra} />
        </div>
        <div className="assigned-states" role="group" aria-label="تصفية القائمة">{p.filters.map(f => <button type="button" key={f.key} aria-pressed={p.queueFilter === f.key} onClick={() => p.onFilter(f.key)}>{f.label}</button>)}</div>
        <div className="assigned-filter-context"><div><span>{p.filters.find(f => f.key === p.queueFilter)?.label}</span>{activeLabels.map(text => <span key={text} className="assigned-tag">{text}</span>)}</div>{p.filtersActive ? <button type="button" onClick={p.onReset}>مسح الفلاتر</button> : null}</div>
      </section>

      <section className="assigned-surface assigned-list" aria-labelledby="assigned-list-title" aria-busy={p.queueState === "loading"}>
        <div className="assigned-list-heading"><div><h2 id="assigned-list-title">{p.queueState === "ready" ? `قائمة المراجعة (${p.total})` : "قائمة المراجعة"}</h2><p>{historyOnly ? "عرض قراراتك السابقة، دون إعادة فتحها كعمل نشط." : "تبدأ المراجعة من أول تعليق في القائمة وتتنقل عبر جميع التعليقات المطابقة، حسب التصفية والترتيب."}</p></div></div>
        {p.queueState === "loading" ? <SystemState kind="loading" className="assigned-state"><span aria-hidden="true" className="assigned-loading-mark" /><p>{STATE_COPY.queueLoading}</p></SystemState> : null}
        {p.queueState === "error" ? <SystemState kind="error" className="assigned-state"><h3>{STATE_COPY.queueFailed}</h3><p>الفلاتر محفوظة. حاول تحميل التعليقات مرة أخرى.</p><button type="button" onClick={p.onRetryQueue}>إعادة تحميل القائمة</button></SystemState> : null}
        {p.queueState === "ready" && !p.queue.length ? p.allCaughtUp && p.queueFilter === "pending" && !p.filtersActive ? <EmptyState {...STATE_COPY.reviewerCaughtUp} /> : <EmptyState kind="noResults" {...STATE_COPY.noResults} action={<button type="button" onClick={p.onReset}>مسح الفلاتر</button>} /> : null}
        {p.queueState === "ready" && p.queue.length > 0 ? <>
          <div className="assigned-column-labels" aria-hidden="true"><span>التعليق</span><span>النية والموضوع</span><span>الأولوية</span></div>
          <div className="assigned-rows" role="list" aria-label="التعليقات المسندة إليك">{p.queue.map(item => <article key={item.comment_id} className="queue-row assigned-row" role="listitem">
            <div className="assigned-comment"><header><span className="assigned-id">#{item.comment_id}</span><span className="assigned-status">{STATUS_LABEL[item.review_status] || item.review_status}</span></header><p className="assigned-preview">{item.comment_preview || "نص التعليق غير متاح"}</p><p className="assigned-time">{item.ingested_at || item.comment_created_at ? <time dateTime={item.ingested_at || item.comment_created_at || undefined} title={formatAbsoluteAr(item.ingested_at || item.comment_created_at)}>{formatRelativeAr(item.ingested_at || item.comment_created_at)}</time> : "تاريخ الورود غير متاح"}</p></div>
            <div className="assigned-classification"><span className="assigned-tag">{item.primary_intent ? INTENT_LABEL[item.primary_intent] || item.primary_intent : "النية غير متاحة"}</span><span>{item.topic_subtype ? TOPIC_LABEL[item.topic_subtype] || item.topic_subtype : "الموضوع غير متاح"}</span></div>
            <div className="assigned-priority"><span className="assigned-mobile-label">الأولوية</span><span className={item.review_priority === 1 ? "is-urgent" : ""}>{item.review_priority ? PRIORITY_LABEL[item.review_priority] || "غير متاح" : "غير متاح"}</span></div>
          </article>)}</div>
          <nav className="assigned-pagination" aria-label="صفحات المسند لي"><span>الصفحة {p.pageIndex + 1} من {p.pages}</span><div><button type="button" disabled={!p.pageIndex} onClick={() => p.onPage(p.pageIndex - 1)}>السابق</button><button type="button" disabled={p.pageIndex + 1 >= p.pages} onClick={() => p.onPage(p.pageIndex + 1)}>التالي</button></div></nav>
        </> : null}
      </section>
      <section className="assigned-start" aria-label="بدء العمل"><div><h3>{historyOnly ? "الاطلاع على المراجعات" : "مجموعة العمل الحالية"}</h3><p>{p.queueState === "ready" ? `${p.total} تعليقًا مطابقًا للفلاتر` : "بانتظار تحميل قائمة العمل"}</p></div><button type="button" disabled={startDisabled} onClick={p.onStart}>{historyOnly ? "عرض المراجعات" : "بدء المراجعة"}<span aria-hidden="true"> ←</span></button></section>
    </>}
  </div>;
}
