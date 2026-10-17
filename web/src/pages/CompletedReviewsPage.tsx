import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api/client";
import type { QueueItem, ReviewDetail, ReviewerSummaryRow } from "../api/types";
import { INTENT_LABEL, PRIORITY_LABEL, REVIEWER_ACTION_LABEL, STATUS_LABEL, TOPIC_LABEL } from "../labels";
import { ClassificationFilter } from "../components/ClassificationFilter";
import { EMPTY_EXTRA, ExtraFilters, extraFilterCount, extraFilterLabels, extraFilterParams, type ExtraFilterValue } from "../components/ExtraFilters";
import { EmptyState } from "../components/EmptyState";
import { formatAbsoluteAr } from "../time/relative";
import "./CompletedReviewsPage.css";

const PAGE_SIZE = 20;
const DECISION_KEYS = Object.keys(REVIEWER_ACTION_LABEL);
const label = (labels: Record<string, string>, key: string | null) => key ? labels[key] || key : "غير متاح";

export function CompletedReviewsPage() {
  const [reviewers, setReviewers] = useState<ReviewerSummaryRow[]>([]);
  const [reviewersState, setReviewersState] = useState<"loading" | "ready" | "error">("loading");
  const [reviewersRevision, setReviewersRevision] = useState(0);
  const [search, setSearch] = useState("");
  const [reviewerUserId, setReviewerUserId] = useState("");
  const [selectedIntents, setSelectedIntents] = useState<string[]>([]);
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [selectedDecisions, setSelectedDecisions] = useState<string[]>([]);
  const [extra, setExtra] = useState<ExtraFilterValue>(EMPTY_EXTRA);
  const extraKey = JSON.stringify(extra);
  const [sort, setSort] = useState("newest");
  const [offset, setOffset] = useState(0);
  const [revision, setRevision] = useState(0);
  const [items, setItems] = useState<QueueItem[]>([]);
  const [total, setTotal] = useState(0);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [openId, setOpenId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");
  const [exportCount, setExportCount] = useState("");
  const [exportNote, setExportNote] = useState("");
  const listRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (window.scrollY !== 0) window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setReviewersState("loading");
    api.adminReviewers().then(res => {
      if (!cancelled) { setReviewers(res.reviewers); setReviewersState("ready"); }
    }).catch(() => { if (!cancelled) setReviewersState("error"); });
    return () => { cancelled = true; };
  }, [reviewersRevision]);

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    api.queue({ filter: "completed", sort, search, offset, limit: PAGE_SIZE, ...extraFilterParams(extra),
      primaryIntents: selectedIntents, topicSubtypes: selectedTopics,
      reviewerActions: selectedDecisions, reviewerUserId: reviewerUserId || undefined,
    }).then(page => {
      if (cancelled) return;
      if (offset > 0 && offset >= page.total) {
        setOffset(Math.max(0, Math.ceil(page.total / PAGE_SIZE) - 1) * PAGE_SIZE);
        return;
      }
      setItems(page.items); setTotal(page.total); setState("ready");
    }).catch(() => { if (!cancelled) setState("error"); });
    return () => { cancelled = true; };
  }, [search, reviewerUserId, selectedIntents, selectedTopics, selectedDecisions, extraKey, sort, offset, revision]);

  function changeFilters(change: () => void) {
    change(); setOffset(0); setExportError(""); setExportNote("");
  }
  function clearFilters() {
    changeFilters(() => { setSearch(""); setReviewerUserId(""); setSelectedIntents([]); setSelectedTopics([]); setSelectedDecisions([]); setExtra(EMPTY_EXTRA); setSort("newest"); });
  }
  // Server-side export of the persisted completed reviews matching the current filters, across ALL pages; an empty count means every match.
  async function exportCurrentResults() {
    if (state !== "ready" || !total || exporting) return;
    const typed = exportCount.trim();
    const count = Number(typed);
    if (typed && (!Number.isInteger(count) || count < 1)) { setExportError("أدخل عددًا صحيحًا موجبًا."); return; }
    if (typed && count > total) { setExportError(`العدد المطلوب أكبر من المتاح (${total}).`); return; }
    setExporting(true); setExportError(""); setExportNote("");
    try {
      const summary = await api.downloadCompletedExport({
        ...(typed ? { count } : { all: true }),
        search: search || undefined, reviewerUserId: reviewerUserId || undefined, primaryIntents: selectedIntents, topicSubtypes: selectedTopics,
        reviewerActions: selectedDecisions, ...extraFilterParams(extra),
      });
      setExportNote(`تم تصدير ${summary.rows} من ${summary.matched} مراجعة مكتملة مطابقة${summary.shortfall ? ` — النقص ${summary.shortfall}` : ""}.`);
    }
    catch { setExportError("تعذر تصدير المراجعات المكتملة. حاول مجددًا."); }
    finally { setExporting(false); }
  }
  function openDetail(id: string) {
    // Give pointer users the same focus return target as keyboard users.
    document.getElementById(`completed-open-${id}`)?.focus();
    setOpenId(id);
  }
  const filtersActive = Boolean(search || reviewerUserId || selectedIntents.length || selectedTopics.length || selectedDecisions.length || extraFilterCount(extra) || sort !== "newest");
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const pageIndex = Math.floor(offset / PAGE_SIZE);
  const activeLabels = [
    ...(search ? [`البحث: ${search}`] : []),
    ...(reviewerUserId ? [`المراجع: ${reviewers.find(r => r.user_id === reviewerUserId)?.display_name || "المحدد"}`] : []),
    ...selectedIntents.map(k => `النية: ${INTENT_LABEL[k]}`),
    ...selectedTopics.map(k => `الموضوع: ${TOPIC_LABEL[k]}`),
    ...selectedDecisions.map(k => `القرار: ${REVIEWER_ACTION_LABEL[k]}`),
    ...extraFilterLabels(extra),
  ];

  return (
    <div className="completed-ui completed-workspace" dir="rtl">
      <header className="completed-heading">
        <div><h2>المراجعات المكتملة</h2><p>التعليقات التي اكتملت مراجعتها البشرية، وردودها النهائية.</p></div>
        <div className="completed-export"><label><span>عدد المراجعات (اتركه فارغًا لتصدير الكل)</span><input type="number" min={1} max={Math.max(total, 1)} inputMode="numeric" aria-label="عدد المراجعات المطلوب تصديرها" value={exportCount} disabled={exporting} onChange={e => { setExportCount(e.target.value); setExportError(""); }} /></label>
        <button type="button" disabled={exporting || state !== "ready" || !total} onClick={() => void exportCurrentResults()}>{exporting ? "جارٍ التصدير…" : `تصدير المراجعات المكتملة (${state === "ready" ? (exportCount.trim() && Number.isInteger(Number(exportCount)) && Number(exportCount) >= 1 ? Math.min(Number(exportCount), total) : total) : "—"})`}</button></div>
      </header>
      {exportError ? <p className="completed-feedback" role="alert">{exportError}</p> : null}
      {exportNote ? <p className="completed-feedback" role="status">{exportNote}</p> : null}

      <section className="completed-surface" aria-label="البحث والتصفية">
        <div className="completed-filters">
          <label className="completed-search"><span>البحث في التعليقات</span><input type="search" aria-label="بحث" placeholder="ابحث في نص التعليق" value={search} onChange={e => changeFilters(() => setSearch(e.target.value))} /></label>
          <label><span>المراجع</span><select aria-label="اختر المراجع" value={reviewerUserId} disabled={reviewersState !== "ready"} onChange={e => changeFilters(() => setReviewerUserId(e.target.value))}><option value="">{reviewersState === "loading" ? "جاري تحميل المراجعين…" : reviewersState === "error" ? "قائمة المراجعين غير متاحة" : "جميع المراجعين"}</option>{reviewers.map(r => <option key={r.user_id} value={r.user_id}>{r.display_name}</option>)}</select></label>
          <div><span>النية والموضوع</span><ClassificationFilter selectedIntents={selectedIntents} selectedTopics={selectedTopics} onChange={(intents, topics) => changeFilters(() => { setSelectedIntents(intents); setSelectedTopics(topics); })} /></div>
          <ExtraFilters value={extra} onChange={next => changeFilters(() => setExtra(next))} />
          <label><span>ترتيب العرض</span><select aria-label="ترتيب" value={sort} onChange={e => changeFilters(() => setSort(e.target.value))}><option value="newest">الأحدث إضافةً</option><option value="oldest">الأقدم إضافةً</option><option value="comment_newest">الأحدث تعليقًا</option><option value="comment_oldest">الأقدم تعليقًا</option></select></label>
        </div>
        {reviewersState === "error" ? <p className="completed-filter-error" role="alert">تعذر تحميل قائمة المراجعين. <button type="button" onClick={() => setReviewersRevision(n => n + 1)}>إعادة المحاولة</button></p> : null}
        <div className="completed-decisions" role="group" aria-label="القرار النهائي"><span>القرار النهائي</span>{DECISION_KEYS.map(key => <button key={key} type="button" aria-pressed={selectedDecisions.includes(key)} onClick={() => changeFilters(() => setSelectedDecisions(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]))}>{REVIEWER_ACTION_LABEL[key]}</button>)}</div>
        {filtersActive ? <div className="completed-filter-context"><div aria-label="الفلاتر النشطة">{activeLabels.map(text => <span className="completed-tag" key={text}>{text}</span>)}</div><button type="button" onClick={clearFilters}>مسح الفلاتر</button></div> : null}
      </section>

      <section className="completed-surface completed-list" ref={listRef} aria-labelledby="completed-list-title" aria-busy={state === "loading"}>
        <div className="completed-list-heading"><div><h3 id="completed-list-title">سجل المراجعات</h3><p>اختر تعليقًا للاطلاع على تفاصيل مراجعته</p></div><span role="status">{state === "ready" ? `${total} ${filtersActive ? "نتيجة مطابقة" : "مراجعة مكتملة"}` : state === "loading" ? "جاري التحميل…" : "غير متاح"}</span></div>
        {state === "loading" ? <div className="completed-state" role="status"><span className="completed-loading-mark" aria-hidden="true" /><p>جاري تحميل المراجعات المكتملة…</p></div> : null}
        {state === "error" ? <div className="completed-state" role="alert"><h3>تعذر تحميل المراجعات المكتملة</h3><p>حاول مجددًا. ستبقى الفلاتر التي اخترتها محفوظة.</p><button type="button" onClick={() => setRevision(n => n + 1)}>إعادة تحميل المراجعات</button></div> : null}
        {state === "ready" && !items.length ? <EmptyState title={filtersActive ? "لا توجد مراجعات تطابق البحث والتصفية" : "لا توجد مراجعات مكتملة بعد"} description={filtersActive ? "جرّب تغيير البحث أو مسح الفلاتر لعرض المراجعات المتاحة." : "ستظهر هنا التعليقات بعد اكتمال مراجعتها البشرية."} action={filtersActive ? <button type="button" onClick={clearFilters}>مسح الفلاتر</button> : undefined} /> : null}
        {state === "ready" && items.length > 0 ? <>
          <table className="completed-table" aria-label="المراجعات المكتملة">
            <thead><tr><th scope="col">التعليق</th><th scope="col">التصنيف</th><th scope="col">المراجع</th><th scope="col">تاريخ المراجعة</th><th scope="col">القرار النهائي</th></tr></thead>
            <tbody>{items.map(item => <tr key={item.comment_id} onClick={() => openDetail(item.comment_id)}>
              <td className="completed-comment"><button id={`completed-open-${item.comment_id}`} type="button" aria-label={`عرض مراجعة ${item.comment_id}`} aria-haspopup="dialog" onClick={e => { e.stopPropagation(); openDetail(item.comment_id); }}><span className="completed-id"><bdi>#{item.comment_id}</bdi></span><span className="completed-preview">{item.comment_preview || "نص التعليق غير متاح"}</span><span className="completed-open-hint">عرض المراجعة <span aria-hidden="true">←</span></span></button></td>
              <td data-label="التصنيف"><div className="completed-classification"><span>{label(INTENT_LABEL, item.primary_intent)}</span><small>{label(TOPIC_LABEL, item.topic_subtype)}</small></div></td>
              <td data-label="المراجع">{item.reviewer_display_name || "غير متاح"}</td>
              <td data-label="تاريخ المراجعة" className="completed-date">{item.reviewed_at ? <time dateTime={item.reviewed_at}>{formatAbsoluteAr(item.reviewed_at)}</time> : "غير متاح"}</td>
              <td data-label="القرار النهائي"><span className={`completed-outcome${item.reviewer_action === "NO_RESPONSE_NEEDED" ? " is-neutral" : ""}`}>{label(REVIEWER_ACTION_LABEL, item.reviewer_action)}</span></td>
            </tr>)}</tbody>
          </table>
          <nav className="completed-pagination" aria-label="صفحات المراجعات"><span>الصفحة {pageIndex + 1} من {pages} · {offset + 1}–{offset + items.length} من {total}</span><div><button type="button" disabled={offset === 0} onClick={() => { setOffset(Math.max(0, offset - PAGE_SIZE)); listRef.current?.scrollIntoView?.({ block: "start" }); }}>السابق</button><button type="button" disabled={pageIndex + 1 >= pages} onClick={() => { setOffset(offset + PAGE_SIZE); listRef.current?.scrollIntoView?.({ block: "start" }); }}>التالي</button></div></nav>
        </> : null}
      </section>
      {openId ? <ReviewDrawer commentId={openId} onClose={() => setOpenId(null)} /> : null}
    </div>
  );
}

function ReviewDrawer({ commentId, onClose }: { commentId: string; onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [item, setItem] = useState<ReviewDetail | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const dialog = dialogRef.current!;
    const trigger = document.getElementById(`completed-open-${commentId}`);
    const overflow = document.body.style.overflow;
    dialog.showModal(); document.body.style.overflow = "hidden";
    return () => { dialog.close(); document.body.style.overflow = overflow; trigger?.focus(); };
  }, []);
  useEffect(() => {
    let cancelled = false;
    setItem(null); setState("loading");
    api.item(commentId).then(result => { if (!cancelled) { setItem(result); setState("ready"); } })
      .catch(() => { if (!cancelled) setState("error"); });
    return () => { cancelled = true; };
  }, [commentId, revision]);

  return createPortal(<dialog ref={dialogRef} className="completed-ui completed-drawer" dir="rtl" aria-labelledby="completed-detail-title" onCancel={e => { e.preventDefault(); onClose(); }} onClick={e => { if (e.target === e.currentTarget) { const bounds = e.currentTarget.getBoundingClientRect(); if (e.clientX > bounds.right || e.clientX < bounds.left) onClose(); } }}>
    <header className="completed-drawer-header"><div><p>المراجعات المكتملة <span aria-hidden="true">/</span> <bdi>#{commentId}</bdi></p><h2 id="completed-detail-title">تفاصيل المراجعة</h2></div><button type="button" autoFocus onClick={onClose} aria-label="إغلاق تفاصيل المراجعة">إغلاق <span aria-hidden="true">×</span></button></header>
    <div className="completed-drawer-body" aria-busy={state === "loading"}>
      {state === "loading" ? <p className="completed-state" role="status">جاري تحميل تفاصيل المراجعة…</p> : null}
      {state === "error" ? <div className="completed-state" role="alert"><h3>تعذر تحميل تفاصيل التعليق</h3><p>يمكنك إعادة المحاولة أو العودة إلى السجل.</p><button type="button" onClick={() => setRevision(n => n + 1)}>إعادة تحميل التفاصيل</button></div> : null}
      {item ? <>
        <section className="completed-detail-section"><h3>تعليق الزائر</h3><p className="completed-text completed-original" dir="auto">{item.original_comment || "نص التعليق غير متاح"}</p>{item.original_title ? <p className="completed-context">المدخل: <bdi>{item.original_title}</bdi></p> : null}</section>
        <section className="completed-detail-section"><h3>التصنيف</h3><dl className="completed-meta"><div><dt>النية</dt><dd>{label(INTENT_LABEL, item.primary_intent)}</dd></div><div><dt>الموضوع</dt><dd>{label(TOPIC_LABEL, item.topic_subtype)}</dd></div><div><dt>الأولوية</dt><dd>{item.review_priority ? PRIORITY_LABEL[item.review_priority] || "غير متاح" : "غير متاح"}</dd></div></dl>{item.priority_reason ? <p className="completed-context">سبب الأولوية: {item.priority_reason}</p> : null}</section>
        <section className="completed-detail-section"><h3>مسودة الرد المقترحة</h3><p className="completed-caption">المسودة المتاحة من الوكيل</p><p className="completed-text" dir="auto">{item.suggested_response || "لا توجد مسودة رد متاحة لهذا التعليق."}</p></section>
        <section className="completed-detail-section completed-human"><h3>المراجعة البشرية</h3><dl className="completed-meta"><div><dt>المراجع</dt><dd>{item.reviewer_display_name || "غير متاح"}</dd></div><div><dt>تاريخ المراجعة</dt><dd>{item.reviewed_at ? <time dateTime={item.reviewed_at}>{formatAbsoluteAr(item.reviewed_at)}</time> : "غير متاح"}</dd></div><div><dt>القرار النهائي</dt><dd>{label(REVIEWER_ACTION_LABEL, item.reviewer_action)}</dd></div><div><dt>الحالة</dt><dd>{label(STATUS_LABEL, item.review_status)}</dd></div></dl><h4>الرد النهائي المسجل</h4><p className="completed-text completed-final" dir="auto">{item.final_response || (item.reviewer_action === "NO_RESPONSE_NEEDED" ? "اكتملت المراجعة بقرار عدم الحاجة إلى رد." : "لا يوجد رد نهائي مسجل متاح.")}</p>{item.no_response_reason ? <p className="completed-context">سبب عدم الرد: {item.no_response_reason}</p> : null}</section>
        <p className="completed-drawer-note">المعروض هو المسودة المتاحة ونتيجة المراجعة المسجلة، وليس سجلًا كاملًا لتغييرات الرد.</p>
      </> : null}
    </div>
  </dialog>, document.body);
}
