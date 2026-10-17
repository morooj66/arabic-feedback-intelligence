import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { AssignableItem, AssignByFilterResult, AssignResult, ReviewerSummaryRow, SourceFileSummary } from "../api/types";
import { INTENT_LABEL, PRIORITY_LABEL, STATUS_LABEL, TOPIC_LABEL } from "../labels";
import { ClassificationFilter } from "../components/ClassificationFilter";
import { EMPTY_EXTRA, ExtraFilters, extraFilterCount, extraFilterLabels, extraFilterParams, type ExtraFilterValue } from "../components/ExtraFilters";
import { EmptyState } from "../components/EmptyState";
import "./AssignmentPage.css";
import { formatAbsoluteAr } from "../time/relative";
import { ReviewerProgressPage } from "./ReviewerProgressPage";

const PAGE_SIZE = 20;

export function AssignmentPage({ initialSourceFile }: { initialSourceFile?: string | null }) {
  const [sources, setSources] = useState<SourceFileSummary[] | null>(null);
  const [reviewers, setReviewers] = useState<ReviewerSummaryRow[] | null>(null);
  const [loadError, setLoadError] = useState("");

  const [search, setSearch] = useState("");
  const [sourceFile, setSourceFile] = useState(initialSourceFile || "");
  const [unassignedOnly, setUnassignedOnly] = useState(true);
  const [selectedIntents, setSelectedIntents] = useState<string[]>([]);
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [extra, setExtra] = useState<ExtraFilterValue>(EMPTY_EXTRA);
  const extraKey = JSON.stringify(extra);
  const [assignedTo, setAssignedTo] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [sort, setSort] = useState("oldest");
  const [offset, setOffset] = useState(0);

  const [items, setItems] = useState<AssignableItem[]>([]);
  const [total, setTotal] = useState(0);
  const [itemsState, setItemsState] = useState<"loading" | "ready" | "error">("loading");

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [allFilteredSelected, setAllFilteredSelected] = useState(false);
  const [targetReviewer, setTargetReviewer] = useState("");
  const requestId = useRef(0);
  const listRef = useRef<HTMLElement>(null);
  const [selectingAll, setSelectingAll] = useState(false);
  const [summaryRevision, setSummaryRevision] = useState(0);
  const [assigning, setAssigning] = useState(false);
  const [assignError, setAssignError] = useState("");
  const [result, setResult] = useState<AssignResult | null>(null);
  const [countText, setCountText] = useState("");
  const [countResult, setCountResult] = useState<AssignByFilterResult | null>(null);

  useEffect(() => {
    // Enter this workspace at its heading even when opened from a lower Home action.
    if (window.scrollY !== 0) window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoadError("");
    setSources(null);
    setReviewers(null);
    Promise.all([api.adminSourceSummary(), api.adminReviewers()])
      .then(([sourceRes, reviewerRes]) => {
        if (cancelled) return;
        setSources(sourceRes.sources);
        setReviewers(reviewerRes.reviewers);
      })
      .catch(() => {
        if (!cancelled) setLoadError("تعذر تحميل قائمة الملفات أو المراجعين.");
      });
    return () => {
      cancelled = true;
    };
  }, [summaryRevision]);

  async function loadItems() {
    const id = ++requestId.current;
    setItemsState("loading");
    try {
      const page = await api.adminAssignableItems({
        unassignedOnly,
        sourceFile: sourceFile || undefined,
        offset,
        limit: PAGE_SIZE,
        primaryIntents: selectedIntents,
        topicSubtypes: selectedTopics,
        search: search || undefined,
        ...extraFilterParams(extra),
        reviewerUserId: assignedTo || undefined,
        statuses: statusFilter ? [statusFilter] : undefined,
        sort,
      });
      if (id !== requestId.current) return;
      if (page.total > 0 && offset >= page.total) {
        setOffset(Math.floor((page.total - 1) / PAGE_SIZE) * PAGE_SIZE);
        return;
      }
      if (page.total === 0 && offset > 0) { setOffset(0); return; }
      setItems(page.items);
      setTotal(page.total);
      setItemsState("ready");
    } catch {
      if (id === requestId.current) setItemsState("error");
    }
  }

  useEffect(() => {
    setOffset(0);
    setAllFilteredSelected(false);
    setSelected(new Set());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, sourceFile, unassignedOnly, selectedIntents, selectedTopics, extraKey, assignedTo, statusFilter, sort]);

  useEffect(() => {
    void loadItems();
    return () => { requestId.current += 1; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, sourceFile, unassignedOnly, selectedIntents, selectedTopics, extraKey, assignedTo, statusFilter, sort, offset]);

  function toggleOne(id: string) {
    setAllFilteredSelected(false);
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectPage() {
    setAllFilteredSelected(false);
    setSelected((prev) => {
      const next = new Set(prev);
      items.forEach((i) => next.add(i.comment_id));
      return next;
    });
  }

  async function selectAllFiltered() {
    // Preserve explicit cross-page selection without requesting an unbounded payload.
    const id = requestId.current;
    setSelectingAll(true);
    setAssignError("");
    try {
      const ids = new Set<string>();
      let cursor = 0;
      let remaining = total;
      while (cursor < remaining) {
        const page = await api.adminAssignableItems({
          unassignedOnly, sourceFile: sourceFile || undefined,
          offset: cursor, limit: PAGE_SIZE,
          primaryIntents: selectedIntents, topicSubtypes: selectedTopics,
          search: search || undefined,
          ...extraFilterParams(extra),
          reviewerUserId: assignedTo || undefined,
          statuses: statusFilter ? [statusFilter] : undefined,
          sort,
        });
        if (id !== requestId.current) return;
        if (page.items.length === 0) throw new Error("selection_changed");
        page.items.forEach(item => ids.add(item.comment_id));
        cursor += page.items.length;
        remaining = page.total;
      }
      setSelected(ids);
      setAllFilteredSelected(true);
    } catch {
      setAssignError("تعذر تحديد جميع نتائج الفلترة. التحديد السابق محفوظ؛ حاول مجددًا.");
    } finally {
      setSelectingAll(false);
    }
  }

  function clearSelection() {
    setSelected(new Set());
    setAllFilteredSelected(false);
  }

  async function assignSelected() {
    if (!targetReviewer || selected.size === 0) return;
    setAssigning(true);
    setAssignError("");
    setResult(null);
    setCountResult(null);
    try {
      const outcome = await api.adminAssign(Array.from(selected), targetReviewer);
      setResult(outcome);
      clearSelection();
      if (window.scrollY !== 0) window.scrollTo(0, 0);
      setSummaryRevision(n => n + 1);
      await loadItems();
    } catch {
      setAssignError("تعذّر تنفيذ الإسناد. التحديد محفوظ؛ تحقق من الاتصال ثم أعد المحاولة.");
    } finally {
      setAssigning(false);
    }
  }

  // Count-based assignment runs on the server over EVERY matching unassigned comment (not only the visible page); it never moves existing assignments.
  async function assignByCount(all: boolean) {
    if (!targetReviewer) return;
    const count = Number(countText);
    if (!all && (!countText.trim() || !Number.isInteger(count) || count < 1)) { setAssignError("أدخل عددًا صحيحًا موجبًا."); return; }
    if (!all && count > total) { setAssignError(`العدد المطلوب أكبر من المتاح (${total}).`); return; }
    setAssigning(true);
    setAssignError("");
    setResult(null);
    setCountResult(null);
    try {
      const outcome = await api.adminAssignByFilter({
        reviewerUserId: targetReviewer, ...(all ? { all: true } : { count }),
        sourceFile: sourceFile || undefined, primaryIntents: selectedIntents, topicSubtypes: selectedTopics, search: search || undefined,
        ...extraFilterParams(extra), statuses: statusFilter ? [statusFilter] : undefined, sort,
      });
      setCountResult(outcome);
      setCountText("");
      clearSelection();
      if (window.scrollY !== 0) window.scrollTo(0, 0);
      setSummaryRevision(n => n + 1);
      await loadItems();
    } catch {
      setAssignError("تعذّر تنفيذ الإسناد بالعدد. تحقق من الاتصال ثم أعد المحاولة.");
    } finally {
      setAssigning(false);
    }
  }

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const pageIndex = Math.floor(offset / PAGE_SIZE);
  const noReviewers = reviewers !== null && reviewers.length === 0;
  const filtersActive = Boolean(search) || Boolean(sourceFile) || !unassignedOnly || selectedIntents.length > 0 || selectedTopics.length > 0 || extraFilterCount(extra) > 0 || Boolean(assignedTo) || Boolean(statusFilter);

  function clearFilters() {
    setSearch("");
    setSourceFile("");
    setUnassignedOnly(true);
    setSelectedIntents([]);
    setSelectedTopics([]);
    setExtra(EMPTY_EXTRA);
    setAssignedTo("");
    setStatusFilter("");
    setSort("oldest");
  }

  const busy = assigning || selectingAll;
  const chosenReviewer = reviewers?.find(r => r.user_id === targetReviewer);
  const hiddenSelection = selected.size - items.filter(item => selected.has(item.comment_id)).length;
  const pageSelected = items.length > 0 && items.every(item => selected.has(item.comment_id));
  const somePageSelected = items.some(item => selected.has(item.comment_id));
  const unassignedTotal = sources?.reduce((sum, source) => sum + source.unassigned, 0);

  return (
    <div className="assignment-workspace" dir="rtl">
      <header className="assignment-heading">
        <div><h2>الإسناد</h2><p>إسناد التعليقات غير المسندة إلى المراجعين</p></div>
        <div className="assignment-summary"><span>غير مسندة <strong>{unassignedTotal ?? "—"}</strong></span><span>المحدد حاليًا <strong>{selected.size}</strong></span></div>
      </header>
      {loadError ? <p className="assignment-feedback" role="alert">{loadError} <button type="button" disabled={busy} onClick={() => setSummaryRevision(n => n + 1)}>إعادة المحاولة</button></p> : null}
      {noReviewers ? <p className="assignment-feedback" role="status">لا يوجد مراجعون نشطون لإسناد العناصر إليهم.</p> : null}
      {assignError ? <p className="assignment-feedback" role="alert">{assignError}</p> : null}
      {countResult ? <div className="assignment-feedback assignment-success" role="status">
        <p>{countResult.requested === null ? <>تم إسناد <strong>{countResult.assigned}</strong> من جميع التعليقات المطابقة.</> : <>تم إسناد <strong>{countResult.assigned}</strong> من <strong>{countResult.requested}</strong> المطلوبة.</>}</p>
        {countResult.shortfall > 0 ? <p>النقص {countResult.shortfall} لعدم توفر تعليقات مؤهلة كافية.</p> : null}
        <p>المتبقي غير مسند ضمن التصفية: {countResult.remaining_eligible}.</p>
      </div> : null}
      {result ? <div className="assignment-feedback assignment-success" role="status">
        <p>{result.assigned.length ? <>تم إسناد <strong>{result.assigned.length}</strong> عنصر بنجاح.</> : "لم يُسند أي تعليق في هذه المحاولة."}</p>
        {result.skipped_already_decided.length ? <p>تم تجاوز {result.skipped_already_decided.length} عنصر لأنه تم اتخاذ قرار مراجعة بشأنها مسبقًا.</p> : null}
        {result.skipped_not_reviewable.length ? <p>تم تجاوز {result.skipped_not_reviewable.length} عنصر لأنها غير قابلة للمراجعة حاليًا.</p> : null}
        {result.not_found.length ? <p>تعذّر العثور على {result.not_found.length} عنصر.</p> : null}
      </div> : null}

      <section className="assignment-surface" aria-label="البحث والتصفية">
        <fieldset className="assignment-filters" disabled={busy}>
          <legend className="assignment-sr-only">البحث والتصفية</legend>
          <label className="assignment-search"><span>البحث في التعليقات</span><input type="search" aria-label="بحث" placeholder="ابحث في نص التعليق أو المعرّف" value={search} onChange={e => setSearch(e.target.value)} /></label>
          <label className="assignment-source"><span>ملف الاستيراد</span><select aria-label="اختر الملف" value={sourceFile} onChange={e => setSourceFile(e.target.value)}><option value="">كل الملفات</option>{sourceFile && !sources?.some(s => s.source_file === sourceFile) ? <option value={sourceFile}>{sourceFile}</option> : null}{(sources || []).map(s => <option key={s.source_file} value={s.source_file}>{s.source_file} ({s.unassigned} غير مسند)</option>)}</select></label>
          <div className="assignment-classification"><span>النية والموضوع</span><ClassificationFilter selectedIntents={selectedIntents} selectedTopics={selectedTopics} onChange={(intents, topics) => { setSelectedIntents(intents); setSelectedTopics(topics); }} /></div>
          <label className="assignment-unassigned"><input type="checkbox" checked={unassignedOnly} onChange={e => { setUnassignedOnly(e.target.checked); if (e.target.checked) setAssignedTo(""); }} /><span>غير المسند فقط</span></label>
          <label><span>المسند إلى</span><select aria-label="المسند إلى" value={assignedTo} disabled={unassignedOnly} onChange={e => setAssignedTo(e.target.value)}><option value="">أي مراجع</option>{(reviewers || []).map(r => <option key={r.user_id} value={r.user_id}>{r.display_name}</option>)}</select></label>
          <label><span>حالة المراجعة</span><select aria-label="حالة المراجعة" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}><option value="">كل الحالات القابلة للإسناد</option>{[["DRAFT_READY", "مسودة جاهزة"], ["REVIEW_ONLY", "مراجعة بدون مسودة"], ["IN_REVIEW", STATUS_LABEL.IN_REVIEW], ["NO_RESPONSE_RECOMMENDED", STATUS_LABEL.NO_RESPONSE_RECOMMENDED]].map(([k, text]) => <option key={k} value={k}>{text}</option>)}</select></label>
          <label><span>ترتيب العرض</span><select aria-label="ترتيب" value={sort} onChange={e => setSort(e.target.value)}><option value="oldest">الأقدم إضافة</option><option value="newest">الأحدث إضافة</option><option value="comment_newest">الأحدث تعليقًا</option><option value="comment_oldest">الأقدم تعليقًا</option><option value="priority">حسب الأولوية</option></select></label>
          <ExtraFilters value={extra} onChange={setExtra} />
        </fieldset>
        <div className="assignment-filter-context"><p>{unassignedOnly ? "تُعرض التعليقات غير المسندة القابلة للمراجعة" : "تُعرض جميع التعليقات القابلة للإسناد، بما فيها المسندة"}{sourceFile ? <> · الملف: <bdi>{sourceFile}</bdi></> : " · كل الملفات"}{search ? <> · البحث: «{search}»</> : null}{selectedIntents.map(key => <span className="assignment-tag" key={key}>{INTENT_LABEL[key] || key}</span>)}{selectedTopics.map(key => <span className="assignment-tag" key={key}>{TOPIC_LABEL[key] || key}</span>)}{extraFilterLabels(extra).map(text => <span className="assignment-tag" key={text}>{text}</span>)}{assignedTo ? <span className="assignment-tag">المسند إلى: {reviewers?.find(r => r.user_id === assignedTo)?.display_name || "المحدد"}</span> : null}{statusFilter ? <span className="assignment-tag">{statusFilter === "DRAFT_READY" ? "مسودة جاهزة" : statusFilter === "REVIEW_ONLY" ? "مراجعة بدون مسودة" : STATUS_LABEL[statusFilter] || statusFilter}</span> : null}</p>{filtersActive ? <button type="button" disabled={busy} onClick={clearFilters}>مسح الفلاتر</button> : null}</div>
      </section>

      <section ref={listRef} className="assignment-surface assignment-list" aria-labelledby="assignment-list-title">
        <div className="assignment-list-heading"><h3 id="assignment-list-title">{unassignedOnly ? "التعليقات غير المسندة" : "التعليقات القابلة للإسناد"}</h3><span>{itemsState === "ready" ? `${total} تعليقًا` : itemsState === "loading" ? "جاري التحميل…" : "غير متاح"}</span></div>
        <fieldset className="assignment-selection" disabled={busy || itemsState !== "ready"}>
          <legend className="assignment-sr-only">تحديد التعليقات</legend>
          <span role="status">{selected.size ? `تم تحديد ${selected.size} تعليقات` : "حدد التعليقات لإسنادها إلى مراجع"}</span>
          <div><button type="button" onClick={selectPage} disabled={!items.length}>تحديد الصفحة الحالية ({itemsState === "ready" ? items.length : "—"})</button>{total > items.length && itemsState === "ready" ? <button type="button" onClick={() => void selectAllFiltered()} disabled={allFilteredSelected}>تحديد جميع نتائج الفلترة ({total})</button> : null}{selected.size ? <button type="button" onClick={clearSelection}>إلغاء التحديد</button> : null}</div>
        </fieldset>
        {selectingAll ? <p className="assignment-state" role="status">جاري تحديد نتائج الفلترة عبر الصفحات…</p> : null}
        {!noReviewers ? <section className="assignment-action" aria-labelledby="assignment-count-title">
          <div><div className="assignment-action-heading"><h3 id="assignment-count-title">إسناد بالعدد</h3></div><p>{unassignedOnly ? `يُسند من التعليقات غير المسندة المطابقة للتصفية (${itemsState === "ready" ? total : "—"}) بحسب الترتيب المختار، عبر جميع الصفحات.` : "الإسناد بالعدد يشمل التعليقات غير المسندة فقط؛ فعّل «غير المسندة فقط»."}</p></div>
          <div className="assignment-action-controls"><label><span>المراجع</span><select aria-label="المراجع للإسناد بالعدد" value={targetReviewer} disabled={busy || !reviewers?.length} onChange={e => setTargetReviewer(e.target.value)}><option value="">{reviewers === null ? "قائمة المراجعين غير متاحة بعد" : "اختر مراجعًا"}</option>{(reviewers || []).map(r => <option key={r.user_id} value={r.user_id}>{r.display_name} — {r.pending} قيد الإنجاز</option>)}</select></label><label><span>العدد</span><input type="number" min={1} max={Math.max(total, 1)} inputMode="numeric" aria-label="عدد التعليقات المطلوب إسنادها" value={countText} disabled={busy || !unassignedOnly} onChange={e => setCountText(e.target.value)} /></label><button type="button" className="assignment-primary" disabled={busy || !chosenReviewer || !unassignedOnly || itemsState !== "ready" || total === 0 || !countText.trim()} onClick={() => void assignByCount(false)}>{assigning ? "جاري الإسناد…" : "إسناد العدد"}</button><button type="button" disabled={busy || !chosenReviewer || !unassignedOnly || itemsState !== "ready" || total === 0} onClick={() => void assignByCount(true)}>{`إسناد كل المطابقة (${itemsState === "ready" ? total : "—"})`}</button></div>
        </section> : null}
        {selected.size > 0 ? <section className="assignment-action" aria-labelledby="assignment-action-title">
          <div><div className="assignment-action-heading"><h3 id="assignment-action-title">إسناد المحدد ({selected.size})</h3><button type="button" disabled={busy} onClick={clearSelection} aria-label="إلغاء تحديد التعليقات">إلغاء</button></div><p>{hiddenSelection > 0 ? `يشمل ${hiddenSelection} تعليقات محددة في صفحات أخرى. ` : ""}اختر المراجع ثم أكّد الإسناد.</p>{!unassignedOnly ? <p>العناصر المسندة المحددة ستُنقل إلى المراجع المختار.</p> : null}</div>
          <div className="assignment-action-controls"><label><span>المراجع</span><select aria-label="اختر المراجع" value={targetReviewer} disabled={busy || !reviewers?.length} onChange={e => setTargetReviewer(e.target.value)}><option value="">{noReviewers ? "لا يوجد مراجعون نشطون" : reviewers === null ? "قائمة المراجعين غير متاحة بعد" : "اختر مراجعًا"}</option>{(reviewers || []).map(r => <option key={r.user_id} value={r.user_id}>{r.display_name} — {r.pending} قيد الإنجاز</option>)}</select></label><button type="button" className="assignment-primary" disabled={busy || !chosenReviewer || itemsState !== "ready"} onClick={() => void assignSelected()}>{assigning ? "جاري الإسناد…" : "إسناد المحدد"}</button></div>
          {chosenReviewer ? <p className="assignment-workload">عبء المراجع الحالي: {chosenReviewer.assigned_total} مسند · {chosenReviewer.pending} قيد الإنجاز · {chosenReviewer.completed} مكتمل</p> : null}
        </section> : null}

        {itemsState === "loading" ? <p className="assignment-state" role="status">جاري تحميل التعليقات…</p> : null}
        {itemsState === "error" ? <div className="assignment-state" role="alert"><p>تعذر تحميل العناصر القابلة للإسناد.</p><button type="button" disabled={busy} onClick={() => void loadItems()}>إعادة تحميل التعليقات</button></div> : null}
        {itemsState === "ready" && items.length === 0 ? <EmptyState title={filtersActive ? "لا توجد تعليقات تطابق البحث والتصفية" : "لا توجد تعليقات غير مسندة حاليًا"} description={filtersActive ? "جرّب تغيير البحث أو مسح الفلاتر لعرض التعليقات المتاحة." : "ستظهر التعليقات الجديدة القابلة للإسناد هنا عند توفرها."} action={filtersActive ? <button type="button" onClick={clearFilters}>مسح الفلاتر</button> : undefined} /> : null}
        {itemsState === "ready" && items.length > 0 ? <>
          <table className="assignment-table" aria-label="العناصر القابلة للإسناد">
            <thead><tr><th scope="col"><label className="assignment-check"><input type="checkbox" aria-label="تحديد جميع عناصر الصفحة" disabled={busy} checked={pageSelected} ref={node => { if (node) node.indeterminate = somePageSelected && !pageSelected; }} onChange={() => { setAllFilteredSelected(false); if (pageSelected) setSelected(prev => new Set([...prev].filter(id => !items.some(item => item.comment_id === id)))); else selectPage(); }} /></label></th><th scope="col">التعليق</th><th scope="col">التصنيف</th><th scope="col">الأولوية</th><th scope="col">الإسناد</th><th scope="col">تاريخ الإضافة</th></tr></thead>
            <tbody>{items.map(item => <tr key={item.comment_id} className={selected.has(item.comment_id) ? "is-selected" : ""}>
              <td className="assignment-select-cell"><label className="assignment-check"><input type="checkbox" aria-label={`تحديد ${item.comment_id}`} disabled={busy} checked={selected.has(item.comment_id)} onChange={() => toggleOne(item.comment_id)} /><span className="assignment-mobile-select">{selected.has(item.comment_id) ? "محدد" : "تحديد التعليق"}</span></label></td>
              <td className="assignment-comment"><span className="assignment-comment-id"><bdi>#{item.comment_id}</bdi></span><p>{item.comment_preview || "نص التعليق غير متاح"}</p><span className="assignment-file">الملف: <bdi>{item.source_file}</bdi></span></td>
              <td className="assignment-category" data-label="التصنيف">{item.primary_intent ? <span className="assignment-tag">{INTENT_LABEL[item.primary_intent] || item.primary_intent}</span> : null}{item.topic_subtype ? <span className="assignment-tag">{TOPIC_LABEL[item.topic_subtype] || item.topic_subtype}</span> : null}{!item.primary_intent && !item.topic_subtype ? "غير متاح" : null}</td>
              <td data-label="الأولوية"><span className={item.review_priority === 1 ? "assignment-priority is-urgent" : "assignment-priority"}>{item.review_priority ? PRIORITY_LABEL[item.review_priority] || "غير متاح" : "غير متاح"}</span></td>
              <td data-label="الإسناد"><span className="assignment-status">{item.reviewer_user_id ? "مسند" : "غير مسند"}</span>{item.reviewer_user_id ? <span className="assignment-status-detail">{STATUS_LABEL[item.review_status] || "حالة غير معروفة"}</span> : null}</td>
              <td className="assignment-time" data-label="تاريخ الإضافة">{item.ingested_at ? <time dateTime={item.ingested_at}>{formatAbsoluteAr(item.ingested_at)}</time> : "غير متاح"}</td>
            </tr>)}</tbody>
          </table>
          <nav className="assignment-pagination" aria-label="صفحات التعليقات"><span>الصفحة {pageIndex + 1} من {pages} · {offset + 1}–{offset + items.length} من {total}</span><div><button type="button" disabled={busy || offset === 0} onClick={() => { setOffset(Math.max(0, offset - PAGE_SIZE)); listRef.current?.scrollIntoView?.({ block: "start" }); }}>السابق</button><button type="button" disabled={busy || pageIndex + 1 >= pages} onClick={() => { setOffset(offset + PAGE_SIZE); listRef.current?.scrollIntoView?.({ block: "start" }); }}>التالي</button></div></nav>
        </> : null}
      </section>
      <details className="assignment-reviewers assignment-surface"><summary>أحمال المراجعين وتقدم العمل</summary><ReviewerProgressPage key={summaryRevision} /></details>
    </div>
  );
}
