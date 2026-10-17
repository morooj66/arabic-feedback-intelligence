import { useEffect, useRef, useState } from "react";
import type { ReviewDetail } from "../api/types";
import { INTENT_LABEL, TOPIC_LABEL, PRIORITY_LABEL, STATUS_LABEL, MAX_RESPONSE_CHARS, NO_RESPONSE_REASONS, CONTEXT_CONFLICT_LABEL, CONTEXT_CONFLICT_FALLBACK, NO_DRAFT_REASON_LABEL } from "../labels";
import { composerActionFlags, isComposerEditable } from "../review/composerActions";
import { attestableText, blockingTexts, SAFETY_COPY } from "../review/replySafety";
import { formatAbsoluteAr, formatRelativeAr } from "../time/relative";
import "./ReviewModePage.css";
import { SystemState } from "../components/SystemState";

/** Reply-safety outcome returned by the server for the last send attempt. `codes` are backend finding codes and are never rendered. */
export type SafetyPanelState = { mode: "as_is" | "edit"; kind: "attest" | "denied" | "blocked"; codes: string[]; confirmed: boolean; stale: boolean };

type Props = {
  detail: ReviewDetail | null; draft: string; loading: boolean; busy: boolean; blocked: boolean;
  error: string; banner: string | null; toast: string; queueError: boolean;
  index: number; pageCount: number; hasPrev: boolean; hasNext: boolean;
  reasonOpen: boolean; reason: string;
  onDraft: (text: string) => void; onBack: () => void; onPrev: () => void; onNext: () => void;
  onReload: () => void; onSend: (mode: "as_is" | "edit") => void;
  safety?: SafetyPanelState | null; onSafetyConfirmed?: (confirmed: boolean) => void; onSafetyConfirm?: () => void; onSafetyDismiss?: () => void;
  onReasonOpen: (open: boolean) => void; onReason: (reason: string) => void; onNoResponse: () => void;
};

/** Focused presentation; mutations and version checks remain in ReviewerWorkspace. */
export function ReviewModePage(p: Props) {
  const d = p.detail;
  const flags = composerActionFlags(d?.allowed_actions, d?.suggested_response, p.draft);
  const editable = isComposerEditable(d?.review_status);
  const [editing, setEditing] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const modal = useRef<HTMLDivElement>(null);
  const noReply = useRef<HTMLButtonElement>(null);
  const disabled = p.busy || p.loading || p.blocked;
  useEffect(() => { setEditing(false); }, [d?.comment_id]);
  useEffect(() => {
    if (!p.reasonOpen) return;
    modal.current?.querySelector<HTMLSelectElement>("select")?.focus();
    return () => noReply.current?.focus();
  }, [p.reasonOpen]);
  const conflicts = d ? [...new Set((d.context_conflicts || []).map(code => CONTEXT_CONFLICT_LABEL[code] || CONTEXT_CONFLICT_FALLBACK))] : [];
  const noDraftReason = d && !d.suggested_response && d.generation_reason_code ? NO_DRAFT_REASON_LABEL[d.generation_reason_code] || "" : "";
  const context = d ? [["العنوان", d.original_title], ["معرف المدخل", d.entry_id], ["نوع المشاركة", d.original_type], ["المنصة", d.platform]].filter(([, value]) => value) : [];
  return <div className="focus-review" dir="rtl">
    <nav className="focus-nav" aria-label="التنقل بين عناصر القائمة">
      <button type="button" disabled={p.busy || p.loading} onClick={p.onBack}>العودة للقائمة</button>
      <div><button type="button" aria-label="التعليق السابق" disabled={p.busy || p.loading || !p.hasPrev} onClick={p.onPrev}>السابق</button><span aria-live="polite">{p.index >= 0 ? `${p.index + 1} من ${p.pageCount}` : "—"}<small>ضمن القائمة المصفّاة</small></span><button type="button" aria-label="التعليق التالي" disabled={p.busy || p.loading || !p.hasNext} onClick={p.onNext}>التالي</button></div>
    </nav>
    {p.toast ? <SystemState kind="success" as="p" className="focus-notice">{p.toast}</SystemState> : null}
    <section className="focus-body" aria-label="المراجعة الحالية" aria-busy={p.loading || p.busy}>
      {p.loading ? <SystemState kind="loading" className="focus-state"><h2>جاري تحميل التعليق…</h2><p>يُجهّز التعليق للمراجعة.</p></SystemState> : <>
        {p.error ? <SystemState kind="error" announce={!p.reasonOpen} className="focus-error"><p>{p.error}</p><button type="button" disabled={p.busy} onClick={p.onReload}>تحديث التعليق</button></SystemState> : null}
        {p.queueError ? <SystemState kind="error" as="p" className="focus-error">تعذر تحديث قائمة العمل. عُد إلى القائمة لإعادة تحميلها؛ لم يُفتح تعليق تالٍ.</SystemState> : null}
        {!d ? !p.error && !p.queueError ? <SystemState kind="empty" className="focus-state"><h2>لا توجد عناصر أخرى في القائمة</h2><p>عُد إلى القائمة للاطلاع على العمل المتبقي ضمن الفلاتر الحالية.</p></SystemState> : null : <>
          <header className="focus-heading"><div><p>مراجعة التعليق</p><h2>تعليق <bdi>#{d.comment_id}</bdi></h2><p>{d.comment_created_at ? <time dateTime={d.comment_created_at} title={formatAbsoluteAr(d.comment_created_at)}>{formatRelativeAr(d.comment_created_at)}</time> : "تاريخ التعليق غير متاح"}</p></div><span className="focus-tag">{STATUS_LABEL[d.review_status] || "حالة المراجعة غير متاحة"}</span></header>
          <section className="focus-comment" aria-labelledby="focus-comment-label"><h3 id="focus-comment-label">نص التعليق</h3><p dir="auto">{d.original_comment || "نص التعليق غير متاح"}</p></section>
          <dl className="focus-classification"><div><dt>النية</dt><dd>{d.primary_intent ? INTENT_LABEL[d.primary_intent] || "غير متاحة" : "غير متاحة"}</dd></div><div><dt>الموضوع</dt><dd>{d.topic_subtype ? TOPIC_LABEL[d.topic_subtype] || "غير متاح" : "غير متاح"}</dd></div>{d.review_priority ? <div><dt>الأولوية</dt><dd>{PRIORITY_LABEL[d.review_priority] || "غير متاحة"}</dd></div> : null}</dl>
          {conflicts.length || noDraftReason ? <section className="focus-flags" aria-label="ملاحظات التصنيف">
            {conflicts.length ? <><h3>ملاحظات على التصنيف</h3><p>تنبيهات آلية تساعد على التحقق من التصنيف أعلاه؛ تقديرك هو المعتمد.</p><ul>{conflicts.map(label => <li key={label}>{label}</li>)}</ul></> : null}
            {noDraftReason ? <p className="focus-flags-reason">{noDraftReason}</p> : null}
          </section> : null}
          {context.length ? <div className="focus-context" aria-label="السياق">{context.map(([label, value]) => <span key={label}><span>{label}: </span><span>{value}</span></span>)}</div> : null}
          {p.banner ? <p className="focus-notice">{p.banner}</p> : null}
          {d.review_status === "NO_RESPONSE_RECOMMENDED" ? <p className="focus-notice">اقترح النظام عدم الرد على هذا التعليق، وهذه توصية وليست قرارًا نهائيًا. أكّد «لا يحتاج رد» أو اكتب ردًا لتجاوز التوصية.</p> : null}
          {!editable && d.review_status === "NO_REPLY" ? <p className="focus-notice">أُغلق بدون إرسال رد{d.no_response_reason ? ` — ${d.no_response_reason}` : ""}</p> : null}
          <section className={`focus-response${editing || flags.dirty ? " is-editing" : ""}`} aria-label="صياغة الرد">
            <header><div><h3 id="focus-response-title">{editable ? editing || flags.dirty ? "تحرير الرد النهائي" : "الرد المقترح" : "الرد المحفوظ"}</h3><p id="focus-response-help">{!editable ? "للاطلاع فقط؛ لا يمكن تغيير القرار من هذا الموضع." : !flags.suggestion ? "لا توجد مسودة مولَّدة. اكتب الرد المؤسسي أو أغلق بدون رد." : editing || flags.dirty ? "لن تُحفظ التعديلات حتى تختار تعديل واعتماد." : "مسودة من الوكيل؛ راجعها قبل اتخاذ القرار."}</p></div>{editable ? <button type="button" disabled={disabled} onClick={() => { setEditing(true); textarea.current?.focus(); }}>تعديل الرد</button> : null}</header>
            <textarea ref={textarea} aria-label="الرد المقترح" aria-describedby="focus-response-help" dir="rtl" value={p.draft} readOnly={!editable || disabled} maxLength={MAX_RESPONSE_CHARS} onFocus={() => { if (editable) setEditing(true); }} onChange={e => p.onDraft(e.target.value.slice(0, MAX_RESPONSE_CHARS))} placeholder="اكتب الرد النهائي هنا…" />
            <footer><span dir="ltr">{p.draft.length} / {MAX_RESPONSE_CHARS}</span>{editable && flags.dirty ? <span>تعديلات غير محفوظة</span> : null}{editable && flags.suggestion ? <button type="button" disabled={disabled || !flags.dirty} onClick={() => { p.onDraft(d.suggested_response || ""); setEditing(false); }}>استعادة الرد المقترح</button> : null}</footer>
            {editable && flags.dirty && flags.suggestion ? <details><summary>الاطلاع على المسودة الأصلية</summary><p>{d.suggested_response}</p></details> : null}
          </section>
          {p.safety ? <SafetyPanel {...p} safety={p.safety} onEdit={() => { p.onSafetyDismiss?.(); setEditing(true); textarea.current?.focus(); }} /> : null}
          <div className="focus-actions"><p>{p.busy ? "جاري حفظ القرار…" : editable ? "اختر قرار المراجعة" : "تم تسجيل القرار؛ يمكنك متابعة القائمة."}</p><div><button type="button" className={!flags.dirty ? "primary" : ""} disabled={disabled || !editable || !flags.canSendAsIs} onClick={() => p.onSend("as_is")}>اعتماد كما هو</button><button type="button" className={flags.dirty ? "primary" : ""} disabled={disabled || !editable || !flags.canEditSend} onClick={() => p.onSend("edit")}>تعديل واعتماد</button><button ref={noReply} type="button" disabled={disabled || !editable || !flags.canNoReply} onClick={() => p.onReasonOpen(true)}>لا يحتاج رد</button></div></div>
        </>}
      </>}
    </section>
    {p.reasonOpen ? <div className="focus-modal-back"><div ref={modal} className="focus-modal" role="dialog" aria-modal="true" aria-labelledby="focus-reason-title" aria-describedby="focus-reason-help" onKeyDown={e => {
      if (e.key === "Escape" && !p.busy) { e.stopPropagation(); p.onReasonOpen(false); }
      if (e.key === "Tab") { const nodes = modal.current?.querySelectorAll<HTMLElement>("button:not(:disabled),select:not(:disabled)"); if (!nodes?.length) { e.preventDefault(); return; } const first = nodes[0], last = nodes[nodes.length - 1]; if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); } }
    }}><h2 id="focus-reason-title">إغلاق بدون رد</h2><p id="focus-reason-help">اختر السبب. لن يُرسل رد إلى الزائر.</p>{flags.dirty ? <p>لن تُحفظ تعديلات الرد عند الإغلاق بدون رد.</p> : null}<label>سبب عدم الرد<select value={p.reason} disabled={p.busy} onChange={e => p.onReason(e.target.value)} aria-label="سبب عدم الرد">{NO_RESPONSE_REASONS.map(r => <option key={r}>{r}</option>)}</select></label>{p.error ? <SystemState kind="error" as="p" className="focus-error">{p.error}</SystemState> : null}<div><button type="button" disabled={disabled || !flags.canNoReply} className="primary" onClick={p.onNoResponse}>تأكيد</button><button type="button" disabled={p.busy} onClick={() => p.onReasonOpen(false)}>إلغاء</button></div></div></div> : null}
  </div>;
}

function SafetyPanel(p: Props & { safety: SafetyPanelState; onEdit: () => void }) {
  const s = p.safety;
  if (s.kind === "attest") {
    return <SystemState kind="permission" className="focus-safety focus-safety--attest">
      <h3>{SAFETY_COPY.attestTitle}</h3>
      <p>{SAFETY_COPY.attestBody}</p>
      {s.stale ? <p className="focus-safety-stale">{SAFETY_COPY.stale}</p> : null}
      <ul>{s.codes.map(c => <li key={c}>{attestableText(c)}</li>)}</ul>
      <label><input type="checkbox" checked={s.confirmed} disabled={p.busy} onChange={e => p.onSafetyConfirmed?.(e.target.checked)} /> {SAFETY_COPY.attestCheckbox}</label>
      <div><button type="button" className="primary" disabled={p.busy || !s.confirmed || s.codes.length === 0} onClick={p.onSafetyConfirm}>{p.busy ? SAFETY_COPY.checking : SAFETY_COPY.attestSend}</button><button type="button" disabled={p.busy} onClick={p.onEdit}>{SAFETY_COPY.attestEdit}</button></div>
    </SystemState>;
  }
  const denied = s.kind === "denied";
  return <SystemState kind="error" className="focus-safety focus-safety--blocked">
    <h3>{denied ? SAFETY_COPY.deniedTitle : SAFETY_COPY.blockedTitle}</h3>
    <p>{denied ? SAFETY_COPY.deniedBody : SAFETY_COPY.blockedBody}</p>
    <ul>{denied ? s.codes.map(c => <li key={c}>{attestableText(c)}</li>) : blockingTexts(s.codes).map(text => <li key={text}>{text}</li>)}</ul>
    <div><button type="button" disabled={p.busy} onClick={p.onEdit}>{SAFETY_COPY.attestEdit}</button></div>
  </SystemState>;
}
