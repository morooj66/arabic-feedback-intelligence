import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { ApiError, DirectMessage, MessageConversationSummary } from "../api/types";
import { EmptyState } from "../components/EmptyState";
import "./MessagesPage.css";
import { STATE_COPY } from "../components/stateCopy";

type PreviewMessage = { id: string; sender: string; role: "ADMIN" | "REVIEWER"; at: string; body: string };
type PreviewThread = { id: string; reviewer: string; preview: string; commentId: string; at: string; messages: PreviewMessage[] };

const PREVIEW_THREADS: PreviewThread[] = [
  { id: "thread-1", reviewer: "مراجع اختبار أول", preview: "هذا التعليق يحتاج مراجعة من الإدارة.", commentId: "review-123", at: "١٠:٢٤ ص", messages: [
    { id: "m1", sender: "مراجع اختبار أول", role: "REVIEWER", at: "١٠:١٨ ص", body: "هذا التعليق يحتاج مراجعة من الإدارة، خصوصًا الجزء المتعلق باقتراح المثال." },
    { id: "m2", sender: "مسؤول التحقق", role: "ADMIN", at: "١٠:٢٤ ص", body: "شكرًا للتنبيه. سنراجع صياغة الرد وسياق المدخل قبل اتخاذ الإجراء." },
  ] },
  { id: "thread-2", reviewer: "مسؤول التحقق", preview: "يرجى تأكيد سياق الرد قبل اعتماده.", commentId: "review-108", at: "أمس", messages: [
    { id: "m3", sender: "مسؤول التحقق", role: "ADMIN", at: "أمس", body: "يرجى تأكيد سياق الرد قبل اعتماده." },
  ] },
];

/**
 * Work Messages: internal admin <-> reviewer text messaging, one conversation per reviewer. The authenticated application renders the live page
 * (`LiveMessages`); `preview` is test/visual-review only and is never passed by it.
 */
export function MessagesPage({ preview = false, role = "REVIEWER" }: { preview?: boolean; role?: "ADMIN" | "REVIEWER" }) {
  if (!preview) return <LiveMessages role={role} />;
  return <PreviewMessages />;
}

function PreviewMessages() {
  const preview = true;
  const [selectedId, setSelectedId] = useState<string | null>(preview ? PREVIEW_THREADS[0].id : null);
  const [search, setSearch] = useState("");
  const [draft, setDraft] = useState("");
  const threads = useMemo(() => PREVIEW_THREADS.filter(thread => !search || `${thread.commentId} ${thread.preview} ${thread.reviewer}`.includes(search)), [search]);
  const selected = threads.find(thread => thread.id === selectedId) || null;

  return (
    <div className="messages-ui messages-workspace" dir="rtl">
      <header className="messages-heading"><div><h2>الرسائل</h2><p>مناقشات العمل المرتبطة بمراجعات تعليقات الزوار.</p></div><span>معاينة تصميمية</span></header>
      <div className="messages-layout">
        <aside className="messages-list" aria-label="محادثات العمل">
          <div className="messages-list-heading"><div><h3>محادثات العمل</h3><span>{threads.length} محادثات</span></div><button type="button" aria-label="إجراءات المحادثات" disabled>⋯</button></div>
          <label className="messages-search"><span>بحث في المحادثات</span><input type="search" aria-label="بحث في المحادثات" placeholder="المعرّف أو نص الرسالة" value={search} onChange={e => setSearch(e.target.value)} /></label>
          <div className="messages-filter-row" role="group" aria-label="تصفية الرسائل"><button type="button" aria-pressed="true">الكل</button><button type="button" aria-pressed="false" disabled>غير المقروءة</button><button type="button" aria-pressed="false" disabled>تحتاج إجراء</button></div>
          {threads.length ? <div className="messages-thread-list">{threads.map(thread => <button type="button" key={thread.id} className={`messages-thread-item ${selected?.id === thread.id ? "is-selected" : ""}`} onClick={() => setSelectedId(thread.id)}><span className="messages-thread-ref"><bdi>#{thread.commentId}</bdi><time>{thread.at}</time></span><strong>{thread.reviewer}</strong><span>{thread.preview}</span></button>)}</div> : <EmptyState title="لا توجد محادثات مطابقة" description="جرّب تغيير البحث." />}
        </aside>
        <section className="messages-conversation" aria-label="المحادثة المحددة">
          {selected ? <><header className="messages-context"><div><span>سياق المراجعة</span><h3><bdi>تعليق #{selected.commentId}</bdi></h3><p>{selected.preview}</p></div><button type="button" disabled title="يتطلب ربطًا وظيفيًا بالمراجعة">عرض المراجعة</button></header><div className="messages-timeline" aria-label="سجل الرسائل">{selected.messages.map(message => <article key={message.id} className={`messages-message ${message.role === "ADMIN" ? "is-admin" : "is-reviewer"}`}><header><strong>{message.sender}</strong><span>{message.role === "ADMIN" ? "مسؤول النظام" : "مراجع"}</span><time>{message.at}</time></header><p>{message.body}</p></article>)}</div><form className="messages-composer" onSubmit={e => e.preventDefault()}><label htmlFor="message-draft">رسالة العمل</label><div><textarea id="message-draft" aria-describedby="messages-composer-note" value={draft} onChange={e => setDraft(e.target.value)} placeholder="اكتب ملاحظة مرتبطة بهذه المراجعة…" disabled /><button type="submit" disabled title="يتطلب تفعيل خدمة الرسائل">إرسال</button></div><p id="messages-composer-note">الإرسال غير متاح حتى يتم ربط خدمة الرسائل.</p></form></> : <EmptyState title="اختر محادثة عمل" description="اختر محادثة للاطلاع على سياق التعليق والرسائل المرتبطة به." />}
        </section>
      </div>
    </div>
  );
}

function UnavailableMessages() {
  return <div className="messages-ui messages-workspace" dir="rtl"><header className="messages-heading"><div><h2>الرسائل</h2><p>مناقشات العمل المرتبطة بمراجعات تعليقات الزوار.</p></div></header><div className="messages-layout messages-unavailable-layout"><aside className="messages-list"><div className="messages-list-heading"><div><h3>محادثات العمل</h3><span>غير متاح</span></div></div><EmptyState kind="unavailable" {...STATE_COPY.messagesEmpty} /></aside><section className="messages-conversation"><EmptyState kind="unavailable" {...STATE_COPY.messagesUnavailable} /></section></div></div>;
}

const POLL_MS = 10000;
const newClientId = () => (globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`).replace(/[^A-Za-z0-9_-]/g, "").slice(0, 64).padEnd(8, "0");
const formatTime = (iso: string | null | undefined) => {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString("ar", { dateStyle: "medium", timeStyle: "short" });
};

function LiveMessages({ role }: { role: "ADMIN" | "REVIEWER" }) {
  const isAdmin = role === "ADMIN";
  const [threads, setThreads] = useState<MessageConversationSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const [messages, setMessages] = useState<DirectMessage[]>([]);
  const [convName, setConvName] = useState("");
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [onlyUnread, setOnlyUnread] = useState(false);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const pendingId = useRef<{ text: string; id: string } | null>(null);   // reused for a retry of the SAME text, so a retried request cannot create a second message
  const selectedRef = useRef<string | undefined>(undefined);
  selectedRef.current = selectedId;

  const fail = useCallback((e: unknown) => {
    const err = e as ApiError;
    if (err?.status === 503) setUnavailable(true);
    else setError("تعذر تحميل الرسائل. حاول مجددًا.");
  }, []);

  const loadThreads = useCallback(async () => {
    if (!isAdmin) return;
    try { setThreads((await api.messagesConversations()).conversations); } catch (e) { fail(e); }
  }, [isAdmin, fail]);

  const loadConversation = useCallback(async (reviewerId: string | undefined, opening: boolean) => {
    try {
      const conv = await api.messagesConversation(reviewerId);
      if (selectedRef.current !== reviewerId) return;                    // the user moved on while this was loading
      setMessages(conv.messages); setConvName(conv.reviewer_name); setError(null);
      const fresh = conv.messages.filter(m => m.sender_role !== role && m.read_at === null).map(m => m.message_id);
      if (opening) setNewIds(new Set(fresh));
      else if (fresh.length) setNewIds(prev => new Set([...prev, ...fresh]));
      if (fresh.length) { await api.messagesMarkRead(reviewerId); void loadThreads(); }
    } catch (e) { fail(e); }
  }, [role, fail, loadThreads]);

  useEffect(() => {                                                       // first load + polling
    let alive = true;
    (async () => { await loadThreads(); if (alive) setLoading(false); })();
    const timer = setInterval(() => { void loadThreads(); if (!isAdmin || selectedRef.current) void loadConversation(selectedRef.current, false); }, POLL_MS);
    return () => { alive = false; clearInterval(timer); };
  }, [loadThreads, loadConversation, isAdmin]);

  useEffect(() => { if (!isAdmin) { setLoading(false); void loadConversation(undefined, true); } }, [isAdmin, loadConversation]);

  const open = (reviewerId: string) => { setSelectedId(reviewerId); selectedRef.current = reviewerId; setMessages([]); setNewIds(new Set()); setDraft(""); void loadConversation(reviewerId, true); };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text || sending || (isAdmin && !selectedId)) return;
    if (!pendingId.current || pendingId.current.text !== text) pendingId.current = { text, id: newClientId() };
    setSending(true); setError(null);
    try {
      await api.messagesSend(text, pendingId.current.id, isAdmin ? selectedId : undefined);
      pendingId.current = null; setDraft("");
      await loadConversation(selectedRef.current, false); void loadThreads();
    } catch (err) {
      const status = (err as ApiError)?.status;
      setError(status === 400 ? "تعذر إرسال الرسالة: تحقق من النص (لا تتجاوز 2000 حرف ولا تكون فارغة)." : "تعذر إرسال الرسالة. يمكنك إعادة المحاولة دون تكرارها.");
    } finally { setSending(false); }
  };

  const shown = useMemo(() => threads.filter(t => (!onlyUnread || t.unread > 0) && (!search || `${t.reviewer_name} ${t.last_message?.body ?? ""}`.includes(search))), [threads, search, onlyUnread]);
  const unreadTotal = threads.reduce((n, t) => n + t.unread, 0);

  if (unavailable) return <UnavailableMessages />;
  const conversationOpen = !isAdmin || !!selectedId;
  const title = isAdmin ? convName || threads.find(t => t.reviewer_user_id === selectedId)?.reviewer_name || "" : "الإدارة";

  return (
    <div className="messages-ui messages-workspace" dir="rtl">
      <header className="messages-heading"><div><h2>الرسائل</h2><p>مراسلات العمل بين المراجعين والإدارة.</p></div></header>
      <div className="messages-layout">
        <aside className="messages-list" aria-label="محادثات العمل">
          <div className="messages-list-heading"><div><h3>محادثات العمل</h3><span>{isAdmin ? (unreadTotal ? `${unreadTotal} غير مقروءة` : `${threads.length} محادثات`) : ""}</span></div></div>
          {isAdmin ? <>
            <label className="messages-search"><span>بحث في المحادثات</span><input type="search" aria-label="بحث في المحادثات" placeholder="اسم المراجع أو نص الرسالة" value={search} onChange={e => setSearch(e.target.value)} /></label>
            <div className="messages-filter-row" role="group" aria-label="تصفية الرسائل"><button type="button" aria-pressed={!onlyUnread} onClick={() => setOnlyUnread(false)}>الكل</button><button type="button" aria-pressed={onlyUnread} onClick={() => setOnlyUnread(true)}>غير المقروءة</button></div>
            {loading ? <EmptyState title="جارٍ التحميل" description="" /> : shown.length ? <div className="messages-thread-list">{shown.map(t => <button type="button" key={t.reviewer_user_id} data-unread={t.unread} className={`messages-thread-item ${selectedId === t.reviewer_user_id ? "is-selected" : ""}`} onClick={() => open(t.reviewer_user_id)}><span className="messages-thread-ref"><bdi>{t.unread ? `${t.unread} غير مقروءة` : ""}</bdi><time>{formatTime(t.last_message?.created_at)}</time></span><strong>{t.reviewer_name}</strong><span>{t.last_message ? t.last_message.body : "لا توجد رسائل بعد"}</span></button>)}</div> : <EmptyState title="لا توجد محادثات مطابقة" description="جرّب تغيير البحث أو التصفية." />}
          </> : <div className="messages-thread-list"><button type="button" className="messages-thread-item is-selected" aria-current="true"><strong>الإدارة</strong><span>محادثتك مع إدارة النظام</span></button></div>}
        </aside>
        <section className="messages-conversation" aria-label="المحادثة المحددة">
          {conversationOpen ? <>
            <header className="messages-context"><div><span>{isAdmin ? "محادثة مع المراجع" : "محادثة مع"}</span><h3>{title}</h3></div></header>
            <div className="messages-timeline" aria-label="سجل الرسائل" aria-live="polite">
              {messages.length ? messages.map(m => <article key={m.message_id} className={`messages-message ${m.sender_role === "ADMIN" ? "is-admin" : "is-reviewer"}`}><header><strong>{m.sender_name}</strong><span>{m.sender_role === "ADMIN" ? "مسؤول النظام" : "مراجع"}</span><time dateTime={m.created_at}>{formatTime(m.created_at)}</time>{newIds.has(m.message_id) ? <span>جديدة</span> : null}</header><p>{m.body}</p></article>) : <EmptyState title="لا توجد رسائل بعد" description="ابدأ المحادثة بإرسال رسالة." />}
            </div>
            <form className="messages-composer" onSubmit={submit}>
              <label htmlFor="message-draft">رسالة العمل</label>
              <div><textarea id="message-draft" value={draft} maxLength={2000} onChange={e => setDraft(e.target.value)} placeholder="اكتب رسالتك…" /><button type="submit" disabled={sending || !draft.trim()}>{sending ? "جارٍ الإرسال…" : "إرسال"}</button></div>
              {error ? <p role="alert">{error}</p> : null}
            </form>
          </> : <EmptyState title="اختر محادثة عمل" description="اختر مراجعًا للاطلاع على الرسائل وإرسال رسالة." />}
        </section>
      </div>
    </div>
  );
}
