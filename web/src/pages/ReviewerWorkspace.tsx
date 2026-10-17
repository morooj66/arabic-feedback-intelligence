import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { ApiError, FactAttestation } from "../api/types";
import type { QueueItem, ReviewDetail, ReviewerSession, WorkflowCounts } from "../api/types";
import { NO_RESPONSE_REASONS, ROLE_LABEL, outboundBanner } from "../labels";
import { Avatar } from "../components/Avatar";
import { WorkspaceIcon } from "../components/WorkspaceIcon";
import "./AuthenticatedShell.css";
import { ReviewerAssignedPage } from "./ReviewerAssignedPage";
import { ReviewModePage } from "./ReviewModePage";
import { SystemState } from "../components/SystemState";
import { STATE_COPY } from "../components/stateCopy";
import { AdminDashboard } from "./AdminDashboard";
import { AgentInspectorPage } from "./AgentInspectorPage";
import { PlaygroundPage } from "./PlaygroundPage";
import { AnalyticsPage } from "./AnalyticsPage";
import { AssignmentPage } from "./AssignmentPage";
import { CompletedReviewsPage } from "./CompletedReviewsPage";
import { ImportPage } from "./ImportPage";
import { MessagesPage } from "./MessagesPage";
import { SettingsPage, readDefaultSort, writeDefaultSort } from "./SettingsPage";
import { composerActionFlags, isComposerEditable } from "../review/composerActions";
import { sameCodes } from "../review/replySafety";
import type { SafetyPanelState } from "./ReviewModePage";
import { EMPTY_EXTRA, extraFilterCount, extraFilterParams, type ExtraFilterValue } from "../components/ExtraFilters";

const PAGE_SIZE = 8;

// ADMIN and REVIEWER intentionally have disjoint nav sets (never one
// list filtered by role) -- each role has exactly the destinations it
// needs and nothing else. "import"/"settings"/"inspector" remain valid
// internal destinations (reached contextually -- Admin Home's import
// CTA, the account menu, Settings' technical section — see below)
// without a primary sidebar entry of their own. Reviewer progress
// ("متابعة المراجعين") is no longer a standalone destination either —
// it is embedded directly inside AssignmentPage, where an admin already
// needs that same reviewer list to pick who to assign to.
type NavKey = "dashboard" | "import" | "assignment" | "completed" | "review" | "analytics" | "messages" | "settings" | "inspector" | "playground";
type QueueFilter = "pending" | "all" | "approved" | "no_reply" | "followup";

const ADMIN_NAV: { key: NavKey; label: string }[] = [
  { key: "dashboard", label: "الرئيسية" },
  { key: "assignment", label: "الإسناد" },
  { key: "completed", label: "المراجعات المكتملة" },
  { key: "analytics", label: "التحليلات" },
  { key: "messages", label: "الرسائل" },
];

const REVIEWER_NAV: { key: NavKey; label: string }[] = [
  { key: "review", label: "المسند لي" },
  { key: "messages", label: "الرسائل" },
];

// Every reachable destination's title, including the contextual ones
// with no primary sidebar entry -- one lookup, never duplicated per role.
const NAV_TITLES: Record<NavKey, string> = {
  dashboard: "الرئيسية",
  import: "استيراد التعليقات",
  assignment: "الإسناد",
  completed: "المراجعات المكتملة",
  review: "المسند لي",
  analytics: "التحليلات",
  messages: "الرسائل",
  settings: "الإعدادات",
  inspector: "فاحص الوكيل",
  playground: "ساحة التقييم",
};

const QUEUE_FILTERS: { key: QueueFilter; label: string }[] = [
  { key: "pending", label: "بانتظار المراجعة" },
  { key: "all", label: "الكل" },
  { key: "approved", label: "تمت الموافقة" },
  { key: "no_reply", label: "لا يحتاج إلى رد" },
  { key: "followup", label: "مراجعة لاحقة" },
];

export function ReviewerWorkspace() {
  const navigate = useNavigate();
  const [session, setSession] = useState<ReviewerSession | null>(null);
  const [nav, setNav] = useState<NavKey>("dashboard");
  const [queueFilter, setQueueFilter] = useState<QueueFilter>("pending");
  const [sort, setSort] = useState(() => readDefaultSort());
  const [defaultSort, setDefaultSort] = useState(() => readDefaultSort());
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [total, setTotal] = useState(0);
  const [counts, setCounts] = useState<WorkflowCounts | null>(null);
  const [countsState, setCountsState] = useState<"loading" | "ready" | "error">("loading");
  const [queueState, setQueueState] = useState<"loading" | "ready" | "error">("loading");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ReviewDetail | null>(null);
  const [draft, setDraft] = useState("");
  const [detailError, setDetailError] = useState("");
  const [safety, setSafety] = useState<SafetyPanelState | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailBlocked, setDetailBlocked] = useState(false);
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);
  const [reasonOpen, setReasonOpen] = useState(false);
  const [reason, setReason] = useState(NO_RESPONSE_REASONS[0]);
  const [menuOpen, setMenuOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  // "list": filter/select a working set (§12). "reviewing": the
  // focused single-item view (§13). A REVIEWER always starts on
  // "list" -- never auto-opened into an item on login.
  const [reviewerViewMode, setReviewerViewMode] = useState<"list" | "reviewing">("list");
  const [selectedIntents, setSelectedIntents] = useState<string[]>([]);
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [extra, setExtra] = useState<ExtraFilterValue>(EMPTY_EXTRA);
  const extraKey = JSON.stringify(extra);

  const flags = composerActionFlags(detail?.allowed_actions, detail?.suggested_response, draft);
  const dirty = flags.dirty;
  const unsaved = isComposerEditable(detail?.review_status) && dirty;
  const title = NAV_TITLES[nav] || NAV_TITLES.dashboard;

  const [activeSourceFile, setActiveSourceFile] = useState<string | null>(null);
  // Guards against openItem(id) firing twice concurrently for the same id
  // (e.g. React StrictMode's dev-only double-invoked effects, or the
  // auto-open effect re-running while a fetch+claim is still in flight):
  // without this, both calls fetch and claim the same item, one claim
  // wins and the other's now-stale expected_version gets a legitimate
  // 409 that overwrites a perfectly successful screen with a spurious
  // error banner.
  const openingIdRef = useRef<string | null>(null);
  // Set when "next"/"previous" crosses a page boundary of the filtered list: once the new page has loaded, open its first / last comment.
  const edgeRef = useRef<"first" | "last" | null>(null);

  // Screen 08 only: leave/list/browser-exit protection for a real unsaved edit.
  function canLeaveReview() {
    if (reviewerViewMode !== "reviewing" || nav !== "review") return true;
    if (busy || detailLoading) return false;
    return !unsaved || window.confirm("هناك تعديلات غير محفوظة. هل تريد المتابعة؟");
  }
  useEffect(() => {
    if (reviewerViewMode !== "reviewing" || nav !== "review" || !unsaved) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [reviewerViewMode, nav, unsaved]);

  useEffect(() => {
    api
      .session()
      .then((s) => {
        setSession(s);
        if (s.role === "ADMIN") {
          // /api/excel/sessions/active is ADMIN-only (Excel import is
          // admin-owned end to end) — a REVIEWER has no "active session"
          // concept at all, so never call it for them.
          void refreshActiveSession();
        } else {
          // A REVIEWER has one destination — their own work — so they
          // land directly in it instead of a separate landing page.
          setNav("review");
        }
      })
      .catch((e) => {
        // A missing/expired/invalid session (401) always lands here as
        // "not logged in" — deliberately the same as a fresh, never-
        // authenticated visit (see auth.py's resolve_session docstring:
        // never reveal which reason applied). A backend/database outage
        // (503 service_unavailable) is a different situation the user
        // should be told about, rather than silently looking like a
        // plain logged-out state — reuses LoginPage's existing ?error=
        // banner mechanism instead of adding a new one.
        const err = e as ApiError;
        navigate(err.code === "service_unavailable" ? "/?error=service_unavailable" : "/");
      });
  }, [navigate]);

  async function refreshActiveSession() {
    try {
      const res = await api.activeExcelSession();
      setActiveSourceFile(res.active?.source_file ?? null);
    } catch {
      setActiveSourceFile(null);
    }
  }

  // Reviewer-scoped (server-enforced, see http_api.py::_require_item_access
  // and workflow_counts(reviewer_user_id=...)) — /api/counts already
  // returns only this reviewer's own assigned items, independent of any
  // source_file, so it works whether or not an active import exists.
  async function refreshReviewerCounts() {
    setCountsState("loading");
    try {
      setCounts(await api.counts());
      setCountsState("ready");
    } catch {
      setCounts(null);
      setCountsState("error");
    }
  }

  async function refreshQueue(nextOffset = offset) {
    setQueueState("loading");
    try {
      const page = await api.queue({
        filter: queueFilter,
        sort,
        search,
        offset: nextOffset,
        limit: PAGE_SIZE,
        sourceFile: activeSourceFile ?? undefined,
        primaryIntents: selectedIntents,
        topicSubtypes: selectedTopics,
        ...extraFilterParams(extra),
      });
      setQueue(page.items);
      setTotal(page.total);
      setQueueState("ready");
      return page.items;
    } catch {
      setQueueState("error");
      return [];
    }
  }

  useEffect(() => {
    setOffset(0);
  }, [nav, queueFilter, sort, search, selectedIntents, selectedTopics, extraKey]);

  useEffect(() => {
    if (nav !== "review") return;
    // A REVIEWER's own summary strip (assigned/pending/completed) lives
    // on this same screen now, above their queue — refresh it alongside
    // the queue instead of on a separate dashboard visit.
    if (session && session.role !== "ADMIN") void refreshReviewerCounts();
    void refreshQueue(offset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nav, queueFilter, sort, search, offset, session, selectedIntents, selectedTopics, extraKey]);

  function go(next: NavKey) {
    if (!canLeaveReview()) return;
    if (next === "review" && nav !== "review") {
      setSort(readDefaultSort());
    }
    setNav(next);
    setMenuOpen(false);
    setProfileOpen(false);
  }

  // ADMIN's next step after import/processing is assignment, not a
  // personal review queue (ADMIN has none — see ADMIN_NAV above); this
  // used to open the reviewer queue directly, which no longer exists
  // as an admin destination.
  function goAssignment(sourceFile?: string) {
    if (sourceFile) setActiveSourceFile(sourceFile);
    setNav("assignment");
  }

  async function openItem(id: string, force = false) {
    if (!force && unsaved && selectedId && selectedId !== id) {
      if (!window.confirm("هناك تعديلات غير محفوظة. هل تريد المتابعة؟")) return;
    }
    if (openingIdRef.current === id) return; // a fetch+claim for this id is already in flight
    openingIdRef.current = id;
    setSafety(null);
    setDetailLoading(true);
    setDetailBlocked(false);
    try {
      const switching = detail?.comment_id !== id;
      if (switching) {
        setDraft("");
        setDetail(null);
      }
      setSelectedId(id);
      setDetailError("");
      try {
        let item = await api.item(id);
        if (["DRAFT_READY", "REVIEW_ONLY"].includes(item.review_status)) {
          try {
            item = await api.claim(id, item.row_version);
          } catch (e) {
            const err = e as ApiError;
            if (err.code === "claimed") {
              setDetailBlocked(true);
              setDetailError("هذا التعليق قيد المراجعة لدى مراجع آخر.");
              setDetail(item);
              setDraft(item.suggested_response || "");
              return;
            }
            throw e;
          }
        }
        const suggestion = item.suggested_response || "";
        const keepTyped = !switching && dirty && !force && isComposerEditable(item.review_status);
        setDetail(item);
        if (!keepTyped) {
          setDraft(isComposerEditable(item.review_status) ? suggestion : item.final_response || suggestion);
        }
      } catch (e) {
        setDetailBlocked(true);
        const err = e as ApiError;
        setDetailError(err.code === "not_found" ? "تعذر العثور على التعليق." : "تعذر تحميل التعليق.");
      }
    } finally {
      if (openingIdRef.current === id) {
        openingIdRef.current = null;
        setDetailLoading(false);
      }
    }
  }

  useEffect(() => {
    // Only while actively reviewing (never in the list-first screen —
    // a REVIEWER is never auto-opened into an item on login/filter
    // change, see beginReview below): keeps the currently-open item in
    // sync with the queue as it refetches (e.g. after an action removes
    // the just-decided item from a "pending" filter).
    if (nav !== "review" || reviewerViewMode !== "reviewing") return;
    if (edgeRef.current && queue.length) {
      const target = edgeRef.current === "last" ? queue[queue.length - 1] : queue[0];
      edgeRef.current = null;
      void openItem(target.comment_id, true);
      return;
    }
    if (queue.length && !selectedId) {
      void openItem(queue[0].comment_id, true);
    } else if (queue.length && selectedId && detail?.comment_id !== selectedId) {
      void openItem(selectedId, true);
    } else if (!queue.length) {
      setDetail(null);
      setSelectedId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue, nav, reviewerViewMode]);

  function beginReview() {
    if (queue.length) void openItem(queue[0].comment_id, true);
    setReviewerViewMode("reviewing");
  }

  function backToList() {
    if (!canLeaveReview()) return;
    // Filters (queueFilter/sort/search/selectedIntents/selectedTopics)
    // are untouched -- returning to the list preserves them exactly.
    setReviewerViewMode("list");
  }

  async function afterAction(next?: ReviewDetail) {
    // The mutation is confirmed. Never leave the old editable version active
    // if refreshing the queue fails afterward.
    if (next) {
      setDetail(next);
      setSelectedId(next.comment_id);
      setDraft(next.final_response || next.suggested_response || "");
    }
    if (session?.role !== "ADMIN") void refreshReviewerCounts();
    let items = await refreshQueue(offset);
    if (!items.length && offset > 0) {            // the decided comment was the last one on this page: continue from the start of the remaining list
      setOffset(0);
      items = await refreshQueue(0);
    }
    const remaining = items.filter((i) => i.comment_id !== next?.comment_id);
    if (remaining[0]) {
      await openItem(remaining[0].comment_id, true);
    } else if (next) {
      setDetail(next);
      setSelectedId(next.comment_id);
      setDraft(next.final_response || next.suggested_response || "");
    } else {
      setDetail(null);
    }
  }

  function showToast(text: string) {
    setToast(text);
    window.setTimeout(() => setToast(""), 3500);
  }

  function codesOf(err: ApiError, key: string): string[] {
    const v = err.payload?.[key];
    return Array.isArray(v) ? v.filter((c): c is string => typeof c === "string") : [];
  }

  /** Server-side safety outcomes. The reviewer's draft is never touched here; only the explanation panel changes. */
  function showSafety(mode: "as_is" | "edit", err: ApiError, stale: boolean): boolean {
    if (err.code === "attestation_required") {
      setSafety({ mode, kind: "attest", codes: codesOf(err, "attestable_findings"), confirmed: false, stale });
      return true;
    }
    if (err.code === "attestation_not_permitted") {
      setSafety({ mode, kind: "denied", codes: codesOf(err, "attestable_findings"), confirmed: false, stale: false });
      return true;
    }
    if (err.code === "draft_validation_failed") {
      setSafety({ mode, kind: "blocked", codes: codesOf(err, "blocking_findings"), confirmed: false, stale: false });
      return true;
    }
    return false;
  }

  /** Revalidate with the server, then send with EXACTLY the findings the reviewer was shown. A changed result re-opens the confirmation. */
  async function confirmAttestation() {
    if (!detail || !safety || safety.kind !== "attest" || !safety.confirmed || busy || detailLoading || detailBlocked) return;
    const mode = safety.mode;
    const text = mode === "edit" ? draft : detail.suggested_response || "";
    setBusy(true);
    try {
      const check = await api.validateReply(detail.comment_id, text);
      if (check.blocking_findings.length) {
        setSafety({ mode, kind: "blocked", codes: check.blocking_findings, confirmed: false, stale: false });
        return;
      }
      if (check.attestable_findings.length && !check.attestation_permitted) {
        setSafety({ mode, kind: "denied", codes: check.attestable_findings, confirmed: false, stale: false });
        return;
      }
      if (check.attestable_findings.length && !sameCodes(check.attestable_findings, safety.codes)) {
        setSafety({ mode, kind: "attest", codes: check.attestable_findings, confirmed: false, stale: true });
        return;
      }
    } catch {
      setDetailError(STATE_COPY.mutationFailed);
      return;
    } finally {
      setBusy(false);
    }
    await send(mode, { confirmed: true, finding_codes: safety.codes });
  }

  async function send(mode: "as_is" | "edit", attestation?: FactAttestation) {
    if (!detail || busy || detailLoading || detailBlocked) return;
    if (mode === "edit" && !draft.trim()) return;
    if (mode === "as_is" && !(detail.suggested_response || "").trim()) return;
    setBusy(true);
    setDetailError("");
    if (!attestation) setSafety(null);
    try {
      const result =
        mode === "edit"
          ? attestation ? await api.editAndSend(detail.comment_id, detail.row_version, draft, attestation) : await api.editAndSend(detail.comment_id, detail.row_version, draft)
          : attestation ? await api.sendAsIs(detail.comment_id, detail.row_version, attestation) : await api.sendAsIs(detail.comment_id, detail.row_version);
      const banner = outboundBanner(result.outbound_status, result.outbound_error_code, result.review_status === "SENT");
      showToast(banner || "تم اعتماد الرد");
      setSafety(null);
      await afterAction(result);
    } catch (e) {
      const err = e as ApiError;
      if (showSafety(mode, err, Boolean(attestation))) return;
      if (err.code === "version_conflict" || err.code === "claimed") setDetailBlocked(true);
      if (err.code === "version_conflict") setDetailError(STATE_COPY.versionConflict);
      else if (err.code === "claimed") setDetailError("هذا التعليق قيد المراجعة لدى مراجع آخر.");
      else setDetailError(STATE_COPY.mutationFailed);
    } finally {
      setBusy(false);
    }
  }

  async function closeNoResponse() {
    if (!detail || busy || detailLoading || detailBlocked) return;
    setBusy(true);
    setDetailError("");
    try {
      const result = await api.noResponse(detail.comment_id, detail.row_version, reason);
      setReasonOpen(false);
      showToast("أُغلق بدون إرسال رد");
      await afterAction(result);
    } catch (e) {
      const err = e as ApiError;
      if (err.code === "version_conflict" || err.code === "claimed") setDetailBlocked(true);
      setDetailError(err.code === "version_conflict" ? STATE_COPY.versionConflict : "تعذر إغلاق التعليق.");
    } finally {
      setBusy(false);
    }
  }

  async function logout() {
    await api.logout();
    navigate("/");
  }

  const pages = useMemo(() => Math.max(1, Math.ceil(total / PAGE_SIZE)), [total]);
  const pageIndex = Math.floor(offset / PAGE_SIZE);
  const banner = detail
    ? outboundBanner(detail.outbound_status, detail.outbound_error_code, detail.review_status === "SENT")
    : null;

  if (!session) {
    return <SystemState kind="loading" className="empty">{STATE_COPY.loading}</SystemState>;
  }

  const isReviewer = session.role !== "ADMIN";
  const reviewerZeroAssigned = isReviewer && countsState === "ready" && (counts?.all ?? 0) === 0;
  const reviewerAllCaughtUp =
    isReviewer && countsState === "ready" && (counts?.all ?? 0) > 0 && (counts?.pending ?? 0) === 0;
  const queueIndex = selectedId ? queue.findIndex((q) => q.comment_id === selectedId) : -1;
  // The review session spans the WHOLE filtered list (not just the 8 comments on the current page): moving past either end of the page loads the neighbouring page.
  const hasPrevInQueue = queueIndex > 0 || (queueIndex === 0 && offset > 0);
  const hasNextInQueue = queueIndex >= 0 && (queueIndex < queue.length - 1 || offset + queue.length < total);
  const filtersActive = queueFilter !== "pending" || sort !== "priority" || search !== "" || selectedIntents.length > 0 || selectedTopics.length > 0 || extraFilterCount(extra) > 0;

  function crossPage(edge: "first" | "last", nextOffset: number) {
    if (unsaved && !window.confirm("هناك تعديلات غير محفوظة. هل تريد المتابعة؟")) return;
    edgeRef.current = edge;
    setOffset(nextOffset);
  }

  function goPrevItem() {
    if (busy || detailLoading || !hasPrevInQueue) return;
    if (queueIndex > 0) void openItem(queue[queueIndex - 1].comment_id);
    else crossPage("last", Math.max(0, offset - PAGE_SIZE));
  }

  function goNextItem() {
    if (busy || detailLoading || !hasNextInQueue) return;
    if (queueIndex < queue.length - 1) void openItem(queue[queueIndex + 1].comment_id);
    else crossPage("first", offset + PAGE_SIZE);
  }

  function resetQueueFilters() {
    setQueueFilter("pending");
    setSort("priority");
    setSearch("");
    setSelectedIntents([]);
    setSelectedTopics([]);
    setExtra(EMPTY_EXTRA);
  }

  return (
    <Shell
      session={session}
      nav={nav}
      go={go}
      title={title}
      menuOpen={menuOpen}
      setMenuOpen={setMenuOpen}
      profileOpen={profileOpen}
      setProfileOpen={setProfileOpen}
      onLogout={() => { if (canLeaveReview()) void logout(); }}
    >
      {nav === "dashboard" ? (
        // ADMIN-only destination (see ADMIN_NAV above) — a REVIEWER
        // never reaches this branch, so it renders the operational
        // home unconditionally.
        <div className="workspace workspace--wide">
          <AdminDashboard session={session} onImport={() => go("import")} onGoAssignment={goAssignment} onGoCompleted={() => go("completed")} />
        </div>
      ) : nav === "import" ? (
        <div className="workspace workspace--wide">
          {session.role === "ADMIN" ? (
            <ImportPage
              onOpenReview={goAssignment}
              onProcessed={() => {
                void refreshActiveSession();
              }}
            />
          ) : (
            <SystemState kind="permission" className="empty">{STATE_COPY.permissionDenied}</SystemState>
          )}
        </div>
      ) : nav === "assignment" ? (
        <div className="workspace workspace--wide">
          {session.role === "ADMIN" ? (
            <AssignmentPage initialSourceFile={activeSourceFile} />
          ) : (
            <SystemState kind="permission" className="empty">{STATE_COPY.permissionDenied}</SystemState>
          )}
        </div>
      ) : nav === "completed" ? (
        <div className="workspace workspace--wide">
          {session.role === "ADMIN" ? <CompletedReviewsPage /> : <SystemState kind="permission" className="empty">{STATE_COPY.permissionDenied}</SystemState>}
        </div>
      ) : nav === "analytics" ? (
        <div className="workspace workspace--wide">
          {session.role === "ADMIN" ? <AnalyticsPage /> : <SystemState kind="permission" className="empty">{STATE_COPY.permissionDenied}</SystemState>}
        </div>
      ) : nav === "messages" ? (
        <div className="workspace workspace--wide">
          <MessagesPage role={session.role === "ADMIN" ? "ADMIN" : "REVIEWER"} />
        </div>
      ) : nav === "inspector" ? (
        <div className="workspace workspace--wide">
          {session.role === "ADMIN" ? <AgentInspectorPage /> : <SystemState kind="permission" className="empty">{STATE_COPY.permissionDenied}</SystemState>}
        </div>
      ) : nav === "playground" ? (
        <div className="workspace workspace--wide">
          {session.role === "ADMIN" ? <PlaygroundPage /> : <SystemState kind="permission" className="empty">{STATE_COPY.permissionDenied}</SystemState>}
        </div>
      ) : nav === "settings" ? (
        <div className="workspace workspace--wide">
          <SettingsPage
            sort={defaultSort}
            onSort={(value) => {
              setDefaultSort(value);
              writeDefaultSort(value);
            }}
            isAdmin={session.role === "ADMIN"}
            onOpenInspector={session.role === "ADMIN" ? () => go("inspector") : undefined}
            onOpenPlayground={session.role === "ADMIN" ? () => go("playground") : undefined}
          />
        </div>
      ) : (
        <>
          {reviewerViewMode === "list" ? (
            <div className="workspace workspace--wide">
              <ReviewerAssignedPage
                displayName={session.display_name} counts={counts} countsState={countsState}
                queue={queue} queueState={queueState} total={total} queueFilter={queueFilter}
                filters={QUEUE_FILTERS} search={search} sort={sort}
                selectedIntents={selectedIntents} selectedTopics={selectedTopics} extra={extra} onExtra={setExtra}
                filtersActive={filtersActive} zeroAssigned={reviewerZeroAssigned} allCaughtUp={reviewerAllCaughtUp}
                pageIndex={pageIndex} pages={pages}
                onFilter={setQueueFilter} onSearch={setSearch} onSort={setSort}
                onClassification={(intents, topics) => { setSelectedIntents(intents); setSelectedTopics(topics); }}
                onReset={resetQueueFilters} onPage={page => setOffset(page * PAGE_SIZE)}
                onRetryQueue={() => void refreshQueue(offset)} onRetryCounts={() => void refreshReviewerCounts()}
                onStart={beginReview}
              />
            </div>
          ) : (
            <div className="workspace workspace--wide">
              <ReviewModePage
                detail={detail} draft={draft} loading={detailLoading} busy={busy} blocked={detailBlocked}
                error={detailError} banner={banner} toast={toast} queueError={queueState === "error"}
                index={queueIndex >= 0 ? offset + queueIndex : -1} pageCount={total} hasPrev={hasPrevInQueue} hasNext={hasNextInQueue}
                reasonOpen={reasonOpen} reason={reason} onDraft={text => { setSafety(null); setDraft(text); }} onBack={backToList}
                onPrev={goPrevItem} onNext={goNextItem} onReload={() => { if (selectedId && canLeaveReview()) void openItem(selectedId); }}
                safety={safety} onSafetyConfirmed={confirmed => setSafety(s => (s ? { ...s, confirmed } : s))} onSafetyConfirm={() => void confirmAttestation()}
                onSafetyDismiss={() => setSafety(null)}
                onSend={mode => void send(mode)} onReasonOpen={setReasonOpen} onReason={setReason}
                onNoResponse={() => void closeNoResponse()}
              />
            </div>
          )}
        </>
      )}
    </Shell>
  );
}

function Shell(props: {
  session: ReviewerSession;
  nav: NavKey;
  go: (k: NavKey) => void;
  title: string;
  menuOpen: boolean;
  setMenuOpen: (v: boolean) => void;
  onLogout: () => void;
  profileOpen: boolean;
  setProfileOpen: (v: boolean) => void;
  children: ReactNode;
}) {
  const { session, nav, go, title, menuOpen, setMenuOpen, onLogout, profileOpen, setProfileOpen, children } = props;
  const navItems = session.role === "ADMIN" ? ADMIN_NAV : REVIEWER_NAV;
  const [navigationOpen, setNavigationOpen] = useState(false);
  const navigationToggle = useRef<HTMLButtonElement>(null);
  const accountToggle = useRef<HTMLButtonElement>(null);
  const profileDialog = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Profile dialog: move focus inside when it opens, give it back to the account control when it closes.
    if (!profileOpen) return;
    profileDialog.current?.querySelector<HTMLElement>("button:not(:disabled)")?.focus();
    return () => accountToggle.current?.focus();
  }, [profileOpen]);
  function onProfileKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.stopPropagation();
      setProfileOpen(false);
      return;
    }
    if (event.key !== "Tab") return;
    const nodes = profileDialog.current?.querySelectorAll<HTMLElement>("button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex='-1'])");
    if (!nodes?.length) {
      event.preventDefault();
      return;
    }
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  useEffect(() => {
    function onEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      if (navigationOpen) { setNavigationOpen(false); navigationToggle.current?.focus(); }
      if (menuOpen) { setMenuOpen(false); accountToggle.current?.focus(); }
    }
    document.addEventListener("keydown", onEscape);
    return () => document.removeEventListener("keydown", onEscape);
  }, [navigationOpen, menuOpen, setMenuOpen]);
  return (
    <div className="shell shell--wide authenticated-shell" dir="rtl">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <img
            className="sidebar-logo"
            src="/login-logo.png"
            alt="شعار المشروع"
          />
          <button ref={navigationToggle} type="button" className="auth-navigation-toggle" aria-label={navigationOpen ? "إغلاق التنقل" : "فتح التنقل"} aria-expanded={navigationOpen} aria-controls="authenticated-navigation" onClick={() => setNavigationOpen(open => !open)}><WorkspaceIcon name={navigationOpen ? "close" : "menu"} /></button>
        </div>
        <nav id="authenticated-navigation" className={navigationOpen ? "auth-navigation is-open" : "auth-navigation"} aria-label="التنقل الرئيسي">
          {navItems.map((item) => (
            <button key={item.key} type="button" className={`nav-btn ${nav === item.key ? "active" : ""}`} aria-current={nav === item.key ? "page" : undefined} onClick={() => { go(item.key); setNavigationOpen(false); }}>
              <WorkspaceIcon name={item.key} /><span>{item.label}</span>
            </button>
          ))}
        </nav>
        <p className="auth-sidebar-note">نظام مراجعة تعليقات الزوار</p>
      </aside>
      <header className="topbar">
        <h1>{title}</h1>
        <div className="account">
          <div className="account-menu">
            <button ref={accountToggle} type="button" className="account-trigger" aria-label="الحساب" aria-expanded={menuOpen} aria-controls="authenticated-account-menu" onClick={() => setMenuOpen(!menuOpen)}>
              <Avatar displayName={session.display_name} />
              <span className="who">
                <strong>{session.display_name}</strong>
                <span>{ROLE_LABEL[session.role] || session.role}</span>
              </span>
              <WorkspaceIcon name="chevron" />
            </button>
            {menuOpen ? (
              <div id="authenticated-account-menu" className="menu" aria-label="قائمة الحساب">
                <p>
                  <strong>{session.display_name}</strong>
                  <span>{ROLE_LABEL[session.role] || session.role}</span>
                </p>
                <button
                  type="button"
                  className="menu-link"
                  onClick={() => {
                    setProfileOpen(true);
                    setMenuOpen(false);
                  }}
                >
                  الملف الشخصي
                </button>
                <button type="button" className="menu-link" onClick={() => go("settings")}>
                  الإعدادات
                </button>
                <button type="button" onClick={onLogout}>
                  تسجيل الخروج
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </header>
      {profileOpen ? (
        <div className="modal-back" role="dialog" aria-modal="true" aria-labelledby="profile-title" onKeyDown={onProfileKeyDown}>
          <div className="modal" ref={profileDialog}>
            <h3 id="profile-title">الملف الشخصي</h3>
            <p>الاسم: {session.display_name}</p>
            <p>الدور: {ROLE_LABEL[session.role] || session.role}</p>
            {session.email ? <p>البريد المؤسسي: {session.email}</p> : <p>البريد المؤسسي: غير متوفر</p>}
            <div className="actions">
              <button className="btn-ghost" type="button" onClick={() => setProfileOpen(false)}>
                إغلاق
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {children}
    </div>
  );
}
