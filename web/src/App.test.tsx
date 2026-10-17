import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api/client";
import type {
  AssignableItem,
  AssignResult,
  ExcelJob,
  ExcelSession,
  ExcelSessionCounts,
  QueueItem,
  ReviewDetail,
  ReviewerSummaryRow,
  SourceFileSummary,
  WorkflowCounts,
} from "./api/types";

const items: Record<string, ReviewDetail> = {};
let queueFilter = "pending";
let queueSort = "priority";
let failNext = "";
let downloadNextError = 0;
// ADMIN by default so the existing general-purpose test suite (written
// before Phase 6 roles existed) keeps exercising every page unchanged;
// dedicated role tests override this to "REVIEWER" explicitly.
let currentMockRole: "REVIEWER" | "ADMIN" = "ADMIN";
let loginShouldFail = false;
let lastLoginArgs: { email: string; password: string } | null = null;
let mockEntraEnabled = false;
// when true, login() reports called but waits for the test to call
// resolvePendingLogin() instead of resolving immediately — mirrors the
// existing blockUpload pattern, used to observe the busy/loading state.
let blockLogin = false;
let pendingLoginResolve: (() => void) | null = null;

function resolvePendingLogin() {
  pendingLoginResolve?.();
  pendingLoginResolve = null;
}

// --- Excel import/session/job mock state -----------------------------------
let excelActiveSession: ExcelSession | null = null;
const excelSessionsById: Record<string, ExcelSession> = {};
const excelJobsById: Record<string, ExcelJob> = {};
let excelSessionSeq = 0;
let excelJobSeq = 0;
let excelLastFileName = "";
let excelUploadInvalid: string[] = [];
let excelUploadTotalRows = 1;
let excelUploadError = "";
let excelUploadErrorMessage = "bad";
// number of "running" polls returned before the job settles to succeeded/failed
let excelJobRunningTicks = 0;
let excelJobShouldFail = false;
let excelJobPollCount = 0;
let excelSessionCounts: ExcelSessionCounts = {
  total: 3,
  excluded: 0,
  duplicate_content: 0,
  canonical_total: 3,
  completed: 0,
  remaining: 3,
};
// when true, uploadExcelWithProgress reports partial progress and then waits
// for the test to call `resolvePendingUpload()` instead of resolving immediately
let blockUpload = false;
let pendingUploadResolve: (() => void) | null = null;

// --- Phase 8: admin assignment mock state -----------------------------------
let mockSources: SourceFileSummary[] = [];
let mockReviewerSummaries: ReviewerSummaryRow[] = [];
let mockAssignableItems: AssignableItem[] = [];
let assignResultOverride: AssignResult | null = null;
const assignCalls: { commentIds: string[]; reviewerUserId: string }[] = [];

function resolvePendingUpload() {
  pendingUploadResolve?.();
  pendingUploadResolve = null;
}

function makeSession(overrides: Partial<ExcelSession> = {}): ExcelSession {
  excelSessionSeq += 1;
  return {
    session_id: `sess-${excelSessionSeq}`,
    original_filename: excelLastFileName || "زوار.xlsx",
    source_file: `uploads/${excelLastFileName || "زوار.xlsx"}`,
    status: "processing",
    created_at: "2026-08-30T10:00:00Z",
    updated_at: "2026-08-30T10:00:00Z",
    counts: { total: 0, excluded: 0, duplicate_content: 0, canonical_total: 0, completed: 0, remaining: 0 },
    latest_job_id: null,
    ...overrides,
  };
}

function makeJob(sessionId: string, overrides: Partial<ExcelJob> = {}): ExcelJob {
  excelJobSeq += 1;
  return {
    job_id: `job-${excelJobSeq}`,
    session_id: sessionId,
    status: "running",
    current_stage: "رفع الملف",
    rows_done: 0,
    rows_total: excelUploadTotalRows,
    started_at: "2026-08-30T10:00:00Z",
    updated_at: "2026-08-30T10:00:00Z",
    finished_at: null,
    error_message: null,
    ...overrides,
  };
}

function detail(id: string, extra: Partial<ReviewDetail> = {}): ReviewDetail {
  return {
    comment_id: id,
    source_reference: "المعجم / نموذج",
    original_comment: extra.original_comment || `نص التعليق ${id}`,
    comment_created_at: extra.comment_created_at || "2026-08-25T17:45:00Z",
    original_title: null,
    entry_id: null,
    platform: null,
    original_type: null,
    primary_intent: "THANKS_PRAISE",
    topic_subtype: "GENERAL",
    sentiment: "POSITIVE",
    confidence: 0.9,
    needs_review: false,
    generation_routing_decision: "GENERATE_DRAFT",
    generation_status: "SUCCEEDED",
    generation_reason_code: "thanks",
    suggested_response: "سعدنا بثقتكم، وملاحظاتكم محل تقدير.",
    review_status: "DRAFT_READY",
    reviewer_id: null,
    reviewer_display_name: null,
    reviewer_action: null,
    reviewed_at: null,
    edited_response: null,
    final_response: null,
    no_response_reason: null,
    ready_to_send: false,
    outbound_status: null,
    outbound_error_code: null,
    review_priority: extra.review_priority ?? 3,
    priority_reason: "routine",
    row_version: 1,
    allowed_actions: ["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"],
    ...extra,
  };
}

function seed() {
  Object.keys(items).forEach((k) => delete items[k]);
  items["8742"] = detail("8742", {
    original_comment: "شكرًا لكم على هذا المعجم الرائع",
    comment_created_at: "2026-08-25T17:45:00Z",
    review_priority: 3,
  });
  items["8743"] = detail("8743", {
    original_comment: "الموقع لا يعمل عند البحث",
    comment_created_at: "2026-08-25T16:00:00Z",
    primary_intent: "COMPLAINT",
    review_priority: 1,
    suggested_response: "نعتذر عن هذا العطل.",
  });
  items["9001"] = detail("9001", {
    original_comment: "إعلان تجاري",
    review_status: "NO_RESPONSE_RECOMMENDED",
    suggested_response: null,
    allowed_actions: ["NO_RESPONSE_NEEDED"],
    generation_routing_decision: "NO_RESPONSE_RECOMMENDED",
  });
  items["8801"] = detail("8801", {
    original_comment: "محتوى غير مكتمل يحتاج مراجعة",
    review_status: "REVIEW_ONLY",
    suggested_response: null,
    allowed_actions: ["EDIT_AND_SEND", "NO_RESPONSE_NEEDED"],
    generation_routing_decision: "REVIEW_ONLY",
    review_priority: 2,
    comment_created_at: "2026-08-25T16:30:00Z",
  });
}

function list(): QueueItem[] {
  let rows = Object.values(items).map((d) => ({
    comment_id: d.comment_id,
    review_status: d.review_status,
    comment_preview: d.original_comment,
    primary_intent: d.primary_intent,
    topic_subtype: d.topic_subtype,
    review_priority: d.review_priority,
    priority_reason: d.priority_reason,
    claimed_by: d.reviewer_id,
    comment_created_at: d.comment_created_at,
    ingested_at: d.comment_created_at,
    suggested_response: d.suggested_response,
    reviewer_action: d.reviewer_action,
    reviewer_display_name: d.reviewer_display_name,
    reviewed_at: d.reviewed_at,
  }));
  if (queueFilter === "pending") rows = rows.filter((r) => ["DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW"].includes(r.review_status));
  if (queueFilter === "awaiting") rows = rows.filter((r) => ["DRAFT_READY", "REVIEW_ONLY"].includes(r.review_status));
  if (queueFilter === "approved") rows = rows.filter((r) => ["APPROVED", "EDITED_AND_APPROVED", "SENT"].includes(r.review_status));
  if (queueFilter === "no_reply") rows = rows.filter((r) => ["NO_REPLY", "NO_RESPONSE_RECOMMENDED"].includes(r.review_status));
  if (queueSort === "newest") rows.sort((a, b) => (b.ingested_at || "").localeCompare(a.ingested_at || ""));
  if (queueSort === "oldest") rows.sort((a, b) => (a.ingested_at || "").localeCompare(b.ingested_at || ""));
  if (queueSort === "priority") {
    rows.sort((a, b) => {
      const byPriority = (a.review_priority || 9) - (b.review_priority || 9);
      if (byPriority !== 0) return byPriority;
      return (a.ingested_at || "").localeCompare(b.ingested_at || "");
    });
  }
  return rows;
}

vi.mock("./api/client", () => ({
  microsoftSignInUrl: "/api/auth/microsoft/start",
  api: {
    health: vi.fn(async () => ({ ok: true, demo_auth: true, entra: mockEntraEnabled })),
    session: vi.fn(async () => ({
      user_id: "u-1",
      display_name: "مروج الراجحي",
      role: currentMockRole,
      auth_mode: "session",
      email: "reviewer@example.com",
    })),
    login: vi.fn(async (body: { email: string; password: string }) => {
      lastLoginArgs = body;
      if (blockLogin) {
        await new Promise<void>((resolve) => {
          pendingLoginResolve = resolve;
        });
      }
      if (loginShouldFail) {
        loginShouldFail = false;
        throw Object.assign(new Error("invalid_credentials"), { status: 401, code: "invalid_credentials" });
      }
      return {
        user_id: "u-1",
        display_name: "مروج الراجحي",
        role: currentMockRole,
        auth_mode: "session",
        email: body.email || "reviewer@example.com",
      };
    }),
    logout: vi.fn(async () => ({ ok: true })),
    accountRequests: vi.fn(async () => ({ requests: [] })),
    adminUsers: vi.fn(async () => ({
      users: [
        { user_id: "u-1", email: "admin@example.com", display_name: "مسؤول تجريبي", role: "ADMIN", is_active: true, created_at: "2026-08-01T00:00:00Z" },
        { user_id: "u-2", email: "reviewer@example.com", display_name: "مراجع تجريبي", role: "REVIEWER", is_active: true, created_at: "2026-08-02T00:00:00Z" },
      ],
    })),
    adminSourceSummary: vi.fn(async () => ({ sources: mockSources })),
    adminReviewers: vi.fn(async () => ({ reviewers: mockReviewerSummaries })),
    adminAssignableItems: vi.fn(
      async (params: { unassignedOnly?: boolean; sourceFile?: string; offset: number; limit: number }) => {
        let rows = mockAssignableItems;
        if (params.sourceFile) rows = rows.filter((r) => r.source_file === params.sourceFile);
        if (params.unassignedOnly) rows = rows.filter((r) => !r.reviewer_user_id);
        const total = rows.length;
        rows = rows.slice(params.offset, params.offset + params.limit);
        return { items: rows, total, offset: params.offset, limit: params.limit };
      },
    ),
    adminAssign: vi.fn(async (commentIds: string[], reviewerUserId: string) => {
      assignCalls.push({ commentIds, reviewerUserId });
      if (assignResultOverride) return assignResultOverride;
      const assigned: string[] = [];
      commentIds.forEach((id) => {
        const item = mockAssignableItems.find((i) => i.comment_id === id);
        if (item) {
          item.reviewer_user_id = reviewerUserId;
          assigned.push(id);
        }
      });
      return { assigned, skipped_already_decided: [], skipped_not_reviewable: [], not_found: [] };
    }),
    analytics: vi.fn(async () => {
      const rows = Object.values(items);
      const pending = rows.filter((r) => ["DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW"].includes(r.review_status)).length;
      const urgent = rows.filter((r) => r.review_priority === 1 && ["DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW"].includes(r.review_status)).length;
      const noReply = rows.filter((r) => ["NO_REPLY", "NO_RESPONSE_RECOMMENDED"].includes(r.review_status)).length;
      return {
        range: "week",
        range_start: "2026-08-19T00:00:00",
        range_end: "2026-08-26T00:00:00",
        kpis: {
          incoming: rows.length,
          reviewed: rows.length - pending,
          pending,
          urgent_pending: urgent,
          overdue_pending: 0,
          no_reply: noReply,
          avg_review_seconds: null,
          avg_review_sample: 0,
        },
        today: {
          reviewed_all: 0,
          reviewed_by_me: 0,
          actions: { SEND_AS_IS: 0, EDIT_AND_SEND: 0, NO_RESPONSE_NEEDED: 0 },
        },
        outcomes: { SEND_AS_IS: 0, EDIT_AND_SEND: 0, NO_RESPONSE_NEEDED: 0 },
        queue_health: { under_day: pending, one_two: 0, three_seven: 0, over_seven: 0, dated: pending, oldest_pending_age_seconds: 3600 },
        quality: { sample: 0, send_as_is_rate: null, edited_rate: null, no_reply_rate: null },
        classification: {
          intent: [
            { key: "THANKS_PRAISE", count: 1 },
            { key: "COMPLAINT", count: 1 },
            { key: "SUGGESTION", count: 0 },
            { key: "INQUIRY", count: 0 },
            { key: "SPAM", count: 0 },
          ],
          topic: [
            { key: "WORD_SUBMISSION", count: 0 },
            { key: "ENTRY_CORRECTION", count: 0 },
            { key: "DICTIONARY_SUGGESTION", count: 0 },
            { key: "TECHNICAL_ISSUE", count: 0 },
            { key: "WORD_OF_YEAR_NOMINATION", count: 0 },
            { key: "OFF_TOPIC_PERSONAL", count: 0 },
            { key: "GENERAL", count: 2 },
            { key: "WORD_OF_YEAR", count: 3 },
          ],
          sentiment: [
            { key: "POSITIVE", count: 1 },
            { key: "NEGATIVE", count: 0 },
            { key: "NEUTRAL", count: 0 },
          ],
        },
        trend: [
          { bucket: "2026-08-25", incoming: 2, reviewed: 0 },
          { bucket: "2026-08-26", incoming: 1, reviewed: 1 },
        ],
        availability: { avg_review_time: "insufficient_timestamps", outcomes: "no_completed_actions", trend: "ok", classification: "ok" },
      };
    }),
    counts: vi.fn(async (): Promise<WorkflowCounts> => {
      const rows = Object.values(items);
      const by_status: Record<string, number> = {};
      rows.forEach((r) => {
        by_status[r.review_status] = (by_status[r.review_status] || 0) + 1;
      });
      return {
        all: rows.length,
        pending: rows.filter((r) => ["DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW"].includes(r.review_status)).length,
        approved: rows.filter((r) => ["APPROVED", "EDITED_AND_APPROVED", "SENT"].includes(r.review_status)).length,
        no_reply: rows.filter((r) => ["NO_REPLY", "NO_RESPONSE_RECOMMENDED"].includes(r.review_status)).length,
        send_failed: 0,
        by_status,
      };
    }),
    queue: vi.fn(async (params: { filter: string; sort: string; search: string; offset: number; limit: number }) => {
      if (failNext === "queue") {
        failNext = "";
        throw Object.assign(new Error("server"), { status: 500, code: "server_error" });
      }
      queueFilter = params.filter;
      queueSort = params.sort;
      let rows = list().filter((r) => !params.search || (r.comment_preview || "").includes(params.search) || r.comment_id.includes(params.search));
      const total = rows.length;
      rows = rows.slice(params.offset, params.offset + params.limit);
      return { items: rows, total, offset: params.offset, limit: params.limit };
    }),
    item: vi.fn(async (id: string) => {
      if (!items[id]) throw Object.assign(new Error("missing"), { status: 404, code: "not_found" });
      return { ...items[id] };
    }),
    claim: vi.fn(async (id: string, version: number) => {
      const row = items[id];
      if (failNext === "claimed") {
        failNext = "";
        throw Object.assign(new Error("claimed"), { status: 409, code: "claimed" });
      }
      if (row.row_version !== version) throw Object.assign(new Error("v"), { status: 409, code: "version_conflict" });
      row.review_status = "IN_REVIEW";
      row.row_version += 1;
      row.allowed_actions = row.suggested_response
        ? ["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"]
        : ["EDIT_AND_SEND", "NO_RESPONSE_NEEDED"];
      return { ...row };
    }),
    sendAsIs: vi.fn(async (id: string, version: number) => {
      if (failNext === "conflict") {
        failNext = "";
        throw Object.assign(new Error("v"), { status: 409, code: "version_conflict" });
      }
      const row = items[id];
      if (row.row_version !== version) throw Object.assign(new Error("v"), { status: 409, code: "version_conflict" });
      row.review_status = "APPROVED";
      row.reviewer_action = "SEND_AS_IS";
      row.final_response = row.suggested_response;
      row.outbound_status = "PENDING";
      row.row_version += 1;
      row.allowed_actions = [];
      return { ...row };
    }),
    editAndSend: vi.fn(async (id: string, version: number, edited: string) => {
      const row = items[id];
      if (row.row_version !== version) throw Object.assign(new Error("v"), { status: 409, code: "version_conflict" });
      row.review_status = "EDITED_AND_APPROVED";
      row.reviewer_action = "EDIT_AND_SEND";
      row.edited_response = edited;
      row.final_response = edited;
      row.outbound_status = "PENDING";
      row.row_version += 1;
      row.allowed_actions = [];
      return { ...row };
    }),
    noResponse: vi.fn(async (id: string, version: number, reason: string) => {
      const row = items[id];
      if (row.row_version !== version) throw Object.assign(new Error("v"), { status: 409, code: "version_conflict" });
      row.review_status = "NO_REPLY";
      row.reviewer_action = "NO_RESPONSE_NEEDED";
      row.no_response_reason = reason;
      row.row_version += 1;
      row.allowed_actions = [];
      return { ...row };
    }),
    uploadExcelWithProgress: vi.fn((file: File, onProgress: (fraction: number) => void) => {
      excelLastFileName = file.name;
      const finish = (resolve: (v: unknown) => void, reject: (e: unknown) => void) => {
        if (excelUploadError) {
          const err = Object.assign(new Error(excelUploadErrorMessage), { code: excelUploadError });
          excelUploadError = "";
          excelUploadErrorMessage = "bad";
          reject(err);
          return;
        }
        onProgress(1);
        resolve({
          upload_token: "a".repeat(32),
          filename: file.name,
          size_bytes: file.size,
          valid: excelUploadInvalid.length === 0,
          total_rows: excelUploadTotalRows,
          errors: excelUploadInvalid,
        });
      };
      return new Promise((resolve, reject) => {
        onProgress(0.35);
        if (blockUpload) {
          pendingUploadResolve = () => finish(resolve, reject);
          return;
        }
        finish(resolve, reject);
      });
    }),
    createExcelSession: vi.fn(async (_uploadToken: string) => {
      const session = makeSession();
      const job = makeJob(session.session_id);
      session.latest_job_id = job.job_id;
      excelSessionsById[session.session_id] = session;
      excelJobsById[job.job_id] = job;
      excelActiveSession = session;
      excelJobPollCount = 0;
      return { session: { ...session }, job_id: job.job_id };
    }),
    activeExcelSession: vi.fn(async () => ({ active: excelActiveSession ? { ...excelActiveSession } : null })),
    getExcelSession: vi.fn(async (sessionId: string) => {
      const session = excelSessionsById[sessionId];
      if (!session) throw Object.assign(new Error("missing"), { status: 404, code: "not_found" });
      return { ...session };
    }),
    resumeExcelSession: vi.fn(async (sessionId: string) => {
      const session = excelSessionsById[sessionId];
      if (!session) throw Object.assign(new Error("missing"), { status: 404, code: "not_found" });
      const job = makeJob(sessionId);
      excelJobsById[job.job_id] = job;
      session.latest_job_id = job.job_id;
      session.status = "processing";
      excelActiveSession = session;
      excelJobPollCount = 0;
      return { job_id: job.job_id };
    }),
    getExcelJob: vi.fn(async (jobId: string) => {
      const job = excelJobsById[jobId];
      if (!job) throw Object.assign(new Error("missing"), { status: 404, code: "not_found" });
      excelJobPollCount += 1;
      if (excelJobPollCount <= excelJobRunningTicks) {
        job.current_stage = excelJobPollCount === 1 ? "تصنيف التعليقات" : "إنشاء الردود المقترحة";
        job.rows_done = Math.min(excelJobPollCount, excelUploadTotalRows);
        job.rows_total = excelUploadTotalRows;
        return { ...job };
      }
      if (excelJobShouldFail) {
        job.status = "failed";
        job.error_message = "تعذر الاتصال بخدمة التصنيف";
        excelJobShouldFail = false;
        return { ...job };
      }
      job.status = "succeeded";
      job.current_stage = "جاهز للمراجعة";
      job.rows_done = excelUploadTotalRows;
      job.rows_total = excelUploadTotalRows;
      job.finished_at = "2026-08-30T10:05:00Z";
      const session = excelSessionsById[job.session_id];
      if (session) {
        session.status = excelSessionCounts.remaining === 0 ? "completed" : "ready";
        session.counts = { ...excelSessionCounts };
        excelActiveSession = session;
      }
      return { ...job };
    }),
    downloadExcelExport: vi.fn(async (_sourceFile?: string) => {
      if (downloadNextError) {
        const err = Object.assign(new Error("no rows"), { status: downloadNextError });
        downloadNextError = 0;
        throw err;
      }
    }),
  },
}));

function renderReview() {
  return render(
    <MemoryRouter initialEntries={["/review"]}>
      <App />
    </MemoryRouter>,
  );
}

async function openReviewWorkspace(user: ReturnType<typeof userEvent.setup>) {
  // A REVIEWER lands directly on the list-first "المسند لي" screen (no
  // nav click needed — see the session-fetch effect); "بدء المراجعة"
  // is the one primary action that enters the focused review mode,
  // starting at whatever is first in the current (unfiltered) queue.
  await user.click(await screen.findByRole("button", { name: "بدء المراجعة" }));
  await screen.findByLabelText("الرد المقترح");
}

// Search-scopes the list to exactly one comment_id, then begins review
// -- deterministic entry into the focused single-item view regardless
// of default sort order (queue-item rows are no longer clickable in
// the list screen; "بدء المراجعة" always opens whatever is first in
// the current filtered set).
async function beginReviewOn(user: ReturnType<typeof userEvent.setup>, commentId: string) {
  const search = await screen.findByLabelText("بحث");
  await user.clear(search);
  await user.type(search, commentId);
  await screen.findByText(`قائمة المراجعة (1)`);
  await user.click(screen.getByRole("button", { name: "بدء المراجعة" }));
  await screen.findByLabelText("الرد المقترح");
}

function resetHarness() {
    seed();
    failNext = "";
    queueFilter = "pending";
    queueSort = "priority";
    downloadNextError = 0;
    currentMockRole = "ADMIN";
    loginShouldFail = false;
    lastLoginArgs = null;
    mockEntraEnabled = false;
    excelActiveSession = null;
    Object.keys(excelSessionsById).forEach((k) => delete excelSessionsById[k]);
    Object.keys(excelJobsById).forEach((k) => delete excelJobsById[k]);
    excelLastFileName = "";
    excelUploadInvalid = [];
    excelUploadTotalRows = 1;
    excelUploadError = "";
    excelUploadErrorMessage = "bad";
    excelJobRunningTicks = 0;
    excelJobShouldFail = false;
    excelJobPollCount = 0;
    excelSessionCounts = { total: 3, excluded: 0, duplicate_content: 0, canonical_total: 3, completed: 0, remaining: 3 };
    blockUpload = false;
    pendingUploadResolve = null;
    mockSources = [
      { source_file: "uploads/batch.xlsx", total_reviewable: 10, unassigned: 4, assigned: 6, completed: 2 },
    ];
    mockReviewerSummaries = [
      { user_id: "rev-1", email: "rev1@example.com", display_name: "مراجع أول", assigned_total: 4, pending: 2, completed: 2 },
      { user_id: "rev-2", email: "rev2@example.com", display_name: "مراجع ثانٍ", assigned_total: 2, pending: 2, completed: 0 },
    ];
    mockAssignableItems = [
      {
        comment_id: "a-1",
        comment_preview: "تعليق قابل للإسناد الأول",
        source_file: "uploads/batch.xlsx",
        review_status: "DRAFT_READY",
        routing_decision: "GENERATE_DRAFT",
        reviewer_user_id: null,
        primary_intent: null,
        topic_subtype: null,
        review_priority: null,
        ingested_at: null,
      },
      {
        comment_id: "a-2",
        comment_preview: "تعليق قابل للإسناد الثاني",
        source_file: "uploads/batch.xlsx",
        review_status: "REVIEW_ONLY",
        routing_decision: "REVIEW_ONLY",
        reviewer_user_id: "rev-1",
        primary_intent: null,
        topic_subtype: null,
        review_priority: null,
        ingested_at: null,
      },
    ];
    assignResultOverride = null;
    assignCalls.length = 0;
    window.localStorage.clear();
}

describe("Reviewer UI", () => {
  beforeEach(() => {
    resetHarness();
  });

  it("renders the login screen in RTL Arabic", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    expect(screen.getByRole("heading", { name: "تسجيل الدخول" })).toBeInTheDocument();
    expect(screen.getByLabelText("البريد الإلكتروني")).toBeInTheDocument();
    expect(screen.getByLabelText("كلمة المرور")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "الدخول عبر Microsoft" })).toBeDisabled();            // retained, but clearly unavailable when SSO is not configured
    expect(screen.getByText("© 2026 RCS")).toBeInTheDocument();
    expect(screen.queryByText("الدعم الفني")).not.toBeInTheDocument();
  });

  it("does not label the shared login page as reviewer-only (ADMIN and REVIEWER share one screen)", async () => {
    // Regression: the heading previously read "تسجيل دخول المراجع"
    // ("reviewer login"), which was factually wrong -- this single page
    // is shared by both ADMIN and REVIEWER, and role-based
    // authorization/routing happens AFTER login, server-side. The
    // heading must be role-neutral and the old wrong wording must never
    // reappear, regardless of which role ultimately logs in here.
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    expect(screen.getByRole("heading", { name: "تسجيل الدخول" })).toBeInTheDocument();
    expect(screen.queryByText("تسجيل دخول المراجع")).not.toBeInTheDocument();
    expect(screen.queryByText(/المراجع/)).not.toBeInTheDocument();
    expect(screen.getByText("استخدم حسابك للوصول إلى نظام مراجعة تعليقات الزوار")).toBeInTheDocument();
  });

  it("forgot-password shows a truthful contact-admin message, not a fake reset flow", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    expect(screen.queryByText(/تواصل مع مسؤول النظام/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "نسيت كلمة المرور؟" }));
    expect(screen.getByText("تواصل مع مسؤول النظام لإعادة تعيين كلمة المرور.")).toBeInTheDocument();
    // Must never imply an email/reset link was sent.
    expect(screen.queryByText(/تم إرسال/)).not.toBeInTheDocument();
    expect(screen.queryByText(/رابط/)).not.toBeInTheDocument();
    // Purely informational — no API call triggered.
    expect(vi.mocked(api).login).not.toHaveBeenCalled();
    // Toggles back off.
    await user.click(screen.getByRole("button", { name: "نسيت كلمة المرور؟" }));
    expect(screen.queryByText("تواصل مع مسؤول النظام لإعادة تعيين كلمة المرور.")).not.toBeInTheDocument();
  });

  it("lands a REVIEWER directly on their own work — one screen, no separate dashboard hop", async () => {
    // REVIEWER navigation now revolves around work: there is no separate
    // "الرئيسية" destination for this role (that stays an ADMIN-only
    // operational overview) — logging in opens the list-first queue
    // screen directly, with a compact summary strip in place of the old
    // stand-alone dashboard page. /api/counts is reviewer-scoped
    // server-side; the mock's shared `items` stands in for "this
    // reviewer's own assigned items" — 4 total, 3 pending, 1 completed
    // (9001 is NO_RESPONSE_RECOMMENDED, counted as not-pending). The
    // composer itself is reached one click later, via "بدء المراجعة" —
    // a REVIEWER is never auto-opened into an item.
    currentMockRole = "REVIEWER";
    renderReview();
    expect(await screen.findByRole("heading", { name: "المسند لي" })).toBeInTheDocument();
    expect(document.querySelector(".dash-greeting")?.textContent).toMatch(/^(صباح الخير|مساء الخير)، مروج$/);
    const summary = within(screen.getByRole("region", { name: "ملخص عملي" }));
    expect(summary.getByText("إجمالي المسند إليّ").nextSibling).toHaveTextContent("4");
    expect(summary.getByText("بانتظار المراجعة").nextSibling).toHaveTextContent("3");
    expect(summary.getByText("مكتمل").nextSibling).toHaveTextContent("1");
    expect(summary.getByText("نسبة الإنجاز").nextSibling).toHaveTextContent("25٪");
    expect(await screen.findByText("قائمة المراجعة (3)")).toBeInTheDocument();
    expect(screen.queryByLabelText("الرد المقترح")).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "بدء المراجعة" }));
    expect(await screen.findByLabelText("الرد المقترح")).toBeInTheDocument();
    // No global/admin-only data source is touched for a REVIEWER.
    expect(vi.mocked(api).analytics).not.toHaveBeenCalled();
    expect(vi.mocked(api).activeExcelSession).not.toHaveBeenCalled();
  });

  it("a REVIEWER's review session spans the whole filtered list, across page boundaries, in both directions", async () => {
    currentMockRole = "REVIEWER";
    Object.keys(items).forEach((k) => delete items[k]);
    for (let i = 0; i < 20; i++) {                                                    // 20 pending comments = 3 pages of 8
      const id = String(7000 + i);
      items[id] = detail(id, { original_comment: `تعليق مراجعة رقم ${i}`, comment_created_at: `2026-08-25T${String(10 + (i % 10)).padStart(2, "0")}:00:00Z`, review_priority: 3 });
    }
    renderReview();
    const user = userEvent.setup();
    expect(await screen.findByText("قائمة المراجعة (20)")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "بدء المراجعة" }));
    await screen.findByLabelText("الرد المقترح");
    const position = () => document.querySelector(".focus-nav span")?.textContent || "";
    expect(position()).toMatch(/^1 من 20/);
    expect(screen.getByLabelText("التعليق السابق")).toBeDisabled();
    for (let step = 2; step <= 20; step++) {                                          // crosses the page boundaries at 8 and 16
      await user.click(screen.getByLabelText("التعليق التالي"));
      await waitFor(() => expect(position()).toMatch(new RegExp(`^${step} من 20`)));
      await screen.findByLabelText("الرد المقترح");
    }
    expect(screen.getByLabelText("التعليق التالي")).toBeDisabled();                   // the true end of the list, not the end of a page
    for (let step = 19; step >= 1; step--) {
      await user.click(screen.getByLabelText("التعليق السابق"));
      await waitFor(() => expect(position()).toMatch(new RegExp(`^${step} من 20`)));
    }
    expect(screen.getByLabelText("التعليق السابق")).toBeDisabled();
    expect(vi.mocked(api).queue.mock.calls.some(([params]) => params.offset === 16)).toBe(true);
    // classification is display-only in the review screen: no control can change intent, topic or sentiment
    expect(screen.queryByRole("combobox", { name: /النية|الموضوع|المشاعر/ })).not.toBeInTheDocument();
  });

  it("shows a clean empty state instead of an import prompt when a REVIEWER has zero assignments", async () => {
    currentMockRole = "REVIEWER";
    Object.keys(items).forEach((k) => delete items[k]);
    renderReview();
    expect(await screen.findByRole("heading", { name: "المسند لي" })).toBeInTheDocument();
    expect(await screen.findByText("لا توجد تعليقات مسندة لك حاليًا")).toBeInTheDocument();
    expect(screen.getByText("ستظهر هنا التعليقات الجديدة عند إسنادها لك.")).toBeInTheDocument();
    // No celebratory language, ever (§18).
    expect(screen.queryByText(/أحسنت/)).not.toBeInTheDocument();
    // No summary stats strip when there is nothing assigned yet.
    expect(screen.queryByRole("region", { name: "ملخص عملي" })?.querySelector(".work-summary-stats")).toBeFalsy();
    // A REVIEWER can never import — no such nav destination exists for this role.
    expect(screen.queryByRole("button", { name: "استيراد التعليقات" })).not.toBeInTheDocument();
  });

  it("shows a clear caught-up state when a REVIEWER has assignments but none pending", async () => {
    currentMockRole = "REVIEWER";
    Object.keys(items).forEach((k) => delete items[k]);
    items["done-1"] = detail("done-1", { review_status: "APPROVED", reviewer_action: "SEND_AS_IS" });
    items["done-2"] = detail("done-2", { review_status: "NO_REPLY", reviewer_action: "NO_RESPONSE_NEEDED" });
    renderReview();
    expect(await screen.findByRole("heading", { name: "المسند لي" })).toBeInTheDocument();
    const summary = within(screen.getByRole("region", { name: "ملخص عملي" }));
    expect(summary.getByText("إجمالي المسند إليّ").nextSibling).toHaveTextContent("2");
    expect(summary.getByText("مكتمل").nextSibling).toHaveTextContent("2");
    expect(summary.getByText("نسبة الإنجاز").nextSibling).toHaveTextContent("100٪");
    // Default filter is "pending", which is now empty — a calm,
    // non-celebratory empty state (§18), never "أحسنت".
    expect(await screen.findByText("اكتملت جميع المراجعات")).toBeInTheDocument();
    expect(screen.getByText("لا توجد تعليقات أخرى بانتظار المراجعة حاليًا.")).toBeInTheDocument();
    expect(screen.queryByText(/أحسنت/)).not.toBeInTheDocument();
    // Switching filters still works — completed history remains reachable
    // on this same screen, unlike the old dead-end "all done" dashboard page.
    await userEvent.setup().click(within(screen.getByLabelText("تصفية القائمة")).getByRole("button", { name: "الكل" }));
    expect(await screen.findByText("قائمة المراجعة (2)")).toBeInTheDocument();
  });

  it("loads the reviewer page, queue, and suggested response", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await openReviewWorkspace(user);
    const composer = await screen.findByLabelText("الرد المقترح");
    expect((composer as HTMLTextAreaElement).value.length).toBeGreaterThan(10);
    expect(screen.getByText("نص التعليق")).toBeInTheDocument();
    expect(screen.getByLabelText("الرد المقترح")).toHaveAttribute("dir", "rtl");
  });

  it("shows the compact السياق context area only when non-empty, with no visitor PII", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    items["8743"] = detail("8743", {
      original_comment: "الموقع لا يعمل عند البحث",
      primary_intent: "COMPLAINT",
      review_priority: 1,
      suggested_response: "نعتذر عن هذا العطل.",
      original_title: "حَسَبَ",
      entry_id: "entry-77",
      platform: "الرياض",
      original_type: "التعليق على المدخل",
    });
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");

    // 8742 has no context fields (all null) — the السياق area must be
    // hidden cleanly, not rendered empty.
    await beginReviewOn(user, "8742");
    expect(screen.queryByLabelText("السياق")).not.toBeInTheDocument();

    // 8743 has all four context fields — each must render, and neither
    // visitor_name nor visitor_email (never sent by the backend/mock)
    // can appear anywhere on the page.
    await user.click(screen.getByRole("button", { name: "العودة للقائمة" }));
    await beginReviewOn(user, "8743");
    const context = await screen.findByLabelText("السياق");
    expect(within(context).getByText("حَسَبَ")).toBeInTheDocument();
    expect(within(context).getByText("entry-77")).toBeInTheDocument();
    expect(within(context).getByText("الرياض")).toBeInTheDocument();
    expect(within(context).getByText("التعليق على المدخل")).toBeInTheDocument();
    expect(screen.queryByText(/visitor/i)).not.toBeInTheDocument();
  });

  it("sorts newest, oldest, and by priority", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await user.selectOptions(screen.getByLabelText("ترتيب"), "newest");
    await waitFor(() => expect(screen.getAllByText("#8742", { selector: ".queue-row span" }).length).toBeGreaterThan(0));
    const firstRow = () => document.querySelector(".queue-row header span")?.textContent;
    expect(firstRow()).toBe("#8742");
    await user.selectOptions(screen.getByLabelText("ترتيب"), "oldest");
    await waitFor(() => expect(firstRow()).toBe("#8743"));
    await user.selectOptions(screen.getByLabelText("ترتيب"), "priority");
    await waitFor(() => expect(firstRow()).toBe("#8743"));
  });

  it("filters and searches the queue", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await user.click(within(screen.getByLabelText("تصفية القائمة")).getByRole("button", { name: "لا يحتاج إلى رد" }));
    expect(await screen.findByText("قائمة المراجعة (1)")).toBeInTheDocument();
    await user.click(within(screen.getByLabelText("تصفية القائمة")).getByRole("button", { name: "بانتظار المراجعة" }));
    await screen.findByText("قائمة المراجعة (3)");
    await user.type(screen.getByLabelText("بحث"), "لا يعمل");
    expect(await screen.findByText("قائمة المراجعة (1)")).toBeInTheDocument();
    await user.clear(screen.getByLabelText("بحث"));
    await user.type(screen.getByLabelText("بحث"), "لا يوجد نص مطابق");
    expect(await screen.findByText("لا توجد نتائج")).toBeInTheDocument();
  });

  it("begins review on a specific filtered item and shows relative time", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8742");
    expect(await screen.findByRole("heading", { name: /تعليق #/ })).toBeInTheDocument();
    expect(screen.getAllByText(/منذ/).length).toBeGreaterThan(0);
  });

  it("SEND_AS_IS keeps the original suggestion and shows outbound-not-configured copy", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8742");
    const composer = await screen.findByLabelText("الرد المقترح");
    const original = (composer as HTMLTextAreaElement).value;
    await user.click(screen.getByRole("button", { name: "اعتماد كما هو" }));
    // The search filter that scoped review to exactly this one item is
    // still active, so both the toast and the (now-only-remaining)
    // item's own banner show the same outbound-not-configured copy —
    // assert at least one, not exactly one.
    expect((await screen.findAllByText("تم اعتماد الرد — بانتظار تكامل الإرسال")).length).toBeGreaterThan(0);
    expect(items["8742"].suggested_response).toBe(original);
    expect(items["8742"].final_response).toBe(original);
    expect(items["8742"].review_status).toBe("APPROVED");
    expect(items["8742"].reviewer_action).toBe("SEND_AS_IS");
  });

  it("switches primary action after edit and preserves original suggestion", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8742");
    const composer = await screen.findByLabelText("الرد المقترح");
    const original = (composer as HTMLTextAreaElement).value;
    await user.clear(composer);
    await user.type(composer, "رد معدّل من المراجع بعد المراجعة.");
    expect(screen.getByRole("button", { name: "تعديل واعتماد" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "تعديل واعتماد" }));
    await waitFor(() => expect(items["8742"].review_status).toBe("EDITED_AND_APPROVED"));
    expect(items["8742"].reviewer_action).toBe("EDIT_AND_SEND");
    expect(items["8742"].suggested_response).toBe(original);
    expect(items["8742"].final_response).toContain("رد معدّل");
  });

  it("enables EDIT_AND_SEND only after the draft changes and restores SEND_AS_IS", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8742");
    const composer = await screen.findByLabelText("الرد المقترح");
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "تعديل واعتماد" })).toBeDisabled();
    await user.clear(composer);
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "تعديل واعتماد" })).toBeDisabled();
    await user.type(composer, "رد معدّل من المراجع بعد المراجعة.");
    expect(screen.getByRole("button", { name: "تعديل واعتماد" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "استعادة الرد المقترح" }));
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "تعديل واعتماد" })).toBeDisabled();
  });

  it("enables EDIT_AND_SEND when the reviewer writes a draft with no Gemini suggestion", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8801");
    const composer = await screen.findByLabelText("الرد المقترح");
    expect((composer as HTMLTextAreaElement).value).toBe("");
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "تعديل واعتماد" })).toBeDisabled();
    await user.type(composer, "يمكنكم استخدام نموذج إضافة كلمة في الموقع.");
    expect(screen.getByRole("button", { name: "تعديل واعتماد" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "استعادة الرد المقترح" })).not.toBeInTheDocument();
  });

  it("does not leak the previous item suggestion when switching queue items", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8742");
    const composer = await screen.findByLabelText("الرد المقترح");
    expect((composer as HTMLTextAreaElement).value).toBe("سعدنا بثقتكم، وملاحظاتكم محل تقدير.");
    await user.click(screen.getByRole("button", { name: "العودة للقائمة" }));
    await beginReviewOn(user, "8743");
    await waitFor(() => expect((screen.getByLabelText("الرد المقترح") as HTMLTextAreaElement).value).toBe("نعتذر عن هذا العطل."));
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "العودة للقائمة" }));
    await beginReviewOn(user, "8801");
    await waitFor(() => expect((screen.getByLabelText("الرد المقترح") as HTMLTextAreaElement).value).toBe(""));
    expect(screen.getByRole("button", { name: "اعتماد كما هو" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "استعادة الرد المقترح" })).not.toBeInTheDocument();
  });

  it("NO_RESPONSE_NEEDED saves a reason", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8742");
    await user.click(within(screen.getByLabelText("المراجعة الحالية")).getByRole("button", { name: "لا يحتاج رد" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "سبب عدم الرد" }), "تكرار");
    await user.click(screen.getByRole("button", { name: "تأكيد" }));
    await waitFor(() => expect(items["8742"].review_status).toBe("NO_REPLY"));
    expect(items["8742"].no_response_reason).toBe("تكرار");
    expect(items["8742"].reviewer_action).toBe("NO_RESPONSE_NEEDED");
  });

  it("shows pagination controls when the queue spans pages", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    for (let i = 0; i < 10; i += 1) {
      items[`extra-${i}`] = detail(`extra-${i}`, { original_comment: `تعليق إضافي ${i}` });
    }
    renderReview();
    await openReviewWorkspace(user);
    await user.click(screen.getByRole("button", { name: "العودة للقائمة" }));
    expect(await screen.findByRole("button", { name: "التالي" })).toBeEnabled();
  });

  it("shows backend and concurrency errors inline", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    failNext = "queue";
    await user.selectOptions(screen.getByLabelText("ترتيب"), "newest");
    expect(await screen.findByText("تعذر تحميل القائمة.")).toBeInTheDocument();
  });

  it("shows a concurrency conflict on send", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByText("قائمة المراجعة (3)");
    await beginReviewOn(user, "8742");
    await screen.findByRole("button", { name: "اعتماد كما هو" });
    failNext = "conflict";
    await user.click(screen.getByRole("button", { name: "اعتماد كما هو" }));
    expect(await screen.findByText(/تعارض في الحفظ/)).toBeInTheDocument();
  });

  it("opens analytics with real operational widgets", async () => {
    const user = userEvent.setup();
    renderReview();
    await user.click(await screen.findByRole("button", { name: "التحليلات" }));
    expect(await screen.findByText("نظرة شاملة على حجم التعليقات، سير المراجعة، ونتائج الردود.")).toBeInTheDocument();
    expect(screen.getByText("التعليقات الواردة")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "اليوم" })).toBeEnabled();
    expect(screen.getByText("اتجاه التعليقات والمراجعة")).toBeInTheDocument();
    expect(screen.getByText("جودة الردود والمراجعة")).toBeInTheDocument();
    expect(screen.getByText("تحليل التعليقات")).toBeInTheDocument();
    expect(screen.getByText("نبرة التعليق")).toBeInTheDocument();
    expect(screen.getByText("تعديل ثم اعتماد")).toBeInTheDocument();
    expect(screen.getByText("ترشيح كلمة العام")).toBeInTheDocument();
    expect(screen.getByText("اقتراح مفردة")).toBeInTheDocument();
    expect(screen.queryByText("كلمة العام")).not.toBeInTheDocument();
    const topicLabels = [...screen.getByTestId("class-dist-topic").querySelectorAll(".analytics-dist-labels > span")].map(
      (el) => el.textContent,
    );
    expect(topicLabels).toEqual([
      "اقتراح مفردة",
      "تصحيح مدخل",
      "اقتراح معجمي",
      "مشكلة تقنية",
      "ترشيح كلمة العام",
      "خارج الموضوع",
      "عام",
    ]);
    expect(screen.queryByText("WORD_OF_YEAR")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("المساعدة")).not.toBeInTheDocument();
    // No inert notification control without a real notification integration.
    expect(screen.queryByLabelText("إشعارات")).not.toBeInTheDocument();
  });

  it("shows exactly seven canonical Arabic topic classes and hides WORD_OF_YEAR", async () => {
    const user = userEvent.setup();
    renderReview();
    await user.click(await screen.findByRole("button", { name: "التحليلات" }));
    await screen.findByText("تحليل التعليقات");
    const topicLabels = [...screen.getByTestId("class-dist-topic").querySelectorAll(".analytics-dist-labels > span")].map(
      (el) => el.textContent,
    );
    expect(topicLabels).toEqual([
      "اقتراح مفردة",
      "تصحيح مدخل",
      "اقتراح معجمي",
      "مشكلة تقنية",
      "ترشيح كلمة العام",
      "خارج الموضوع",
      "عام",
    ]);
    expect(screen.queryByText("WORD_OF_YEAR")).not.toBeInTheDocument();
  });

  it("opens local settings preferences", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    // Settings now lives in the account menu, not the primary sidebar —
    // open the account menu first, then its "الإعدادات" link.
    await user.click(await screen.findByRole("button", { name: "الحساب" }));
    await user.click(screen.getByRole("button", { name: "الإعدادات" }));
    expect(await screen.findByText("ترتيب قائمة المراجعة الافتراضي")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "الإعدادات" })).toBeInTheDocument();
    expect(screen.getByText("حدد الترتيب الذي تظهر به التعليقات عند فتح صفحة المراجعة.")).toBeInTheDocument();
    expect(screen.getByText("المساعدة والمعلومات")).toBeInTheDocument();
    expect(screen.getByText("الدعم الفني")).toBeInTheDocument();
    expect(screen.getByText("اسم النظام")).toBeInTheDocument();
    // No demo/environment wording surfaced to a normal user.
    expect(screen.queryByText(/نسخة تجريبية|بيئة العرض|demo/i)).not.toBeInTheDocument();
    expect(screen.queryByText("0.1.0")).not.toBeInTheDocument();
    expect(screen.queryByText("الإشعارات")).not.toBeInTheDocument();
  });

  it("applies the settings default sort when the review queue is opened", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await user.click(await screen.findByRole("button", { name: "الحساب" }));
    await user.click(screen.getByRole("button", { name: "الإعدادات" }));
    await user.selectOptions(await screen.findByLabelText("الترتيب الافتراضي"), "oldest");
    await user.click(screen.getByRole("button", { name: "المسند لي" }));
    await waitFor(() => {
      const rows = screen.getAllByText(/^#\d+$/, { selector: ".queue-row span" });
      expect(rows[0]).toHaveTextContent("#8743");
    });
    expect(window.localStorage.getItem("rcs.defaultQueueSort")).toBe("oldest");
  });

  it("shows the authenticated user's real name, role, and email in the profile panel", async () => {
    const user = userEvent.setup();
    renderReview();
    await user.click(await screen.findByRole("button", { name: "الحساب" }));
    await user.click(screen.getByRole("button", { name: "الملف الشخصي" }));
    expect(screen.getByText("الاسم: مروج الراجحي")).toBeInTheDocument();
    expect(screen.queryByText(/الاسم الأول/)).not.toBeInTheDocument();
    expect(screen.getByText("الدور: مسؤول النظام")).toBeInTheDocument();
    expect(screen.getByText("البريد المؤسسي: reviewer@example.com")).toBeInTheDocument();
  });

  it("rejects an inverted custom analytics range without swapping dates", async () => {
    const user = userEvent.setup();
    renderReview();
    await user.click(await screen.findByRole("button", { name: "التحليلات" }));
    await user.click(screen.getByRole("button", { name: "نطاق مخصص" }));
    const from = await screen.findByLabelText("من تاريخ");
    const to = screen.getByLabelText("إلى تاريخ");
    fireEvent.change(from, { target: { value: "2026-08-20" } });
    fireEvent.change(to, { target: { value: "2026-08-01" } });
    expect(await screen.findByText("تاريخ البداية يجب أن يسبق تاريخ النهاية أو يساويه.")).toBeInTheDocument();
    expect((from as HTMLInputElement).value).toBe("2026-08-20");
    expect((to as HTMLInputElement).value).toBe("2026-08-01");
  });

  describe("Excel import workflow", () => {
    async function openImportPage(user: ReturnType<typeof userEvent.setup>) {
      renderReview();
      // Import is no longer a standalone sidebar destination — it's
      // reached contextually from Admin Home's own CTA (§4).
      await user.click(await screen.findByRole("button", { name: "استيراد ملف جديد" }));
    }

    async function uploadFile(user: ReturnType<typeof userEvent.setup>, name = "زوار.xlsx") {
      const input = screen.getByTestId("excel-file-input") as HTMLInputElement;
      const file = new File(["dummy"], name, {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      });
      await user.upload(input, file);
    }

    it("shows the empty state with the import CTA before any file is active", async () => {
      const user = userEvent.setup();
      await openImportPage(user);
      expect(await screen.findByRole("heading", { name: "استيراد ملف Excel" })).toBeInTheDocument();
      expect(screen.getByTestId("excel-file-input")).toBeInTheDocument();
    });

    it("uses simple, non-technical reviewer-facing Arabic copy instead of implementation terminology", async () => {
      const user = userEvent.setup();
      await openImportPage(user);
      expect(
        await screen.findByText(
          "استورد ملف تعليقات الزوار لتهيئته للمراجعة. سيقوم النظام بمعالجة التعليقات وتصنيفها وإنشاء الردود المقترحة تلقائيًا.",
        ),
      ).toBeInTheDocument();
      const pageText = document.body.textContent || "";
      for (const technicalTerm of ["pipeline", "الفلترة المسبقة", "خط الأنابيب", "Gemini", "frozen model", "prefilter", "routing"]) {
        expect(pageText).not.toContain(technicalTerm);
      }
    });

    it("shows real upload progress, then auto-starts processing once the upload finishes", async () => {
      const user = userEvent.setup();
      blockUpload = true;
      excelJobRunningTicks = 1;
      await openImportPage(user);
      await uploadFile(user, "تعليقات_الزوار.xlsx");

      expect(await screen.findByText("جاري رفع الملف")).toBeInTheDocument();
      expect(screen.getAllByText("تعليقات_الزوار.xlsx").length).toBeGreaterThanOrEqual(1);
      const fill = document.querySelector(".import-progress-fill") as HTMLElement;
      expect(fill.style.width).toBe("35%");

      await act(async () => {
        resolvePendingUpload();
      });
      await screen.findByRole("list", { name: "مراحل المعالجة" });
      expect(await screen.findByText("تصنيف التعليقات")).toBeInTheDocument();
      expect(vi.mocked(api).createExcelSession).toHaveBeenCalled();
    });

    it("shows a clear Arabic error and stays on the empty state for an invalid workbook", async () => {
      const user = userEvent.setup();
      excelUploadInvalid = ["الملف لا يحتوي على الأعمدة المتوقعة: المحتوى"];
      await openImportPage(user);
      const callsBefore = vi.mocked(api).createExcelSession.mock.calls.length;
      await uploadFile(user);
      expect(await screen.findByText(excelUploadInvalid[0])).toBeInTheDocument();
      expect(await screen.findByRole("heading", { name: "استيراد ملف Excel" })).toBeInTheDocument();
      expect(vi.mocked(api).createExcelSession.mock.calls.length).toBe(callsBefore);
    });

    it("rejects an unsupported file type before ever uploading it", async () => {
      const user = userEvent.setup({ applyAccept: false });
      await openImportPage(user);
      const callsBefore = vi.mocked(api).uploadExcelWithProgress.mock.calls.length;
      const input = screen.getByTestId("excel-file-input") as HTMLInputElement;
      const file = new File(["MZ"], "برنامج.exe", { type: "application/octet-stream" });
      await user.upload(input, file);
      expect(await screen.findByText("نوع الملف غير مدعوم — الملفات المدعومة حاليًا: .xlsx فقط")).toBeInTheDocument();
      expect(vi.mocked(api).uploadExcelWithProgress.mock.calls.length).toBe(callsBefore);
    });

    it("surfaces the real error detail instead of a misleading 'corrupt file' message for an unexpected failure", async () => {
      const user = userEvent.setup();
      excelUploadError = "invalid_request";
      excelUploadErrorMessage = "'utf-8' codec can't decode byte 0x90 in position 22: invalid start byte";
      await openImportPage(user);
      await uploadFile(user);
      expect(
        await screen.findByText(
          "تعذّر رفع الملف: 'utf-8' codec can't decode byte 0x90 in position 22: invalid start byte",
        ),
      ).toBeInTheDocument();
      expect(screen.queryByText(/تأكد أنه ملف Excel/)).not.toBeInTheDocument();
    });

    it("advances the processing checklist through real backend stages, then shows the ready state with counts and a start-review CTA", async () => {
      const user = userEvent.setup();
      excelUploadTotalRows = 5;
      excelJobRunningTicks = 1;
      excelSessionCounts = { total: 5, excluded: 1, duplicate_content: 0, canonical_total: 4, completed: 0, remaining: 4 };
      await openImportPage(user);
      await uploadFile(user, "batch.xlsx");

      await screen.findByText("تصنيف التعليقات");
      await waitFor(() => expect(screen.getByText(/تمت معالجة/)).toBeInTheDocument());

      expect(await screen.findByText("✓ الملف جاهز للمراجعة", {}, { timeout: 2500 })).toBeInTheDocument();
      expect(screen.getByText("إجمالي التعليقات").nextSibling).toHaveTextContent("4");
      expect(screen.getByText("المستبعد").nextSibling).toHaveTextContent("1");
      expect(screen.getByText("تمت مراجعته").nextSibling).toHaveTextContent("0");
      expect(screen.getByText("المتبقي").nextSibling).toHaveTextContent("4");
      expect(screen.getByRole("button", { name: "ابدأ المراجعة" })).toBeInTheDocument();
      // Internal ML/pipeline details must never leak into this reviewer-facing UI.
      expect(screen.queryByText(/gemini|rg-v7|marbertv2|tfidf/i)).not.toBeInTheDocument();
    });

    it("shows a failed processing state with a retry action instead of a misleading success", async () => {
      const user = userEvent.setup();
      excelJobShouldFail = true;
      await openImportPage(user);
      await uploadFile(user);
      expect(await screen.findByText("تعذّرت المعالجة: تعذر الاتصال بخدمة التصنيف")).toBeInTheDocument();
      const retry = screen.getByRole("button", { name: "إعادة المحاولة" });
      excelJobShouldFail = false;
      await user.click(retry);
      expect(await screen.findByRole("button", { name: "ابدأ المراجعة" })).toBeInTheDocument();
    });

    it("opens assignment scoped to the active file from the ready state's start-review CTA", async () => {
      // ADMIN's next step after processing is assignment, not a personal
      // review queue (ADMIN has none — see §4/§7): "ابدأ المراجعة" now
      // takes the admin straight to Assignment, scoped to that file.
      const user = userEvent.setup();
      await openImportPage(user);
      await uploadFile(user, "batch.xlsx");
      await user.click(await screen.findByRole("button", { name: "ابدأ المراجعة" }));
      await screen.findByRole("heading", { name: "الإسناد", level: 2 });
      expect(vi.mocked(api).adminAssignableItems).toHaveBeenCalledWith(
        expect.objectContaining({ sourceFile: "uploads/batch.xlsx" }),
      );
    });

    it("shows the resume state with remaining/completed counts for an already-active file across a reload", async () => {
      const user = userEvent.setup();
      excelLastFileName = "زوار_سابق.xlsx";
      const session = makeSession({
        status: "ready",
        counts: { total: 10, excluded: 2, duplicate_content: 0, canonical_total: 8, completed: 3, remaining: 5 },
      });
      excelSessionsById[session.session_id] = session;
      excelActiveSession = session;

      await openImportPage(user);
      expect(await screen.findByRole("heading", { name: "زوار_سابق.xlsx" })).toBeInTheDocument();
      expect(screen.getByText("تمت مراجعته").nextSibling).toHaveTextContent("3");
      expect(screen.getByText("المتبقي").nextSibling).toHaveTextContent("5");
      expect(screen.getByRole("button", { name: "متابعة المراجعة" })).toBeInTheDocument();
    });

    it("shows the completed state and allows downloading the reviewed workbook", async () => {
      const user = userEvent.setup();
      excelLastFileName = "مكتمل.xlsx";
      const session = makeSession({
        status: "completed",
        counts: { total: 4, excluded: 0, duplicate_content: 0, canonical_total: 4, completed: 4, remaining: 0 },
      });
      excelSessionsById[session.session_id] = session;
      excelActiveSession = session;

      await openImportPage(user);
      expect(await screen.findByText("✓ اكتملت مراجعة الملف")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "ابدأ المراجعة" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "متابعة المراجعة" })).not.toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "تنزيل ملف الردود" }));
      await waitFor(() => {
        expect(vi.mocked(api).downloadExcelExport).toHaveBeenCalledWith(session.source_file);
      });
    });

    it("shows a clear message instead of a misleading export when nothing is ready yet", async () => {
      const user = userEvent.setup();
      downloadNextError = 409;
      excelLastFileName = "مكتمل.xlsx";
      const session = makeSession({
        status: "completed",
        counts: { total: 1, excluded: 0, duplicate_content: 0, canonical_total: 1, completed: 1, remaining: 0 },
      });
      excelSessionsById[session.session_id] = session;
      excelActiveSession = session;

      await openImportPage(user);
      await screen.findByText("✓ اكتملت مراجعة الملف");
      await user.click(screen.getByRole("button", { name: "تنزيل ملف الردود" }));
      expect(await screen.findByText("لا توجد نتائج مراجعة جاهزة للتصدير")).toBeInTheDocument();
    });

    it("resets to the empty state and allows importing a new file after completion", async () => {
      const user = userEvent.setup();
      excelLastFileName = "مكتمل.xlsx";
      const session = makeSession({
        status: "completed",
        counts: { total: 1, excluded: 0, duplicate_content: 0, canonical_total: 1, completed: 1, remaining: 0 },
      });
      excelSessionsById[session.session_id] = session;
      excelActiveSession = session;

      await openImportPage(user);
      await screen.findByText("✓ اكتملت مراجعة الملف");
      await user.click(screen.getByRole("button", { name: "استيراد ملف جديد" }));
      expect(await screen.findByRole("heading", { name: "استيراد ملف Excel" })).toBeInTheDocument();
    });
  });
});

describe("Phase 6 — real authentication and roles", () => {
  beforeEach(() => {
    seed();
    currentMockRole = "ADMIN";
    loginShouldFail = false;
    lastLoginArgs = null;
    mockEntraEnabled = false;
    excelActiveSession = null;
    blockLogin = false;
    pendingLoginResolve = null;
    window.localStorage.clear();
  });

  it("submits the real email/password form to the backend and navigates to the workspace on success", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    await user.type(screen.getByLabelText("البريد الإلكتروني"), "reviewer@example.com");
    await user.type(screen.getByLabelText("كلمة المرور"), "correct-password");
    await user.click(screen.getByRole("button", { name: "تسجيل الدخول" }));
    await screen.findByRole("heading", { name: "الرئيسية" });
    expect(lastLoginArgs).toEqual({ email: "reviewer@example.com", password: "correct-password" });
  });

  it("shows a loading state on the submit button while the login request is in flight", async () => {
    const user = userEvent.setup();
    blockLogin = true;
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    await user.type(screen.getByLabelText("البريد الإلكتروني"), "reviewer@example.com");
    await user.type(screen.getByLabelText("كلمة المرور"), "correct-password");
    await user.click(screen.getByRole("button", { name: "تسجيل الدخول" }));

    const submitButton = await screen.findByRole("button", { name: "جاري تسجيل الدخول…" });
    expect(submitButton).toBeDisabled();

    await act(async () => {
      resolvePendingLogin();
    });
    await screen.findByRole("heading", { name: "الرئيسية" });
  });

  it("shows one generic error for invalid credentials without revealing which field was wrong", async () => {
    const user = userEvent.setup();
    loginShouldFail = true;
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    await user.type(screen.getByLabelText("البريد الإلكتروني"), "someone@example.com");
    await user.type(screen.getByLabelText("كلمة المرور"), "wrong-password");
    await user.click(screen.getByRole("button", { name: "تسجيل الدخول" }));
    expect(await screen.findByText("البريد الإلكتروني أو كلمة المرور غير صحيحة.")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "الرئيسية" })).not.toBeInTheDocument();
  });

  it("hides admin-only navigation and blocks the analytics page for a REVIEWER", async () => {
    const user = userEvent.setup();
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByRole("heading", { name: "المسند لي" });
    expect(screen.queryByRole("button", { name: "الرئيسية" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "التحليلات" })).not.toBeInTheDocument();
    // role label shown in the account menu reflects the real role
    await user.click(screen.getByRole("button", { name: "الحساب" }));
    expect(screen.getAllByText("مراجع").length).toBeGreaterThan(0);
  });

  it("shows admin-only navigation, global analytics, and the read-only users list for an ADMIN", async () => {
    const user = userEvent.setup();
    currentMockRole = "ADMIN";
    renderReview();
    await user.click(await screen.findByRole("button", { name: "التحليلات" }));
    await screen.findByRole("heading", { name: "التحليلات", level: 2 });

    await user.click(screen.getByRole("button", { name: "الحساب" }));
    await user.click(screen.getByRole("button", { name: "الإعدادات" }));
    await screen.findByRole("heading", { name: "المستخدمون" });
    expect(await screen.findByText(/admin@example.com/)).toBeInTheDocument();
    expect(await screen.findByText(/reviewer@example.com/)).toBeInTheDocument();
  });

  it("restores an authenticated session on reload without re-prompting for login", async () => {
    renderReview();
    expect(await screen.findByRole("heading", { name: "الرئيسية" })).toBeInTheDocument();
    expect(vi.mocked(api).session).toHaveBeenCalled();
  });

  it("logout clears the session and returns to the login screen", async () => {
    const user = userEvent.setup();
    renderReview();
    await user.click(await screen.findByRole("button", { name: "الحساب" }));
    await user.click(screen.getByRole("button", { name: "تسجيل الخروج" }));
    expect(vi.mocked(api).logout).toHaveBeenCalled();
  });
});

describe("Phase 8 — Admin React UI", () => {
  beforeEach(() => {
    seed();
    currentMockRole = "ADMIN";
    excelActiveSession = null;
    mockSources = [
      { source_file: "uploads/batch.xlsx", total_reviewable: 10, unassigned: 4, assigned: 6, completed: 2 },
    ];
    mockReviewerSummaries = [
      { user_id: "rev-1", email: "rev1@example.com", display_name: "مراجع أول", assigned_total: 4, pending: 2, completed: 2 },
      { user_id: "rev-2", email: "rev2@example.com", display_name: "مراجع ثانٍ", assigned_total: 2, pending: 2, completed: 0 },
    ];
    mockAssignableItems = [
      {
        comment_id: "a-1",
        comment_preview: "تعليق قابل للإسناد الأول",
        source_file: "uploads/batch.xlsx",
        review_status: "DRAFT_READY",
        routing_decision: "GENERATE_DRAFT",
        reviewer_user_id: null,
        primary_intent: null,
        topic_subtype: null,
        review_priority: null,
        ingested_at: null,
      },
      {
        comment_id: "a-2",
        comment_preview: "تعليق قابل للإسناد الثاني",
        source_file: "uploads/batch.xlsx",
        review_status: "REVIEW_ONLY",
        routing_decision: "REVIEW_ONLY",
        reviewer_user_id: "rev-1",
        primary_intent: null,
        topic_subtype: null,
        review_priority: null,
        ingested_at: null,
      },
    ];
    assignResultOverride = null;
    assignCalls.length = 0;
    window.localStorage.clear();
  });

  it("shows the admin navigation and a real-data operational overview for an ADMIN", async () => {
    renderReview();
    await screen.findByRole("heading", { name: "الرئيسية" });
    const sidebar = within(document.querySelector(".sidebar")!);
    expect(sidebar.getByRole("button", { name: "الإسناد" })).toBeInTheDocument();
    expect(sidebar.getByRole("button", { name: "المراجعات المكتملة" })).toBeInTheDocument();
    expect(sidebar.getByRole("button", { name: "التحليلات" })).toBeInTheDocument();
    expect(sidebar.getByRole("button", { name: "الرسائل" })).toBeInTheDocument();
    // No standalone import/reviewer-progress nav destinations anymore —
    // reached contextually instead (§4).
    expect(screen.queryByRole("button", { name: "استيراد التعليقات" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "متابعة المراجعين" })).not.toBeInTheDocument();

    // Keep contracts distinct: global total=4, source completed=2,
    // source unassigned=4, and actual IN_REVIEW=0 (not all pending work).
    const metrics = within(await screen.findByRole("region", { name: "حالة عبء المراجعة" }));
    await waitFor(() => {
      expect(metrics.getByText("إجمالي التعليقات").closest("article")).toHaveTextContent("٤");
      expect(metrics.getByText("مكتملة").closest("article")).toHaveTextContent("٢");
      expect(metrics.getByText("قيد المراجعة").closest("article")).toHaveTextContent("٠");
      expect(metrics.getByText("غير مسندة").closest("article")).toHaveTextContent("٤");
    });
  });

  it("hides every admin-only nav item and admin page content for a REVIEWER", async () => {
    currentMockRole = "REVIEWER";
    renderReview();
    await screen.findByRole("heading", { name: "المسند لي" });
    expect(screen.queryByRole("button", { name: "الرئيسية" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "الإسناد" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "المراجعات المكتملة" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "التحليلات" })).not.toBeInTheDocument();
    // The reviewer's own work screen renders instead of any admin
    // operational content — not just a hidden nav button.
    expect(screen.queryByText("دورة العمل الحالية")).not.toBeInTheDocument();
    expect(screen.queryByTestId("excel-file-input")).not.toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "العناصر القابلة للإسناد" })).not.toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "تقدم المراجعين" })).not.toBeInTheDocument();
  });

  it("shows real reviewer progress with completion percentage embedded in the assignment workspace", async () => {
    // متابعة المراجعين is no longer a standalone destination — it is
    // embedded directly in Assignment (§4/§9), where an admin already
    // needs the same reviewer list to pick who to assign to.
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "الرئيسية" });
    await user.click(within(document.querySelector(".sidebar")!).getByRole("button", { name: "الإسناد" }));
    await screen.findByRole("heading", { name: "الإسناد", level: 2 });
    await user.click(screen.getByText("أحمال المراجعين وتقدم العمل"));
    const progressTable = await screen.findByRole("table", { name: "تقدم المراجعين" });
    // Scoped: the Assignment filter bar's "assigned to" select also lists reviewer names.
    expect(within(progressTable).getByText("مراجع أول")).toBeInTheDocument();
    expect(screen.getByText("rev1@example.com")).toBeInTheDocument();
    // completed=2 / assigned_total=4 -> 50%
    expect(screen.getByText("50٪")).toBeInTheDocument();
    expect(screen.getByText("0٪")).toBeInTheDocument();
  });

  it("lists real assignable items scoped by source file and unassigned-only filter", async () => {
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "الرئيسية" });
    await user.click(within(document.querySelector(".sidebar")!).getByRole("button", { name: "الإسناد" }));
    await screen.findByRole("table", { name: "العناصر القابلة للإسناد" });
    expect(screen.getByText("تعليق قابل للإسناد الأول")).toBeInTheDocument();
    expect(screen.queryByText("تعليق قابل للإسناد الثاني")).not.toBeInTheDocument();

    await user.click(screen.getByLabelText("غير المسند فقط"));
    await screen.findByText("تعليق قابل للإسناد الثاني");
    expect(screen.getByText("تعليق قابل للإسناد الأول")).toBeInTheDocument();
  });

  it("assigns selected items to the chosen reviewer and shows the real backend result", async () => {
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "الرئيسية" });
    await user.click(within(document.querySelector(".sidebar")!).getByRole("button", { name: "الإسناد" }));
    await screen.findByRole("table", { name: "العناصر القابلة للإسناد" });

    await user.click(screen.getByLabelText("تحديد a-1"));
    await user.selectOptions(screen.getByLabelText("اختر المراجع"), "rev-1");
    await user.click(screen.getByRole("button", { name: "إسناد المحدد" }));

    await waitFor(() => expect(vi.mocked(api).adminAssign).toHaveBeenCalledWith(["a-1"], "rev-1"));
    expect(assignCalls[0]).toEqual({ commentIds: ["a-1"], reviewerUserId: "rev-1" });
    expect(await screen.findByText(/تم إسناد/)).toBeInTheDocument();
    expect(screen.getByText("1", { selector: "strong" })).toBeInTheDocument();
  });

  it("shows the real skipped/not-found counts from the backend instead of assuming full success", async () => {
    const user = userEvent.setup();
    assignResultOverride = {
      assigned: ["a-1"],
      skipped_already_decided: ["a-2"],
      skipped_not_reviewable: [],
      not_found: ["missing-1"],
    };
    renderReview();
    await screen.findByRole("heading", { name: "الرئيسية" });
    await user.click(within(document.querySelector(".sidebar")!).getByRole("button", { name: "الإسناد" }));
    await screen.findByRole("table", { name: "العناصر القابلة للإسناد" });
    await user.click(screen.getByLabelText("تحديد جميع عناصر الصفحة"));
    await user.selectOptions(screen.getByLabelText("اختر المراجع"), "rev-2");
    await user.click(screen.getByRole("button", { name: "إسناد المحدد" }));

    expect(await screen.findByText(/تم تجاوز 1 عنصر لأنه تم اتخاذ قرار مراجعة بشأنها مسبقًا/)).toBeInTheDocument();
    expect(screen.getByText(/تعذّر العثور على 1 عنصر/)).toBeInTheDocument();
  });
});

describe("Phase 9 — Microsoft Entra SSO", () => {
  beforeEach(() => {
    seed();
    currentMockRole = "ADMIN";
    loginShouldFail = false;
    lastLoginArgs = null;
    mockEntraEnabled = false;
    window.localStorage.clear();
  });

  it("keeps the Microsoft button but disabled, with an explicit unavailable note, when the backend reports Entra is not configured", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "تسجيل الدخول" });
    await waitFor(() => expect(vi.mocked(api).health).toHaveBeenCalled());
    expect(screen.getByRole("button", { name: "الدخول عبر Microsoft" })).toBeDisabled();
    expect(screen.queryByRole("link", { name: "الدخول عبر Microsoft" })).not.toBeInTheDocument();    // never a working link, never a simulated login
    expect(screen.getByText("الدخول عبر Microsoft غير متاح حاليًا — لم يتم إعداده بعد.")).toBeInTheDocument();
  });

  it("shows a Microsoft sign-in link that targets the backend auth-start endpoint when Entra is configured", async () => {
    mockEntraEnabled = true;
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    const link = await screen.findByRole("link", { name: /الدخول عبر Microsoft/ });
    expect(link).toHaveAttribute("href", "/api/auth/microsoft/start");
    // Email/password login stays visible alongside it.
    expect(screen.getByLabelText("البريد الإلكتروني")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "تسجيل الدخول" })).toBeInTheDocument();
  });

  it.each([
    ["not_configured", "تسجيل الدخول عبر Microsoft غير متاح حاليًا."],
    ["sso_cancelled_or_expired", "تم إلغاء تسجيل الدخول عبر Microsoft أو انتهت صلاحية المحاولة. يرجى المحاولة مرة أخرى."],
    ["sso_failed", "تعذر تسجيل الدخول عبر Microsoft. حاول مرة أخرى أو استخدم البريد الإلكتروني وكلمة المرور."],
    ["account_not_registered", "هذا الحساب غير مسجَّل في النظام. يرجى التواصل مع مسؤول النظام."],
    ["some_unknown_code", "تعذر تسجيل الدخول عبر Microsoft. حاول مرة أخرى أو استخدم البريد الإلكتروني وكلمة المرور."],
  ])("renders a clean Arabic message for ?error=%s without leaking the raw code", async (code, expected) => {
    render(
      <MemoryRouter initialEntries={[`/?error=${code}`]}>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByText(expected)).toBeInTheDocument();
    // Never surface the raw machine-readable code to the user.
    expect(screen.queryByText(code)).not.toBeInTheDocument();
  });

  it("still supports normal email/password login when Entra is configured", async () => {
    mockEntraEnabled = true;
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByRole("link", { name: /الدخول عبر Microsoft/ });
    await user.type(screen.getByLabelText("البريد الإلكتروني"), "reviewer@example.com");
    await user.type(screen.getByLabelText("كلمة المرور"), "correct-password");
    await user.click(screen.getByRole("button", { name: "تسجيل الدخول" }));
    await screen.findByRole("heading", { name: "الرئيسية" });
    expect(lastLoginArgs).toEqual({ email: "reviewer@example.com", password: "correct-password" });
  });

});

describe("UAT remediation", () => {
  beforeEach(() => {
    resetHarness();
  });

  it("profile dialog: focus moves in, Tab is trapped, Escape closes it and focus returns to the account control", async () => {
    currentMockRole = "REVIEWER";
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "المسند لي" });
    const trigger = screen.getByRole("button", { name: "الحساب" });
    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "الملف الشخصي" }));
    const dialog = await screen.findByRole("dialog", { name: "الملف الشخصي" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    const close = within(dialog).getByRole("button", { name: "إغلاق" });
    expect(close).toHaveFocus();
    await user.tab();
    expect(close).toHaveFocus();                                  // a single control: Tab and Shift+Tab stay inside
    await user.tab({ shift: true });
    expect(close).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "الملف الشخصي" })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("profile dialog: the close button also restores focus", async () => {
    currentMockRole = "REVIEWER";
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "المسند لي" });
    const trigger = screen.getByRole("button", { name: "الحساب" });
    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "الملف الشخصي" }));
    await user.click(within(await screen.findByRole("dialog", { name: "الملف الشخصي" })).getByRole("button", { name: "إغلاق" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("review detail shows stored classification conflicts as Arabic notes - no raw codes, no confidence", async () => {
    currentMockRole = "REVIEWER";
    items["8743"].context_conflicts = ["intent:calm_correction_predicted_complaint", "topic:something_new_and_unknown", "intent:calm_correction_predicted_complaint"];
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "المسند لي" });
    await beginReviewOn(user, "8743");
    const notes = await screen.findByRole("region", { name: "ملاحظات التصنيف" });
    expect(within(notes).getByText("صُنّف التعليق كشكوى، لكن صياغته تبدو تصحيحًا هادئًا دون نبرة استياء.")).toBeInTheDocument();
    expect(within(notes).getByText("تنبيه آلي: قد لا يطابق التصنيف المعروض نص التعليق.")).toBeInTheDocument();   // unknown code -> generic sentence
    expect(within(notes).getAllByRole("listitem")).toHaveLength(2);                                             // duplicates collapsed
    expect(document.body.textContent).not.toMatch(/calm_correction|intent:|topic:|something_new/);
    expect(document.body.textContent).not.toMatch(/ثقة|confidence/i);
  });

  it("review detail shows no notes section when there are no conflicts", async () => {
    currentMockRole = "REVIEWER";
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "المسند لي" });
    await beginReviewOn(user, "8743");
    expect(screen.queryByRole("region", { name: "ملاحظات التصنيف" })).not.toBeInTheDocument();
  });

  it("explains why there is no draft only for reason codes the system really emits", async () => {
    currentMockRole = "REVIEWER";
    items["9001"].generation_reason_code = "factual_question_without_grounding";
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "المسند لي" });
    await user.click(screen.getByRole("button", { name: "الكل" }));
    await beginReviewOn(user, "9001");
    expect(await screen.findByText(/يطلب التعليق معلومة لغوية ولا يتوفر مصدر رسمي معتمد/)).toBeInTheDocument();
  });

  it("admin dashboard states reviewed-of-total, not 'finished processing'", async () => {
    excelActiveSession = { session_id: "s-1", original_filename: "زوار.xlsx", source_file: "uploads/زوار.xlsx", status: "ready", created_at: "2026-08-30T10:00:00Z", updated_at: "2026-08-30T10:00:00Z", counts: { total: 39, excluded: 0, duplicate_content: 0, canonical_total: 39, completed: 3, remaining: 36 }, latest_job_id: null };
    renderReview();
    expect(await screen.findByText(/جاهز للإسناد · تمت مراجعة ٣ من ٣٩ تعليقًا/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("منتهية المعالجة");
  });

  it("settings: help rows say honestly that no official link is configured, and never invent one", async () => {
    const user = userEvent.setup();
    renderReview();
    await screen.findByRole("heading", { name: "الرئيسية" });
    await user.click(screen.getByRole("button", { name: "الحساب" }));
    await user.click(screen.getByRole("button", { name: "الإعدادات" }));
    const help = (await screen.findByRole("heading", { name: "المساعدة والمعلومات" })).closest("section") as HTMLElement;
    expect(within(help).getAllByText("غير متاح بعد — لم يُعتمد رابط رسمي")).toHaveLength(3);
    expect(within(help).queryAllByRole("link")).toHaveLength(0);
    expect(help.textContent).not.toContain("غير متصل");
  });

  it("settings: a configured https link renders as a safe external link; unsafe values are ignored", async () => {
    vi.stubEnv("VITE_PRIVACY_URL", "https://example.rcs-demo.net/privacy");
    vi.stubEnv("VITE_TERMS_URL", "javascript:alert(1)");
    vi.stubEnv("VITE_SUPPORT_URL", "http://insecure.example");
    try {
      const user = userEvent.setup();
      renderReview();
      await screen.findByRole("heading", { name: "الرئيسية" });
      await user.click(screen.getByRole("button", { name: "الحساب" }));
      await user.click(screen.getByRole("button", { name: "الإعدادات" }));
      const help = (await screen.findByRole("heading", { name: "المساعدة والمعلومات" })).closest("section") as HTMLElement;
      const links = within(help).getAllByRole("link");
      expect(links).toHaveLength(1);
      expect(links[0]).toHaveAttribute("href", "https://example.rcs-demo.net/privacy");
      expect(links[0]).toHaveAttribute("rel", "noopener noreferrer");
      expect(links[0]).toHaveAttribute("target", "_blank");
      expect(within(help).getAllByText("غير متاح بعد — لم يُعتمد رابط رسمي")).toHaveLength(2);
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("configuredHelpUrl accepts only https (and mailto for support)", async () => {
    const { configuredHelpUrl } = await import("./pages/SettingsPage");
    expect(configuredHelpUrl("https://a.rcs-demo.net/x", false)).toBe("https://a.rcs-demo.net/x");
    expect(configuredHelpUrl("  https://a.rcs-demo.net  ", false)).toBe("https://a.rcs-demo.net/");
    for (const bad of ["", undefined, null, "http://a.rcs-demo.net", "javascript:alert(1)", "data:text/html,x", "//a.rcs-demo.net", "not a url", "mailto:help@a.rcs-demo.net"]) expect(configuredHelpUrl(bad, false)).toBeNull();
    expect(configuredHelpUrl("mailto:help@a.rcs-demo.net", true)).toBe("mailto:help@a.rcs-demo.net");
    expect(configuredHelpUrl("mailto:bad", true)).toBeNull();
  });
});
