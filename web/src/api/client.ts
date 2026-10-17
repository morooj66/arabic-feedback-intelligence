import type {
  AdminUser,
  AnalyticsRange,
  AnalyticsSnapshot,
  AssignableItemsPage,
  AssignByFilterResult,
  CompletedExportSummary,
  AssignResult,
  ExcelJob,
  ExcelProcessingSummary,
  ExcelSession,
  ExcelUploadResult,
  HealthStatus,
  QueuePage,
  ReviewDetail,
  ReviewerSession,
  ReviewerSummaryRow,
  SourceFileSummary,
  WorkflowCounts,
} from "./types";
import type { ApiError, DirectMessage, PendingAccountRequest, FactAttestation, MessageConversation, MessageConversationSummary, ReplyValidation } from "./types";

// Phase 7: empty (relative /api/... paths, today's dev behavior — the
// Vite dev server proxies them to the local backend) unless the
// production build sets VITE_API_BASE_URL, needed once frontend and
// backend are on different origins (Vercel + Render). No trailing slash.
export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const err: ApiError = new Error(payload.detail || payload.error || "request_failed");
    err.status = response.status;
    err.code = payload.error;
    err.payload = payload;
    throw err;
  }
  return payload as T;
}

// A real full-page navigation target (never fetched) — the browser has
// to follow the actual redirect chain through Microsoft's own login
// page, which no fetch/XHR call can do.
export const microsoftSignInUrl = apiUrl("/api/auth/microsoft/start");

export const api = {
  // Self-service account request (public) and its admin approval. The requester never supplies a role; approval always creates a REVIEWER.
  requestAccount: (email: string, displayName: string, password: string) =>
    request<{ ok: boolean }>("/api/account-requests", { method: "POST", body: JSON.stringify({ email, display_name: displayName, password }) }),
  accountRequests: () => request<{ requests: PendingAccountRequest[] }>("/api/admin/account-requests"),
  approveAccountRequest: (requestId: string) =>
    request<{ request_id: string; user_id: string; email: string; role: string }>("/api/admin/account-requests/approve", { method: "POST", body: JSON.stringify({ request_id: requestId }) }),
  rejectAccountRequest: (requestId: string) =>
    request<{ request_id: string; status: string }>("/api/admin/account-requests/reject", { method: "POST", body: JSON.stringify({ request_id: requestId }) }),
  // Internal admin <-> reviewer messaging. `reviewerUserId` is required for an ADMIN and ignored (own conversation) for a REVIEWER.
  messagesConversations: () => request<{ conversations: MessageConversationSummary[] }>("/api/messages/conversations"),
  messagesConversation: (reviewerUserId?: string) =>
    request<MessageConversation>(`/api/messages/conversation${reviewerUserId ? `?reviewer_user_id=${encodeURIComponent(reviewerUserId)}` : ""}`),
  messagesSend: (body: string, clientMessageId: string, reviewerUserId?: string) =>
    request<DirectMessage>("/api/messages/send", { method: "POST", body: JSON.stringify({ body, client_message_id: clientMessageId, reviewer_user_id: reviewerUserId }) }),
  messagesMarkRead: (reviewerUserId?: string) =>
    request<{ reviewer_user_id: string; marked: number }>("/api/messages/read", { method: "POST", body: JSON.stringify({ reviewer_user_id: reviewerUserId }) }),
  // Phase DEV-INSPECT: admin-only Agent Inspector / Test Console. Every
  // call is read-only with respect to review_items — see
  // rcs.response_intelligence.inspector's module docstring.
  inspectorArchitecture: () => request<Record<string, unknown>>("/api/admin/inspector/architecture"),
  inspectorConfig: () => request<Record<string, unknown>>("/api/admin/inspector/config"),
  inspectorClassification: (comment_text: string) =>
    request<Record<string, unknown>>("/api/admin/inspector/classification", {
      method: "POST",
      body: JSON.stringify({ comment_text }),
    }),
  inspectorEligibility: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>("/api/admin/inspector/eligibility", { method: "POST", body: JSON.stringify(body) }),
  inspectorRetrieval: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>("/api/admin/inspector/retrieval", { method: "POST", body: JSON.stringify(body) }),
  inspectorGrounding: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>("/api/admin/inspector/grounding", { method: "POST", body: JSON.stringify(body) }),
  inspectorValidation: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>("/api/admin/inspector/validation", { method: "POST", body: JSON.stringify(body) }),
  inspectorFullRun: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>("/api/admin/inspector/full-run", { method: "POST", body: JSON.stringify(body) }),
  // Internal AI Evaluation Playground (admin-only, disabled by default on the
  // server; see rcs.response_intelligence.playground). Nothing here is saved
  // unless playgroundSaveCase is called explicitly.
  playgroundStatus: () => request<Record<string, any>>("/api/admin/playground/status"),
  playgroundRun: (body: Record<string, unknown>) =>
    request<Record<string, any>>("/api/admin/playground/run", { method: "POST", body: JSON.stringify(body) }),
  playgroundCompare: (body: Record<string, unknown>) =>
    request<Record<string, any>>("/api/admin/playground/compare", { method: "POST", body: JSON.stringify(body) }),
  playgroundSaveCase: (body: Record<string, unknown>) =>
    request<Record<string, any>>("/api/admin/playground/cases", { method: "POST", body: JSON.stringify(body) }),
  playgroundCases: () => request<Record<string, any>>("/api/admin/playground/cases"),
  playgroundDeleteCase: (case_id: string) =>
    request<Record<string, any>>("/api/admin/playground/cases/delete", { method: "POST", body: JSON.stringify({ case_id }) }),
  playgroundMetrics: () => request<Record<string, any>>("/api/admin/playground/metrics"),
  session: () => request<ReviewerSession>("/api/session"),
  health: () => request<HealthStatus>("/api/health"),
  login: (body: { email: string; password: string }) =>
    request<ReviewerSession>("/api/login", { method: "POST", body: JSON.stringify(body) }),
  logout: () => request<{ ok: boolean }>("/api/logout", { method: "POST", body: "{}" }),
  // Public password-reset endpoints (no session). The request answer is identical for every address by design.
  requestPasswordReset: (email: string) =>
    request<{ ok: boolean }>("/api/password-reset/request", { method: "POST", body: JSON.stringify({ email }) }),
  confirmPasswordReset: (body: { token: string; password: string }) =>
    request<{ ok: boolean }>("/api/password-reset/confirm", { method: "POST", body: JSON.stringify(body) }),
  adminUsers: () => request<{ users: AdminUser[] }>("/api/admin/users"),
  adminReviewers: () => request<{ reviewers: ReviewerSummaryRow[] }>("/api/admin/reviewers"),
  adminSourceSummary: () => request<{ sources: SourceFileSummary[] }>("/api/admin/source-summary"),
  adminAssignableItems: (params: {
    unassignedOnly?: boolean;
    sourceFile?: string;
    offset: number;
    limit: number;
    primaryIntents?: string[];
    topicSubtypes?: string[];
    search?: string;
    sentiments?: string[];
    priorities?: string[];
    createdFrom?: string;
    createdTo?: string;
    // Currently assigned to this reviewer (cannot be combined with unassignedOnly).
    reviewerUserId?: string;
    statuses?: string[];
    sort?: string;
  }) => {
    const qs = new URLSearchParams({ offset: String(params.offset), limit: String(params.limit) });
    if (params.sentiments?.length) qs.set("sentiment", params.sentiments.join(","));
    if (params.priorities?.length) qs.set("priority", params.priorities.join(","));
    if (params.createdFrom) qs.set("created_from", params.createdFrom);
    if (params.createdTo) qs.set("created_to", params.createdTo);
    if (params.reviewerUserId) qs.set("reviewer_user_id", params.reviewerUserId);
    if (params.statuses?.length) qs.set("review_status", params.statuses.join(","));
    if (params.sort) qs.set("sort", params.sort);
    if (params.unassignedOnly) qs.set("unassigned_only", "true");
    if (params.sourceFile) qs.set("source_file", params.sourceFile);
    if (params.primaryIntents?.length) qs.set("primary_intent", params.primaryIntents.join(","));
    if (params.topicSubtypes?.length) qs.set("topic_subtype", params.topicSubtypes.join(","));
    if (params.search) qs.set("search", params.search);
    return request<AssignableItemsPage>(`/api/admin/assignable-items?${qs.toString()}`);
  },
  adminAssign: (commentIds: string[], reviewerUserId: string) =>
    request<AssignResult>("/api/admin/assign", {
      method: "POST",
      body: JSON.stringify({ comment_ids: commentIds, reviewer_user_id: reviewerUserId }),
    }),
  // Server-side count/filter assignment across ALL matching unassigned comments (not just the visible page). Exactly one of `count` / `all`.
  adminAssignByFilter: (params: {
    reviewerUserId: string;
    count?: number;
    all?: boolean;
    sourceFile?: string;
    primaryIntents?: string[];
    topicSubtypes?: string[];
    search?: string;
    sentiments?: string[];
    priorities?: string[];
    createdFrom?: string;
    createdTo?: string;
    statuses?: string[];
    sort?: string;
  }) => {
    const filters: Record<string, string> = {};
    if (params.sentiments?.length) filters.sentiment = params.sentiments.join(",");
    if (params.priorities?.length) filters.priority = params.priorities.join(",");
    if (params.createdFrom) filters.created_from = params.createdFrom;
    if (params.createdTo) filters.created_to = params.createdTo;
    if (params.statuses?.length) filters.review_status = params.statuses.join(",");
    if (params.sort) filters.sort = params.sort;
    if (params.sourceFile) filters.source_file = params.sourceFile;
    if (params.primaryIntents?.length) filters.primary_intent = params.primaryIntents.join(",");
    if (params.topicSubtypes?.length) filters.topic_subtype = params.topicSubtypes.join(",");
    if (params.search) filters.search = params.search;
    return request<AssignByFilterResult>("/api/admin/assign-by-filter", {
      method: "POST",
      body: JSON.stringify({ reviewer_user_id: params.reviewerUserId, ...(params.all ? { all: true } : { count: params.count }), filters }),
    });
  },
  counts: () => request<WorkflowCounts>("/api/counts"),
  analytics: (params: { range: AnalyticsRange; start?: string; end?: string; sourceFile?: string }) => {
    const qs = new URLSearchParams({ range: params.range });
    if (params.start) qs.set("start", params.start);
    if (params.end) qs.set("end", params.end);
    if (params.sourceFile) qs.set("source_file", params.sourceFile);
    return request<AnalyticsSnapshot>(`/api/analytics?${qs.toString()}`);
  },
  queue: (params: {
    filter: string;
    sort: string;
    search: string;
    offset: number;
    limit: number;
    sourceFile?: string;
    primaryIntents?: string[];
    topicSubtypes?: string[];
    reviewerActions?: string[];
    sentiments?: string[];
    priorities?: string[];
    createdFrom?: string;
    createdTo?: string;
    // ADMIN-only: view one specific reviewer's items (e.g. Completed
    // Reviews filtered by reviewer). Silently ignored server-side for a
    // REVIEWER, whose own scope is always forced to their own id.
    reviewerUserId?: string;
  }) => {
    const qs = new URLSearchParams({
      filter: params.filter,
      sort: params.sort,
      search: params.search,
      offset: String(params.offset),
      limit: String(params.limit),
    });
    if (params.sourceFile) qs.set("source_file", params.sourceFile);
    if (params.primaryIntents?.length) qs.set("primary_intent", params.primaryIntents.join(","));
    if (params.topicSubtypes?.length) qs.set("topic_subtype", params.topicSubtypes.join(","));
    if (params.reviewerActions?.length) qs.set("reviewer_action", params.reviewerActions.join(","));
    if (params.sentiments?.length) qs.set("sentiment", params.sentiments.join(","));
    if (params.priorities?.length) qs.set("priority", params.priorities.join(","));
    if (params.createdFrom) qs.set("created_from", params.createdFrom);
    if (params.createdTo) qs.set("created_to", params.createdTo);
    if (params.reviewerUserId) qs.set("reviewer_user_id", params.reviewerUserId);
    return request<QueuePage>(`/api/queue?${qs.toString()}`);
  },
  item: (id: string) => request<ReviewDetail>(`/api/items/${encodeURIComponent(id)}`),
  claim: (id: string, expected_version: number) =>
    request<ReviewDetail>(`/api/items/${encodeURIComponent(id)}/claim`, {
      method: "POST",
      body: JSON.stringify({ expected_version }),
    }),
  sendAsIs: (id: string, expected_version: number, attestation?: FactAttestation) =>
    request<ReviewDetail>(`/api/items/${encodeURIComponent(id)}/send-as-is`, {
      method: "POST",
      body: JSON.stringify(attestation ? { expected_version, attestation } : { expected_version }),
    }),
  editAndSend: (id: string, expected_version: number, edited_response: string, attestation?: FactAttestation) =>
    request<ReviewDetail>(`/api/items/${encodeURIComponent(id)}/edit-and-send`, {
      method: "POST",
      body: JSON.stringify(attestation ? { expected_version, edited_response, attestation } : { expected_version, edited_response }),
    }),
  validateReply: (id: string, reply: string) =>
    request<ReplyValidation>(`/api/items/${encodeURIComponent(id)}/validate-reply`, {
      method: "POST",
      body: JSON.stringify({ reply }),
    }),
  noResponse: (id: string, expected_version: number, reason: string) =>
    request<ReviewDetail>(`/api/items/${encodeURIComponent(id)}/no-response`, {
      method: "POST",
      body: JSON.stringify({ expected_version, reason }),
    }),
  uploadExcel: (file: File) =>
    request<ExcelUploadResult>("/api/excel/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    }),
  // XHR (not fetch) specifically because it's the only web API that
  // exposes real upload byte-progress events — fetch has no upload
  // progress hook. Used so the "جاري رفع الملف" state can show a real
  // percentage rather than an indeterminate spinner.
  uploadExcelWithProgress: (file: File, onProgress: (fraction: number) => void) =>
    new Promise<ExcelUploadResult>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", apiUrl("/api/excel/upload"));
      xhr.withCredentials = true;
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let payload: any = {};
        try {
          payload = JSON.parse(xhr.responseText || "{}");
        } catch {
          // ignore — payload stays {}
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress(1);
          resolve(payload as ExcelUploadResult);
        } else {
          const err: ApiError = new Error(payload.detail || payload.error || "upload_failed");
          err.status = xhr.status;
          err.code = payload.error;
          reject(err);
        }
      };
      xhr.onerror = () => {
        const err: ApiError = new Error("network_error");
        err.status = 0;
        reject(err);
      };
      xhr.send(file);
    }),
  processExcel: (uploadToken: string, limit?: number) =>
    request<ExcelProcessingSummary>("/api/excel/process", {
      method: "POST",
      body: JSON.stringify({ upload_token: uploadToken, ...(limit ? { limit } : {}) }),
    }),
  createExcelSession: (uploadToken: string) =>
    request<{ session: ExcelSession; job_id: string }>("/api/excel/sessions", {
      method: "POST",
      body: JSON.stringify({ upload_token: uploadToken }),
    }),
  activeExcelSession: () => request<{ active: ExcelSession | null }>("/api/excel/sessions/active"),
  getExcelSession: (sessionId: string) => request<ExcelSession>(`/api/excel/sessions/${encodeURIComponent(sessionId)}`),
  resumeExcelSession: (sessionId: string) =>
    request<{ job_id: string }>(`/api/excel/sessions/${encodeURIComponent(sessionId)}/process`, {
      method: "POST",
      body: "{}",
    }),
  getExcelJob: (jobId: string) => request<ExcelJob>(`/api/excel/jobs/${encodeURIComponent(jobId)}`),
  // `commentIds`, when given, exports exactly that explicit set instead
  // of a whole source_file (e.g. "export my current selection/filtered
  // result"). The backend caps this at 200 ids per request (a GET
  // query string has a practical length ceiling) — callers with a
  // larger filtered set should export by source_file instead, or in
  // batches; this client does not silently truncate.
  // Completed-only export, selected on the server across ALL matching persisted decisions (not the displayed page). Exactly one of `count` / `all`.
  downloadCompletedExport: async (params: {
    count?: number;
    all?: boolean;
    search?: string;
    reviewerUserId?: string;
    primaryIntents?: string[];
    topicSubtypes?: string[];
    reviewerActions?: string[];
    sentiments?: string[];
    priorities?: string[];
    createdFrom?: string;
    createdTo?: string;
    sourceFile?: string;
  }): Promise<CompletedExportSummary> => {
    const qs = new URLSearchParams();
    if (params.all) qs.set("all", "1"); else qs.set("count", String(params.count));
    if (params.search) qs.set("search", params.search);
    if (params.reviewerUserId) qs.set("reviewer_user_id", params.reviewerUserId);
    if (params.primaryIntents?.length) qs.set("primary_intent", params.primaryIntents.join(","));
    if (params.topicSubtypes?.length) qs.set("topic_subtype", params.topicSubtypes.join(","));
    if (params.reviewerActions?.length) qs.set("reviewer_action", params.reviewerActions.join(","));
    if (params.sentiments?.length) qs.set("sentiment", params.sentiments.join(","));
    if (params.priorities?.length) qs.set("priority", params.priorities.join(","));
    if (params.createdFrom) qs.set("created_from", params.createdFrom);
    if (params.createdTo) qs.set("created_to", params.createdTo);
    if (params.sourceFile) qs.set("source_file", params.sourceFile);
    const response = await fetch(apiUrl(`/api/excel/export/completed?${qs.toString()}`), { credentials: "include" });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const err: ApiError = new Error(payload.detail || payload.error || "export_failed");
      err.status = response.status;
      err.code = payload.error;
      throw err;
    }
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : "completed_reviews.xlsx";
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    try {
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } finally {
      URL.revokeObjectURL(url);
    }
    const num = (name: string) => Number(response.headers.get(name) || 0);
    return { rows: num("X-Export-Rows"), matched: num("X-Export-Matched"), shortfall: num("X-Export-Shortfall") };
  },
  downloadExcelExport: async (sourceFile?: string, commentIds?: string[]): Promise<void> => {
    const qs = commentIds?.length
      ? `?comment_ids=${encodeURIComponent(commentIds.join(","))}`
      : sourceFile
      ? `?source_file=${encodeURIComponent(sourceFile)}`
      : "";
    const response = await fetch(apiUrl(`/api/excel/export${qs}`), { credentials: "include" });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const err: ApiError = new Error(payload.detail || payload.error || "export_failed");
      err.status = response.status;
      err.code = payload.error;
      throw err;
    }
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : "reviewed_responses.xlsx";
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    try {
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } finally {
      URL.revokeObjectURL(url);
    }
  },
};
