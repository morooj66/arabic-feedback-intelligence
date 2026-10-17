export type Role = "REVIEWER" | "ADMIN";

export type ReviewerSession = {
  user_id: string;
  display_name: string;
  role: Role;
  auth_mode: string;
  email?: string | null;
};

export type AdminUser = {
  user_id: string;
  email: string;
  display_name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
};

export type QueueItem = {
  comment_id: string;
  review_status: string;
  comment_preview: string | null;
  primary_intent: string | null;
  topic_subtype: string | null;
  review_priority: number | null;
  priority_reason: string | null;
  claimed_by: string | null;
  comment_created_at: string | null;
  ingested_at: string | null;
  suggested_response: string | null;
  // Present for a decided item (Completed Reviews); null for a still-
  // pending item.
  reviewer_action: string | null;
  reviewer_display_name: string | null;
  reviewed_at: string | null;
};

export type QueuePage = {
  items: QueueItem[];
  total: number;
  offset: number;
  limit: number;
};

export type ReviewDetail = {
  comment_id: string;
  source_reference: string | null;
  original_comment: string | null;
  comment_created_at: string | null;
  // Non-PII entry-identifying context — never visitor_name/visitor_email,
  // which never leave the backend.
  original_title: string | null;
  entry_id: string | null;
  platform: string | null;
  original_type: string | null;
  primary_intent: string | null;
  topic_subtype: string | null;
  sentiment: string | null;
  confidence: number | null;
  needs_review: boolean | null;
  generation_routing_decision: string | null;
  generation_status: string | null;
  generation_reason_code: string | null;
  suggested_response: string | null;
  review_status: string;
  reviewer_id: string | null;
  reviewer_display_name: string | null;
  reviewer_action: string | null;
  reviewed_at: string | null;
  edited_response: string | null;
  final_response: string | null;
  no_response_reason: string | null;
  ready_to_send: boolean;
  outbound_status: string | null;
  outbound_error_code: string | null;
  review_priority: number | null;
  priority_reason: string | null;
  row_version: number;
  allowed_actions: string[];
  // Contextual-signal conflicts stored with the classification ("dimension:code"); display-only, absent/empty when none.
  context_conflicts?: string[];
};

export type WorkflowCounts = {
  all: number;
  pending: number;
  approved: number;
  no_reply: number;
  send_failed: number;
  by_status: Record<string, number>;
};

export type AnalyticsRange = "today" | "week" | "month" | "custom";

export type AnalyticsSnapshot = {
  range: string;
  range_start: string | null;
  range_end: string;
  kpis: {
    incoming: number;
    reviewed: number;
    pending: number;
    urgent_pending: number;
    overdue_pending: number;
    no_reply: number;
    avg_review_seconds: number | null;
    avg_review_sample: number;
  };
  today: {
    reviewed_all: number;
    reviewed_by_me: number;
    actions: Record<string, number>;
  };
  outcomes: Record<string, number>;
  queue_health: {
    under_day: number;
    one_two: number;
    three_seven: number;
    over_seven: number;
    dated: number;
    oldest_pending_age_seconds: number | null;
  };
  quality: {
    sample: number;
    send_as_is_rate: number | null;
    edited_rate: number | null;
    no_reply_rate: number | null;
  };
  classification: {
    intent: { key: string; count: number }[];
    topic: { key: string; count: number }[];
    sentiment: { key: string; count: number }[];
  };
  data_quality?: {
    unmapped_classification: {
      intent: { key: string; count: number }[];
      topic: { key: string; count: number }[];
      sentiment: { key: string; count: number }[];
    };
    note: string | null;
  };
  trend: { bucket: string; incoming: number; reviewed: number }[];
  availability: Record<string, string>;
};

export type ApiError = Error & { status?: number; code?: string; payload?: Record<string, unknown> };

/** Reviewer confirmation of factual claims that cannot be verified against approved grounding (server-enforced). */
export type FactAttestation = { confirmed: boolean; finding_codes: string[] };

/** Result of POST /api/items/:id/validate-reply — read-only; nothing is stored or sent. */
export type ReplyValidation = {
  valid: boolean;
  blocking_findings: string[];
  attestable_findings: string[];
  attestation_required: boolean;
  attestation_permitted: boolean;
};

export type HealthStatus = {
  ok: boolean;
  demo_auth: boolean;
  // Whether Microsoft Entra SSO is configured on this server — the
  // frontend uses this only to decide whether to show the Microsoft
  // sign-in button; it is never trusted for authorization.
  entra: boolean;
  // Whether self-service password reset is usable on this server (an email provider is configured). Absent/false -> the login screen keeps the
  // "contact your administrator" message. Never trusted for authorization.
  password_reset?: boolean;
  // Whether the public "new account request" form is available (the account stays inactive until an ADMIN approves it).
  account_requests?: boolean;
  // True only on a synthetic-data demo instance (rcs.demo); shows a one-line notice. Never trusted for authorization.
  demo?: boolean;
};

export type ExcelUploadResult = {
  upload_token: string;
  filename: string;
  size_bytes: number;
  valid: boolean;
  total_rows: number;
  errors: string[];
};

export type ExcelProcessingSummary = {
  file_name: string;
  total_rows: number;
  valid_rows: number;
  inserted_rows: number;
  duplicate_rows: number;
  excluded_rows: number;
  classification_processed: number;
  generated_drafts: number;
  no_response_recommended: number;
  review_only: number;
  failed_rows: number;
  output_available: boolean;
};

export type ExcelSessionCounts = {
  total: number;
  excluded: number;
  duplicate_content: number;
  canonical_total: number;
  completed: number;
  remaining: number;
};

export type ExcelSessionStatus = "uploaded" | "processing" | "ready" | "completed";

export type ExcelSession = {
  session_id: string;
  original_filename: string;
  source_file: string;
  status: ExcelSessionStatus;
  created_at: string;
  updated_at: string;
  counts: ExcelSessionCounts;
  latest_job_id: string | null;
};

// --- Phase 8 — admin assignment (reviewer isolation foundation) ------------

export type ReviewerSummaryRow = {
  user_id: string;
  email: string;
  display_name: string;
  assigned_total: number;
  pending: number;
  completed: number;
};

export type SourceFileSummary = {
  source_file: string;
  total_reviewable: number;
  unassigned: number;
  assigned: number;
  completed: number;
};

export type AssignableItem = {
  comment_id: string;
  comment_preview: string | null;
  source_file: string;
  review_status: string;
  routing_decision: string | null;
  reviewer_user_id: string | null;
  primary_intent: string | null;
  topic_subtype: string | null;
  review_priority: number | null;
  ingested_at: string | null;
};

export type AssignableItemsPage = {
  items: AssignableItem[];
  total: number;
  offset: number;
  limit: number;
};

export type AssignResult = {
  assigned: string[];
  skipped_already_decided: string[];
  skipped_not_reviewable: string[];
  not_found: string[];
};

export type AssignByFilterResult = {
  requested: number | null;      // null = "all matching"
  assigned: number;
  shortfall: number;             // requested - assigned, never negative
  matched_before: number;
  remaining_eligible: number;
};

export type CompletedExportSummary = {
  rows: number;        // completed reviews written to the file
  matched: number;     // completed reviews matching the filters on the server
  shortfall: number;   // requested - rows (0 when everything was exported)
};

export type ExcelJobStatus = "running" | "succeeded" | "failed";

export type ExcelJob = {
  job_id: string;
  session_id: string;
  status: ExcelJobStatus;
  current_stage: string | null;
  rows_done: number;
  rows_total: number | null;
  started_at: string;
  updated_at: string;
  finished_at: string | null;
  error_message: string | null;
};

export type DirectMessage = {
  message_id: string;
  reviewer_user_id: string;
  sender_user_id: string;
  sender_role: "ADMIN" | "REVIEWER";
  sender_name: string;
  body: string;
  created_at: string;
  read_at: string | null;
  duplicate?: boolean;
};

export type MessageConversationSummary = {
  reviewer_user_id: string;
  reviewer_name: string;
  last_message: { body: string; sender_role: "ADMIN" | "REVIEWER"; created_at: string } | null;
  unread: number;
};

export type MessageConversation = {
  reviewer_user_id: string;
  reviewer_name: string;
  messages: DirectMessage[];
  unread: number;
};

export type PendingAccountRequest = { request_id: string; email: string; display_name: string; created_at: string };
