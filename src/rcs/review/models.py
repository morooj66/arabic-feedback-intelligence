"""
SQLAlchemy ORM model for the `review_items` table.

This is the single persistence table described in Section 2 of
DESIGN_SPEC.md (v2). Phase 2 (ingest + cleaning) only populates the
ingest-related columns and leaves every classification/review column
NULL with review_status="UNPROCESSED". Later phases fill in the rest —
the schema is defined in full now so no migration is needed as phases
land.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ReviewItem(Base):
    __tablename__ = "review_items"

    # --- identity -----------------------------------------------------
    comment_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    source_row_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- original data (Phase 2) ---------------------------------------
    original_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_type: Mapped[str | None] = mapped_column(String, nullable=True)
    entry_id: Mapped[str | None] = mapped_column(String, nullable=True)
    lemma: Mapped[str | None] = mapped_column(String, nullable=True)
    visitor_name: Mapped[str | None] = mapped_column(String, nullable=True)
    visitor_email: Mapped[str | None] = mapped_column(String, nullable=True)
    platform: Mapped[str | None] = mapped_column(String, nullable=True)
    comment_created_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    # --- cleaning (Phase 2) ---------------------------------------------
    is_exact_duplicate_of: Mapped[str | None] = mapped_column(String, nullable=True)
    ingested_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    comment_age_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    age_group: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- prefilter (Phase 3) --------------------------------------------
    is_prefiltered_junk: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    prefilter_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- AI output, immutable once written (Phase 4) --------------------
    # Hierarchical taxonomy (MIGRATION_PLAN.md): primary_intent (communicative
    # intent) x topic_subtype (what it's about) replace the old flat
    # ai_category. ai_immediate_reply_draft/ai_suggested_final_reply columns
    # are kept (nullable) for a future Phase 6 to populate — Phase 4 no
    # longer writes them; classification and drafting are separate calls.
    ai_primary_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    ai_topic_subtype: Mapped[str | None] = mapped_column(String, nullable=True)
    ai_subcategory: Mapped[str | None] = mapped_column(String, nullable=True)
    ai_secondary_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    ai_routing_decision: Mapped[str | None] = mapped_column(String, nullable=True)
    ai_classification_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # analytics only, not a gate
    ai_immediate_reply_draft: Mapped[str | None] = mapped_column(Text, nullable=True)  # reserved for Phase 6
    ai_suggested_final_reply: Mapped[str | None] = mapped_column(Text, nullable=True)  # reserved for Phase 6

    # --- classification-tier provenance, immutable once written ----------
    # LOCAL_FIRST_ARCHITECTURE_PLAN.md Phase A: which classifier tier
    # (RULE_BASED / TFIDF_LOGREG / TFIDF_SVM / TRANSFORMER_AR /
    # GEMINI_FALLBACK) actually produced ai_primary_intent, and that tier's
    # own opinion on whether the result needs escalation/human review.
    # Distinct from review_status/routing_decision, which are business-rule
    # concepts — this is a technical/audit signal about the classifier
    # itself. Not yet populated by anything (classify_row/ClassificationOutcome
    # unchanged in Phase A) — reserved for the hybrid router wiring in a
    # later phase.
    ai_classification_source: Mapped[str | None] = mapped_column(String, nullable=True)
    ai_needs_review: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # --- working values, mutable by reviewer (Phase 5-7) -----------------
    primary_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    topic_subtype: Mapped[str | None] = mapped_column(String, nullable=True)
    subcategory: Mapped[str | None] = mapped_column(String, nullable=True)
    secondary_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    classification_source: Mapped[str | None] = mapped_column(String, nullable=True)
    needs_review: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    routing_decision: Mapped[str | None] = mapped_column(String, nullable=True)
    immediate_reply_style: Mapped[str | None] = mapped_column(String, nullable=True)
    immediate_reply_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_final_reply: Mapped[str | None] = mapped_column(Text, nullable=True)
    exclusion_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- review workflow (Phase 7) ---------------------------------------
    review_status: Mapped[str] = mapped_column(String, nullable=False, default="UNPROCESSED")
    reviewer_action: Mapped[str | None] = mapped_column(String, nullable=True)  # 6 values, see settings.REVIEWER_ACTIONS
    reviewer_edit: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_reply: Mapped[str | None] = mapped_column(Text, nullable=True)
    ready_to_send: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reviewer_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- model / experiment traceability (Phase 4, Correction 5) ---------
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String, nullable=True)
    inference_run_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- timestamps --------------------------------------------------------
    processed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    reviewed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- sentiment (additive; classification may leave null) --------------
    ai_sentiment: Mapped[str | None] = mapped_column(String, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- response generation (immutable suggestion; working copy in suggested_final_reply)
    generation_routing_decision: Mapped[str | None] = mapped_column(String, nullable=True)
    generation_status: Mapped[str | None] = mapped_column(String, nullable=True)
    generation_reason_code: Mapped[str | None] = mapped_column(String, nullable=True)
    generation_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    generation_model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    generation_prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    generated_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    # --- reviewer claim / identity (provider-agnostic) -------------------
    reviewer_display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewer_started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    no_response_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- outbound (dictionary-platform connector later; never called on approve) -------
    outbound_status: Mapped[str | None] = mapped_column(String, nullable=True)
    outbound_attempted_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    outbound_sent_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    outbound_external_id: Mapped[str | None] = mapped_column(String, nullable=True)
    outbound_error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    outbound_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    outbound_idempotency_key: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- queue / concurrency ---------------------------------------------
    review_priority: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    source_reference: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- exact-content deduplication (Excel operational flow) -------------
    # content_dedupe_key: sha256 of the stripped original_comment text.
    # NULL for blank/near-empty content (nothing meaningful to dedupe).
    # dedupe_canonical_id: comment_id of the canonical representative for
    # this exact-content group. Equal to this row's own comment_id when
    # this row IS the canonical (including "group of one" rows). NULL
    # until dedup has run for this row, and always NULL for EXCLUDED rows
    # (prefiltered-junk rows never become or reference a canonical).
    # See rcs.review.excel_review_flow.run_content_deduplication.
    content_dedupe_key: Mapped[str | None] = mapped_column(String, nullable=True)
    dedupe_canonical_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- Phase 6: authenticated reviewer identity (NEW decisions only) ----
    # Nullable and never backfilled: every row created before Phase 6 has
    # NULL here and keeps its historical reviewer_id/reviewer_display_name
    # (plain strings, unauthenticated) exactly as before — this column is
    # additive, not a replacement. Only actions taken through the Phase 6
    # authenticated session ever set it, to users.user_id.
    reviewer_user_id: Mapped[str | None] = mapped_column(String, nullable=True)


class User(Base):
    """Phase 6 — app-owned account: the backend-authoritative source of
    truth for identity and role. Never created via public sign-up —
    accounts are provisioned only through scripts/manage_users.py."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    # "REVIEWER" | "ADMIN" — enforced in application code (rcs.review.auth);
    # a CHECK constraint would need per-dialect DDL, so validation happens
    # at every write path (manage_users.py, and nowhere else — there is no
    # HTTP endpoint that lets anyone set their own role).
    role: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class UserSession(Base):
    """Phase 6 — server-side session store. The cookie carries only an
    opaque, high-entropy token (rcs.review.auth.new_session_token — never
    derived from user data, so it cannot be forged); this table is the
    only place that maps a token to a user and an expiry. Logout deletes
    the row here, which is what makes logout an actual invalidation
    rather than "wait for a signed cookie to expire on its own"."""

    __tablename__ = "user_sessions"

    session_token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class PasswordResetToken(Base):
    """Self-service password reset. Only the SHA-256 HASH of the emailed token is stored (the raw token exists only in the email), so a database read
    cannot be turned into a working reset link. Single use (`used_at`) and short-lived (`expires_at`, at most 30 minutes after `created_at`). New table: created
    by Base.metadata.create_all and by migration b3d9f0a1c7e2; no existing table is altered."""

    __tablename__ = "password_reset_tokens"
    __table_args__ = (Index("ix_password_reset_tokens_user_created", "user_id", "created_at"),)

    token_hash: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class ReviewSession(Base):
    """Phase 4 — "which uploaded workbook is the reviewer currently
    working on". Deliberately thin: identity + status only. Total/
    completed/remaining counts are NEVER stored here — always computed
    live from review_items (filtered by source_file), so there is a
    single source of truth and no cache-invalidation class of bug.
    New table (created automatically by Base.metadata.create_all in
    init_db — no ALTER TABLE needed since review_items is untouched).
    """

    __tablename__ = "review_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    source_file: Mapped[str] = mapped_column(String, nullable=False)  # matches ReviewItem.source_file exactly
    upload_token: Mapped[str] = mapped_column(String, nullable=False)
    # "uploaded" -> "processing" -> "ready" -> "completed" (or "failed")
    status: Mapped[str] = mapped_column(String, nullable=False, default="uploaded")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class ProcessingJob(Base):
    """Phase 4 — progress record for one background processing run of a
    ReviewSession. A lightweight persisted job, not an external queue:
    run by a single in-process background thread (see
    rcs.review.job_runner), polled via GET /api/excel/jobs/{job_id}.
    New table — no ALTER TABLE needed.
    """

    __tablename__ = "processing_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    # "running" -> "succeeded" | "failed"
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    current_stage: Mapped[str | None] = mapped_column(String, nullable=True)
    rows_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentExecutionTrace(Base):
    """Phase M5E — one row per `rcs.response_intelligence.orchestrator.
    run_agent_pipeline` execution. Sanitized audit/observability data
    ONLY: outcome codes, counts, and booleans — see
    `rcs.response_intelligence.audit.AgentAuditRecord` (the pure,
    storage-agnostic shape this table mirrors) for the enforced
    guarantee that no prompt, comment, retrieved content, or raw
    exception text is ever persisted here.

    Append-only from the application's perspective: `execution_id` is
    the primary key and is minted fresh (uuid4 hex) once per execution,
    so one `review_item_id` may have many rows here (one per genuine
    generation attempt) and no row is ever updated after insertion.
    `review_item_id` is a plain reference column, not an enforced
    foreign key — matching this codebase's existing convention for
    `ProcessingJob.session_id`."""

    __tablename__ = "agent_execution_traces"
    __table_args__ = (
        Index("ix_agent_execution_traces_review_item_created", "review_item_id", "created_at"),
    )

    execution_id: Mapped[str] = mapped_column(String, primary_key=True)
    review_item_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)

    grounding_mode: Mapped[str] = mapped_column(String, nullable=False)

    official_retrieval_outcome: Mapped[str] = mapped_column(String, nullable=False)
    official_result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    official_grounding_available: Mapped[bool] = mapped_column(Boolean, nullable=False)

    style_retrieval_outcome: Mapped[str] = mapped_column(String, nullable=False)
    style_result_count: Mapped[int] = mapped_column(Integer, nullable=False)

    logical_generation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    refinement_attempted: Mapped[bool] = mapped_column(Boolean, nullable=False)

    initial_validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    initial_failure_code: Mapped[str | None] = mapped_column(String, nullable=True)
    final_validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    final_failure_code: Mapped[str | None] = mapped_column(String, nullable=True)

    generation_status: Mapped[str] = mapped_column(String, nullable=False)


class ReviewItemEvent(Base):
    """Append-only audit log of reviewer/admin actions on a review item
    (see `rcs.review.audit_events`). Rows are inserted in the SAME
    transaction as the state change they describe and are never updated
    or deleted by the application. Holds identifiers, enumerated codes
    and timestamps only -- no visitor text, reply text or free-form
    reasons. `review_item_id` is a plain reference column, not an
    enforced foreign key, matching `AgentExecutionTrace`."""

    __tablename__ = "review_item_events"
    __table_args__ = (
        Index("ix_review_item_events_item_created", "review_item_id", "created_at"),
        Index("ix_review_item_events_actor_created", "actor_user_id", "created_at"),
    )

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    review_item_id: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    actor_user_id: Mapped[str] = mapped_column(String, nullable=False)
    actor_role: Mapped[str | None] = mapped_column(String, nullable=True)
    from_status: Mapped[str | None] = mapped_column(String, nullable=True)
    to_status: Mapped[str | None] = mapped_column(String, nullable=True)
    from_assignee_user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    to_assignee_user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    overrode_recommendation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class DirectMessage(Base):
    """Internal admin <-> reviewer text message. A conversation is identified by the REVIEWER's user id (one conversation per reviewer, shared by all admins).
    `sender_role` is the sender's server-verified role; `read_at` is set when the OTHER side opened the conversation. (sender_user_id, client_message_id) is
    unique, so a retried send (same client-generated id) cannot create a second row. Timestamps are naive UTC, like the auth tables. New table: created by
    Base.metadata.create_all and by migration c4e8a2d17f35; no existing table is altered."""

    __tablename__ = "direct_messages"
    __table_args__ = (
        UniqueConstraint("sender_user_id", "client_message_id", name="uq_direct_messages_sender_client_id"),
        Index("ix_direct_messages_conversation_created", "reviewer_user_id", "created_at"),
    )

    message_id: Mapped[str] = mapped_column(String, primary_key=True)
    reviewer_user_id: Mapped[str] = mapped_column(String, nullable=False)
    sender_user_id: Mapped[str] = mapped_column(String, nullable=False)
    sender_role: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    client_message_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    read_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class AccountRequest(Base):
    """A self-service request for an account. It is NOT a login: the person cannot authenticate until an ADMIN approves it, and approval always creates a
    REVIEWER (the requester never supplies or influences the role; ADMIN accounts are provisioned only through scripts/manage_users.py). The password the
    requester chose is stored only as a bcrypt hash and becomes the account's hash on approval. `email` is unique, so a repeated request never creates a
    second row. New table: created by Base.metadata.create_all and by migration d5f1b3c8a2e6; no existing table is altered."""

    __tablename__ = "account_requests"
    __table_args__ = (Index("ix_account_requests_status_created", "status", "created_at"),)

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)  # PENDING | APPROVED | REJECTED
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    decided_by_user_id: Mapped[str | None] = mapped_column(String, nullable=True)
