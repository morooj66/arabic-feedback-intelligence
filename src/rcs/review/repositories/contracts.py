"""
Phase M4A — repository contracts (Protocols), extracted 1:1 from the
persistence operations the review workflow already performs today.

Deliberately NOT a generic repository/query-builder framework: every
method here mirrors one specific business operation this codebase
already has a name for (claim, send_as_is, assign a batch, compute
session counts, ...) with the exact parameters/return shape the current
SQLAlchemy code already produces. There is no generic
get/filter/query() method anywhere in this file.

Phase M4D.1 extracted the `users` table (UserRepository); Phase M4D.2
extracted `user_sessions` (UserSessionRepository below) --
role/authorization semantics and password hashing stay exactly where
they are (rcs.review.auth). Microsoft Entra (rcs.review.entra_auth) was
originally, deliberately left out of this contract -- documented at the
time as "PostgreSQL remains the sole, unabstracted authority for the
Entra provider flow". The production login/DB boundary audit closed
that exception: rcs.review.entra_auth now resolves its local-user lookup
through UserRepository.get_by_email via the same
rcs.review.persistence.resolve_bundle seam every other auth path uses,
so it needs no separate contract entry of its own.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from rcs.response_generation.schema import GenerationResult
from rcs.review.audit_events import ReviewEvent
from rcs.review.models import ProcessingJob, ReviewItem, ReviewSession, User, UserSession

# Exactly the four ORDER BY outcomes service.py's list_review_queue has
# always produced (see the original mode/newest_first branching) --
# resolved to one unambiguous literal by the caller (pure logic, no SQL)
# before it ever reaches a repository.
QueueSort = str  # one of: "newest", "oldest", "priority_newest", "priority_oldest"


@dataclass(frozen=True)
class FilterAssignmentResult:
    """Outcome of one server-side count/filter assignment. `matched_before` = comments eligible for NEW assignment (unassigned, undecided, reviewable)
    that matched the filters when the call started; `requested` is None for "all matching"; `shortfall` = requested - assigned (never negative)."""

    matched_before: int
    requested: int | None
    assigned: int
    shortfall: int
    remaining_eligible: int


@dataclass(frozen=True)
class AssignmentResult:
    """Outcome of one admin assign/reassign batch call. Moved here
    unchanged from service.py -- this is the repository's output shape,
    not a service-layer concept, since both today's Postgres batch and
    any future Mongo transactional batch must produce exactly this."""

    assigned: tuple[str, ...]
    skipped_already_decided: tuple[str, ...]
    skipped_not_reviewable: tuple[str, ...]
    not_found: tuple[str, ...]


@dataclass(frozen=True)
class SessionCounts:
    """Moved unchanged from sessions.py — no caller ever imported this
    type by name (only the functions that return it), so relocating it
    here is a zero-risk move, not a shape change."""

    total: int
    excluded: int
    duplicate_content: int
    canonical_total: int
    completed: int
    remaining: int


@dataclass(frozen=True)
class ReviewerAssignmentCounts:
    """Persistence-decoupling milestone — the exact two numbers
    service.py::reviewer_summaries needs per reviewer: every row ever
    assigned to them, and how many of those have no reviewer_action yet.
    Replaces the inline `select(ReviewItem.reviewer_user_id, func.count())`
    aggregate that used to live directly in service.py."""

    total: int
    pending: int


@dataclass(frozen=True)
class SourceFileStatusFields:
    """Mongo runtime boundary — the exact per-row fields
    rcs.review.excel_review_flow._source_file_aggregate needs to compute
    its whole-file cumulative counters (processed/excluded/generated/
    no-response/needs-review/duplicate-content totals). A read shape,
    not a decision: the counting rule itself (which combination of
    review_status/generation_routing_decision/generation_status counts
    as which bucket) stays entirely in excel_review_flow.py, computed
    once from this shape regardless of backend."""

    review_status: str
    generation_routing_decision: str | None
    generation_status: str | None


@dataclass(frozen=True)
class DedupCandidate:
    """Persistence-decoupling milestone — the exact row shape
    rcs.review.excel_review_flow.run_content_deduplication's grouping/
    tiebreak business logic reads. Field-for-field the same columns the
    original inline `select(ReviewItem.comment_id, ...)` already
    selected; carrying the business logic itself (grouping, protected/
    decided tiebreak, blank-content singleton rule) stays entirely in
    excel_review_flow.py — this is a read shape, not a decision."""

    comment_id: str
    source_file: str
    source_row_index: int
    original_comment: str | None
    review_status: str
    reviewer_action: str | None
    dedupe_canonical_id: str | None


class ReviewItemRepository(Protocol):
    def get(self, comment_id: str) -> ReviewItem | None: ...

    def apply_generation_result(
        self,
        result: GenerationResult,
        *,
        intent: str | None,
        topic: str | None,
        sentiment: str | None,
        confidence: float | None,
        needs_review: bool | None,
    ) -> ReviewItem:
        """Phase M4C.4 -- idempotent-by-skip write of one frozen
        response-generation result onto its already-classified (or still
        UNPROCESSED) review item. No `expected_version` parameter: the
        current service.py code has never taken one for this method --
        unlike claim/send_as_is/edit_and_send/mark_no_response_needed,
        there is no caller-supplied optimistic-concurrency check here.
        Skips (returns the item completely unchanged, no write, no
        row_version bump) if EITHER (a) `reviewer_action` is already set
        or `ready_to_send` is already True, OR (b) `generation_routing_
        decision` is already set AND `review_status` is one of the
        "generation already landed" statuses (DRAFT_READY, REVIEW_ONLY,
        NO_RESPONSE_RECOMMENDED, IN_REVIEW, APPROVED, EDITED_AND_APPROVED,
        NO_REPLY, SENT). Otherwise: derives the transition action from
        `result` (apply_generate_draft / apply_no_response_recommended /
        apply_review_only), advances review_status via next_status
        (raises InvalidTransition if the item's current status --
        anything other than CLASSIFIED/UNPROCESSED -- has no such edge),
        writes the generation_* metadata fields unconditionally, writes
        generated_at/suggested_final_reply+ai_suggested_final_reply only
        when `result` supplies them, coalesces (`or`) primary_intent/
        ai_primary_intent/topic_subtype/ai_topic_subtype/ai_sentiment
        only-if-currently-unset when the matching argument is given, but
        overwrites sentiment/ai_confidence/needs_review unconditionally
        when given, recomputes review_priority/priority_reason via the
        unmodified `compute_priority` business rule, sets
        outbound_status="NOT_APPLICABLE" only when the final status is
        NO_RESPONSE_RECOMMENDED, and bumps row_version by exactly 1 in
        the same write. Raises ItemNotFound if `comment_id` does not
        exist. Never touches reviewer/assignment/outbound-send fields
        (those belong to claim/send_as_is/edit_and_send/dispatch_
        outbound, all out of scope here)."""
        ...

    def list_queue(
        self,
        *,
        statuses: tuple[str, ...],
        source_file: str | None,
        reviewer_user_id: str | None,
        search: str | None,
        sort: QueueSort,
        offset: int,
        limit: int,
        primary_intents: tuple[str, ...] | None = None,
        topic_subtypes: tuple[str, ...] | None = None,
        reviewer_actions: tuple[str, ...] | None = None,
        sentiments: tuple[str, ...] | None = None,
        priorities: tuple[int, ...] | None = None,
        created_from: dt.datetime | None = None,
        created_to: dt.datetime | None = None,
    ) -> tuple[list[ReviewItem], int]:
        """`sort` is one of newest / oldest (ingestion time), comment_newest /
        comment_oldest (`comment_created_at`), priority_newest /
        priority_oldest. Every ordering ends with the unique comment_id and
        sorts NULLs of its leading key last, so pagination is stable and
        identical across backends. `sentiments`/`priorities` (OR within,
        AND across) and `created_from`/`created_to` (inclusive range on
        `comment_created_at`; NULL dates never match an active range) are
        further optional read filters. A NULL value never matches a
        filtered dimension.

        `primary_intents`/`topic_subtypes`/`reviewer_actions`, when
        given (a non-empty tuple), additionally restrict the result to
        rows whose own `primary_intent`/`topic_subtype`/`reviewer_action`
        is one of the given values (multi-select, OR semantics within
        each dimension, AND semantics across dimensions) -- composed
        with every other filter here (`statuses`, `source_file`,
        `reviewer_user_id`, `search`), never replacing them. Omitted or
        empty (None/(), the default) reproduces the original behavior
        unchanged. This is a read filter only -- it never changes which
        rows are eligible for any workflow transition."""
        ...

    def counts_by_status(self, reviewer_user_id: str | None) -> dict[str, int]: ...

    def claim(
        self,
        comment_id: str,
        *,
        reviewer_id: str,
        display_name: str,
        user_id: str | None,
        expected_version: int | None,
        actor_role: str | None = None,
    ) -> ReviewItem: ...

    def send_as_is(
        self,
        comment_id: str,
        *,
        reviewer_id: str,
        display_name: str,
        user_id: str | None,
        expected_version: int | None,
        actor_role: str | None = None,
        attested: bool = False,
    ) -> ReviewItem: ...

    def edit_and_send(
        self,
        comment_id: str,
        *,
        reviewer_id: str,
        display_name: str,
        user_id: str | None,
        edited_response: str,
        expected_version: int | None,
        actor_role: str | None = None,
        attested: bool = False,
    ) -> ReviewItem: ...

    def mark_no_response_needed(
        self,
        comment_id: str,
        *,
        reviewer_id: str,
        display_name: str,
        user_id: str | None,
        reason: str | None,
        expected_version: int | None,
        actor_role: str | None = None,
    ) -> ReviewItem: ...

    def begin_outbound_dispatch(self, comment_id: str, expected_version: int | None) -> ReviewItem:
        """Mongo runtime boundary — persistence half of
        service.py::dispatch_outbound's pre-send step, extracted so the
        external side effect (calling the outbound sender) never has to
        happen inside a repository. Optimistic-concurrency check first
        (raises VersionConflict on mismatch, exactly like claim/
        send_as_is/etc — never retried). If the item is already fully
        SENT with an idempotency key recorded, returns it completely
        unchanged (no write, no row_version bump) — the caller uses this
        to short-circuit and skip calling the sender again. Otherwise
        transitions review_status via next_status(current, "dispatch"),
        sets outbound_status="SENDING", records outbound_attempted_at,
        assigns outbound_idempotency_key if not already set (from
        comment_id + reviewer_action + final_reply, deterministic and
        stable across retries), and bumps row_version. Raises
        ItemNotFound if `comment_id` does not exist. This method commits
        the "attempting to send" state — it never calls anything
        external itself. See docs/persistence.md's outbound-dispatch
        section for the reliability boundary this leaves in place (no
        transactional outbox: a process crash between this write and the
        external call, or between the external call and
        complete_outbound_dispatch below, leaves the item stuck at
        SENDING until manually reconciled — pre-existing behavior,
        unchanged by this extraction)."""
        ...

    def complete_outbound_dispatch(
        self,
        comment_id: str,
        *,
        ok: bool,
        external_id: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ReviewItem:
        """Mongo runtime boundary — persistence half of
        service.py::dispatch_outbound's post-send step: records the
        outcome of a send attempt already made by the caller (never
        calls anything external itself). No `expected_version` — the
        current code has never taken one here either; the SENDING state
        `begin_outbound_dispatch` wrote is what makes this call
        meaningful, not a version check. `ok=True` transitions via
        next_status(current, "mark_sent") and sets outbound_status=
        "SENT", outbound_sent_at, outbound_external_id, clearing any
        prior error fields. `ok=False` transitions via
        next_status(current, "mark_failed") and sets outbound_status=
        "FAILED" plus the given error_code/error_message. Either way,
        final_reply is preserved unchanged, `ready_to_send` is
        recomputed from the resulting review_status, and row_version is
        bumped. Raises ItemNotFound if `comment_id` does not exist."""
        ...

    def assign_batch(
        self,
        comment_ids: list[str],
        reviewer_user_id: str,
        *,
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ) -> AssignmentResult:
        """Audit contract (all state-changing methods above and this one):
        every successful state change appends exactly one ReviewEvent per
        affected item in the SAME transaction as the change; a failed
        event write aborts the change; idempotent no-op repeats, version
        conflicts, forbidden/invalid transitions write no event."""
        ...

    def assign_by_filter(
        self,
        reviewer_user_id: str,
        *,
        count: int | None,
        source_file: str | None = None,
        primary_intents: tuple[str, ...] | None = None,
        topic_subtypes: tuple[str, ...] | None = None,
        search: str | None = None,
        sentiments: tuple[str, ...] | None = None,
        priorities: tuple[int, ...] | None = None,
        created_from: Any = None,
        created_to: Any = None,
        statuses: tuple[str, ...] | None = None,
        sort: str = "oldest",
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ) -> FilterAssignmentResult:
        """Assign up to `count` (None = every match) comments to one reviewer, server-side across ALL matching rows in a deterministic order.
        Only comments that are unassigned, undecided and at a reviewable status are touched -- an existing assignment or decision is never changed.
        Each row is claimed with a compare-and-set (still unassigned and undecided at write time), so concurrent callers can never both get the
        same comment; exactly one ASSIGNED event per claimed row is written in the same transaction. If some candidates were claimed by someone else
        in the meantime the call selects further candidates (bounded) before reporting a shortfall."""
        ...

    def list_events(self, comment_id: str) -> list[ReviewEvent]:
        """Audit history of one item, oldest first. Read-only."""
        ...

    def list_assignable(
        self,
        *,
        unassigned_only: bool,
        source_file: str | None,
        offset: int,
        limit: int,
        primary_intents: tuple[str, ...] | None = None,
        topic_subtypes: tuple[str, ...] | None = None,
        search: str | None = None,
        sentiments: tuple[str, ...] | None = None,
        priorities: tuple[int, ...] | None = None,
        created_from: dt.datetime | None = None,
        created_to: dt.datetime | None = None,
        reviewer_user_id: str | None = None,
        statuses: tuple[str, ...] | None = None,
        sort: str = "oldest",
    ) -> tuple[list[ReviewItem], int]:
        """Same optional filters and the same deterministic `sort` keys as
        list_queue; additionally `reviewer_user_id` (currently assigned to)
        and `statuses` (narrowing within the assignable-status universe).

        `primary_intents`/`topic_subtypes`, when given (a non-empty
        tuple), additionally restrict the assignable set to rows whose
        own classification is one of the given values -- composed with
        `unassigned_only`/`source_file`, never replacing the existing
        assignability rule (reviewer_action IS NULL and status in the
        assignable-statuses universe). `search`, when given, matches
        `comment_id`/`original_comment` case-sensitively via substring
        -- the exact same semantics `list_queue`'s own `search` already
        has, never a second search implementation. Omitted/empty
        reproduces the original behavior unchanged."""
        ...

    def distinct_source_files(self) -> list[str]: ...

    def unassigned_counts_by_source_file(self) -> dict[str, int]: ...

    def count_all(self) -> int: ...

    def list_ordered(self, comment_ids: list[str] | None) -> list[ReviewItem]: ...

    def get_many(self, comment_ids: list[str]) -> dict[str, ReviewItem]: ...

    def comment_ids_for_source_file(self, source_file: str) -> list[str]: ...

    def reviewer_assignment_counts(self) -> dict[str, ReviewerAssignmentCounts]:
        """Persistence-decoupling milestone — replaces service.py's
        inline SQLAlchemy aggregate (`reviewer_summaries`'s two
        `select(ReviewItem.reviewer_user_id, func.count())` queries).
        One entry per reviewer_user_id that has at least one assigned
        row (rows with no reviewer at all are excluded, matching the
        original `.where(reviewer_user_id.is_not(None))` filter).
        `pending` counts only rows among those with `reviewer_action`
        still unset, exactly as the original second query did."""
        ...

    def get_dedup_candidates(self, comment_ids: list[str]) -> list[DedupCandidate]:
        """Persistence-decoupling milestone — read side of
        run_content_deduplication's exact-content dedup. Returns one
        DedupCandidate per existing comment_id in `comment_ids` (unknown
        ids are simply absent, never an error — matches the original
        `.where(comment_id.in_(comment_ids))` semantics). No ordering
        guarantee; the caller sorts by source_row_index itself."""
        ...

    def find_existing_dedup_canonical(self, source_file: str, content_dedupe_key: str) -> str | None:
        """Persistence-decoupling milestone — cross-batch/historical
        lookup: is there ALREADY a resolved canonical for this exact
        (source_file, content_dedupe_key) pair, from any earlier batch or
        backfill run? Returns its comment_id, or None if this exact-
        content group has never been resolved before."""
        ...

    def set_dedupe_canonical_if_unset(self, comment_id: str, content_dedupe_key: str | None) -> bool:
        """Persistence-decoupling milestone — idempotent, atomic:
        sets `content_dedupe_key`/`dedupe_canonical_id=comment_id` (this
        row becomes its own canonical) ONLY if `dedupe_canonical_id` is
        currently unset on that row. Returns True if it actually wrote,
        False if the row already had a canonical (already resolved by an
        earlier call — never re-touched, matching the original code's
        `if canon_item.dedupe_canonical_id is None:` guard, now enforced
        server-side as a single conditional write instead of a
        read-then-write in application code). Raises ItemNotFound if
        `comment_id` does not exist."""
        ...

    def mark_dedupe_duplicate(self, comment_id: str, content_dedupe_key: str, canonical_id: str) -> None:
        """Persistence-decoupling milestone — marks a non-canonical
        exact-content duplicate: sets `content_dedupe_key`,
        `dedupe_canonical_id=canonical_id`, `review_status=
        "DUPLICATE_CONTENT"`, `outbound_status="NOT_APPLICABLE"`.
        Unconditional (the caller only ever calls this for rows already
        confirmed unresolved and non-canonical this batch — see
        run_content_deduplication). Raises ItemNotFound if `comment_id`
        does not exist."""
        ...

    def review_statuses(self, comment_ids: list[str]) -> dict[str, str]:
        """Mongo runtime boundary — bulk `review_status` lookup for a
        known batch of ids (used by excel_review_flow.py to find which
        of a just-ingested/just-classified batch are currently at a
        particular status, e.g. "still UNPROCESSED" or "now CLASSIFIED",
        without pulling full rows). Unknown ids are simply absent from
        the returned dict, never an error and never a None entry — same
        semantics as `.where(comment_id.in_(comment_ids))` matching
        nothing for an unknown id."""
        ...

    def source_file_status_fields(self, source_file: str) -> list[SourceFileStatusFields]:
        """Mongo runtime boundary — one SourceFileStatusFields per row
        currently persisted for `source_file`, read-only. This is the
        raw shape `_source_file_aggregate`'s cumulative whole-file
        counters are computed from; no aggregation happens here, so the
        counting rule is never duplicated across backends. No ordering
        guarantee (the caller only counts, never orders, this)."""
        ...

    def upsert_ingest_batch(self, rows: list[dict]) -> dict:
        """Phase M4C.1 — idempotent ingest write keyed by comment_id (the
        `_id`/`comment_id` field of each row is required). For each row:
        new comment_id -> insert the full row. Existing row with
        `reviewer_action` falsy AND `review_status` in ("UNPROCESSED",
        None) -> update ONLY the ingest-owned field set (never
        row_version, never any review/classification/generation field).
        Existing row that is locked by either condition -> left
        completely untouched (not even a no-op write attempted). Never
        modifies row_version. Returns {"inserted": n, "updated": n,
        "skipped": n} -- the exact shape
        rcs.review.repository.upsert_ingest_items already returns."""
        ...

    def load_for_prefilter(self, comment_ids: list[str] | None = None) -> pd.DataFrame:
        """Phase M4C.2 -- exact columns the deterministic prefilter rules
        need (comment_id, original_comment, visitor_email, visitor_name,
        comment_created_at, is_exact_duplicate_of), as a DataFrame.
        `comment_ids`, when given, restricts the read to that set;
        omitted (None) reads the whole collection. No ordering guarantee
        (the original SQL has none either -- none is invented here).
        Read-only; may use the standard Mongo read-retry layer."""
        ...

    def apply_prefilter_batch(self, results: list[dict]) -> dict:
        """Phase M4C.2 -- idempotent UPDATE-only write of prefilter
        output. Each dict: {comment_id, is_prefiltered_junk,
        prefilter_reason}. For every row in `results`: sets
        is_prefiltered_junk/prefilter_reason/processed_at (one shared
        timestamp for the whole call, matching the original); junk rows
        additionally get primary_intent=SPAM, routing_decision=EXCLUDE,
        review_status=EXCLUDED, exclusion_reason=<reason>; non-junk rows
        get those four fields cleared back to (None, None, UNPROCESSED,
        None). Deliberately enforces NO review_status guard -- matches
        the current persistence function exactly; caller discipline is
        the only safety boundary, as today. Never touches row_version.
        Raises (aborting the whole call, nothing persisted) if any
        comment_id does not exist -- mirrors the current KeyError
        exactly, never a silent skip."""
        ...

    def load_for_classification(self) -> pd.DataFrame:
        """Phase M4C.3 -- rows eligible for frozen classification:
        review_status == "UNPROCESSED" AND is_prefiltered_junk is
        exactly False (never None/missing -- mirrors PostgreSQL's
        `.is_(False)`, which is strict `IS FALSE`, not `!= True`). No
        comment_ids parameter -- the current free function has none
        either; callers filter client-side (unchanged). No ordering
        guarantee. Read-only; may use the standard Mongo read-retry
        layer."""
        ...

    def apply_classification_batch(
        self,
        results: list[dict],
        *,
        model_name: str,
        model_version: str | None,
        prompt_version: str,
        pipeline_version: str,
        inference_run_id: str,
    ) -> dict:
        """Phase M4C.3 -- idempotent, UPDATE-only write of classification
        output. Skips a row entirely (no field touched) if
        `ai_primary_intent` is already set OR `review_status` is no
        longer "UNPROCESSED" -- ai_* fields are immutable once written.
        On success: writes the full ai_* set, initializes the matching
        working fields, sets review_status="CLASSIFIED". On failure:
        writes only ai_routing_decision="REVIEW_ONLY" +
        ai_classification_reason, leaves review_status as "UNPROCESSED"
        (retry-eligible). Model/prompt/pipeline/run metadata and
        processed_at are written on BOTH the success and failure paths,
        for every non-skipped row -- one shared processed_at timestamp
        per call, exactly as the original. Never touches row_version,
        never touches reviewer/generation/outbound fields. Raises
        (aborting the whole call, nothing persisted) if any comment_id
        does not exist. Returns {"classified": n, "failed": n,
        "skipped_already_classified": n}."""
        ...


class ReviewSessionRepository(Protocol):
    def create(self, *, original_filename: str, source_file: str, upload_token: str) -> ReviewSession: ...

    def get(self, session_id: str) -> ReviewSession | None: ...

    def get_active(self) -> ReviewSession | None: ...

    def set_status(self, session_id: str, status: str) -> None: ...

    def compute_counts(self, source_file: str) -> SessionCounts: ...


class ProcessingJobRepository(Protocol):
    def create(self, session_id: str) -> ProcessingJob: ...

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        current_stage: str | None = None,
        rows_done: int | None = None,
        rows_total: int | None = None,
        error_message: str | None = None,
        finished: bool = False,
    ) -> None: ...

    def get(self, job_id: str) -> ProcessingJob | None: ...

    def latest_for_session(self, session_id: str) -> ProcessingJob | None: ...

    def list_running(self) -> list[ProcessingJob]: ...


class AnalyticsRepository(Protocol):
    """Phase M4C.5 -- the one currently-served runtime analytics entry
    point (rcs.review.analytics.compute_operational_analytics), extracted
    behind its own contract, separate from ReviewItemRepository, since
    this is a cross-row aggregation concern rather than a single-item/
    batch persistence concern. `reviewer_summaries` (service.py, admin-
    only) used to be dependency-gated here on the identity/auth
    migration -- as of Phase M4D.1 its User-side read now goes through
    UserRepository below (see the M4D.1 report); this contract itself is
    unchanged."""

    def compute_operational_analytics(
        self,
        *,
        range_key: str = "week",
        now: dt.datetime | None = None,
        reviewer_id: str | None = None,
        custom_start: dt.datetime | None = None,
        custom_end: dt.datetime | None = None,
        source_file: str | None = None,
        reviewer_user_id: str | None = None,
    ) -> dict: ...


class UserRepository(Protocol):
    """Phase M4D.1 -- the `users` table only. `user_sessions` and the
    Microsoft Entra provider flow are explicitly NOT covered (see this
    module's docstring) -- role/authorization semantics, password
    hashing, and session lifecycle all stay exactly where they are
    (rcs.review.auth), calling into this contract only for the raw
    `users` row lookups/insert they already performed inline.

    Methods are derived only from actual runtime callers found in the
    M4D.1 audit: rcs.review.auth (authenticate/create_user),
    http_api.py's /api/admin/users listing and /api/admin/assign target
    validation, and service.py's reviewer_summaries. scripts/
    manage_users.py (an offline CLI, never served over HTTP) and
    rcs.review.entra_auth (explicitly protected this phase) are
    deliberately NOT rerouted through this contract -- see the M4D.1
    report. No update/activate/deactivate method exists here because no
    runtime HTTP endpoint performs one today; only the CLI does, and it
    stays untouched."""

    def get_by_id(self, user_id: str) -> User | None: ...

    def get_by_email(self, email: str) -> User | None: ...

    def list_all(self) -> list[User]: ...

    def list_active_reviewers(self) -> list[User]: ...

    def create(
        self,
        *,
        user_id: str,
        email: str,
        password_hash: str,
        display_name: str,
        role: str,
        is_active: bool,
        created_at: dt.datetime,
        updated_at: dt.datetime,
    ) -> User:
        """Raises ValueError (same message/type as the original inline
        code) if `email` already exists. Phase M4D.1 audit finding,
        preserved not fixed: this is a read-then-insert check, not an
        atomic constraint -- two concurrent calls with the same email
        can both pass the check before either commits. A real unique
        index (Postgres already has one; Mongo does not yet) is required
        before this is safe under real concurrency -- see the M4D.1
        report's index recommendations."""
        ...

    def set_password_hash(self, user_id: str, password_hash: str, updated_at: dt.datetime) -> bool:
        """Self-service password reset: replace the stored hash. Returns False if no such user. Never changes role/active state."""
        ...


class UserSessionRepository(Protocol):
    """Phase M4D.2 -- the `user_sessions` table only: the four functions
    rcs.review.auth already exposes (start_session/resolve_session/
    end_session/purge_expired_sessions) and nothing else. Token
    generation (rcs.review.auth.new_session_token) stays in auth.py --
    it is a pure function with no DB dependency, not persistence.
    Active-account revalidation stays in auth.py too: this contract
    returns the raw session row only, and auth.py's resolve_session
    still separately calls UserRepository.get_by_id + checks is_active,
    exactly as before -- a session document never caches/replaces that
    check."""

    def create(
        self,
        *,
        session_token: str,
        user_id: str,
        created_at: dt.datetime,
        expires_at: dt.datetime,
    ) -> UserSession:
        """Blind insert -- no existing-token check, matching the
        original inline code exactly: it relies entirely on the
        underlying store's own primary-key uniqueness (Postgres:
        session_token is the PK, violated inserts raise IntegrityError;
        Mongo: session_token as `_id`, violated inserts raise
        DuplicateKeyError). Not expected to ever be exercised in
        practice given new_session_token()'s 256 bits of entropy."""
        ...

    def get_by_token(self, session_token: str) -> UserSession | None: ...

    def delete_by_token(self, session_token: str) -> None:
        """Idempotent: deleting an already-gone/unknown token is a
        silent no-op, matching the original `delete(...).where(...)`
        with no rowcount check."""
        ...

    def delete_expired(self, now: dt.datetime) -> int:
        """expires_at <= now (inclusive), server-side delete predicate.
        Returns the number of rows/documents actually deleted."""
        ...

    def delete_for_user(self, user_id: str) -> int:
        """Revoke EVERY session of one user (password reset). Returns how many were removed; other users' sessions are untouched."""
        ...


class PasswordResetTokenRepository(Protocol):
    """Single-use, hashed, short-lived password-reset tokens. Callers pass the token HASH only; the raw token never reaches a repository."""

    def create(self, *, token_hash: str, user_id: str, created_at: dt.datetime, expires_at: dt.datetime) -> None: ...

    def consume(self, token_hash: str, now: dt.datetime) -> str | None:
        """Atomically mark the token used and return its user_id, but ONLY if it exists, is unused and `expires_at > now`. Any other case -> None (indistinguishable
        to the caller: unknown, expired and already-used tokens all look the same). Two concurrent calls can never both succeed."""
        ...

    def invalidate_for_user(self, user_id: str, now: dt.datetime) -> int:
        """Mark every still-unused token of this user as used (a newer request, or a completed reset, supersedes older links). Returns the count."""
        ...

    def delete_expired(self, now: dt.datetime) -> int:
        """Housekeeping: delete tokens whose `expires_at <= now`. Returns the count."""
        ...


@dataclass(frozen=True)
class DirectMessageRecord:
    message_id: str
    reviewer_user_id: str
    sender_user_id: str
    sender_role: str
    body: str
    client_message_id: str
    created_at: dt.datetime
    read_at: dt.datetime | None


@dataclass(frozen=True)
class ConversationSummary:
    reviewer_user_id: str
    last_message: DirectMessageRecord
    unread_from_reviewer: int  # what an ADMIN has not yet read
    unread_from_admin: int  # what the REVIEWER has not yet read


class DirectMessageRepository(Protocol):
    """Internal admin <-> reviewer text messages. One conversation per reviewer (keyed by the reviewer's user id)."""

    def add(self, *, message_id: str, reviewer_user_id: str, sender_user_id: str, sender_role: str, body: str, client_message_id: str,
            created_at: dt.datetime) -> tuple[DirectMessageRecord, bool]:
        """Stores the message and returns (record, True). If (sender_user_id, client_message_id) already exists the stored record is returned UNCHANGED with False
        (a retried request never creates a second row, even when two identical requests race)."""
        ...

    def list_conversation(self, reviewer_user_id: str, limit: int) -> list[DirectMessageRecord]:
        """The most recent `limit` messages of one conversation, oldest first."""
        ...

    def mark_read(self, reviewer_user_id: str, sender_role: str, read_at: dt.datetime) -> int:
        """Marks the still-unread messages SENT BY `sender_role` in this conversation as read. Returns how many changed."""
        ...

    def unread_count(self, reviewer_user_id: str, sender_role: str) -> int: ...

    def summaries(self) -> dict[str, ConversationSummary]:
        """One entry per reviewer that has at least one message."""
        ...


@dataclass(frozen=True)
class AccountRequestRecord:
    request_id: str
    email: str
    display_name: str
    password_hash: str
    status: str
    created_at: dt.datetime
    decided_at: dt.datetime | None
    decided_by_user_id: str | None


class AccountRequestRepository(Protocol):
    """Self-service account requests. A request is never a login; see rcs.review.account_requests for the rules."""

    def create(self, *, request_id: str, email: str, display_name: str, password_hash: str, created_at: dt.datetime) -> bool:
        """Stores a PENDING request. Returns False (stores nothing, changes nothing) if a request for this e-mail already exists in ANY state."""
        ...

    def get(self, request_id: str) -> AccountRequestRecord | None: ...

    def list_pending(self, limit: int) -> list[AccountRequestRecord]:
        """Oldest first."""
        ...

    def count_pending(self) -> int: ...

    def decide(self, request_id: str, status: str, decided_by_user_id: str, decided_at: dt.datetime) -> AccountRequestRecord | None:
        """Atomic PENDING -> APPROVED|REJECTED. Returns the request as it was BEFORE the change, or None if it does not exist or was already decided (so two admins
        can never both approve, or approve and reject, the same request)."""
        ...
