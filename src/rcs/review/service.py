"""
Reviewer workflow domain service.

No HTTP framework: these methods are the API contract the future reviewer
page will call (list/get/claim/send-as-is/edit-and-send/no-response).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Optional

from config import settings
from rcs.response_generation.schema import GenerationResult
from rcs.response_intelligence.audit import AgentAuditRepository
from rcs.review.dto import (
    AssignableItemSummary,
    ReviewerItemDetail,
    ReviewerQueueItem,
    to_assignable_item_summary,
    to_detail,
    to_internal,
    to_queue_item,
)
from rcs.review.identity import ReviewerIdentity
from rcs.response_generation.claim_gates import REASON_MESSAGES_AR
from rcs.response_generation.validate import ReplyFindings, collect_reply_findings
from rcs.review.outbound import OutboundSendResult, OutboundSender, UnconfiguredOutboundSender
from rcs.review.persistence import resolve_bundle
from rcs.review.repositories.contracts import (
    AnalyticsRepository,
    AssignmentResult,
    ReviewerAssignmentCounts,
    ReviewItemRepository,
    UserRepository,
)
from rcs.review.sessions import compute_session_counts
from rcs.review.workflow import AttestationNotPermitted, AttestationRequired, DraftValidationError, ItemNotFound


@dataclass(frozen=True)
class FactAttestation:
    """A reviewer's explicit confirmation that the factual claims named by `finding_codes` are accurate. Only meaningful for the findings it
    names; the server compares it with the current findings."""

    confirmed: bool
    finding_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplyAssessment:
    blocking: tuple[str, ...]
    attestable: tuple[str, ...]
    may_attest: bool

    @property
    def attestation_required(self) -> bool:
        return not self.blocking and bool(self.attestable)


@dataclass(frozen=True)
class QueuePage:
    items: tuple[ReviewerQueueItem, ...]
    total: int
    offset: int
    limit: int


# AssignmentResult moved to rcs.review.repositories.contracts (Phase M4A)
# -- it is the repository's output shape (both today's Postgres batch and
# any future Mongo transactional batch must produce exactly this), not a
# service-layer concept. Re-imported above for backward-compatible
# `from rcs.review.service import AssignmentResult`.


@dataclass(frozen=True)
class ReviewerSummary:
    user_id: str
    email: str
    display_name: str
    assigned_total: int
    pending: int
    completed: int


@dataclass(frozen=True)
class AssignableItemsPage:
    items: tuple[AssignableItemSummary, ...]
    total: int
    offset: int
    limit: int


@dataclass(frozen=True)
class SourceFileSummary:
    source_file: str
    total_reviewable: int
    unassigned: int
    assigned: int
    completed: int


_SORT_ALIASES = {"ingested_newest": "newest", "ingested_oldest": "oldest"}
_SORT_MODES = (
    "priority", "newest", "oldest", "ingested_newest", "ingested_oldest", "comment_newest", "comment_oldest"
)


def _assignable_sort(sort: str) -> str:
    """Repository sort key for an assignment list. `priority` means
    priority-then-oldest, exactly like the queue; an unknown value raises
    instead of silently using another ordering."""
    mode = (sort or "oldest").strip().lower()
    if mode not in _SORT_MODES:
        raise ValueError(f"unknown sort: {sort!r}")
    if mode == "priority":
        return "priority_oldest"
    return _SORT_ALIASES.get(mode, mode)


class ReviewWorkflowService:
    def __init__(
        self,
        engine,
        *,
        repo: ReviewItemRepository | None = None,
        analytics_repo: AnalyticsRepository | None = None,
        user_repo: UserRepository | None = None,
        audit_repo: AgentAuditRepository | None = None,
    ):
        self.engine = engine
        # Persistence extracted behind the repository contracts
        # (rcs.review.repositories.contracts). `repo`/`analytics_repo`/
        # `user_repo`/`audit_repo` remain injectable seams for tests;
        # every current caller (ReviewWorkflowService(engine)) is
        # unchanged and gets its defaults from the persistence bundle
        # (rcs.review.persistence.resolve_bundle) instead of importing a
        # concrete Postgres class directly here — the bundle is the only
        # place that decides which backend those defaults come from.
        # dispatch_outbound below also goes through self.repo now
        # (begin_outbound_dispatch/complete_outbound_dispatch) — no
        # method on this class opens a SQLAlchemy Session directly.
        bundle = resolve_bundle(engine)
        self.repo: ReviewItemRepository = repo or bundle.review_items
        self.analytics_repo: AnalyticsRepository = analytics_repo or bundle.analytics
        self.user_repo: UserRepository = user_repo or bundle.users
        self.audit_repo: AgentAuditRepository = audit_repo or bundle.audit

    def get_review_item(self, comment_id: str, *, internal: bool = False):
        item = self.repo.get(comment_id)
        if item is None:
            raise ItemNotFound(comment_id)
        if internal:
            return to_detail(item), to_internal(item)
        return to_detail(item)

    def apply_generation_result(
        self,
        result: GenerationResult,
        *,
        intent: str | None = None,
        topic: str | None = None,
        sentiment: str | None = None,
        confidence: float | None = None,
        needs_review: bool | None = None,
    ) -> ReviewerItemDetail:
        """Phase M4C.4: the actual persistence now lives in
        PostgresReviewItemRepository.apply_generation_result (verbatim
        move); this method is kept as the orchestration entry point --
        it delegates to self.repo and converts the returned item to the
        reviewer-facing DTO, exactly as before."""
        item = self.repo.apply_generation_result(
            result,
            intent=intent,
            topic=topic,
            sentiment=sentiment,
            confidence=confidence,
            needs_review=needs_review,
        )
        return to_detail(item)

    def list_review_queue(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
        include_no_response_recommended: bool = False,
        search: str | None = None,
        newest_first: bool = False,
        sort: str | None = None,
        offset: int = 0,
        limit: int = 50,
        source_file: str | None = None,
        reviewer_user_id: str | None = None,
        primary_intents: tuple[str, ...] | None = None,
        topic_subtypes: tuple[str, ...] | None = None,
        reviewer_actions: tuple[str, ...] | None = None,
        sentiments: tuple[str, ...] | None = None,
        priorities: tuple[int, ...] | None = None,
        created_from: dt.datetime | None = None,
        created_to: dt.datetime | None = None,
    ) -> QueuePage:
        """`source_file`, when given (Phase 4 — active review session
        scoping), restricts the queue to that one workbook's rows only.
        `reviewer_user_id`, when given (Phase 8 — reviewer isolation),
        restricts the queue to items assigned to exactly that reviewer
        (the HTTP layer passes the caller's own id for every REVIEWER).
        `primary_intents`/`topic_subtypes`/`reviewer_actions`/`sentiments`/
        `priorities` narrow by value (OR within a dimension, AND across
        all filters) and `created_from`/`created_to` by comment date.

        `sort`: priority (default), newest / oldest / ingested_newest /
        ingested_oldest (ingestion time), comment_newest / comment_oldest
        (the visitor comment's own date). An unrecognised sort raises
        ValueError rather than silently using another ordering."""
        pending = statuses if statuses is not None else settings.QUEUE_PENDING_STATUSES
        if include_no_response_recommended:
            pending = tuple(dict.fromkeys((*pending, "NO_RESPONSE_RECOMMENDED")))
        mode = (sort or "").strip().lower()
        if mode not in _SORT_MODES and mode != "":
            raise ValueError(f"unknown sort: {sort!r}")
        if not mode:
            mode = "newest" if newest_first else "priority"
        mode = _SORT_ALIASES.get(mode, mode)
        if mode in ("newest", "oldest", "comment_newest", "comment_oldest"):
            resolved_sort = mode
        elif newest_first:
            resolved_sort = "priority_newest"
        else:
            resolved_sort = "priority_oldest"
        page, total = self.repo.list_queue(
            statuses=pending,
            source_file=source_file,
            reviewer_user_id=reviewer_user_id,
            search=search,
            sort=resolved_sort,
            offset=offset,
            limit=limit,
            primary_intents=primary_intents,
            topic_subtypes=topic_subtypes,
            reviewer_actions=reviewer_actions,
            sentiments=sentiments,
            priorities=priorities,
            created_from=created_from,
            created_to=created_to,
        )
        return QueuePage(
            items=tuple(to_queue_item(r) for r in page),
            total=total,
            offset=offset,
            limit=limit,
        )

    def workflow_counts(self, *, reviewer_user_id: str | None = None) -> dict[str, Any]:
        """Small status aggregations for dashboard cards. Not an analytics
        product. `reviewer_user_id`, when given, scopes the counts to one
        reviewer's own assigned items (Phase 8) — omitted (None, the
        default, what ADMIN gets) reproduces the original global counts."""
        by_status = self.repo.counts_by_status(reviewer_user_id)

        def _sum(*keys: str) -> int:
            return sum(by_status.get(k, 0) for k in keys)

        return {
            "all": sum(by_status.values()),
            "pending": _sum("DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW", "NO_RESPONSE_RECOMMENDED"),
            "approved": _sum("APPROVED", "EDITED_AND_APPROVED", "SENT", "SENDING"),
            "no_reply": _sum("NO_REPLY"),
            "send_failed": _sum("SEND_FAILED"),
            "by_status": by_status,
        }

    def operational_analytics(
        self,
        *,
        range_key: str = "week",
        reviewer_id: str | None = None,
        custom_start: dt.datetime | None = None,
        custom_end: dt.datetime | None = None,
        now: dt.datetime | None = None,
        source_file: str | None = None,
        reviewer_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Phase M4C.5: the actual query logic now lives in
        PostgresAnalyticsRepository.compute_operational_analytics
        (verbatim move); this method delegates to self.analytics_repo,
        exactly as apply_generation_result delegates to self.repo."""
        return self.analytics_repo.compute_operational_analytics(
            range_key=range_key,
            now=now,
            reviewer_id=reviewer_id,
            custom_start=custom_start,
            custom_end=custom_end,
            source_file=source_file,
            reviewer_user_id=reviewer_user_id,
        )

    def claim_review(
        self,
        comment_id: str,
        reviewer: ReviewerIdentity,
        *,
        expected_version: int | None = None,
    ) -> ReviewerItemDetail:
        item = self.repo.claim(
            comment_id,
            reviewer_id=reviewer.reviewer_id,
            display_name=reviewer.display_name,
            user_id=reviewer.user_id,
            expected_version=expected_version,
            actor_role=reviewer.role,
        )
        return to_detail(item)

    # ---- reply-safety validation (deterministic; no model, no network) ------------------------------------------------------------
    @staticmethod
    def _official_grounding_possible() -> bool:
        """True only when an official-knowledge source is configured. Without one, no draft can have been grounded, so re-validating
        stored text with empty grounding is exactly equivalent to validating it at generation time."""
        return bool(settings.OFFICIAL_KNOWLEDGE_SOURCE_PATH)

    def assess_reply(self, comment_id: str, text: str | None, reviewer: ReviewerIdentity | None = None) -> ReplyAssessment:
        """All findings for `text`, split into never-attestable and attestable, plus whether THIS reviewer may attest. Pure: stores nothing,
        never rewrites the text."""
        item = self.repo.get(comment_id)
        if item is None:
            raise ItemNotFound(comment_id)
        findings = collect_reply_findings(text, trusted_grounding_text=None, comment_text=item.original_comment)
        return ReplyAssessment(
            blocking=findings.blocking,
            attestable=findings.attestable,
            may_attest=self._may_attest(reviewer),
        )

    @staticmethod
    def _may_attest(reviewer: ReviewerIdentity | None) -> bool:
        role = (getattr(reviewer, "role", None) or "").strip().upper()
        return bool(settings.FACT_ATTESTATION_ENABLED and role and role in settings.FACT_ATTESTATION_ROLES)

    def validate_reply_text(self, comment_id: str, text: str | None) -> tuple[bool, str | None, str | None]:
        """(ok, reason_code, message_ar) - ok only when there are NO findings at all (attestable ones included). Never persists anything."""
        a = self.assess_reply(comment_id, text)
        first = (a.blocking + a.attestable)[:1]
        if not first:
            return True, None, None
        return False, first[0], REASON_MESSAGES_AR.get(first[0], "لم يجتز الرد فحص السلامة التلقائي.")

    def _enforce_reply_policy(
        self,
        comment_id: str,
        text: str | None,
        reviewer: ReviewerIdentity,
        attestation: FactAttestation | None,
        *,
        grounding_checked_at_generation: bool = False,
    ) -> bool:
        """ONE safety policy for send_as_is and edit_and_send. Returns True when the send must be recorded as ATTESTED.

        1. Any never-attestable finding (URL, commitment / promise, institutional policy, operational claim, security, shape) blocks. No attestation
           can change that.
        2. Attestable findings (unverifiable definitions / verdicts / numbers) need: attestation switched on, an authenticated role in
           FACT_ATTESTATION_ROLES, an explicit confirmation, and finding codes EXACTLY equal to the current findings (so the confirmation is for
           these claims, not blanket and not stale).
        `grounding_checked_at_generation`: a stored generated draft was already checked WITH grounding when it was produced, so the
        grounding-dependent (attestable) categories are not re-evaluated against empty grounding; the never-attestable categories always are."""
        findings = collect_reply_findings(text, trusted_grounding_text=None, comment_text=self._comment_text(comment_id))
        blocking = findings.blocking
        attestable = () if grounding_checked_at_generation else findings.attestable
        if blocking:
            raise DraftValidationError(
                blocking[0], REASON_MESSAGES_AR.get(blocking[0], "لم يجتز الرد فحص السلامة التلقائي."), blocking=blocking, attestable=attestable
            )
        if not attestable:
            return False
        if not self._may_attest(reviewer):
            raise AttestationNotPermitted(attestable)
        if attestation is None or not attestation.confirmed or set(attestation.finding_codes) != set(attestable):
            raise AttestationRequired(attestable, tuple(REASON_MESSAGES_AR.get(c, c) for c in attestable))
        return True

    def _comment_text(self, comment_id: str) -> str | None:
        item = self.repo.get(comment_id)
        if item is None:
            raise ItemNotFound(comment_id)
        return item.original_comment

    def send_as_is(
        self,
        comment_id: str,
        reviewer: ReviewerIdentity,
        *,
        expected_version: int | None = None,
        attestation: FactAttestation | None = None,
    ) -> ReviewerItemDetail:
        attested = False
        stored = self.repo.get(comment_id)
        if stored is not None and (stored.suggested_final_reply or "").strip():
            # Same policy as edit_and_send. A draft generated while an official source is configured was grounded then, so only its
            # grounding-independent categories are re-checked; without a source no draft could have been grounded and everything is.
            attested = self._enforce_reply_policy(
                comment_id, stored.suggested_final_reply, reviewer, attestation,
                grounding_checked_at_generation=self._official_grounding_possible(),
            )
        item = self.repo.send_as_is(
            comment_id,
            reviewer_id=reviewer.reviewer_id,
            display_name=reviewer.display_name,
            user_id=reviewer.user_id,
            expected_version=expected_version,
            actor_role=reviewer.role,
            attested=attested,
        )
        return to_detail(item)

    def edit_and_send(
        self,
        comment_id: str,
        reviewer: ReviewerIdentity,
        edited_response: str,
        *,
        expected_version: int | None = None,
        attestation: FactAttestation | None = None,
    ) -> ReviewerItemDetail:
        attested = False
        if (edited_response or "").strip():                        # empty input keeps its existing InvalidTransition error from the repository
            attested = self._enforce_reply_policy(comment_id, edited_response, reviewer, attestation)   # BEFORE any state change; text never rewritten
        item = self.repo.edit_and_send(
            comment_id,
            reviewer_id=reviewer.reviewer_id,
            display_name=reviewer.display_name,
            user_id=reviewer.user_id,
            edited_response=edited_response,
            expected_version=expected_version,
            actor_role=reviewer.role,
            attested=attested,
        )
        return to_detail(item)

    def mark_no_response_needed(
        self,
        comment_id: str,
        reviewer: ReviewerIdentity,
        *,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> ReviewerItemDetail:
        item = self.repo.mark_no_response_needed(
            comment_id,
            reviewer_id=reviewer.reviewer_id,
            display_name=reviewer.display_name,
            user_id=reviewer.user_id,
            reason=reason,
            expected_version=expected_version,
            actor_role=reviewer.role,
        )
        return to_detail(item)

    def dispatch_outbound(
        self,
        comment_id: str,
        sender: OutboundSender | None = None,
        *,
        expected_version: int | None = None,
    ) -> tuple[ReviewerItemDetail, OutboundSendResult]:
        """Persistence state transition, concurrency check, and the
        external side effect are three distinct steps, kept separate on
        purpose: self.repo.begin_outbound_dispatch/complete_outbound_
        dispatch (both backend-neutral, see rcs.review.repositories.
        contracts) own the first two; only the adapter.send_response
        call in the middle ever reaches outside this process, and it
        never happens while a repository write is in flight either side
        of it. expected_version is checked once, in begin_outbound_
        dispatch, before any mutation -- a version conflict there raises
        immediately and is never retried.

        Reliability boundary (unchanged by this extraction, see
        docs/persistence.md): there is no transactional outbox. A crash
        between begin_outbound_dispatch's commit and the adapter call,
        or between the adapter call and complete_outbound_dispatch's
        commit, leaves the item stuck at outbound_status="SENDING" until
        an operator reconciles it by hand. UnconfiguredOutboundSender
        remains the only sender ever wired in this application, so this
        path has never executed a real external send."""
        adapter = sender or UnconfiguredOutboundSender()
        item = self.repo.begin_outbound_dispatch(comment_id, expected_version)
        if item.review_status == "SENT" and item.outbound_idempotency_key:
            result = OutboundSendResult(
                ok=True,
                comment_id=comment_id,
                idempotency_key=item.outbound_idempotency_key,
                external_id=item.outbound_external_id,
            )
            return to_detail(item), result

        send_result = adapter.send_response(
            comment_id=comment_id,
            source_reference=item.source_reference or item.source_file,
            final_response=item.final_reply or "",
            idempotency_key=item.outbound_idempotency_key,
        )

        item = self.repo.complete_outbound_dispatch(
            comment_id,
            ok=send_result.ok,
            external_id=send_result.external_id,
            error_code=send_result.error_code,
            error_message=send_result.error_message,
        )
        return to_detail(item), send_result

    def assign_reviewer(
        self,
        comment_ids: list[str],
        reviewer_user_id: str,
        *,
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ) -> AssignmentResult:
        """Admin-only (enforced by the caller — http_api.py's
        _require_admin) assign/reassign. One transaction covering the
        whole batch (atomic); re-running with the same arguments is a
        no-op for rows already correctly assigned (idempotent). Eligible
        = row exists, has no reviewer decision yet (reviewer_action IS
        NULL — checked first, so a decided item is always reported as
        skipped_already_decided regardless of its current status), and
        is at a genuinely reviewable status (_ASSIGNABLE_STATUSES —
        excludes both terminal non-content rows like EXCLUDED/
        DUPLICATE_CONTENT and pipeline-in-progress rows like
        UNPROCESSED/CLASSIFIED). This single rule covers both a
        brand-new assignment and a reassignment identically, and is
        exactly what makes "never overwrite an already-decided item"
        true by construction rather than by a separate check."""
        return self.repo.assign_batch(
            comment_ids, reviewer_user_id, actor_user_id=actor_user_id, actor_role=actor_role
        )

    def assign_by_filter(
        self,
        reviewer_user_id: str,
        *,
        count: int | None,
        source_file: str | None = None,
        primary_intents=None,
        topic_subtypes=None,
        search: str | None = None,
        sentiments=None,
        priorities=None,
        created_from=None,
        created_to=None,
        statuses=None,
        sort: str = "oldest",
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ):
        """Admin-only (enforced by the caller): assign up to `count` (None = all) matching comments that are still unassigned, undecided and
        reviewable. Never reassigns or touches a decided item (use assign_reviewer with explicit ids for that). Server-side across every matching row,
        in the deterministic order of `sort` (default oldest first, ties by comment id). Returns repositories.contracts.FilterAssignmentResult."""
        return self.repo.assign_by_filter(
            reviewer_user_id, count=count, source_file=source_file, primary_intents=primary_intents, topic_subtypes=topic_subtypes,
            search=search, sentiments=sentiments, priorities=priorities, created_from=created_from, created_to=created_to,
            statuses=statuses, sort=_assignable_sort(sort), actor_user_id=actor_user_id, actor_role=actor_role,
        )

    def item_history(self, comment_id: str):
        """Admin-only (enforced by the caller): the item's audit events,
        oldest first. Raises ItemNotFound for an unknown comment_id."""
        if self.repo.get(comment_id) is None:
            raise ItemNotFound(comment_id)
        return tuple(self.repo.list_events(comment_id))

    def reviewer_summaries(self) -> tuple[ReviewerSummary, ...]:
        """Admin-only (enforced by the caller): every active REVIEWER
        with their current assignment counts. `pending` = assigned but no
        decision yet; `completed` = assigned and decided.

        The User-side read goes through self.user_repo.
        list_active_reviewers() (role=REVIEWER + is_active filter,
        ordered by display_name); the ReviewItem-side aggregate counts
        go through self.repo.reviewer_assignment_counts()
        (ReviewItemRepository, implemented for both backends) — no raw
        SQLAlchemy left in this method, and no aggregation business
        logic duplicated here: the "assigned but no decision yet" /
        "assigned and decided" split is computed once, in the
        repository, from the same two fields (reviewer_user_id,
        reviewer_action) either backend already stores."""
        reviewers = self.user_repo.list_active_reviewers()
        counts = self.repo.reviewer_assignment_counts()
        return tuple(
            ReviewerSummary(
                user_id=r.user_id,
                email=r.email,
                display_name=r.display_name,
                assigned_total=counts.get(r.user_id, ReviewerAssignmentCounts(0, 0)).total,
                pending=counts.get(r.user_id, ReviewerAssignmentCounts(0, 0)).pending,
                completed=counts.get(r.user_id, ReviewerAssignmentCounts(0, 0)).total
                - counts.get(r.user_id, ReviewerAssignmentCounts(0, 0)).pending,
            )
            for r in reviewers
        )

    def list_assignable_items(
        self,
        *,
        unassigned_only: bool = False,
        source_file: str | None = None,
        offset: int = 0,
        limit: int = 50,
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
    ) -> AssignableItemsPage:
        """Admin-only (enforced by the caller): items eligible for
        assignment right now — same eligibility rule as assign_reviewer
        (reviewer_action IS NULL, status in the assignable-statuses
        universe), so everything this lists genuinely can be assigned.
        Read-only. `primary_intents`/`topic_subtypes` narrow this same
        eligible set by classification (multi-select); `search` reuses
        `list_review_queue`'s own text-match semantics. See
        ReviewItemRepository.list_assignable's docstring."""
        page, total = self.repo.list_assignable(
            unassigned_only=unassigned_only,
            source_file=source_file,
            offset=offset,
            limit=limit,
            primary_intents=primary_intents,
            topic_subtypes=topic_subtypes,
            search=search,
            sentiments=sentiments,
            priorities=priorities,
            created_from=created_from,
            created_to=created_to,
            reviewer_user_id=reviewer_user_id,
            statuses=statuses,
            sort=_assignable_sort(sort),
        )
        return AssignableItemsPage(
            items=tuple(to_assignable_item_summary(i) for i in page),
            total=total,
            offset=offset,
            limit=limit,
        )

    def source_file_summaries(self) -> tuple[SourceFileSummary, ...]:
        """Admin-only (enforced by the caller): one row per distinct
        source_file. Reuses compute_session_counts (Phase 4) for
        total_reviewable/completed rather than re-deriving that logic —
        only the unassigned/assigned split is new here."""
        source_files = self.repo.distinct_source_files()
        unassigned_by_file = self.repo.unassigned_counts_by_source_file()
        summaries = []
        for source_file in source_files:
            counts = compute_session_counts(self.engine, source_file)
            unassigned = unassigned_by_file.get(source_file, 0)
            summaries.append(
                SourceFileSummary(
                    source_file=source_file,
                    total_reviewable=counts.canonical_total,
                    unassigned=unassigned,
                    assigned=counts.canonical_total - unassigned,
                    completed=counts.completed,
                )
            )
        return tuple(summaries)
