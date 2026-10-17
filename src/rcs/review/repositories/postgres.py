"""
Phase M4A — PostgreSQL implementation of the repository contracts.

Every method body here is a verbatim move of code that used to live
inline in service.py / sessions.py / http_api.py / excel_review_flow.py
-- not a rewrite. `_now`/`_bump`/`_check_version`/`_idempotency_key`/
`_generation_apply_action` are local to this module rather than imported
from service.py because a repository must not import from the service
layer it is meant to sit underneath. Both this module and
rcs.review.repositories.mongo must stay behaviorally identical for every
contract method; if one changes, mirror the change in the other.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd
from sqlalchemy import case, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import set_committed_value

from config import settings
from rcs.response_generation.schema import GenerationResult
from rcs.response_intelligence.audit import AgentAuditRecord
from rcs.review.audit_events import (
    EVENT_ASSIGNED,
    EVENT_CLAIMED,
    EVENT_EDIT_AND_SEND,
    EVENT_EDIT_AND_SEND_ATTESTED,
    EVENT_SEND_AS_IS_ATTESTED,
    EVENT_NO_RESPONSE_NEEDED,
    EVENT_REASSIGNED,
    EVENT_SEND_AS_IS,
    ReviewEvent,
    build_event_fields,
    event_from_fields,
)
from rcs.review.analytics import OUTCOME_ACTIONS, OVERDUE_DAYS, PENDING, _bucket_age, _complete_counts, range_bounds
from rcs.review.clock import iso_app, now_naive
from rcs.review.models import (
    AgentExecutionTrace,
    ProcessingJob,
    ReviewItem,
    ReviewItemEvent,
    AccountRequest,
    DirectMessage,
    PasswordResetToken,
    ReviewSession,
    User,
    UserSession,
)
from rcs.review.priority import compute_priority
from rcs.review.repository import INGEST_OWNED_FIELDS, count_rows
from rcs.review.workflow import InvalidTransition, ItemNotFound, VersionConflict, next_status

from .contracts import (
    AccountRequestRecord,
    AssignmentResult,
    ConversationSummary,
    DedupCandidate,
    DirectMessageRecord,
    FilterAssignmentResult,
    QueueSort,
    ReviewerAssignmentCounts,
    SessionCounts,
    SourceFileStatusFields,
)

# Same eligibility rule as service.py's module-level constants (moved
# here verbatim -- assign_batch/list_assignable are the only callers).
_NOT_ASSIGNABLE_STATUSES = ("EXCLUDED", "DUPLICATE_CONTENT")

# The genuinely "ready for a human reviewer" statuses -- same universe as
# service.py's original _ASSIGNABLE_STATUSES constant (moved here
# verbatim, since assign_batch/list_assignable are the only callers).
_ASSIGNABLE_STATUSES = settings.QUEUE_PENDING_STATUSES


def _now() -> dt.datetime:
    return now_naive()


def _bump(item: ReviewItem) -> None:
    item.row_version = int(item.row_version or 1) + 1
    item.updated_at = _now()


def _add_event(
    session: Session,
    item: ReviewItem,
    event_type: str,
    *,
    actor_user_id: str | None,
    actor_role: str | None,
    from_status: str | None,
    from_assignee: str | None,
) -> None:
    """Stages one audit event on the CALLER's open Session, so it commits
    (or rolls back) together with the state change it describes. No
    try/except anywhere: a failure here aborts the whole action."""
    fields = build_event_fields(
        review_item_id=item.comment_id,
        event_type=event_type,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        from_status=from_status,
        to_status=item.review_status,
        from_assignee_user_id=from_assignee,
        to_assignee_user_id=item.reviewer_user_id,
        created_at=_now(),
    )
    session.add(ReviewItemEvent(**fields))


def _check_version(item: ReviewItem, expected: int | None) -> None:
    if expected is None:
        return
    current = int(item.row_version or 1)
    if expected != current:
        raise VersionConflict(f"row_version {current} != expected {expected}")


def _apply_classification_filters(
    stmt,
    *,
    primary_intents=None,
    topic_subtypes=None,
    sentiments=None,
    priorities=None,
    created_from=None,
    created_to=None,
):
    """Optional multi-select / range filters shared by list_queue and
    list_assignable: OR within a dimension, AND across dimensions. Rows
    with a NULL value in a filtered dimension never match it. The
    comment-date range filters `comment_created_at` (the visitor
    comment's own date), inclusive at both ends."""
    if primary_intents:
        stmt = stmt.where(ReviewItem.primary_intent.in_(primary_intents))
    if topic_subtypes:
        stmt = stmt.where(ReviewItem.topic_subtype.in_(topic_subtypes))
    if sentiments:
        stmt = stmt.where(ReviewItem.sentiment.in_(sentiments))
    if priorities:
        stmt = stmt.where(ReviewItem.review_priority.in_(priorities))
    if created_from is not None:
        stmt = stmt.where(ReviewItem.comment_created_at >= created_from)
    if created_to is not None:
        stmt = stmt.where(ReviewItem.comment_created_at <= created_to)
    return stmt


def _order_clauses(sort: str) -> list:
    """Deterministic ORDER BY for every public sort. The leading key sorts
    NULLs last in both directions (MongoDB parity, see
    MongoReviewItemRepository._find_ordered) and the final key is the
    unique comment_id, so OFFSET/LIMIT pagination never repeats or skips
    rows even when many rows share an ingestion or comment timestamp."""
    ing, cc, pr = ReviewItem.ingested_at, ReviewItem.comment_created_at, ReviewItem.review_priority
    tie = ReviewItem.comment_id.asc()
    if sort == "newest":
        return [ing.desc().nulls_last(), tie]
    if sort == "oldest":
        return [ing.asc().nulls_last(), tie]
    if sort == "comment_newest":
        return [cc.desc().nulls_last(), ing.desc(), tie]
    if sort == "comment_oldest":
        return [cc.asc().nulls_last(), ing.asc(), tie]
    if sort == "priority_newest":
        return [pr.asc().nulls_last(), ing.desc(), tie]
    if sort == "priority_oldest":
        return [pr.asc().nulls_last(), ing.asc(), tie]
    raise ValueError(f"unknown queue sort: {sort!r}")


def _idempotency_key(comment_id: str, action: str, body: str) -> str:
    payload = f"{comment_id}|{action}|{body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _generation_apply_action(result: GenerationResult) -> str:
    """Verbatim move of service.py's module-level helper of the same
    name -- duplicated (not imported) for the same reason `_now`/`_bump`
    /`_check_version` are duplicated above: a repository must not import
    from the service layer it sits underneath."""
    if result.routing_decision == "GENERATE_DRAFT" and result.generation_status == "SUCCEEDED":
        return "apply_generate_draft"
    if result.routing_decision == "NO_RESPONSE_RECOMMENDED":
        return "apply_no_response_recommended"
    return "apply_review_only"


_GENERATION_ALREADY_LANDED_STATUSES = (
    "DRAFT_READY",
    "REVIEW_ONLY",
    "NO_RESPONSE_RECOMMENDED",
    "IN_REVIEW",
    "APPROVED",
    "EDITED_AND_APPROVED",
    "NO_REPLY",
    "SENT",
)


class PostgresReviewItemRepository:
    def __init__(self, engine):
        self.engine = engine

    def get(self, comment_id: str) -> ReviewItem | None:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is not None:
                session.expunge(item)
            return item

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
        with Session(self.engine) as session:
            item = session.get(ReviewItem, result.comment_id)
            if item is None:
                raise ItemNotFound(result.comment_id)
            if item.reviewer_action or item.ready_to_send:
                session.expunge(item)
                return item
            if item.generation_routing_decision and item.review_status in _GENERATION_ALREADY_LANDED_STATUSES:
                session.expunge(item)
                return item

            action = _generation_apply_action(result)
            item.review_status = next_status(item.review_status, action)
            item.generation_routing_decision = result.routing_decision
            item.generation_status = result.generation_status
            item.generation_reason_code = result.reason_code
            item.generation_provider = result.provider
            item.generation_model_name = result.model_name
            item.generation_prompt_version = result.prompt_version
            if result.generated_at:
                try:
                    item.generated_at = dt.datetime.fromisoformat(result.generated_at.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    item.generated_at = _now()
            if result.suggested_response:
                item.suggested_final_reply = result.suggested_response
                item.ai_suggested_final_reply = result.suggested_response
            if intent:
                item.primary_intent = item.primary_intent or intent
                item.ai_primary_intent = item.ai_primary_intent or intent
            if topic:
                item.topic_subtype = item.topic_subtype or topic
                item.ai_topic_subtype = item.ai_topic_subtype or topic
            if sentiment:
                item.sentiment = sentiment
                item.ai_sentiment = item.ai_sentiment or sentiment
            if confidence is not None:
                item.ai_confidence = confidence
            if needs_review is not None:
                item.needs_review = needs_review
            item.source_reference = item.source_reference or item.source_file
            prio, reason = compute_priority(
                primary_intent=item.primary_intent,
                topic_subtype=item.topic_subtype,
                generation_routing=item.generation_routing_decision,
                needs_review=item.needs_review,
                confidence=item.ai_confidence,
            )
            item.review_priority = prio
            item.priority_reason = reason
            if item.review_status == "NO_RESPONSE_RECOMMENDED":
                item.outbound_status = "NOT_APPLICABLE"
            _bump(item)
            session.commit()
            session.refresh(item)
            session.expunge(item)
            return item

    def begin_outbound_dispatch(self, comment_id: str, expected_version: int | None) -> ReviewItem:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            _check_version(item, expected_version)
            if item.review_status == "SENT" and item.outbound_idempotency_key:
                session.expunge(item)
                return item
            item.review_status = next_status(item.review_status, "dispatch")
            item.outbound_status = "SENDING"
            item.outbound_attempted_at = _now()
            item.outbound_idempotency_key = item.outbound_idempotency_key or _idempotency_key(
                comment_id, item.reviewer_action or "dispatch", item.final_reply or ""
            )
            _bump(item)
            session.commit()
            session.refresh(item)
            session.expunge(item)
            return item

    def complete_outbound_dispatch(
        self,
        comment_id: str,
        *,
        ok: bool,
        external_id: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ReviewItem:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            preserved = item.final_reply
            if ok:
                item.review_status = next_status(item.review_status, "mark_sent")
                item.outbound_status = "SENT"
                item.outbound_sent_at = _now()
                item.outbound_external_id = external_id
                item.outbound_error_code = None
                item.outbound_error_message = None
            else:
                item.review_status = next_status(item.review_status, "mark_failed")
                item.outbound_status = "FAILED"
                item.outbound_error_code = error_code
                item.outbound_error_message = error_message
            item.final_reply = preserved
            item.ready_to_send = item.review_status in settings.READY_TO_SEND_STATUSES or item.review_status == "SEND_FAILED"
            _bump(item)
            session.commit()
            session.refresh(item)
            session.expunge(item)
            return item

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
        with Session(self.engine) as session:
            stmt = select(ReviewItem).where(ReviewItem.review_status.in_(statuses))
            if source_file:
                stmt = stmt.where(ReviewItem.source_file == source_file)
            if reviewer_user_id:
                stmt = stmt.where(ReviewItem.reviewer_user_id == reviewer_user_id)
            stmt = _apply_classification_filters(
                stmt,
                primary_intents=primary_intents,
                topic_subtypes=topic_subtypes,
                sentiments=sentiments,
                priorities=priorities,
                created_from=created_from,
                created_to=created_to,
            )
            if reviewer_actions:
                stmt = stmt.where(ReviewItem.reviewer_action.in_(reviewer_actions))
            if search:
                like = f"%{search}%"
                stmt = stmt.where(
                    or_(ReviewItem.comment_id.like(like), ReviewItem.original_comment.like(like))
                )
            filtered = stmt
            stmt = filtered.order_by(*_order_clauses(sort))
            total = session.execute(
                select(func.count()).select_from(filtered.order_by(None).subquery())
            ).scalar_one()
            page = session.execute(stmt.offset(offset).limit(limit)).scalars().all()
            for item in page:
                session.expunge(item)
            return list(page), int(total)

    def counts_by_status(self, reviewer_user_id: str | None) -> dict[str, int]:
        with Session(self.engine) as session:
            stmt = select(ReviewItem.review_status, func.count()).group_by(ReviewItem.review_status)
            if reviewer_user_id:
                stmt = stmt.where(ReviewItem.reviewer_user_id == reviewer_user_id)
            rows = session.execute(stmt).all()
        return {str(status or ""): int(n) for status, n in rows}

    def claim(self, comment_id, *, reviewer_id, display_name, user_id, expected_version, actor_role=None) -> ReviewItem:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            _check_version(item, expected_version)
            if item.review_status == "IN_REVIEW" and item.reviewer_id == reviewer_id:
                session.expunge(item)
                return item
            if item.review_status == "IN_REVIEW" and item.reviewer_id not in (None, reviewer_id):
                raise InvalidTransition(f"claimed by {item.reviewer_id}")
            prev_status, prev_assignee = item.review_status, item.reviewer_user_id
            item.review_status = next_status(item.review_status, "claim")
            item.reviewer_id = reviewer_id
            item.reviewer_display_name = display_name
            item.reviewer_user_id = user_id
            item.reviewer_started_at = item.reviewer_started_at or _now()
            _add_event(session, item, EVENT_CLAIMED, actor_user_id=user_id or reviewer_id, actor_role=actor_role,
                       from_status=prev_status, from_assignee=prev_assignee)
            _bump(item)
            session.commit()
            session.refresh(item)
            session.expunge(item)
            return item

    def send_as_is(self, comment_id, *, reviewer_id, display_name, user_id, expected_version, actor_role=None, attested=False) -> ReviewItem:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            _check_version(item, expected_version)
            if item.reviewer_action == "SEND_AS_IS" and item.review_status == "APPROVED":
                session.expunge(item)
                return item
            suggestion = (item.suggested_final_reply or "").strip()
            if not suggestion:
                raise InvalidTransition("SEND_AS_IS requires a suggested_response")
            prev_status, prev_assignee = item.review_status, item.reviewer_user_id
            self._ensure_claim(session, item, reviewer_id=reviewer_id, display_name=display_name)
            item.review_status = next_status(item.review_status, "SEND_AS_IS")
            item.reviewer_action = "SEND_AS_IS"
            item.reviewer_id = reviewer_id
            item.reviewer_display_name = display_name
            item.reviewer_user_id = user_id
            item.final_reply = suggestion
            item.ready_to_send = True
            item.reviewed_at = _now()
            item.outbound_status = "PENDING"
            item.outbound_idempotency_key = _idempotency_key(comment_id, "SEND_AS_IS", suggestion)
            _add_event(session, item, EVENT_SEND_AS_IS_ATTESTED if attested else EVENT_SEND_AS_IS, actor_user_id=user_id or reviewer_id, actor_role=actor_role,
                       from_status=prev_status, from_assignee=prev_assignee)
            _bump(item)
            session.commit()
            session.refresh(item)
            session.expunge(item)
            return item

    def edit_and_send(self, comment_id, *, reviewer_id, display_name, user_id, edited_response, expected_version, actor_role=None, attested=False) -> ReviewItem:
        edited = (edited_response or "").strip()
        if not edited:
            raise InvalidTransition("EDIT_AND_SEND requires non-empty edited_response")
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            _check_version(item, expected_version)
            original = item.suggested_final_reply
            if (
                item.reviewer_action == "EDIT_AND_SEND"
                and item.review_status == "EDITED_AND_APPROVED"
                and item.final_reply == edited
            ):
                session.expunge(item)
                return item
            prev_status, prev_assignee = item.review_status, item.reviewer_user_id
            self._ensure_claim(session, item, reviewer_id=reviewer_id, display_name=display_name)
            item.review_status = next_status(item.review_status, "EDIT_AND_SEND")
            item.reviewer_action = "EDIT_AND_SEND"
            item.reviewer_id = reviewer_id
            item.reviewer_display_name = display_name
            item.reviewer_user_id = user_id
            item.reviewer_edit = edited
            item.final_reply = edited
            item.suggested_final_reply = original
            item.ready_to_send = True
            item.reviewed_at = _now()
            item.outbound_status = "PENDING"
            item.outbound_idempotency_key = _idempotency_key(comment_id, "EDIT_AND_SEND", edited)
            _add_event(session, item, EVENT_EDIT_AND_SEND_ATTESTED if attested else EVENT_EDIT_AND_SEND, actor_user_id=user_id or reviewer_id, actor_role=actor_role,
                       from_status=prev_status, from_assignee=prev_assignee)
            _bump(item)
            session.commit()
            session.refresh(item)
            session.expunge(item)
            return item

    def mark_no_response_needed(self, comment_id, *, reviewer_id, display_name, user_id, reason, expected_version, actor_role=None) -> ReviewItem:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            _check_version(item, expected_version)
            if item.reviewer_action == "NO_RESPONSE_NEEDED" and item.review_status == "NO_REPLY":
                session.expunge(item)
                return item
            prev_status, prev_assignee = item.review_status, item.reviewer_user_id
            item.review_status = next_status(item.review_status, "NO_RESPONSE_NEEDED")
            item.reviewer_action = "NO_RESPONSE_NEEDED"
            item.reviewer_id = reviewer_id
            item.reviewer_display_name = display_name
            item.reviewer_user_id = user_id
            item.no_response_reason = reason
            item.ready_to_send = False
            item.final_reply = None
            item.reviewed_at = _now()
            item.outbound_status = "NOT_APPLICABLE"
            _add_event(session, item, EVENT_NO_RESPONSE_NEEDED, actor_user_id=user_id or reviewer_id, actor_role=actor_role,
                       from_status=prev_status, from_assignee=prev_assignee)
            _bump(item)
            session.commit()
            session.refresh(item)
            session.expunge(item)
            return item

    def _ensure_claim(self, session: Session, item: ReviewItem, *, reviewer_id: str, display_name: str) -> None:
        """Verbatim move of service.py's ReviewWorkflowService._ensure_claim."""
        if item.review_status == "IN_REVIEW":
            if item.reviewer_id not in (None, reviewer_id):
                raise InvalidTransition(f"claimed by {item.reviewer_id}")
            return
        item.review_status = next_status(item.review_status, "claim")
        item.reviewer_id = reviewer_id
        item.reviewer_display_name = display_name
        item.reviewer_started_at = item.reviewer_started_at or _now()

    def assign_batch(
        self,
        comment_ids: list[str],
        reviewer_user_id: str,
        *,
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ) -> AssignmentResult:
        assigned: list[str] = []
        skipped_decided: list[str] = []
        skipped_not_reviewable: list[str] = []
        not_found: list[str] = []
        with Session(self.engine) as session:
            for comment_id in comment_ids:
                # Compare-and-set: the UPDATE only applies if the assignee is STILL what we just read (and the row is still undecided and reviewable), so a
                # concurrent assignment can never be overwritten from a stale read. On a lost race we re-read and decide again (bounded). The audit event is
                # staged AFTER the winning UPDATE in the same transaction, so its from-assignee, its timestamp order and row_version all follow the real
                # commit order.
                outcome = "stale"
                for _attempt in range(10):
                    item = session.get(ReviewItem, comment_id, populate_existing=True)
                    if item is None:
                        outcome = "not_found"
                        break
                    if item.reviewer_action is not None:
                        outcome = "decided"
                        break
                    if item.review_status not in _ASSIGNABLE_STATUSES:
                        outcome = "not_reviewable"
                        break
                    prev_assignee = item.reviewer_user_id
                    if prev_assignee == reviewer_user_id:
                        outcome = "unchanged"
                        break
                    prev_match = ReviewItem.reviewer_user_id.is_(None) if prev_assignee is None else ReviewItem.reviewer_user_id == prev_assignee
                    changed = session.execute(
                        update(ReviewItem)
                        .where(
                            ReviewItem.comment_id == comment_id, prev_match, ReviewItem.reviewer_action.is_(None),
                            ReviewItem.review_status == item.review_status,
                        )
                        .values(reviewer_user_id=reviewer_user_id, row_version=ReviewItem.row_version + 1, updated_at=_now())
                        .execution_options(synchronize_session=False)
                    ).rowcount
                    if changed:
                        set_committed_value(item, "reviewer_user_id", reviewer_user_id)       # the event records the NEW assignee; not a pending ORM change
                        _add_event(
                            session, item, EVENT_ASSIGNED if prev_assignee is None else EVENT_REASSIGNED,
                            actor_user_id=actor_user_id, actor_role=actor_role,
                            from_status=item.review_status, from_assignee=prev_assignee,
                        )
                        outcome = "assigned"
                        break
                if outcome == "not_found":
                    not_found.append(comment_id)
                elif outcome == "decided":
                    skipped_decided.append(comment_id)
                elif outcome == "not_reviewable":
                    skipped_not_reviewable.append(comment_id)
                elif outcome in ("assigned", "unchanged"):
                    assigned.append(comment_id)
                else:                                                   # still contended after the bounded retries: report it, never overwrite blindly
                    raise RuntimeError(f"assignment of {comment_id} kept losing to concurrent updates")
            session.commit()
        return AssignmentResult(
            assigned=tuple(assigned),
            skipped_already_decided=tuple(skipped_decided),
            skipped_not_reviewable=tuple(skipped_not_reviewable),
            not_found=tuple(not_found),
        )

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
        created_from: dt.datetime | None = None,
        created_to: dt.datetime | None = None,
        statuses: tuple[str, ...] | None = None,
        sort: str = "oldest",
        actor_user_id: str | None = None,
        actor_role: str | None = None,
    ) -> FilterAssignmentResult:
        """Server-side count/filter assignment (see the contract). Eligibility = exactly list_assignable(unassigned_only=True): reviewable status,
        no decision, no assignee. One transaction: a single `UPDATE ... WHERE comment_id IN (<ordered, limited candidate select>) AND still unassigned
        AND still undecided RETURNING` claims rows atomically (PostgreSQL additionally locks the candidates with FOR UPDATE SKIP LOCKED so two admins
        never wait on or double-claim the same rows; SQLite serialises writers). The same transaction appends one ASSIGNED event per claimed row."""
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
            raise ValueError("count must be a positive integer or None")
        max_rounds = 25

        def candidates():
            stmt = select(ReviewItem.comment_id).where(
                ReviewItem.review_status.in_(_ASSIGNABLE_STATUSES),
                ReviewItem.reviewer_action.is_(None),
                ReviewItem.reviewer_user_id.is_(None),
            )
            if statuses:
                stmt = stmt.where(ReviewItem.review_status.in_(statuses))
            if source_file:
                stmt = stmt.where(ReviewItem.source_file == source_file)
            stmt = _apply_classification_filters(
                stmt, primary_intents=primary_intents, topic_subtypes=topic_subtypes, sentiments=sentiments,
                priorities=priorities, created_from=created_from, created_to=created_to,
            )
            if search:
                like = f"%{search}%"
                stmt = stmt.where(or_(ReviewItem.comment_id.like(like), ReviewItem.original_comment.like(like)))
            return stmt

        with Session(self.engine) as session, session.begin():
            matched_before = int(session.execute(select(func.count()).select_from(candidates().subquery())).scalar_one())
            claimed: list[tuple[str, str]] = []
            for _ in range(max_rounds):
                want = None if count is None else count - len(claimed)
                if want is not None and want <= 0:
                    break
                pick = candidates().order_by(*_order_clauses(sort))
                if want is not None:
                    pick = pick.limit(want)
                pick = pick.with_for_update(skip_locked=True)          # no-op on SQLite
                # Two steps in ONE transaction: first select-and-lock exactly the rows to claim, then update exactly those ids. (Embedding the LIMIT ... FOR UPDATE SKIP LOCKED
                # select as a sub-select of the UPDATE is NOT safe on PostgreSQL: the planner may re-evaluate it per row, so `LIMIT 1` could claim every eligible row.
                # Verified on PostgreSQL 16; SQLite ignores FOR UPDATE and hid the problem.) The row locks held until commit keep the claim atomic, and the UPDATE keeps its
                # compare-and-set conditions as a second line of defence.
                ids = list(session.execute(pick).scalars().all())
                if not ids:
                    break
                now = _now()
                rows = []
                for start in range(0, len(ids), 1000):
                    rows.extend(session.execute(
                        update(ReviewItem)
                        .where(
                            ReviewItem.comment_id.in_(ids[start:start + 1000]),
                            ReviewItem.reviewer_user_id.is_(None),
                            ReviewItem.reviewer_action.is_(None),
                        )
                        .values(reviewer_user_id=reviewer_user_id, row_version=ReviewItem.row_version + 1, updated_at=now)
                        .returning(ReviewItem.comment_id, ReviewItem.review_status)
                        .execution_options(synchronize_session=False)
                    ).all())
                if not rows:
                    continue        # every picked row was claimed by someone else between select and update (possible without row locks, e.g. SQLite): select again (bounded by max_rounds)
                claimed.extend((cid, status) for cid, status in rows)
            for cid, status in claimed:
                fields = build_event_fields(
                    review_item_id=cid, event_type=EVENT_ASSIGNED, actor_user_id=actor_user_id, actor_role=actor_role,
                    from_status=status, to_status=status, from_assignee_user_id=None, to_assignee_user_id=reviewer_user_id, created_at=_now(),
                )
                session.add(ReviewItemEvent(**fields))
            remaining = int(session.execute(select(func.count()).select_from(candidates().subquery())).scalar_one())
        assigned = len(claimed)
        shortfall = 0 if count is None else max(0, count - assigned)
        return FilterAssignmentResult(matched_before, count, assigned, shortfall, remaining)

    def list_events(self, comment_id: str) -> list[ReviewEvent]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(ReviewItemEvent)
                .where(ReviewItemEvent.review_item_id == comment_id)
                .order_by(ReviewItemEvent.created_at, ReviewItemEvent.event_id)
            ).scalars().all()
            return [
                event_from_fields({c.name: getattr(r, c.name) for c in ReviewItemEvent.__table__.columns})
                for r in rows
            ]

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
        with Session(self.engine) as session:
            stmt = select(ReviewItem).where(
                ReviewItem.review_status.in_(_ASSIGNABLE_STATUSES),
                ReviewItem.reviewer_action.is_(None),
            )
            if statuses:
                stmt = stmt.where(ReviewItem.review_status.in_(statuses))
            if unassigned_only:
                stmt = stmt.where(ReviewItem.reviewer_user_id.is_(None))
            if reviewer_user_id:
                stmt = stmt.where(ReviewItem.reviewer_user_id == reviewer_user_id)
            if source_file:
                stmt = stmt.where(ReviewItem.source_file == source_file)
            stmt = _apply_classification_filters(
                stmt,
                primary_intents=primary_intents,
                topic_subtypes=topic_subtypes,
                sentiments=sentiments,
                priorities=priorities,
                created_from=created_from,
                created_to=created_to,
            )
            if search:
                like = f"%{search}%"
                stmt = stmt.where(or_(ReviewItem.comment_id.like(like), ReviewItem.original_comment.like(like)))
            filtered = stmt
            total = session.execute(
                select(func.count()).select_from(filtered.order_by(None).subquery())
            ).scalar_one()
            page = session.execute(
                filtered.order_by(*_order_clauses(sort)).offset(offset).limit(limit)
            ).scalars().all()
            for item in page:
                session.expunge(item)
            return list(page), int(total)

    def distinct_source_files(self) -> list[str]:
        with Session(self.engine) as session:
            return sorted({row[0] for row in session.execute(select(ReviewItem.source_file).distinct()).all()})

    def unassigned_counts_by_source_file(self) -> dict[str, int]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(ReviewItem.source_file, func.count())
                .where(~ReviewItem.review_status.in_(_NOT_ASSIGNABLE_STATUSES), ReviewItem.reviewer_user_id.is_(None))
                .group_by(ReviewItem.source_file)
            ).all()
        return {sf: int(n) for sf, n in rows}

    def count_all(self) -> int:
        return count_rows(self.engine)

    def list_ordered(self, comment_ids: list[str] | None) -> list[ReviewItem]:
        with Session(self.engine) as session:
            q = session.query(ReviewItem)
            if comment_ids is not None:
                q = q.filter(ReviewItem.comment_id.in_(list(comment_ids)))
            items = q.order_by(ReviewItem.source_row_index, ReviewItem.comment_id).all()
            for item in items:
                session.expunge(item)
            return items

    def get_many(self, comment_ids: list[str]) -> dict[str, ReviewItem]:
        if not comment_ids:
            return {}
        with Session(self.engine) as session:
            rows = session.query(ReviewItem).filter(ReviewItem.comment_id.in_(comment_ids)).all()
            for row in rows:
                session.expunge(row)
            return {r.comment_id: r for r in rows}

    def comment_ids_for_source_file(self, source_file: str) -> list[str]:
        with Session(self.engine) as session:
            return [r[0] for r in session.execute(select(ReviewItem.comment_id).where(ReviewItem.source_file == source_file)).all()]

    def reviewer_assignment_counts(self) -> dict[str, ReviewerAssignmentCounts]:
        with Session(self.engine) as session:
            total_rows = session.execute(
                select(ReviewItem.reviewer_user_id, func.count())
                .where(ReviewItem.reviewer_user_id.is_not(None))
                .group_by(ReviewItem.reviewer_user_id)
            ).all()
            pending_rows = session.execute(
                select(ReviewItem.reviewer_user_id, func.count())
                .where(ReviewItem.reviewer_user_id.is_not(None), ReviewItem.reviewer_action.is_(None))
                .group_by(ReviewItem.reviewer_user_id)
            ).all()
        pending = {uid: int(n) for uid, n in pending_rows}
        return {uid: ReviewerAssignmentCounts(total=int(n), pending=pending.get(uid, 0)) for uid, n in total_rows}

    def get_dedup_candidates(self, comment_ids: list[str]) -> list[DedupCandidate]:
        if not comment_ids:
            return []
        with Session(self.engine) as session:
            rows = session.execute(
                select(
                    ReviewItem.comment_id,
                    ReviewItem.source_file,
                    ReviewItem.source_row_index,
                    ReviewItem.original_comment,
                    ReviewItem.review_status,
                    ReviewItem.reviewer_action,
                    ReviewItem.dedupe_canonical_id,
                ).where(ReviewItem.comment_id.in_(comment_ids))
            ).all()
        return [
            DedupCandidate(
                comment_id=cid, source_file=source_file, source_row_index=row_idx,
                original_comment=text, review_status=status, reviewer_action=reviewer_action,
                dedupe_canonical_id=canon,
            )
            for cid, source_file, row_idx, text, status, reviewer_action, canon in rows
        ]

    def find_existing_dedup_canonical(self, source_file: str, content_dedupe_key: str) -> str | None:
        with Session(self.engine) as session:
            existing = session.execute(
                select(ReviewItem.dedupe_canonical_id).where(
                    ReviewItem.source_file == source_file,
                    ReviewItem.content_dedupe_key == content_dedupe_key,
                    ReviewItem.dedupe_canonical_id.is_not(None),
                )
            ).first()
        return existing[0] if existing is not None else None

    def set_dedupe_canonical_if_unset(self, comment_id: str, content_dedupe_key: str | None) -> bool:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            if item.dedupe_canonical_id is not None:
                return False
            item.content_dedupe_key = content_dedupe_key
            item.dedupe_canonical_id = comment_id
            session.commit()
            return True

    def mark_dedupe_duplicate(self, comment_id: str, content_dedupe_key: str, canonical_id: str) -> None:
        with Session(self.engine) as session:
            item = session.get(ReviewItem, comment_id)
            if item is None:
                raise ItemNotFound(comment_id)
            item.content_dedupe_key = content_dedupe_key
            item.dedupe_canonical_id = canonical_id
            item.review_status = "DUPLICATE_CONTENT"
            item.outbound_status = "NOT_APPLICABLE"
            session.commit()

    def review_statuses(self, comment_ids: list[str]) -> dict[str, str]:
        if not comment_ids:
            return {}
        with Session(self.engine) as session:
            rows = session.execute(
                select(ReviewItem.comment_id, ReviewItem.review_status).where(
                    ReviewItem.comment_id.in_(list(comment_ids))
                )
            ).all()
        return {comment_id: status for comment_id, status in rows}

    def source_file_status_fields(self, source_file: str) -> list[SourceFileStatusFields]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(
                    ReviewItem.review_status,
                    ReviewItem.generation_routing_decision,
                    ReviewItem.generation_status,
                ).where(ReviewItem.source_file == source_file)
            ).all()
        return [
            SourceFileStatusFields(
                review_status=review_status,
                generation_routing_decision=routing_decision,
                generation_status=generation_status,
            )
            for review_status, routing_decision, generation_status in rows
        ]

    def upsert_ingest_batch(self, rows: list[dict]) -> dict:
        """Phase M4C.1 -- verbatim move of repository.py::upsert_ingest_items.
        One Session, one commit for the whole batch (unchanged whole-call
        atomicity). Lock definition preserved exactly: `bool(reviewer_action)`
        (not `is not None` -- an empty string would not count as locked,
        matching the original), OR review_status not in ("UNPROCESSED", None)."""
        inserted = 0
        updated = 0
        skipped = 0
        with Session(self.engine) as session:
            for row in rows:
                existing = session.get(ReviewItem, row["comment_id"])
                if existing is None:
                    payload = {k: v for k, v in row.items() if hasattr(ReviewItem, k)}
                    session.add(ReviewItem(**payload))
                    inserted += 1
                    continue
                locked = bool(existing.reviewer_action) or existing.review_status not in ("UNPROCESSED", None)
                if locked:
                    skipped += 1
                    continue
                for key in INGEST_OWNED_FIELDS:
                    if key in row:
                        setattr(existing, key, row[key])
                updated += 1
            session.commit()
        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def load_for_prefilter(self, comment_ids: list[str] | None = None) -> pd.DataFrame:
        """Phase M4C.2 -- verbatim move of repository.py::load_for_prefilter.
        No ORDER BY in the original -- none added here."""
        cols = [
            ReviewItem.comment_id,
            ReviewItem.original_comment,
            ReviewItem.visitor_email,
            ReviewItem.visitor_name,
            ReviewItem.comment_created_at,
            ReviewItem.is_exact_duplicate_of,
        ]
        stmt = select(*cols)
        if comment_ids is not None:
            stmt = stmt.where(ReviewItem.comment_id.in_(list(comment_ids)))
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        df = pd.DataFrame(rows, columns=[c.key for c in cols])
        df["comment_created_at"] = pd.to_datetime(df["comment_created_at"])
        return df

    def apply_prefilter_batch(self, results: list[dict]) -> dict:
        """Phase M4C.2 -- verbatim move of repository.py::apply_prefilter_results,
        adapted to iterate list[dict] instead of DataFrame.iterrows() (no
        behavior change: same per-row dict-like access either way). `now`
        is computed ONCE, shared across the whole batch -- exactly as the
        original. No review_status guard -- exactly as the original.
        Never touches row_version. Raises KeyError (whole call aborts,
        nothing committed) on any missing comment_id -- exactly as the
        original."""
        now = dt.datetime.utcnow()
        excluded = 0
        passed = 0
        with Session(self.engine) as session:
            for row in results:
                item = session.get(ReviewItem, row["comment_id"])
                if item is None:
                    raise KeyError(f"comment_id {row['comment_id']} not found in review_items")

                item.is_prefiltered_junk = bool(row["is_prefiltered_junk"])
                item.prefilter_reason = row["prefilter_reason"]
                item.processed_at = now

                if row["is_prefiltered_junk"]:
                    item.primary_intent = "SPAM"
                    item.routing_decision = "EXCLUDE"
                    item.review_status = "EXCLUDED"
                    item.exclusion_reason = row["prefilter_reason"]
                    excluded += 1
                else:
                    item.primary_intent = None
                    item.routing_decision = None
                    item.review_status = "UNPROCESSED"
                    item.exclusion_reason = None
                    passed += 1
            session.commit()
        return {"excluded": excluded, "passed": passed}

    def load_for_classification(self) -> pd.DataFrame:
        """Phase M4C.3 -- verbatim move of repository.py::load_for_classification.
        Eligibility: review_status == "UNPROCESSED" AND is_prefiltered_junk
        IS FALSE (strict -- excludes NULL, not just != True). No
        comment_ids parameter -- matches the original exactly; callers
        filter client-side. No ORDER BY in the original -- none added here."""
        cols = [
            ReviewItem.comment_id,
            ReviewItem.original_comment,
            ReviewItem.original_title,
            ReviewItem.original_type,
            ReviewItem.age_group,
            ReviewItem.platform,
            ReviewItem.entry_id,
            ReviewItem.lemma,
        ]
        stmt = select(*cols).where(
            ReviewItem.review_status == "UNPROCESSED",
            ReviewItem.is_prefiltered_junk.is_(False),
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        return pd.DataFrame(rows, columns=[c.key for c in cols])

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
        """Phase M4C.3 -- verbatim move of repository.py::apply_classification_results.
        `now` computed ONCE, shared across the whole batch. Skip guard:
        ai_primary_intent already set OR review_status != "UNPROCESSED".
        Model/prompt/pipeline/run metadata + processed_at written on
        BOTH success and failure paths, for every non-skipped row. Never
        touches row_version. Raises KeyError (whole call aborts, nothing
        committed) on any missing comment_id."""
        now = dt.datetime.utcnow()
        classified = 0
        failed = 0
        skipped = 0

        with Session(self.engine) as session:
            for result in results:
                item = session.get(ReviewItem, result["comment_id"])
                if item is None:
                    raise KeyError(f"comment_id {result['comment_id']} not found in review_items")

                if item.ai_primary_intent is not None or item.review_status != "UNPROCESSED":
                    skipped += 1
                    continue

                item.model_name = model_name
                item.model_version = model_version
                item.prompt_version = prompt_version
                item.pipeline_version = pipeline_version
                item.inference_run_id = inference_run_id
                item.processed_at = now

                if result["success"]:
                    item.ai_primary_intent = result["ai_primary_intent"]
                    item.ai_topic_subtype = result["ai_topic_subtype"]
                    item.ai_subcategory = result["ai_subcategory"]
                    item.ai_secondary_intent = result["ai_secondary_intent"]
                    item.ai_routing_decision = result["ai_routing_decision"]
                    item.ai_classification_reason = result["ai_classification_reason"]
                    item.ai_confidence = result["ai_confidence"]
                    item.ai_classification_source = result.get("ai_classification_source")
                    item.ai_needs_review = result.get("ai_needs_review")
                    if result.get("ai_sentiment"):
                        item.ai_sentiment = result["ai_sentiment"]
                        item.sentiment = item.sentiment or result["ai_sentiment"]

                    item.primary_intent = result["ai_primary_intent"]
                    item.topic_subtype = result["ai_topic_subtype"]
                    item.subcategory = result["ai_subcategory"]
                    item.secondary_intent = result["ai_secondary_intent"]
                    item.classification_source = result.get("ai_classification_source")
                    item.needs_review = result.get("ai_needs_review")
                    item.routing_decision = result["ai_routing_decision"]

                    if result["ai_routing_decision"] == "EXCLUDE":
                        item.exclusion_reason = result.get("routing_reason_code")
                    else:
                        item.exclusion_reason = None

                    item.review_status = "CLASSIFIED"
                    classified += 1
                else:
                    item.ai_routing_decision = "REVIEW_ONLY"
                    item.ai_classification_reason = result["ai_classification_reason"]
                    # review_status intentionally left as "UNPROCESSED" -- retry-eligible.
                    failed += 1

            session.commit()

        return {"classified": classified, "failed": failed, "skipped_already_classified": skipped}


class PostgresReviewSessionRepository:
    def __init__(self, engine):
        self.engine = engine

    def _now(self) -> dt.datetime:
        return dt.datetime.utcnow()

    def create(self, *, original_filename: str, source_file: str, upload_token: str) -> ReviewSession:
        import uuid

        session_id = uuid.uuid4().hex
        now = self._now()
        row = ReviewSession(
            session_id=session_id,
            original_filename=original_filename,
            source_file=source_file,
            upload_token=upload_token,
            status="uploaded",
            created_at=now,
            updated_at=now,
        )
        with Session(self.engine, expire_on_commit=False) as db:
            db.add(row)
            db.commit()
        return row

    def get(self, session_id: str) -> ReviewSession | None:
        with Session(self.engine) as db:
            row = db.get(ReviewSession, session_id)
            if row is not None:
                db.expunge(row)
            return row

    def get_active(self) -> ReviewSession | None:
        active_statuses = ("uploaded", "processing", "ready", "completed")
        with Session(self.engine) as db:
            row = db.execute(
                select(ReviewSession)
                .where(ReviewSession.status.in_(active_statuses))
                .order_by(ReviewSession.created_at.desc())
            ).scalars().first()
            if row is not None:
                db.expunge(row)
            return row

    def set_status(self, session_id: str, status: str) -> None:
        with Session(self.engine) as db:
            row = db.get(ReviewSession, session_id)
            if row is None:
                return
            row.status = status
            row.updated_at = self._now()
            db.commit()

    def compute_counts(self, source_file: str) -> SessionCounts:
        """Verbatim move of sessions.py::compute_session_counts -- SQL-side
        conditional aggregates in ONE round trip. This exact shape was a
        deliberate fix for an observed ~7.5s regression when counts were
        computed by fetching every row; it must not be reintroduced by a
        future refactor (including a Mongo port) fetching rows instead of
        aggregating server-side."""
        from sqlalchemy import case

        excluded_expr = ReviewItem.review_status == "EXCLUDED"
        duplicate_expr = ReviewItem.review_status == "DUPLICATE_CONTENT"
        completed_expr = ReviewItem.review_status.not_in(("EXCLUDED", "DUPLICATE_CONTENT")) & (
            ReviewItem.reviewer_action.is_not(None)
        )
        auto_resolved_expr = ReviewItem.review_status == "NO_RESPONSE_RECOMMENDED"
        with Session(self.engine) as db:
            total, excluded, duplicate_content, completed, auto_resolved = db.execute(
                select(
                    func.count(),
                    func.sum(case((excluded_expr, 1), else_=0)),
                    func.sum(case((duplicate_expr, 1), else_=0)),
                    func.sum(case((completed_expr, 1), else_=0)),
                    func.sum(case((auto_resolved_expr, 1), else_=0)),
                ).where(ReviewItem.source_file == source_file)
            ).one()
        total = int(total or 0)
        excluded = int(excluded or 0)
        duplicate_content = int(duplicate_content or 0)
        completed = int(completed or 0)
        auto_resolved = int(auto_resolved or 0)
        canonical_total = total - excluded - duplicate_content
        # NO_RESPONSE_RECOMMENDED is pending human review, not completed
        # work: it is deliberately not added to `completed`.
        completed_or_auto = completed
        remaining = max(0, canonical_total - completed_or_auto)
        return SessionCounts(
            total=total,
            excluded=excluded,
            duplicate_content=duplicate_content,
            canonical_total=canonical_total,
            completed=completed_or_auto,
            remaining=remaining,
        )


class PostgresProcessingJobRepository:
    def __init__(self, engine):
        self.engine = engine

    def _now(self) -> dt.datetime:
        return dt.datetime.utcnow()

    def create(self, session_id: str) -> ProcessingJob:
        import uuid

        from rcs.review.sessions import STAGE_PREPARE

        job_id = uuid.uuid4().hex
        now = self._now()
        row = ProcessingJob(
            job_id=job_id,
            session_id=session_id,
            status="running",
            current_stage=STAGE_PREPARE,
            rows_done=0,
            rows_total=None,
            started_at=now,
            updated_at=now,
        )
        with Session(self.engine, expire_on_commit=False) as db:
            db.add(row)
            db.commit()
        return row

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
    ) -> None:
        with Session(self.engine) as db:
            row = db.get(ProcessingJob, job_id)
            if row is None:
                return
            if status is not None:
                row.status = status
            if current_stage is not None:
                row.current_stage = current_stage
            if rows_done is not None:
                row.rows_done = rows_done
            if rows_total is not None:
                row.rows_total = rows_total
            if error_message is not None:
                row.error_message = error_message
            row.updated_at = self._now()
            if finished:
                row.finished_at = row.updated_at
            db.commit()

    def get(self, job_id: str) -> ProcessingJob | None:
        with Session(self.engine) as db:
            row = db.get(ProcessingJob, job_id)
            if row is not None:
                db.expunge(row)
            return row

    def latest_for_session(self, session_id: str) -> ProcessingJob | None:
        with Session(self.engine) as db:
            row = db.execute(
                select(ProcessingJob)
                .where(ProcessingJob.session_id == session_id)
                .order_by(ProcessingJob.started_at.desc())
            ).scalars().first()
            if row is not None:
                db.expunge(row)
            return row

    def list_running(self) -> list[ProcessingJob]:
        with Session(self.engine) as db:
            rows = db.execute(select(ProcessingJob).where(ProcessingJob.status == "running")).scalars().all()
            for row in rows:
                db.expunge(row)
            return list(rows)


class PostgresAnalyticsRepository:
    """Phase M4C.5 -- verbatim move of rcs.review.analytics's
    compute_operational_analytics/_trend_series query bodies. See that
    module's docstring: it keeps only the pure, backend-neutral helpers
    (range_bounds/_bucket_age/_complete_counts/PENDING/OUTCOME_ACTIONS/
    OVERDUE_DAYS) imported here rather than duplicated, since they have
    no DB dependency and are not "the service layer" -- the same
    precedent already set by importing next_status from workflow.py."""

    def __init__(self, engine):
        self.engine = engine

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
    ) -> dict:
        with Session(self.engine) as session:
            now = now_naive(now)
            start, end = range_bounds(range_key, now, start=custom_start, end=custom_end)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            overdue_cut = now - dt.timedelta(days=OVERDUE_DAYS)

            scope = [ReviewItem.source_file == source_file] if source_file else []
            if reviewer_user_id:
                scope.append(ReviewItem.reviewer_user_id == reviewer_user_id)

            pending_q = ReviewItem.review_status.in_(PENDING)
            pending = int(session.execute(select(func.count()).where(pending_q, *scope)).scalar_one() or 0)
            urgent = int(
                session.execute(
                    select(func.count()).where(pending_q, ReviewItem.review_priority == 1, *scope)
                ).scalar_one()
                or 0
            )
            overdue = int(
                session.execute(
                    select(func.count()).where(
                        pending_q, ReviewItem.ingested_at.is_not(None), ReviewItem.ingested_at <= overdue_cut, *scope
                    )
                ).scalar_one()
                or 0
            )
            oldest_pending = session.execute(select(func.min(ReviewItem.ingested_at)).where(pending_q, *scope)).scalar_one()
            oldest_age = int((now - oldest_pending).total_seconds()) if oldest_pending else None

            incoming_q = select(func.count()).where(ReviewItem.ingested_at.is_not(None), *scope)
            reviewed_q = select(func.count()).where(ReviewItem.reviewed_at.is_not(None), *scope)
            if start is not None:
                incoming_q = incoming_q.where(ReviewItem.ingested_at >= start)
                reviewed_q = reviewed_q.where(ReviewItem.reviewed_at >= start)
            incoming_q = incoming_q.where(ReviewItem.ingested_at <= end)
            reviewed_q = reviewed_q.where(ReviewItem.reviewed_at <= end)
            incoming = int(session.execute(incoming_q).scalar_one() or 0)
            reviewed = int(session.execute(reviewed_q).scalar_one() or 0)

            duration_rows = session.execute(
                select(ReviewItem.reviewer_started_at, ReviewItem.reviewed_at).where(
                    ReviewItem.reviewer_started_at.is_not(None),
                    ReviewItem.reviewed_at.is_not(None),
                    ReviewItem.reviewed_at <= end,
                    *([ReviewItem.reviewed_at >= start] if start is not None else []),
                    *scope,
                )
            ).all()
            durations = [
                (done - started).total_seconds()
                for started, done in duration_rows
                if started and done and done >= started
            ]
            avg_seconds = round(sum(durations) / len(durations), 1) if durations else None

            action_rows = session.execute(
                select(ReviewItem.reviewer_action, func.count()).where(
                    ReviewItem.reviewer_action.in_(OUTCOME_ACTIONS),
                    ReviewItem.reviewed_at.is_not(None),
                    ReviewItem.reviewed_at <= end,
                    *([ReviewItem.reviewed_at >= start] if start is not None else []),
                    *scope,
                ).group_by(ReviewItem.reviewer_action)
            ).all()
            outcomes = {key: 0 for key in OUTCOME_ACTIONS}
            for action, n in action_rows:
                outcomes[str(action)] = int(n)
            outcome_total = sum(outcomes.values())

            today_actions = {key: 0 for key in OUTCOME_ACTIONS}
            today_action_rows = session.execute(
                select(ReviewItem.reviewer_action, func.count()).where(
                    ReviewItem.reviewer_action.in_(OUTCOME_ACTIONS),
                    ReviewItem.reviewed_at >= today_start,
                    ReviewItem.reviewed_at <= now,
                    *scope,
                ).group_by(ReviewItem.reviewer_action)
            ).all()
            for action, n in today_action_rows:
                today_actions[str(action)] = int(n)
            reviewed_today = int(
                session.execute(
                    select(func.count()).where(ReviewItem.reviewed_at >= today_start, ReviewItem.reviewed_at <= now, *scope)
                ).scalar_one()
                or 0
            )
            reviewed_by_me = 0
            if reviewer_id:
                reviewed_by_me = int(
                    session.execute(
                        select(func.count()).where(
                            ReviewItem.reviewed_at >= today_start,
                            ReviewItem.reviewed_at <= now,
                            ReviewItem.reviewer_id == reviewer_id,
                            *scope,
                        )
                    ).scalar_one()
                    or 0
                )

            pending_times = session.execute(select(ReviewItem.ingested_at).where(pending_q, *scope)).scalars().all()
            health = {"under_day": 0, "one_two": 0, "three_seven": 0, "over_seven": 0}
            dated = 0
            for stamp in pending_times:
                bucket = _bucket_age(stamp, now)
                if not bucket:
                    continue
                health[bucket] += 1
                dated += 1

            class_filters = [ReviewItem.ingested_at.is_not(None), ReviewItem.ingested_at <= end, *scope]
            if start is not None:
                class_filters.append(ReviewItem.ingested_at >= start)
            intent_rows = session.execute(
                select(ReviewItem.primary_intent, func.count())
                .where(ReviewItem.primary_intent.is_not(None), *class_filters)
                .group_by(ReviewItem.primary_intent)
            ).all()
            topic_rows = session.execute(
                select(ReviewItem.topic_subtype, func.count())
                .where(ReviewItem.topic_subtype.is_not(None), *class_filters)
                .group_by(ReviewItem.topic_subtype)
            ).all()
            sentiment_rows = session.execute(
                select(ReviewItem.sentiment, func.count())
                .where(ReviewItem.sentiment.is_not(None), *class_filters)
                .group_by(ReviewItem.sentiment)
            ).all()

            trend = self._trend_series(
                session, start, end, now, range_key, source_file=source_file, reviewer_user_id=reviewer_user_id
            )
            quality_sample = outcome_total
            quality = {
                "sample": quality_sample,
                "send_as_is_rate": round(outcomes["SEND_AS_IS"] / quality_sample, 3) if quality_sample else None,
                "edited_rate": round(outcomes["EDIT_AND_SEND"] / quality_sample, 3) if quality_sample else None,
                "no_reply_rate": round(outcomes["NO_RESPONSE_NEEDED"] / quality_sample, 3) if quality_sample else None,
            }

            no_reply_all = int(
                session.execute(
                    select(func.count()).where(
                        ReviewItem.review_status == "NO_REPLY", *scope
                    )
                ).scalar_one()
                or 0
            )

            intent, intent_unmapped = _complete_counts(settings.PRIMARY_INTENTS, intent_rows)
            topic, topic_unmapped = _complete_counts(settings.TOPIC_SUBTYPES, topic_rows)
            sentiment, sentiment_unmapped = _complete_counts(settings.SENTIMENTS, sentiment_rows)
            unmapped = {
                "intent": intent_unmapped,
                "topic": topic_unmapped,
                "sentiment": sentiment_unmapped,
            }
            has_unmapped = any(unmapped[dim] for dim in unmapped)

            return {
                "range": range_key,
                "range_start": iso_app(start) if start else None,
                "range_end": iso_app(end) or end.isoformat(),
                "kpis": {
                    "incoming": incoming,
                    "reviewed": reviewed,
                    "pending": pending,
                    "urgent_pending": urgent,
                    "overdue_pending": overdue,
                    "no_reply": no_reply_all,
                    "avg_review_seconds": avg_seconds,
                    "avg_review_sample": len(durations),
                },
                "today": {
                    "reviewed_all": reviewed_today,
                    "reviewed_by_me": reviewed_by_me,
                    "actions": today_actions,
                },
                "outcomes": outcomes,
                "queue_health": {
                    **health,
                    "dated": dated,
                    "oldest_pending_age_seconds": oldest_age,
                },
                "quality": quality,
                "classification": {
                    "intent": intent,
                    "topic": topic,
                    "sentiment": sentiment,
                },
                "data_quality": {
                    "unmapped_classification": unmapped,
                    "note": (
                        "legacy_unmapped_labels_excluded_from_canonical_taxonomy"
                        if has_unmapped
                        else None
                    ),
                },
                "trend": trend,
                "availability": {
                    "avg_review_time": "ok" if durations else "insufficient_timestamps",
                    "outcomes": "ok" if outcome_total else "no_completed_actions",
                    "trend": "ok" if any(p["incoming"] or p["reviewed"] for p in trend) else "empty_window",
                    "classification": "ok" if any(n for _k, n in intent_rows) else "no_labels",
                },
            }

    def _trend_series(
        self,
        session: Session,
        start: dt.datetime | None,
        end: dt.datetime,
        now: dt.datetime,
        range_key: str,
        *,
        source_file: str | None = None,
        reviewer_user_id: str | None = None,
    ) -> list[dict]:
        scope = [ReviewItem.source_file == source_file] if source_file else []
        if reviewer_user_id:
            scope.append(ReviewItem.reviewer_user_id == reviewer_user_id)
        hourly = (range_key or "").lower() == "today"
        if (range_key or "").lower() == "custom" and start is not None:
            hourly = (end - start) <= dt.timedelta(days=2)
        fmt = "%Y-%m-%d %H" if hourly else "%Y-%m-%d"
        # Bucketed in Python, not SQL: SQLite's date-formatting SQL function
        # has no PostgreSQL equivalent. Fetching the raw timestamps with a plain,
        # dialect-neutral select() and bucketing client-side keeps this
        # database-portable without reaching for dialect-specific functions
        # like PostgreSQL's date_trunc(). Row volumes here (operational
        # review queue, not raw event logs) make this a non-issue.
        incoming_stamps = session.execute(
            select(ReviewItem.ingested_at).where(
                ReviewItem.ingested_at.is_not(None),
                ReviewItem.ingested_at <= end,
                *([ReviewItem.ingested_at >= start] if start is not None else []),
                *scope,
            )
        ).scalars().all()
        reviewed_stamps = session.execute(
            select(ReviewItem.reviewed_at).where(
                ReviewItem.reviewed_at.is_not(None),
                ReviewItem.reviewed_at <= end,
                *([ReviewItem.reviewed_at >= start] if start is not None else []),
                *scope,
            )
        ).scalars().all()
        incoming_map: dict[str, int] = {}
        for stamp in incoming_stamps:
            key = stamp.strftime(fmt)
            incoming_map[key] = incoming_map.get(key, 0) + 1
        reviewed_map: dict[str, int] = {}
        for stamp in reviewed_stamps:
            key = stamp.strftime(fmt)
            reviewed_map[key] = reviewed_map.get(key, 0) + 1
        keys: list[str] = []
        window_start = start or (now - dt.timedelta(days=13))
        if hourly:
            cursor = window_start.replace(minute=0, second=0, microsecond=0)
            while cursor <= end:
                keys.append(cursor.strftime("%Y-%m-%d %H"))
                cursor += dt.timedelta(hours=1)
                if len(keys) > 48:
                    break
        else:
            cursor = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
            last = end.replace(hour=0, minute=0, second=0, microsecond=0)
            while cursor <= last:
                keys.append(cursor.strftime("%Y-%m-%d"))
                cursor += dt.timedelta(days=1)
                if len(keys) > 62:
                    break
        return [
            {"bucket": key, "incoming": incoming_map.get(key, 0), "reviewed": reviewed_map.get(key, 0)}
            for key in keys
        ]


class PostgresUserRepository:
    """Phase M4D.1 -- verbatim move of the `users`-table persistence
    that used to live inline in rcs.review.auth / http_api.py /
    service.py. `"REVIEWER"` is hardcoded (not imported from
    rcs.review.auth.REVIEWER) for the same reason `_now`/`_bump` are
    duplicated elsewhere in this file: auth.py is about to import THIS
    module to delegate its own persistence, so this module must not
    import back from auth.py."""

    def __init__(self, engine):
        self.engine = engine

    def get_by_id(self, user_id: str) -> User | None:
        with Session(self.engine) as session:
            user = session.get(User, user_id)
            if user is not None:
                session.expunge(user)
            return user

    def get_by_email(self, email: str) -> User | None:
        with Session(self.engine) as session:
            user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if user is not None:
                session.expunge(user)
            return user

    def list_all(self) -> list[User]:
        with Session(self.engine) as session:
            rows = session.execute(select(User).order_by(User.created_at)).scalars().all()
            for row in rows:
                session.expunge(row)
            return list(rows)

    def list_active_reviewers(self) -> list[User]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(User).where(User.role == "REVIEWER", User.is_active.is_(True)).order_by(User.display_name)
            ).scalars().all()
            for row in rows:
                session.expunge(row)
            return list(rows)

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
        row = User(
            user_id=user_id,
            email=email,
            password_hash=password_hash,
            display_name=display_name,
            role=role,
            is_active=is_active,
            created_at=created_at,
            updated_at=updated_at,
        )
        with Session(self.engine, expire_on_commit=False) as session:
            existing = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if existing is not None:
                raise ValueError(f"a user with email {email!r} already exists")
            session.add(row)
            session.commit()
        return row

    def set_password_hash(self, user_id: str, password_hash: str, updated_at: dt.datetime) -> bool:
        with Session(self.engine) as session:
            result = session.execute(update(User).where(User.user_id == user_id).values(password_hash=password_hash, updated_at=updated_at))
            session.commit()
            return (result.rowcount or 0) == 1


class PostgresUserSessionRepository:
    """Phase M4D.2 -- verbatim move of the `user_sessions`-table
    persistence that used to live inline in rcs.review.auth. Token
    generation and active-account revalidation are NOT here -- they
    stay in auth.py, exactly as documented on UserSessionRepository."""

    def __init__(self, engine):
        self.engine = engine

    def create(self, *, session_token: str, user_id: str, created_at: dt.datetime, expires_at: dt.datetime) -> UserSession:
        row = UserSession(session_token=session_token, user_id=user_id, created_at=created_at, expires_at=expires_at)
        with Session(self.engine, expire_on_commit=False) as session:
            session.add(row)
            session.commit()
        return row

    def get_by_token(self, session_token: str) -> UserSession | None:
        with Session(self.engine) as session:
            sess = session.get(UserSession, session_token)
            if sess is not None:
                session.expunge(sess)
            return sess

    def delete_by_token(self, session_token: str) -> None:
        with Session(self.engine) as session:
            session.execute(delete(UserSession).where(UserSession.session_token == session_token))
            session.commit()

    def delete_expired(self, now: dt.datetime) -> int:
        with Session(self.engine) as session:
            result = session.execute(delete(UserSession).where(UserSession.expires_at <= now))
            session.commit()
            return result.rowcount or 0

    def delete_for_user(self, user_id: str) -> int:
        with Session(self.engine) as session:
            result = session.execute(delete(UserSession).where(UserSession.user_id == user_id))
            session.commit()
            return result.rowcount or 0


class PostgresPasswordResetTokenRepository:
    """`password_reset_tokens` table (see models.PasswordResetToken). Works on any SQLAlchemy dialect (PostgreSQL in production, SQLite in tests)."""

    def __init__(self, engine):
        self.engine = engine

    def create(self, *, token_hash: str, user_id: str, created_at: dt.datetime, expires_at: dt.datetime) -> None:
        with Session(self.engine) as session:
            session.add(PasswordResetToken(token_hash=token_hash, user_id=user_id, created_at=created_at, expires_at=expires_at, used_at=None))
            session.commit()

    def consume(self, token_hash: str, now: dt.datetime) -> str | None:
        with Session(self.engine) as session:
            # one conditional UPDATE: only the single caller that flips used_at from NULL wins, so a token can never be redeemed twice
            result = session.execute(
                update(PasswordResetToken)
                .where(PasswordResetToken.token_hash == token_hash, PasswordResetToken.used_at.is_(None), PasswordResetToken.expires_at > now)
                .values(used_at=now)
            )
            if (result.rowcount or 0) != 1:
                session.rollback()
                return None
            user_id = session.execute(select(PasswordResetToken.user_id).where(PasswordResetToken.token_hash == token_hash)).scalar_one()
            session.commit()
            return user_id

    def invalidate_for_user(self, user_id: str, now: dt.datetime) -> int:
        with Session(self.engine) as session:
            result = session.execute(
                update(PasswordResetToken).where(PasswordResetToken.user_id == user_id, PasswordResetToken.used_at.is_(None)).values(used_at=now)
            )
            session.commit()
            return result.rowcount or 0

    def delete_expired(self, now: dt.datetime) -> int:
        with Session(self.engine) as session:
            result = session.execute(delete(PasswordResetToken).where(PasswordResetToken.expires_at <= now))
            session.commit()
            return result.rowcount or 0


def _trace_row_to_record(row: AgentExecutionTrace) -> AgentAuditRecord:
    return AgentAuditRecord(
        execution_id=row.execution_id,
        review_item_id=row.review_item_id,
        created_at=row.created_at,
        grounding_mode=row.grounding_mode,
        official_retrieval_outcome=row.official_retrieval_outcome,
        official_result_count=row.official_result_count,
        official_grounding_available=row.official_grounding_available,
        style_retrieval_outcome=row.style_retrieval_outcome,
        style_result_count=row.style_result_count,
        logical_generation_count=row.logical_generation_count,
        refinement_attempted=row.refinement_attempted,
        initial_validation_passed=row.initial_validation_passed,
        initial_failure_code=row.initial_failure_code,
        final_validation_passed=row.final_validation_passed,
        final_failure_code=row.final_failure_code,
        generation_status=row.generation_status,
    )


class PostgresAgentAuditRepository:
    """Phase M5E. Satisfies `rcs.response_intelligence.audit.
    AgentAuditRepository` structurally (a plain Protocol -- no explicit
    inheritance, matching this codebase's convention). `record()` is
    idempotent on `execution_id` (the primary key): re-inserting the
    same id (e.g. a caller retrying a failed write) hits `IntegrityError`
    and is treated as "already recorded", not an error -- a genuinely
    new execution always has a fresh `execution_id` and always inserts
    a new row (append-only, never updated)."""

    def __init__(self, engine):
        self.engine = engine

    def record(self, record: AgentAuditRecord) -> None:
        row = AgentExecutionTrace(
            execution_id=record.execution_id,
            review_item_id=record.review_item_id,
            created_at=record.created_at,
            grounding_mode=record.grounding_mode,
            official_retrieval_outcome=record.official_retrieval_outcome,
            official_result_count=record.official_result_count,
            official_grounding_available=record.official_grounding_available,
            style_retrieval_outcome=record.style_retrieval_outcome,
            style_result_count=record.style_result_count,
            logical_generation_count=record.logical_generation_count,
            refinement_attempted=record.refinement_attempted,
            initial_validation_passed=record.initial_validation_passed,
            initial_failure_code=record.initial_failure_code,
            final_validation_passed=record.final_validation_passed,
            final_failure_code=record.final_failure_code,
            generation_status=record.generation_status,
        )
        with Session(self.engine) as session:
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()

    def list_for_review_item(self, review_item_id: str) -> tuple[AgentAuditRecord, ...]:
        with Session(self.engine) as session:
            rows = (
                session.query(AgentExecutionTrace)
                .filter(AgentExecutionTrace.review_item_id == review_item_id)
                .order_by(AgentExecutionTrace.created_at)
                .all()
            )
            return tuple(_trace_row_to_record(row) for row in rows)

    def get_by_execution_id(self, execution_id: str) -> AgentAuditRecord | None:
        with Session(self.engine) as session:
            row = session.get(AgentExecutionTrace, execution_id)
            return _trace_row_to_record(row) if row is not None else None


def _message_record(r: DirectMessage) -> DirectMessageRecord:
    return DirectMessageRecord(r.message_id, r.reviewer_user_id, r.sender_user_id, r.sender_role, r.body, r.client_message_id, r.created_at, r.read_at)


class PostgresDirectMessageRepository:
    """`direct_messages` table (see models.DirectMessage). Works on any SQLAlchemy dialect (PostgreSQL in production, SQLite in tests)."""

    def __init__(self, engine):
        self.engine = engine

    def _existing(self, session, sender_user_id, client_message_id):
        return session.execute(select(DirectMessage).where(DirectMessage.sender_user_id == sender_user_id, DirectMessage.client_message_id == client_message_id)).scalar_one_or_none()

    def add(self, *, message_id, reviewer_user_id, sender_user_id, sender_role, body, client_message_id, created_at):
        with Session(self.engine) as session:
            found = self._existing(session, sender_user_id, client_message_id)
            if found is not None:
                return _message_record(found), False
            row = DirectMessage(message_id=message_id, reviewer_user_id=reviewer_user_id, sender_user_id=sender_user_id, sender_role=sender_role, body=body,
                                client_message_id=client_message_id, created_at=created_at, read_at=None)
            session.add(row)
            try:
                session.commit()
            except IntegrityError:  # a concurrent identical request won the unique constraint -> return ITS row
                session.rollback()
                found = self._existing(session, sender_user_id, client_message_id)
                if found is None:
                    raise
                return _message_record(found), False
            return _message_record(row), True

    def list_conversation(self, reviewer_user_id, limit):
        with Session(self.engine) as session:
            rows = session.execute(
                select(DirectMessage).where(DirectMessage.reviewer_user_id == reviewer_user_id)
                .order_by(DirectMessage.created_at.desc(), DirectMessage.message_id.desc()).limit(limit)
            ).scalars().all()
            return [_message_record(r) for r in reversed(rows)]

    def mark_read(self, reviewer_user_id, sender_role, read_at):
        with Session(self.engine) as session, session.begin():
            result = session.execute(
                update(DirectMessage).where(DirectMessage.reviewer_user_id == reviewer_user_id, DirectMessage.sender_role == sender_role, DirectMessage.read_at.is_(None))
                .values(read_at=read_at)
            )
            return result.rowcount or 0

    def unread_count(self, reviewer_user_id, sender_role):
        with Session(self.engine) as session:
            return int(session.execute(select(func.count()).select_from(DirectMessage).where(
                DirectMessage.reviewer_user_id == reviewer_user_id, DirectMessage.sender_role == sender_role, DirectMessage.read_at.is_(None))).scalar_one())

    def summaries(self):
        with Session(self.engine) as session:
            counts = session.execute(
                select(
                    DirectMessage.reviewer_user_id,
                    func.sum(case((DirectMessage.sender_role == "REVIEWER", 1), else_=0)),
                    func.sum(case((DirectMessage.sender_role == "ADMIN", 1), else_=0)),
                ).where(DirectMessage.read_at.is_(None)).group_by(DirectMessage.reviewer_user_id)
            ).all()
            unread = {rid: (int(a or 0), int(b or 0)) for rid, a, b in counts}
            out = {}
            for rid in session.execute(select(DirectMessage.reviewer_user_id).distinct()).scalars().all():
                last = session.execute(select(DirectMessage).where(DirectMessage.reviewer_user_id == rid).order_by(DirectMessage.created_at.desc(), DirectMessage.message_id.desc()).limit(1)).scalar_one()
                a, b = unread.get(rid, (0, 0))
                out[rid] = ConversationSummary(rid, _message_record(last), a, b)
            return out


def _account_request_record(r: AccountRequest) -> AccountRequestRecord:
    return AccountRequestRecord(r.request_id, r.email, r.display_name, r.password_hash, r.status, r.created_at, r.decided_at, r.decided_by_user_id)


class PostgresAccountRequestRepository:
    """`account_requests` table (see models.AccountRequest). Works on any SQLAlchemy dialect."""

    def __init__(self, engine):
        self.engine = engine

    def create(self, *, request_id, email, display_name, password_hash, created_at):
        with Session(self.engine) as session:
            if session.execute(select(AccountRequest.request_id).where(AccountRequest.email == email)).first() is not None:
                return False
            session.add(AccountRequest(request_id=request_id, email=email, display_name=display_name, password_hash=password_hash, status="PENDING", created_at=created_at))
            try:
                session.commit()
            except IntegrityError:                    # a concurrent identical request won the unique e-mail constraint
                session.rollback()
                return False
            return True

    def get(self, request_id):
        with Session(self.engine) as session:
            r = session.get(AccountRequest, request_id)
            return None if r is None else _account_request_record(r)

    def list_pending(self, limit):
        with Session(self.engine) as session:
            rows = session.execute(select(AccountRequest).where(AccountRequest.status == "PENDING").order_by(AccountRequest.created_at, AccountRequest.request_id).limit(limit)).scalars().all()
            return [_account_request_record(r) for r in rows]

    def count_pending(self):
        with Session(self.engine) as session:
            return int(session.execute(select(func.count()).select_from(AccountRequest).where(AccountRequest.status == "PENDING")).scalar_one())

    def decide(self, request_id, status, decided_by_user_id, decided_at):
        with Session(self.engine) as session, session.begin():
            before = session.get(AccountRequest, request_id)
            if before is None:
                return None
            snapshot = _account_request_record(before)
            changed = session.execute(
                update(AccountRequest).where(AccountRequest.request_id == request_id, AccountRequest.status == "PENDING")
                .values(status=status, decided_at=decided_at, decided_by_user_id=decided_by_user_id).execution_options(synchronize_session=False)
            ).rowcount
            return snapshot if changed else None
