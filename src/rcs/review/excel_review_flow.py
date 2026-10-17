"""
Read-only Excel ingest → ReviewItems → reviewer decisions → new output workbook.

Never writes the source workbook. Does not call Gemini unless the caller
passes live GenerationResult objects. Offline stubs are demo-only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from openpyxl import load_workbook

from config import settings
from rcs.classification.runtime import (
    apply_optional_topic_layer,
    predict_intent,
    predict_sentiment,
    predict_topic,
    provenance,
    with_layer_provenance,
)
from rcs.classification.redact import redact_phone_numbers
from rcs.ingest.cleaning import clean
from rcs.ingest.loader import _comment_id, load_raw
from rcs.prefilter.spam_rules import BurstConfig, apply_prefilter
from rcs.response_generation.eligibility import (
    GENERATE_DRAFT,
    EligibilityInput,
    determine_response_eligibility,
)
from rcs.response_generation.schema import GenerationResult
from rcs.response_generation.service import GenerationRequest, build_live_gemini_client
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from rcs.response_intelligence.orchestrator import generate_response_with_orchestrator
from rcs.review.import_identity import assert_no_collision
from rcs.review.persistence import resolve_bundle
from rcs.review.service import ReviewWorkflowService
from rcs.routing.engine import determine_routing

INGEST_COLUMNS = [
    "comment_id",
    "source_file",
    "source_row_index",
    "original_comment",
    "original_title",
    "original_type",
    "entry_id",
    "lemma",
    "visitor_name",
    "visitor_email",
    "platform",
    "comment_created_at",
    "is_exact_duplicate_of",
    "ingested_at",
    "comment_age_days",
    "age_group",
    "review_status",
    "ready_to_send",
    "source_reference",
]


# Test/fixture stub ONLY (see apply_offline_generation below). Never used by
# the live Excel E2E path — that path requires a real rg-v7 Gemini call via
# apply_live_generation(). Kept only because tests/test_excel_review_flow.py
# exercises the offline stub path deliberately (no network in unit tests).
# Offline stub for tests. Must pass every reply-safety gate: it deliberately does NOT claim the message reached a team (that claim is rejected as unverifiable).
_TEST_OFFLINE_STUB_DRAFT = "شكرًا لتواصلكم، وصلت رسالتكم."


def _none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().replace(tzinfo=None)
    return value


def records_from_cleaned(cleaned: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in cleaned.to_dict(orient="records"):
        out = {col: _none(rec.get(col)) for col in INGEST_COLUMNS if col in rec or col == "source_reference"}
        src = rec.get("source_file") or ""
        idx = rec.get("source_row_index")
        out["source_reference"] = rec.get("source_reference") or f"{src}:{idx}"
        out["comment_id"] = rec["comment_id"]
        rows.append(out)
    return rows


def load_source_subset(
    source_path: Path,
    *,
    limit: int = 6,
    row_indices: Sequence[int] | None = None,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """Read-only load of a small subset. Source file is never opened for write."""
    raw = load_raw(source_path, settings.SOURCE_COLUMNS)
    if row_indices is not None:
        raw = raw.iloc[list(row_indices)].copy()
    else:
        meaningful = raw["original_comment"].fillna("").astype(str).str.strip().str.len() > 8
        raw = raw.loc[meaningful].head(limit).copy()
    as_of = as_of or dt.datetime.utcnow()
    return clean(
        raw,
        as_of=as_of,
        recent_max_days=settings.AGE_RECENT_MAX_DAYS,
        delayed_max_days=settings.AGE_DELAYED_MAX_DAYS,
    )


def import_subset(engine, source_path: Path, **kwargs) -> dict:
    cleaned = load_source_subset(source_path, **kwargs)
    records = records_from_cleaned(cleaned)
    result = resolve_bundle(engine).review_items.upsert_ingest_batch(records)
    return {
        **result,
        "comment_ids": [r["comment_id"] for r in records],
        "source_path": str(source_path),
        "row_count": len(records),
    }


def apply_offline_generation(svc: ReviewWorkflowService, comment_ids: Iterable[str]) -> None:
    """TEST/FIXTURE STUB ONLY — eligibility + canned draft, no Gemini call.

    Not part of the live Excel E2E path. Production/demo runs must use
    run_frozen_classification() + apply_live_generation() instead, which
    call the real frozen classifiers and a real rg-v7 Gemini client. This
    function exists only so unit tests can exercise import/export without
    network access or a GEMINI_API_KEY.
    """
    for comment_id in comment_ids:
        detail = svc.get_review_item(comment_id)
        elig = determine_response_eligibility(
            EligibilityInput(
                comment_text=detail.original_comment or "",
                primary_intent=detail.primary_intent,
                topic_subtype=detail.topic_subtype,
                sentiment=detail.sentiment,
                confidence=detail.confidence,
                needs_review=bool(detail.needs_review),
                classification_routing_decision=detail.classification_routing_decision,
            )
        )
        draft = _TEST_OFFLINE_STUB_DRAFT if elig.routing_decision == GENERATE_DRAFT else None
        svc.apply_generation_result(
            GenerationResult(
                comment_id=comment_id,
                routing_decision=elig.routing_decision,
                generation_status="SUCCEEDED" if draft else "SKIPPED",
                suggested_response=draft,
                review_required=True,
                reason_code=elig.reason_code,
                provider=None if not draft else "offline_demo",
                model_name=None,
                prompt_version=None,
                generated_at=dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            ),
            intent=detail.primary_intent or "INQUIRY",
            topic=detail.topic_subtype or "GENERAL",
            sentiment=detail.sentiment,
        )


def _ids_with_status(engine, comment_ids: Sequence[str], status: str) -> list[str]:
    """Which of `comment_ids` are currently at `status`, in the same
    relative order as `comment_ids` itself (deterministic — the original
    SQL had no ORDER BY, so this is at least as well-defined as before).
    Ids that don't exist, or exist at a different status, are simply
    absent — never an error."""
    comment_ids = list(comment_ids)
    if not comment_ids:
        return []
    statuses = resolve_bundle(engine).review_items.review_statuses(comment_ids)
    return [comment_id for comment_id in comment_ids if statuses.get(comment_id) == status]


def run_deterministic_prefilter(engine, comment_ids: Sequence[str]) -> dict:
    """Run the EXISTING Phase 3 deterministic prefilter
    (rcs.prefilter.spam_rules.apply_prefilter) over just this batch of
    freshly-ingested comment_ids, then persist via the existing,
    unmodified apply_prefilter_results().

    Scoped to `comment_ids` (rather than the whole table, which is what
    scripts/run_pipeline.py's one-shot use of load_for_prefilter(engine)
    does) so that re-running this for a new upload never touches rows
    from a previous upload that a reviewer has since acted on — those
    rows are no longer "UNPROCESSED" and callers of this function are
    expected to have already restricted `comment_ids` to freshly-inserted,
    still-UNPROCESSED rows only (see process_uploaded_excel below).

    Same rule function, same persistence function, same exclusion
    semantics (EXCLUDE routing, review_status=EXCLUDED,
    primary_intent=SPAM) as the main pipeline — no second policy.
    """
    comment_ids = list(comment_ids)
    if not comment_ids:
        return {"excluded": 0, "passed": 0}

    repo = resolve_bundle(engine).review_items
    df = repo.load_for_prefilter(comment_ids)
    if df.empty:
        return {"excluded": 0, "passed": 0}

    burst_config = BurstConfig(
        min_count=settings.BURST_MIN_COUNT,
        window_hours=settings.BURST_WINDOW_HOURS,
    )
    flagged = apply_prefilter(df, settings.KNOWN_JUNK_STRINGS, burst_config)
    results = flagged[["comment_id", "is_prefiltered_junk", "prefilter_reason"]].to_dict("records")
    return repo.apply_prefilter_batch(results)


def _content_dedupe_key(text: str | None) -> str | None:
    """sha256 of the stripped comment text. None for blank/near-empty
    content — nothing meaningful to deduplicate, and such rows are
    normally prefiltered/excluded before this ever runs anyway."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    return hashlib.sha256(stripped.encode("utf-8")).hexdigest()


# Any status that reflects real reviewer engagement (claimed, decided, or
# dispatched) — a row in one of these must NEVER be converted to
# DUPLICATE_CONTENT, even during a retroactive backfill over rows that
# went through generation/review before dedup existed. It may still be
# recognized/used as the canonical if it happens to be the earliest or
# only engaged member of its group.
_DEDUPE_PROTECTED_STATUSES = frozenset(
    {
        "IN_REVIEW",
        "APPROVED",
        "EDITED_AND_APPROVED",
        "NO_REPLY",
        "SENDING",
        "SENT",
        "SEND_FAILED",
        "REJECTED",
    }
)


def run_content_deduplication(engine, comment_ids: Sequence[str]) -> dict:
    """Exact-content dedup for a batch of freshly-prefiltered rows (or,
    for a one-time backfill, any existing rows for a source_file).

    Only rows still UNPROCESSED (i.e. NOT excluded by the deterministic
    prefilter above) participate — EXCLUDED rows are left completely
    untouched: they never become, and never reference, a canonical.

    Canonical selection: for each exact content key (scoped per
    source_file):
      1. An already-resolved canonical from an earlier call (looked up in
         the DB, so this is correct across batch/resume boundaries) wins
         if one exists.
      2. Otherwise, if ANY member of this group is already
         reviewer-engaged (_DEDUPE_PROTECTED_STATUSES — claimed, decided,
         or dispatched), that row becomes the canonical instead of an
         earlier-but-untouched row — preferring one with an actual
         reviewer_action (a real, terminal decision) over one that's
         merely IN_REVIEW/claimed with no decision yet, earliest
         source_row_index as the final tiebreak. This is what makes the
         backfill safe to run over rows a reviewer has already acted on:
         their claim or decision is never displaced or discarded.
      3. Otherwise, the lowest source_row_index among this batch's own
         rows sharing that content becomes canonical — the original,
         batch-processing-time rule (deterministic and stable because
         process_uploaded_excel_batch() always advances through the
         workbook in increasing row-index order).

    Non-canonical rows are only ever converted if they are themselves NOT
    protected — a protected-but-not-canonical row (e.g. two independently
    reviewed duplicates from before dedup existed) is left completely
    untouched, keeping its own real decision intact rather than being
    silently merged into someone else's.

    Convertible non-canonical duplicates are marked
    review_status="DUPLICATE_CONTENT" — a terminal status that (by
    construction of the existing load_for_classification()/
    apply_live_generation() eligibility queries, both unchanged) is never
    picked up for classification, generation, or the reviewer queue. The
    canonical row is left exactly as it was and proceeds through
    classification -> routing -> Gemini -> reviewer queue (or stays at
    whatever reviewer state it already reached) completely normally.

    Idempotent: a row that already has dedupe_canonical_id set is never
    touched again.

    Returns {"canonical": n, "duplicate_content": n, "skipped_excluded": n}.
    """
    comment_ids = list(comment_ids)
    if not comment_ids:
        return {"canonical": 0, "duplicate_content": 0, "skipped_excluded": 0}

    canonical_count = 0
    duplicate_count = 0
    skipped_excluded = 0

    # Persistence-decoupling milestone: all grouping/tiebreak business
    # logic below is unchanged, plain Python -- only the actual reads/
    # writes now go through ReviewItemRepository (implemented for both
    # PostgreSQL and MongoDB) instead of a raw SQLAlchemy Session.
    repo = resolve_bundle(engine).review_items
    batch_rows = repo.get_dedup_candidates(comment_ids)

    by_key: dict[tuple[str, str], list[tuple[int, str, bool, bool]]] = {}
    for row in batch_rows:
        if row.dedupe_canonical_id is not None:
            continue  # already resolved; never re-touch
        if row.review_status == "EXCLUDED":
            skipped_excluded += 1
            continue
        key = _content_dedupe_key(row.original_comment)
        protected = row.review_status in _DEDUPE_PROTECTED_STATUSES
        decided = row.reviewer_action is not None
        if key is None:
            # Blank content: its own singleton group. content_dedupe_key
            # stays None here -- the original code never set it for this
            # branch either, only dedupe_canonical_id.
            if repo.set_dedupe_canonical_if_unset(row.comment_id, None):
                canonical_count += 1
            continue
        by_key.setdefault((row.source_file, key), []).append(
            (row.source_row_index, row.comment_id, protected, decided)
        )

    for (source_file, key), members in by_key.items():
        existing = repo.find_existing_dedup_canonical(source_file, key)
        members.sort(key=lambda m: m[0])
        protected_members = [m for m in members if m[2]]
        # Among protected members, a row with an actual reviewer_action
        # (a real, terminal decision) wins over one that's merely
        # IN_REVIEW/claimed with no decision yet; earliest row_index
        # is the final tiebreak.
        protected_members.sort(key=lambda m: (0 if m[3] else 1, m[0]))

        if existing is not None:
            canonical_id = existing
        elif protected_members:
            canonical_id = protected_members[0][1]
        else:
            canonical_id = members[0][1]

        if repo.set_dedupe_canonical_if_unset(canonical_id, key):
            canonical_count += 1

        for _row_idx, cid, protected, _decided in members:
            if cid == canonical_id:
                continue
            if protected:
                continue  # independently reviewer-engaged; leave entirely untouched
            repo.mark_dedupe_duplicate(cid, key, canonical_id)
            duplicate_count += 1

    return {"canonical": canonical_count, "duplicate_content": duplicate_count, "skipped_excluded": skipped_excluded}


def run_frozen_classification(engine, comment_ids: Sequence[str]) -> dict:
    """Classify real Excel rows using the configured classifier composition.

    The default `legacy_rcs` mode is the existing frozen selection. The
    experimental `hybrid_jev_topic` mode changes only Topic; Intent and
    Sentiment remain frozen local inference. External/redaction/provider
    failures propagate before persistence: there is no silent classifier
    fallback and therefore no routing decision from mixed or partial results.

    Persists primary_intent/topic_subtype/routing via the existing,
    unmodified apply_classification_results() (same function
    scripts/run_classification.py uses for its Gemini-based path — this
    function only changes what produced the predictions, not how they're
    written or how routing is computed).

    Only classifies rows load_for_classification() (existing, unmodified
    query: is_prefiltered_junk == False AND review_status == UNPROCESSED)
    considers eligible — i.e. rows the prefilter already excluded are
    never sent through the frozen classifiers, matching main-pipeline
    behavior. Restricted to the intersection with `comment_ids` so a
    second upload's classification never reprocesses earlier rows.

    Returns {"sentiments": {comment_id: sentiment}, "classified": n,
    "failed": n, "skipped_already_classified": n} — sentiment has no
    dedicated persistence function and is instead threaded through
    ReviewWorkflowService.apply_generation_result(..., sentiment=...).
    """
    comment_ids = set(comment_ids)
    if not comment_ids:
        return {"sentiments": {}, "classified": 0, "failed": 0, "skipped_already_classified": 0}

    repo = resolve_bundle(engine).review_items
    eligible = repo.load_for_classification()
    eligible = eligible[eligible["comment_id"].isin(comment_ids)]
    if eligible.empty:
        return {"sentiments": {}, "classified": 0, "failed": 0, "skipped_already_classified": 0}

    ordered_ids = list(eligible["comment_id"])
    texts = [(t or "") for t in eligible["original_comment"]]

    intent_preds = predict_intent(texts)
    topic_preds = predict_topic(ordered_ids, texts)
    # Opt-in WS/EC boundary layer (default OFF -> topic_preds is returned unchanged). `lemma` is stored dictionary-entry metadata.
    lemmas = list(eligible["lemma"]) if "lemma" in eligible.columns else None
    topic_preds, layer_applied = apply_optional_topic_layer(topic_preds, ordered_ids, texts, lemmas)
    sentiment_preds = predict_sentiment(ordered_ids, texts)
    classification_provenance = with_layer_provenance(provenance(), layer_applied)

    text_by_id = dict(zip(ordered_ids, texts))
    results = []
    sentiments: dict[str, str] = {}
    for cid, (intent_label, intent_conf) in zip(ordered_ids, intent_preds):
        topic_label, _topic_conf = topic_preds.get(cid, (None, None))
        sentiment_label, _sent_conf = sentiment_preds.get(cid, (None, None))
        if sentiment_label:
            sentiments[cid] = sentiment_label

        routing = determine_routing(
            primary_intent=intent_label,
            topic_subtype=topic_label,
            subcategory=None,
            secondary_intent=None,
            is_content_meaningful=True,
        )
        routing_decision, routing_reason = routing.routing_decision, routing.reason_code
        needs_review = intent_conf is not None and intent_conf < settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD
        class_reason = classification_provenance["reason"]
        if settings.CONTEXT_SIGNALS_ENABLED:                       # opt-in, default OFF: flags only, labels are never changed
            from rcs.classification import context_signals

            conflicts = context_signals.detect_conflicts(
                context_signals.analyze(text_by_id[cid]), intent_label, topic_label, sentiment_label)
            routing_decision, routing_reason, needs_review = context_signals.apply_to_decision(conflicts, routing_decision, routing_reason, needs_review)
            if conflicts:
                class_reason = f"{class_reason}; context_conflicts={','.join(context_signals.conflict_codes(conflicts))}"
        results.append(
            {
                "comment_id": cid,
                "success": True,
                "ai_primary_intent": intent_label,
                "ai_topic_subtype": topic_label,
                "ai_subcategory": None,
                "ai_secondary_intent": None,
                "ai_routing_decision": routing_decision,
                "ai_classification_reason": class_reason,
                "ai_confidence": intent_conf,
                # Persisted at classification time (before generation), into
                # the existing ai_sentiment/sentiment columns, so sentiment
                # filters and analytics cover rows that never reach generation.
                "ai_sentiment": sentiment_label,
                "ai_classification_source": classification_provenance["source"],
                "ai_needs_review": needs_review,
                "routing_reason_code": routing_reason,
            }
        )

    write_result = repo.apply_classification_batch(
        results,
        model_name=classification_provenance["model_name"],
        model_version=None,
        prompt_version=classification_provenance["prompt_version"],
        pipeline_version=settings.PIPELINE_VERSION,
        inference_run_id=str(uuid.uuid4()),
    )
    return {"sentiments": sentiments, **write_result}


def apply_live_generation(
    svc: ReviewWorkflowService,
    comment_ids: Iterable[str],
    client,
    sentiments: dict[str, str] | None = None,
) -> dict:
    """Real rg-v7 Gemini generation for the live Excel E2E path.

    Phase M5A.1: calls generate_response_with_orchestrator() instead of
    generate_suggested_response() directly. This is a structural seam
    only — the orchestrator delegates unchanged to the same
    generate_suggested_response() for every actual decision (eligibility,
    prompt, Gemini call/retry, validation); behavior here is identical
    to before this phase.

    Phase M5E.1: passes `audit_repository=svc.audit_repo`
    (a `PostgresAgentAuditRepository` by default — see
    `ReviewWorkflowService.__init__`) so every real generation attempt
    through this path now persists a sanitized `AgentExecutionTrace`
    row. Purely additive observability: a failure to record that trace
    is caught inside the orchestrator and never affects `result` or
    this function's own behavior.

    `client` may be a real GenAI client, a scripted test double, or None.
    There is no offline/canned fallback here: passing client=None simply
    means generate_suggested_response() (reached via the orchestrator)
    takes its own existing "no client provided" branch, which fails that
    one row closed as FAILED_REVIEW_ONLY (review_status -> REVIEW_ONLY)
    rather than inventing a reply — the same behavior a live Gemini
    outage would already produce via that function, unchanged.

    Only attempts generation for comment_ids currently CLASSIFIED (the
    state apply_classification_results() advances a row to on success) —
    rows the prefilter already excluded, or rows a reviewer already
    resolved, are skipped rather than passed to the workflow state
    machine (which would otherwise raise InvalidTransition for e.g. an
    EXCLUDED row).

    Returns {"generated_drafts": n, "no_response_recommended": n,
    "review_only": n, "attempted": n}.
    """
    sentiments = sentiments or {}
    classified_ids = _ids_with_status(engine=svc.engine, comment_ids=list(comment_ids), status="CLASSIFIED")
    stats = {"generated_drafts": 0, "no_response_recommended": 0, "review_only": 0, "attempted": 0}
    for comment_id in classified_ids:
        detail = svc.get_review_item(comment_id)
        sentiment = sentiments.get(comment_id) or detail.sentiment
        # Phase M6.2: structured metadata only -- the orchestrator's
        # Context/Trust Policies step (identifying_context_policy.
        # evaluate_identifying_context) decides whether this reaches
        # Gemini at all; rendering (build_identifying_grounding) happens
        # inside the orchestrator, only after that decision allows it.
        # `trusted_grounding_available`/`trusted_grounding_text` are
        # deliberately not set here: the orchestrator always overwrites
        # both with Official-Knowledge-derived values before eligibility
        # or generation ever sees them (see orchestrator.py's
        # `enriched_request`), so setting them from identifying context
        # here would have had no effect even before this change.
        request = GenerationRequest(
            comment_id=comment_id,
            comment_text=detail.original_comment or "",
            primary_intent=detail.primary_intent,
            topic_subtype=detail.topic_subtype,
            sentiment=sentiment,
            confidence=detail.confidence,
            needs_review=bool(detail.needs_review),
            classification_routing_decision=detail.classification_routing_decision,
            identifying_metadata=IdentifyingMetadata(detail.original_title, detail.entry_id),
        )
        result = generate_response_with_orchestrator(request, client=client, audit_repository=svc.audit_repo)
        svc.apply_generation_result(
            result,
            intent=detail.primary_intent,
            topic=detail.topic_subtype,
            sentiment=sentiment,
            confidence=detail.confidence,
            needs_review=detail.needs_review,
        )
        stats["attempted"] += 1
        if result.routing_decision == GENERATE_DRAFT and result.generation_status == "SUCCEEDED":
            stats["generated_drafts"] += 1
        elif result.routing_decision == "NO_RESPONSE_RECOMMENDED":
            stats["no_response_recommended"] += 1
        else:
            stats["review_only"] += 1
    return stats


@dataclass(frozen=True)
class ExcelProcessingSummary:
    """Structured result of process_uploaded_excel(), safe to serialize
    directly for the reviewer app's UI (no model names, no prompt
    internals, no confidence internals)."""

    file_name: str
    total_rows: int
    valid_rows: int
    inserted_rows: int
    duplicate_rows: int
    duplicate_content_rows: int
    excluded_rows: int
    classification_processed: int
    generated_drafts: int
    no_response_recommended: int
    review_only: int
    failed_rows: int
    output_available: bool
    comment_ids: list[str] = field(default_factory=list)


def process_uploaded_excel(
    engine,
    source_path: Path,
    *,
    file_name: str | None = None,
    limit: int | None = None,
    row_indices: Sequence[int] | None = None,
    client=None,
) -> ExcelProcessingSummary:
    """The one Excel application-level operation: ingest+clean -> existing
    deterministic prefilter -> frozen classification -> existing
    eligibility -> real rg-v7 Gemini generation.

    Reuses import_subset/run_deterministic_prefilter/
    run_frozen_classification/apply_live_generation exactly as defined
    above — this function only sequences them and scopes each stage to
    the rows still eligible for it, so a duplicate upload of the same
    file never reprocesses already-ingested/classified/reviewed rows.

    `row_indices`, when given, selects an exact set of source-row
    positions to import this call (used by process_uploaded_excel_batch()
    for resumable full-workbook batches); otherwise `limit` picks the
    first N meaningful rows, as before.

    `client`: pass an explicit SupportsGenerate for tests. If omitted,
    attempts build_live_gemini_client() (requires GEMINI_API_KEY); if
    that fails (key missing/invalid), generation proceeds with
    client=None, which fails each eligible row closed
    (FAILED_REVIEW_ONLY) rather than fabricating a draft — see
    apply_live_generation.
    """
    kwargs: dict = {}
    if row_indices is not None:
        kwargs["row_indices"] = row_indices
    elif limit is not None:
        kwargs["limit"] = limit
    imported = import_subset(engine, source_path, **kwargs)
    comment_ids = imported["comment_ids"]
    svc = ReviewWorkflowService(engine)

    fresh_ids = _ids_with_status(engine, comment_ids, "UNPROCESSED")
    prefilter_result = run_deterministic_prefilter(engine, fresh_ids)

    # Exact-content dedup, scoped to whatever is still UNPROCESSED after
    # prefilter (i.e. not excluded). Non-canonical duplicates are marked
    # DUPLICATE_CONTENT here, which is why they're naturally invisible to
    # both run_frozen_classification (below) and apply_live_generation
    # (further down) without either needing to change at all.
    still_unprocessed = _ids_with_status(engine, comment_ids, "UNPROCESSED")
    dedup_result = run_content_deduplication(engine, still_unprocessed)

    classification_result = run_frozen_classification(engine, comment_ids)

    if client is None:
        try:
            client = build_live_gemini_client()
        except Exception:
            client = None  # generate_suggested_response fails each row closed; never a canned fallback

    generation_result = apply_live_generation(svc, comment_ids, client, classification_result["sentiments"])

    return ExcelProcessingSummary(
        file_name=file_name or Path(source_path).name,
        total_rows=imported["row_count"],
        valid_rows=imported["row_count"],
        inserted_rows=imported["inserted"],
        duplicate_rows=imported["skipped"],
        duplicate_content_rows=dedup_result["duplicate_content"],
        excluded_rows=prefilter_result["excluded"],
        classification_processed=classification_result["classified"],
        generated_drafts=generation_result["generated_drafts"],
        no_response_recommended=generation_result["no_response_recommended"],
        review_only=generation_result["review_only"],
        failed_rows=classification_result["failed"],
        output_available=bool(comment_ids),
        comment_ids=comment_ids,
    )


DEFAULT_BATCH_SIZE = 30


def _workbook_row_stats(source_path: Path) -> tuple[int, list[int]]:
    """Read-only pass over the whole workbook: (total_raw_rows,
    meaningful_row_indices). `meaningful` reproduces load_source_subset's
    own filter (original_comment stripped length > 8) so batch counting
    matches exactly what would ever be ingested."""
    raw = load_raw(source_path, settings.SOURCE_COLUMNS)
    meaningful_mask = raw["original_comment"].fillna("").astype(str).str.strip().str.len() > 8
    meaningful_indices = raw.loc[meaningful_mask, "source_row_index"].tolist()
    return len(raw), meaningful_indices


def _next_batch_row_indices(
    engine, source_path: Path, source_file_name: str, batch_size: int
) -> tuple[list[int], int, int]:
    """Returns (next_batch_row_indices, total_raw_rows, meaningful_row_count).

    Resumability: comment_id is sha256(source_file_name + "|" +
    source_row_index) (rcs.ingest.loader._comment_id) — deterministic and
    stable across calls for the same file/row. So "already processed" is
    computed by checking which of the file's meaningful row indices'
    comment_ids already exist in review_items for this source_file, with
    no dependence on run order or any separate progress-tracking table.
    """
    total_rows, meaningful_indices = _workbook_row_stats(source_path)
    existing = set(resolve_bundle(engine).review_items.comment_ids_for_source_file(source_file_name))
    pending = [i for i in meaningful_indices if _comment_id(source_file_name, i) not in existing]
    return pending[:batch_size], total_rows, len(meaningful_indices)


def _source_file_aggregate(engine, source_file_name: str) -> dict:
    """Cumulative, whole-file counters for this source_file — read-only
    aggregation over whatever has been persisted so far across every
    batch, never recomputed from the source workbook itself. The
    counting rule (which combination of fields counts as which bucket)
    lives here, in plain Python, computed once from the repository's raw
    per-row read — never duplicated across backends."""
    rows = resolve_bundle(engine).review_items.source_file_status_fields(source_file_name)
    processed_total = len(rows)
    excluded_total = sum(1 for r in rows if r.review_status == "EXCLUDED")
    generated_drafts_total = sum(
        1 for r in rows if r.generation_routing_decision == GENERATE_DRAFT and r.generation_status == "SUCCEEDED"
    )
    no_response_recommended_total = sum(1 for r in rows if r.review_status in ("NO_RESPONSE_RECOMMENDED", "NO_REPLY"))
    needs_review_total = sum(1 for r in rows if r.review_status in ("DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW"))
    duplicate_content_total = sum(1 for r in rows if r.review_status == "DUPLICATE_CONTENT")
    return {
        "processed_total": processed_total,
        "excluded_total": excluded_total,
        "generated_drafts_total": generated_drafts_total,
        "no_response_recommended_total": no_response_recommended_total,
        "needs_review_total": needs_review_total,
        "duplicate_content_total": duplicate_content_total,
    }


@dataclass(frozen=True)
class ExcelBatchProcessingSummary:
    """Resumable-batch processing result, safe to serialize directly for
    the reviewer app's UI. All *_total fields are cumulative across every
    batch run so far for this source file, not just this call."""

    file_name: str
    total_file_rows: int
    batch_processed: int
    processed_total: int
    remaining: int
    excluded_total: int
    generated_drafts_total: int
    no_response_recommended_total: int
    needs_review_total: int
    duplicate_content_total: int
    comment_ids: list[str] = field(default_factory=list)
    # Accurate accounting of rows that were deliberately NOT imported by this call (not failures):
    invalid_rows_skipped: int = 0     # rows too short to be a real comment (the existing "meaningful" rule), never ingested
    already_imported: int = 0         # rows of this workbook that were already stored under this batch name before this call (re-upload / resume)


def process_uploaded_excel_batch(
    engine,
    source_path: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    client=None,
) -> ExcelBatchProcessingSummary:
    """Resumable batch entry point for a large workbook (e.g. the full
    8,413-row source): processes at most `batch_size` NEW/unprocessed
    rows per call, then reports cumulative progress for the whole file.

    Reuses process_uploaded_excel() unchanged for the actual pipeline
    (ingest -> prefilter -> frozen classification -> rg-v7 generation) —
    this function only selects which row_indices are still pending and
    aggregates cumulative counters afterward. Calling it repeatedly with
    the same source_path continues from wherever the previous call left
    off; already classified/generated/reviewed rows are never reselected
    (comment_id-stable, per _next_batch_row_indices above), so reviewer
    decisions are never touched and no row is ever sent to Gemini twice.
    """
    source_path = Path(source_path)
    source_file_name = source_path.name
    # Last line of defence: rows already stored under this batch name must belong to this workbook. Raises ImportIdentityCollision (nothing written)
    # instead of silently skipping or overwriting another batch's rows.
    assert_no_collision(engine, source_path)
    already_imported = len(resolve_bundle(engine).review_items.comment_ids_for_source_file(source_file_name))
    batch_indices, total_rows, meaningful_count = _next_batch_row_indices(
        engine, source_path, source_file_name, batch_size
    )

    batch_comment_ids: list[str] = []
    if batch_indices:
        batch_result = process_uploaded_excel(
            engine,
            source_path,
            file_name=source_file_name,
            row_indices=batch_indices,
            client=client,
        )
        batch_comment_ids = batch_result.comment_ids

    agg = _source_file_aggregate(engine, source_file_name)
    remaining = max(0, meaningful_count - agg["processed_total"])

    return ExcelBatchProcessingSummary(
        file_name=source_file_name,
        total_file_rows=total_rows,
        batch_processed=len(batch_indices),
        processed_total=agg["processed_total"],
        remaining=remaining,
        excluded_total=agg["excluded_total"],
        generated_drafts_total=agg["generated_drafts_total"],
        no_response_recommended_total=agg["no_response_recommended_total"],
        needs_review_total=agg["needs_review_total"],
        duplicate_content_total=agg["duplicate_content_total"],
        comment_ids=batch_comment_ids,
        invalid_rows_skipped=max(0, total_rows - meaningful_count),
        already_imported=already_imported,
    )


_ID_CHUNK = 5000
_EXPORT_SHEET1_TITLE = "الردود المراجعة"
_EXPORT_SHEET2_TITLE = "بيانات التدقيق"

_EXPORT_SHEET1_HEADERS = [
    "المرجع المصدري",
    "تعليق الزائر",
    "معرف المدخل",
    "اللما",
    "التصنيف الرئيسي",
    "الموضوع",
    "نبرة التعليق",
    "الرد المقترح",
    "إجراء المراجع",
    "الرد بعد التعديل",
    "الرد النهائي",
    "اسم المراجع",
    "تاريخ المراجعة",
    "حالة المراجعة",
    # Appended (not inserted) so every existing column keeps its exact
    # position — smallest compatible change, see export privacy/source-date
    # fix: العنوان helps identify the reviewed entry; تاريخ التعليق is the
    # visitor's original submission date (distinct from تاريخ المراجعة,
    # which is when the reviewer decided).
    "العنوان",
    "تاريخ التعليق",
]

_EXPORT_SHEET2_HEADERS = [
    "comment_id",
    "dedupe_canonical_id",
    "classification_model",
    "classification_source",
    "classification_confidence",
    "generation_provider",
    "generation_model_name",
    "generation_prompt_version",
    "generated_at",
    "outbound_status",
    "outbound_idempotency_key",
    "no_response_reason",
]

_EXPORT_HEADER_FILL_HEX = "1F5C3E"  # brand-green, matches rg_v8 review workbook convention


def _export_row_height(*texts) -> float:
    """Same heuristic used by the rg-v8 AB-eval review workbook builder —
    duplicated here deliberately rather than shared, since the two live
    in different lifecycle contexts (this one runs on every reviewer
    export, the other is a one-off research script)."""
    chars_per_line = 55
    line_height = 15.0
    max_lines = 1
    for text in texts:
        if not text:
            continue
        lines = 0
        for physical_line in str(text).split("\n"):
            lines += max(1, -(-len(physical_line) // chars_per_line))
        max_lines = max(max_lines, lines)
    return min(max(max_lines * line_height, 26), 260)


def _style_export_sheet(ws, *, headers: list[str], data_row_count: int, wrap_columns: set[int]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    ws.sheet_view.rightToLeft = True
    header_fill = PatternFill("solid", fgColor=_EXPORT_HEADER_FILL_HEX)
    header_font = Font(bold=True, size=11, color="FFFFFF")
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True, readingOrder=2)
    ws.row_dimensions[1].height = 30

    for col_idx in range(1, len(headers) + 1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = 46 if col_idx in wrap_columns else 18

    for row_idx in range(2, data_row_count + 2):
        for cell in ws[row_idx]:
            cell.alignment = Alignment(
                wrap_text=cell.column in wrap_columns, vertical="top", horizontal="right", readingOrder=2
            )

    last_row = data_row_count + 1
    ws.freeze_panes = "A2"
    if last_row >= 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last_row}"


def export_reviewed_workbook(engine, dest: Path, comment_ids: Sequence[str] | None = None) -> Path:
    """Two-sheet reviewer-facing export (Phase 4):

      Sheet 1 "الردود المراجعة" — business-facing columns only, Arabic
      headers, one row per ORIGINAL source row (full provenance — never
      collapsed, see the duplicate-handling note below).
      Sheet 2 "بيانات التدقيق" — technical/audit fields (comment_id,
      dedupe/classification/generation provenance, outbound status).

    Exact-content duplicate rows (review_status="DUPLICATE_CONTENT") keep
    their own source_reference/comment_text (full source-row provenance
    is never lost — every original row still gets its own export line in
    both sheets) but display the CANONICAL row's classification/decision
    fields, since the duplicate row itself was never independently
    classified or generated — only the canonical was. This is "Option B"
    from the dedup design: re-expand to every original source row rather
    than collapsing the export.

    Never touches the source workbook — writes a brand-new file at `dest`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Phase M4A: the two query shapes this export needs are extracted
    # behind PostgresReviewItemRepository (list_ordered / get_many) --
    # everything below (row expansion, redaction, canonical
    # substitution, sheet styling) is unchanged, unextracted business
    # logic operating on plain ORM attribute access, exactly as before.
    repo = resolve_bundle(engine).review_items
    if comment_ids is not None and len(comment_ids) > _ID_CHUNK:
        # keep every IN (...) list well below the database parameter limits; same ordering as list_ordered (source_row_index, comment_id)
        ids_all = list(comment_ids)
        items = [it for k in range(0, len(ids_all), _ID_CHUNK) for it in repo.list_ordered(ids_all[k:k + _ID_CHUNK])]
        items.sort(key=lambda it: (it.source_row_index, it.comment_id))
    else:
        items = repo.list_ordered(list(comment_ids) if comment_ids is not None else None)
    canonical_ids = list(
        {
            item.dedupe_canonical_id
            for item in items
            if item.dedupe_canonical_id and item.dedupe_canonical_id != item.comment_id
        }
    )
    canonicals = {}
    for k in range(0, len(canonical_ids), _ID_CHUNK):
        canonicals.update(repo.get_many(canonical_ids[k:k + _ID_CHUNK]))

    sheet1_rows = []
    sheet2_rows = []
    for item in items:
        source = item
        if item.dedupe_canonical_id and item.dedupe_canonical_id != item.comment_id:
            source = canonicals.get(item.dedupe_canonical_id, item)

        final = source.final_reply
        if source.reviewer_action == "NO_RESPONSE_NEEDED":
            final = None

        sheet1_rows.append(
            [
                item.source_reference,
                # Redacted, same as every other reviewer/admin-facing view
                # of this text (rcs.classification.redact.redact_phone_numbers)
                # — the normal reviewed-responses export must not expose
                # inline phone numbers a visitor happened to type.
                redact_phone_numbers(item.original_comment),
                item.entry_id,
                item.lemma,
                source.primary_intent,
                source.topic_subtype,
                source.sentiment,
                source.suggested_final_reply,
                source.reviewer_action,
                source.reviewer_edit,
                final,
                source.reviewer_display_name,
                source.reviewed_at,
                source.review_status,
                item.original_title,
                item.comment_created_at,
            ]
        )
        sheet2_rows.append(
            [
                item.comment_id,
                item.dedupe_canonical_id,
                source.model_name,
                source.ai_classification_source,
                source.ai_confidence,
                source.generation_provider,
                source.generation_model_name,
                source.generation_prompt_version,
                source.generated_at,
                source.outbound_status,
                source.outbound_idempotency_key,
                source.no_response_reason,
            ]
        )

    from openpyxl import Workbook

    wb = Workbook()
    ws1 = wb.active
    ws1.title = _EXPORT_SHEET1_TITLE
    ws1.append(_EXPORT_SHEET1_HEADERS)
    for row in sheet1_rows:
        ws1.append(row)
    # comment_text (2), suggested_response (8), edited_response (10), final_response (11)
    _style_export_sheet(ws1, headers=_EXPORT_SHEET1_HEADERS, data_row_count=len(sheet1_rows), wrap_columns={2, 8, 10, 11})
    for i, row in enumerate(sheet1_rows, start=2):
        ws1.row_dimensions[i].height = _export_row_height(row[1], row[7], row[9], row[10])

    ws2 = wb.create_sheet(_EXPORT_SHEET2_TITLE)
    ws2.append(_EXPORT_SHEET2_HEADERS)
    for row in sheet2_rows:
        ws2.append(row)
    _style_export_sheet(ws2, headers=_EXPORT_SHEET2_HEADERS, data_row_count=len(sheet2_rows), wrap_columns=set())

    # openpyxl turns any string that starts with "=" into a live formula; visitor comments and reviewer replies are untrusted text, so keep them text.
    for ws in (ws1, ws2):
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    cell.data_type = "s"

    dest = Path(dest)
    wb.save(dest)
    return dest


def assert_source_unmodified(source_path: Path, digest_before: bytes) -> None:
    after = Path(source_path).read_bytes()
    if after != digest_before:
        raise RuntimeError(f"Source workbook was modified: {source_path}")


def source_digest(source_path: Path) -> bytes:
    return Path(source_path).read_bytes()


def write_fixture_workbook(path: Path, rows: list[dict]) -> Path:
    """Tiny SOURCE_COLUMNS workbook for tests. Not the historical export."""
    headers = list(settings.SOURCE_COLUMNS.keys())
    internal = list(settings.SOURCE_COLUMNS.values())
    data = {h: [] for h in headers}
    for row in rows:
        mapped = {internal[i]: row.get(internal[i]) for i in range(len(internal))}
        for h, key in settings.SOURCE_COLUMNS.items():
            data[h].append(mapped.get(key))
    pd.DataFrame(data).to_excel(path, index=False)
    return path


def workbook_headers(path: Path) -> list[str]:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = wb.active
        return [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    finally:
        wb.close()
