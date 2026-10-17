"""
Phase M5B.3 — Approved Response Memory corpus: fail-closed eligibility
governance and sanitized corpus records.

This module owns the ONLY eligibility logic for what may enter the
Approved Response Memory corpus (style/response-pattern reference only —
see rcs.response_intelligence.retrieval.contracts's module docstring for
the trust boundary this whole retrieval family exists to protect).

Storage-agnostic by construction: nothing here imports sqlalchemy,
pymongo, or rcs.review.models. A caller (a future repository-side
adapter, not built in this phase — see the M5B.3 final report's known
limitations) is responsible for populating `RawApprovedResponseCandidate`
from whichever backend it uses.

--- Governance facts this module encodes, and their evidence ------------

Every rule below was taken from a direct audit of config/settings.py and
src/rcs/review/{models.py,repositories/postgres.py} — not guessed:

1. Only currently proven workflow values `SEND_AS_IS` and `EDIT_AND_SEND`
   (`config.settings.WORKFLOW_REVIEWER_ACTIONS`, minus
   `NO_RESPONSE_NEEDED`, a "no reply" terminal decision, never eligible)
   are treated as an approved/sent decision. All unknown values fail
   closed. This module does NOT assert or rely on any characterization
   of a historical/legacy reviewer_action vocabulary: a fresh M5B.3a
   re-audit found `config.settings.REVIEWER_ACTIONS`'s six lowercase
   strings have no genuine production evidence anywhere in this
   repository — the sole occurrence of one of them ("approve", in
   tests/test_classification.py:455) is an arbitrary placeholder value
   in an unrelated field-overwrite-protection test, not evidence that
   any reviewer/workflow code path ever produces it. Consistent with
   M5B.1's contracts.py, which already declined to encode that
   vocabulary for the same reason (see its "Corrections applied from
   the M5B.0 review", item 1) — this module makes no behavior
   allowance for it either, guessed or otherwise.
2. Genuinely-completed review approval is
   `config.settings.READY_TO_SEND_STATUSES` = ("APPROVED",
   "EDITED_AND_APPROVED"). Checked independently of reviewer_action as a
   second, redundant governance layer — this project's own established
   multi-layer style (see contracts.py's trust-boundary docstring).
3. `source_file` is uploader-controlled free text with NO existing
   allowlist concept anywhere in the codebase before this phase (see
   src/rcs/ingest/loader.py, src/rcs/review/excel_upload.py — neither
   validates source_file against any trusted-source list). Per
   instruction, this module MUST NOT invent a filename heuristic
   ("contains 'prod'", "not a fixture", etc.) — eligibility is gated
   ONLY by exact membership in a positive, organization-supplied
   allowlist (config.settings.APPROVED_RESPONSE_SOURCE_ALLOWLIST),
   which defaults to EMPTY. An empty allowlist makes every candidate
   fail rule 5 below, so `build_governed_corpus` returns zero records —
   never a silent fallback to untrusted data.
4. Canonical/non-duplicate: `ReviewItem.dedupe_canonical_id` equals a
   row's own comment_id only when that row IS the canonical
   representative of its exact-content dedup group (models.py's
   documented semantics). This module requires that equality exactly;
   NULL (dedup not yet run for this row) or a mismatch both fail closed
   — never assumed safe.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rcs.response_generation.prompt import redact_for_generation
from rcs.response_generation.validate import _boilerplate_gate, _internal_leak_gate

# Only currently proven workflow values are eligible; all unknown values
# (including "NO_RESPONSE_NEEDED" and every value outside this set) fail
# closed. See module docstring point 1.
APPROVED_REVIEWER_ACTIONS = frozenset({"SEND_AS_IS", "EDIT_AND_SEND"})


def _approved_review_statuses() -> frozenset[str]:
    # Imported lazily, inside a function, so this module's own import
    # graph does not force config.settings to be imported at module
    # load time in a context where that's undesirable (matches this
    # file's storage/config-agnostic spirit) -- see also build_governed_
    # corpus's `allowlist` parameter, which is caller-supplied for the
    # same reason rather than read from settings internally.
    from config import settings

    return frozenset(settings.READY_TO_SEND_STATUSES)


@dataclass(frozen=True)
class RawApprovedResponseCandidate:
    """Plain, storage-agnostic input record for ONE historical review
    item, exactly as governance eligibility needs to see it. A future
    repository-side adapter (Postgres or Mongo, not built in this phase)
    would populate `approved_at` from `ReviewItem.reviewed_at` — the
    only proven-existing timestamp for "when this row reached its final
    reviewer decision"; there is no separate `approved_at` column in the
    current schema.

    `visitor_name`/`visitor_email`/`original_comment` are intentionally
    NOT fields here at all -- this dataclass cannot leak what it has no
    field for. Only what eligibility checking and corpus-record building
    actually need is present."""

    review_item_id: str
    source_file: str
    review_status: str
    reviewer_action: str | None
    final_reply: str | None
    dedupe_canonical_id: str | None
    approved_at: dt.datetime | None
    primary_intent: str | None = None
    topic_subtype: str | None = None
    sentiment: str | None = None


class ApprovedResponseCandidateSource(Protocol):
    """Narrow, storage-agnostic provider interface -- an alternative to
    passing already-loaded records directly into `build_governed_corpus`.
    No concrete Postgres/Mongo implementation exists in this phase (see
    the M5B.3 final report's known limitations); a future repository-side
    adapter would satisfy this Protocol structurally without this
    package ever importing sqlalchemy/pymongo."""

    def fetch_candidates(self) -> Iterable[RawApprovedResponseCandidate]: ...


@dataclass(frozen=True)
class ApprovedResponseCorpusRecord:
    """Sanitized, storage-agnostic corpus record -- the ONLY shape ever
    handed to the embedding index (see embeddings.py/approved_memory.py).
    `build_governed_corpus` is the sole producer of this type and never
    reads visitor/reviewer-identity fields or raw original_comment from
    the raw candidate, so this record cannot carry them.

    `final_reply_id` is a deterministic sha256 hex digest of the
    sanitized `response_text` -- there is no separate final-reply id
    column in the current schema, and a content hash is stable across
    rebuilds for identical text without inventing a new identity
    concept."""

    review_item_id: str
    final_reply_id: str
    response_text: str
    reviewer_action: str
    approved_at: dt.datetime
    primary_intent: str | None = None
    topic_subtype: str | None = None
    sentiment: str | None = None


def _passes_structural_eligibility(
    candidate: RawApprovedResponseCandidate,
    *,
    allowlist: frozenset[str],
    approved_review_statuses: frozenset[str],
) -> bool:
    if not candidate.final_reply or not candidate.final_reply.strip():
        return False
    if candidate.review_status not in approved_review_statuses:
        return False
    if candidate.reviewer_action not in APPROVED_REVIEWER_ACTIONS:
        return False
    if candidate.dedupe_canonical_id != candidate.review_item_id:
        return False
    if candidate.source_file not in allowlist:
        return False
    if candidate.approved_at is None:
        return False
    return True


def build_governed_corpus(
    candidates: Iterable[RawApprovedResponseCandidate],
    *,
    allowlist: frozenset[str],
) -> tuple[ApprovedResponseCorpusRecord, ...]:
    """Pure function: raw candidates in, sanitized+eligible corpus
    records out, sorted by review_item_id for determinism. Every
    exclusion reason is enforced here and only here -- see module
    docstring for the evidence behind each rule.

    An empty `allowlist` (the default, config.settings.
    APPROVED_RESPONSE_SOURCE_ALLOWLIST) makes every candidate fail the
    source_file check, so this always returns an empty tuple -- fails
    closed, never falls back to an unvalidated source."""
    approved_review_statuses = _approved_review_statuses()
    records: list[ApprovedResponseCorpusRecord] = []
    for candidate in candidates:
        if not _passes_structural_eligibility(
            candidate, allowlist=allowlist, approved_review_statuses=approved_review_statuses
        ):
            continue

        # Rule 6 (PII/redaction discipline): reuse the EXACT existing
        # generation-time redaction function, not a new regex. This is
        # a build-time content transform, unrelated to query_text
        # redaction (see contracts.py's "query text vs corpus content"
        # note) -- redacting here determines what gets STORED.
        sanitized = redact_for_generation(candidate.final_reply).strip()
        if not sanitized:
            continue

        # Rules 7-8: reuse the exact existing internal-leak and
        # boilerplate gates from response_generation.validate (the same
        # module scripts/run_response_quality_eval_30.py already reaches
        # into for _LEAK_PATTERNS -- an established precedent in this
        # codebase for importing these specific names directly, rather
        # than duplicating the regex/constant lists here). Deliberately
        # NOT using the full run_validation_gates/validate_draft pipeline:
        # its content_shape gate is tuned for AI-draft length, and its
        # unsupported-operational-claim gate assumes no matching source
        # text is ever available to check against, which would
        # over-exclude ordinary approved replies that happen to contain
        # an operational phrase this corpus was never asked to fact-check.
        if not _internal_leak_gate(sanitized).passed:
            continue
        if not _boilerplate_gate(sanitized).passed:
            continue

        records.append(
            ApprovedResponseCorpusRecord(
                review_item_id=candidate.review_item_id,
                final_reply_id=hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
                response_text=sanitized,
                reviewer_action=candidate.reviewer_action,
                approved_at=candidate.approved_at,
                primary_intent=candidate.primary_intent,
                topic_subtype=candidate.topic_subtype,
                sentiment=candidate.sentiment,
            )
        )

    return tuple(sorted(records, key=lambda r: r.review_item_id))
