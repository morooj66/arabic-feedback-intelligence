"""
Phase M5B.1 — pure typed contracts for the future retrieval layer.

This module defines TYPES ONLY. It contains no retrieval logic, no
embedding, no vector math, no database access, and no corpus data. It
imports nothing beyond `dataclasses`/`typing`. No sqlalchemy, no
pymongo, no google.genai, no httpx, no embedding/vector SDK, no
`rcs.review.models`. See test_retrieval_contracts.py's static guard for
an enforced check of this.

--- The trust boundary this module exists to make hard to violate ------

Per the M5B.0 governance audit, the project must keep two retrieval
domains structurally separate:

  A. OFFICIAL KNOWLEDGE      -- may support factual/operational claims.
  B. APPROVED RESPONSE MEMORY -- style/response-pattern reference ONLY,
                                  never factual authority.

That separation is enforced here by THREE independent layers, not one:
  1. Two distinct, unrelated result dataclasses (not one generic record
     with a "type" field switching its meaning).
  2. Two distinct, unrelated result-wrapper dataclasses.
  3. Two distinct Protocol interfaces (no shared `Retriever[T]` that a
     caller could satisfy with either implementation interchangeably).
`CorpusType`/`corpus_type` below is a fourth, explicit self-labeling
layer for logging/serialization convenience -- it is deliberately NOT
the only thing standing between the two domains, per instruction.

--- Corrections applied from the M5B.0 review (do not re-introduce) ----

1. No "legacy lowercase reviewer_action vocabulary" is asserted or
   encoded anywhere in this module. That claim was not backed by direct
   evidence in the M5B.0 audit and is not repeated here. Nothing here
   depends on it either.
2. `trusted_grounding_text` / `build_identifying_grounding`
   (rcs.response_generation.prompt, existing and unmodified) are NOT
   part of this contract family and must never be wrapped in
   `OfficialKnowledgeResult`. That existing mechanism only identifies
   WHICH dictionary entry a comment concerns -- it asserts nothing
   factual and has no version/effective-date/authoritative concept. No
   new "IdentifyingContext" type is introduced here either, since
   nothing in this module's actual scope touches it; this paragraph is
   the boundary statement the M5B.0 review asked for.
3. No production-source allowlist, denylist, or filename heuristic is
   implemented or implied anywhere in this module. Query/result types
   below carry no `source_file` field and no fixture/production
   classification logic. That governance decision (an explicit,
   organization-approved allowlist) belongs to a later, separate phase.

--- Query text vs. corpus content (do not conflate) ---------------------

`query_text` on both query types below is the text a caller supplies for
ONE retrieval call, at runtime -- e.g. the CURRENT visitor comment,
already redacted by the caller the same way `redact_for_generation`
already redacts prompt input today. That is unrelated to, and does not
imply anything about, what raw text a future corpus is allowed to
store. The M5B.0 audit recommended AGAINST storing raw historical
visitor comments inside the Approved Response Memory corpus; using
redacted current-comment text as a single query's input carries no such
long-term-storage concern and is not restricted by that recommendation.
Redaction of `query_text` is the CALLER's responsibility -- this module
contains no redaction logic (it is a pure contract, not business logic).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal, Protocol

# --- retrieval outcome vocabulary -------------------------------------
#
# Plain string constants + a Literal alias, not enum.Enum -- matching
# this codebase's own established convention (see
# rcs.response_generation.eligibility's GENERATE_DRAFT/REVIEW_ONLY/
# NO_RESPONSE_RECOMMENDED and rcs.evaluation.genai_client's failure
# categories; enum.Enum is not used anywhere in this codebase today).
# A Literal union is exhaustiveness-checkable by a type checker exactly
# like an Enum would be, without introducing a second vocabulary style.

SUCCESS_WITH_RESULTS = "SUCCESS_WITH_RESULTS"
SUCCESS_EMPTY = "SUCCESS_EMPTY"
TIMEOUT = "TIMEOUT"
UNAVAILABLE = "UNAVAILABLE"
ERROR = "ERROR"

RetrievalOutcome = Literal[
    "SUCCESS_WITH_RESULTS",
    "SUCCESS_EMPTY",
    "TIMEOUT",
    "UNAVAILABLE",
    "ERROR",
]

_FAILURE_OUTCOMES = frozenset({TIMEOUT, UNAVAILABLE, ERROR})

# --- corpus self-labeling (one of four layers, see module docstring) ---

OFFICIAL_KNOWLEDGE = "OFFICIAL_KNOWLEDGE"
APPROVED_RESPONSE_MEMORY = "APPROVED_RESPONSE_MEMORY"

CorpusType = Literal["OFFICIAL_KNOWLEDGE", "APPROVED_RESPONSE_MEMORY"]


def _require_non_blank(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} is required and cannot be blank")


def _require_positive(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_retrieval_invariant(outcome: str, results: tuple, failure_detail: str | None) -> None:
    """Shared by both result-wrapper dataclasses below (see module
    docstring: two concrete wrapper types, not one generic template --
    this function is the single source of truth for the invariant they
    both enforce, without sharing the type itself).

    SUCCESS_WITH_RESULTS requires >= 1 result. Every other outcome
    (including SUCCESS_EMPTY) requires exactly 0 results. A failure
    outcome may optionally carry a sanitized `failure_detail`; a
    success outcome (either kind) never does -- there is nothing to
    detail about a call that completed."""
    if outcome == SUCCESS_WITH_RESULTS:
        if len(results) == 0:
            raise ValueError("SUCCESS_WITH_RESULTS requires at least one result")
    elif len(results) != 0:
        raise ValueError(f"{outcome} must carry zero results, got {len(results)}")

    if outcome not in _FAILURE_OUTCOMES and failure_detail is not None:
        raise ValueError(f"{outcome} must not carry a failure_detail")


# --- Official Knowledge (Corpus A) --------------------------------------


@dataclass(frozen=True)
class OfficialKnowledgeResult:
    """One retrieved factual/operational knowledge item. This is the
    ONLY result type in this module that a future groundedness check
    may treat as factual authority.

    Datetime semantics: `effective_from`/`effective_to` are plain
    `datetime.datetime` values with no timezone conversion performed or
    assumed by this contract -- exactly like every other DTO in this
    codebase (e.g. GenerationRequest/GenerationResult carry datetimes
    the same way). Which timezone convention a value uses is decided by
    whichever future repository actually populates it, and must be
    documented there, not invented here.

    Deliberately excludes: any visitor field, any reviewer field, any
    embedding/vector. `content` is the retrieved text a future
    generation step would use -- not an embedding."""

    corpus_type: CorpusType = field(default=OFFICIAL_KNOWLEDGE, init=False)

    source_id: str
    source_title: str
    source_type: str
    content: str
    version: str
    effective_from: dt.datetime
    authoritative: bool
    retrieval_score: float
    effective_to: dt.datetime | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.source_id, "source_id")
        _require_non_blank(self.source_title, "source_title")
        _require_non_blank(self.source_type, "source_type")
        _require_non_blank(self.content, "content")
        _require_non_blank(self.version, "version")


@dataclass(frozen=True)
class OfficialKnowledgeQuery:
    """Input to `OfficialKnowledgeRetriever.retrieve`. `max_results` has
    no contract-level default or ceiling (per instruction: illustrative
    top-K numbers from the M5B.0 design are not encoded here as hard
    requirements) -- a future implementation's own configuration decides
    a production cap; this contract only requires the caller to make an
    explicit, positive choice."""

    query_text: str
    max_results: int
    topic: str | None = None
    intent: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.query_text, "query_text")
        _require_positive(self.max_results, "max_results")


@dataclass(frozen=True)
class OfficialKnowledgeRetrievalResult:
    """Wraps zero-or-more `OfficialKnowledgeResult`s with an explicit
    outcome. See `_validate_retrieval_invariant` for the enforced
    contract between `outcome` and `results`."""

    outcome: RetrievalOutcome
    results: tuple[OfficialKnowledgeResult, ...] = ()
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        _validate_retrieval_invariant(self.outcome, self.results, self.failure_detail)


class OfficialKnowledgeRetriever(Protocol):
    """Corpus A only. A future groundedness/policy check may treat this
    interface's results as factual authority. `ApprovedResponseMemoryRetriever`
    below is a structurally separate interface -- nothing satisfies both."""

    def retrieve(self, query: OfficialKnowledgeQuery) -> OfficialKnowledgeRetrievalResult: ...


# --- Approved Response Memory (Corpus B) --------------------------------


@dataclass(frozen=True)
class ApprovedResponseMemoryResult:
    """One retrieved human-approved historical response, for STYLE/
    PATTERN reference only. This type has NO `authoritative` field and
    NO factual-grounding flag of any kind, by design -- a future
    groundedness check must be structurally unable to treat this as
    factual evidence, per the M5B.0 trust boundary.

    Deliberately excludes: visitor_name, visitor_email, raw
    original_comment, reviewer display name, reviewer email, any
    identity field beyond `reviewer_action` (a decision-type value like
    "SEND_AS_IS", not an identity). `response_text` is expected to be
    the already-sanitized final reply text -- this contract does not
    perform sanitization itself.

    Datetime semantics for `approved_at`: same note as
    `OfficialKnowledgeResult.effective_from` above -- plain datetime, no
    conversion assumed here."""

    corpus_type: CorpusType = field(default=APPROVED_RESPONSE_MEMORY, init=False)

    review_item_id: str
    final_reply_id: str
    response_text: str
    reviewer_action: str
    approved_at: dt.datetime
    retrieval_score: float
    primary_intent: str | None = None
    topic_subtype: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.review_item_id, "review_item_id")
        _require_non_blank(self.final_reply_id, "final_reply_id")
        _require_non_blank(self.response_text, "response_text")
        _require_non_blank(self.reviewer_action, "reviewer_action")


@dataclass(frozen=True)
class ApprovedResponseMemoryQuery:
    """Input to `ApprovedResponseMemoryRetriever.retrieve`. `query_text`
    is the CALLER-supplied, already-redacted text for this one call
    (e.g. the current comment) -- see the module docstring's "query text
    vs. corpus content" note. No `source_file`/allowlist field exists
    here; production-source governance is a separate, later phase."""

    query_text: str
    max_results: int
    intent: str | None = None
    topic: str | None = None
    sentiment: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.query_text, "query_text")
        _require_positive(self.max_results, "max_results")


@dataclass(frozen=True)
class ApprovedResponseMemoryRetrievalResult:
    """Wraps zero-or-more `ApprovedResponseMemoryResult`s with an
    explicit outcome. Same invariant as
    `OfficialKnowledgeRetrievalResult`, enforced by the same shared
    validator, kept as a distinct type per the module's trust-boundary
    design (see docstring)."""

    outcome: RetrievalOutcome
    results: tuple[ApprovedResponseMemoryResult, ...] = ()
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        _validate_retrieval_invariant(self.outcome, self.results, self.failure_detail)


class ApprovedResponseMemoryRetriever(Protocol):
    """Corpus B only. Results from this interface must NEVER be treated
    as factual authority by any future validation/generation code --
    enforced by `ApprovedResponseMemoryResult` structurally having no
    authoritative-style field for such code to even check."""

    def retrieve(self, query: ApprovedResponseMemoryQuery) -> ApprovedResponseMemoryRetrievalResult: ...
