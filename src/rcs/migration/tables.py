"""
Central migration metadata for the five source tables (Phase M1/M2
baseline). Column lists are derived from the live ORM models
(rcs.review.models) at import time -- never re-typed by hand -- so
extraction/transformation can never silently drift from the real schema.

The naive-datetime timezone convention map below is the one piece of
metadata that genuinely cannot be derived from SQLAlchemy's column
metadata: it was established by reading each column's actual write path
(see the comment on each entry). It is a real, mixed convention across
this schema -- not a simplification -- and the transformer must honor it
per-column rather than applying one blanket rule.
"""

from __future__ import annotations

from sqlalchemy import DateTime

from rcs.review.models import ProcessingJob, ReviewItem, ReviewSession, User, UserSession

# table name -> (ORM class, primary-key / business-id column name)
TABLE_SPECS: dict[str, tuple[type, str]] = {
    "users": (User, "user_id"),
    "user_sessions": (UserSession, "session_token"),
    "review_sessions": (ReviewSession, "session_id"),
    "processing_jobs": (ProcessingJob, "job_id"),
    "review_items": (ReviewItem, "comment_id"),
}

# Fixed, stable read order for one extraction pass (no FK dependency
# order is required -- M1 finding: zero ForeignKey() declarations exist
# anywhere in models.py -- this order is kept only for reproducible logs).
TABLE_ORDER: tuple[str, ...] = (
    "users",
    "user_sessions",
    "review_sessions",
    "processing_jobs",
    "review_items",
)


def orm_class(table_name: str) -> type:
    return TABLE_SPECS[table_name][0]


def id_field(table_name: str) -> str:
    return TABLE_SPECS[table_name][1]


def columns_for(table_name: str):
    """All columns of the table, in declaration order, including the id
    column (callers that map id -> Mongo `_id` separately should skip it
    by name)."""
    return list(orm_class(table_name).__table__.columns)


def datetime_columns(table_name: str) -> set[str]:
    return {c.name for c in columns_for(table_name) if isinstance(c.type, DateTime)}


def not_null_columns(table_name: str) -> set[str]:
    return {c.name for c in columns_for(table_name) if not c.nullable}


# --- naive-datetime timezone convention, per (table, column) -------------
# "utc":    the app writer uses dt.datetime.utcnow() -- naive UTC.
#           Confirmed at: src/rcs/review/auth.py:46 (_now, user_sessions),
#           src/rcs/review/sessions.py:47 (_now, review_sessions +
#           processing_jobs), src/rcs/review/repository.py (apply_prefilter_
#           results / apply_classification_results, both `now = dt.datetime.
#           utcnow()`), src/rcs/ingest/cleaning.py (`ingested_at = as_of`,
#           caller-supplied `as_of` following the same utcnow() convention).
# "riyadh": the app writer uses rcs.review.clock.now_naive(), i.e. Asia/
#           Riyadh wall-clock stored as a naive value (see clock.py's own
#           docstring: "Stored naive datetimes are treated as wall-clock
#           times in APP_TIMEZONE"). Confirmed at: src/rcs/review/
#           service.py:52 (_now = now_naive()), used for updated_at,
#           generated_at, reviewer_started_at, reviewed_at,
#           outbound_attempted_at, outbound_sent_at.
# "source": copied directly from the source Excel column via
#           pandas.to_datetime(df["comment_created_at"]) (src/rcs/ingest/
#           cleaning.py:106) with NO timezone applied by the app at all.
#           Neither "utc" nor "riyadh" is an accurate label for this
#           value -- it is preserved as-is; the migration must not invent
#           a timezone for it, only preserve the wall-clock value exactly.
DATETIME_TZ_CONVENTION: dict[tuple[str, str], str] = {
    ("review_items", "comment_created_at"): "source",
    ("review_items", "ingested_at"): "utc",
    ("review_items", "processed_at"): "utc",
    ("review_items", "updated_at"): "riyadh",
    ("review_items", "generated_at"): "riyadh",
    ("review_items", "reviewer_started_at"): "riyadh",
    ("review_items", "reviewed_at"): "riyadh",
    ("review_items", "outbound_attempted_at"): "riyadh",
    ("review_items", "outbound_sent_at"): "riyadh",
    ("users", "created_at"): "utc",
    ("users", "updated_at"): "utc",
    ("user_sessions", "created_at"): "utc",
    ("user_sessions", "expires_at"): "utc",
    ("review_sessions", "created_at"): "utc",
    ("review_sessions", "updated_at"): "utc",
    ("processing_jobs", "started_at"): "utc",
    ("processing_jobs", "updated_at"): "utc",
    ("processing_jobs", "finished_at"): "utc",
}

# --- fields whose full value must never appear in any migration log, ------
# report, exception message, or debug output (Phase M2 amendment Section
# B). Extraction/transform/reconciliation code must route these through
# rcs.migration.logging_safe.short_ref() instead of printing them.
NEVER_LOG_FIELDS: set[tuple[str, str]] = {
    ("user_sessions", "session_token"),
    ("review_sessions", "upload_token"),
    ("users", "password_hash"),
    ("review_items", "visitor_name"),
    ("review_items", "visitor_email"),
    ("review_items", "original_comment"),
    ("review_items", "original_title"),
    ("review_items", "reviewer_edit"),
    ("review_items", "final_reply"),
    ("review_items", "immediate_reply_draft"),
    ("review_items", "suggested_final_reply"),
    ("review_items", "ai_immediate_reply_draft"),
    ("review_items", "ai_suggested_final_reply"),
    ("review_items", "notes"),
    ("review_items", "no_response_reason"),
}

# --- known enum-shaped columns whose membership is validated -------------
# High severity (block that document from staging) if violated -- these
# specifically drive the workflow state machine or access control
# (rcs.review.workflow.TRANSITIONS, rcs.review.auth role checks), so an
# unrecognized value is a real correctness risk, not cosmetic.
def _settings_enums():
    from config import settings

    return {
        ("review_items", "review_status"): set(settings.REVIEW_STATUSES),
        ("users", "role"): {"ADMIN", "REVIEWER"},
        ("review_sessions", "status"): {"uploaded", "processing", "ready", "completed", "failed"},
        ("processing_jobs", "status"): {"running", "succeeded", "failed"},
        ("review_items", "outbound_status"): set(settings.OUTBOUND_STATUSES),
        ("review_items", "reviewer_action"): set(settings.REVIEWER_ACTIONS) | set(settings.WORKFLOW_REVIEWER_ACTIONS),
    }


# Medium severity (informational only, migrate as-is) -- these describe
# classification/AI output, not workflow-state or access control.
def _informational_enums():
    from config import settings

    return {
        ("review_items", "primary_intent"): set(settings.PRIMARY_INTENTS),
        ("review_items", "ai_primary_intent"): set(settings.PRIMARY_INTENTS),
        ("review_items", "topic_subtype"): set(settings.TOPIC_SUBTYPES),
        ("review_items", "ai_topic_subtype"): set(settings.TOPIC_SUBTYPES),
        ("review_items", "sentiment"): set(settings.SENTIMENTS),
        ("review_items", "ai_sentiment"): set(settings.SENTIMENTS),
    }


HIGH_SEVERITY_ENUMS = _settings_enums
INFORMATIONAL_ENUMS = _informational_enums
