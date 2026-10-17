"""
Central configuration for the Response Classification System (RCS).

Phase 2 (ingest + cleaning) only relies on the PATHS, SOURCE_FILE, and
AGE_THRESHOLDS sections below. Everything else is defined here now so later
phases (prefilter, classification, routing, drafting, review) don't need to
invent new config files, and so the taxonomy/state enums exist in exactly
one place for the whole project.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "review.sqlite"

# ---------------------------------------------------------------------------
# Database configuration (Phase 5 — data-layer portability)
# ---------------------------------------------------------------------------
# The one env-driven boundary between the application and the relational
# database underneath it. Unset (the normal case today): every entry point
# that opts into this boundary (see resolve_database_url below) falls back
# to a local SQLite file. Set to a full SQLAlchemy URL — e.g.
# "postgresql+psycopg://user:pass@host/dbname" — to point the running
# server at PostgreSQL instead, with no application-code changes.
#
# Deliberately NOT read automatically by rcs.review.repository.get_engine()
# itself: that function keeps its exact current Path -> sqlite:// behavior
# for every existing caller (scripts, tests) so a stray DATABASE_URL in a
# developer's shell can never silently redirect a test run or a one-off
# script at another database. Only scripts/serve_reviewer_api.py (the
# actual served application) calls resolve_database_url() to opt in.
DATABASE_URL = os.environ.get("DATABASE_URL")


class DatabaseConfigError(RuntimeError):
    """The database configuration is unsafe or unusable. Raised instead of falling back to a different database."""


def _is_production_env() -> bool:
    # Read directly (not via rcs.review.identity) so configuration stays free of application imports.
    return os.environ.get("RCS_ENV", "").strip().lower() in {"production", "prod"}


def normalize_database_url(url: str) -> str:
    """Pin PostgreSQL URLs to the driver that is actually installed (psycopg v3).

    Hosts hand out "postgresql://..." or "postgres://..."; SQLAlchemy would then pick psycopg2, which this project does not install, and fail at first connect.
    A URL that already names a driver is left alone; anything that is not a PostgreSQL or SQLite URL is returned unchanged for SQLAlchemy to judge."""
    u = (url or "").strip()
    for prefix in ("postgresql://", "postgres://"):
        if u.lower().startswith(prefix):
            return "postgresql+psycopg://" + u[len(prefix):]
    return u


def resolve_database_url(default_path: Path) -> str:
    """DATABASE_URL, when set, wins outright (PostgreSQL in deployment). Otherwise falls back to a local SQLite file at default_path - the local-development default.
    This is the single place that decides which relational database backend a served instance uses.

    In production (RCS_ENV=production) a missing DATABASE_URL is a configuration ERROR, never a silent fallback: a container's local SQLite file is ephemeral, so
    reviewers' work would be lost on the next restart. A production URL must also be PostgreSQL on the installed psycopg v3 driver (SQLite URLs are refused)."""
    raw = DATABASE_URL                       # the module constant stays the single source (a dotenv-provided value is synced into it by the loaders), as before
    if _is_production_env():
        if not (raw or "").strip():
            raise DatabaseConfigError("DATABASE_URL is required when RCS_ENV=production (refusing to fall back to a local SQLite file).")
        url = normalize_database_url(raw)
        if not url.startswith("postgresql+psycopg://"):
            raise DatabaseConfigError("In production DATABASE_URL must be a PostgreSQL URL (postgresql+psycopg://...); got an unsupported scheme.")
        return url
    return normalize_database_url(raw) if (raw or "").strip() else f"sqlite:///{default_path}"


# ---------------------------------------------------------------------------
# Persistence backend selection (rcs.review.persistence)
# ---------------------------------------------------------------------------
# Which concrete repository implementation the served application's
# composition root (rcs.review.persistence.build_repository_bundle)
# constructs. "postgres" (the default, unchanged current behavior) uses
# DATABASE_URL/resolve_database_url exactly as before. "mongo" requires
# both MONGODB_URI and MONGODB_DATABASE below — there is no automatic
# fallback from mongo to postgres on missing/invalid Mongo config; see
# rcs.review.persistence for the fail-closed behavior.
PERSISTENCE_BACKEND = os.environ.get("PERSISTENCE_BACKEND", "postgres").strip().lower()

# MongoDB connection string. Never given a default, never logged, never
# read by anything outside rcs.review.persistence /
# rcs.review.repositories.mongo_client — the exact same "never printed"
# discipline already established for DATABASE_URL and used throughout
# rcs.migration.
MONGODB_URI = os.environ.get("MONGODB_URI")

# The Mongo database name is deployment configuration, never a business-
# logic constant: switching MONGODB_DATABASE (with a compatible
# MONGODB_URI) points the whole application at a different Mongo
# database with no code change anywhere in services/auth/sessions/jobs/
# the Excel flow/HTTP API. No default — an unset value fails closed
# (see rcs.review.persistence.build_repository_bundle) rather than
# silently landing writes in whatever default database MONGODB_URI's own
# connection string might otherwise imply.
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE")


# ---------------------------------------------------------------------------
# Phase 7 — deployment / CORS
# ---------------------------------------------------------------------------
# Comma-separated exact origins, e.g.
# "https://reviewer.example.com,https://reviewer-preview.vercel.app". Empty
# (the local-dev default) falls back to just the Vite dev server origins —
# never a wildcard, and never "reflect whatever Origin the request sent"
# (the pre-Phase-7 behavior). Set this in the real deployment environment;
# never widen it beyond the exact frontend origin(s) actually in use.
_DEFAULT_DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
# Reverse proxies whose X-Forwarded-For header may be believed (comma-separated IPs / CIDRs, e.g. "10.0.0.0/8,203.0.113.7"). EMPTY by default: the header is then ignored
# entirely and the direct socket peer is the client address. Never a wildcard. Used only for the login rate limiter.
TRUSTED_PROXIES = tuple(p.strip() for p in os.environ.get("RCS_TRUSTED_PROXIES", "").split(",") if p.strip())

# Self-service password reset. Needs BOTH a configured email sender (rcs.review.email_sender; none is wired by default, so the feature is OFF) and the public
# frontend origin the emailed link points at (e.g. https://rcs.example.org - no trailing slash). Token lifetime is capped at 30 minutes whatever the env says.
PASSWORD_RESET_LINK_BASE = os.environ.get("RCS_PASSWORD_RESET_LINK_BASE", "").strip().rstrip("/")
# SMTP delivery for password-reset e-mail. All empty by default = no sender = password reset stays OFF (fail closed). TLS is mandatory: "starttls" (port 587, the default)
# or "ssl" (port 465); there is deliberately no setting that allows plain-text SMTP. Credentials come from the environment only.
SMTP_HOST = os.environ.get("RCS_SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("RCS_SMTP_PORT", "587") or 587)
SMTP_SECURITY = (os.environ.get("RCS_SMTP_SECURITY", "starttls") or "starttls").strip().lower()
SMTP_USERNAME = os.environ.get("RCS_SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.environ.get("RCS_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("RCS_SMTP_FROM", "").strip()
PASSWORD_RESET_TOKEN_TTL_MINUTES = max(1, min(30, int(os.environ.get("RCS_PASSWORD_RESET_TTL_MINUTES", "30") or 30)))

# Public-endpoint request hardening. JSON request bodies are small by design (the largest legitimate one is a reviewer's reply, <= 600 characters, plus metadata), so a
# 256 KB cap is generous; Excel uploads keep their own cap (EXCEL_UPLOAD_MAX_BYTES). The per-socket timeout bounds slow-connection (slowloris-style) resource use:
# it applies to each blocking read/write, so a legitimate 10 MB upload on a slow link is unaffected while a stalled connection is dropped.
MAX_JSON_BODY_BYTES = max(1024, int(os.environ.get("RCS_MAX_JSON_BODY_BYTES", str(256 * 1024)) or 256 * 1024))
REQUEST_SOCKET_TIMEOUT_SECONDS = max(5, int(os.environ.get("RCS_REQUEST_SOCKET_TIMEOUT_SECONDS", "60") or 60))
# Origin enforcement for state-changing (POST) requests is ALWAYS on in production; RCS_ENFORCE_ORIGIN=1 turns it on elsewhere (e.g. a staging run).
ENFORCE_ORIGIN_CHECK = os.environ.get("RCS_ENFORCE_ORIGIN", "").strip().lower() in ("1", "true", "yes")

ALLOWED_ORIGINS = tuple(
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
) or _DEFAULT_DEV_ORIGINS

# ---------------------------------------------------------------------------
# Microsoft Entra ID SSO (Phase 9 — additive, alongside app-owned auth)
# ---------------------------------------------------------------------------
# Entra never creates or authorizes anyone on its own — it only verifies
# "this really is you" (an Authorization Code + PKCE OIDC flow, entirely
# server-side; see rcs.review.entra_auth). The existing users/user_sessions
# tables and REVIEWER/ADMIN roles remain the sole source of authorization
# after that — see entra_auth.resolve_local_user_for_entra_login(). All
# unset (the default, e.g. a fresh local checkout with no Entra app
# registered yet): ENTRA_CONFIGURED is False, the callback/start endpoints
# refuse cleanly, and the frontend simply doesn't show the Microsoft
# button (see /api/health's "entra" field) — password login is completely
# unaffected either way.
ENTRA_TENANT_ID = os.environ.get("ENTRA_TENANT_ID") or None
ENTRA_CLIENT_ID = os.environ.get("ENTRA_CLIENT_ID") or None
# Never logged, never sent to the frontend — used only for the
# server-to-server authorization-code token exchange.
ENTRA_CLIENT_SECRET = os.environ.get("ENTRA_CLIENT_SECRET") or None
# This backend's own callback URL — must be registered in the Entra app
# registration's Redirect URIs exactly as configured here. Never a
# hardcoded production domain.
#
# Deliberately on the SPA's own origin (localhost:5173), not the backend's
# real bind address (127.0.0.1:8765). Microsoft's post-login redirect is a
# real top-level browser navigation straight to this URL — it never goes
# through the Vite dev proxy the way the SPA's own fetch() calls do — so
# whatever host receives that navigation is the host the session cookie
# gets scoped to. The Vite dev server proxies any /api/* path (including
# this callback) through to the backend at 127.0.0.1:8765 (see
# web/vite.config.ts), so registering the callback on port 5173 keeps the
# whole login round trip on one browser-visible origin: the same one the
# SPA's later fetch("/api/session") calls use. Pointing this at
# 127.0.0.1:8765 instead would set the cookie on a host the SPA never
# talks to, and the user would land back on the app silently logged out.
ENTRA_REDIRECT_URI = os.environ.get("ENTRA_REDIRECT_URI") or "http://localhost:5173/api/auth/microsoft/callback"
# Where the browser is sent after a successful (or failed) SSO round trip
# — the frontend origin, not an API path. Local-dev default matches the
# Vite dev server.
ENTRA_FRONTEND_RETURN_URL = os.environ.get("ENTRA_FRONTEND_RETURN_URL") or "http://localhost:5173/"
ENTRA_CONFIGURED = bool(ENTRA_TENANT_ID and ENTRA_CLIENT_ID and ENTRA_CLIENT_SECRET)

# Normal (non-demo) local Excel operation: one persistent DB that survives
# server restarts and is never wiped by --seed. Distinct from DB_PATH
# (review.sqlite, the original batch-classification pipeline's DB — not
# used by the reviewer app) and from rcs.review.demo_seed.DEMO_DB
# (reviewer_ui_demo.sqlite, which --seed unlinks and recreates on every
# invocation). See scripts/serve_reviewer_api.py.
EXCEL_OPERATIONAL_DB = DB_DIR / "excel_review_demo.sqlite"

# Reviewer-app Excel upload (local/demo only — see rcs.review.excel_upload).
# Each upload gets its own unique subdirectory here; the original filename
# is preserved (sanitized) inside it, never the client-supplied path.
# RCS_UPLOAD_DIR points uploads at a persistent volume in a deployment (a container filesystem is ephemeral and an interrupted import could not resume after a restart).
EXCEL_UPLOAD_DIR = Path(os.environ["RCS_UPLOAD_DIR"]) if os.environ.get("RCS_UPLOAD_DIR") else DATA_DIR / "uploads"
EXCEL_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB — reject larger uploads outright

# Reviewer UI presentation and "today"/greeting calculations.
APP_TIMEZONE = "Asia/Riyadh"

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
# Per the data-discovery report: the second uploaded file
# (_تفاعل-الزوار-بعد-2026-03-31.xlsx) is a 100%-overlapping subset of this
# file and is deliberately NOT ingested, to avoid double-counting.
SOURCE_FILE = "historical-export.xlsx"

# Original (Arabic) -> internal (English) column name mapping.
# Keys must match the source workbook's header row exactly.
SOURCE_COLUMNS = {
    "العنوان": "original_title",
    "معرف المدخل": "entry_id",
    "اللما": "lemma",
    "المحتوى": "original_comment",
    "اسم الزائر": "visitor_name",
    " البريد الالكتروني للزائر": "visitor_email",  # note: leading space in source header
    "النوع": "original_type",
    "التاريخ": "comment_created_at",
    "المنصة": "platform",
}

# ---------------------------------------------------------------------------
# Age buckets (Section 17 of the data-discovery report; configurable)
# ---------------------------------------------------------------------------
AGE_RECENT_MAX_DAYS = 14      # 0-14 days  -> "recent"
AGE_DELAYED_MAX_DAYS = 180    # 14-180 days -> "delayed"; 180+ -> "old"

AGE_GROUPS = ("recent", "delayed", "old")

# ---------------------------------------------------------------------------
# Versioning / traceability (Correction 5)
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "0.1.0"
PROMPT_VERSION = "v1"  # not used until Phase 4, defined here so it's stable

# ---------------------------------------------------------------------------
# Taxonomy — hierarchical (MIGRATION_PLAN.md): PRIMARY_INTENT x TOPIC_SUBTYPE.
#
# Replaces the old flat 9-category CATEGORIES enum. PRIMARY_INTENT is the
# visitor's communicative intent (thanking, complaining, asking, etc.);
# TOPIC_SUBTYPE is what the message is about (a word submission, an entry
# correction, a technical bug, etc.). These are two independently-classified
# dimensions — Gemini must not infer one from the other's metadata.
# ---------------------------------------------------------------------------
# DUA_BLESSING was merged into THANKS_PRAISE 2026-08-24 (approved taxonomy
# decision, see FINAL_PRE_TRAINING_QA_AND_MERGE_AUDIT.md): identical routing
# (AUTO_REPLY_CANDIDATE) and no drafting-stage code ever differentiated them;
# it was also the sparsest class in the project after two dedicated sourcing
# rounds. All existing DUA_BLESSING gold rows outside human_test were
# relabeled THANKS_PRAISE in data/gold/gold_dataset_final_training.csv.
# human_test was NOT touched — it may still contain the old 6-class scheme
# and must be remapped at final-evaluation time, not before.
PRIMARY_INTENTS = (
    "THANKS_PRAISE",
    "COMPLAINT",
    "SUGGESTION",
    "INQUIRY",
    "SPAM",
)

TOPIC_SUBTYPES = (
    "WORD_SUBMISSION",
    "ENTRY_CORRECTION",
    "DICTIONARY_SUGGESTION",
    "TECHNICAL_ISSUE",
    "WORD_OF_YEAR_NOMINATION",
    "OFF_TOPIC_PERSONAL",
    "GENERAL",
)

SUBCATEGORIES = (
    "MeaningFault",
    "TechFault",
    "typo",
    "GrammerFault",
    "Other",
)

# Only populated when topic_subtype is one of these (was keyed off
# primary_category before the hierarchical migration; now keyed off topic).
TOPICS_REQUIRING_SUBCATEGORY = ("ENTRY_CORRECTION", "TECHNICAL_ISSUE")

# Sentiment — a separate signal from PRIMARY_INTENT (REVISED_ARCHITECTURE_PLAN.md
# v2, Section 3): the comment's TONE, independent of its communicative intent
# or topic. Used by the gold-labeling workbook (gold_sentiment) and, from
# Phase B.2 on, by the AI-assisted suggestion schema (ai_suggested_sentiment).
SENTIMENTS = ("POSITIVE", "NEGATIVE", "NEUTRAL")

# Classification provider composition. The frozen all-local stack remains the
# default. `hybrid_jev_topic` is experimental and replaces only Topic with the
# pinned Vendor Jev model through OpenRouter; Intent and Sentiment remain the
# existing frozen local classifiers. Invalid values fail closed when the
# classification runtime is invoked (never silently select another mode).
# "local_v2_pilot" = Intent I10 + frozen MARBERTv2 topic + Sentiment S5 (experimental classification_v2 artifacts). Local only: refused when RCS_ENV=production.
CLASSIFICATION_MODES = ("legacy_rcs", "hybrid_jev_topic", "local_v2_pilot")
V2_ARTIFACT_DIR = os.environ.get("RCS_V2_ARTIFACT_DIR") or None
CLASSIFICATION_MODE = os.environ.get("RCS_CLASSIFICATION_MODE", "legacy_rcs").strip().lower()
JEV_TOPIC_MODEL = "vendor/topic-model-1"
JEV_TOPIC_EXPECTED_SNAPSHOT = "vendor/topic-model-1-snapshot"
JEV_TOPIC_EXPECTED_PROVIDER = "Vendor"
JEV_TOPIC_ENDPOINT = "https://provider.example.invalid/api/decisions"
JEV_TOPIC_REQUEST_TIMEOUT_SECONDS = 30.0
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or None

# OPT-IN, DEFAULT OFF. Post-classification WORD_SUBMISSION / ENTRY_CORRECTION boundary layer
# (rcs.classification.topic_boundary_layer). Development evidence only; never enable in production before the independent
# blind evaluation (data/evaluation/blind_eval_v1) has been labelled and scored.
TOPIC_BOUNDARY_LAYER_ENABLED = os.environ.get("RCS_TOPIC_BOUNDARY_LAYER", "0").strip().lower() in ("1", "true", "yes")
# Contextual classification safeguards (rcs.classification.context_signals). DEFAULT OFF. When on, conflicts between a comment's structure and its predicted
# labels only ADD review flags (ai_needs_review) and move an EXCLUDE routing to REVIEW_ONLY; no label, confidence or model output is ever changed.
CONTEXT_SIGNALS_ENABLED = os.environ.get("RCS_CONTEXT_SIGNALS", "0").strip().lower() in ("1", "true", "yes")

ROUTING_DECISIONS = (
    "AUTO_REPLY_CANDIDATE",
    "ACKNOWLEDGE_THEN_REVIEW",
    "REVIEW_ONLY",
    "EXCLUDE",
)

# Deterministic primary_intent -> routing_decision lookup (MIGRATION_PLAN.md
# Section 4). Driven by intent, not topic — topic only overrides in the two
# cases below, both applied after this lookup.
INTENT_ROUTING = {
    "SPAM": "EXCLUDE",
    "THANKS_PRAISE": "AUTO_REPLY_CANDIDATE",
    # Leftover alias after the 2026-08-24 merge into THANKS_PRAISE (same routing).
    "DUA_BLESSING": "AUTO_REPLY_CANDIDATE",
    "COMPLAINT": "ACKNOWLEDGE_THEN_REVIEW",
    "SUGGESTION": "ACKNOWLEDGE_THEN_REVIEW",
    "INQUIRY": "ACKNOWLEDGE_THEN_REVIEW",
}

# Topic-based routing overrides — applied AFTER the intent lookup above,
# regardless of primary_intent. Only these two topics override; every other
# topic_subtype leaves the intent-driven routing_decision untouched.
# WORD_OF_YEAR_NOMINATION: low-volume, campaign-specific, needs a human call
# on whether the campaign is even active. OFF_TOPIC_PERSONAL: no dictionary/
# service action needed regardless of tone (decided, MIGRATION_PLAN.md
# Open Question B).
TOPIC_ROUTING_OVERRIDES = {
    "WORD_OF_YEAR_NOMINATION": "REVIEW_ONLY",
    "OFF_TOPIC_PERSONAL": "EXCLUDE",
}

# ---------------------------------------------------------------------------
# Review workflow (Section 3 of design spec v2 — no READY_TO_SEND status)
# ---------------------------------------------------------------------------
REVIEW_STATUSES = (
    "UNPROCESSED",
    "CLASSIFIED",
    "EXCLUDED",
    # Non-canonical member of an exact-content duplicate group (Excel
    # operational flow). Terminal, non-reviewer-facing — set directly by
    # rcs.review.excel_review_flow.run_content_deduplication (same pattern
    # as EXCLUDED), never via the reviewer workflow state machine. Bypasses
    # classification, generation, and the reviewer queue entirely; the
    # canonical row (dedupe_canonical_id) carries the real decision.
    "DUPLICATE_CONTENT",
    "DRAFT_READY",
    "REVIEW_ONLY",
    "NO_RESPONSE_RECOMMENDED",
    "IN_REVIEW",
    "APPROVED",
    "EDITED_AND_APPROVED",
    "NO_REPLY",
    "SENDING",
    "SENT",
    "SEND_FAILED",
    "REJECTED",
)

READY_TO_SEND_STATUSES = ("APPROVED", "EDITED_AND_APPROVED")

# Legacy six-action vocabulary (classification change / exclude / unused reject).
REVIEWER_ACTIONS = (
    "approve",
    "edit_approve",
    "change_classification",
    "exclude",
    "no_reply",
    "reject",
)

# Operational reviewer actions for the response-generation workflow.
WORKFLOW_REVIEWER_ACTIONS = (
    "SEND_AS_IS",
    "EDIT_AND_SEND",
    "NO_RESPONSE_NEEDED",
)

REVIEWER_ACTION_TO_STATUS = {
    "approve": "APPROVED",
    "edit_approve": "EDITED_AND_APPROVED",
    "change_classification": "CLASSIFIED",
    "exclude": "EXCLUDED",
    "no_reply": "NO_REPLY",
    "reject": "REJECTED",
    "SEND_AS_IS": "APPROVED",
    "EDIT_AND_SEND": "EDITED_AND_APPROVED",
    "NO_RESPONSE_NEEDED": "NO_REPLY",
}

# NO_RESPONSE_RECOMMENDED is a model suggestion, not a human decision: it
# stays in the human work queue until a reviewer confirms (NO_REPLY) or
# overrides it (claim / EDIT_AND_SEND).
QUEUE_PENDING_STATUSES = ("DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW", "NO_RESPONSE_RECOMMENDED")
OUTBOUND_STATUSES = ("NOT_APPLICABLE", "PENDING", "SENDING", "SENT", "FAILED")

# Phase M5B.3 — Approved Response Memory governance: source_file is
# uploader-controlled free text (src/rcs/ingest/loader.py,
# src/rcs/review/excel_upload.py) with no existing trusted-source
# concept. This is a POSITIVE allowlist only — no filename heuristic is
# or may be used instead. Default is EMPTY: with no organization-
# approved sources configured, the approved-response corpus has zero
# eligible entries by design (fails closed), never falling back to an
# unlisted source. Set via a comma-separated env var of exact
# `source_file` values this organization has explicitly approved as
# production-trustworthy for style-memory retrieval.
APPROVED_RESPONSE_SOURCE_ALLOWLIST = frozenset(
    s.strip() for s in os.environ.get("APPROVED_RESPONSE_SOURCE_ALLOWLIST", "").split(",") if s.strip()
)

# Phase M5B.4 — Official Knowledge governance: a local audit (see
# official_corpus.py's module docstring) found no existing trustworthy
# local dictionary/lexicon file in this repository. Default is unset:
# with no path configured, no source is even attempted and Official
# Knowledge retrieval always returns SUCCESS_EMPTY (fails closed, never
# falls back to Approved Response Memory or any other data). Set this
# to a local JSON file path only once an organization-approved,
# provenance-tagged official knowledge export exists — see
# official_corpus.py for the required schema.
OFFICIAL_KNOWLEDGE_SOURCE_PATH = os.environ.get("OFFICIAL_KNOWLEDGE_SOURCE_PATH") or None

# Phase M5C — fixed, deterministic per-request retrieval caps (not an
# agent decision, not runtime-tunable per call). Small on purpose: this
# is evidence assembled into ONE bounded prompt, not a retrieval-heavy
# workflow.
OFFICIAL_KNOWLEDGE_MAX_RESULTS_PER_QUERY = 3
APPROVED_RESPONSE_MEMORY_MAX_RESULTS_PER_QUERY = 2

# ---------------------------------------------------------------------------
# Deterministic prefilter (Phase 3 — constants defined here in advance)
# ---------------------------------------------------------------------------
KNOWN_JUNK_STRINGS = (
    # Confirmed bot flood from data discovery: 3,875 identical rows on
    # 2025-02-17 from a single "test" account.
    "jhnjkn njnun Phone Number: 111111111111j",
)
BURST_MIN_COUNT = 10   # same account/content repeated >= this many times
BURST_WINDOW_HOURS = 24

# ---------------------------------------------------------------------------
# Gemini (Phase 4)
# ---------------------------------------------------------------------------
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_API_KEY = os.environ.get(GEMINI_API_KEY_ENV_VAR)  # None until set; only needed for a live call

# ---------------------------------------------------------------------------
# Response generation (visitor-facing drafts — never auto-sent)
# ---------------------------------------------------------------------------
# Distinct from INTENT_ROUTING / ROUTING_DECISIONS (classification "what to
# do operationally"). GENERATE_DRAFT for meaningful comments that deserve a
# reply or reviewer assistance. NO_RESPONSE_RECOMMENDED skips Gemini for
# spam, off-topic, empty/invalid, and other non-actionable content.
# REVIEW_ONLY is used when generation itself fails closed.
RESPONSE_ELIGIBILITY_DECISIONS = (
    "GENERATE_DRAFT",
    "REVIEW_ONLY",
    "NO_RESPONSE_RECOMMENDED",
)

# Future reviewer actions once a draft (or a no-draft routing) is queued.
# Not wired to persistence or an external platform in this phase. Distinct from
# REVIEWER_ACTIONS (approve / edit_approve / no_reply / …).
FUTURE_REVIEWER_RESPONSE_ACTIONS = (
    "SEND_AS_IS",
    "EDIT_AND_SEND",
    "NO_RESPONSE_NEEDED",
)

# v1: never send an automatic acknowledgement. Keep the flag so ACK can be
# enabled later without inventing a new config surface.
AUTOMATIC_ACK_ENABLED = False

# rg-v11: promoted from rg-v10 because that change altered actual
# prompt content (a new Hard Rule plus tightened situation hints in
# rcs.response_generation.prompt.SYSTEM_INSTRUCTIONS) -- see that
# module's docstring for the exact content diff. rg-v10 remains the
# historically-accurate label for every execution/record/evaluation
# artifact produced before this change; it is not renamed or migrated.
#
# rg-v12: promoted from rg-v11 for the same reason -- a new Hard Rule
# and situation hint were added to SYSTEM_INSTRUCTIONS (see that
# module's docstring). rg-v11 remains the historically-accurate label
# for every execution/record/evaluation artifact produced before this
# change.
#
# rg-v13: promoted from rg-v12 for the same reason -- the existing
# rg-v10 composition instruction in SYSTEM_INSTRUCTIONS was extended to
# name an additional detached-topic-preamble connector-verb family (see
# that module's docstring). rg-v12 remains the historically-accurate
# label for every execution/record/evaluation artifact produced before
# this change.
#
# rg-v14: context-aware response selection and the explicitly approved
# dictionary homepage were added after the first live baseline. Definitions
# and disputed linguistic facts remain Official-Knowledge-only; see
# prompt.py for the exact constraints and historical rationale.
#
# rg-v14.1: focused quality corrections from the completed 18-case live
# evaluation: more natural concise Arabic, stricter navigation intent,
# and explicit suggestion-evidence phrases in the deterministic gate.
# Routing, eligibility, grounding, and review policy are unchanged.
#
# rg-v14.2: focused corrections from the rg-v14.1 five-case live
# regression. Routing, eligibility, grounding, and review policy remain
# unchanged.
RESPONSE_GENERATION_PROMPT_VERSION = "rg-v14.4"
RESPONSE_GENERATION_PROVIDER = "gemini"

# The single homepage URL that generated replies may mention (the response prompt
# names it, and the claim gate treats it as the only approved link). Neutral
# placeholder by default; set RCS_REFERENCE_HOME_URL to your own dictionary homepage.
REFERENCE_HOME_URL = os.environ.get("RCS_REFERENCE_HOME_URL", "https://dictionary-platform.invalid/home").strip()

RESPONSE_DRAFT_MIN_CHARS = 12
RESPONSE_DRAFT_MAX_CHARS = 600
RESPONSE_DRAFT_MIN_ARABIC_LETTERS = 8
# Frozen classifiers are weak; treat mid/low self-reported or model
# probability as unsafe to auto-draft. None confidence skips this gate.
RESPONSE_LOW_CONFIDENCE_THRESHOLD = 0.45

# Reply-safety / classification-safety policy (default ON; the flags exist so the previous behaviour can be restored deliberately and
# so legacy unit tests of downstream machinery can run without an official-knowledge source).
#  * A comment that asks for facts is routed to a human (no generated draft) while no approved OFFICIAL KNOWLEDGE is available.
#  * A SPAM / OFF_TOPIC_PERSONAL label that is not corroborated by a second signal is routed to human review, never to "no response".
RESPONSE_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS = os.environ.get("RCS_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS", "1") != "0"
# Reviewer attestation of factual statements that cannot be verified against approved grounding (definitions, verdicts on an entry,
# statistics). Never bypasses URLs, commitments/promises, institutional policies, operational claims, security checks or role permissions.
# Roles are matched against the AUTHENTICATED session role; a caller with no role can never attest.
FACT_ATTESTATION_ENABLED = os.environ.get("RCS_FACT_ATTESTATION_ENABLED", "1") != "0"
FACT_ATTESTATION_ROLES = frozenset(
    r.strip().upper() for r in os.environ.get("RCS_FACT_ATTESTATION_ROLES", "REVIEWER,ADMIN").split(",") if r.strip()
)
RESPONSE_REQUIRE_CORROBORATION_FOR_EXCLUSION = os.environ.get("RCS_REQUIRE_CORROBORATION_FOR_EXCLUSION", "1") != "0"

# Phase B.2 — AI-assisted gold-label suggestion. Separate version string from
# PROMPT_VERSION (Phase 4's live-classification prompt) since it's a
# different prompt/schema (see ai_suggestion_schema.py, suggestion_prompt.py).
# Recorded verbatim in ai_suggestion_model so every suggestion is traceable
# to exactly the model+prompt version that produced it.
#
# b2-v3: adds LABELING_GUIDE-derived disambiguation block to suggestion_prompt.py
# (WORD_SUBMISSION vs ENTRY_CORRECTION, COMPLAINT vs SUGGESTION, INQUIRY cues,
# SPAM scope, DUA vs THANKS). No taxonomy or original_type metadata change.
# b2-v2: migrated from the deprecated `google-generativeai` SDK/gemini-2.5-flash
# to the current `google-genai` SDK (src/rcs/evaluation/genai_client.py) and
# gemini-3.6-flash, after a live local smoke test against gemini-2.5-flash
# returned 404 (model no longer available to new users; confirmed via
# ai.google.dev docs that it shuts down Oct 2026 and gemini-3.6-flash is the
# current GA Flash model). Phase 4 (src/rcs/classification/) intentionally
# NOT touched by this change — out of scope, still on the old SDK/model.
AI_SUGGESTION_PROMPT_VERSION = "b2-v3"

# Separate from GEMINI_MODEL_NAME (Phase 4's model, untouched) on purpose —
# Phase B.2 and Phase 4 are allowed to move to new models independently.
AI_SUGGESTION_MODEL_NAME = "gemini-3.6-flash"

# Default follows Phase B.2 Flash (gemini-2.5-flash is 404 for new keys).
RESPONSE_GENERATION_MODEL_NAME = os.environ.get(
    "GEMINI_RESPONSE_MODEL_NAME",
    AI_SUGGESTION_MODEL_NAME,
)

# Bounded retry for transient API/network failures during a single
# classification call. Not a retry-forever loop — after this many attempts,
# the row falls back to REVIEW_ONLY (Section 4 of DESIGN_SPEC.md) and is
# left eligible for a future pipeline run to retry it again.
GEMINI_MAX_RETRIES = 2

# Phase M5A.2 — per-attempt request timeout (seconds), passed straight to
# the google-genai SDK's own `timeout` call parameter. Per-attempt, not a
# total request budget: each of the (at most GEMINI_MAX_RETRIES + 1)
# attempts in _call_with_retry gets this full bound again, since a
# retry is a fresh HTTP request. One structured, short (<= a few hundred
# Arabic characters) draft should complete well inside this; 20s leaves
# real headroom without ever waiting on a hung connection indefinitely.
GEMINI_REQUEST_TIMEOUT_SECONDS = 20.0

# Phase M5A.2 — bounded backoff between RETRYABLE attempts only (never
# before the first attempt, never after the final one). Simple bounded
# exponential: base * 2**attempt_index, capped at the max below.
# attempt_index 0 -> 0.5s, 1 -> 1.0s, 2 -> 2.0s, ... capped at 4.0s.
# No jitter: at this call volume (one Excel batch job at a time, not a
# high-QPS service under real contention) jitter's thundering-herd
# benefit doesn't apply, and it would only make backoff timing
# non-deterministic in tests for no real production benefit here.
GEMINI_RETRY_BACKOFF_BASE_SECONDS = 0.5
GEMINI_RETRY_BACKOFF_MAX_SECONDS = 4.0

# Digit-sequence length (after stripping separators) at or above which a
# run of digits inside a comment is treated as a phone number and redacted
# before the comment is sent to Gemini. Chosen to catch Saudi mobile/
# landline numbers (8-10 digits) and international formats while leaving
# short numbers (years, counts) alone. See Open Assumptions in
# DESIGN_SPEC.md.
PHONE_REDACTION_MIN_DIGITS = 7
PHONE_REDACTION_PLACEHOLDER = "[phone number removed]"

# ---------------------------------------------------------------------------
# Interactive AI Evaluation Playground (internal, admin-only, local-dev tool).
# DISABLED BY DEFAULT: nothing under /api/admin/playground/* does anything
# unless RCS_PLAYGROUND_ENABLED is explicitly set, and it is additionally
# hard-disabled when RCS_ENV=production regardless of this flag. Results are
# stored only when a user explicitly saves a case, in JSONL files that live
# outside every gold/production dataset (and are git-ignored).
# ---------------------------------------------------------------------------
PLAYGROUND_ENABLED = os.environ.get("RCS_PLAYGROUND_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
PLAYGROUND_DATA_DIR = Path(os.environ.get("RCS_PLAYGROUND_DATA_DIR") or (BASE_DIR / "data" / "evaluation" / "playground"))
PLAYGROUND_MAX_VARIANTS = 10
PLAYGROUND_MAX_CHARS = 1000
