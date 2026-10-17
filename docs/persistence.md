# Persistence backend: configuration and handoff

This document is for the technical team that will operate this application's
database going forward. It explains how the persistence backend is selected,
what Mongo needs, how to change which Mongo deployment/database is used, how
to run the index bootstrap, and what to verify before relying on Mongo in
production. It contains no credentials.

## How backend selection works

One environment variable decides everything: `PERSISTENCE_BACKEND`.

- Unset, or `PERSISTENCE_BACKEND=postgres` (default): the application uses
  PostgreSQL (or SQLite in dev/tests, via the same SQLAlchemy engine path),
  exactly as it always has. This is the fully backward-compatible path — no
  other configuration changes are required.
- `PERSISTENCE_BACKEND=mongo`: the application constructs every repository
  (`review_items`, `users`, `user_sessions`, `review_sessions`,
  `processing_jobs`, `analytics`, `audit`) from MongoDB instead. There is no
  partial/mixed mode — selecting `mongo` means all seven repositories come
  from Mongo, never a mix of the two.

This single read happens in exactly one place:
`rcs.review.persistence.build_repository_bundle()`, called once at process
startup from `scripts/serve_reviewer_api.py`. No other module — `auth.py`,
`sessions.py`, `job_runner.py`, `service.py`, `excel_review_flow.py`, the
HTTP handlers, the frontend, or the Agent code — reads `PERSISTENCE_BACKEND`
or contains any `if backend == "mongo"` branching. They all receive a
`RepositoryBundle` (or, unchanged, a raw SQLAlchemy engine for the Postgres
path) and call its repositories through the same typed contracts either way.

## Required Mongo configuration

Two environment variables, both required together:

- `MONGODB_URI` — the connection string for the target MongoDB deployment.
- `MONGODB_DATABASE` — the name of the database within that deployment to
  use.

If `PERSISTENCE_BACKEND=mongo` and either variable is missing or blank, the
application fails to start (`MissingPersistenceConfig`) rather than falling
back to PostgreSQL or guessing a database name. There is no default database
name baked into the application — it must always be supplied via
`MONGODB_DATABASE`.

If `PERSISTENCE_BACKEND` is set to anything other than `postgres` or
`mongo`, startup fails (`UnsupportedPersistenceBackend`) for the same
fail-closed reason.

## Changing which Mongo deployment/database is used

This is a configuration change only:

1. Update `MONGODB_URI` and/or `MONGODB_DATABASE` to point at the new
   deployment/database.
2. Run the index bootstrap against the new target (see below) before
   sending it real traffic.
3. Restart the application.

No application code, service logic, HTTP handler, frontend code, or Agent
code changes. No database name appears anywhere in `auth.py`, `sessions.py`,
`job_runner.py`, `service.py`, `excel_review_flow.py`, the HTTP API, the
frontend, or the Agent modules — grep for `MONGODB_DATABASE` and
`settings.MONGODB_URI` and the only hits are `config/settings.py` (where the
env vars are read), `rcs/review/persistence.py` (the composition boundary),
`scripts/serve_reviewer_api.py` (the one startup call site), and
`scripts/bootstrap_mongo_indexes.py` (the intentional bootstrap tool). Any
staging/test database name used by the migration tooling
(`rcs.migration.config.STAGING_DB_NAME`) lives only in that tool's own
configuration and is never read by the served application.

## Running the index bootstrap

Indexes are never created automatically — importing or starting the
application never touches indexes. A human runs the bootstrap script
intentionally, against whichever database they've configured:

```bash
# Review the plan first — this makes no network connection at all.
.venv/bin/python scripts/bootstrap_mongo_indexes.py --dry-run

# Then, with MONGODB_URI and MONGODB_DATABASE set in the environment:
.venv/bin/python scripts/bootstrap_mongo_indexes.py
```

The plan (`scripts/bootstrap_mongo_indexes.py::INDEX_PLAN`) creates:

| Collection | Index | Purpose |
|---|---|---|
| `users` | unique on `email` | enforce one account per email (Postgres already enforces this; Mongo currently doesn't without this index) |
| `review_items` | `reviewer_user_id` + `review_status` | reviewer-scoped queue queries |
| `review_items` | `source_file` | per-upload scoping |
| `review_items` | `review_status` | admin's global queue/status view |
| `review_items` | `source_file` + `content_dedupe_key` | exact-dedup canonical lookup |
| `processing_jobs` | `session_id` | job lookups scoped to one review session |
| `user_sessions` | TTL on `expires_at` (0s) | automatic cleanup of expired sessions |

`create_index` is idempotent — re-running the script against a database that
already has these indexes is always a safe no-op. It never drops or
silently replaces an existing index with an incompatible definition; a name
collision with a different definition raises loudly instead.

## Verifying configuration

- `scripts/bootstrap_mongo_indexes.py --dry-run` — confirms the index plan
  without connecting to anything; useful to sanity-check the script itself.
- Starting `scripts/serve_reviewer_api.py` with `PERSISTENCE_BACKEND=mongo`
  set prints `mongo database=<name>` (never the URI) on successful startup,
  or fails closed with a clear error if configuration is missing/invalid.
- `tests/test_persistence_bundle.py` proves the selection logic itself
  (fails closed on missing/invalid config, never mixes backends) without
  needing a real Mongo connection — run it any time to re-verify that
  contract.

## How PostgreSQL rollback works

PostgreSQL is not a legacy path being phased out — it is the default and the
reference implementation, and it remains fully functional. Simply leave
`PERSISTENCE_BACKEND` unset (or set it to `postgres`) to use it; no other
change is required. It should stay the operational backend until this
technical team has performed and verified their own Mongo cutover.

## What must be verified before Mongo cutover

This milestone (persistence decoupling) proves the application is *capable*
of running on Mongo — it does not itself constitute a cutover, and none of
the following has been done as part of it:

1. **Live connectivity** — the application has only been proven against an
   in-memory Mongo fake in tests. Real connectivity, credentials, and network
   access to the target Mongo deployment must be established and verified
   independently.
2. **Index creation** — run `scripts/bootstrap_mongo_indexes.py` against the
   target database and confirm all 7 indexes exist before sending it
   production traffic (especially the `users.email` unique index — without
   it, two concurrent signups with the same email can both succeed).
3. **Data parity/reconciliation** — any previously migrated or staging data
   in the target Mongo database must not be assumed current. Use the
   existing M3 migration tooling (`rcs.migration`) to migrate and reconcile
   data, and verify record counts/spot-check content, before cutover.
4. **Outbound-dispatch reliability boundary** (unchanged by the Mongo
   runtime boundary milestone that closed the persistence side of this —
   see below): `dispatch_outbound` is a two-phase write
   (`begin_outbound_dispatch` → external send → `complete_outbound_dispatch`)
   with no transactional outbox. A process crash between the first commit
   and the external call, or between the external call and the second
   commit, leaves the item stuck at `outbound_status="SENDING"` until an
   operator reconciles it by hand — on either backend, identically. This is
   pre-existing behavior, not introduced or worsened by making it
   backend-neutral. `UnconfiguredOutboundSender` remains the only outbound
   implementation ever wired in this application, so this path has never
   executed a real external send in production.

### Closed in the Mongo runtime boundary milestone

The following were flagged as remaining boundaries in the persistence
decoupling milestone and have since been closed — direct SQLAlchemy usage
was removed from the business layer in all four cases, so they are no
longer a Mongo-cutover blocker:

- `excel_review_flow.py`'s three raw-SQL helpers (`_ids_with_status`,
  `_next_batch_row_indices`, `_source_file_aggregate`) now read through
  `ReviewItemRepository.review_statuses` / `.comment_ids_for_source_file` /
  `.source_file_status_fields` — implemented for both backends.
- `service.py`'s `dispatch_outbound` now writes through
  `ReviewItemRepository.begin_outbound_dispatch` /
  `.complete_outbound_dispatch` — also implemented for both backends, with
  the reliability boundary above carried forward unchanged (never made
  worse, and not silently claimed to be fixed).
- The `comment_created_at` text-formatted-date defect (`compute_age()`
  parsing the column but not writing the parsed value back) is fixed in
  `rcs.ingest.cleaning.compute_age` — the column itself is normalized to a
  real datetime in place, and an unparseable date now raises a dedicated
  `InvalidCommentCreatedAtError` instead of silently reaching persistence
  with the wrong type.

`tests/test_persistence_boundary_architecture.py` structurally enforces that
`auth.py`, `sessions.py`, `job_runner.py`, `service.py`,
`excel_review_flow.py`, and `http_api.py` never import SQLAlchemy or a
concrete backend repository module directly — run it any time to re-verify
this boundary hasn't regressed. `tests/test_mongo_runtime_boundary_parity.py`
proves Postgres/Mongo behavioral parity for every operation above, against
the same in-memory Mongo fake used throughout this project's test suite —
not against a real MongoDB server.
