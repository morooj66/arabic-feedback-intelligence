"""
Phase M3 — PostgreSQL -> MongoDB migration tooling.

Isolated from the application's normal runtime paths: nothing under
rcs.review / rcs.ingest / rcs.response_generation imports this package,
and nothing here is imported by the served application
(scripts/serve_reviewer_api.py) or its HTTP routes. It exists purely to
extract, transform, reconcile, and load a MongoDB staging copy of the
data for migration-readiness verification -- PostgreSQL remains the
sole authoritative datastore throughout.

No module in this package may ever log, print, or include in any report
a raw secret value (MONGODB_URI, DATABASE_URL, password_hash,
session_token, upload_token) or visitor PII / raw comment/reply text.
See rcs.migration.logging_safe and rcs.migration.tables.NEVER_LOG_FIELDS.
"""
