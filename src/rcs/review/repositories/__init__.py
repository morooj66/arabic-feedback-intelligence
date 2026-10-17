"""
Phase M4A — narrow persistence contracts extracted from the review
workflow, so PostgreSQL-specific query/session code lives in one place
(postgres.py) instead of scattered across service.py/sessions.py/
http_api.py/excel_review_flow.py.

PostgreSQL-only in this phase: there is no backend selector here, and no
Mongo implementation exists yet (see the Phase M4 audit report). `users`/
`user_sessions`/authentication are deliberately NOT part of this package
-- they remain direct PostgreSQL access in rcs.review.auth, unabstracted,
per the M4A scope correction (PostgreSQL stays the sole authority for
identity/roles while review data migrates).
"""
