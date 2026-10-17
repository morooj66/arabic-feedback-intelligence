"""Serve the thin Reviewer UI HTTP adapter (stdlib http.server).

Two distinct startup modes, kept intentionally separate:

  Normal (default, no flags) — local Excel operation. Never reseeds.
  Persists across restarts on settings.EXCEL_OPERATIONAL_DB
  (data/db/excel_review_demo.sqlite). Loads .env explicitly (GEMINI_API_KEY
  etc.) regardless of --seed, so live rg-v7 generation doesn't silently
  depend on demo seeding ever having run in this process.

    python scripts/serve_reviewer_api.py

  Demo (opt-in) — synthetic seed rows for UI demos only. --seed
  unconditionally deletes and recreates its own DB
  (rcs.review.demo_seed.DEMO_DB, reviewer_ui_demo.sqlite) on every launch;
  it must never be the default for real operational use.

    python scripts/serve_reviewer_api.py --seed [--live-seed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from rcs.review.demo_seed import DEMO_DB, load_dotenv_if_present, seed_demo_db  # noqa: E402
from rcs.review.http_api import serve  # noqa: E402
from rcs.review.job_runner import resume_orphaned_jobs  # noqa: E402
from rcs.review.persistence import build_repository_bundle  # noqa: E402
from rcs.review.repository import get_engine, init_db  # noqa: E402


def main() -> None:
    # Render (and most container hosts) inject the port to listen on via
    # $PORT and expect a bind to 0.0.0.0 — RCS_ENV=production is the same
    # flag identity.py already uses to distinguish real deployments from
    # local dev, so it doubles as the signal for these defaults too.
    # --host/--port always override both, for any environment.
    import os as _os

    _deployed = _os.environ.get("RCS_ENV", "").strip().lower() in {"production", "prod"}
    _default_host = "0.0.0.0" if _deployed else "127.0.0.1"
    _default_port = int(_os.environ.get("PORT", "8765"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=_default_host)
    parser.add_argument("--port", type=int, default=_default_port)
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "Explicit DB path override. Default (no flag): "
            "excel_review_demo.sqlite normally, or reviewer_ui_demo.sqlite "
            "with --seed."
        ),
    )
    parser.add_argument("--seed", action="store_true", help="Recreate isolated demo SQLite before serving")
    parser.add_argument("--live-seed", action="store_true", help="With --seed, generate drafts via current Gemini pipeline")
    parser.add_argument(
        "--production-db",
        action="store_true",
        help="Use data/db/review.sqlite instead of the isolated UI demo file",
    )
    args = parser.parse_args()

    # Production fails closed: the local-file selectors exist for development and demos (--seed even DELETES the file it points at), so they are refused outright
    # on a deployed instance. A production start must name its database through DATABASE_URL (or PERSISTENCE_BACKEND=mongo) and nothing else.
    if _deployed and (args.seed or args.live_seed or args.db is not None or args.production_db):
        raise SystemExit("FATAL: --seed/--live-seed/--db/--production-db are development-only and are refused when RCS_ENV=production. Configure DATABASE_URL instead.")

    # Synthetic-data DEMO MODE (rcs.demo): off unless an RCS_DEMO_* variable is present. When requested, every guard in rcs.demo.guard must pass or the process exits;
    # a demo process never reads .env (no secrets can leak into it) and never runs with the development database flags. A normal/production start skips all of this
    # and never imports the demo package beyond the False-by-default state flag.
    from rcs.demo.guard import demo_requested

    _demo = demo_requested(_os.environ)
    if _demo:
        from rcs.demo import activation as _demo_activation

        _demo_activation.preflight_or_exit(
            _os.environ, _os.environ.get("DATABASE_URL"),
            dev_db_flags_used=bool(args.seed or args.live_seed or args.db is not None or args.production_db),
        )
    else:
        # Explicit, unconditional — Gemini availability must not depend on
        # --seed having been passed (seed_demo_db() also calls this, but a
        # normal/non-seed startup must not rely on that side effect).
        load_dotenv_if_present()

    # Persistence-decoupling milestone: config.settings.PERSISTENCE_BACKEND
    # is the ONE switch that decides Postgres vs. Mongo, read exactly once,
    # here, by rcs.review.persistence.build_repository_bundle — never
    # scattered as an "if backend == mongo" through auth/sessions/
    # job_runner/service/excel_review_flow. --db/--seed/--production-db
    # below are PostgreSQL-only local-file selectors and stay exactly as
    # they were for that backend; they have no meaning for Mongo (whose
    # target is MONGODB_URI/MONGODB_DATABASE only) and are rejected
    # outright rather than silently ignored if both are given together.
    if settings.PERSISTENCE_BACKEND == "mongo":
        if args.db is not None or args.seed or args.production_db:
            raise SystemExit(
                "--db/--seed/--production-db are PostgreSQL-only flags; "
                "PERSISTENCE_BACKEND=mongo does not use a local DB file. "
                "Unset PERSISTENCE_BACKEND for the local/Postgres path, or "
                "drop these flags."
            )
        bundle, engine = build_repository_bundle()
        mode = "mongo"
        # Never the connection string — only the database name, matching
        # the "never print MONGODB_URI" discipline used everywhere else
        # this project touches Mongo (rcs.migration, mongo_client.py).
        db_display = f"mongo database={settings.MONGODB_DATABASE!r}"
    else:
        # DATABASE_URL only applies to the normal (no-flag) served-
        # application path — Phase 5's actual portability boundary.
        # --seed/--db/--production-db are explicit local-file developer/
        # demo overrides (--seed in particular deletes-and-recreates a
        # SQLite *file*, which has no meaning for a non-file DATABASE_URL)
        # and stay exactly as they were: plain local SQLite Path
        # selection, unaffected.
        if args.production_db:
            db_target = settings.DB_PATH
            mode = "production-db"
        elif args.db is not None:
            db_target = args.db
            mode = "explicit --db"
        elif args.seed:
            db_target = DEMO_DB
            mode = "demo (--seed)"
        else:
            try:
                db_target = settings.resolve_database_url(settings.EXCEL_OPERATIONAL_DB)
            except settings.DatabaseConfigError as exc:
                raise SystemExit(f"FATAL: {exc}") from None
            mode = "normal Excel operation (no reseed)"

        if args.seed and not args.production_db:
            db_target = seed_demo_db(db_target, live=True if args.live_seed else None)

        bundle, engine = build_repository_bundle(engine_factory=lambda: get_engine(db_target))
        init_db(engine)
        # engine.url renders with the password masked (***) — never log a
        # raw DATABASE_URL, which may carry real PostgreSQL credentials.
        db_display = f"db={engine.url}"

    # Phase 7: recover any job a previous process left stuck at "running"
    # (a container restart mid-job) — see job_runner.resume_orphaned_jobs
    # docstring. Safe/idempotent to run on every startup, including the
    # common case where there was nothing to resume. Works identically
    # for either backend: resume_orphaned_jobs normalizes `bundle` through
    # the same rcs.review.persistence.resolve_bundle() every other
    # runtime module uses.
    if _demo:
        _demo_activation.activate()
    resumed = resume_orphaned_jobs(bundle)
    if resumed:
        print(f"resumed {len(resumed)} orphaned processing job(s) for session(s): {resumed}")
    httpd = serve(bundle, host=args.host, port=args.port)
    print(f"reviewer API on http://{args.host}:{args.port} {db_display} mode={mode}")
    print(f"GEMINI_API_KEY available: {bool(settings.GEMINI_API_KEY)}")
    if _demo:
        print("DEMO MODE: synthetic data only; rule-based classifier and local draft generator; invite-only accounts; outbound network disabled")
    print("demo auth only — Microsoft Entra is not connected")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
