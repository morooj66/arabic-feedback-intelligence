"""Operator tool: prepare a DEDICATED demo database (local SQLite or hosted PostgreSQL) for the synthetic-data demo mode. Run it on the operator's own machine.

    DATABASE_URL=postgresql://USER:PASS@HOST/rcs_demo_online \\
    python scripts/demo/init_demo_environment.py --admin-email a@org --reviewer-email r1@org [--reviewer-email r2@org] --credentials-out ./demo_credentials.json [--reset]

  * refuses to touch any database whose name does not start with "rcs_demo" or when a production secret is in the environment (rcs.demo.guard);
  * applies the Alembic migrations (head);
  * creates ONE admin and the listed reviewers with strong random passwords written ONLY to --credentials-out (mode 600). Passwords are never printed, logged or put in Git;
  * imports a synthetic workbook through the real pipeline (rule classifier + local draft generator) so the demo starts with classified, drafted, unassigned comments;
  * --reset first empties every application table of that demo database (never anything else).
Invite-only: there are no default or public accounts; whoever holds the credentials file decides who gets which password."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--admin-email", required=True)
    ap.add_argument("--reviewer-email", action="append", required=True)
    ap.add_argument("--credentials-out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=36)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    from rcs.demo import guard

    env = {**os.environ, guard.MODE_VAR: guard.MODE_VALUE, guard.ACK_VAR: guard.ACK_VALUE}
    found = guard.problems(env, args.database_url)
    if found:
        raise SystemExit("refused:\n  - " + "\n  - ".join(found))
    if args.credentials_out.exists():
        raise SystemExit(f"refusing to overwrite {args.credentials_out}")
    os.environ.update({guard.MODE_VAR: guard.MODE_VALUE, guard.ACK_VAR: guard.ACK_VALUE, "DATABASE_URL": args.database_url})

    done = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=os.environ, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit("alembic upgrade head failed:\n" + done.stderr[-600:])

    from config import settings
    from rcs.demo import activation, synthetic
    from rcs.review.auth import ADMIN, REVIEWER, create_user
    from rcs.review.excel_review_flow import process_uploaded_excel_batch
    from rcs.review.models import Base
    from rcs.review.persistence import build_repository_bundle
    from rcs.review.repository import get_engine

    bundle, engine = build_repository_bundle(engine_factory=lambda: get_engine(settings.resolve_database_url(args.database_url)))
    if args.reset:
        with engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())
    else:
        with engine.begin() as conn:
            if conn.execute(Base.metadata.tables["users"].select().limit(1)).first() is not None:
                raise SystemExit("the demo database already has accounts; use --reset to rebuild it from scratch")
    activation.activate()                                       # from here on: rule classifier, local draft generator, no network

    creds = {}
    for role, email in [(ADMIN, args.admin_email)] + [(REVIEWER, e) for e in args.reviewer_email]:
        pw = secrets.token_urlsafe(16)
        u = create_user(bundle, email=email, password=pw, display_name=email.split("@")[0], role=role)
        creds[email.lower()] = {"role": role, "password": pw, "user_id": u.user_id}
    args.credentials_out.write_text(json.dumps(creds, ensure_ascii=False, indent=1))
    args.credentials_out.chmod(0o600)

    with tempfile.TemporaryDirectory() as td:
        wb = synthetic.write_demo_workbook(Path(td) / "synthetic_comments.xlsx", args.rows)
        from rcs.review.excel_review_flow import build_live_gemini_client

        summary = process_uploaded_excel_batch(engine, wb, batch_size=args.rows, client=build_live_gemini_client())
    print(f"demo database ready: {summary.batch_processed} synthetic comments imported, {len(creds)} accounts created; credentials file: {args.credentials_out} (mode 600; never printed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
