"""Phase 6 — the ONLY way to create/manage accounts. No public sign-up
exists anywhere in this application; every user is provisioned here, by
whoever has shell access to the server, not by an HTTP endpoint.

Usage:
    python scripts/manage_users.py create --email a@b.com --role REVIEWER --name "..."
        (prompts for a password interactively — never pass it as an argument,
        it would end up in shell history)
    python scripts/manage_users.py list
    python scripts/manage_users.py deactivate --email a@b.com
    python scripts/manage_users.py activate --email a@b.com
    python scripts/manage_users.py set-role --email a@b.com --role ADMIN
    python scripts/manage_users.py reset-password --email a@b.com
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from config import settings  # noqa: E402
from rcs.review.auth import ROLES, create_user, hash_password  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402
from rcs.review.models import User  # noqa: E402
from rcs.review.repository import get_engine  # noqa: E402


def _engine():
    load_dotenv_if_present()
    url = settings.resolve_database_url(settings.EXCEL_OPERATIONAL_DB)
    return get_engine(url)


def _prompt_password() -> str:
    while True:
        pw = getpass.getpass("Password (input hidden, min 8 chars): ")
        if len(pw) < 8:
            print("Password must be at least 8 characters.", file=sys.stderr)
            continue
        confirm = getpass.getpass("Confirm password: ")
        if pw != confirm:
            print("Passwords did not match, try again.", file=sys.stderr)
            continue
        return pw


def cmd_create(args: argparse.Namespace) -> None:
    if args.role not in ROLES:
        print(f"invalid role {args.role!r}, must be one of {ROLES}", file=sys.stderr)
        sys.exit(1)
    password = _prompt_password()
    try:
        user = create_user(_engine(), email=args.email, password=password, display_name=args.name, role=args.role)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"created user_id={user.user_id} email={user.email} role={user.role} is_active={user.is_active}")


def cmd_list(_args: argparse.Namespace) -> None:
    with Session(_engine()) as db:
        users = db.execute(select(User).order_by(User.created_at)).scalars().all()
    if not users:
        print("(no users)")
        return
    for u in users:
        # never print password_hash
        print(f"{u.user_id}  {u.email:<30}  {u.role:<10}  active={u.is_active}  {u.display_name}")


def _find_user(db: Session, email: str) -> User:
    email = email.strip().lower()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        print(f"no user with email {email!r}", file=sys.stderr)
        sys.exit(1)
    return user


def cmd_activate(args: argparse.Namespace, *, active: bool) -> None:
    with Session(_engine()) as db:
        user = _find_user(db, args.email)
        user.is_active = active
        user.updated_at = dt.datetime.utcnow()
        db.commit()
    print(f"{'activated' if active else 'deactivated'} {args.email}")


def cmd_set_role(args: argparse.Namespace) -> None:
    if args.role not in ROLES:
        print(f"invalid role {args.role!r}, must be one of {ROLES}", file=sys.stderr)
        sys.exit(1)
    with Session(_engine()) as db:
        user = _find_user(db, args.email)
        old_role = user.role
        user.role = args.role
        user.updated_at = dt.datetime.utcnow()
        db.commit()
    print(f"changed role for {args.email}: {old_role} -> {args.role}")


def cmd_reset_password(args: argparse.Namespace) -> None:
    password = _prompt_password()
    with Session(_engine()) as db:
        user = _find_user(db, args.email)
        user.password_hash = hash_password(password)
        user.updated_at = dt.datetime.utcnow()
        db.commit()
    print(f"password reset for {args.email}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new user (password prompted interactively)")
    p_create.add_argument("--email", required=True)
    p_create.add_argument("--name", required=True, help="Display name")
    p_create.add_argument("--role", required=True, choices=ROLES)

    sub.add_parser("list", help="List all users (never prints password hashes)")

    p_deact = sub.add_parser("deactivate", help="Deactivate a user (cannot log in while inactive)")
    p_deact.add_argument("--email", required=True)

    p_act = sub.add_parser("activate", help="Reactivate a user")
    p_act.add_argument("--email", required=True)

    p_role = sub.add_parser("set-role", help="Change a user's role")
    p_role.add_argument("--email", required=True)
    p_role.add_argument("--role", required=True, choices=ROLES)

    p_reset = sub.add_parser("reset-password", help="Reset a user's password (prompted interactively)")
    p_reset.add_argument("--email", required=True)

    args = parser.parse_args()
    if args.command == "create":
        cmd_create(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "deactivate":
        cmd_activate(args, active=False)
    elif args.command == "activate":
        cmd_activate(args, active=True)
    elif args.command == "set-role":
        cmd_set_role(args)
    elif args.command == "reset-password":
        cmd_reset_password(args)


if __name__ == "__main__":
    main()
