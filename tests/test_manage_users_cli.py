"""Phase 6 — scripts/manage_users.py, the only way to provision accounts
(no public sign-up exists). Exercises create/list/deactivate/activate/
set-role/reset-password against an isolated temp DB."""

from __future__ import annotations

import argparse
import importlib

import pytest
from sqlalchemy.orm import Session

from rcs.review.auth import verify_password
from rcs.review.models import User
from rcs.review.repository import get_engine, init_db

manage_users = importlib.import_module("scripts.manage_users")


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    eng = get_engine(tmp_path / "cli.sqlite")
    init_db(eng)
    monkeypatch.setattr(manage_users, "_engine", lambda: eng)
    return eng


def test_create_never_prints_password_or_hash(engine, monkeypatch, capsys):
    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "a-real-password-123")
    args = argparse.Namespace(email="a@x.com", name="مراجع", role="REVIEWER")
    manage_users.cmd_create(args)
    out = capsys.readouterr().out
    assert "a-real-password-123" not in out
    assert "$2b$" not in out  # bcrypt hash prefix — never printed either

    with Session(engine) as db:
        user = db.execute(User.__table__.select().where(User.email == "a@x.com")).first()
        assert user is not None
        assert verify_password("a-real-password-123", user.password_hash)


def test_create_rejects_duplicate_email(engine, monkeypatch, capsys):
    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "a-real-password-123")
    manage_users.cmd_create(argparse.Namespace(email="a@x.com", name="A", role="REVIEWER"))
    with pytest.raises(SystemExit):
        manage_users.cmd_create(argparse.Namespace(email="a@x.com", name="A again", role="ADMIN"))


def test_create_rejects_invalid_role(engine, monkeypatch):
    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "a-real-password-123")
    with pytest.raises(SystemExit):
        manage_users.cmd_create(argparse.Namespace(email="a@x.com", name="A", role="SUPERUSER"))


def test_list_never_prints_password_hash(engine, monkeypatch, capsys):
    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "a-real-password-123")
    manage_users.cmd_create(argparse.Namespace(email="a@x.com", name="A", role="REVIEWER"))
    capsys.readouterr()
    manage_users.cmd_list(argparse.Namespace())
    out = capsys.readouterr().out
    assert "a@x.com" in out
    assert "REVIEWER" in out
    assert "$2b$" not in out


def test_activate_deactivate_and_disabled_login_rejected(engine, monkeypatch):
    from rcs.review.auth import authenticate

    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "a-real-password-123")
    manage_users.cmd_create(argparse.Namespace(email="a@x.com", name="A", role="REVIEWER"))
    assert authenticate(engine, email="a@x.com", password="a-real-password-123") is not None

    manage_users.cmd_activate(argparse.Namespace(email="a@x.com"), active=False)
    assert authenticate(engine, email="a@x.com", password="a-real-password-123") is None

    manage_users.cmd_activate(argparse.Namespace(email="a@x.com"), active=True)
    assert authenticate(engine, email="a@x.com", password="a-real-password-123") is not None


def test_set_role_changes_role(engine, monkeypatch):
    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "a-real-password-123")
    manage_users.cmd_create(argparse.Namespace(email="a@x.com", name="A", role="REVIEWER"))
    manage_users.cmd_set_role(argparse.Namespace(email="a@x.com", role="ADMIN"))
    with Session(engine) as db:
        user = db.execute(User.__table__.select().where(User.email == "a@x.com")).first()
        assert user.role == "ADMIN"


def test_reset_password_changes_the_hash(engine, monkeypatch):
    from rcs.review.auth import authenticate

    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "original-password-123")
    manage_users.cmd_create(argparse.Namespace(email="a@x.com", name="A", role="REVIEWER"))

    monkeypatch.setattr(manage_users, "_prompt_password", lambda: "brand-new-password-456")
    manage_users.cmd_reset_password(argparse.Namespace(email="a@x.com"))

    assert authenticate(engine, email="a@x.com", password="original-password-123") is None
    assert authenticate(engine, email="a@x.com", password="brand-new-password-456") is not None
