"""
Phase M4D.1 — UserRepository contract + parity tests.

Postgres and Mongo scenarios are placed side by side per behavior, same
convention as the other repository test files. The Mongo tests run
against the same in-memory fake used in test_mongo_repository_unit.py --
they prove THIS CODE's logic, not MongoDB's own server-side uniqueness
guarantee (there is no real unique index yet -- see the M4D.1 report).

Scope note: no `update`/`activate`/`deactivate` method exists on
UserRepository (category "I. update user" is therefore not applicable
and intentionally not tested here) because the M4D.1 audit found no
runtime HTTP endpoint that performs one -- only scripts/manage_users.py
(an offline CLI) does, and it stays untouched, still talking to the
`users` table directly. Security: every password used below is a
synthetic bcrypt hash literal, never a real credential; no test prints
or logs a password_hash value.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rcs.review.models import User
from rcs.review.repositories import contracts
from rcs.review.repositories.mongo import MongoUserRepository
from rcs.review.repositories.postgres import PostgresUserRepository
from rcs.review.repository import get_engine, init_db

from test_mongo_repository_unit import _FakeDatabase  # noqa: E402 -- reuse the one in-memory fake

_SYNTHETIC_HASH = "$2b$12$synthetichashsynthetichashsynthetichashsyn"


def _engine(tmp_path, name="users.sqlite"):
    engine = get_engine(tmp_path / name)
    init_db(engine)
    return engine


def _pg_repo(tmp_path):
    return PostgresUserRepository(_engine(tmp_path))


def _mongo_repo():
    return MongoUserRepository(_FakeDatabase())


def _create_kwargs(**overrides):
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    kwargs = dict(
        user_id="u-1",
        email="a@x.com",
        password_hash=_SYNTHETIC_HASH,
        display_name="A",
        role="REVIEWER",
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    kwargs.update(overrides)
    return kwargs


# A. Contract/signature ------------------------------------------------


def test_contract_declares_user_repository_methods():
    for name in ("get_by_id", "get_by_email", "list_all", "list_active_reviewers", "create"):
        assert hasattr(contracts.UserRepository, name)


def test_both_backends_implement_user_repository():
    for name in ("get_by_id", "get_by_email", "list_all", "list_active_reviewers", "create"):
        assert hasattr(PostgresUserRepository, name)
        assert hasattr(MongoUserRepository, name)


# B. get by id -----------------------------------------------------------


def test_get_by_id_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    created = repo.create(**_create_kwargs())
    found = repo.get_by_id(created.user_id)
    assert found is not None
    assert found.email == "a@x.com"
    assert found.role == "REVIEWER"


def test_get_by_id_mongo():
    repo = _mongo_repo()
    created = repo.create(**_create_kwargs())
    found = repo.get_by_id(created.user_id)
    assert found is not None
    assert found.email == "a@x.com"
    assert found.role == "REVIEWER"


# C. get by email ---------------------------------------------------------


def test_get_by_email_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_create_kwargs())
    found = repo.get_by_email("a@x.com")
    assert found is not None
    assert found.user_id == "u-1"


def test_get_by_email_mongo():
    repo = _mongo_repo()
    repo.create(**_create_kwargs())
    found = repo.get_by_email("a@x.com")
    assert found is not None
    assert found.user_id == "u-1"


# D. Email lookup is exact (case-sensitive), not normalized here --------


def test_email_lookup_is_exact_not_case_insensitive_postgres(tmp_path):
    """Normalization (.strip().lower()) is auth.py's business-logic
    responsibility, applied BEFORE calling the repository -- the
    repository itself does a plain exact-string lookup on both backends,
    proven here so no accidental case-insensitive behavior is added at
    either persistence layer."""
    repo = _pg_repo(tmp_path)
    repo.create(**_create_kwargs(email="lower@x.com"))
    assert repo.get_by_email("LOWER@x.com") is None
    assert repo.get_by_email("lower@x.com") is not None


def test_email_lookup_is_exact_not_case_insensitive_mongo():
    repo = _mongo_repo()
    repo.create(**_create_kwargs(email="lower@x.com"))
    assert repo.get_by_email("LOWER@x.com") is None
    assert repo.get_by_email("lower@x.com") is not None


# E. list ordering --------------------------------------------------------


def test_list_all_ordered_by_created_at_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_create_kwargs(user_id="u-2", email="second@x.com", created_at=dt.datetime(2026, 2, 1)))
    repo.create(**_create_kwargs(user_id="u-1", email="first@x.com", created_at=dt.datetime(2026, 1, 1)))
    rows = repo.list_all()
    assert [r.email for r in rows] == ["first@x.com", "second@x.com"]


def test_list_all_ordered_by_created_at_mongo():
    repo = _mongo_repo()
    repo.create(**_create_kwargs(user_id="u-2", email="second@x.com", created_at=dt.datetime(2026, 2, 1)))
    repo.create(**_create_kwargs(user_id="u-1", email="first@x.com", created_at=dt.datetime(2026, 1, 1)))
    rows = repo.list_all()
    assert [r.email for r in rows] == ["first@x.com", "second@x.com"]


def test_list_active_reviewers_ordered_by_display_name_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_create_kwargs(user_id="u-1", email="z@x.com", display_name="Zed"))
    repo.create(**_create_kwargs(user_id="u-2", email="a@x.com", display_name="Amal"))
    rows = repo.list_active_reviewers()
    assert [r.display_name for r in rows] == ["Amal", "Zed"]


def test_list_active_reviewers_ordered_by_display_name_mongo():
    repo = _mongo_repo()
    repo.create(**_create_kwargs(user_id="u-1", email="z@x.com", display_name="Zed"))
    repo.create(**_create_kwargs(user_id="u-2", email="a@x.com", display_name="Amal"))
    rows = repo.list_active_reviewers()
    assert [r.display_name for r in rows] == ["Amal", "Zed"]


# F. Active/inactive and role filtering -----------------------------------


def test_list_active_reviewers_excludes_inactive_and_admin_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_create_kwargs(user_id="u-1", email="active@x.com", role="REVIEWER", is_active=True))
    repo.create(**_create_kwargs(user_id="u-2", email="inactive@x.com", role="REVIEWER", is_active=False))
    repo.create(**_create_kwargs(user_id="u-3", email="admin@x.com", role="ADMIN", is_active=True))
    rows = repo.list_active_reviewers()
    assert [r.email for r in rows] == ["active@x.com"]


def test_list_active_reviewers_excludes_inactive_and_admin_mongo():
    repo = _mongo_repo()
    repo.create(**_create_kwargs(user_id="u-1", email="active@x.com", role="REVIEWER", is_active=True))
    repo.create(**_create_kwargs(user_id="u-2", email="inactive@x.com", role="REVIEWER", is_active=False))
    repo.create(**_create_kwargs(user_id="u-3", email="admin@x.com", role="ADMIN", is_active=True))
    rows = repo.list_active_reviewers()
    assert [r.email for r in rows] == ["active@x.com"]


def test_list_all_includes_inactive_users_postgres(tmp_path):
    """Audited behavior: /api/admin/users lists EVERY user, active or
    not -- list_all() must not silently filter inactive accounts."""
    repo = _pg_repo(tmp_path)
    repo.create(**_create_kwargs(user_id="u-1", email="active@x.com", is_active=True))
    repo.create(**_create_kwargs(user_id="u-2", email="inactive@x.com", is_active=False))
    rows = repo.list_all()
    assert {r.email for r in rows} == {"active@x.com", "inactive@x.com"}


def test_list_all_includes_inactive_users_mongo():
    repo = _mongo_repo()
    repo.create(**_create_kwargs(user_id="u-1", email="active@x.com", is_active=True))
    repo.create(**_create_kwargs(user_id="u-2", email="inactive@x.com", is_active=False))
    rows = repo.list_all()
    assert {r.email for r in rows} == {"active@x.com", "inactive@x.com"}


# G. Role preservation -----------------------------------------------------


@pytest.mark.parametrize("role", ["REVIEWER", "ADMIN"])
def test_role_round_trips_exactly_postgres(tmp_path, role):
    repo = _pg_repo(tmp_path)
    created = repo.create(**_create_kwargs(role=role))
    assert repo.get_by_id(created.user_id).role == role


@pytest.mark.parametrize("role", ["REVIEWER", "ADMIN"])
def test_role_round_trips_exactly_mongo(role):
    repo = _mongo_repo()
    created = repo.create(**_create_kwargs(role=role))
    assert repo.get_by_id(created.user_id).role == role


# H. Create user -----------------------------------------------------------


def test_create_returns_full_row_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    created = repo.create(**_create_kwargs(created_at=now, updated_at=now))
    assert created.user_id == "u-1"
    assert created.created_at == now
    assert created.updated_at == now
    assert created.password_hash == _SYNTHETIC_HASH


def test_create_returns_full_row_mongo():
    repo = _mongo_repo()
    now = dt.datetime(2026, 1, 1, 9, 0, 0)
    created = repo.create(**_create_kwargs(created_at=now, updated_at=now))
    assert created.user_id == "u-1"
    assert created.created_at == now
    assert created.updated_at == now
    assert created.password_hash == _SYNTHETIC_HASH


# J. Duplicate email --------------------------------------------------------


def test_duplicate_email_raises_value_error_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    repo.create(**_create_kwargs(user_id="u-1", email="dup@x.com"))
    with pytest.raises(ValueError):
        repo.create(**_create_kwargs(user_id="u-2", email="dup@x.com"))


def test_duplicate_email_raises_value_error_mongo():
    repo = _mongo_repo()
    repo.create(**_create_kwargs(user_id="u-1", email="dup@x.com"))
    with pytest.raises(ValueError):
        repo.create(**_create_kwargs(user_id="u-2", email="dup@x.com"))


# K. Missing user -----------------------------------------------------------


def test_missing_user_returns_none_not_raise_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    assert repo.get_by_id("does-not-exist") is None
    assert repo.get_by_email("nobody@x.com") is None


def test_missing_user_returns_none_not_raise_mongo():
    repo = _mongo_repo()
    assert repo.get_by_id("does-not-exist") is None
    assert repo.get_by_email("nobody@x.com") is None


# L. reviewer_summaries parity using UserRepository ------------------------


def test_reviewer_summaries_works_with_either_user_repo_backend(tmp_path):
    """Proves service.py::reviewer_summaries's User-side migration: the
    SAME ReviewItem data (Postgres, unchanged) combined with the SAME
    logical reviewer accounts produces an identical summary whether
    self.user_repo is Postgres or Mongo -- exercising the actual
    dependency-injection seam ReviewWorkflowService.__init__ now offers."""
    from rcs.review.repository import upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    engine = _engine(tmp_path)
    now = dt.datetime(2026, 1, 1)
    upsert_review_items(engine, [
        dict(comment_id="c1", source_file="f", source_row_index=0, review_status="APPROVED",
             reviewer_user_id="rev-1", reviewer_action="SEND_AS_IS", ingested_at=now),
        dict(comment_id="c2", source_file="f", source_row_index=1, review_status="IN_REVIEW",
             reviewer_user_id="rev-1", reviewer_action=None, ingested_at=now),
    ])
    pg_user_repo = PostgresUserRepository(engine)
    pg_user_repo.create(**_create_kwargs(user_id="rev-1", email="rev1@x.com", display_name="Reviewer One"))

    pg_summary = ReviewWorkflowService(engine, user_repo=pg_user_repo).reviewer_summaries()

    mongo_user_repo = _mongo_repo()
    mongo_user_repo.create(**_create_kwargs(user_id="rev-1", email="rev1@x.com", display_name="Reviewer One"))
    mongo_summary = ReviewWorkflowService(engine, user_repo=mongo_user_repo).reviewer_summaries()

    assert pg_summary == mongo_summary
    assert len(pg_summary) == 1
    row = pg_summary[0]
    assert row.user_id == "rev-1"
    assert row.assigned_total == 2
    assert row.pending == 1
    assert row.completed == 1


# M. Assignment-target validation parity (mirrors http_api.py's checks) ----


@pytest.mark.parametrize(
    "role,is_active,expected_valid",
    [("REVIEWER", True, True), ("REVIEWER", False, False), ("ADMIN", True, False)],
)
def test_assignment_target_validation_scenarios_postgres(tmp_path, role, is_active, expected_valid):
    repo = _pg_repo(tmp_path)
    created = repo.create(**_create_kwargs(role=role, is_active=is_active))
    target = repo.get_by_id(created.user_id)
    is_valid = target is not None and target.role == "REVIEWER" and target.is_active
    assert is_valid == expected_valid


@pytest.mark.parametrize(
    "role,is_active,expected_valid",
    [("REVIEWER", True, True), ("REVIEWER", False, False), ("ADMIN", True, False)],
)
def test_assignment_target_validation_scenarios_mongo(role, is_active, expected_valid):
    repo = _mongo_repo()
    created = repo.create(**_create_kwargs(role=role, is_active=is_active))
    target = repo.get_by_id(created.user_id)
    is_valid = target is not None and target.role == "REVIEWER" and target.is_active
    assert is_valid == expected_valid


def test_assignment_target_missing_user_id_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    assert repo.get_by_id("does-not-exist") is None


def test_assignment_target_missing_user_id_mongo():
    repo = _mongo_repo()
    assert repo.get_by_id("does-not-exist") is None


# N. Protected password-hash behavior ---------------------------------------


def test_password_hash_round_trips_exactly_postgres(tmp_path):
    repo = _pg_repo(tmp_path)
    created = repo.create(**_create_kwargs())
    assert repo.get_by_id(created.user_id).password_hash == _SYNTHETIC_HASH
    assert repo.get_by_email("a@x.com").password_hash == _SYNTHETIC_HASH


def test_password_hash_round_trips_exactly_mongo():
    repo = _mongo_repo()
    created = repo.create(**_create_kwargs())
    assert repo.get_by_id(created.user_id).password_hash == _SYNTHETIC_HASH
    assert repo.get_by_email("a@x.com").password_hash == _SYNTHETIC_HASH


# O. No secret/token logging -------------------------------------------------


def test_repository_source_never_logs_password_hash():
    """Static guard: neither repository implementation may print/log a
    password_hash value anywhere in its own source."""
    import inspect

    import rcs.review.repositories.postgres as pg_module
    import rcs.review.repositories.mongo as mongo_module

    for module in (pg_module, mongo_module):
        source = inspect.getsource(module)
        for line in source.splitlines():
            if "password_hash" in line and ("print(" in line or "logging." in line or "log." in line):
                pytest.fail(f"possible password_hash logging in {module.__name__}: {line!r}")


# P. Mongo create() never uses blind application-level write retry ---------


def test_mongo_create_never_uses_transient_retry(monkeypatch):
    import rcs.review.repositories.mongo as mongo_module

    def _boom(*_a, **_kw):
        raise AssertionError("with_transient_retry must not be used by UserRepository.create")

    monkeypatch.setattr(mongo_module, "with_transient_retry", _boom)
    repo = MongoUserRepository(_FakeDatabase())
    created = repo.create(**_create_kwargs())
    assert created.email == "a@x.com"
