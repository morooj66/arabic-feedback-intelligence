"""
Mongo runtime boundary milestone -- structural enforcement.

Business/runtime modules must reach persistence only through repository
contracts (rcs.review.persistence / rcs.review.repositories.contracts),
never directly through SQLAlchemy. This protects that one boundary --
it does not grep the whole project: repository implementations
(repositories/postgres.py, repositories/mongo.py), migrations,
bootstrap scripts, and tests all use SQLAlchemy/pymongo directly and
are correctly outside its scope.

The check works at the AST-import level rather than substring
grep: a protected module that imports nothing from `sqlalchemy` and
nothing from the concrete `repositories.postgres`/`repositories.mongo`
modules cannot construct a `Session`, call `select()`/`update()`/
`delete()`, or call `engine.begin()`/`engine.connect()` -- there would
be no name bound to do it with. That makes the import check both
necessary and sufficient for this boundary, and robust to aliasing
(`import sqlalchemy as sa`, `from sqlalchemy.orm import Session as S`)
in a way a plain text grep for "Session(" is not.
"""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest

PROTECTED_MODULES = (
    "rcs.review.auth",
    "rcs.review.sessions",
    "rcs.review.job_runner",
    "rcs.review.service",
    "rcs.review.excel_review_flow",
    "rcs.review.http_api",
)

_CONCRETE_BACKEND_MODULES = (
    "rcs.review.repositories.postgres",
    "rcs.review.repositories.mongo",
)


def _import_roots(tree: ast.Module) -> set[str]:
    """Every distinct module a file imports from, as import statements
    actually name them -- 'sqlalchemy.orm' from `from sqlalchemy.orm
    import Session`, 'sqlalchemy' from `import sqlalchemy`, etc."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
    return roots


@pytest.mark.parametrize("module_name", PROTECTED_MODULES)
def test_protected_module_never_imports_sqlalchemy(module_name):
    """No `import sqlalchemy`, no `from sqlalchemy... import ...`,
    under any alias -- the only way this module could otherwise open a
    Session or build a select()/update()/delete() statement."""
    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module), filename=module_name)
    roots = _import_roots(tree)
    sqlalchemy_imports = {r for r in roots if r == "sqlalchemy" or r.startswith("sqlalchemy.")}
    assert not sqlalchemy_imports, f"{module_name} imports SQLAlchemy directly: {sqlalchemy_imports}"


@pytest.mark.parametrize("module_name", PROTECTED_MODULES)
def test_protected_module_never_imports_concrete_backend_repository(module_name):
    """The backend choice belongs to rcs.review.persistence alone --
    a protected module may import `rcs.review.persistence` and
    `rcs.review.repositories.contracts`, but never the concrete
    PostgreSQL or Mongo repository implementation modules themselves."""
    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module), filename=module_name)
    roots = _import_roots(tree)
    leaks = {r for r in roots if r in _CONCRETE_BACKEND_MODULES}
    assert not leaks, f"{module_name} imports a concrete backend repository module directly: {leaks}"


def test_composition_root_is_the_one_place_allowed_to_import_both_backends():
    """Sanity check on the test itself: rcs.review.persistence (the
    composition root, deliberately NOT in PROTECTED_MODULES) is
    expected to import both concrete backend modules -- if it didn't,
    that would mean backend selection had been scattered somewhere else
    instead, which is the exact failure mode this file exists to catch
    in the six protected modules."""
    module = importlib.import_module("rcs.review.persistence")
    tree = ast.parse(inspect.getsource(module), filename="rcs.review.persistence")
    roots = _import_roots(tree)
    assert _CONCRETE_BACKEND_MODULES[0] in roots
    assert _CONCRETE_BACKEND_MODULES[1] in roots
