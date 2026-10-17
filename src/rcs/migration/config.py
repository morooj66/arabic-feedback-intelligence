"""
Migration environment access -- reads DATABASE_URL / MONGODB_URI from the
process environment only (via the app's own load_dotenv_if_present(),
which never prints anything). Neither value is ever printed, logged, or
included in any report by this module or any caller of it.

The MongoDB staging database name is a constant chosen by this tooling
(STAGING_DB_NAME), never derived from whatever default database name may
be present in MONGODB_URI's own path -- so migration writes can never
land in a production-sounding database purely because the URI happened
to declare one as its default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from config import settings
from rcs.review.demo_seed import load_dotenv_if_present

STAGING_DB_NAME = "rcs_migration_staging"


@dataclass(frozen=True)
class MigrationEnv:
    postgres_url: str
    mongodb_uri: str
    staging_db_name: str = STAGING_DB_NAME


class MissingEnvError(RuntimeError):
    pass


def load_migration_env() -> MigrationEnv:
    load_dotenv_if_present()
    postgres_url = settings.resolve_database_url(settings.EXCEL_OPERATIONAL_DB)
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        raise MissingEnvError("MONGODB_URI is not set in the environment.")
    return MigrationEnv(postgres_url=postgres_url, mongodb_uri=mongodb_uri)
