import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from rcs.review.demo_seed import load_dotenv_if_present  # noqa: E402
from rcs.review.models import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MUST run before resolve_database_url() below: settings.DATABASE_URL is
# captured once at config.settings import time, before .env is ever read.
# scripts/serve_reviewer_api.py calls this explicitly for the same reason.
# Omitting it here caused a real incident: a bare `alembic upgrade head`
# silently fell back to the local SQLite default instead of the intended
# DATABASE_URL, because .env had never been loaded into os.environ for
# that process. Always load .env first so the `alembic` CLI targets
# exactly the same database the served application would.
load_dotenv_if_present()

# Same DATABASE_URL boundary the served application uses (config.settings.
# resolve_database_url) — Alembic always targets whatever database the app
# itself would connect to, local SQLite by default. Overrides whatever
# static sqlalchemy.url happens to be in alembic.ini.
config.set_main_option(
    "sqlalchemy.url", settings.resolve_database_url(settings.EXCEL_OPERATIONAL_DB)
)

# ORM metadata drives autogenerate — every model in rcs.review.models.
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
