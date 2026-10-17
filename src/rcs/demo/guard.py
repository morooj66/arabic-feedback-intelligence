"""Triple lock that makes demo mode impossible to enable by accident.

Demo mode starts only if ALL of these hold; otherwise the process exits before serving anything (it never silently falls back to normal mode):
  1. RCS_DEMO_MODE == "synthetic-demo"                      (explicit, exact value)
  2. RCS_DEMO_ACK  == the acknowledgement sentence below     (a human typed it; no default exists anywhere)
  3. the database is a DEDICATED demo database: its name (PostgreSQL) or file stem (SQLite) starts with "rcs_demo"
  4. no production secret/provider is present in the environment (Gemini, OpenRouter, MongoDB, SMTP, Entra, Hugging Face token/model repo)
  5. PERSISTENCE_BACKEND is not "mongo"; no development-only database flags (--seed/--db/--production-db) are used
A production deployment does none of 1-2, so `demo_requested()` is False and the demo package is never even imported. A production environment that carries SOME
RCS_DEMO_* variable but not all of the above refuses to start (a half-configured demo is treated as a mistake, not as a normal start)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

MODE_VAR, MODE_VALUE = "RCS_DEMO_MODE", "synthetic-demo"
ACK_VAR, ACK_VALUE = "RCS_DEMO_ACK", "THIS_INSTANCE_HOLDS_SYNTHETIC_DATA_ONLY"
DB_NAME_PREFIX = "rcs_demo"
FORBIDDEN_ENV = ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "MONGODB_URI", "RCS_SMTP_HOST", "RCS_SMTP_PASSWORD", "ENTRA_CLIENT_SECRET", "ENTRA_CLIENT_ID", "ENTRA_TENANT_ID",
                 "HF_TOKEN", "HF_MODEL_REPO_ID")


def demo_requested(env=None) -> bool:
    env = os.environ if env is None else env
    return any(k.startswith("RCS_DEMO_") and str(v).strip() != "" for k, v in env.items())


def database_name(database_url: str | Path | None) -> str | None:
    """PostgreSQL: the database name (last path segment). SQLite (URL or plain path): the file name without extension. Anything else: None."""
    if database_url is None or str(database_url).strip() == "":
        return None
    raw = str(database_url).strip()
    if "://" not in raw:
        return Path(raw).stem
    parts = urlsplit(raw)
    scheme = parts.scheme.split("+")[0]
    if scheme == "sqlite":
        return Path(parts.path).stem or None
    if scheme in ("postgresql", "postgres"):
        name = parts.path.lstrip("/").split("/")[0]
        return name or None
    return None


def problems(env, database_url, *, dev_db_flags_used: bool = False) -> list[str]:
    out: list[str] = []
    if env.get(MODE_VAR, "") != MODE_VALUE:
        out.append(f"{MODE_VAR} must be exactly {MODE_VALUE!r}")
    if env.get(ACK_VAR, "") != ACK_VALUE:
        out.append(f"{ACK_VAR} must carry the acknowledgement sentence")
    name = database_name(database_url)
    if name is None:
        out.append("a DATABASE_URL naming a dedicated demo database is required")
    elif not re.match(rf"^{DB_NAME_PREFIX}", name):
        out.append(f"the database name must start with {DB_NAME_PREFIX!r} (a dedicated demo database); refusing to run demo mode against a database not named for it")
    present = [k for k in FORBIDDEN_ENV if str(env.get(k, "")).strip() != ""]
    if present:
        out.append("production secrets/providers must not be present in a demo environment: " + ", ".join(present))
    if str(env.get("PERSISTENCE_BACKEND", "")).strip().lower() == "mongo":
        out.append("PERSISTENCE_BACKEND=mongo is not supported in demo mode")
    if dev_db_flags_used:
        out.append("--seed/--live-seed/--db/--production-db cannot be combined with demo mode (use scripts/demo/init_demo_environment.py)")
    return out
