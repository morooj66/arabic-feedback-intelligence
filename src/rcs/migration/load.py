"""
MongoDB staging connectivity + idempotent loader (Phase M3).

Always writes to config.STAGING_DB_NAME -- never to whatever default
database name MONGODB_URI's own path may declare -- so a migration run
can never land in a production-sounding database. The URI value itself
is read once (via config.load_migration_env(), env-only) and handed
straight to pymongo.MongoClient(); this module never prints, logs, or
includes it in any returned report/exception message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import certifi
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import BulkWriteError, PyMongoError

from . import tables
from .config import STAGING_DB_NAME
from .logging_safe import log

_URI_SHAPED = re.compile(r"\w+://\S+")


def _sanitize_error_text(text: str, max_len: int = 300) -> str:
    """Defensive scrub for an exception message before it is ever
    returned/logged: server-generated auth/ops error text (e.g. Atlas's
    'Authentication failed.', code 18) does not itself contain
    credentials, but this strips anything URI-shaped just in case a
    future error message ever echoes part of the input back."""
    scrubbed = _URI_SHAPED.sub("<redacted-uri-shaped-text>", text)
    return scrubbed[:max_len]

# This venv's Python does not inherit the macOS system trust store (a
# known python.org-installer gap, root-caused during the Phase M1
# preflight: raw TCP + a system-openssl TLS handshake both succeeded
# against the Atlas cluster, but Python's own ssl.create_default_context()
# failed with CERTIFICATE_VERIFY_FAILED). certifi's bundle is the
# standard, non-destructive fix -- it does not touch the OS trust store
# or any credential, it only tells this process which CA bundle to trust
# for its own outgoing TLS connections.
_TLS_CA_FILE = certifi.where()


def _client(mongodb_uri: str, **kwargs) -> MongoClient:
    return MongoClient(mongodb_uri, serverSelectionTimeoutMS=8000, tlsCAFile=_TLS_CA_FILE, **kwargs)


@dataclass
class ConnectivityResult:
    ok: bool
    server_version: str | None = None
    error_type: str | None = None
    error_detail: str | None = None
    latency_ms: float | None = None


def check_connectivity(mongodb_uri: str) -> ConnectivityResult:
    """Sanitized connectivity probe. Never raises with the URI in the
    exception text -- pymongo's own ConfigurationError/ServerSelection
    messages can occasionally echo back parts of the input URI, so the
    URI is deliberately not interpolated into anything this function
    returns; error_detail is passed through _sanitize_error_text() as a
    defensive scrub."""
    import time

    client = None
    try:
        client = _client(mongodb_uri)
        t0 = time.perf_counter()
        client.admin.command("ping")
        latency_ms = (time.perf_counter() - t0) * 1000
        info = client.server_info()
        return ConnectivityResult(ok=True, server_version=info.get("version"), latency_ms=round(latency_ms, 1))
    except PyMongoError as exc:
        return ConnectivityResult(ok=False, error_type=type(exc).__name__, error_detail=_sanitize_error_text(str(exc)))
    finally:
        if client is not None:
            client.close()


def get_staging_db(mongodb_uri: str):
    client = _client(mongodb_uri)
    return client, client[STAGING_DB_NAME]


@dataclass
class TargetConflictCheck:
    safe_to_write: bool
    existing_collections: list[str]
    conflicting_target_names: list[str]


def check_no_pre_existing_targets(db) -> TargetConflictCheck:
    """Enumerate what already exists in the staging database BEFORE any
    write. If any of the five target collection names already exist,
    this reports safe_to_write=False -- the caller must STOP, never
    reuse, clear, or drop that collection. This check only ever reads
    (list_collection_names); it performs no write and no delete."""
    existing = sorted(db.list_collection_names())
    conflicts = sorted(set(existing) & set(tables.TABLE_ORDER))
    return TargetConflictCheck(
        safe_to_write=len(conflicts) == 0,
        existing_collections=existing,
        conflicting_target_names=conflicts,
    )


@dataclass
class LoadResult:
    collection: str
    attempted: int
    upserted: int = 0
    matched: int = 0
    error_count: int = 0
    error_types: list[str] = field(default_factory=list)


def load_documents(db, collection_name: str, documents: list[dict]) -> LoadResult:
    """Idempotent load: replaceOne(..., upsert=True) per document, keyed
    by the already-preserved business _id. Re-running with the same
    documents is a safe no-op (Phase M2 Section E)."""
    result = LoadResult(collection=collection_name, attempted=len(documents))
    if not documents:
        return result

    ops = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in documents]
    try:
        write_result = db[collection_name].bulk_write(ops, ordered=False)
        result.upserted = len(write_result.upserted_ids or {})
        result.matched = write_result.matched_count
    except BulkWriteError as exc:
        details = exc.details or {}
        write_errors = details.get("writeErrors", [])
        result.error_count = len(write_errors)
        result.error_types = sorted({e.get("code") for e in write_errors})
        result.upserted = details.get("nUpserted", 0)
        result.matched = details.get("nMatched", 0)
        log("load.bulk_write_partial_failure", collection=collection_name, error_count=result.error_count)
    return result
