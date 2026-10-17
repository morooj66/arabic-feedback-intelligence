"""
Phase M4B — MongoDB connection helper for the RUNTIME review
repositories (rcs.review.repositories.mongo).

Deliberately separate from rcs.migration.load: that package's own module
docstring states it is "isolated... nothing here is imported by the
served application" — this module is the opposite by design, since M4B
builds repositories meant to eventually run inside the served
application (not yet wired in — no backend selector exists this phase).

Reuses the same certifi CA-bundle fix established during the Phase M1
connectivity preflight and used throughout Phase M3: this venv's Python
does not inherit the macOS system trust store, so pymongo's TLS
handshake against Atlas fails without an explicit CA bundle.
"""

from __future__ import annotations

import certifi
from pymongo import MongoClient

_TLS_CA_FILE = certifi.where()


def get_mongo_database(uri: str, database_name: str, *, server_selection_timeout_ms: int = 8000):
    """Returns (client, database). Caller owns the client's lifecycle
    (close it when done). Never logs or returns the URI itself."""
    client = MongoClient(uri, serverSelectionTimeoutMS=server_selection_timeout_ms, tlsCAFile=_TLS_CA_FILE)
    return client, client[database_name]
