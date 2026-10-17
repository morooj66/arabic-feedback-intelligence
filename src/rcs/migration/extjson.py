"""
MongoDB Extended JSON v2 (relaxed mode) serialization for manual Atlas
import packages (Phase M-MANUAL).

Relaxed mode represents int32/double values as plain JSON numbers and
booleans/strings/null as plain JSON literals -- only BSON Date needs an
explicit wrapper ({"$date": "<ISO-8601 UTC, millisecond precision>"}),
since plain JSON has no Date type of its own. This module intentionally
does not need $numberLong/$numberInt/$oid: every _id in this schema is
already a plain string (never ObjectId -- see the M1 finding that no
column anywhere uses a DB-generated surrogate key), and every integer
column in review_items/review_sessions/processing_jobs/users is small
enough to round-trip correctly as a bare JSON number.
"""

from __future__ import annotations

import datetime as dt
import json


def _to_extended_json_value(value):
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            raise ValueError("refusing to serialize a naive datetime -- must be tz-aware by this point")
        utc_value = value.astimezone(dt.timezone.utc)
        iso = utc_value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return {"$date": iso}
    return value


def document_to_extended_json(doc: dict) -> dict:
    return {key: _to_extended_json_value(value) for key, value in doc.items()}


def write_extended_json_array(documents: list[dict], path) -> None:
    payload = [document_to_extended_json(doc) for doc in documents]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
