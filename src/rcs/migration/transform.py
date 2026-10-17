"""
Pure, deterministic, side-effect-free transform layer (Phase M3):

    extracted PostgreSQL row (dict)
        -> validated intermediate record
        -> Mongo-ready document

Never touches PostgreSQL or MongoDB. Never "fixes" an anomalous source
value -- a NOT-NULL violation, an unrecognized enum value, or a
non-normalized email is recorded as an Anomaly and (for High severity)
the document is excluded from the output; the value itself is never
altered. Running this module twice over the same input always produces
the same output -- no clock reads, no randomness, no environment-
dependent branching.

Field mapping / type-conversion contract: see the Phase M2 report,
Sections A and B. This module is the implementation of that design.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from rcs.review import clock

from . import tables
from .logging_safe import short_ref


@dataclass
class Anomaly:
    table: str
    row_id_ref: object
    field_name: str
    kind: str
    severity: str  # "high" | "medium"
    detail: str


@dataclass
class TableTransformResult:
    documents: dict[object, dict] = field(default_factory=dict)  # id -> document (blocked ids absent)
    blocked_ids: set = field(default_factory=set)
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def high_severity_anomalies(self) -> list[Anomaly]:
        return [a for a in self.anomalies if a.severity == "high"]


def _safe_row_ref(table_name: str, id_field: str, raw_id):
    if (table_name, id_field) in tables.NEVER_LOG_FIELDS:
        return short_ref(raw_id)
    return raw_id


def _convert_datetime(value: dt.datetime, convention: str) -> dt.datetime:
    if convention == "utc":
        return value.replace(tzinfo=dt.timezone.utc)
    if convention == "riyadh":
        # clock.as_app() attaches Asia/Riyadh tzinfo to the naive wall-clock
        # value exactly as the app itself interprets it (clock.py). pymongo
        # converts a tz-aware datetime to the correct UTC instant when
        # encoding as BSON Date -- this is a genuine, correct conversion,
        # not a storage hack.
        return clock.as_app(value)
    if convention == "source":
        # No timezone convention was ever applied to this column by the
        # app (see tables.DATETIME_TZ_CONVENTION docstring) -- it is a raw
        # wall-clock value from the source Excel file. BSON has no naive
        # datetime type, so UTC is attached here purely as a storage
        # mechanism -- this is NOT a claim that the source data was
        # actually in UTC. Reconciliation must compare wall-clock
        # components (year/month/day/hour/minute/second), never treat this
        # as an absolute instant equivalent to the original naive value.
        return value.replace(tzinfo=dt.timezone.utc)
    raise AssertionError(f"unhandled datetime convention: {convention!r}")


def transform_table(table_name: str, rows: list[dict]) -> TableTransformResult:
    result = TableTransformResult()
    id_field = tables.id_field(table_name)
    nn_cols = tables.not_null_columns(table_name)
    dt_cols = tables.datetime_columns(table_name)
    high_enums = tables.HIGH_SEVERITY_ENUMS()
    info_enums = tables.INFORMATIONAL_ENUMS()

    for row in rows:
        raw_id = row[id_field]
        row_ref = _safe_row_ref(table_name, id_field, raw_id)
        doc = {"_id": raw_id}
        blocked = False

        for col in tables.columns_for(table_name):
            name = col.name
            if name == id_field:
                continue
            value = row[name]

            if name in nn_cols and value is None:
                result.anomalies.append(
                    Anomaly(table_name, row_ref, name, "not_null_violation", "high",
                            f"{name} is NULL but the schema declares it NOT NULL")
                )
                blocked = True
                continue

            if name in dt_cols and value is not None:
                convention = tables.DATETIME_TZ_CONVENTION.get((table_name, name))
                if convention is None:
                    result.anomalies.append(
                        Anomaly(table_name, row_ref, name, "missing_tz_convention", "high",
                                f"no timezone convention configured for datetime column {name}")
                    )
                    blocked = True
                    continue
                value = _convert_datetime(value, convention)

            key = (table_name, name)
            if row[name] is not None and key in high_enums and row[name] not in high_enums[key]:
                result.anomalies.append(
                    Anomaly(table_name, row_ref, name, "invalid_enum_value", "high",
                            f"{name} value not in the known set for this workflow-critical field")
                )
                blocked = True
            elif row[name] is not None and key in info_enums and row[name] not in info_enums[key]:
                result.anomalies.append(
                    Anomaly(table_name, row_ref, name, "invalid_enum_value", "medium",
                            f"{name} value not in the currently-configured taxonomy (informational; migrate as-is)")
                )

            if table_name == "users" and name == "email" and value is not None:
                normalized = value.strip().lower()
                if value != normalized:
                    result.anomalies.append(
                        Anomaly(table_name, row_ref, name, "email_not_normalized", "medium",
                                "stored email differs from the app's normalize-on-write form "
                                "(strip+lower, see auth.py); copied unchanged, not corrected")
                    )

            doc[name] = value

        if blocked:
            result.blocked_ids.add(raw_id)
        else:
            result.documents[raw_id] = doc

    return result


def transform_all(rows_by_table: dict[str, list[dict]]) -> dict[str, TableTransformResult]:
    return {table_name: transform_table(table_name, rows) for table_name, rows in rows_by_table.items()}
