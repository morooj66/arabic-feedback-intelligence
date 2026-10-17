"""
Read-only PostgreSQL snapshot extractor (Phase M3, corrected per the M2
amendment).

Contract (see PHASE_M2 amendment, Section A.1/A.2 -- reproduced here
because this module is the implementation of that contract):

  * One full extraction pass = ONE PostgreSQL transaction = one snapshot.
    The transaction is opened at REPEATABLE READ before the first row is
    read and held open, on the same connection, until the last table's
    last chunk has been read.
  * Keyset pagination (chunked reads) happens INSIDE that one
    transaction -- it is a memory/batching technique, not a resumability
    mechanism.
  * There is no cross-process checkpoint/resume. If the connection or
    transaction is lost, the caller must discard everything and start a
    brand-new extract_all() call from scratch. This module never writes
    a "last seen pk" checkpoint anywhere.
  * Given the current confirmed data volume (review_items on the order
    of a few thousand rows; the other four tables far smaller), the
    whole snapshot is accumulated in memory rather than streamed to
    disk -- correct and simple at this size. Do not reuse this module
    unmodified if the dataset grows enough for full-snapshot in-memory
    accumulation to become impractical.
  * Start/end MAX(updated_at) (or row-count, for tables with no
    updated_at column) is recorded as INFORMATIONAL metadata only -- a
    difference proves the live application kept writing during
    extraction, which is normal and expected under REPEATABLE READ. It
    is never treated as evidence the extractor ran outside its snapshot.

No secret values are ever read from these tables by this module and
printed -- see tables.NEVER_LOG_FIELDS; this module only logs table
names, chunk boundaries (as short_ref() for token-shaped ids), and row
counts.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.engine import Connection, Engine

from . import tables
from .logging_safe import log, short_ref

CHUNK_SIZE = 500

_TABLES_WITH_UPDATED_AT = {"users", "review_sessions", "processing_jobs", "review_items"}


@dataclass
class DriftSignal:
    table: str
    kind: str  # "max_updated_at" | "row_count"
    before: object
    after: object | None = None

    @property
    def changed(self) -> bool:
        return self.after is not None and self.before != self.after


@dataclass
class ExtractionResult:
    rows_by_table: dict[str, list[dict]] = field(default_factory=dict)
    drift_signals: list[DriftSignal] = field(default_factory=list)
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None

    def counts(self) -> dict[str, int]:
        return {t: len(rows) for t, rows in self.rows_by_table.items()}


def _require_postgres(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        raise RuntimeError(
            "Snapshot extraction requires PostgreSQL (REPEATABLE READ semantics); "
            f"got dialect={engine.dialect.name!r}. Refusing to fake snapshot "
            "isolation on a different backend."
        )


def _capture_drift_baseline(engine: Engine, table_names: tuple[str, ...]) -> dict[str, DriftSignal]:
    """Ordinary (non-snapshot) connection, read BEFORE the snapshot
    transaction opens. Informational only -- see module docstring. Only
    touches the given table_names -- never a table outside the caller's
    requested scope, not even for this read-only informational signal."""
    signals: dict[str, DriftSignal] = {}
    with engine.connect() as conn:
        for table_name in table_names:
            table = tables.orm_class(table_name).__table__
            if table_name in _TABLES_WITH_UPDATED_AT:
                value = conn.execute(select(func.max(table.c.updated_at))).scalar()
                signals[table_name] = DriftSignal(table_name, "max_updated_at", value)
            else:
                value = conn.execute(select(func.count()).select_from(table)).scalar()
                signals[table_name] = DriftSignal(table_name, "row_count", value)
    return signals


def _capture_drift_after(engine: Engine, baseline: dict[str, DriftSignal]) -> list[DriftSignal]:
    with engine.connect() as conn:
        for table_name, sig in baseline.items():
            table = tables.orm_class(table_name).__table__
            if sig.kind == "max_updated_at":
                sig.after = conn.execute(select(func.max(table.c.updated_at))).scalar()
            else:
                sig.after = conn.execute(select(func.count()).select_from(table)).scalar()
    return list(baseline.values())


def _read_table_keyset(connection: Connection, table_name: str, chunk_size: int) -> list[dict]:
    orm_cls = tables.orm_class(table_name)
    table = orm_cls.__table__
    id_col = table.c[tables.id_field(table_name)]

    rows: list[dict] = []
    last_id = None
    chunk_index = 0
    while True:
        stmt = select(table).order_by(id_col.asc()).limit(chunk_size)
        if last_id is not None:
            stmt = stmt.where(id_col > last_id)
        chunk = [dict(m) for m in connection.execute(stmt).mappings().all()]
        if not chunk:
            break
        rows.extend(chunk)
        last_id = chunk[-1][tables.id_field(table_name)]
        boundary_ref = short_ref(last_id) if (table_name, tables.id_field(table_name)) in tables.NEVER_LOG_FIELDS else last_id
        log("extract.chunk", table=table_name, chunk_index=chunk_index, chunk_rows=len(chunk), boundary_id_ref=boundary_ref)
        chunk_index += 1
        if len(chunk) < chunk_size:
            break
    return rows


def extract_tables(engine: Engine, table_names: tuple[str, ...], chunk_size: int = CHUNK_SIZE) -> ExtractionResult:
    """Run exactly one full extraction pass over ONLY the given tables --
    every other table is never opened, never read, not even read-only.
    Use this (rather than extract_all) whenever a caller has been asked
    to scope a run to a subset of tables; it keeps that scoping real
    rather than "extract everything, use only a subset in memory".
    Raises on any error -- callers must not attempt to resume a failed
    pass; they must call this function again from scratch."""
    _require_postgres(engine)
    for name in table_names:
        if name not in tables.TABLE_SPECS:
            raise KeyError(f"unknown table: {name!r}")

    result = ExtractionResult()
    baseline = _capture_drift_baseline(engine, table_names)
    result.started_at = dt.datetime.utcnow()

    connection = engine.connect()
    try:
        connection = connection.execution_options(isolation_level="REPEATABLE READ")
        with connection.begin():
            log("extract.snapshot_open", isolation_level="REPEATABLE READ", tables=list(table_names))
            for table_name in table_names:
                rows = _read_table_keyset(connection, table_name, chunk_size)
                result.rows_by_table[table_name] = rows
                log("extract.table_done", table=table_name, row_count=len(rows))
        log("extract.snapshot_closed_committed")
    finally:
        connection.close()

    result.finished_at = dt.datetime.utcnow()
    result.drift_signals = _capture_drift_after(engine, baseline)
    for sig in result.drift_signals:
        log(
            "extract.drift_signal_informational",
            table=sig.table,
            kind=sig.kind,
            changed=sig.changed,
        )
    return result


def extract_all(engine: Engine, chunk_size: int = CHUNK_SIZE) -> ExtractionResult:
    """Run exactly one full extraction pass over all five tables."""
    return extract_tables(engine, tables.TABLE_ORDER, chunk_size)
