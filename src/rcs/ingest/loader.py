"""
Phase 2 — historical data loader.

Reads the single source-of-truth workbook (visitors-reports-*.xlsx),
renames Arabic headers to the internal English field names defined in
config/settings.py, and assigns each row a stable identity:

- source_row_index: 0-based position in the source file, stable across
  re-runs because the source file is a static historical export.
- comment_id: sha256(source_file + "|" + source_row_index) — see
  DESIGN_SPEC.md Section 2 for why this is scoped to this MVP only.

Deliberately NOT loaded here: the second uploaded file
(_تفاعل-الزوار-بعد-2026-03-31.xlsx), confirmed in data discovery to be a
100%-overlapping subset of this one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def _comment_id(source_file: str, source_row_index: int) -> str:
    payload = f"{source_file}|{source_row_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_raw(path: Path, source_columns: dict[str, str]) -> pd.DataFrame:
    """
    Load the raw workbook and rename columns to internal names.
    Does not mutate the source file. Returns one row per source row,
    in original order, with source_row_index and comment_id assigned.
    """
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_excel(path)

    missing = set(source_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Source file {path.name} is missing expected column(s): {sorted(missing)}. "
            "Column names may have changed upstream — update config/settings.py::SOURCE_COLUMNS."
        )

    df = df.rename(columns=source_columns)
    # Keep only the columns we know about, in a stable order.
    df = df[list(source_columns.values())]

    df = df.reset_index(drop=True)
    df["source_row_index"] = df.index
    df["source_file"] = path.name
    df["comment_id"] = df["source_row_index"].apply(
        lambda i: _comment_id(path.name, int(i))
    )

    return df
