"""
Phase M5B.4 — Official Knowledge corpus: fail-closed provenance
governance and sanitized corpus records.

--- Local-source audit result (read before touching this module) --------

Before writing any code, the repository was searched for an existing
local dictionary/lexicon/knowledge file (by filename: *lexicon*,
*dictionary*, *knowledge*; by JSON content: any file with a "lemma" or
"entry_id" key; by script: any *convert*/lexicon-builder script; by doc:
any *.md mentioning "lexicon"/"official knowledge"). Result: **no such
file or converter exists in this repository.** The only two things that
came up:
  - `scripts/reference_api_smoke.py` -- a smoke test against the REMOTE
    external dictionary platform's public API (not a local file, not used by this phase, no
    no external API base URL or key is required anywhere in this module).
  - `data/feedback_research/rg_v8_context_ab_eval_key.json` -- an A/B
    response-generation evaluation key derived from
    `data/db/excel_review_demo.sqlite` (its own `source_db` field says
    so). This is demo/research evaluation data about visitor comments,
    not a curated factual dictionary corpus, and it is never read by
    this module.

Conclusion: there is currently no trustworthy local official source.
This module therefore implements the retrieval architecture and
provider boundary ONLY, fails closed to `SUCCESS_EMPTY` when no source
is configured (the default), and never invents official facts or
treats any repository fixture/demo file as authoritative.

--- What "provenance-acceptable" means here, if a source is ever added -

`OFFICIAL_KNOWLEDGE_SOURCE_PATH` (config.settings, default unset) is the
FIRST governance layer: no path configured -> nothing is ever loaded.
Each record read from a configured source must ALSO carry an explicit
`provenance_tag == "official_approved"` (a second, per-record layer) --
this is a positive self-declared tag, not a filename heuristic. A
record's `authoritative` field must be explicitly `True` in the source
data; this module never defaults or infers it. Required contract fields
(source_id, source_title, source_type, content, version, effective_from)
must be present and non-blank/non-None in the source data; missing ones
exclude that record rather than inventing a placeholder value.

Expected source-file schema (documented, not shipped as data): a JSON
array of objects with string keys `source_id`, `source_title`,
`source_type`, `content`, `version`, `effective_from` (ISO 8601),
`authoritative` (bool), `provenance_tag` (must equal
"official_approved"), and optionally `effective_to` (ISO 8601),
`entry_id`, `lemma`, `topic`, `intent` (the latter two matching
`config.settings.PRIMARY_INTENTS`/`TOPIC_SUBTYPES` if set, for
structured filtering -- see official_knowledge.py).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_ACCEPTED_PROVENANCE_TAG = "official_approved"


class OfficialSourceLoadError(Exception):
    """Raised when a CONFIGURED source cannot be read/parsed. Distinct
    from "no source configured" (which is not an error -- see
    official_knowledge.py, where that case is SUCCESS_EMPTY and this
    exception's case is UNAVAILABLE)."""


@dataclass(frozen=True)
class RawOfficialSourceRecord:
    """Plain, unvalidated record as read from a source. Every field is
    Optional here on purpose -- validation/exclusion happens explicitly
    in `build_governed_official_corpus`, never by silently defaulting a
    missing value."""

    source_id: str | None = None
    source_title: str | None = None
    source_type: str | None = None
    content: str | None = None
    version: str | None = None
    effective_from: dt.datetime | None = None
    effective_to: dt.datetime | None = None
    authoritative: bool | None = None
    provenance_tag: str | None = None
    entry_id: str | None = None
    lemma: str | None = None
    topic: str | None = None
    intent: str | None = None


class OfficialSourceProvider(Protocol):
    """Narrow, storage-agnostic provider interface. `JsonFileOfficialSourceProvider`
    below is the only production implementation; tests substitute a
    simple in-memory fake."""

    def fetch_records(self) -> Iterable[RawOfficialSourceRecord]: ...


def _parse_datetime(value: object) -> dt.datetime | None:
    if value is None or isinstance(value, dt.datetime):
        return value  # type: ignore[return-value]
    try:
        return dt.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None  # malformed date -> treated as absent; governance excludes the record


def _parse_raw_record(item: dict) -> RawOfficialSourceRecord:
    return RawOfficialSourceRecord(
        source_id=item.get("source_id"),
        source_title=item.get("source_title"),
        source_type=item.get("source_type"),
        content=item.get("content"),
        version=item.get("version"),
        effective_from=_parse_datetime(item.get("effective_from")),
        effective_to=_parse_datetime(item.get("effective_to")),
        authoritative=item.get("authoritative"),
        provenance_tag=item.get("provenance_tag"),
        entry_id=item.get("entry_id"),
        lemma=item.get("lemma"),
        topic=item.get("topic"),
        intent=item.get("intent"),
    )


class JsonFileOfficialSourceProvider:
    """Reads a local JSON file (a list of objects matching this module's
    documented schema) into `RawOfficialSourceRecord`s. No format
    guessing, no schema inference -- a missing field simply parses as
    None, and `build_governed_official_corpus` excludes the record if
    that field was required."""

    def __init__(self, path: str):
        self._path = path

    def fetch_records(self) -> tuple[RawOfficialSourceRecord, ...]:
        try:
            raw_list = json.loads(Path(self._path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OfficialSourceLoadError("official knowledge source could not be read or parsed") from exc

        if not isinstance(raw_list, list):
            raise OfficialSourceLoadError("official knowledge source must be a JSON array of objects")

        try:
            return tuple(_parse_raw_record(item) for item in raw_list if isinstance(item, dict))
        except Exception as exc:  # noqa: BLE001 -- never leak a raw parse exception to the caller
            raise OfficialSourceLoadError("official knowledge source contained an unreadable record") from exc


@dataclass(frozen=True)
class OfficialCorpusRecord:
    """Sanitized, storage-agnostic corpus record -- the ONLY shape ever
    handed to the semantic index (see official_knowledge.py).
    `build_governed_official_corpus` is the sole producer of this type.
    `authoritative` is always `True` here (never `False`) by
    construction -- a record that wasn't explicitly authoritative in
    the source never reaches this type at all."""

    source_id: str
    source_title: str
    source_type: str
    content: str
    version: str
    effective_from: dt.datetime
    effective_to: dt.datetime | None = None
    entry_id: str | None = None
    lemma: str | None = None
    topic: str | None = None
    intent: str | None = None


def _passes_governance(raw: RawOfficialSourceRecord) -> bool:
    if raw.provenance_tag != _ACCEPTED_PROVENANCE_TAG:
        return False
    if raw.authoritative is not True:
        return False
    for text_field in (raw.source_id, raw.source_title, raw.source_type, raw.content, raw.version):
        if not text_field or not text_field.strip():
            return False
    if raw.effective_from is None:
        return False
    return True


def build_governed_official_corpus(
    raw_records: Iterable[RawOfficialSourceRecord],
) -> tuple[OfficialCorpusRecord, ...]:
    """Pure function: raw records in, sanitized+eligible corpus records
    out, sorted by source_id for determinism. A record is excluded
    (never repaired/defaulted) if it fails ANY governance check -- see
    module docstring for the evidence/reasoning behind each one."""
    records: list[OfficialCorpusRecord] = []
    for raw in raw_records:
        if not _passes_governance(raw):
            continue
        records.append(
            OfficialCorpusRecord(
                source_id=raw.source_id.strip(),
                source_title=raw.source_title.strip(),
                source_type=raw.source_type.strip(),
                content=raw.content.strip(),
                version=raw.version.strip(),
                effective_from=raw.effective_from,
                effective_to=raw.effective_to,
                entry_id=raw.entry_id,
                lemma=raw.lemma,
                topic=raw.topic,
                intent=raw.intent,
            )
        )
    return tuple(sorted(records, key=lambda r: r.source_id))
