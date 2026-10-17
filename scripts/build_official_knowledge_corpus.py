#!/usr/bin/env python3
"""
Official Knowledge corpus preparation tool — ONE-TIME, offline, no
runtime framework. Converts an approved lexical export (Excel) into the
EXISTING, documented `RawOfficialSourceRecord` JSON schema (see
rcs.response_intelligence.retrieval.official_corpus's module docstring)
so it can be loaded, unmodified, by `JsonFileOfficialSourceProvider` and
retrieved by the existing, unmodified `OfficialKnowledgeRetrieverImpl`.

This tool does not talk to Gemini, the orchestrator, a database, or the
network. It reads one local file and writes one local JSON file. It
never decides on its own that a source is authoritative -- see
"THE ONE GOVERNANCE RULE" below.

--- Why this exists (audit context) --------------------------------------

The "Official Knowledge Source Audit" milestone found: (1) no
trustworthy local dictionary/lexicon file exists in this repository
today: the one dictionary-shaped file found
(data/uploads/.../المعجم_الطلابي_ابتدائي_ـ_اسم_ذات.xlsx) has unverified
provenance, is a differently-named product ("المعجم الطلابي") from the
dictionary visitors actually comment on ("المعجم"), and MUST NOT be
treated as authoritative; (2) an external dictionary platform's public API requires an
API key not available in this environment; (3) no lexicon/dictionary
table exists in any project database. The recommended, approved
strategy is: when an organization-approved المعجم export exists,
convert it with THIS tool into the existing governed-JSON-corpus
contract -- no new provider, retriever, or Agent code needed.

--- THE ONE GOVERNANCE RULE (read before changing anything below) --------

`authoritative=True` and `provenance_tag="official_approved"` are NEVER
inferred from the input file itself -- not from its filename, its
folder, its column names, or the mere presence of "معجم" in any string.
They are set ONLY when the human operator running this tool passes
`--confirm-approved` on the command line, explicitly attesting that
this specific export has been organizationally approved as an
authoritative source. Without that flag, this tool will VALIDATE and
report on the input, but refuses to write ANY output file claiming
records are official_approved -- it fails closed, exactly like the
runtime governance in official_corpus.py.

--- Output schema (unchanged from official_corpus.py; nothing invented) --

A JSON array of objects with keys: source_id, source_title, source_type,
content, version, effective_from (ISO 8601), authoritative (bool),
provenance_tag, and optionally effective_to (ISO 8601), entry_id,
lemma, topic, intent -- the exact fields of RawOfficialSourceRecord.
`source_title`/`source_type`/`version`/`effective_from`/`effective_to`
are operator-supplied constants describing the SOURCE (e.g. "which
dictionary, which export date"), not per-row values, since a lexical
export does not carry its own per-entry versioning. `content` is built
from the row's definition text, optionally prefixed with its
part-of-speech and suffixed with its example -- both are existing free
text concatenated into the single `content` field the contract already
has; no new field is added for either.

--- Usage -------------------------------------------------------------

  # Dry run: validate only, write nothing, no approval flag needed.
  python scripts/build_official_knowledge_corpus.py \\
      --input approved_export.xlsx --validate-only \\
      --headword-column "الكلمة" --definition-column "التعريف"

  # Real write: requires explicit operator approval confirmation.
  python scripts/build_official_knowledge_corpus.py \\
      --input approved_export.xlsx \\
      --output data/official_knowledge/dictionary_snapshot_2026_01.json \\
      --confirm-approved \\
      --source-title "المعجم" --source-type dictionary_entry \\
      --version "2026-01" --effective-from 2026-01-01 \\
      --headword-column "الكلمة" --definition-column "التعريف"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

_ACCEPTED_PROVENANCE_TAG = "official_approved"  # must match official_corpus.py exactly
_REQUIRED_RECORD_FIELDS = ("source_id", "source_title", "source_type", "content", "version")


@dataclass(frozen=True)
class RowRejection:
    row_number: int  # 1-based, matching the source spreadsheet's own row numbering
    reason: str


@dataclass
class ValidationSummary:
    """Counts only -- never holds row content, so it is always safe to
    print or log (see module docstring: "no secrets", "do not print
    sensitive source content unnecessarily")."""

    input_rows: int = 0
    valid_rows: int = 0
    rejected_missing_content: int = 0
    rejected_duplicate_source_id: int = 0
    rejected_invalid_effective_date: int = 0
    missing_lexical_key: int = 0  # warning only, not a rejection -- lemma/entry_id are optional
    approved: bool = False
    rejections: list[RowRejection] = field(default_factory=list)

    @property
    def rejected_rows(self) -> int:
        return self.input_rows - self.valid_rows

    def to_dict(self) -> dict:
        return {
            "input_rows": self.input_rows,
            "valid_rows": self.valid_rows,
            "rejected_rows": self.rejected_rows,
            "rejected_missing_content": self.rejected_missing_content,
            "rejected_duplicate_source_id": self.rejected_duplicate_source_id,
            "rejected_invalid_effective_date": self.rejected_invalid_effective_date,
            "missing_lexical_key_warning": self.missing_lexical_key,
            "approved": self.approved,
            # Row NUMBERS and REASON CODES only -- never cell content.
            "rejections": [{"row": r.row_number, "reason": r.reason} for r in self.rejections],
        }


def _clean(value: object) -> str | None:
    """pandas represents a blank Excel cell as NaN/None inconsistently
    across dtypes -- normalize both to None, and strip everything else."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_iso_date(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _build_content(*, pos: str | None, definition: str | None, example: str | None) -> str | None:
    """Concatenates existing free text into the ONE `content` field the
    contract already has -- no new field. `definition` is required for
    any content at all; `pos`/`example` are optional prefixes/suffixes."""
    if not definition:
        return None
    parts = []
    if pos:
        parts.append(f"({pos})")
    parts.append(definition)
    text = " ".join(parts)
    if example:
        text = f"{text}\n{example}"
    return text


def _read_dataframe(input_path: Path, sheet: str | int | None) -> pd.DataFrame:
    return pd.read_excel(input_path, sheet_name=sheet if sheet is not None else 0)


def _referenced_columns(args: argparse.Namespace) -> list[str]:
    """Every column name the operator asked this run to read -- the
    required `--definition-column` plus every optional column that was
    actually set. Used to fail closed with a clear message BEFORE
    processing any row, rather than silently rejecting every row with
    "missing_required_content" when a column name was simply mistyped."""
    names = [args.definition_column]
    for optional in (
        args.headword_column, args.example_column, args.pos_column, args.entry_id_column,
        args.source_id_column, args.topic_column, args.intent_column,
        args.effective_from_column, args.effective_to_column,
    ):
        if optional:
            names.append(optional)
    return names


def _missing_columns(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    referenced = _referenced_columns(args)
    return [name for name in referenced if name not in df.columns]


def _normalize(
    rows: list[dict],
    *,
    args: argparse.Namespace,
    approved: bool,
) -> tuple[list[dict], ValidationSummary]:
    summary = ValidationSummary(input_rows=len(rows), approved=approved)
    seen_source_ids: set[str] = set()
    output_records: list[dict] = []

    global_effective_from = _parse_iso_date(args.effective_from)
    global_effective_to = _parse_iso_date(args.effective_to) if args.effective_to else None

    for i, row in enumerate(rows):
        row_number = i + 2  # 1-based, +1 for the header row Excel shows

        headword = _clean(row.get(args.headword_column)) if args.headword_column else None
        definition = _clean(row.get(args.definition_column)) if args.definition_column else None
        example = _clean(row.get(args.example_column)) if args.example_column else None
        pos = _clean(row.get(args.pos_column)) if args.pos_column else None
        entry_id = _clean(row.get(args.entry_id_column)) if args.entry_id_column else None
        topic = _clean(row.get(args.topic_column)) if args.topic_column else None
        intent = _clean(row.get(args.intent_column)) if args.intent_column else None

        content = _build_content(pos=pos, definition=definition, example=example)
        if not content:
            summary.rejected_missing_content += 1
            summary.rejections.append(RowRejection(row_number, "missing_required_content"))
            continue

        # Per-row effective-date override, if the export carries one;
        # otherwise every record uses the operator-supplied constant.
        if args.effective_from_column:
            row_effective_from = _parse_iso_date(_clean(row.get(args.effective_from_column)))
            if row_effective_from is None:
                summary.rejected_invalid_effective_date += 1
                summary.rejections.append(RowRejection(row_number, "invalid_effective_from"))
                continue
        else:
            row_effective_from = global_effective_from

        if args.effective_to_column:
            row_effective_to_raw = _clean(row.get(args.effective_to_column))
            if row_effective_to_raw is not None:
                row_effective_to = _parse_iso_date(row_effective_to_raw)
                if row_effective_to is None:
                    summary.rejected_invalid_effective_date += 1
                    summary.rejections.append(RowRejection(row_number, "invalid_effective_to"))
                    continue
            else:
                row_effective_to = None
        else:
            row_effective_to = global_effective_to

        source_id = (
            _clean(row.get(args.source_id_column)) if args.source_id_column else None
        ) or entry_id or f"{args.source_id_prefix}-{row_number}"

        if source_id in seen_source_ids:
            summary.rejected_duplicate_source_id += 1
            summary.rejections.append(RowRejection(row_number, "duplicate_source_id"))
            continue
        seen_source_ids.add(source_id)

        if entry_id is None and headword is None:
            summary.missing_lexical_key += 1  # warning only -- record still included

        record = {
            "source_id": source_id,
            "source_title": args.source_title,
            "source_type": args.source_type,
            "content": content,
            "version": args.version,
            "effective_from": row_effective_from.isoformat() if row_effective_from else None,
            "authoritative": True if approved else False,
            "provenance_tag": _ACCEPTED_PROVENANCE_TAG if approved else None,
        }
        if row_effective_to is not None:
            record["effective_to"] = row_effective_to.isoformat()
        if entry_id is not None:
            record["entry_id"] = entry_id
        if headword is not None:
            record["lemma"] = headword
        if topic is not None:
            record["topic"] = topic
        if intent is not None:
            record["intent"] = intent

        output_records.append(record)
        summary.valid_rows += 1

    return output_records, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert an approved lexical export into the governed Official Knowledge JSON corpus schema.",
    )
    p.add_argument("--input", required=True, help="Path to the approved source export (.xlsx).")
    p.add_argument("--sheet", default=None, help="Sheet name/index to read (default: first sheet).")
    p.add_argument("--output", default=None, help="Path to write the governed JSON corpus (required unless --validate-only).")
    p.add_argument("--validate-only", action="store_true", help="Validate and print a summary; never write output.")
    p.add_argument(
        "--confirm-approved",
        action="store_true",
        help=(
            "Explicit operator attestation that this source has been organizationally approved as "
            "authoritative. Required to write any output file. Without it, records are still validated "
            "and reported on, but authoritative/provenance_tag are never set, so the tool refuses to write."
        ),
    )
    p.add_argument("--source-title", default=None, help="Name of the source/dictionary (e.g. \"المعجم\").")
    p.add_argument("--source-type", default=None, help="e.g. \"dictionary_entry\".")
    p.add_argument("--version", default=None, help="Version label for this export (e.g. \"2026-01\").")
    p.add_argument("--effective-from", default=None, help="ISO 8601 date/datetime this export becomes effective.")
    p.add_argument("--effective-to", default=None, help="ISO 8601 date/datetime this export stops being effective.")
    p.add_argument("--headword-column", default="الكلمة", help="Column holding the headword/lemma.")
    p.add_argument("--definition-column", default="التعريف", help="Column holding the definition text (required).")
    p.add_argument("--example-column", default="الشاهد", help="Column holding an example/citation. Pass '' to disable.")
    p.add_argument("--pos-column", default=None, help="Optional column holding part-of-speech.")
    p.add_argument("--entry-id-column", default=None, help="Optional column holding a stable entry id.")
    p.add_argument("--source-id-column", default=None, help="Optional column holding a stable source id.")
    p.add_argument("--topic-column", default=None, help="Optional column matching config.settings.TOPIC_SUBTYPES.")
    p.add_argument("--intent-column", default=None, help="Optional column matching config.settings.PRIMARY_INTENTS.")
    p.add_argument("--effective-from-column", default=None, help="Optional per-row effective_from override column.")
    p.add_argument("--effective-to-column", default=None, help="Optional per-row effective_to override column.")
    p.add_argument("--source-id-prefix", default="official-knowledge", help="Prefix for auto-generated source ids.")
    return p


def run(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.example_column == "":
        args.example_column = None

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    if not args.validate_only:
        if not args.output:
            print("error: --output is required unless --validate-only is set", file=sys.stderr)
            return 2
        missing_meta = [
            name
            for name, value in (
                ("--source-title", args.source_title),
                ("--source-type", args.source_type),
                ("--version", args.version),
                ("--effective-from", args.effective_from),
            )
            if not value
        ]
        if missing_meta:
            print(f"error: missing required metadata for a real write: {', '.join(missing_meta)}", file=sys.stderr)
            return 2
        if not args.confirm_approved:
            print(
                "error: refusing to write output without --confirm-approved. "
                "This tool never infers organizational approval from the input file itself -- "
                "pass --confirm-approved only after the source has actually been approved.",
                file=sys.stderr,
            )
            return 2

    df = _read_dataframe(input_path, args.sheet)
    missing_columns = _missing_columns(df, args)
    if missing_columns:
        print(f"error: input file is missing expected column(s): {missing_columns}", file=sys.stderr)
        return 2

    rows = df.to_dict(orient="records")
    # Approval is driven ONLY by the explicit operator flag -- never by
    # validate-only vs. write mode. --validate-only --confirm-approved
    # previews exactly what a real write would contain; --validate-only
    # alone previews it with authoritative/provenance_tag left unset.
    approved = bool(args.confirm_approved)
    records, summary = _normalize(rows, args=args, approved=approved)

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))

    if args.validate_only:
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} governed record(s) to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
