#!/usr/bin/env python3
"""
Phase 3 — OFFLINE A/B evaluation: current rg-v7 vs the research-only
rg-v8-candidate-context prompt (src/rcs/response_generation/prompt_candidates.py).

Read-only over review_items: response A is the ALREADY-PERSISTED real
rg-v7 draft for each selected case (never regenerated — same model, same
temperature/config, same classification, by definition, since it's the
actual historical output). Response B is one NEW live Gemini call per
case using the candidate prompt, same client/model. Nothing here writes
to review_items, the reviewer queue, or the source workbook.

Case selection (hardcoded, curated from already-processed real rows —
NOT human_test / frozen gold data):
  - 7 cases where entry_id/lemma linkage plausibly matters: short
    comments linked to a dictionary entry, missing-definition
    complaints, entry-correction comments, one ambiguous/garbled comment.
  - 3 control cases with no entry_id/lemma at all, so the candidate is
    not unfairly advantaged by every case having extra context.

Output (data/feedback_research/, new directory, nothing else touched):
  - rg_v8_context_ab_eval_blind.csv   — for human judgment. No prompt-
    version labels, no comment_id, no entry_id, no visitor PII (comment
    text redacted the same way it already is before reaching Gemini).
  - rg_v8_context_ab_eval_key.json    — hidden mapping of which physical
    column (response_A/response_B) held which prompt version per case,
    plus comment_id/entry_id for reproducibility. Not for the blind judge.

  python scripts/run_rg_v8_context_ab_eval.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from rcs.response_generation.prompt import build_generation_prompt, redact_for_generation  # noqa: E402
from rcs.response_generation.prompt_candidates import PROMPT_VERSION as CANDIDATE_VERSION  # noqa: E402
from rcs.response_generation.prompt_candidates import build_generation_prompt_v8_candidate  # noqa: E402
from rcs.response_generation.schema import JSON_SCHEMA, GeminiDraftResponse  # noqa: E402
from rcs.response_generation.service import build_live_gemini_client  # noqa: E402
from rcs.response_generation.validate import validate_draft  # noqa: E402
from rcs.review.models import ReviewItem  # noqa: E402
from rcs.review.repository import get_engine, init_db  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

RANDOM_SEED = 42
SOURCE_FILE = "historical-export.xlsx"

# (source_row_index, why it was picked)
CASE_ROW_INDICES = [
    (1, "entry_correction_linked_to_entry"),
    (10, "inquiry_linked_to_entry"),
    (68, "ambiguous_comment_linked_to_entry"),
    (74, "short_correction_linked_to_entry"),
    (93, "suggestion_linked_to_entry"),
    (101, "missing_definition_complaint_linked_to_entry"),
    (102, "missing_definition_complaint_linked_to_entry"),
    (18, "control_no_entry_context"),
    (84, "control_no_entry_context"),
    (180, "control_no_entry_context"),
]

OUT_DIR = PROJECT_ROOT / "data" / "feedback_research"
BLIND_CSV = OUT_DIR / "rg_v8_context_ab_eval_blind.csv"
KEY_JSON = OUT_DIR / "rg_v8_context_ab_eval_key.json"


def _parse_lemma(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
        if isinstance(parsed, str):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return str(raw).strip("[]\"' ") or None


def _generate_candidate(client, *, comment_text, intent, topic, sentiment, lemma) -> tuple[str | None, str | None]:
    """Returns (draft_or_None, error_reason_or_None). Never fabricates a
    draft on failure — mirrors generate_suggested_response()'s fail-closed
    behavior."""
    prompt = build_generation_prompt_v8_candidate(
        comment_text=comment_text,
        primary_intent=intent,
        topic_subtype=topic,
        sentiment=sentiment,
        reason_code="offline_ab_eval",
        lemma=lemma,
    )
    try:
        raw = client.generate(prompt, JSON_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return None, f"api_error: {type(exc).__name__}: {exc}"
    try:
        parsed = GeminiDraftResponse.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        return None, f"malformed_output: {exc}"
    ok, why = validate_draft(parsed.suggested_response)
    if not ok:
        return None, f"validation_failed: {why}"
    return parsed.suggested_response.strip(), None


FIELDNAMES = [
    "case_id",
    "comment_text",
    "lemma",
    "intent",
    "topic",
    "sentiment",
    "response_A",
    "response_B",
    "judgment",
    "note",
]


def build_ab_eval_artifacts(
    engine,
    *,
    source_file: str,
    case_row_indices: list[tuple[int, str]],
    client,
    seed: int = RANDOM_SEED,
) -> dict:
    """Core, testable logic: read-only over `engine` (never writes to
    review_items), one live `client.generate()` call per case for the
    candidate side only. Returns {"blind_rows": [...], "key_payload": {...}}
    — callers decide how/whether to persist them (main() writes CSV/JSON
    files; tests can inspect the dicts directly)."""
    row_indices = [idx for idx, _ in case_row_indices]
    with Session(engine) as session:
        rows = (
            session.execute(
                select(ReviewItem).where(
                    ReviewItem.source_file == source_file,
                    ReviewItem.source_row_index.in_(row_indices),
                )
            )
            .scalars()
            .all()
        )
        by_row = {r.source_row_index: r for r in rows}

        rng = random.Random(seed)
        blind_rows = []
        key_entries = {}
        skipped = []

        for i, (row_idx, reason_tag) in enumerate(case_row_indices, start=1):
            item = by_row.get(row_idx)
            if item is None or not item.suggested_final_reply:
                skipped.append({"row_index": row_idx, "reason": "no_persisted_rg_v7_draft_found"})
                continue

            case_id = f"case-{i:02d}"
            lemma = _parse_lemma(item.lemma)
            comment_text_redacted = redact_for_generation(item.original_comment)

            response_v7 = item.suggested_final_reply.strip()
            response_candidate, error = _generate_candidate(
                client,
                comment_text=item.original_comment or "",
                intent=item.ai_primary_intent,
                topic=item.ai_topic_subtype,
                sentiment=item.ai_sentiment,
                lemma=lemma,
            )
            if response_candidate is None:
                skipped.append({"row_index": row_idx, "case_id": case_id, "reason": error})
                continue

            # True blind A/B: which physical column holds which version is
            # randomized per case (fixed seed -> reproducible), not fixed.
            if rng.random() < 0.5:
                col_a, col_b = response_v7, response_candidate
                version_a, version_b = "rg-v7", CANDIDATE_VERSION
            else:
                col_a, col_b = response_candidate, response_v7
                version_a, version_b = CANDIDATE_VERSION, "rg-v7"

            blind_rows.append(
                {
                    "case_id": case_id,
                    "comment_text": comment_text_redacted,
                    "lemma": lemma or "",
                    "intent": item.ai_primary_intent or "",
                    "topic": item.ai_topic_subtype or "",
                    "sentiment": item.ai_sentiment or "",
                    "response_A": col_a,
                    "response_B": col_b,
                    "judgment": "",  # human fills in: A better / B better / Equivalent / Neither acceptable
                    "note": "",
                }
            )
            key_entries[case_id] = {
                "comment_id": item.comment_id,
                "source_row_index": row_idx,
                "case_tag": reason_tag,
                "entry_id": item.entry_id,  # provenance only; never shown to the blind judge
                "response_A_version": version_a,
                "response_B_version": version_b,
            }

    key_payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z",
        "random_seed": seed,
        "source_db": str(getattr(engine, "url", "")),
        "candidate_prompt_version": CANDIDATE_VERSION,
        "production_prompt_version": "rg-v7",
        "cases": key_entries,
        "skipped": skipped,
    }
    return {"blind_rows": blind_rows, "key_payload": key_payload}


def write_ab_eval_files(artifacts: dict, *, blind_csv: Path, key_json: Path) -> None:
    import csv

    blind_csv.parent.mkdir(parents=True, exist_ok=True)
    with blind_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in artifacts["blind_rows"]:
            writer.writerow(row)
    key_json.parent.mkdir(parents=True, exist_ok=True)
    key_json.write_text(json.dumps(artifacts["key_payload"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if not settings.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is required for the live rg-v8-candidate-context calls.", file=sys.stderr)
        return 2

    engine = get_engine(settings.EXCEL_OPERATIONAL_DB)
    init_db(engine)
    client = build_live_gemini_client()

    artifacts = build_ab_eval_artifacts(
        engine,
        source_file=SOURCE_FILE,
        case_row_indices=CASE_ROW_INDICES,
        client=client,
    )
    write_ab_eval_files(artifacts, blind_csv=BLIND_CSV, key_json=KEY_JSON)

    print(
        json.dumps(
            {
                "cases_written": len(artifacts["blind_rows"]),
                "skipped": artifacts["key_payload"]["skipped"],
                "blind_csv": str(BLIND_CSV),
                "key_json": str(KEY_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
