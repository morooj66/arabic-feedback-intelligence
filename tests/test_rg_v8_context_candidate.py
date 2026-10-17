"""Phase 3 — rg-v8-candidate-context prompt + offline A/B eval artifact.

Covers: candidate prompt construction (lemma included, entry_id NEVER
sent to the model), fail-closed generation, no PII in the blind
artifact, no prompt-version leakage into the blind artifact, and — most
importantly — that building the artifact never writes to review_items
(production rg-v7 records/state remain byte-for-byte untouched).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS
from rcs.response_generation.prompt_candidates import (
    PROMPT_VERSION,
    build_generation_prompt_v8_candidate,
)
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_ab_eval_module():
    spec = importlib.util.spec_from_file_location(
        "rg_v8_context_ab_eval_script", PROJECT_ROOT / "scripts" / "run_rg_v8_context_ab_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubClient:
    """Deterministic offline stand-in for the real Gemini client — no network."""

    def __init__(self, *, fail_for: set[str] | None = None):
        self.calls: list[str] = []
        self.fail_for = fail_for or set()

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.calls.append(prompt)
        for marker in self.fail_for:
            if marker in prompt:
                raise RuntimeError("simulated failure")
        return json.dumps({"suggested_response": "شكرًا لتعليقكم المرتبط بهذا المدخل."}, ensure_ascii=False)


def test_candidate_prompt_reuses_rg_v7_system_instructions_verbatim():
    prompt = build_generation_prompt_v8_candidate(
        comment_text="تعليق تجريبي",
        primary_intent="SUGGESTION",
        topic_subtype="WORD_SUBMISSION",
        sentiment="NEUTRAL",
        reason_code="test",
        lemma="كلمة",
    )
    assert SYSTEM_INSTRUCTIONS in prompt  # not rewritten, not duplicated-with-drift


def test_candidate_prompt_includes_lemma_when_available():
    prompt = build_generation_prompt_v8_candidate(
        comment_text="تعليق تجريبي",
        primary_intent="SUGGESTION",
        topic_subtype="WORD_SUBMISSION",
        sentiment="NEUTRAL",
        reason_code="test",
        lemma="اِعْتِبَار",
    )
    assert "اِعْتِبَار" in prompt
    assert "DICTIONARY ENTRY CONTEXT" in prompt


def test_candidate_prompt_omits_entry_block_without_lemma():
    prompt = build_generation_prompt_v8_candidate(
        comment_text="تعليق تجريبي",
        primary_intent="SUGGESTION",
        topic_subtype="WORD_SUBMISSION",
        sentiment="NEUTRAL",
        reason_code="test",
        lemma=None,
    )
    assert "--- DICTIONARY ENTRY CONTEXT (system-provided" not in prompt


def test_candidate_prompt_never_receives_entry_id():
    # build_generation_prompt_v8_candidate has no entry_id parameter at all —
    # structural guarantee, not just a runtime check.
    import inspect

    sig = inspect.signature(build_generation_prompt_v8_candidate)
    assert "entry_id" not in sig.parameters


def _seed_case_row(engine, *, comment_id: str, row_index: int, lemma_json: str | None, entry_id: str | None):
    item = ReviewItem(
        comment_id=comment_id,
        source_file="ab_eval_test.xlsx",
        source_row_index=row_index,
        original_comment="أقترح كلمة جديدة مرتبطة بمدخل معجمي محدد",
        entry_id=entry_id,
        lemma=lemma_json,
        review_status="DRAFT_READY",
        ai_primary_intent="SUGGESTION",
        ai_topic_subtype="WORD_SUBMISSION",
        ai_sentiment="NEUTRAL",
        suggested_final_reply="مسودة الذكاء الاصطناعي الأصلية المحفوظة فعليًا.",
        ai_suggested_final_reply="مسودة الذكاء الاصطناعي الأصلية المحفوظة فعليًا.",
        generation_status="SUCCEEDED",
        generation_routing_decision="GENERATE_DRAFT",
        generation_provider="gemini",
        generation_model_name="gemini-3.6-flash",
        generation_prompt_version="rg-v7",
        source_reference=f"ab_eval_test.xlsx:{row_index}",
    )
    with Session(engine) as session:
        session.add(item)
        session.commit()


def test_ab_eval_artifact_has_no_pii_no_version_labels_and_never_writes_to_review_items(tmp_path):
    module = _load_ab_eval_module()
    engine = get_engine(tmp_path / "ab_eval.sqlite")
    init_db(engine)
    _seed_case_row(
        engine,
        comment_id="case1",
        row_index=1,
        lemma_json=json.dumps(["اِعْتِبَار"], ensure_ascii=False),
        entry_id="6b746072-ca71-483f-8b73-c52da0efbdde",
    )
    _seed_case_row(engine, comment_id="case2", row_index=2, lemma_json=None, entry_id=None)

    with Session(engine) as session:
        before = {
            item.comment_id: (item.suggested_final_reply, item.review_status, item.reviewer_action)
            for item in session.query(ReviewItem).all()
        }

    client = _StubClient()
    artifacts = module.build_ab_eval_artifacts(
        engine,
        source_file="ab_eval_test.xlsx",
        case_row_indices=[(1, "entry_linked"), (2, "control_no_entry")],
        client=client,
    )

    blind_rows = artifacts["blind_rows"]
    assert len(blind_rows) == 2
    for row in blind_rows:
        row_text = json.dumps(row, ensure_ascii=False)
        assert "comment_id" not in row  # no traceable ID in the blind file
        assert "entry_id" not in row
        assert "rg-v7" not in row_text  # no version label leaked into the blind file
        assert "rg-v8" not in row_text
        assert "6b746072-ca71-483f-8b73-c52da0efbdde" not in row_text  # raw entry_id never present
        assert row["judgment"] == ""
        assert row["note"] == ""

    key = artifacts["key_payload"]
    assert set(key["cases"].keys()) == {"case-01", "case-02"}
    for case_id, entry in key["cases"].items():
        assert {entry["response_A_version"], entry["response_B_version"]} == {"rg-v7", module.CANDIDATE_VERSION}
    assert key["cases"]["case-01"]["entry_id"] == "6b746072-ca71-483f-8b73-c52da0efbdde"
    assert key["cases"]["case-02"]["entry_id"] is None

    # Production rg-v7 records are byte-for-byte untouched.
    with Session(engine) as session:
        after = {
            item.comment_id: (item.suggested_final_reply, item.review_status, item.reviewer_action)
            for item in session.query(ReviewItem).all()
        }
    assert before == after


def test_ab_eval_fails_closed_when_candidate_generation_fails(tmp_path):
    module = _load_ab_eval_module()
    engine = get_engine(tmp_path / "ab_eval_fail.sqlite")
    init_db(engine)
    _seed_case_row(engine, comment_id="case1", row_index=1, lemma_json=None, entry_id=None)

    client = _StubClient(fail_for={"case1", "أقترح"})  # forces the .generate() call to raise
    artifacts = module.build_ab_eval_artifacts(
        engine,
        source_file="ab_eval_test.xlsx",
        case_row_indices=[(1, "forced_failure")],
        client=client,
    )
    assert artifacts["blind_rows"] == []  # never fabricated a stand-in draft
    assert len(artifacts["key_payload"]["skipped"]) == 1
    assert "api_error" in artifacts["key_payload"]["skipped"][0]["reason"]


def test_write_ab_eval_files_produces_valid_csv_and_json(tmp_path):
    module = _load_ab_eval_module()
    engine = get_engine(tmp_path / "ab_eval_write.sqlite")
    init_db(engine)
    _seed_case_row(engine, comment_id="case1", row_index=1, lemma_json=None, entry_id=None)

    client = _StubClient()
    artifacts = module.build_ab_eval_artifacts(
        engine,
        source_file="ab_eval_test.xlsx",
        case_row_indices=[(1, "control_no_entry")],
        client=client,
    )
    blind_csv = tmp_path / "blind.csv"
    key_json = tmp_path / "key.json"
    module.write_ab_eval_files(artifacts, blind_csv=blind_csv, key_json=key_json)

    assert blind_csv.exists() and key_json.exists()
    import csv

    with blind_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(module.FIELDNAMES)

    key_data = json.loads(key_json.read_text(encoding="utf-8"))
    assert key_data["candidate_prompt_version"] == PROMPT_VERSION
    assert key_data["production_prompt_version"] == "rg-v7"
