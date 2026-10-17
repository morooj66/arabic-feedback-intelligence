"""
Phase B.2 tests — AI-assisted gold-label suggestion.

Covers: AISuggestionResponse schema/business-rule validation, the
LABELED-only/is_few_shot_example-only few-shot filter and its determinism,
suggest_row's success/failure orchestration (mocked Gemini client, same
pattern as tests/test_classification.py), and scripts/run_ai_suggestions.py's
end-to-end safety guarantees using a synthetic gold CSV and an injected
fake client (no real API key/network call anywhere in this file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rcs.evaluation.ai_suggestion_schema import AISuggestionResponse  # noqa: E402
from rcs.evaluation.ai_suggester import suggest_row  # noqa: E402
from rcs.evaluation.few_shot import eligible_few_shot_pool, select_few_shot_examples  # noqa: E402
from rcs.evaluation.genai_client import GenAIClient  # noqa: E402
from rcs.evaluation.gold_dataset import AI_SUGGESTED_COLUMNS  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_ai_suggestions  # noqa: E402


# ---------------------------------------------------------------------------
# 1. AISuggestionResponse — schema + business rules
# ---------------------------------------------------------------------------

def _valid_payload(**overrides):
    payload = {
        "primary_intent": "SUGGESTION",
        "topic_subtype": "WORD_SUBMISSION",
        "secondary_intent": None,
        "subcategory": None,
        "sentiment": "NEUTRAL",
        "confidence": 0.8,
    }
    payload.update(overrides)
    return payload


def test_valid_payload_accepted():
    parsed = AISuggestionResponse(**_valid_payload())
    assert parsed.primary_intent == "SUGGESTION"


def test_invalid_primary_intent_rejected():
    with pytest.raises(ValidationError):
        AISuggestionResponse(**_valid_payload(primary_intent="NOT_A_REAL_INTENT"))


def test_invalid_sentiment_rejected():
    with pytest.raises(ValidationError):
        AISuggestionResponse(**_valid_payload(sentiment="MIXED"))


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        AISuggestionResponse(**_valid_payload(confidence=1.5))


def test_secondary_equal_primary_rejected():
    with pytest.raises(ValidationError):
        AISuggestionResponse(**_valid_payload(primary_intent="COMPLAINT", secondary_intent="COMPLAINT"))


def test_secondary_different_from_primary_accepted():
    parsed = AISuggestionResponse(**_valid_payload(primary_intent="COMPLAINT", secondary_intent="THANKS_PRAISE"))
    assert parsed.secondary_intent == "THANKS_PRAISE"


def test_subcategory_missing_when_required_rejected():
    with pytest.raises(ValidationError):
        AISuggestionResponse(**_valid_payload(topic_subtype="TECHNICAL_ISSUE", subcategory=None))


def test_subcategory_present_when_required_accepted():
    parsed = AISuggestionResponse(**_valid_payload(topic_subtype="TECHNICAL_ISSUE", subcategory="TechFault"))
    assert parsed.subcategory == "TechFault"


def test_subcategory_present_when_not_allowed_rejected():
    with pytest.raises(ValidationError):
        AISuggestionResponse(**_valid_payload(topic_subtype="GENERAL", subcategory="TechFault"))


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        AISuggestionResponse(**_valid_payload(), reply_draft="not allowed")


# ---------------------------------------------------------------------------
# 1b. genai_client.py — current google-genai SDK wrapper (construction only;
#     no real network call is ever made in this test file)
# ---------------------------------------------------------------------------

def test_genai_client_requires_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GenAIClient(api_key=None, model_name="gemini-3.6-flash")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GenAIClient(api_key="", model_name="gemini-3.6-flash")


def test_genai_client_construction_does_not_import_sdk_or_connect():
    # Constructing (without calling .generate()) must never require the
    # google-genai package to even be importable yet, same lazy-import
    # discipline as gemini_client.py -- this just checks it doesn't raise.
    client = GenAIClient(api_key="fake-key-not-a-real-secret", model_name="gemini-3.6-flash")
    assert client is not None


# ---------------------------------------------------------------------------
# 2. few_shot.py — eligibility filter + deterministic selection
# ---------------------------------------------------------------------------

def _synthetic_gold(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "comment_id": "", "comment_text": "", "gold_primary_intent": "",
        "gold_topic_subtype": "", "gold_secondary_intent": "", "gold_subcategory": "",
        "gold_sentiment": "", "gold_annotation_status": "", "labeler_notes": "",
        "labeled_by": "", "labeled_at": "", "split": "train",
        "annotation_stage": "human_train_seed", "is_few_shot_example": "True",
    }
    for col in AI_SUGGESTED_COLUMNS:
        defaults[col] = ""
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


def test_eligible_pool_excludes_insufficient_context():
    gold = _synthetic_gold([
        {"comment_id": "a", "gold_primary_intent": "SPAM", "gold_annotation_status": "LABELED", "is_few_shot_example": "True"},
        {"comment_id": "b", "gold_primary_intent": "SPAM", "gold_annotation_status": "INSUFFICIENT_CONTEXT", "is_few_shot_example": "True"},
    ])
    pool = eligible_few_shot_pool(gold)
    assert set(pool["comment_id"]) == {"a"}


def test_eligible_pool_excludes_non_seed_rows():
    gold = _synthetic_gold([
        {"comment_id": "a", "gold_primary_intent": "SPAM", "gold_annotation_status": "LABELED", "is_few_shot_example": "True"},
        {"comment_id": "b", "gold_primary_intent": "SPAM", "gold_annotation_status": "LABELED", "is_few_shot_example": "False", "annotation_stage": "ai_assisted_train"},
    ])
    pool = eligible_few_shot_pool(gold)
    assert set(pool["comment_id"]) == {"a"}


def test_eligible_pool_excludes_uncertain_status():
    gold = _synthetic_gold([
        {"comment_id": "a", "gold_annotation_status": "LABELED"},
        {"comment_id": "b", "gold_annotation_status": "UNCERTAIN"},
    ])
    pool = eligible_few_shot_pool(gold)
    assert set(pool["comment_id"]) == {"a"}


def test_select_few_shot_caps_per_intent_and_never_pads():
    rows = [
        {"comment_id": f"spam_{i}", "gold_primary_intent": "SPAM", "gold_annotation_status": "LABELED"}
        for i in range(5)
    ] + [
        {"comment_id": "thanks_1", "gold_primary_intent": "THANKS_PRAISE", "gold_annotation_status": "LABELED"},
    ]
    gold = _synthetic_gold(rows)
    selected = select_few_shot_examples(gold, max_per_intent=2, seed=42)
    by_intent = {}
    for ex in selected:
        by_intent.setdefault(ex["gold_primary_intent"], []).append(ex["comment_id"])
    assert len(by_intent["SPAM"]) == 2  # capped, not all 5
    assert len(by_intent["THANKS_PRAISE"]) == 1  # never padded past what's available


def test_select_few_shot_is_deterministic():
    rows = [
        {"comment_id": f"spam_{i}", "gold_primary_intent": "SPAM", "gold_annotation_status": "LABELED"}
        for i in range(8)
    ]
    gold = _synthetic_gold(rows)
    first = select_few_shot_examples(gold, max_per_intent=2, seed=42)
    second = select_few_shot_examples(gold, max_per_intent=2, seed=42)
    assert [e["comment_id"] for e in first] == [e["comment_id"] for e in second]


def test_select_few_shot_empty_pool_returns_empty_list():
    gold = _synthetic_gold([{"comment_id": "a", "gold_annotation_status": "INSUFFICIENT_CONTEXT"}])
    assert select_few_shot_examples(gold) == []


# ---------------------------------------------------------------------------
# 3. suggest_row — mocked Gemini client, same double pattern as Phase 4
# ---------------------------------------------------------------------------

class FakeClient:
    def __init__(self, response_text=None, raises=False, fail_times=0):
        self._response_text = response_text
        self._raises = raises
        self._fail_times = fail_times
        self.calls = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.calls += 1
        if self._raises:
            raise ConnectionError("simulated network failure")
        if self.calls <= self._fail_times:
            raise ConnectionError("simulated transient failure")
        return self._response_text


_VALID_JSON = json.dumps({
    "primary_intent": "COMPLAINT",
    "topic_subtype": "TECHNICAL_ISSUE",
    "secondary_intent": None,
    "subcategory": "TechFault",
    "sentiment": "NEGATIVE",
    "confidence": 0.91,
})


def test_suggest_row_success():
    client = FakeClient(response_text=_VALID_JSON)
    outcome = suggest_row({"comment_text": "لا يعمل"}, few_shot_examples=[], client=client, max_retries=2)
    assert outcome.success is True
    assert outcome.primary_intent == "COMPLAINT"
    assert outcome.subcategory == "TechFault"
    assert outcome.confidence == 0.91
    assert outcome.attempts == 1


def test_suggest_row_retries_then_succeeds():
    client = FakeClient(response_text=_VALID_JSON, fail_times=1)
    outcome = suggest_row({"comment_text": "لا يعمل"}, few_shot_examples=[], client=client, max_retries=2)
    assert outcome.success is True
    assert outcome.attempts == 2


def test_suggest_row_api_error_exhausts_retries():
    client = FakeClient(raises=True)
    outcome = suggest_row({"comment_text": "x"}, few_shot_examples=[], client=client, max_retries=2)
    assert outcome.success is False
    assert "api_error" in outcome.error
    assert client.calls == 3  # max_retries=2 -> 3 attempts


def test_suggest_row_invalid_json_fails_cleanly():
    client = FakeClient(response_text="not json")
    outcome = suggest_row({"comment_text": "x"}, few_shot_examples=[], client=client, max_retries=0)
    assert outcome.success is False
    assert "validation_failed" in outcome.error


def test_suggest_row_schema_violation_fails_cleanly():
    bad_json = json.dumps({
        "primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE",
        "secondary_intent": "COMPLAINT",  # equals primary -> business-rule violation
        "subcategory": "TechFault", "sentiment": "NEGATIVE", "confidence": 0.5,
    })
    client = FakeClient(response_text=bad_json)
    outcome = suggest_row({"comment_text": "x"}, few_shot_examples=[], client=client, max_retries=0)
    assert outcome.success is False
    assert "validation_failed" in outcome.error


def test_suggest_row_never_writes_partial_result_on_failure():
    client = FakeClient(raises=True)
    outcome = suggest_row({"comment_text": "x"}, few_shot_examples=[], client=client, max_retries=0)
    assert outcome.primary_intent is None
    assert outcome.topic_subtype is None
    assert outcome.confidence is None


# ---------------------------------------------------------------------------
# 4. scripts/run_ai_suggestions.run() — end-to-end safety, synthetic CSV
# ---------------------------------------------------------------------------

def _write_synthetic_gold_csv(tmp_path) -> Path:
    rows = [
        # few-shot source (human_train_seed, LABELED) — enough per-intent coverage
        {"comment_id": "seed1", "comment_text": "شكرا", "gold_primary_intent": "THANKS_PRAISE",
         "gold_topic_subtype": "GENERAL", "gold_sentiment": "POSITIVE", "gold_annotation_status": "LABELED",
         "split": "train", "annotation_stage": "human_train_seed", "is_few_shot_example": "True"},
        {"comment_id": "seed2", "comment_text": "الخدمة سيئة", "gold_primary_intent": "COMPLAINT",
         "gold_topic_subtype": "GENERAL", "gold_sentiment": "NEGATIVE", "gold_annotation_status": "LABELED",
         "split": "train", "annotation_stage": "human_train_seed", "is_few_shot_example": "True"},
        # a val/test row — must NEVER receive an ai_suggested_* write
        {"comment_id": "val1", "comment_text": "سؤال عن كلمة", "gold_primary_intent": "INQUIRY",
         "gold_topic_subtype": "WORD_SUBMISSION", "gold_sentiment": "NEUTRAL", "gold_annotation_status": "LABELED",
         "split": "val", "annotation_stage": "human_val", "is_few_shot_example": "False"},
        {"comment_id": "test1", "comment_text": "اقتراح جديد", "gold_primary_intent": "SUGGESTION",
         "gold_topic_subtype": "DICTIONARY_SUGGESTION", "gold_sentiment": "NEUTRAL", "gold_annotation_status": "LABELED",
         "split": "test", "annotation_stage": "human_test", "is_few_shot_example": "False"},
        # ai_assisted_train targets
        {"comment_id": "ai1", "comment_text": "لا يعمل الموقع", "gold_primary_intent": "", "gold_topic_subtype": "",
         "gold_sentiment": "", "gold_annotation_status": "", "split": "train",
         "annotation_stage": "ai_assisted_train", "is_few_shot_example": "False"},
        {"comment_id": "ai2", "comment_text": "بارك الله فيكم", "gold_primary_intent": "", "gold_topic_subtype": "",
         "gold_sentiment": "", "gold_annotation_status": "", "split": "train",
         "annotation_stage": "ai_assisted_train", "is_few_shot_example": "False"},
    ]
    defaults = {"gold_secondary_intent": "", "gold_subcategory": "", "labeler_notes": "", "labeled_by": "", "labeled_at": ""}
    for col in AI_SUGGESTED_COLUMNS:
        defaults[col] = ""
    full_rows = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(full_rows)
    path = tmp_path / "gold_dataset_round1.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


class SequencedFakeClient:
    """Returns one canned response per call, in order — lets a test control
    exactly what each of the N target rows in a run gets back."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        response = self._responses[self.calls]
        self.calls += 1
        if response is None:
            raise ConnectionError("simulated failure")
        return response


def test_run_writes_only_ai_suggested_columns_on_target_rows(tmp_path):
    csv_path = _write_synthetic_gold_csv(tmp_path)
    responses = [
        json.dumps({"primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE", "secondary_intent": None,
                     "subcategory": "TechFault", "sentiment": "NEGATIVE", "confidence": 0.85}),
        json.dumps({"primary_intent": "THANKS_PRAISE", "topic_subtype": "GENERAL", "secondary_intent": None,
                     "subcategory": None, "sentiment": "POSITIVE", "confidence": 0.4}),
    ]
    client = SequencedFakeClient(responses)

    result = run_ai_suggestions.run(client=client, gold_csv_path=csv_path)

    assert result["succeeded"] == 2
    assert result["failed"] == 0
    assert result["low_confidence_count"] == 1  # the 0.4 one, threshold 0.6

    updated = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    ai1 = updated[updated["comment_id"] == "ai1"].iloc[0]
    assert ai1["ai_suggested_primary_intent"] == "COMPLAINT"
    assert ai1["ai_suggested_sentiment"] == "NEGATIVE"
    assert ai1["gold_primary_intent"] == ""  # gold_* never touched

    # val/test rows: zero ai_suggested_* writes, defensively re-checked
    for cid in ["val1", "test1", "seed1", "seed2"]:
        row = updated[updated["comment_id"] == cid].iloc[0]
        for col in AI_SUGGESTED_COLUMNS:
            assert row[col] == "", f"{cid}.{col} should still be blank"


def test_run_leaves_row_blank_on_suggestion_failure(tmp_path):
    csv_path = _write_synthetic_gold_csv(tmp_path)
    responses = [None, None, None, None, None, None]  # exhaust retries -> failure for ai1
    responses += [json.dumps({"primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE",
                                "secondary_intent": None, "subcategory": "TechFault",
                                "sentiment": "NEGATIVE", "confidence": 0.9})] * 3
    client = SequencedFakeClient(responses)

    result = run_ai_suggestions.run(client=client, gold_csv_path=csv_path, limit=1)

    assert result["targets_attempted"] == 1
    assert result["succeeded"] == 0
    assert result["failed"] == 1

    updated = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    ai1 = updated[updated["comment_id"] == "ai1"].iloc[0]
    for col in AI_SUGGESTED_COLUMNS:
        assert ai1[col] == ""


def test_run_respects_limit(tmp_path):
    csv_path = _write_synthetic_gold_csv(tmp_path)
    response = json.dumps({"primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE",
                            "secondary_intent": None, "subcategory": "TechFault",
                            "sentiment": "NEGATIVE", "confidence": 0.9})
    client = SequencedFakeClient([response])
    result = run_ai_suggestions.run(client=client, gold_csv_path=csv_path, limit=1)
    assert result["targets_attempted"] == 1
    assert client.calls == 1


def test_run_skips_already_suggested_rows_by_default(tmp_path):
    csv_path = _write_synthetic_gold_csv(tmp_path)
    response = json.dumps({"primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE",
                            "secondary_intent": None, "subcategory": "TechFault",
                            "sentiment": "NEGATIVE", "confidence": 0.9})
    # first run: suggest both ai1 and ai2
    client1 = SequencedFakeClient([response, response])
    result1 = run_ai_suggestions.run(client=client1, gold_csv_path=csv_path)
    assert result1["succeeded"] == 2
    assert result1["already_suggested_skipped"] == 0

    # second run (resume): both already have a suggestion -> zero calls made, zero targets
    client2 = SequencedFakeClient([])
    result2 = run_ai_suggestions.run(client=client2, gold_csv_path=csv_path)
    assert result2["already_suggested_skipped"] == 2
    assert result2["targets_attempted"] == 0
    assert client2.calls == 0


def test_run_force_reprocesses_already_suggested_rows(tmp_path):
    csv_path = _write_synthetic_gold_csv(tmp_path)
    response = json.dumps({"primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE",
                            "secondary_intent": None, "subcategory": "TechFault",
                            "sentiment": "NEGATIVE", "confidence": 0.9})
    client1 = SequencedFakeClient([response, response])
    run_ai_suggestions.run(client=client1, gold_csv_path=csv_path)

    updated_response = json.dumps({"primary_intent": "SUGGESTION", "topic_subtype": "GENERAL",
                                     "secondary_intent": None, "subcategory": None,
                                     "sentiment": "NEUTRAL", "confidence": 0.7})
    client2 = SequencedFakeClient([updated_response, updated_response])
    result2 = run_ai_suggestions.run(client=client2, gold_csv_path=csv_path, force=True)
    assert result2["targets_attempted"] == 2
    assert result2["succeeded"] == 2

    updated = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    ai1 = updated[updated["comment_id"] == "ai1"].iloc[0]
    assert ai1["ai_suggested_primary_intent"] == "SUGGESTION"  # overwritten by --force


def test_run_resume_after_partial_failure_only_retries_failed_rows(tmp_path):
    csv_path = _write_synthetic_gold_csv(tmp_path)
    response = json.dumps({"primary_intent": "COMPLAINT", "topic_subtype": "TECHNICAL_ISSUE",
                            "secondary_intent": None, "subcategory": "TechFault",
                            "sentiment": "NEGATIVE", "confidence": 0.9})
    # ai1 succeeds, ai2 fails (exhausts retries: max_retries default is settings.GEMINI_MAX_RETRIES = 2 -> 3 attempts)
    client1 = SequencedFakeClient([response, None, None, None])
    result1 = run_ai_suggestions.run(client=client1, gold_csv_path=csv_path)
    assert result1["succeeded"] == 1
    assert result1["failed"] == 1

    updated = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert updated[updated["comment_id"] == "ai1"].iloc[0]["ai_suggested_primary_intent"] == "COMPLAINT"
    assert updated[updated["comment_id"] == "ai2"].iloc[0]["ai_suggested_primary_intent"] == ""

    # resume: only ai2 (the failed one) should be retried, not ai1
    client2 = SequencedFakeClient([response])
    result2 = run_ai_suggestions.run(client=client2, gold_csv_path=csv_path)
    assert result2["already_suggested_skipped"] == 1
    assert result2["targets_attempted"] == 1
    assert result2["succeeded"] == 1
    assert client2.calls == 1

    final = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert final[final["comment_id"] == "ai2"].iloc[0]["ai_suggested_primary_intent"] == "COMPLAINT"


def test_run_raises_without_few_shot_pool(tmp_path):
    rows = [
        {"comment_id": "ai1", "comment_text": "x", "split": "train",
         "annotation_stage": "ai_assisted_train", "is_few_shot_example": "False",
         "gold_primary_intent": "", "gold_topic_subtype": "", "gold_sentiment": "", "gold_annotation_status": ""},
    ]
    defaults = {"gold_secondary_intent": "", "gold_subcategory": "", "labeler_notes": "", "labeled_by": "", "labeled_at": ""}
    for col in AI_SUGGESTED_COLUMNS:
        defaults[col] = ""
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    path = tmp_path / "gold_dataset_round1.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(RuntimeError, match="no eligible few-shot examples"):
        run_ai_suggestions.run(client=SequencedFakeClient([]), gold_csv_path=path)
