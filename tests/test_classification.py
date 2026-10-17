"""
Phase 4 tests: Gemini classification, kept strictly mocked — no live API
calls anywhere in this file.

Hierarchical taxonomy (MIGRATION_PLAN.md): PRIMARY_INTENT x TOPIC_SUBTYPE
replaces the old flat 9-category taxonomy. Gemini's structured response is
classification ONLY — no reply drafting (immediate_reply_draft /
suggested_final_reply removed from the schema entirely).

Covers all required scenarios:
  1. each of the 6 primary intents classifies/routes correctly, including
     the two topic-based routing overrides (WORD_OF_YEAR_NOMINATION,
     OFF_TOPIC_PERSONAL) and the worked examples from the product spec
  2. only non-excluded rows are selected for classification
  3. excluded rows never trigger a Gemini call
  4. phone-number redaction happens before the model ever sees the text
  5. the original stored comment is never modified by classification
  6. valid structured Gemini output is parsed and persisted correctly
  7. malformed/invalid output fails safely (no invented intent/topic)
  8. API/network failure fails safely and is distinguishable from (7)
  9. bounded retry behavior
 10. re-running never overwrites an existing successful AI classification
 11. reviewer fields are never touched by classification
 12. the schema has no reply-draft fields, and a stray one is rejected
"""

from __future__ import annotations

import datetime as dt

import pytest

from config import settings
from rcs.classification.classifier import ClassificationOutcome, classify_row
from rcs.classification.prompt import build_prompt
from rcs.classification.redact import redact_phone_numbers
from rcs.classification.schema import GeminiClassificationResponse
from rcs.review.repository import (
    apply_classification_results,
    apply_prefilter_results,
    get_engine,
    init_db,
    load_for_classification,
    upsert_review_items,
)
from rcs.routing.engine import determine_routing


# ---------------------------------------------------------------------------
# Test doubles — no real Gemini SDK involved anywhere in this file
# ---------------------------------------------------------------------------

class FakeGeminiClient:
    """Returns a pre-scripted sequence of responses/exceptions, one per
    call, and counts how many times .generate() was invoked."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.call_count = 0
        self.prompts_seen = []

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.call_count += 1
        self.prompts_seen.append(prompt)
        if not self._scripted:
            raise AssertionError("FakeGeminiClient called more times than scripted")
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _valid_json(**overrides) -> str:
    import json
    base = {
        "primary_intent": "THANKS_PRAISE",
        "topic_subtype": "GENERAL",
        "subcategory": None,
        "secondary_intent": None,
        "is_content_meaningful": True,
        "confidence": 0.95,
        "classification_reason": "Visitor expresses simple gratitude.",
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def _row(**overrides) -> dict:
    row = {
        "original_comment": "شكرًا لكم على هذا المجهود الرائع",
        "original_title": "إثراء المعجم",
        "original_type": "أرجو النظر في ترشيح كلمة جديدة شائعة بين الناس لإضافتها في المعجم",
        "age_group": "old",
        "platform": "الرياض",
        "entry_id": None,
        "lemma": None,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# 1. Each primary intent classifies/routes correctly, incl. topic overrides
#    and the worked examples from MIGRATION_PLAN.md / the product spec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "primary_intent,topic_subtype,subcategory,secondary_intent,meaningful,expected_routing",
    [
        # one baseline case per primary intent
        ("SPAM", "GENERAL", None, None, True, "EXCLUDE"),
        ("THANKS_PRAISE", "GENERAL", None, None, True, "AUTO_REPLY_CANDIDATE"),  # "شكرًا لكم، عمل رائع"
        # leftover DUA_BLESSING (merged 2026-08-24) keeps THANKS routing
        ("DUA_BLESSING", "GENERAL", None, None, True, "AUTO_REPLY_CANDIDATE"),
        ("COMPLAINT", "ENTRY_CORRECTION", "MeaningFault", None, True, "ACKNOWLEDGE_THEN_REVIEW"),  # incorrect entry, reported as a complaint
        ("SUGGESTION", "ENTRY_CORRECTION", "MeaningFault", None, True, "ACKNOWLEDGE_THEN_REVIEW"),  # same topic, reported neutrally as a suggestion
        ("SUGGESTION", "WORD_SUBMISSION", None, None, True, "ACKNOWLEDGE_THEN_REVIEW"),  # proposing a new word
        ("COMPLAINT", "TECHNICAL_ISSUE", "TechFault", None, True, "ACKNOWLEDGE_THEN_REVIEW"),  # broken website feature
        ("INQUIRY", "WORD_SUBMISSION", None, None, True, "ACKNOWLEDGE_THEN_REVIEW"),  # asking how to submit a word
        # topic-based routing overrides (apply regardless of intent)
        ("COMPLAINT", "WORD_OF_YEAR_NOMINATION", None, None, True, "REVIEW_ONLY"),
        ("THANKS_PRAISE", "OFF_TOPIC_PERSONAL", None, None, True, "EXCLUDE"),  # would be AUTO_REPLY_CANDIDATE by intent alone
    ],
)
def test_intent_routing_matches_migration_plan(
    primary_intent, topic_subtype, subcategory, secondary_intent, meaningful, expected_routing
):
    result = determine_routing(
        primary_intent=primary_intent,
        topic_subtype=topic_subtype,
        subcategory=subcategory,
        secondary_intent=secondary_intent,
        is_content_meaningful=meaningful,
    )
    assert result.routing_decision == expected_routing


def test_secondary_intent_does_not_change_routing_decision():
    # Unlike the old flat taxonomy's THANKS_PRAISE + secondary_category
    # special case, routing is driven by primary_intent alone now —
    # a genuine second intent is recorded for context/audit but does not
    # redirect routing to follow the secondary instead.
    result = determine_routing(
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        subcategory=None,
        secondary_intent="COMPLAINT",
        is_content_meaningful=True,
    )
    assert result.routing_decision == "AUTO_REPLY_CANDIDATE"


def test_end_to_end_classify_row_for_each_primary_intent_via_fake_client():
    for intent in settings.PRIMARY_INTENTS:
        raw = _valid_json(primary_intent=intent, topic_subtype="GENERAL")
        client = FakeGeminiClient([raw])
        outcome = classify_row(_row(), client, max_retries=settings.GEMINI_MAX_RETRIES)
        assert outcome.success is True
        assert outcome.ai_primary_intent == intent
        assert outcome.ai_routing_decision in settings.ROUTING_DECISIONS
        assert client.call_count == 1  # exactly one call for a clean success


# ---------------------------------------------------------------------------
# 2 & 3. Only non-excluded rows are selected; excluded rows never call Gemini
# ---------------------------------------------------------------------------

def _seed_mixed_rows(tmp_path):
    db_path = tmp_path / "phase4_test.sqlite"
    engine = get_engine(db_path)
    init_db(engine)

    now = dt.datetime(2026, 8, 10, 12, 0, 0)
    rows = []
    for i in range(5):
        rows.append({
            "comment_id": f"excluded-{i}",
            "source_file": "f.xlsx", "source_row_index": i,
            "original_comment": "jhnjkn njnun Phone Number: 111111111111j",
            "original_title": None, "original_type": "أتمنى إضافة أمثلة تطبيقية توضّح استخدام الكلمة في سياقات مختلفة",
            "entry_id": None, "lemma": None, "visitor_name": "test", "visitor_email": "test@test.com",
            "platform": "الرياض", "comment_created_at": now, "is_exact_duplicate_of": None,
            "ingested_at": now, "comment_age_days": 1.0, "age_group": "recent",
            "review_status": "UNPROCESSED", "ready_to_send": False,
        })
    for i in range(3):
        rows.append({
            "comment_id": f"eligible-{i}",
            "source_file": "f.xlsx", "source_row_index": 100 + i,
            "original_comment": f"كلمة رقم {i}", "original_title": None,
            "original_type": "إقتراح مدخل", "entry_id": None, "lemma": None,
            "visitor_name": "شخص", "visitor_email": f"user{i}@x.com", "platform": "الرياض",
            "comment_created_at": now, "is_exact_duplicate_of": None, "ingested_at": now,
            "comment_age_days": 1.0, "age_group": "recent", "review_status": "UNPROCESSED", "ready_to_send": False,
        })
    upsert_review_items(engine, rows)

    prefilter_df = __import__("pandas").DataFrame([
        {"comment_id": f"excluded-{i}", "is_prefiltered_junk": True, "prefilter_reason": "exact_match_known_bot_string"}
        for i in range(5)
    ] + [
        {"comment_id": f"eligible-{i}", "is_prefiltered_junk": False, "prefilter_reason": None}
        for i in range(3)
    ])
    apply_prefilter_results(engine, prefilter_df)
    return engine


def test_load_for_classification_excludes_prefiltered_rows(tmp_path):
    engine = _seed_mixed_rows(tmp_path)
    eligible = load_for_classification(engine)
    assert len(eligible) == 3
    assert set(eligible["comment_id"]) == {"eligible-0", "eligible-1", "eligible-2"}


def test_excluded_rows_never_trigger_gemini_call(tmp_path):
    engine = _seed_mixed_rows(tmp_path)
    eligible = load_for_classification(engine)

    client = FakeGeminiClient([_valid_json() for _ in range(len(eligible))])
    for _, row in eligible.iterrows():
        classify_row(row.to_dict(), client, max_retries=0)

    assert client.call_count == 3  # exactly the eligible rows, never the 5 excluded ones


# ---------------------------------------------------------------------------
# 4. Phone-number redaction before model input
# ---------------------------------------------------------------------------

def test_redact_phone_numbers_removes_long_digit_runs():
    assert redact_phone_numbers("اتصل بي على 0555123456 من فضلك") == "اتصل بي على [phone number removed] من فضلك"


def test_redact_phone_numbers_preserves_short_numbers():
    text = "صدر عام 2024 وكان جيدا"
    assert redact_phone_numbers(text) == text


def test_build_prompt_never_contains_raw_phone_number():
    row = _row(original_comment="تجربة رائعة! رقمي 0555123456 للتواصل")
    prompt = build_prompt(row)
    assert "0555123456" not in prompt
    assert "[phone number removed]" in prompt


def test_build_prompt_never_sends_visitor_identity_fields():
    row = _row()
    prompt = build_prompt(row)
    # visitor_name/visitor_email are not even accepted by build_prompt's
    # row contract — assert none of the known real values from other
    # fixtures could have leaked in via string formatting mistakes.
    assert "visitor_email" not in prompt
    assert "visitor_name" not in prompt


# ---------------------------------------------------------------------------
# 5. Original stored comment is never modified by classification
# ---------------------------------------------------------------------------

def test_original_comment_unchanged_after_classification(tmp_path):
    engine = _seed_mixed_rows(tmp_path)
    eligible = load_for_classification(engine)
    before = dict(zip(eligible["comment_id"], eligible["original_comment"]))

    results = []
    for _, row in eligible.iterrows():
        client = FakeGeminiClient([_valid_json()])
        outcome = classify_row(row.to_dict(), client, max_retries=0)
        results.append({"comment_id": row["comment_id"], **outcome.__dict__})

    apply_classification_results(
        engine, results, model_name="fake-model", model_version=None,
        prompt_version="v1", pipeline_version="0.1.0", inference_run_id="run-1",
    )

    from sqlalchemy.orm import Session
    from rcs.review.models import ReviewItem
    with Session(engine) as session:
        for comment_id, original in before.items():
            item = session.get(ReviewItem, comment_id)
            assert item.original_comment == original


# ---------------------------------------------------------------------------
# 6. Valid structured output
# ---------------------------------------------------------------------------

def test_valid_structured_output_parses_and_routes():
    client = FakeGeminiClient([_valid_json(primary_intent="SUGGESTION", topic_subtype="DICTIONARY_SUGGESTION")])
    outcome = classify_row(_row(), client, max_retries=0)
    assert outcome.success is True
    assert outcome.ai_primary_intent == "SUGGESTION"
    assert outcome.ai_topic_subtype == "DICTIONARY_SUGGESTION"
    assert outcome.ai_routing_decision == "ACKNOWLEDGE_THEN_REVIEW"
    assert outcome.error is None


# ---------------------------------------------------------------------------
# 7. Malformed / invalid output fails safely
# ---------------------------------------------------------------------------

def test_invalid_json_fails_safely():
    client = FakeGeminiClient(["this is not json at all"])
    outcome = classify_row(_row(), client, max_retries=0)
    assert outcome.success is False
    assert outcome.ai_primary_intent is None  # never invents an intent
    assert outcome.ai_routing_decision == "REVIEW_ONLY"
    assert "classification_failed" in outcome.ai_classification_reason


def test_unknown_intent_value_fails_safely():
    import json
    bad = json.dumps({
        "primary_intent": "NOT_A_REAL_INTENT",
        "topic_subtype": "GENERAL",
        "is_content_meaningful": True, "confidence": 0.9,
        "classification_reason": "x",
    })
    client = FakeGeminiClient([bad])
    outcome = classify_row(_row(), client, max_retries=0)
    assert outcome.success is False
    assert outcome.ai_primary_intent is None
    assert outcome.ai_routing_decision == "REVIEW_ONLY"


def test_missing_required_field_fails_safely():
    import json
    bad = json.dumps({"primary_intent": "THANKS_PRAISE"})  # missing required fields
    client = FakeGeminiClient([bad])
    outcome = classify_row(_row(), client, max_retries=0)
    assert outcome.success is False
    assert outcome.ai_primary_intent is None


# ---------------------------------------------------------------------------
# 8. API / network failure fails safely
# ---------------------------------------------------------------------------

def test_network_failure_fails_safely():
    client = FakeGeminiClient([ConnectionError("simulated network failure")])
    outcome = classify_row(_row(), client, max_retries=0)
    assert outcome.success is False
    assert outcome.ai_primary_intent is None
    assert outcome.ai_routing_decision == "REVIEW_ONLY"
    assert "api_error" in outcome.ai_classification_reason


# ---------------------------------------------------------------------------
# 9. Retry behavior
# ---------------------------------------------------------------------------

def test_retries_then_succeeds():
    client = FakeGeminiClient([
        ConnectionError("transient failure 1"),
        ConnectionError("transient failure 2"),
        _valid_json(),
    ])
    outcome = classify_row(_row(), client, max_retries=2)
    assert outcome.success is True
    assert outcome.attempts == 3
    assert client.call_count == 3


def test_retries_exhausted_fails_safely():
    client = FakeGeminiClient([
        ConnectionError("fail 1"),
        ConnectionError("fail 2"),
        ConnectionError("fail 3"),
    ])
    outcome = classify_row(_row(), client, max_retries=2)  # 2 retries -> 3 total attempts
    assert outcome.success is False
    assert outcome.attempts == 3
    assert client.call_count == 3


# ---------------------------------------------------------------------------
# 10. Re-running never overwrites an existing successful AI classification
# ---------------------------------------------------------------------------

def test_rerun_does_not_overwrite_existing_ai_classification(tmp_path):
    engine = _seed_mixed_rows(tmp_path)
    eligible = load_for_classification(engine)
    target = eligible.iloc[0].to_dict()

    first_result = [{
        "comment_id": target["comment_id"],
        **ClassificationOutcome(
            success=True, ai_primary_intent="SUGGESTION", ai_topic_subtype="WORD_SUBMISSION",
            ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="ACKNOWLEDGE_THEN_REVIEW",
            ai_classification_reason="first run", ai_confidence=0.8,
        ).__dict__,
    }]
    apply_classification_results(engine, first_result, "model-a", None, "v1", "0.1.0", "run-1")

    # Simulate a re-run with a DIFFERENT (would-be) result for the same row.
    second_result = [{
        "comment_id": target["comment_id"],
        **ClassificationOutcome(
            success=True, ai_primary_intent="SPAM", ai_topic_subtype="GENERAL",
            ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="EXCLUDE",
            ai_classification_reason="second run - should never apply", ai_confidence=0.99,
        ).__dict__,
    }]
    write_result = apply_classification_results(engine, second_result, "model-b", None, "v2", "0.2.0", "run-2")

    assert write_result == {"classified": 0, "failed": 0, "skipped_already_classified": 1}

    from sqlalchemy.orm import Session
    from rcs.review.models import ReviewItem
    with Session(engine) as session:
        item = session.get(ReviewItem, target["comment_id"])
        assert item.ai_primary_intent == "SUGGESTION"  # unchanged, first run wins
        assert item.ai_classification_reason == "first run"
        assert item.review_status == "CLASSIFIED"


def test_failed_classification_stays_unprocessed_and_is_retry_eligible(tmp_path):
    engine = _seed_mixed_rows(tmp_path)
    eligible = load_for_classification(engine)
    target = eligible.iloc[0].to_dict()

    failed_result = [{
        "comment_id": target["comment_id"],
        **ClassificationOutcome(
            success=False, ai_routing_decision="REVIEW_ONLY",
            ai_classification_reason="api_error: simulated", attempts=3, error="simulated",
        ).__dict__,
    }]
    write_result = apply_classification_results(engine, failed_result, "model-a", None, "v1", "0.1.0", "run-1")
    assert write_result == {"classified": 0, "failed": 1, "skipped_already_classified": 0}

    # Still eligible for classification next run — this is what makes retry-across-runs work.
    still_eligible = load_for_classification(engine)
    assert target["comment_id"] in set(still_eligible["comment_id"])


# ---------------------------------------------------------------------------
# 11. Reviewer fields are never touched by classification
# ---------------------------------------------------------------------------

def test_reviewer_fields_never_overwritten(tmp_path):
    engine = _seed_mixed_rows(tmp_path)
    eligible = load_for_classification(engine)
    target_id = eligible.iloc[0]["comment_id"]

    # Simulate a row a (future) reviewer has already acted on: no longer
    # UNPROCESSED, and carries reviewer-owned field values.
    from sqlalchemy.orm import Session
    from rcs.review.models import ReviewItem
    reviewed_at = dt.datetime(2026, 8, 11, 9, 0, 0)
    with Session(engine) as session:
        item = session.get(ReviewItem, target_id)
        item.review_status = "APPROVED"
        item.reviewer_action = "approve"
        item.reviewer_edit = None
        item.final_reply = "الرد النهائي المعتمد من المراجع"
        item.ready_to_send = True
        item.reviewed_at = reviewed_at
        item.reviewer_id = "reviewer_1"
        session.commit()

    attempted_result = [{
        "comment_id": target_id,
        **ClassificationOutcome(
            success=True, ai_primary_intent="SUGGESTION", ai_topic_subtype="WORD_SUBMISSION",
            ai_subcategory=None, ai_secondary_intent=None, ai_routing_decision="ACKNOWLEDGE_THEN_REVIEW",
            ai_classification_reason="should never apply", ai_confidence=0.5,
        ).__dict__,
    }]
    write_result = apply_classification_results(engine, attempted_result, "model-a", None, "v1", "0.1.0", "run-1")
    assert write_result == {"classified": 0, "failed": 0, "skipped_already_classified": 1}

    with Session(engine) as session:
        item = session.get(ReviewItem, target_id)
        assert item.review_status == "APPROVED"
        assert item.reviewer_action == "approve"
        assert item.final_reply == "الرد النهائي المعتمد من المراجع"
        assert item.ready_to_send is True
        assert item.reviewed_at == reviewed_at
        assert item.reviewer_id == "reviewer_1"
        assert item.ai_primary_intent is None  # classification never ran on this row


# ---------------------------------------------------------------------------
# Defensive routing-engine unit tests (MIGRATION_PLAN.md Section 4)
# ---------------------------------------------------------------------------

def test_duplicate_secondary_intent_is_ambiguous():
    result = determine_routing(
        primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION",
        subcategory=None, secondary_intent="SUGGESTION", is_content_meaningful=True,
    )
    assert result.routing_decision == "REVIEW_ONLY"
    assert result.reason_code == "invalid_duplicate_intent"


def test_missing_subcategory_is_ambiguous():
    result = determine_routing(
        primary_intent="COMPLAINT", topic_subtype="TECHNICAL_ISSUE",
        subcategory=None, secondary_intent=None, is_content_meaningful=True,
    )
    assert result.routing_decision == "REVIEW_ONLY"
    assert result.reason_code == "missing_required_subcategory"


def test_meaningless_content_excluded_even_if_intent_says_otherwise():
    result = determine_routing(
        primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION",
        subcategory=None, secondary_intent=None, is_content_meaningful=False,
    )
    assert result.routing_decision == "EXCLUDE"
    assert result.reason_code == "model_detected_meaningless"


def test_meaningless_flag_does_not_double_flag_already_spam_intent():
    # SPAM already routes EXCLUDE via the intent lookup — the safety
    # override shouldn't fire separately and produce a different reason.
    result = determine_routing(
        primary_intent="SPAM", topic_subtype="GENERAL",
        subcategory=None, secondary_intent=None, is_content_meaningful=False,
    )
    assert result.routing_decision == "EXCLUDE"
    assert result.reason_code == "ai_intent_spam"


def test_meaningless_flag_does_not_double_flag_already_off_topic():
    # OFF_TOPIC_PERSONAL already forces EXCLUDE via the topic override —
    # same non-double-flagging guarantee, on the topic-override path.
    result = determine_routing(
        primary_intent="COMPLAINT", topic_subtype="OFF_TOPIC_PERSONAL",
        subcategory=None, secondary_intent=None, is_content_meaningful=False,
    )
    assert result.routing_decision == "EXCLUDE"
    assert result.reason_code == "ai_topic_off_topic_personal"


def test_unsupported_output_defensive_fallback():
    result = determine_routing(
        primary_intent="SOMETHING_NOT_IN_THE_ENUM", topic_subtype="GENERAL",
        subcategory=None, secondary_intent=None, is_content_meaningful=True,
    )
    assert result.routing_decision == "REVIEW_ONLY"
    assert result.reason_code == "unsupported_output"


# ---------------------------------------------------------------------------
# 12. Responsibility boundary: classification schema has no reply-draft
#     fields, and a stray one in the model's response is rejected outright
#     rather than silently accepted (replaces the old
#     test_exclude_reply_drafts_forced_null_even_if_model_provided_them,
#     which tested a null-out step that no longer exists — the schema
#     doesn't accept those fields at all anymore).
# ---------------------------------------------------------------------------

def test_schema_has_no_reply_draft_fields():
    fields = GeminiClassificationResponse.model_fields
    assert "immediate_reply_draft" not in fields
    assert "suggested_final_reply" not in fields


def test_stray_reply_draft_field_rejected_by_validation():
    import json
    payload = json.dumps({
        "primary_intent": "SPAM",
        "topic_subtype": "GENERAL",
        "is_content_meaningful": True, "confidence": 0.9,
        "classification_reason": "spam",
        "immediate_reply_draft": "this should never be accepted",
    })
    client = FakeGeminiClient([payload])
    outcome = classify_row(_row(), client, max_retries=0)
    # extra="forbid" rejects the stray field at validation, so this fails
    # safely exactly like any other malformed response — no invented intent.
    assert outcome.success is False
    assert outcome.ai_primary_intent is None
