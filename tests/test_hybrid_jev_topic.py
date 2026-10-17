"""Deterministic tests for the experimental Jev Topic composition.

All provider responses and failures are injected. This module performs no
network calls, uses synthetic comments only, and never loads frozen models.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from config import settings
from rcs.classification.external_redaction import ExternalRedactionError, redact_external_text
from rcs.classification.jev_topic import JevTopicAdapter, JevTopicError
from rcs.classification import runtime
from rcs.response_intelligence import inspector, playground
from rcs.review import excel_review_flow
from rcs.review.demo_seed import load_dotenv_if_present


def _response(label: str, *, confidence: float = 0.91) -> dict:
    labels = list(settings.TOPIC_SUBTYPES)
    rest = (1.0 - confidence) / (len(labels) - 1)
    probabilities = {item: (confidence if item == label else rest) for item in labels}
    return {
        "model": settings.JEV_TOPIC_EXPECTED_SNAPSHOT,
        "provider": settings.JEV_TOPIC_EXPECTED_PROVIDER,
        "answers": {
            "topic": {
                "type": "choice",
                "choice": label,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
    }


class CapturingTransport:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body, timeout))
        return self.response


@pytest.mark.parametrize("label", settings.TOPIC_SUBTYPES)
def test_all_seven_topic_labels_preserve_the_frozen_taxonomy(label):
    transport = CapturingTransport(_response(label))
    adapter = JevTopicAdapter(api_key="test-only-key", transport=transport)
    result = adapter.predict(["synthetic-1"], ["اقتراح اصطناعي لاختبار التصنيف فقط"])
    assert result == {"synthetic-1": (label, 0.91)}
    assert len(transport.calls) == 1


def test_outbound_request_contains_only_redacted_text_and_topic_question():
    transport = CapturingTransport(_response("GENERAL"))
    adapter = JevTopicAdapter(api_key="test-only-key", transport=transport)
    synthetic = (
        "اسمي محمد العتيبي، بريدي user@example.com ورقمي 0501234567 "
        "وحسابي @visitor_name وموقعي https://example.com/profile/visitor"
    )
    adapter.predict(["internal-comment-id"], [synthetic])
    _, headers, body, _ = transport.calls[0]
    request = json.loads(body.decode("utf-8"))
    sent = request["state"]["visitor_comment_text"]

    assert set(request) == {"model", "state", "questions"}
    assert set(request["state"]) == {"visitor_comment_text"}
    assert set(request["questions"]) == {"topic"}
    assert set(request["questions"]["topic"]["criteria"]) == set(settings.TOPIC_SUBTYPES)
    assert "internal-comment-id" not in body.decode("utf-8")
    assert "gold" not in body.decode("utf-8").lower()
    for private_value in (
        "محمد العتيبي", "user@example.com", "0501234567", "@visitor_name",
        "https://example.com/profile/visitor",
    ):
        assert private_value not in sent
    assert "[PERSON]" in sent and "[EMAIL]" in sent and "[ID]" in sent
    assert "[HANDLE]" in sent and "[URL]" in sent
    assert headers["Authorization"].startswith("Bearer ")


def test_redactor_masks_ids_addresses_and_signatures():
    result = redact_external_text(
        "رقم الهوية 1234567890. عنواني شارع الملك 12، الرياض\nمع التحية، سارة أحمد"
    )
    assert "1234567890" not in result.text
    assert "شارع الملك" not in result.text
    assert "سارة أحمد" not in result.text
    assert "[ID]" in result.text and "[ADDRESS]" in result.text and "[PERSON]" in result.text


def test_missing_credentials_fails_before_transport():
    transport = CapturingTransport(_response("GENERAL"))
    adapter = JevTopicAdapter(api_key=None, transport=transport)
    with pytest.raises(JevTopicError, match="OPENROUTER_API_KEY"):
        adapter.predict(["synthetic-1"], ["تعليق اصطناعي"])
    assert transport.calls == []


def test_redaction_failure_fails_before_transport():
    transport = CapturingTransport(_response("GENERAL"))

    def broken_redactor(_text):
        raise ExternalRedactionError("synthetic failure")

    adapter = JevTopicAdapter(api_key="test-only-key", transport=transport, redactor=broken_redactor)
    with pytest.raises(JevTopicError, match="redaction failed closed"):
        adapter.predict(["synthetic-1"], ["تعليق اصطناعي"])
    assert transport.calls == []


def test_provider_failure_has_no_retry_and_no_fallback():
    calls = 0

    def failing_transport(*_args):
        nonlocal calls
        calls += 1
        raise JevTopicError("synthetic provider failure")

    adapter = JevTopicAdapter(api_key="test-only-key", transport=failing_transport)
    with pytest.raises(JevTopicError, match="synthetic provider failure"):
        adapter.predict(["synthetic-1"], ["تعليق اصطناعي"])
    assert calls == 1


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda raw: raw.update(provider="Other"), "provider validation"),
        (lambda raw: raw.update(model="vendor/jev-latest"), "snapshot validation"),
        (lambda raw: raw.update(answers={}), "exactly one Topic"),
        (lambda raw: raw["answers"]["topic"].update(choice="UNKNOWN"), "outside the frozen taxonomy"),
        (lambda raw: raw["answers"]["topic"].update(confidence=2), "confidence is invalid"),
        (lambda raw: raw["answers"]["topic"].update(probabilities={"GENERAL": 1.0}), "probabilities do not match"),
    ],
)
def test_malformed_provider_responses_fail_closed(mutate, match):
    raw = _response("GENERAL")
    mutate(raw)
    adapter = JevTopicAdapter(api_key="test-only-key", transport=CapturingTransport(raw))
    with pytest.raises(JevTopicError, match=match):
        adapter.predict(["synthetic-1"], ["تعليق اصطناعي"])


def test_legacy_mode_is_default_and_uses_all_existing_frozen_classifiers(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "legacy_rcs")
    monkeypatch.setattr(runtime.frozen_inference, "predict_intent", lambda texts: [("INQUIRY", 0.8)])
    monkeypatch.setattr(runtime.frozen_inference, "predict_topic", lambda ids, texts: {ids[0]: ("GENERAL", 0.7)})
    monkeypatch.setattr(runtime.frozen_inference, "predict_sentiment", lambda ids, texts: {ids[0]: ("NEUTRAL", 0.9)})

    assert runtime.predict_intent(["x"]) == [("INQUIRY", 0.8)]
    assert runtime.predict_topic(["c1"], ["x"]) == {"c1": ("GENERAL", 0.7)}
    assert runtime.predict_sentiment(["c1"], ["x"]) == {"c1": ("NEUTRAL", 0.9)}
    assert runtime.provenance() == {
        "reason": "frozen_model_inference",
        "source": "FROZEN_TFIDF_LOGREG+MARBERTV2+CAMELBERT",
        "model_name": "tfidf_logistic_intent+marbertv2_topic+camelbert_sentiment",
        "prompt_version": "frozen-2026-08-25",
    }


def test_hybrid_mode_changes_only_topic(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "hybrid_jev_topic")
    monkeypatch.setattr(runtime.frozen_inference, "predict_intent", lambda texts: [("SUGGESTION", 0.81)])
    monkeypatch.setattr(runtime.frozen_inference, "predict_sentiment", lambda ids, texts: {ids[0]: ("POSITIVE", 0.82)})
    monkeypatch.setattr(
        runtime.frozen_inference,
        "predict_topic",
        lambda *_args: pytest.fail("legacy Topic must not be a hybrid fallback"),
    )
    adapter = SimpleNamespace(predict=lambda ids, texts: {ids[0]: ("DICTIONARY_SUGGESTION", 0.83)})

    assert runtime.predict_intent(["x"]) == [("SUGGESTION", 0.81)]
    assert runtime.predict_topic(["c1"], ["x"], adapter=adapter) == {
        "c1": ("DICTIONARY_SUGGESTION", 0.83)
    }
    assert runtime.predict_sentiment(["c1"], ["x"]) == {"c1": ("POSITIVE", 0.82)}
    assert runtime.provenance()["source"] == "FROZEN_TFIDF_LOGREG+JEV_TOPIC+CAMELBERT"


def test_inspector_and_playground_use_the_configured_runtime(monkeypatch):
    calls = {"intent": 0, "topic": 0, "sentiment": 0}

    def intent(texts):
        calls["intent"] += 1
        return [("INQUIRY", 0.81) for _ in texts]

    def topic(ids, texts):
        calls["topic"] += 1
        return {cid: ("TECHNICAL_ISSUE", 0.82) for cid in ids}

    def sentiment(ids, texts):
        calls["sentiment"] += 1
        return {cid: ("NEGATIVE", 0.83) for cid in ids}

    monkeypatch.setattr(runtime, "predict_intent", intent)
    monkeypatch.setattr(runtime, "predict_topic", topic)
    monkeypatch.setattr(runtime, "predict_sentiment", sentiment)

    inspected = inspector.inspect_classification("تعليق اصطناعي لاختبار نقطة الدخول")
    assert (
        inspected.primary_intent,
        inspected.topic_subtype,
        inspected.sentiment,
    ) == ("INQUIRY", "TECHNICAL_ISSUE", "NEGATIVE")

    suite = playground.default_classifiers()
    assert suite.intent is intent
    assert suite.topic is topic
    assert suite.sentiment is sentiment
    assert calls == {"intent": 1, "topic": 1, "sentiment": 1}


def test_invalid_mode_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "unexpected")
    with pytest.raises(runtime.ClassificationModeError, match="unsupported classification mode"):
        runtime.predict_topic(["c1"], ["تعليق اصطناعي"])


def test_existing_dotenv_loader_synchronizes_hybrid_settings(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("RCS_IGNORE_DOTENV", raising=False)       # this test exercises the loader itself on a throw-away .env with a dummy key
    (tmp_path / ".env").write_text(
        "RCS_CLASSIFICATION_MODE=hybrid_jev_topic\nOPENROUTER_API_KEY=test-only-key\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "legacy_rcs")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    monkeypatch.delenv("RCS_CLASSIFICATION_MODE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    try:
        load_dotenv_if_present()

        assert settings.CLASSIFICATION_MODE == "hybrid_jev_topic"
        assert settings.OPENROUTER_API_KEY == "test-only-key"
    finally:
        # load_dotenv_if_present() writes directly to os.environ.  Because
        # both keys were initially absent, monkeypatch.delenv(...,
        # raising=False) had no original mutation to undo.  Remove the test
        # values explicitly so later tests cannot inherit hybrid mode or a
        # synthetic credential.
        os.environ.pop("RCS_CLASSIFICATION_MODE", None)
        os.environ.pop("OPENROUTER_API_KEY", None)


def test_excel_pipeline_does_not_persist_or_route_after_topic_failure(monkeypatch):
    class Repo:
        apply_calls = 0

        def load_for_classification(self):
            return pd.DataFrame([{"comment_id": "c1", "original_comment": "تعليق اصطناعي"}])

        def apply_classification_batch(self, *_args, **_kwargs):
            self.apply_calls += 1
            return {"classified": 1, "failed": 0, "skipped_already_classified": 0}

    repo = Repo()
    monkeypatch.setattr(excel_review_flow, "resolve_bundle", lambda _engine: SimpleNamespace(review_items=repo))
    monkeypatch.setattr(excel_review_flow, "predict_intent", lambda texts: [("INQUIRY", 0.8)])
    monkeypatch.setattr(excel_review_flow, "predict_topic", lambda ids, texts: (_ for _ in ()).throw(JevTopicError("provider failed")))
    monkeypatch.setattr(excel_review_flow, "predict_sentiment", lambda *_args: pytest.fail("must stop immediately"))

    with pytest.raises(JevTopicError, match="provider failed"):
        excel_review_flow.run_frozen_classification(object(), ["c1"])
    assert repo.apply_calls == 0


def test_hybrid_excel_success_preserves_persistence_and_routing_contract(monkeypatch):
    class Repo:
        def __init__(self):
            self.results = None
            self.metadata = None

        def load_for_classification(self):
            return pd.DataFrame([{"comment_id": "c1", "original_comment": "اقتراح اصطناعي"}])

        def apply_classification_batch(self, results, **metadata):
            self.results = results
            self.metadata = metadata
            return {"classified": 1, "failed": 0, "skipped_already_classified": 0}

    repo = Repo()
    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "hybrid_jev_topic")
    monkeypatch.setattr(excel_review_flow, "resolve_bundle", lambda _engine: SimpleNamespace(review_items=repo))
    monkeypatch.setattr(excel_review_flow, "predict_intent", lambda texts: [("SUGGESTION", 0.81)])
    monkeypatch.setattr(
        excel_review_flow,
        "predict_topic",
        lambda ids, texts: {ids[0]: ("DICTIONARY_SUGGESTION", 0.83)},
    )
    monkeypatch.setattr(
        excel_review_flow,
        "predict_sentiment",
        lambda ids, texts: {ids[0]: ("NEUTRAL", 0.82)},
    )

    result = excel_review_flow.run_frozen_classification(object(), ["c1"])
    assert result == {
        "sentiments": {"c1": "NEUTRAL"},
        "classified": 1,
        "failed": 0,
        "skipped_already_classified": 0,
    }
    assert set(repo.results[0]) == {
        "comment_id", "success", "ai_primary_intent", "ai_topic_subtype",
        "ai_subcategory", "ai_secondary_intent", "ai_routing_decision",
        "ai_classification_reason", "ai_confidence", "ai_classification_source",
        "ai_needs_review", "routing_reason_code", "ai_sentiment",
    }
    assert repo.results[0]["ai_sentiment"] == "NEUTRAL"  # persisted at classification, before generation
    assert repo.results[0]["ai_primary_intent"] == "SUGGESTION"
    assert repo.results[0]["ai_topic_subtype"] == "DICTIONARY_SUGGESTION"
    assert repo.results[0]["ai_classification_source"] == "FROZEN_TFIDF_LOGREG+JEV_TOPIC+CAMELBERT"
    assert repo.results[0]["ai_routing_decision"] == "ACKNOWLEDGE_THEN_REVIEW"
    assert repo.metadata["model_name"] == "tfidf_logistic_intent+vendor_jev_1_13_topic+camelbert_sentiment"


def test_hybrid_classification_reaches_agent_and_human_review_without_auto_send(tmp_path, monkeypatch):
    """Synthetic offline E2E: runtime -> routing -> Agent -> review workflow."""
    from rcs.review.identity import ReviewerIdentity
    from rcs.review.repository import get_engine, init_db, upsert_review_items
    from rcs.review.service import ReviewWorkflowService

    class SyntheticJevAdapter:
        def __init__(self, *, api_key):
            assert api_key == "test-only-key"

        def predict(self, comment_ids, texts):
            assert texts == ["أقترح إضافة أمثلة أوضح في المدخل"]
            return {comment_ids[0]: ("DICTIONARY_SUGGESTION", 0.93)}

    class SyntheticGeminiClient:
        def __init__(self):
            self.call_count = 0

        def generate(self, prompt, response_schema):
            self.call_count += 1
            return json.dumps(
                    {"suggested_response": "شكرًا لاقتراحكم بشأن أمثلة المدخل."},
                ensure_ascii=False,
            )

    monkeypatch.setattr(settings, "CLASSIFICATION_MODE", "hybrid_jev_topic")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(runtime, "JevTopicAdapter", SyntheticJevAdapter)
    monkeypatch.setattr(
        runtime.frozen_inference,
        "predict_intent",
        lambda texts: [("SUGGESTION", 0.92) for _ in texts],
    )
    monkeypatch.setattr(
        runtime.frozen_inference,
        "predict_sentiment",
        lambda ids, texts: {cid: ("NEUTRAL", 0.91) for cid in ids},
    )
    monkeypatch.setattr(
        runtime.frozen_inference,
        "predict_topic",
        lambda *_args: pytest.fail("hybrid mode must not use the legacy Topic classifier"),
    )

    engine = get_engine(tmp_path / "hybrid-e2e.sqlite")
    init_db(engine)
    comment_id = "synthetic-hybrid-e2e"
    upsert_review_items(
        engine,
        [
            {
                "comment_id": comment_id,
                "source_file": "synthetic.xlsx",
                "source_row_index": 0,
                "original_comment": "أقترح إضافة أمثلة أوضح في المدخل",
                "review_status": "UNPROCESSED",
                "is_prefiltered_junk": False,
                "ready_to_send": False,
                "row_version": 1,
            }
        ],
    )

    classified = excel_review_flow.run_frozen_classification(engine, [comment_id])
    assert classified["classified"] == 1
    assert classified["sentiments"] == {comment_id: "NEUTRAL"}

    service = ReviewWorkflowService(engine)
    client = SyntheticGeminiClient()
    generated = excel_review_flow.apply_live_generation(
        service,
        [comment_id],
        client,
        classified["sentiments"],
    )
    assert generated == {
        "generated_drafts": 1,
        "no_response_recommended": 0,
        "review_only": 0,
        "attempted": 1,
    }
    assert client.call_count == 1

    draft = service.get_review_item(comment_id)
    assert (draft.primary_intent, draft.topic_subtype, draft.sentiment) == (
        "SUGGESTION",
        "DICTIONARY_SUGGESTION",
        "NEUTRAL",
    )
    assert draft.review_status == "DRAFT_READY"
    assert draft.final_response is None
    assert draft.outbound_status is None
    assert service.list_review_queue().items[0].comment_id == comment_id

    reviewer = ReviewerIdentity("synthetic-reviewer", "مراجع اختباري")
    approved = service.send_as_is(comment_id, reviewer, expected_version=draft.row_version)
    assert approved.review_status == "APPROVED"
    assert approved.reviewer_action == "SEND_AS_IS"
    assert approved.outbound_status == "PENDING"
