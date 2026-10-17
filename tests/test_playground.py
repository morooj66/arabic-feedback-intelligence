"""
Interactive AI Evaluation Playground tests. No network, no Gemini, no frozen
model loading anywhere in this file: classifiers, retrievers and Gemini
clients are all injected fakes.
"""

from __future__ import annotations

import inspect as pyinspect
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from sqlalchemy import inspect as sa_inspect, text as sa_text

from config import settings
from rcs.response_generation.eligibility import determine_response_eligibility, EligibilityInput
from rcs.response_intelligence import playground, playground_eval, playground_store
from rcs.response_intelligence.audit import FORBIDDEN_FIELD_SUBSTRINGS
from rcs.response_intelligence.retrieval.contracts import (
    SUCCESS_EMPTY,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeRetrievalResult,
)
from rcs.review import playground_api
from rcs.review.auth import create_user
from rcs.review.demo_seed import seed_demo_db
from rcs.review.http_api import serve_in_thread
from rcs.review.repository import get_engine
from rcs.routing.engine import determine_routing

GOOD = "التطبيق ممتاز"


# ---------------------------------------------------------------- fakes
class FakeSuite(playground.ClassifierSuite):
    """Deterministic classifiers. `table` maps text -> (intent, topic, sentiment, conf)."""

    def __init__(self, table=None, default=("SUGGESTION", "WORD_SUBMISSION", "NEUTRAL", 0.8)):
        self.table, self.default = table or {}, default
        self.calls = {"intent": 0, "topic": 0, "sentiment": 0}
        self.batch_sizes = []
        super().__init__(self._intent, self._topic, self._sentiment)

    def _row(self, text):
        return self.table.get(text, self.default)

    def _intent(self, texts):
        self.calls["intent"] += 1
        self.batch_sizes.append(len(texts))
        return [(self._row(t)[0], self._row(t)[3]) for t in texts]

    def _topic(self, ids, texts):
        self.calls["topic"] += 1
        return {i: (self._row(t)[1], 0.7) for i, t in zip(ids, texts)}

    def _sentiment(self, ids, texts):
        self.calls["sentiment"] += 1
        return {i: (self._row(t)[2], 0.9) for i, t in zip(ids, texts)}


class _Retr:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return self.result


def fake_retrievers():
    return {
        "official_knowledge_retriever": _Retr(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)),
        "approved_response_retriever": _Retr(ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)),
    }


class FakeLiveClient:
    def __init__(self, text="شكرًا لملاحظتكم القيّمة حول هذا الموضوع.", exc=None):
        self.text, self.exc, self.call_count = text, exc, 0

    def generate(self, prompt, response_schema):
        self.call_count += 1
        if self.exc:
            raise self.exc
        return json.dumps({"suggested_response": self.text}, ensure_ascii=False)


def run(body, suite=None, **kw):
    kw.setdefault("retrievers", fake_retrievers())
    return playground.run_playground(body, classifiers=suite or FakeSuite(), cache=playground.RunCache(), **kw)


@pytest.fixture(autouse=True)
def _no_live_client(monkeypatch):
    """Any accidental live-client construction fails the test loudly."""

    def boom():
        raise AssertionError("live Gemini client must not be built in tests")

    monkeypatch.setattr(playground, "build_live_gemini_client", boom)
    monkeypatch.setattr(settings, "GEMINI_RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setattr(settings, "GEMINI_RETRY_BACKOFF_MAX_SECONDS", 0.0)
    playground.RUN_CACHE.clear()


# ================================================= classification output integrity
def test_classification_outputs_use_frozen_taxonomy_and_pass_confidence_through():
    out = run({"variants": [GOOD], "mode": "classification_only"})
    c = out["variants"][0]["classification"]
    assert c["intent"] == {"label": "SUGGESTION", "confidence": 0.8}
    assert c["topic"]["label"] in settings.TOPIC_SUBTYPES and c["sentiment"]["label"] in settings.SENTIMENTS
    assert out["variants"][0]["errors"] == []


def test_variants_are_classified_as_one_batch_in_input_order():
    table = {f"variant text {i}": (settings.PRIMARY_INTENTS[i % 5], "GENERAL", "NEUTRAL", 0.9) for i in range(6)}
    suite = FakeSuite(table)
    out = run({"variants": [f"variant text {i}" for i in range(6)]}, suite)
    assert suite.calls == {"intent": 1, "topic": 1, "sentiment": 1}
    assert suite.batch_sizes == [6]
    assert [v["classification"]["intent"]["label"] for v in out["variants"]] == [settings.PRIMARY_INTENTS[i % 5] for i in range(6)]


def test_out_of_taxonomy_prediction_is_rejected_not_displayed():
    suite = FakeSuite(default=("NOT_A_REAL_INTENT", "GENERAL", "NEUTRAL", 0.9))
    v = run({"variants": [GOOD]}, suite)["variants"][0]
    assert v["classification"]["intent"] is None
    assert any(e["stage"] == "classification.intent" and "outside the frozen taxonomy" in e["message"] for e in v["errors"])
    assert v["routing"] is None  # routing needs both intent and topic


def test_invalid_confidence_is_dropped_to_none():
    v = run({"variants": [GOOD]}, FakeSuite(default=("SUGGESTION", "GENERAL", "NEUTRAL", 1.7)))["variants"][0]
    assert v["classification"]["intent"]["confidence"] is None


def test_taxonomy_comes_from_settings_not_a_copy():
    t = playground.taxonomy()
    assert t["intents"] == list(settings.PRIMARY_INTENTS) and t["topics"] == list(settings.TOPIC_SUBTYPES)
    assert t["sentiments"] == list(settings.SENTIMENTS)


# ================================================= production parity
def test_text_goes_through_production_normalize_text_and_prefilter_rules():
    p = playground.preprocess("  hello \r\n\r\n\r\n\r\nworld  ")
    assert p["normalized_text"] == "hello\n\nworld" and p["changed"]
    assert playground.preprocess("  ")["prefilter_reason"] == "null_content"
    assert playground.preprocess("ab")["prefilter_reason"] == "near_empty_content"
    assert playground.preprocess("Phone Number: 0555555555")["prefilter_reason"] == "phone_number_only_stub"
    assert playground.preprocess(settings.KNOWN_JUNK_STRINGS[0])["prefilter_reason"] == "exact_match_known_bot_string"
    assert playground.preprocess("asdf qwer zxcv")["prefilter_reason"] == "keyboard_mash_sequence"
    assert playground.preprocess("HTTP API SDK")["prefilter_reason"] is None
    assert playground.preprocess("الموقع يعرض Error 500")["prefilter_reason"] is None
    assert playground.preprocess(GOOD)["prefilter_reason"] is None


def test_prefiltered_text_is_not_classified_like_production_unless_forced():
    suite = FakeSuite()
    v = run({"variants": ["ab"]}, suite)["variants"][0]
    assert suite.calls["intent"] == 0 and v["classification"] is None
    assert v["preprocessing"]["production_outcome"].startswith("EXCLUDED_BY_PREFILTER")
    suite2 = FakeSuite()
    v2 = run({"variants": ["ab"], "force_classify_prefiltered": True}, suite2)["variants"][0]
    assert suite2.calls["intent"] == 1 and v2["preprocessing"]["forced_classification"] is True


def test_classifiers_receive_the_normalized_text():
    suite = FakeSuite()
    seen = []
    orig = suite.intent
    suite.intent = lambda texts: (seen.extend(texts), orig(texts))[1]
    run({"variants": ["  التطبيق ممتاز \n\n\n\n"]}, suite)
    assert seen == ["التطبيق ممتاز"]


def test_needs_review_uses_production_low_confidence_threshold(monkeypatch):
    captured = []
    real = playground.run_agent_pipeline
    monkeypatch.setattr(playground, "run_agent_pipeline", lambda req, *a, **k: (captured.append(req), real(req, *a, **k))[1])
    low = settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD - 0.01
    run({"variants": [GOOD], "mode": "full_pipeline"}, FakeSuite(default=("SUGGESTION", "GENERAL", "NEUTRAL", low)))
    run({"variants": [GOOD], "mode": "full_pipeline"}, FakeSuite(default=("SUGGESTION", "GENERAL", "NEUTRAL", 0.9)))
    assert [r.needs_review for r in captured] == [True, False]
    assert captured[0].confidence == low  # intent confidence, exactly like production


def test_agent_request_matches_production_request_shape(monkeypatch):
    captured = {}
    real = playground.run_agent_pipeline

    def spy(req, client, retries, **kw):
        captured.update(req=req, kw=kw)
        return real(req, client, retries, **kw)

    monkeypatch.setattr(playground, "run_agent_pipeline", spy)
    out = run({"variants": [GOOD], "mode": "full_pipeline"})
    req, kw = captured["req"], captured["kw"]
    routing = out["variants"][0]["routing"]["decision"]
    assert req.classification_routing_decision == routing
    assert (req.primary_intent, req.topic_subtype, req.sentiment) == ("SUGGESTION", "WORD_SUBMISSION", "NEUTRAL")
    assert req.comment_id.startswith("playground-")
    assert kw["audit_repository"] is None and kw["capture_trace"] is True  # no persistence of any kind


# ================================================= routing correctness
@pytest.mark.parametrize(
    "intent,topic",
    [("SPAM", "GENERAL"), ("SUGGESTION", "OFF_TOPIC_PERSONAL"), ("THANKS_PRAISE", "GENERAL"), ("COMPLAINT", "TECHNICAL_ISSUE"), ("INQUIRY", "WORD_OF_YEAR_NOMINATION"), ("SUGGESTION", "ENTRY_CORRECTION")],
)
def test_routing_equals_production_determine_routing(intent, topic):
    v = run({"variants": [GOOD]}, FakeSuite(default=(intent, topic, "NEUTRAL", 0.9)))["variants"][0]
    expected = determine_routing(primary_intent=intent, topic_subtype=topic, subcategory=None, secondary_intent=None, is_content_meaningful=True)
    assert (v["routing"]["decision"], v["routing"]["reason_code"]) == (expected.routing_decision, expected.reason_code)
    assert v["routing"]["partial_classification"] is False


def test_spam_prediction_routes_exclude_and_agent_never_calls_generation():
    suite = FakeSuite(default=("SPAM", "GENERAL", "NEUTRAL", 0.9))
    v = run({"variants": [GOOD], "mode": "full_pipeline"}, suite)["variants"][0]
    assert v["routing"]["decision"] == "EXCLUDE"
    a = v["agent_runs"]["predicted"]
    assert a["decision"]["routing_decision"] == "REVIEW_ONLY"      # SPAM alone is uncorroborated: a human decides, no generation
    assert a["decision"]["reason_code"] == "uncertain_exclusion_requires_review"
    assert a["llm_calls"] == 0 and a["response"] is None


def test_eligibility_preview_equals_production_function():
    v = run({"variants": [GOOD]})["variants"][0]
    ref = determine_response_eligibility(
        EligibilityInput(comment_text=GOOD, primary_intent="SUGGESTION", topic_subtype="WORD_SUBMISSION", sentiment="NEUTRAL", confidence=0.8, needs_review=False,
                         classification_routing_decision=v["routing"]["decision"])
    )
    assert v["eligibility_preview"]["rule_id"] == ref.rule_id and v["eligibility_preview"]["routing_decision"] == ref.routing_decision


# ================================================= mock / live separation
def test_default_agent_mode_is_mock_and_never_builds_a_live_client():
    out = run({"variants": [GOOD], "mode": "full_pipeline"})  # autouse fixture makes live construction fail the test
    assert out["agent_mode"] == "mock" and "MOCK" in out["mode_banner"]
    a = out["variants"][0]["agent_runs"]["predicted"]
    assert a["agent_mode"] == "mock" and "NOT Gemini" in a["mode_banner"]
    assert a["response"] == playground.MOCK_RESPONSE_TEXT
    assert a["tool_calls"][-1]["provider"] == "mock"


def test_mock_response_text_passes_production_validation_gates():
    from rcs.response_generation.validate import run_validation_gates

    assert run_validation_gates(playground.MOCK_RESPONSE_TEXT).passed


def test_live_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    with pytest.raises(playground.PlaygroundError, match="live_confirmed"):
        run({"variants": [GOOD], "mode": "full_pipeline", "agent_mode": "live"}, live_client_factory=lambda: pytest.fail("must not build"))


def test_live_requires_configured_api_key(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    with pytest.raises(playground.PlaygroundError, match="no API key"):
        run({"variants": [GOOD], "mode": "full_pipeline", "agent_mode": "live", "live_confirmed": True}, live_client_factory=lambda: pytest.fail("must not build"))


def test_live_mode_uses_injected_client_and_is_labelled_live(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    client = FakeLiveClient()
    out = run({"variants": [GOOD], "mode": "full_pipeline", "agent_mode": "live", "live_confirmed": True}, live_client_factory=lambda: client)
    a = out["variants"][0]["agent_runs"]["predicted"]
    assert out["agent_mode"] == "live" and a["agent_mode"] == "live" and "LIVE" in a["mode_banner"] and "MOCK" not in a["mode_banner"]
    assert client.call_count == 1 and a["llm_calls"] == 1 and a["response"] == client.text and a["tool_calls"][-1]["provider"] == "gemini"


def test_classification_only_never_creates_any_agent_client(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    out = run({"variants": [GOOD], "mode": "classification_only", "agent_mode": "live", "live_confirmed": True}, live_client_factory=lambda: pytest.fail("must not build"))
    assert out["variants"][0]["agent_runs"] == {} and out["agent_mode"] is None


def test_live_agent_failure_surfaces_fallback_and_never_leaks_the_key(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "sk-secret-123456")
    monkeypatch.setattr(settings, "GEMINI_MAX_RETRIES", 0)
    client = FakeLiveClient(exc=RuntimeError("401 bad credentials key=sk-secret-123456"))
    out = run({"variants": [GOOD], "mode": "full_pipeline", "agent_mode": "live", "live_confirmed": True}, live_client_factory=lambda: client)
    a = out["variants"][0]["agent_runs"]["predicted"]
    assert a["fallback"]["kind"] == "failed_review_only" and a["decision"]["generation_status"] == "FAILED_REVIEW_ONLY"
    assert "sk-secret-123456" not in json.dumps(out, ensure_ascii=False)


def test_sanitize_error_text_redacts_key_shapes(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "abc-key-999")
    s = playground.sanitize_error_text("boom abc-key-999 and AIzaSyA1234567890123456789012 and api_key=hello and Bearer tok123")
    assert "abc-key-999" not in s and "AIza" not in s and "hello" not in s and "tok123" not in s


# ================================================= fixed-label agent testing
def test_fixed_label_mode_bypasses_classifiers_and_uses_same_routing():
    suite = FakeSuite()
    body = {"variants": [GOOD], "mode": "full_pipeline", "label_source": "verified",
            "fixed_labels": [{"verified": True, "labels": {"intent": "THANKS_PRAISE", "topic": "GENERAL", "sentiment": "POSITIVE"}}]}
    v = run(body, suite)["variants"][0]
    assert suite.calls == {"intent": 0, "topic": 0, "sentiment": 0}
    b = v["agent_runs"]["verified"]
    ref = determine_routing(primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", subcategory=None, secondary_intent=None, is_content_meaningful=True)
    assert b["classification_routing"]["decision"] == ref.routing_decision and b["label_source"] == "verified"
    assert b["labels_used"]["intent"] == "THANKS_PRAISE" and "predicted" not in v["agent_runs"]


def test_fixed_vs_predicted_comparison_isolates_classifier_error():
    suite = FakeSuite(default=("SPAM", "GENERAL", "NEUTRAL", 0.9))  # classifier is wrong
    body = {"variants": [GOOD], "mode": "full_pipeline", "label_source": "both",
            "fixed_labels": [{"verified": True, "labels": {"intent": "THANKS_PRAISE", "topic": "GENERAL", "sentiment": "POSITIVE"}}]}
    v = run(body, suite)["variants"][0]
    assert v["agent_runs"]["predicted"]["decision"]["routing_decision"] == "REVIEW_ONLY"
    assert v["agent_runs"]["verified"]["decision"]["routing_decision"] == "GENERATE_DRAFT"
    assert v["fixed_label_comparison"]["decision_differs"] is True


def test_unverified_labels_are_never_used_as_fixed_labels():
    body = {"variants": [GOOD], "mode": "full_pipeline", "label_source": "verified",
            "fixed_labels": [{"verified": False, "labels": {"intent": "THANKS_PRAISE", "topic": "GENERAL", "sentiment": "POSITIVE"}}]}
    with pytest.raises(playground.PlaygroundError, match="verified labels"):
        run(body)


def test_fixed_labels_are_validated_against_the_taxonomy():
    body = {"variants": [GOOD], "mode": "full_pipeline", "label_source": "verified",
            "fixed_labels": [{"verified": True, "labels": {"intent": "DUA_BLESSING", "topic": "GENERAL", "sentiment": "POSITIVE"}}]}
    with pytest.raises(playground.PlaygroundError, match="invalid intent"):
        run(body)


def test_fixed_label_testing_requires_full_pipeline():
    with pytest.raises(playground.PlaygroundError, match="full_pipeline"):
        run({"variants": [GOOD], "mode": "classification_only", "label_source": "verified"})


# ================================================= partial-classification regression (Full Pipeline must not run the Agent)
def _suite_failing(which: str) -> FakeSuite:
    suite = FakeSuite()

    def broken(*a, **k):
        raise OSError(f"{which} checkpoint could not be loaded")

    setattr(suite, which, broken)
    return suite


@pytest.mark.parametrize("which,stage", [("intent", "classification.intent"), ("topic", "classification.topic"), ("sentiment", "classification.sentiment")])
def test_full_pipeline_never_runs_the_agent_when_any_classifier_fails(which, stage, monkeypatch):
    agent_calls = []
    real = playground.run_agent_pipeline
    monkeypatch.setattr(playground, "run_agent_pipeline", lambda *a, **k: (agent_calls.append(1), real(*a, **k))[1])
    out = run({"variants": [GOOD], "mode": "full_pipeline"}, _suite_failing(which))
    v = out["variants"][0]
    assert agent_calls == [] and v["agent_runs"] == {}
    assert v["pipeline_status"]["classification_complete"] is False and v["pipeline_status"]["production_equivalent"] is False
    assert v["pipeline_status"]["failed_stages"] == [stage]
    skipped = v["agent_skipped"]
    assert skipped["reason"] == "classification_incomplete" and skipped["failed_stages"] == [stage] and skipped["production_equivalent"] is False
    assert stage in skipped["detail"] and "OSError" in skipped["detail"] and "could not be loaded" in skipped["detail"]
    assert any(e["stage"] == "agent" for e in v["errors"]) and any(e["stage"] == stage and e["type"] == "OSError" for e in v["errors"])


def test_successful_classifier_results_are_preserved_when_another_one_fails():
    v = run({"variants": [GOOD], "mode": "full_pipeline"}, _suite_failing("sentiment"))["variants"][0]
    c = v["classification"]
    assert c["intent"]["label"] == "SUGGESTION" and c["topic"]["label"] == "WORD_SUBMISSION" and c["sentiment"] is None
    assert v["routing"]["decision"] and v["routing"]["partial_classification"] is True and v["eligibility_preview"]["partial_classification"] is True


def test_classification_only_keeps_working_with_a_failed_sentiment_model_and_flags_it_partial():
    v = run({"variants": [GOOD], "mode": "classification_only"}, _suite_failing("sentiment"))["variants"][0]
    assert v["classification"]["intent"] and v["classification"]["topic"] and v["classification"]["sentiment"] is None
    assert v["pipeline_status"]["classification_complete"] is False and "agent_skipped" not in v and v["agent_runs"] == {}


def test_complete_classification_is_flagged_production_equivalent():
    v = run({"variants": [GOOD], "mode": "full_pipeline"})["variants"][0]
    assert v["pipeline_status"] == {"classification_complete": True, "failed_stages": [], "production_equivalent": True, "note": None}
    assert v["agent_runs"]["predicted"]["production_equivalent"] is True and "agent_skipped" not in v


def test_fixed_label_run_stays_independent_and_is_never_production_equivalent_when_classifier_fails():
    body = {"variants": [GOOD], "mode": "full_pipeline", "label_source": "both",
            "fixed_labels": [{"verified": True, "labels": {"intent": "THANKS_PRAISE", "topic": "GENERAL", "sentiment": "POSITIVE"}}]}
    v = run(body, _suite_failing("topic"))["variants"][0]
    assert "predicted" not in v["agent_runs"] and v["agent_skipped"]["label_source"] == "predicted"
    b = v["agent_runs"]["verified"]
    assert b["production_equivalent"] is False and "ISOLATION TEST" in b["path_note"] and b["decision"]["routing_decision"] == "GENERATE_DRAFT"
    assert "fixed_label_comparison" not in v  # A/B comparison needs both runs


def test_partial_failure_in_one_variant_does_not_stop_the_others():
    suite = FakeSuite()
    real_topic = suite.topic
    suite.topic = lambda ids, texts: (_ for _ in ()).throw(OSError("boom"))
    out = run({"variants": [GOOD, "ab"], "mode": "full_pipeline"}, suite)  # second variant is prefiltered, never classified
    assert out["variants"][0]["agent_runs"] == {} and out["variants"][1]["preprocessing"]["prefilter_reason"]
    suite2 = FakeSuite()
    out2 = run({"variants": [GOOD, "variant two text"], "mode": "full_pipeline"}, suite2)
    assert all("predicted" in v["agent_runs"] for v in out2["variants"]) and real_topic


def test_expected_response_decision_is_error_not_pass_when_agent_was_skipped():
    v = run({"variants": [GOOD], "mode": "full_pipeline"}, _suite_failing("sentiment"))["variants"][0]
    c = playground_eval.compare({"response_decision": "GENERATE_DRAFT"}, "VERIFIED", v)
    assert c["verdict"] == "ERROR"
    c2 = playground_eval.compare({"intent": "SUGGESTION", "sentiment": "NEUTRAL"}, "VERIFIED", v)
    assert c2["verdict"] == "ERROR" and c2["fields"]["sentiment"]["evaluated"] is False  # never PASS on a failed model


# ================================================= limits & label validation
def test_variant_and_length_limits_enforced():
    with pytest.raises(playground.PlaygroundError, match="at most 10"):
        run({"variants": ["abcdef"] * 11})
    with pytest.raises(playground.PlaygroundError, match="1000"):
        run({"variants": ["أ" * 1001]})
    with pytest.raises(playground.PlaygroundError):
        run({"variants": []})
    assert len(run({"variants": ["abcdef"] * 10})["variants"]) == 10
    run({"variants": ["أ" * 1000]})


@pytest.mark.parametrize("labels", [{"intent": "NOPE"}, {"topic": "OFF_TOPIC"}, {"sentiment": "HAPPY"}])
def test_validate_labels_rejects_values_outside_taxonomy(labels):
    with pytest.raises(playground.PlaygroundError):
        playground.validate_labels(labels)


def test_validate_labels_accepts_every_taxonomy_value():
    for v in settings.PRIMARY_INTENTS:
        assert playground.validate_labels({"intent": v}) == {"intent": v}
    for v in settings.TOPIC_SUBTYPES:
        assert playground.validate_labels({"topic": v}) == {"topic": v}
    for v in settings.SENTIMENTS:
        assert playground.validate_labels({"sentiment": v}) == {"sentiment": v}


def test_expected_validation_covers_routing_vocabularies():
    assert playground_eval.validate_expected({"routing": "EXCLUDE", "response_decision": "GENERATE_DRAFT"})
    with pytest.raises(ValueError):
        playground_eval.validate_expected({"routing": "GENERATE_DRAFT"})  # wrong vocabulary
    with pytest.raises(ValueError):
        playground_eval.validate_expected({"colour": "red"})


# ================================================= error handling
def test_classifier_exception_is_reported_per_stage_and_run_continues():
    suite = FakeSuite()

    def broken(ids, texts):
        raise FileNotFoundError("Frozen MARBERTv2 topic checkpoint not found")

    suite.topic = broken
    out = run({"variants": [GOOD], "mode": "full_pipeline"}, suite)
    v = out["variants"][0]
    assert v["classification"]["intent"] and v["classification"]["topic"] is None
    assert any(e["stage"] == "classification.topic" and e["type"] == "FileNotFoundError" for e in v["errors"])
    assert v["agent_runs"] == {}  # production would fail this row; the playground does not invent a label


def test_agent_exception_is_caught_and_reported_as_fallback(monkeypatch):
    monkeypatch.setattr(playground, "run_agent_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaput")))
    a = run({"variants": [GOOD], "mode": "full_pipeline"})["variants"][0]["agent_runs"]["predicted"]
    assert a["error"]["type"] == "RuntimeError" and a["fallback"]["kind"] == "agent_exception" and a["decision"] is None


def test_agent_tool_outcomes_and_timing_are_reported():
    out = run({"variants": [GOOD], "mode": "full_pipeline"})
    a = out["variants"][0]["agent_runs"]["predicted"]
    tools = {t["tool"]: t for t in a["tool_calls"]}
    assert tools["official_knowledge_retrieval"]["outcome"] == SUCCESS_EMPTY
    assert tools["approved_response_memory_retrieval"]["outcome"] == SUCCESS_EMPTY
    assert a["execution_time_ms"] is not None and out["total_ms"] >= 0 and out["classification_timings_ms"]["total"] >= 0
    assert a["decision"]["routing_decision"] == "GENERATE_DRAFT" and a["response"]


def test_result_payload_exposes_no_forbidden_field_names_or_prompts():
    blob = json.dumps(run({"variants": [GOOD], "mode": "full_pipeline"}), ensure_ascii=False).lower()
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k.lower())
                walk(v)
        elif isinstance(o, list):
            for i in o:
                walk(i)

    walk(json.loads(blob))
    for k in keys:
        assert not any(bad in k for bad in FORBIDDEN_FIELD_SUBSTRINGS), k


# ================================================= expected vs actual / ambiguity
def _variant(intent="SUGGESTION", topic="WORD_SUBMISSION", sentiment="NEUTRAL", routing="ACKNOWLEDGE_THEN_REVIEW", decision=None):
    v = {"classification": {"intent": {"label": intent, "confidence": .8}, "topic": {"label": topic, "confidence": .7}, "sentiment": {"label": sentiment, "confidence": .9}},
         "routing": {"decision": routing, "reason_code": None}, "agent_runs": {}}
    if decision:
        v["agent_runs"]["predicted"] = {"decision": {"routing_decision": decision}}
    return v


def test_pass_requires_verified_status_and_all_set_fields_matching():
    c = playground_eval.compare({"intent": "SUGGESTION", "topic": "WORD_SUBMISSION"}, "VERIFIED", _variant())
    assert c["verdict"] == "PASS" and c["evaluated_fields"] == 2
    assert "sentiment" not in c["fields"]  # unset field is not evaluated, not auto-passed


def test_fail_when_any_verified_field_differs():
    c = playground_eval.compare({"intent": "INQUIRY", "topic": "WORD_SUBMISSION"}, "VERIFIED", _variant())
    assert c["verdict"] == "FAIL" and c["fields"]["intent"]["match"] is False and c["fields"]["topic"]["match"] is True


def test_unverified_labels_are_never_scored():
    c = playground_eval.compare({"intent": "INQUIRY"}, "UNVERIFIED", _variant())
    assert c["verdict"] == "NOT_EVALUATED" and c["fields"] == {}


def test_ambiguous_cases_are_needs_human_review_never_pass_or_fail():
    c = playground_eval.compare({"intent": "INQUIRY"}, "NEEDS_HUMAN_REVIEW", _variant())
    assert c["verdict"] == "NEEDS_HUMAN_REVIEW"


def test_verified_without_any_expected_label_is_not_evaluated():
    assert playground_eval.compare({}, "VERIFIED", _variant())["verdict"] == "NOT_EVALUATED"


def test_missing_actual_never_yields_pass():
    v = _variant()
    v["classification"]["topic"] = None
    c = playground_eval.compare({"intent": "SUGGESTION", "topic": "GENERAL"}, "VERIFIED", v)
    assert c["verdict"] == "ERROR" and c["fields"]["topic"]["evaluated"] is False
    c2 = playground_eval.compare({"response_decision": "GENERATE_DRAFT"}, "VERIFIED", _variant())
    assert c2["verdict"] == "ERROR"  # classification-only run has no Agent decision


def test_response_decision_compares_against_the_predicted_label_agent_run_only():
    v = _variant(decision="GENERATE_DRAFT")
    v["agent_runs"]["verified"] = {"decision": {"routing_decision": "NO_RESPONSE_RECOMMENDED"}}
    assert playground_eval.compare({"response_decision": "GENERATE_DRAFT"}, "VERIFIED", v)["verdict"] == "PASS"


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        playground_eval.validate_status("MAYBE")


# ================================================= metrics
def _case(i, exp, act, status="VERIFIED", task="intent", agent_mode=None, extra_expected=None, extra_actual=None):
    return {"case_id": f"c{i}", "status": status, "expected": {task: exp, **(extra_expected or {})},
            "result": {"agent_mode": agent_mode, "actual": {task: act, **(extra_actual or {})}}, "verdict": {"verdict": "PASS"}}


def test_metrics_match_sklearn_reference():
    from sklearn.metrics import accuracy_score, f1_score

    exp = ["SUGGESTION"] * 5 + ["INQUIRY"] * 4 + ["SPAM"] * 3 + ["COMPLAINT"]
    act = ["SUGGESTION"] * 4 + ["INQUIRY"] + ["INQUIRY"] * 3 + ["SUGGESTION"] + ["SPAM"] * 2 + ["INQUIRY"] + ["COMPLAINT"]
    cases = [_case(i, e, a) for i, (e, a) in enumerate(zip(exp, act))]
    m = playground_eval.compute_evaluation(cases)["classification"]["intent"]
    present = sorted(set(exp))
    assert m["sufficient"] and m["n"] == 13
    assert m["accuracy"] == pytest.approx(accuracy_score(exp, act))
    assert m["macro_f1_supported_classes"] == pytest.approx(f1_score(exp, act, labels=present, average="macro", zero_division=0))
    assert m["weighted_f1"] == pytest.approx(f1_score(exp, act, labels=present, average="weighted", zero_division=0))
    cm = m["confusion_matrix"]
    assert cm["matrix"][cm["labels"].index("SUGGESTION")][cm["labels"].index("INQUIRY")] == 1
    thanks = next(c for c in m["per_class"] if c["label"] == "THANKS_PRAISE")
    assert thanks["support"] == 0 and thanks["f1"] is None  # unsupported class is not scored as 0


def test_aggregates_withheld_below_minimum_sample_but_counts_reported():
    cases = [_case(i, "SUGGESTION", "SUGGESTION") for i in range(5)] + [_case(9, "INQUIRY", "SUGGESTION")]
    m = playground_eval.compute_evaluation(cases)["classification"]["intent"]
    assert m["n"] == 6 and m["sufficient"] is False and m["accuracy"] is None and m["macro_f1_supported_classes"] is None
    assert "need at least 10" in m["withheld_reason"] and sum(map(sum, m["confusion_matrix"]["matrix"])) == 6


def test_single_class_sample_withholds_macro_and_weighted():
    m = playground_eval.compute_evaluation([_case(i, "SPAM", "SPAM") for i in range(12)])["classification"]["intent"]
    assert m["sufficient"] is False and "single class" in m["withheld_reason"]


def test_unverified_and_ambiguous_cases_are_excluded_from_every_metric():
    cases = [_case(i, "SUGGESTION", "SUGGESTION") for i in range(6)] + [_case(10 + i, "INQUIRY", "SPAM", status="UNVERIFIED") for i in range(5)] + [_case(20 + i, "INQUIRY", "SPAM", status="NEEDS_HUMAN_REVIEW") for i in range(5)]
    ev = playground_eval.compute_evaluation(cases)
    assert ev["classification"]["intent"]["n"] == 6
    assert ev["case_counts"] == {"total": 16, "verified": 6, "unverified": 5, "needs_human_review": 5}


def test_agent_summaries_keep_mock_and_live_separate():
    cases = [
        _case(1, "SUGGESTION", "SUGGESTION", agent_mode="mock", extra_expected={"response_decision": "GENERATE_DRAFT"}, extra_actual={"response_decision": "GENERATE_DRAFT"}),
        _case(2, "SUGGESTION", "SUGGESTION", agent_mode="mock", extra_expected={"response_decision": "GENERATE_DRAFT"}, extra_actual={"response_decision": "NO_RESPONSE_RECOMMENDED"}),
        _case(3, "SUGGESTION", "SUGGESTION", agent_mode="live", extra_expected={"response_decision": "GENERATE_DRAFT"}, extra_actual={"response_decision": "GENERATE_DRAFT"}),
    ]
    agent = playground_eval.compute_evaluation(cases)["agent"]
    assert (agent["mock"]["n"], agent["mock"]["correct"]) == (2, 1) and (agent["live"]["n"], agent["live"]["correct"]) == (1, 1)
    assert "plumbing" in agent["mock"]["note"]


# ================================================= storage & privacy
def test_running_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PLAYGROUND_DATA_DIR", tmp_path / "pg")
    run({"variants": [GOOD], "mode": "full_pipeline"})
    assert not (tmp_path / "pg").exists()


def test_store_save_list_version_and_delete(tmp_path):
    s = playground_store.PlaygroundStore(tmp_path / "pg")
    a = s.save_case({"text": "x", "status": "UNVERIFIED", "expected": {}}, saved_by="a@x")
    b = s.save_case({"case_id": a["case_id"], "text": "x", "status": "VERIFIED", "expected": {"intent": "SPAM"}}, saved_by="a@x")
    assert (a["version"], b["version"]) == (1, 2)
    assert [c["status"] for c in s.list_cases()] == ["VERIFIED"]
    assert s.delete_case(a["case_id"], deleted_by="a@x") and s.list_cases() == []
    assert not s.delete_case("nope", deleted_by="a@x")
    assert len((tmp_path / "pg" / "cases.jsonl").read_text().splitlines()) == 3  # append-only history


def test_store_survives_a_torn_line(tmp_path):
    s = playground_store.PlaygroundStore(tmp_path / "pg")
    s.save_case({"text": "x", "status": "UNVERIFIED", "expected": {}}, saved_by=None)
    with open(s.path, "a", encoding="utf-8") as fh:
        fh.write('{"truncated": \n')
    assert len(s.list_cases()) == 1


@pytest.mark.parametrize("bad", ["data/gold", "data/db/x", "data/uploads", "data/raw", "data/models/final_test", "data/gold/sub/playground"])
def test_store_refuses_gold_and_production_data_locations(tmp_path, bad):
    with pytest.raises(playground_store.UnsafeStorageLocation):
        playground_store.PlaygroundStore(tmp_path / bad)


def test_default_storage_dir_is_outside_gold_and_git_ignored():
    playground_store.assert_safe_dir(Path(settings.BASE_DIR) / "data" / "evaluation" / "playground")
    gi = (Path(settings.BASE_DIR) / ".gitignore").read_text()
    assert "data/evaluation/playground/" in gi


def test_playground_modules_have_no_database_or_production_write_surface():
    """AST-level: no persistence/review/outbound import and no such identifier
    anywhere in the playground modules (docstrings and comments excluded)."""
    import ast

    import rcs.response_intelligence.playground as m1
    import rcs.response_intelligence.playground_eval as m2
    import rcs.response_intelligence.playground_store as m3
    import rcs.review.playground_api as m4

    banned_modules = ("sqlalchemy", "psycopg", "pymongo", "rcs.review.repository", "rcs.review.service", "rcs.review.workflow", "rcs.review.outbound",
                      "rcs.response_generation.outbound", "rcs.response_intelligence.audit", "rcs.review.persistence")
    banned_names = {"ReviewWorkflowService", "apply_generation_result", "send_as_is", "edit_and_send", "upsert_review_items", "get_engine", "AgentAuditRepository", "PostgresAgentAuditRepository"}
    for mod in (m1, m2, m3, m4):
        tree = ast.parse(pyinspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                names = []
            for n in names:
                assert not n.startswith(banned_modules), (mod.__name__, n)
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, (mod.__name__, node.id)
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_names, (mod.__name__, node.attr)
    # gold / sealed-test locations are never referenced as string literals either
    for mod in (m1, m2, m4):
        for node in ast.walk(ast.parse(pyinspect.getsource(mod))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 80:
                assert "gold" not in node.value.lower() and "human_test" not in node.value.lower(), (mod.__name__, node.value)


def test_disabled_by_default_and_hard_disabled_in_production(monkeypatch):
    monkeypatch.delenv("RCS_ENV", raising=False)
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", False)
    assert playground.playground_available() is False
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", True)
    assert playground.playground_available() is True
    monkeypatch.setenv("RCS_ENV", "production")
    assert playground.playground_available() is False
    assert playground.status_payload()["enabled"] is False and playground.status_payload()["taxonomy"] is None


def test_default_setting_is_disabled_without_the_env_flag():
    """Fresh interpreter with the flag removed from the environment: the shipped default is OFF."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith("RCS_PLAYGROUND")}
    code = "import sys; sys.path[:0]=['src','.']; from config import settings; print(settings.PLAYGROUND_ENABLED, settings.PLAYGROUND_DATA_DIR.parts[-3:])"
    out = subprocess.run([sys.executable, "-c", code], cwd=settings.BASE_DIR, env=env, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "False ('data', 'evaluation', 'playground')", out.stdout + out.stderr


def test_status_payload_never_contains_the_api_key(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "sk-very-secret")
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", True)
    payload = json.dumps(playground.status_payload())
    assert "sk-very-secret" not in payload and playground.status_payload()["live_configured"] is True


# ================================================= HTTP: auth, disabled, no production writes
def _http(url, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode()), e.headers


def _login(base, email, password="a-real-password-123"):
    _s, _p, h = _http(f"{base}/api/login", method="POST", data={"email": email, "password": password})
    return h.get("Set-Cookie").split(";")[0]


@pytest.fixture
def server(tmp_path, monkeypatch):
    engine = get_engine(seed_demo_db(tmp_path / "demo.sqlite"))
    create_user(engine, email="admin@example.org", password="a-real-password-123", display_name="Admin", role="ADMIN")
    create_user(engine, email="rev@example.org", password="a-real-password-123", display_name="Rev", role="REVIEWER")
    monkeypatch.delenv("RCS_ENV", raising=False)
    monkeypatch.setattr(settings, "PLAYGROUND_DATA_DIR", tmp_path / "pg")
    monkeypatch.setattr(playground_api, "_classifier_provider", lambda: FakeSuite())
    monkeypatch.setattr(playground_api, "_retrievers", fake_retrievers())
    srv, base = serve_in_thread(engine)
    yield engine, base, tmp_path
    srv.shutdown()


GET_ROUTES = ["/api/admin/playground/status", "/api/admin/playground/cases", "/api/admin/playground/metrics"]
POST_ROUTES = ["/api/admin/playground/run", "/api/admin/playground/compare", "/api/admin/playground/cases", "/api/admin/playground/cases/delete"]


def test_unauthenticated_and_reviewer_access_is_denied_on_every_route(server, monkeypatch):
    _engine, base, _tmp = server
    for enabled in (False, True):
        monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", enabled)
        rev = _login(base, "rev@example.org")
        for route in GET_ROUTES:
            assert _http(base + route)[0] == 401
            assert _http(base + route, cookie=rev)[0] == 403
        for route in POST_ROUTES:
            assert _http(base + route, method="POST", data={"variants": [GOOD]})[0] == 401
            assert _http(base + route, method="POST", data={"variants": [GOOD]}, cookie=rev)[0] == 403


def test_disabled_playground_answers_404_for_admin_and_does_nothing(server, monkeypatch):
    _engine, base, tmp = server
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", False)
    admin = _login(base, "admin@example.org")
    s, body, _ = _http(base + "/api/admin/playground/status", cookie=admin)
    assert s == 200 and body["enabled"] is False and body["taxonomy"] is None
    for route in GET_ROUTES[1:]:
        assert _http(base + route, cookie=admin)[:2] == (404, {"error": "playground_disabled"})
    for route in POST_ROUTES:
        assert _http(base + route, method="POST", data={"variants": [GOOD]}, cookie=admin)[:2] == (404, {"error": "playground_disabled"})
    assert not (tmp / "pg").exists()


def test_production_env_disables_playground_even_when_flag_is_on(server, monkeypatch):
    _engine, base, _tmp = server
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", True)
    admin = _login(base, "admin@example.org")
    monkeypatch.setenv("RCS_ENV", "production")
    assert _http(base + "/api/admin/playground/run", method="POST", data={"variants": [GOOD]}, cookie=admin)[0] == 404


def _table_fingerprint(engine):
    with engine.connect() as conn:
        out = {}
        for t in sa_inspect(engine).get_table_names():
            if t in ("user_sessions",):
                continue
            out[t] = [tuple(map(str, r)) for r in conn.execute(sa_text(f'SELECT * FROM "{t}"')).fetchall()]
        return out


def test_full_http_workflow_never_writes_to_the_production_database(server, monkeypatch):
    engine, base, tmp = server
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", True)
    admin = _login(base, "admin@example.org")
    before = _table_fingerprint(engine)

    s, run_out, _ = _http(base + "/api/admin/playground/run", method="POST", cookie=admin, data={"variants": [GOOD, GOOD + " 😊"], "mode": "full_pipeline"})
    assert s == 200 and run_out["agent_mode"] == "mock" and len(run_out["variants"]) == 2
    assert not (tmp / "pg").exists()  # running never autosaves comment text
    rid = run_out["run_id"]
    s, cmp_out, _ = _http(base + "/api/admin/playground/compare", method="POST", cookie=admin,
                          data={"run_id": rid, "variant_index": 0, "status": "VERIFIED", "expected": {"intent": "SUGGESTION"}})
    assert s == 200 and cmp_out["verdict"] == "PASS"
    s, saved, _ = _http(base + "/api/admin/playground/cases", method="POST", cookie=admin,
                        data={"run_id": rid, "variant_index": 0, "status": "VERIFIED", "expected": {"intent": "SUGGESTION"}, "notes": "n"})
    assert s == 200 and saved["case"]["verified_by"] == "admin@example.org" and saved["case"]["text"] == GOOD
    assert (tmp / "pg" / "cases.jsonl").exists()
    s, listing, _ = _http(base + "/api/admin/playground/cases", cookie=admin)
    assert [c["case_id"] for c in listing["cases"]] == [saved["case"]["case_id"]]
    s, metrics, _ = _http(base + "/api/admin/playground/metrics", cookie=admin)
    assert s == 200 and metrics["classification"]["intent"]["n"] == 1 and metrics["classification"]["intent"]["sufficient"] is False
    s, deleted, _ = _http(base + "/api/admin/playground/cases/delete", method="POST", cookie=admin, data={"case_id": saved["case"]["case_id"]})
    assert deleted == {"deleted": True}

    assert _table_fingerprint(engine) == before  # review_items, traces, assignments... all untouched


def test_save_uses_server_side_predictions_not_client_supplied_results(server, monkeypatch):
    _engine, base, _tmp = server
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", True)
    admin = _login(base, "admin@example.org")
    _s, run_out, _ = _http(base + "/api/admin/playground/run", method="POST", cookie=admin, data={"variants": [GOOD]})
    _s, saved, _ = _http(base + "/api/admin/playground/cases", method="POST", cookie=admin,
                         data={"run_id": run_out["run_id"], "variant_index": 0, "status": "UNVERIFIED", "expected": {},
                               "result": {"actual": {"intent": "SPAM"}}, "actual": {"intent": "SPAM"}})
    assert saved["case"]["result"]["actual"]["intent"] == "SUGGESTION"
    assert _http(base + "/api/admin/playground/cases", method="POST", cookie=admin, data={"run_id": "run-nope", "variant_index": 0})[0] == 400


def test_http_validation_errors_are_400_and_verified_needs_a_label(server, monkeypatch):
    _engine, base, _tmp = server
    monkeypatch.setattr(settings, "PLAYGROUND_ENABLED", True)
    admin = _login(base, "admin@example.org")
    assert _http(base + "/api/admin/playground/run", method="POST", cookie=admin, data={"variants": ["abcdef"] * 11})[0] == 400
    assert _http(base + "/api/admin/playground/run", method="POST", cookie=admin, data={"variants": [GOOD], "mode": "full_pipeline", "agent_mode": "live"})[0] == 400
    _s, run_out, _ = _http(base + "/api/admin/playground/run", method="POST", cookie=admin, data={"variants": [GOOD]})
    base_body = {"run_id": run_out["run_id"], "variant_index": 0}
    assert _http(base + "/api/admin/playground/cases", method="POST", cookie=admin, data={**base_body, "status": "VERIFIED", "expected": {}})[0] == 400
    assert _http(base + "/api/admin/playground/cases", method="POST", cookie=admin, data={**base_body, "status": "VERIFIED", "expected": {"intent": "BOGUS"}})[0] == 400
    assert _http(base + "/api/admin/playground/cases", method="POST", cookie=admin, data={**base_body, "status": "NEEDS_HUMAN_REVIEW", "expected": {"intent": "INQUIRY"}})[0] == 200
