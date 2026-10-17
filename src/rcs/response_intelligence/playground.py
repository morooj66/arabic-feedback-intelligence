"""
Interactive AI Evaluation Playground -- core run logic (internal, admin-only).

This module is NOT a new classifier, router or agent and contains NO new
decision logic. It chains the exact production functions, in the exact
production order, for typed-in custom comments:

    production (rcs.review.excel_review_flow)        playground (this module)
    ------------------------------------------        --------------------------------------
    ingest.cleaning.normalize_text                    same function (whitespace/control cleanup)
    prefilter.spam_rules single-row rules 1-5         same functions, same order, same junk list
    classification.runtime intent/topic/sentiment     same configured runtime, ONE batch per model
    routing.engine.determine_routing                  same function, same arguments
    needs_review = intent_conf < LOW_CONFIDENCE       same expression
    GenerationRequest(...) -> run_agent_pipeline      same request shape, capture_trace=True

Unavoidable differences from production (also returned to the UI):
  * prefilter rule 6 (burst repetition) needs submitter identity/time and the
    database; content de-duplication is database-backed. Neither is reproduced.
  * No title/entry_id metadata exists for typed text, so the identifying-context
    policy has nothing to evaluate.
  * Production classifies a whole upload as one batch; the playground classifies
    the variants of one run as one batch. Local frozen-model weights are loaded
    by the unchanged inference functions; experimental hybrid Topic uses the
    configured redacted provider boundary. Timings include either cost.
  * Agent retrieval uses the real default retrievers (read-only). Their corpora
    are whatever is configured locally; the approved-memory corpus is empty today.

Safety contract:
  * Never touches review_items, sessions, assignments, the production audit
    repository, outbound/send paths, gold data or any database. The Agent is
    called with audit_repository=None.
  * MOCK is the default Agent mode. LIVE needs explicit selection AND an
    explicit confirmation flag, and an already-configured API key.
    This controls Gemini generation only; the opt-in hybrid classification
    mode may still call Jev for Topic after its mandatory PII-redaction gate.
  * Nothing is written to disk by running. Persistence happens only through
    playground_store, on an explicit save request.
"""

from __future__ import annotations

import collections
import json
import re
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable

from config import settings
from rcs.ingest.cleaning import normalize_text
from rcs.prefilter import spam_rules
from rcs.response_generation.eligibility import EligibilityInput, determine_response_eligibility
from rcs.response_generation.service import GenerationRequest, build_live_gemini_client
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from rcs.response_intelligence.orchestrator import run_agent_pipeline
from rcs.routing.engine import determine_routing

MODE_CLASSIFICATION_ONLY = "classification_only"
MODE_FULL_PIPELINE = "full_pipeline"
AGENT_MOCK = "mock"
AGENT_LIVE = "live"
SOURCE_PREDICTED = "predicted"
SOURCE_VERIFIED = "verified"
SOURCE_BOTH = "both"

MOCK_RESPONSE_TEXT = "شكرًا لتواصلكم، وصلت رسالتكم."   # passes every production gate; no delivery-to-team claim
MOCK_BANNER = "MOCK – scripted placeholder text, NOT Gemini output. Says nothing about real response quality."
LIVE_BANNER = "LIVE – real Gemini call (billed; typed text is sent to an external service)."

PARITY_NOTES = (
    "Reproduced from production: text normalization, prefilter rules 1-5, the configured classification runtime, determine_routing, "
    "the low-confidence needs_review rule, GenerationRequest construction and run_agent_pipeline.",
    "NOT reproduced: prefilter rule 6 (burst repetition: needs submitter identity/time), database content de-duplication, "
    "title/entry_id identifying metadata (typed text has none).",
    "Confidence values are the top-class probability only (that is all the frozen inference functions return).",
    "Local frozen Topic (legacy mode) and Sentiment inference include model-loading time; hybrid Topic includes provider latency.",
    "Pre-retrieval eligibility is a preview; the Agent's own decision (after retrieval) is authoritative.",
    "If ANY classifier (intent, topic or sentiment) fails, the Full Pipeline Agent stage is NOT run; successful classification results are still shown but labelled partial.",
    "Fixed-label Agent runs (manually verified labels) are isolation tests that bypass the classifiers; they are never production-equivalent.",
)


class PlaygroundError(ValueError):
    """Invalid playground request (maps to HTTP 400)."""


# ---------------------------------------------------------------------------
# Availability / taxonomy
# ---------------------------------------------------------------------------
def playground_available() -> bool:
    from rcs.review.identity import is_production_env

    return bool(settings.PLAYGROUND_ENABLED) and not is_production_env()


def taxonomy() -> dict[str, list[str]]:
    return {
        "intents": list(settings.PRIMARY_INTENTS),
        "topics": list(settings.TOPIC_SUBTYPES),
        "sentiments": list(settings.SENTIMENTS),
        "routing_decisions": list(settings.ROUTING_DECISIONS),
        "response_decisions": list(settings.RESPONSE_ELIGIBILITY_DECISIONS),
    }


def status_payload() -> dict[str, Any]:
    from rcs.review.identity import is_production_env

    available = playground_available()
    return {
        "enabled": available,
        "disabled_reason": None
        if available
        else ("production_environment" if is_production_env() else "RCS_PLAYGROUND_ENABLED_not_set"),
        "limits": {"max_variants": settings.PLAYGROUND_MAX_VARIANTS, "max_chars": settings.PLAYGROUND_MAX_CHARS},
        "modes": [MODE_CLASSIFICATION_ONLY, MODE_FULL_PIPELINE],
        "agent_modes": [AGENT_MOCK, AGENT_LIVE],
        "default_agent_mode": AGENT_MOCK,
        "live_configured": bool(settings.GEMINI_API_KEY),  # boolean only -- never the key
        "taxonomy": taxonomy() if available else None,
        "parity_notes": list(PARITY_NOTES),
        "low_confidence_threshold": settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Error sanitization (never leak credentials through error text)
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|authorization)\s*[=:]\s*\S+"),
    re.compile(r"(?i)bearer\s+\S+"),
)


def sanitize_error_text(text: str) -> str:
    out = str(text)
    key = settings.GEMINI_API_KEY
    if key:
        out = out.replace(key, "[redacted]")
    for rx in _SECRET_PATTERNS:
        out = rx.sub("[redacted]", out)
    return out[:300]


def _error(stage: str, exc: BaseException | str) -> dict[str, str]:
    if isinstance(exc, BaseException):
        return {"stage": stage, "type": type(exc).__name__, "message": sanitize_error_text(str(exc))}
    return {"stage": stage, "type": "Error", "message": sanitize_error_text(exc)}


# ---------------------------------------------------------------------------
# Classifier suite (injectable; defaults are the real configured runtime)
# ---------------------------------------------------------------------------
class ClassifierSuite:
    def __init__(self, intent: Callable, topic: Callable, sentiment: Callable):
        self.intent, self.topic, self.sentiment = intent, topic, sentiment


def default_classifiers() -> ClassifierSuite:
    from rcs.classification import runtime

    return ClassifierSuite(runtime.predict_intent, runtime.predict_topic, runtime.predict_sentiment)


# ---------------------------------------------------------------------------
# Mock Gemini client (scripted; never networks)
# ---------------------------------------------------------------------------
class MockGeminiClient:
    def __init__(self, text: str = MOCK_RESPONSE_TEXT):
        self._payload = json.dumps({"suggested_response": text}, ensure_ascii=False)
        self.call_count = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.call_count += 1
        return self._payload


class _CountingClient:
    """Counts actual generate() invocations (including retries) so the UI can
    show real LLM request counts. The orchestrator's own
    `logical_generation_count` also counts passes that eligibility skipped."""

    def __init__(self, inner):
        self._inner, self.calls = inner, 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        self.calls += 1
        return self._inner.generate(prompt, response_schema)


# ---------------------------------------------------------------------------
# In-memory run cache (process-local; lets "save" take predictions from the
# server's own record of a run instead of trusting client-supplied results)
# ---------------------------------------------------------------------------
class RunCache:
    def __init__(self, max_runs: int = 100):
        self._runs: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self._max = max_runs
        self._lock = threading.Lock()

    def put(self, run_id: str, payload: dict) -> None:
        with self._lock:
            self._runs[run_id] = payload
            while len(self._runs) > self._max:
                self._runs.popitem(last=False)

    def get(self, run_id: str) -> dict | None:
        with self._lock:
            return self._runs.get(run_id)

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


RUN_CACHE = RunCache()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_labels(labels: dict[str, Any] | None, *, require_all: bool = False) -> dict[str, str]:
    """Validate intent/topic/sentiment against the frozen taxonomy in config.settings."""
    labels = labels or {}
    allowed = {
        "intent": settings.PRIMARY_INTENTS,
        "topic": settings.TOPIC_SUBTYPES,
        "sentiment": settings.SENTIMENTS,
    }
    out: dict[str, str] = {}
    for key, values in allowed.items():
        v = labels.get(key)
        if v in (None, ""):
            continue
        if v not in values:
            raise PlaygroundError(f"invalid {key} label {v!r}; allowed: {list(values)}")
        out[key] = v
    if require_all:
        missing = [k for k in allowed if k not in out]
        if missing:
            raise PlaygroundError(f"missing labels: {missing}")
    return out


def _parse_variants(body: dict[str, Any]) -> list[str]:
    raw = body.get("variants")
    if not isinstance(raw, list) or not raw:
        raise PlaygroundError("variants must be a non-empty list")
    if len(raw) > settings.PLAYGROUND_MAX_VARIANTS:
        raise PlaygroundError(f"at most {settings.PLAYGROUND_MAX_VARIANTS} variants per run")
    texts = []
    for item in raw:
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str):
            raise PlaygroundError("each variant must be text")
        if len(text) > settings.PLAYGROUND_MAX_CHARS:
            raise PlaygroundError(f"comment exceeds {settings.PLAYGROUND_MAX_CHARS} characters")
        texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Production-parity preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> dict[str, Any]:
    """Phase 2 normalization + Phase 3 single-row prefilter rules 1-6, in
    production order (first match wins), using the production functions."""
    normalized = normalize_text(text)
    reason = None
    for rule in (
        lambda c: spam_rules.detect_known_junk_string(c, settings.KNOWN_JUNK_STRINGS),
        spam_rules.detect_null_or_near_empty,
        spam_rules.detect_phone_number_stub,
        spam_rules.detect_unfilled_template,
        spam_rules.detect_test_marker,
        spam_rules.detect_keyboard_mash_sequence,
    ):
        reason = rule(normalized)
        if reason:
            break
    return {"normalized_text": normalized, "changed": normalized != text, "prefilter_reason": reason}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _valid_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 else None


# ---------------------------------------------------------------------------
# Classification (one batch per model, exactly like run_frozen_classification)
# ---------------------------------------------------------------------------
def _classify_batch(texts: list[str], suite: ClassifierSuite) -> tuple[list[dict], dict[str, float]]:
    n = len(texts)
    ids = [f"playground-{uuid.uuid4().hex}" for _ in texts]
    timings: dict[str, float] = {}
    out: list[dict] = [{"intent": None, "topic": None, "sentiment": None, "errors": []} for _ in range(n)]

    def field(key, taxonomy_values, fetch):
        t0 = time.perf_counter()
        try:
            rows = fetch()
        except Exception as exc:  # model missing, OOM, etc. -> per-field error, run continues
            for o in out:
                o["errors"].append(_error(f"classification.{key}", exc))
            timings[key] = _ms(t0)
            return
        timings[key] = _ms(t0)
        for i, row in enumerate(rows):
            label, conf = row if row is not None else (None, None)
            if label not in taxonomy_values:
                out[i]["errors"].append(_error(f"classification.{key}", f"label {label!r} is outside the frozen taxonomy"))
                continue
            out[i][key] = {"label": label, "confidence": _valid_confidence(conf)}

    def topic_rows():
        m = suite.topic(ids, texts)
        return [m.get(cid) for cid in ids]

    def sent_rows():
        m = suite.sentiment(ids, texts)
        return [m.get(cid) for cid in ids]

    field("intent", settings.PRIMARY_INTENTS, lambda: list(suite.intent(texts)))
    field("topic", settings.TOPIC_SUBTYPES, topic_rows)
    field("sentiment", settings.SENTIMENTS, sent_rows)
    timings["total"] = round(sum(timings.values()), 2)
    return out, timings


def _routing_for(intent: str | None, topic: str | None) -> dict[str, Any] | None:
    if intent is None or topic is None:
        return None
    r = determine_routing(
        primary_intent=intent,
        topic_subtype=topic,
        subcategory=None,
        secondary_intent=None,
        is_content_meaningful=True,
    )
    return {"decision": r.routing_decision, "reason_code": r.reason_code}


def _eligibility_preview(text, intent, topic, sentiment, confidence, needs_review, routing_decision) -> dict[str, Any]:
    r = determine_response_eligibility(
        EligibilityInput(
            comment_text=text,
            primary_intent=intent,
            topic_subtype=topic,
            sentiment=sentiment,
            confidence=confidence,
            needs_review=needs_review,
            classification_routing_decision=routing_decision,
        )
    )
    return {**asdict(r), "note": "pre-retrieval preview; the Agent decides after retrieval"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
def _run_agent(
    *,
    text: str,
    labels: dict[str, str],
    confidence: float | None,
    needs_review: bool,
    classification_routing_decision: str | None,
    client,
    agent_mode: str,
    label_source: str,
    retrievers: dict[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label_source": label_source,
        "production_equivalent": label_source == SOURCE_PREDICTED,
        "path_note": "classifier predictions -> routing -> Agent (production-equivalent path)"
        if label_source == SOURCE_PREDICTED
        else "ISOLATION TEST with manually verified labels: classifiers bypassed, NOT the production path",
        "agent_mode": agent_mode,
        "mode_banner": MOCK_BANNER if agent_mode == AGENT_MOCK else LIVE_BANNER,
        "labels_used": {**labels, "confidence": confidence, "needs_review": needs_review},
        "classification_routing_decision": classification_routing_decision,
        "decision": None,
        "fallback": None,
        "tool_calls": [],
        "drafts": None,
        "response": None,
        "logical_generation_count": 0,
        "llm_calls": 0,
        "execution_time_ms": None,
        "error": None,
    }
    request = GenerationRequest(
        comment_id=f"playground-{uuid.uuid4().hex}",
        comment_text=text,
        primary_intent=labels.get("intent"),
        topic_subtype=labels.get("topic"),
        sentiment=labels.get("sentiment"),
        confidence=confidence,
        needs_review=needs_review,
        classification_routing_decision=classification_routing_decision,
        identifying_metadata=IdentifyingMetadata(None, None),
    )
    kwargs = {}
    if retrievers:
        kwargs.update(retrievers)
    counting = _CountingClient(client) if client is not None else None
    t0 = time.perf_counter()
    try:
        result = run_agent_pipeline(request, counting, None, audit_repository=None, capture_trace=True, **kwargs)
    except Exception as exc:
        record["llm_calls"] = counting.calls if counting else 0
        record["execution_time_ms"] = _ms(t0)
        record["error"] = _error("agent", exc)
        record["fallback"] = {"kind": "agent_exception", "detail": "Agent raised; a production caller would fail this row closed to REVIEW_ONLY."}
        return record
    record["execution_time_ms"] = _ms(t0)
    record["llm_calls"] = counting.calls if counting else 0
    gr, md, trace = result.generation_result, result.metadata, result.trace
    record["decision"] = {
        "routing_decision": gr.routing_decision,
        "generation_status": gr.generation_status,
        "reason_code": gr.reason_code,
        "eligibility_rule_id": gr.eligibility_rule_id,
        "review_required": gr.review_required,
    }
    record["logical_generation_count"] = md.logical_generation_count
    record["response"] = gr.suggested_response
    if gr.generation_status == "FAILED_REVIEW_ONLY":
        record["fallback"] = {"kind": "failed_review_only", "detail": "Generation failed closed; production routes this to REVIEW_ONLY."}
        record["error"] = _error("agent.generation", gr.error or gr.reason_code)
    record["tool_calls"] = [
        {"tool": "official_knowledge_retrieval", "outcome": md.official_retrieval_outcome, "result_count": md.official_result_count},
        {"tool": "approved_response_memory_retrieval", "outcome": md.style_retrieval_outcome, "result_count": md.style_result_count, "note": "style only, never factual"},
        {
            "tool": "validation_gates",
            "initial_passed": md.initial_validation_passed,
            "initial_failure_code": md.initial_failure_code,
            "refinement_attempted": md.refinement_attempted,
            "final_passed": md.final_validation_passed,
            "final_failure_code": md.final_failure_code,
        },
        {
            "tool": "generation",
            "provider": "mock" if agent_mode == AGENT_MOCK else "gemini",
            "llm_requests": record["llm_calls"],
            "orchestrator_logical_generations": md.logical_generation_count,
            "grounding_mode": md.grounding_mode,
        },
    ]
    if trace is not None:
        record["drafts"] = {"draft_1": trace.initial_draft, "draft_2": trace.refinement_draft}
    return record


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_playground(
    body: dict[str, Any],
    *,
    classifiers: ClassifierSuite | None = None,
    live_client_factory: Callable[[], Any] | None = None,
    retrievers: dict[str, Any] | None = None,
    cache: RunCache | None = None,
    user_email: str | None = None,
) -> dict[str, Any]:
    t_run = time.perf_counter()
    texts = _parse_variants(body)
    mode = body.get("mode") or MODE_CLASSIFICATION_ONLY
    if mode not in (MODE_CLASSIFICATION_ONLY, MODE_FULL_PIPELINE):
        raise PlaygroundError(f"unknown mode {mode!r}")
    agent_mode = body.get("agent_mode") or AGENT_MOCK
    if agent_mode not in (AGENT_MOCK, AGENT_LIVE):
        raise PlaygroundError(f"unknown agent_mode {agent_mode!r}")
    label_source = body.get("label_source") or SOURCE_PREDICTED
    if label_source not in (SOURCE_PREDICTED, SOURCE_VERIFIED, SOURCE_BOTH):
        raise PlaygroundError(f"unknown label_source {label_source!r}")
    if mode == MODE_CLASSIFICATION_ONLY and label_source != SOURCE_PREDICTED:
        raise PlaygroundError("fixed-label testing needs mode=full_pipeline")
    force = bool(body.get("force_classify_prefiltered", False))

    # Per-variant fixed (verified) labels, used ONLY for Agent isolation testing.
    fixed_raw = body.get("fixed_labels") or [None] * len(texts)
    if not isinstance(fixed_raw, list) or len(fixed_raw) != len(texts):
        raise PlaygroundError("fixed_labels must be a list aligned with variants")
    fixed: list[dict[str, str] | None] = []
    for item in fixed_raw:
        if not item:
            fixed.append(None)
            continue
        if item.get("verified") is not True:
            fixed.append(None)  # unverified labels are never used as ground truth
            continue
        fixed.append(validate_labels(item.get("labels"), require_all=True))
    if label_source in (SOURCE_VERIFIED, SOURCE_BOTH) and not any(fixed):
        raise PlaygroundError("fixed-label testing needs at least one variant with verified labels")

    # ---- Agent client selection (mock default; live needs explicit confirmation)
    client = None
    if mode == MODE_FULL_PIPELINE:
        if agent_mode == AGENT_LIVE:
            if body.get("live_confirmed") is not True:
                raise PlaygroundError("live Gemini requires live_confirmed=true (cost and privacy warning acknowledged)")
            if not settings.GEMINI_API_KEY:
                raise PlaygroundError("live Gemini requested but no API key is configured")
            client = (live_client_factory or build_live_gemini_client)()
        else:
            client = MockGeminiClient()

    suite = classifiers or default_classifiers()

    # ---- preprocessing per variant
    pre = []
    for text in texts:
        t0 = time.perf_counter()
        p = preprocess(text)
        p["preprocess_ms"] = _ms(t0)
        pre.append(p)

    needs_predicted = label_source in (SOURCE_PREDICTED, SOURCE_BOTH) or mode == MODE_CLASSIFICATION_ONLY
    classify_idx = [i for i, p in enumerate(pre) if p["normalized_text"] and (p["prefilter_reason"] is None or force)] if needs_predicted else []
    cls_by_idx: dict[int, dict] = {}
    cls_timings: dict[str, float] = {}
    if classify_idx:
        rows, cls_timings = _classify_batch([pre[i]["normalized_text"] for i in classify_idx], suite)
        cls_by_idx = dict(zip(classify_idx, rows))

    variants = []
    for i, text in enumerate(texts):
        p = pre[i]
        v: dict[str, Any] = {
            "index": i,
            "text_length": len(text),
            "preprocessing": {
                "normalized_text": p["normalized_text"],
                "changed": p["changed"],
                "prefilter_reason": p["prefilter_reason"],
                "production_outcome": "EXCLUDED_BY_PREFILTER (never classified in production)" if p["prefilter_reason"] else "CLASSIFIED",
                "forced_classification": bool(p["prefilter_reason"] and force and i in cls_by_idx),
            },
            "classification": None,
            "routing": None,
            "eligibility_preview": None,
            "agent_runs": {},
            "timings_ms": {"preprocess": p["preprocess_ms"]},
            "errors": [],
        }
        can_run = p["normalized_text"] is not None and (p["prefilter_reason"] is None or force)
        predicted_labels = None
        pred_conf = None
        pred_needs_review = False
        if i in cls_by_idx:
            c = cls_by_idx[i]
            v["classification"] = {k: c[k] for k in ("intent", "topic", "sentiment")}
            v["errors"].extend(c["errors"])
            failed_stages = sorted({e["stage"] for e in c["errors"]} | {f"classification.{k}" for k in ("intent", "topic", "sentiment") if not c[k]})
            classification_complete = not failed_stages
            v["pipeline_status"] = {
                "classification_complete": classification_complete,
                "failed_stages": failed_stages,
                "production_equivalent": classification_complete,
                "note": None if classification_complete else "partial classification: production would fail this row at the first failing model, so nothing downstream is production-equivalent",
            }
            t0 = time.perf_counter()
            if c["intent"] and c["topic"]:
                v["routing"] = _routing_for(c["intent"]["label"], c["topic"]["label"])
                v["routing"]["partial_classification"] = not classification_complete
                pred_conf = c["intent"]["confidence"]
                pred_needs_review = pred_conf is not None and pred_conf < settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD
                predicted_labels = {"intent": c["intent"]["label"], "topic": c["topic"]["label"]}
                if c["sentiment"]:
                    predicted_labels["sentiment"] = c["sentiment"]["label"]
                v["eligibility_preview"] = _eligibility_preview(
                    p["normalized_text"],
                    predicted_labels["intent"],
                    predicted_labels["topic"],
                    predicted_labels.get("sentiment"),
                    pred_conf,
                    pred_needs_review,
                    v["routing"]["decision"],
                )
                v["eligibility_preview"]["partial_classification"] = not classification_complete
            else:
                v["errors"].append(_error("routing", "intent and topic are both required for routing; a production row would fail here"))
            v["timings_ms"]["routing"] = _ms(t0)

        if mode == MODE_FULL_PIPELINE and can_run:
            # A: actual classifier predictions
            if label_source in (SOURCE_PREDICTED, SOURCE_BOTH):
                status = v.get("pipeline_status") or {}
                if predicted_labels is None or not status.get("classification_complete"):
                    failed = status.get("failed_stages") or ["classification"]
                    diag = next((e for e in v["errors"] if e["stage"] in failed), None)
                    detail = f"{diag['stage']}: {diag['type']} - {diag['message']}" if diag else ", ".join(failed)
                    v["agent_skipped"] = {
                        "label_source": SOURCE_PREDICTED,
                        "reason": "classification_incomplete",
                        "failed_stages": failed,
                        "detail": f"Agent not run because classification failed at {', '.join(failed)}. First diagnostic: {detail}",
                        "production_equivalent": False,
                    }
                    v["errors"].append(_error("agent", v["agent_skipped"]["detail"]))
                else:
                    v["agent_runs"][SOURCE_PREDICTED] = _run_agent(
                        text=p["normalized_text"], labels=predicted_labels, confidence=pred_conf, needs_review=pred_needs_review,
                        classification_routing_decision=v["routing"]["decision"], client=client, agent_mode=agent_mode,
                        label_source=SOURCE_PREDICTED, retrievers=retrievers,
                    )
            # B: manually verified labels (classifiers bypassed, routing via the SAME determine_routing)
            if label_source in (SOURCE_VERIFIED, SOURCE_BOTH) and fixed[i]:
                fl = fixed[i]
                fr = _routing_for(fl["intent"], fl["topic"])
                v["agent_runs"][SOURCE_VERIFIED] = _run_agent(
                    text=p["normalized_text"], labels=fl, confidence=None, needs_review=False,
                    classification_routing_decision=fr["decision"] if fr else None, client=client, agent_mode=agent_mode,
                    label_source=SOURCE_VERIFIED, retrievers=retrievers,
                )
                v["agent_runs"][SOURCE_VERIFIED]["classification_routing"] = fr
        v["timings_ms"]["agent"] = round(sum(r["execution_time_ms"] or 0 for r in v["agent_runs"].values()), 2)
        if len(v["agent_runs"]) == 2:
            a, b = v["agent_runs"][SOURCE_PREDICTED], v["agent_runs"][SOURCE_VERIFIED]
            da, db = a.get("decision") or {}, b.get("decision") or {}
            v["fixed_label_comparison"] = {
                "decision_differs": da.get("routing_decision") != db.get("routing_decision"),
                "rule_differs": da.get("eligibility_rule_id") != db.get("eligibility_rule_id"),
                "response_differs": a.get("response") != b.get("response"),
                "note": "differences caused by classifier errors (A vs B); Mock responses are placeholders and say nothing about generation quality",
            }
        variants.append(v)

    run_id = f"run-{uuid.uuid4().hex}"
    payload = {
        "run_id": run_id,
        "mode": mode,
        "agent_mode": agent_mode if mode == MODE_FULL_PIPELINE else None,
        "label_source": label_source,
        "mode_banner": (MOCK_BANNER if agent_mode == AGENT_MOCK else LIVE_BANNER) if mode == MODE_FULL_PIPELINE else None,
        "batch_size": len(texts),
        "classification_timings_ms": cls_timings or None,
        "total_ms": _ms(t_run),
        "variants": variants,
    }
    (cache or RUN_CACHE).put(run_id, {"user": user_email, "texts": texts, "payload": payload})
    return payload
