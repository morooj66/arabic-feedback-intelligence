"""LIVE Gemini reply-quality pilot (explicitly authorised by the project owner for the 12 comments in gemini_sample_v2.json ONLY).
Production generation path (run_agent_pipeline: eligibility -> prompt (current version) -> outbound privacy boundary -> Gemini -> validation gates -> at most ONE bounded repair),
empty official knowledge (none exists), no database, nothing sent to visitors. Transport retries are disabled so each comment makes one call (plus at most one repair call).
Refuses to run without --authorized and aborts on any privacy-boundary failure."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for line in open(ROOT / ".env", encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402
from rcs.response_generation.outbound_privacy import OutboundPrivacyError, guard_prompt  # noqa: E402
from rcs.response_generation.service import GenerationRequest, build_live_gemini_client  # noqa: E402
from rcs.response_generation.validate import collect_reply_findings, validate_draft  # noqa: E402
from rcs.response_intelligence.orchestrator import run_agent_pipeline  # noqa: E402
from rcs.response_intelligence.retrievers import default_approved_response_memory_retriever, default_official_knowledge_retriever  # noqa: E402
from rcs.routing.engine import determine_routing  # noqa: E402

SAMPLE = ROOT / "data/evaluation/reply_quality/gemini_sample_v2.json"
OUT = ROOT / "data/evaluation/reply_quality"
PRED = ROOT / "data/evaluation/classification_v2_final_validation/predictions_v2.json"
UNIVERSE = ROOT / "data/evaluation/classification_v2_final_validation/universe.jsonl"


class PrivacyStop(RuntimeError):
    pass


class RecordingClient:
    """Sits at the transport seam: re-asserts the privacy boundary on the exact outbound payload, records payload / raw response / timing."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def generate(self, prompt: str, response_schema: dict) -> str:
        try:
            if guard_prompt(prompt) != prompt:
                raise PrivacyStop("payload changed under a second privacy pass")
        except OutboundPrivacyError as exc:
            raise PrivacyStop(f"privacy boundary failed: {exc.reason}") from exc
        t0 = time.perf_counter()
        try:
            raw = self.inner.generate(prompt, response_schema)
            err = None
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for the production error path
            raw, err = None, f"{type(exc).__name__}: {str(exc)[:300]}"
            self.calls.append({"payload_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "payload": prompt, "raw_response": None, "error": err, "seconds": round(time.perf_counter() - t0, 2)})
            raise
        self.calls.append({"payload_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "payload": prompt, "raw_response": raw, "error": None, "seconds": round(time.perf_counter() - t0, 2)})
        return raw


def main():
    dry = "--dry-run" in sys.argv
    if "--authorized" not in sys.argv and not dry:
        raise SystemExit("refusing to call Gemini without --authorized")
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert len(sample) == 12, "the approved sample has exactly 12 comments"
    sample_sha = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    rows = {json.loads(l)["comment_id"]: i for i, l in enumerate(open(UNIVERSE, encoding="utf-8"))}
    pred = json.load(open(PRED))
    settings.GEMINI_MAX_RETRIES = 0                       # one request per comment; the only extra call allowed is the bounded repair
    out_path = OUT / (f"dry_run_{settings.RESPONSE_GENERATION_PROMPT_VERSION}.json" if dry else f"live_pilot_results_{settings.RESPONSE_GENERATION_PROMPT_VERSION}.json")
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path.name}")
    if dry:                                                # no network: a local stub returns a fixed draft; payloads still pass the real privacy boundary
        class _Stub:
            def generate(self, prompt, response_schema):
                return json.dumps({"suggested_response": os.environ.get("DRY_RUN_DRAFT", "نشكركم على اقتراحكم.")}, ensure_ascii=False)
        base = RecordingClient(_Stub())
    else:
        base = RecordingClient(build_live_gemini_client())
    ok_r, arm_r = default_official_knowledge_retriever(), default_approved_response_memory_retriever()
    results, stop = [], None
    for s in sample:
        i = rows[s["comment_id"]]
        intent, conf = pred["intent"][i]
        topic, sentiment = pred["topic"][i][0], pred["sentiment"][i][0]
        assert (intent, topic, sentiment) == (s["intent"], s["topic"], s["sentiment"]), "prediction drift - refusing to run"
        route = determine_routing(primary_intent=intent, topic_subtype=topic, subcategory=None, secondary_intent=None, is_content_meaningful=True)
        req = GenerationRequest(comment_id=s["comment_id"], comment_text=s["comment_sanitized"], primary_intent=intent, topic_subtype=topic, sentiment=sentiment,
                                confidence=float(conf), needs_review=float(conf) < settings.RESPONSE_LOW_CONFIDENCE_THRESHOLD, classification_routing_decision=route.routing_decision)
        base.calls = []
        t0 = time.perf_counter()
        try:
            out = run_agent_pipeline(req, client=base, official_knowledge_retriever=ok_r, approved_response_retriever=arm_r, audit_repository=None)
            gr, meta = out.generation_result, out.metadata
            rec = {"generation_status": gr.generation_status, "routing_decision": gr.routing_decision, "reason_code": gr.reason_code, "error": gr.error,
                   "suggested_response": gr.suggested_response, "prompt_version": gr.prompt_version, "model": gr.model_name,
                   "refinement_attempted": bool(meta.refinement_attempted), "logical_generation_count": getattr(meta, "logical_generation_count", None),
                   "grounding_mode": getattr(meta, "grounding_mode", None)}
        except PrivacyStop as exc:
            stop = str(exc); rec = {"generation_status": "ABORTED_PRIVACY", "error": stop}
        except Exception as exc:  # noqa: BLE001
            rec = {"generation_status": "EXCEPTION", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        drafts = []
        for c in base.calls:
            try:
                drafts.append(json.loads(c["raw_response"])["suggested_response"] if c["raw_response"] else None)
            except Exception:  # noqa: BLE001
                drafts.append(None)
        final = rec.get("suggested_response")
        independent = None
        if final:
            ok, why = validate_draft(final, comment_text=s["comment_sanitized"], trusted_grounding_text=None)
            f = collect_reply_findings(final, comment_text=s["comment_sanitized"])
            independent = {"passes_current_gates": ok, "first_failure": why, "blocking_findings": list(f.blocking), "attestable_findings": list(f.attestable)}
        results.append({"n": s["n"], "intended_category": s["intended_category"], "comment_sanitized": s["comment_sanitized"], "predicted": {"intent": intent, "topic": topic, "sentiment": sentiment, "intent_confidence": round(float(conf), 3)},
                        "eligibility_reason": s["eligibility_reason"], "classification_routing_decision": route.routing_decision, "pipeline": rec,
                        "gemini_calls": [{"payload_sha256": c["payload_sha256"], "seconds": c["seconds"], "error": c["error"], "draft_returned": d} for c, d in zip(base.calls, drafts)],
                        "payloads": [c["payload"] for c in base.calls], "independent_revalidation_of_final_draft": independent, "seconds_total": round(time.perf_counter() - t0, 2)})
        print(s["n"], rec.get("generation_status"), rec.get("reason_code"), "calls", len(base.calls), flush=True)
        if stop:
            break
    meta = {"sample_file_sha256": sample_sha, "prompt_version": settings.RESPONSE_GENERATION_PROMPT_VERSION, "model": settings.RESPONSE_GENERATION_MODEL_NAME, "transport_retries": 0,
            "total_gemini_requests": sum(len(r["gemini_calls"]) for r in results), "stopped_on_privacy_failure": stop, "database_writes": 0, "sent_to_visitors": 0}
    meta["dry_run"] = dry
    out_path.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
