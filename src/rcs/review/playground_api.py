"""
HTTP-shape adapter for the internal AI Evaluation Playground. Mirrors
rcs.review.inspector_api: translation only, no business logic and no
database access. ADMIN authentication is enforced in http_api.py before any
function here runs; the "playground enabled" gate is enforced there too.
"""

from __future__ import annotations

from typing import Any

from config import settings
from rcs.response_intelligence import playground, playground_eval
from rcs.response_intelligence.playground_store import PlaygroundStore

# Seams for tests (monkeypatched); production wiring uses the real functions.
_classifier_provider = playground.default_classifiers
_live_client_factory = None  # None -> rcs.response_generation.service.build_live_gemini_client
_retrievers = None


def _store() -> PlaygroundStore:
    return PlaygroundStore(settings.PLAYGROUND_DATA_DIR)


def handle_status() -> dict[str, Any]:
    return playground.status_payload()


def handle_run(body: dict[str, Any], user_email: str | None) -> dict[str, Any]:
    return playground.run_playground(
        body,
        classifiers=_classifier_provider(),
        live_client_factory=_live_client_factory,
        retrievers=_retrievers,
        user_email=user_email,
    )


def _variant_from_run(body: dict[str, Any], user_email: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    run = playground.RUN_CACHE.get(str(body.get("run_id") or ""))
    if run is None:
        raise playground.PlaygroundError("unknown or expired run_id; re-run the comment first")
    try:
        index = int(body.get("variant_index"))
        variant = run["payload"]["variants"][index]
        text = run["texts"][index]
    except (TypeError, ValueError, IndexError):
        raise playground.PlaygroundError("invalid variant_index") from None
    return text, variant, run["payload"]


def handle_compare(body: dict[str, Any], user_email: str | None) -> dict[str, Any]:
    _text, variant, _payload = _variant_from_run(body, user_email)
    try:
        return playground_eval.compare(body.get("expected") or {}, body.get("status"), variant)
    except ValueError as exc:
        raise playground.PlaygroundError(str(exc)) from None


def handle_save_case(body: dict[str, Any], user_email: str | None) -> dict[str, Any]:
    """Explicit save only. Predictions are taken from the server's own record
    of the run, never from client-supplied result data."""
    text, variant, payload = _variant_from_run(body, user_email)
    try:
        expected = playground_eval.validate_expected(body.get("expected") or {})
        status = playground_eval.validate_status(body.get("status"))
    except ValueError as exc:
        raise playground.PlaygroundError(str(exc)) from None
    if status == playground_eval.STATUS_VERIFIED and not expected:
        raise playground.PlaygroundError("a VERIFIED case needs at least one expected label")
    verdict = playground_eval.compare(expected, status, variant)
    record = {
        "case_id": body.get("case_id"),
        "text": text,
        "expected": expected,
        "status": status,
        "verified_by": user_email if status == playground_eval.STATUS_VERIFIED else None,
        "notes": str(body.get("notes") or "")[:1000],
        "result": {
            "mode": payload["mode"],
            "agent_mode": payload["agent_mode"],
            "actual": playground_eval.actual_values(variant),
            "variant": variant,
        },
        "verdict": verdict,
    }
    saved = _store().save_case(record, saved_by=user_email)
    return {"case": saved}


def handle_list_cases() -> dict[str, Any]:
    return {"cases": _store().list_cases()}


def handle_delete_case(body: dict[str, Any], user_email: str | None) -> dict[str, Any]:
    case_id = str(body.get("case_id") or "")
    if not case_id:
        raise playground.PlaygroundError("case_id is required")
    return {"deleted": _store().delete_case(case_id, deleted_by=user_email)}


def handle_metrics() -> dict[str, Any]:
    return playground_eval.compute_evaluation(_store().list_cases())
