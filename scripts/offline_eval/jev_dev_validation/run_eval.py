"""Guarded Jev executor. DRY-RUN BY DEFAULT: with no flags it builds and validates every
payload, prints the exact request count and cost estimate, and sends NOTHING.

Real calls require ALL of: --execute, --approve-sha256 <hash of the sanitized inputs you
inspected>, --max-requests, --max-usd, and OPENROUTER_API_KEY in the environment. Calls are
sequential, never retried, and stop at the first provider/validation failure; predictions are
written incrementally so a partial run is never lost or silently completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_common import OUT_DIR, ROOT  # noqa: E402
from config import settings  # noqa: E402
from jev_protocol import JevProtocolError, build_request, estimate_cost, validate_response  # noqa: E402
from rcs.classification.jev_topic import JevTopicError  # noqa: E402
from jev_redaction import Quarantined, redact_strict  # noqa: E402

ALLOWED_PAYLOAD_KEYS = {"model", "state", "questions"}


def certifi_https_post(url: str, headers: dict, body: bytes, timeout: float) -> dict:
    """Exactly one HTTPS POST, no redirects, no retries -- the production adapter's transport, but with the
    certifi CA bundle (this Python does not inherit the macOS trust store; same fix as mongo_client).
    Certificate verification stays ON."""
    import http.client
    import ssl
    from urllib.parse import urlsplit

    import certifi

    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise JevTopicError("endpoint must be HTTPS")
    conn = http.client.HTTPSConnection(parts.hostname, port=parts.port, timeout=timeout,
                                       context=ssl.create_default_context(cafile=certifi.where()))
    try:
        conn.request("POST", (parts.path or "/") + (f"?{parts.query}" if parts.query else ""), body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read(2_000_001)
        status = resp.status
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise JevTopicError(f"request failed without retry: {type(exc).__name__}") from None
    finally:
        conn.close()
    if len(raw) > 2_000_000:
        raise JevTopicError("response exceeded the size limit")
    if not 200 <= status < 300:
        raise JevTopicError(f"provider returned HTTP {status}")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JevTopicError("provider returned invalid JSON") from None
    if not isinstance(parsed, dict):
        raise JevTopicError("provider returned a non-object response")
    return parsed


def key_from_dotenv(path: Path = ROOT / ".env") -> str | None:
    """Read ONLY the OPENROUTER_API_KEY line from .env (no other variable is loaded, nothing is exported
    or printed). The process environment takes precedence over the file."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip("'\"") or None
    return None


def load_inputs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_payload_clean(payload: dict, sanitized_text: str) -> None:
    """Defence in depth: the payload is exactly model + sanitized text + questions, and the text
    is a fixed point of the strict redactor (re-redacting changes nothing)."""
    if set(payload) != ALLOWED_PAYLOAD_KEYS or set(payload["state"]) != {"visitor_comment_text"}:
        raise SystemExit("payload contains unexpected fields; aborting before any request")
    if payload["state"]["visitor_comment_text"] != sanitized_text:
        raise SystemExit("payload text differs from the inspected sanitized text")
    try:
        again = redact_strict(sanitized_text)
    except Quarantined as q:
        raise SystemExit(f"payload failed the redaction re-check ({q.reason}); aborting") from None
    if again.text != sanitized_text:
        raise SystemExit("payload text is not a redaction fixed point; aborting")


def run(args, *, transport=certifi_https_post, env=os.environ) -> dict:
    inputs_path = Path(args.inputs)
    digest = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    rows = load_inputs(inputs_path)[getattr(args, "skip", 0) :]
    rows = rows[: args.limit] if args.limit else rows
    payloads = [build_request(r["text"], getattr(args, "criteria", "v1")) for r in rows]
    for r, p in zip(rows, payloads):
        assert_payload_clean(p, r["text"])
    cost = estimate_cost(payloads)
    summary = {**cost, "sanitized_inputs_sha256": digest, "mode": "execute" if args.execute else "dry-run",
               "calls_executed": 0}
    if not args.execute:
        return summary

    if args.approve_sha256 != digest:
        raise SystemExit("--approve-sha256 does not match the sanitized inputs on disk; re-inspect and re-approve")
    if not args.max_requests or not args.max_usd:
        raise SystemExit("--max-requests and --max-usd are required with --execute")
    if len(rows) > args.max_requests:
        raise SystemExit(f"{len(rows)} requests exceed --max-requests {args.max_requests}")
    if cost["upper_bound_usd"] > args.max_usd:
        raise SystemExit(f"estimated upper bound ${cost['upper_bound_usd']} exceeds --max-usd {args.max_usd}")
    api_key = env.get("OPENROUTER_API_KEY") or (key_from_dotenv() if env is os.environ else None)
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    out_path = Path(args.predictions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds: dict[str, dict] = {}
    with open(out_path, "w", encoding="utf-8") as sink:
        for r, p in zip(rows, payloads):
            body = json.dumps(p, ensure_ascii=False).encode("utf-8")
            headers = {
                "Authorization": "Bearer " + api_key, "Accept": "application/json",
                "Content-Type": "application/json", "Content-Length": str(len(body)),
                "User-Agent": "rcs-jev-dev-validation/1.0.0",
            }
            try:
                raw = transport(settings.JEV_TOPIC_ENDPOINT, headers, body, settings.JEV_TOPIC_REQUEST_TIMEOUT_SECONDS)
                answers = validate_response(raw)
            except (JevTopicError, JevProtocolError) as exc:
                summary.update(stopped_at=r["eval_id"], error=str(exc))
                break
            summary["calls_executed"] += 1
            rec = {"eval_id": r["eval_id"], **{k: v[0] for k, v in answers.items()},
                   "confidence": {k: v[1] for k, v in answers.items()},
                   # provider-reported accounting, if any (never request headers or the key)
                   "provider_meta": {k: raw[k] for k in ("id", "usage", "cost", "latency_ms") if k in raw}}
            preds[r["eval_id"]] = rec
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", default=str(OUT_DIR / "sanitized_inputs.jsonl"))
    ap.add_argument("--predictions-out", default=str(OUT_DIR / "jev_predictions.jsonl"))
    ap.add_argument("--criteria", choices=("v1", "v2"), default="v1", help="v2 = prepared candidate, requires its own approval")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N rows (e.g. 5 after the canary)")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (use 5 for the canary run)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--approve-sha256", default="")
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.0)
    print(json.dumps(run(ap.parse_args()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
