"""Opt-in rg-v14 live evaluation through the authenticated Admin Playground.

This file is preparation only. It performs no work when imported. Running it
still fails closed unless RUN_RG_V14_LIVE_EVAL=1 is set and the operator types
the exact interactive confirmation phrase. It never calls the Playground case
save endpoint and writes future results only under /private/tmp.

Before a future run, restart the local backend so it has loaded rg-v14 and
verify that it still points to the isolated temporary SQLite database.
"""

from __future__ import annotations

import datetime as dt
import getpass
import json
import os
import sys
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener


BASE_URL = os.environ.get("RCS_LOCAL_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
CASES_PATH = Path(__file__).with_name("rg_v14_live_cases.json")
CONFIRMATION = "RUN-RG-V14-18-LIVE-CASES"
RESULTS_DIR = Path("/private/tmp")

EXPECTED_INTENTS = {"THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM"}
EXPECTED_TOPICS = {
    "WORD_SUBMISSION",
    "ENTRY_CORRECTION",
    "DICTIONARY_SUGGESTION",
    "TECHNICAL_ISSUE",
    "WORD_OF_YEAR_NOMINATION",
    "OFF_TOPIC_PERSONAL",
    "GENERAL",
}
EXPECTED_SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


class LocalApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.cookies = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookies))

    def request(self, method: str, path: str, payload=None, *, timeout: int = 300):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1000]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _load_and_validate_cases() -> list[dict]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if payload.get("prompt_version") != "rg-v14":
        raise RuntimeError("Case manifest is not marked for rg-v14.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 18:
        raise RuntimeError("The rg-v14 live manifest must contain exactly 18 cases.")

    ids = [case.get("case_id") for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Every evaluation case_id must be unique.")

    intents = {case["verified_labels"]["intent"] for case in cases}
    topics = {case["verified_labels"]["topic"] for case in cases}
    sentiments = {case["verified_labels"]["sentiment"] for case in cases}
    if intents != EXPECTED_INTENTS:
        raise RuntimeError(f"Intent coverage mismatch: {sorted(intents)}")
    if topics != EXPECTED_TOPICS:
        raise RuntimeError(f"Topic coverage mismatch: {sorted(topics)}")
    if sentiments != EXPECTED_SENTIMENTS:
        raise RuntimeError(f"Sentiment coverage mismatch: {sorted(sentiments)}")
    return cases


def _playground_payload(case: dict) -> dict:
    return {
        "variants": [case["comment"]],
        "mode": "full_pipeline",
        "agent_mode": "live",
        "live_confirmed": True,
        "force_classify_prefiltered": False,
        "label_source": "verified",
        "fixed_labels": [
            {
                "verified": True,
                "labels": case["verified_labels"],
            }
        ],
    }


def _result_record(case: dict, result: dict) -> dict:
    variants = result.get("variants") or []
    variant = variants[0] if len(variants) == 1 else {}
    agent = (variant.get("agent_runs") or {}).get("verified") or {}
    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "comment": case["comment"],
        "verified_labels": case["verified_labels"],
        "quality_expectation": case["quality_expectation"],
        "url_expectation": case["url_expectation"],
        "run_id": result.get("run_id"),
        "classification_routing": agent.get("classification_routing"),
        "decision": agent.get("decision"),
        "response": agent.get("response"),
        "tool_calls": agent.get("tool_calls"),
        "drafts": agent.get("drafts"),
        "llm_calls": agent.get("llm_calls"),
        "error": agent.get("error"),
        "fallback": agent.get("fallback"),
        "variant_errors": variant.get("errors"),
    }


def main() -> int:
    if os.environ.get("RUN_RG_V14_LIVE_EVAL") != "1":
        print(
            "STOPPED: live evaluation is disabled. After explicit approval, set "
            "RUN_RG_V14_LIVE_EVAL=1 and run this script interactively."
        )
        return 0

    cases = _load_and_validate_cases()
    client = LocalApiClient(BASE_URL)
    logged_in = False

    print("This will submit exactly 18 verified-only Playground run requests.")
    print("Gemini may make internal retry/refinement calls; this runner never retries a Playground request.")
    print("Confirm the backend was restarted on rg-v14 and still uses isolated temporary SQLite.")
    confirmation = input(f"Type {CONFIRMATION} to continue: ")
    if confirmation != CONFIRMATION:
        print("Aborted before authentication or any Playground run.")
        return 1

    try:
        email = input("Local ADMIN email: ").strip()
        password = getpass.getpass("Local ADMIN password: ")
        session = client.request(
            "POST", "/api/login", {"email": email, "password": password}, timeout=30
        )
        password = ""
        logged_in = True
        if session.get("role") != "ADMIN":
            raise RuntimeError("Authenticated account is not an ADMIN.")

        status = client.request("GET", "/api/admin/playground/status", timeout=30)
        if status.get("enabled") is not True:
            raise RuntimeError(f"Playground disabled: {status.get('disabled_reason')}")
        if status.get("live_configured") is not True:
            raise RuntimeError("Gemini is not configured on the local backend.")

        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"rcs_rg_v14_live_eval_{timestamp}.jsonl"

        with output_path.open("x", encoding="utf-8") as output:
            for index, case in enumerate(cases, start=1):
                print(f"[{index:02d}/18] {case['case_id']}")
                # Exactly one run request per case. Never automatically retry this call:
                # after a client timeout, the backend may still have completed it.
                result = client.request(
                    "POST",
                    "/api/admin/playground/run",
                    _playground_payload(case),
                    timeout=300,
                )
                output.write(json.dumps(_result_record(case, result), ensure_ascii=False) + "\n")
                output.flush()

        print(f"Completed. Results: {output_path}")
        return 0
    finally:
        if logged_in:
            try:
                client.request("POST", "/api/logout", {}, timeout=10)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())

