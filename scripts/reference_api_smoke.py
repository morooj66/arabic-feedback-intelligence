#!/usr/bin/env python3
"""
Non-destructive smoke of a PUBLIC, read-only dictionary/lexicon list endpoint (one GET).

The endpoint is not hard-coded. Point it at any public lexicon list API with
REFERENCE_API_ENDPOINT (for example https://dictionary-platform.invalid/api/v1/public/lexicons)
and, optionally, REFERENCE_API_DOCS (a documentation URL echoed in the report).

Does not send visitor replies. Does not call private endpoints.
Does not guess undocumented comment/feedback APIs. Nothing in the application depends on it.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("REFERENCE_API_ENDPOINT", "").strip()
DOCS = os.environ.get("REFERENCE_API_DOCS", "").strip() or None


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _summarize_error_body(snippet: str) -> str:
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError:
        return "non-JSON error body (not dumped)"
    if isinstance(parsed, dict):
        return f"JSON object keys={list(parsed.keys())[:8]}; message={parsed.get('message')}"
    return f"JSON {type(parsed).__name__}"


def smoke() -> dict:
    if not ENDPOINT.lower().startswith(("https://", "http://")):
        return {
            "endpoint": None,
            "error": "Set REFERENCE_API_ENDPOINT to a public lexicon list URL (http/https) to run this smoke test.",
            "note": "No request was made.",
        }
    req = urllib.request.Request(ENDPOINT, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as resp:
            body = resp.read(4096)
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        snippet = exc.read(400).decode("utf-8", errors="replace")
        return {
            "endpoint": ENDPOINT,
            "docs": DOCS,
            "method": "GET",
            "request_parameters": {},
            "http_status": exc.code,
            "auth_header_sent": False,
            "content_type": exc.headers.get("Content-Type") if exc.headers else None,
            "schema_summary": _summarize_error_body(snippet),
            "note": "No visitor/comment/reply endpoints were called.",
        }
    except Exception as exc:
        return {
            "endpoint": ENDPOINT,
            "docs": DOCS,
            "method": "GET",
            "request_parameters": {},
            "http_status": None,
            "error": f"{type(exc).__name__}: {exc}",
            "auth_header_sent": False,
        }

    summary = "opaque bytes"
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(parsed, list):
            first = parsed[0] if parsed and isinstance(parsed[0], dict) else {}
            summary = f"JSON array, length={len(parsed)}, first_keys={list(first.keys())[:8]}"
        elif isinstance(parsed, dict):
            summary = f"JSON object, keys={list(parsed.keys())[:12]}"
        else:
            summary = f"JSON {type(parsed).__name__}"
    except json.JSONDecodeError:
        summary = f"non-JSON, content_type={content_type}"

    return {
        "endpoint": ENDPOINT,
        "docs": DOCS,
        "method": "GET",
        "request_parameters": {},
        "http_status": status,
        "auth_header_sent": False,
        "content_type": content_type,
        "schema_summary": summary,
        "note": "No visitor/comment/reply endpoints were called.",
    }


if __name__ == "__main__":
    print(json.dumps(smoke(), ensure_ascii=False, indent=2))
