"""Turns demo mode on. Called from scripts/serve_reviewer_api.py (and the demo initialiser) only after `guard.problems()` returned nothing.

What it changes, all by replacing module attributes at start-up (no production file contains demo logic):
  * classification  -> rcs.demo.classifier (rule based, labelled DEMO_RULES_SYNTHETIC_ONLY)
  * response draft  -> rcs.demo.provider.DemoProvider (local, deterministic; the real LLM client builder is replaced, so no key is ever read)
  * workbook upload -> synthetic-marker + size/row policy (rcs.demo.workbook)
  * account requests, password reset, Microsoft sign-in -> unavailable (invite-only: accounts are created by the operator with scripts/demo/init_demo_environment.py)
  * admin inspector entries that need the frozen models -> "not available in demo"
  * network: every Python-level connection to a non-loopback address is refused (database drivers talk through their own C library and are unaffected), and the ML /
    LLM client libraries can no longer be imported lazily.
Authentication and authorization are NOT touched: the same login, roles, reviewer isolation and CSRF/origin checks apply."""

from __future__ import annotations

import importlib.abc
import os
import socket
import sys

from rcs.demo import classifier, guard, provider, state, workbook

_BLOCKED_IMPORTS = ("torch", "transformers", "sklearn", "joblib", "huggingface_hub", "google.genai", "google.generativeai", "openai")


class DemoModeRefused(SystemExit):
    pass


def preflight_or_exit(env, database_url, *, dev_db_flags_used: bool = False) -> None:
    found = guard.problems(env, database_url, dev_db_flags_used=dev_db_flags_used)
    if found:
        raise DemoModeRefused("FATAL: demo mode refused to start:\n  - " + "\n  - ".join(found))


def _install_socket_guard() -> None:
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    allowed = {"127.0.0.1", "::1", "localhost"}

    def _check(address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if isinstance(address, str) and address.startswith("/"):
            return                                                    # unix domain socket
        if host not in allowed:
            raise ConnectionRefusedError("demo mode: outbound network connections are disabled")

    def connect(self, address):
        _check(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        _check(address)
        return real_connect_ex(self, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


class _BlockMlImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == b or fullname.startswith(b + ".") for b in _BLOCKED_IMPORTS):
            raise ImportError(f"{fullname} is disabled in demo mode")
        return None


def activate() -> None:
    import rcs.review.account_requests as account_requests
    import rcs.review.excel_review_flow as erf
    import rcs.review.http_api as http_api
    from config import settings

    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    erf.predict_intent = classifier.predict_intent
    erf.predict_topic = classifier.predict_topic
    erf.predict_sentiment = classifier.predict_sentiment
    erf.provenance = classifier.provenance
    erf.apply_optional_topic_layer = classifier.apply_optional_topic_layer
    erf.with_layer_provenance = classifier.with_layer_provenance
    erf.build_live_gemini_client = lambda: provider.DemoProvider()
    http_api.validate_workbook = workbook.wrap_validator(http_api.validate_workbook)
    settings.EXCEL_UPLOAD_MAX_BYTES = workbook.MAX_BYTES
    account_requests.ENABLED = False

    def unavailable(_body):
        return {"available": False, "detail": "غير متاح في وضع العرض التجريبي"}

    for name in ("handle_inspector_classification", "handle_inspector_full_run"):
        if hasattr(http_api, name):
            setattr(http_api, name, unavailable)
    _install_socket_guard()
    sys.meta_path.insert(0, _BlockMlImports())
    state._set_active()
