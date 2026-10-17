"""E2E verification server: the REAL HTTP app + REAL frozen classifiers on a scratch SQLite DB, with Gemini replaced by a deterministic local fake and every non-loopback
network connection blocked (and counted). Verification tooling only: it changes no product code and no default; nothing is sent to visitors.

  python scripts/e2e/serve_e2e_mock.py --workdir <scratch dir> --port 8791 [--context-signals]

Creates (idempotently) one ADMIN and two REVIEWERs with throw-away passwords written to <workdir>/e2e_users.json. Counters are written to <workdir>/e2e_counters.json."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", type=Path, required=True)
ap.add_argument("--port", type=int, default=8791)
ap.add_argument("--context-signals", action="store_true")
ap.add_argument("--echo-gemini", action="store_true", help="mock Gemini that ties each reply to the comment it saw (\"رقم N\"), records which comments it was called for, "
                "never sees phone numbers / e-mail addresses, fails on the marker فشل-التوليد and answers non-JSON on the marker رد-معطوب (full-workflow verification)")
ap.add_argument("--playground", action="store_true", help="enable the admin Playground (mock Gemini, data dir inside the workdir)")
ap.add_argument("--fake-email-base", default="", help="enable password reset with the in-memory FAKE sender; links point at this frontend origin; outbox mirrored to <workdir>/outbox.json")
args = ap.parse_args()
args.workdir.mkdir(parents=True, exist_ok=True)

for k in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL", "MONGODB_URI", "HF_TOKEN"):
    os.environ[k] = ""
os.environ.update({"PERSISTENCE_BACKEND": "postgres", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "RCS_ENV": "development",
                   "RCS_CONTEXT_SIGNALS": "1" if args.context_signals else "0", "RCS_PLAYGROUND_ENABLED": "1" if args.playground else "", "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

COUNTERS = {"mock_gemini_calls": 0, "blocked_external_connections": 0, "external_hosts_attempted": []}
COUNTER_FILE = args.workdir / "e2e_counters.json"


def _flush():
    COUNTER_FILE.write_text(json.dumps(COUNTERS, ensure_ascii=False))


_real_connect = socket.socket.connect


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    if self.family in (socket.AF_INET, socket.AF_INET6) and host not in ("127.0.0.1", "::1", "localhost"):
        COUNTERS["blocked_external_connections"] += 1
        COUNTERS["external_hosts_attempted"].append(str(host))
        _flush()
        raise ConnectionRefusedError("E2E harness: external network is blocked")
    return _real_connect(self, address)


socket.socket.connect = _guarded_connect

from config import settings  # noqa: E402

settings.EXCEL_UPLOAD_DIR = args.workdir / "uploads"
settings.EXCEL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class FakeGemini:
    """Deterministic, safe, local. A plain acknowledgement that passes the validators; counts calls.
    With --echo-gemini it additionally traces every call by comment number (see the module help)."""

    def generate(self, prompt: str, response_schema: dict) -> str:
        COUNTERS["mock_gemini_calls"] += 1
        if getattr(args, "echo_gemini", False):
            import re as _re

            m = _re.search(r"--- UNTRUSTED VISITOR COMMENT[^\n]*---\n(.*?)\n--- END UNTRUSTED VISITOR COMMENT", prompt, _re.S)
            comment = m.group(1) if m else ""
            num = _re.search(r"رقم\s*(\d+)", comment)
            COUNTERS.setdefault("called_numbers", []).append(num.group(1) if num else None)
            if "0500000013" in prompt or "example.invalid" in prompt:
                COUNTERS["prompts_containing_pii"] = COUNTERS.get("prompts_containing_pii", 0) + 1
            _flush()
            if "فشل-التوليد" in comment:
                raise RuntimeError("simulated provider failure")
            if "رد-معطوب" in comment:
                return "هذا ليس JSON"
            return json.dumps({"suggested_response": f"شكرًا لكم على ملاحظتكم رقم {num.group(1)}." if num else "شكرًا لكم على ملاحظتكم."}, ensure_ascii=False)
        _flush()
        return json.dumps({"suggested_response": "شكرًا لكم على ملاحظتكم."}, ensure_ascii=False)


import rcs.review.excel_review_flow as erf  # noqa: E402
import rcs.review.inspector_api as insp  # noqa: E402

erf.build_live_gemini_client = lambda: FakeGemini()
if args.playground:
    import rcs.review.playground_api as pg_api

    settings.PLAYGROUND_DATA_DIR = args.workdir / "playground"
    pg_api._live_client_factory = lambda: FakeGemini()
insp.build_live_gemini_client = lambda: FakeGemini()

from rcs.review.email_sender import FakeEmailSender, set_email_sender  # noqa: E402

if args.fake_email_base:
    class _FileFake(FakeEmailSender):
        def send(self, message):
            super().send(message)
            (args.workdir / "outbox.json").write_text(json.dumps([m.__dict__ for m in self.outbox], ensure_ascii=False))

    settings.PASSWORD_RESET_LINK_BASE = args.fake_email_base.rstrip("/")
    set_email_sender(_FileFake())

from rcs.review.auth import ADMIN, REVIEWER, create_user  # noqa: E402
from rcs.review.http_api import serve  # noqa: E402
from rcs.review.persistence import build_repository_bundle, resolve_bundle  # noqa: E402
from rcs.review.repository import get_engine, init_db  # noqa: E402

DB = args.workdir / "e2e.sqlite"
bundle, engine = build_repository_bundle(engine_factory=lambda: get_engine(DB))
init_db(engine)
users_file = args.workdir / "e2e_users.json"
if not users_file.exists():
    users = {}
    for key, role in (("admin", ADMIN), ("reviewer_a", REVIEWER), ("reviewer_b", REVIEWER)):
        pw = secrets.token_urlsafe(12)
        u = create_user(bundle, email=f"{key}@e2e.invalid", password=pw, display_name=f"E2E {key}", role=role)
        users[key] = {"email": f"{key}@e2e.invalid", "password": pw, "user_id": u.user_id, "role": role}
    users_file.write_text(json.dumps(users))
server = serve(bundle, host="127.0.0.1", port=args.port)
print(f"E2E server on http://127.0.0.1:{args.port} db={DB} context_signals={args.context_signals}", flush=True)
server.serve_forever()
