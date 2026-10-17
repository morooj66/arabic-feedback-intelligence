"""Pre-deployment hardening: database configuration (fail closed + driver), request-body limits, media-type and Origin (CSRF) checks, production startup guards,
.gitignore protection and the server-build configuration. Local only - no network, no real database."""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from config import settings
from rcs.review import auth
from rcs.review.auth import create_user
from rcs.review.http_api import ReviewerAPIHandler, serve_in_thread
from rcs.review.repository import get_engine, init_db

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- database configuration
@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", None)
    monkeypatch.delenv("RCS_ENV", raising=False)
    return monkeypatch


@pytest.mark.parametrize("given,expected", [
    ("postgresql://u:p@h:5432/d?sslmode=require", "postgresql+psycopg://u:p@h:5432/d?sslmode=require"),
    ("postgres://u@h/d", "postgresql+psycopg://u@h/d"),
    ("POSTGRESQL://u@h/d", "postgresql+psycopg://u@h/d"),
    ("postgresql+psycopg://u@h/d", "postgresql+psycopg://u@h/d"),
    ("sqlite:///x.db", "sqlite:///x.db"),
])
def test_postgres_urls_are_pinned_to_the_installed_psycopg_v3_driver(given, expected):
    assert settings.normalize_database_url(given) == expected


def test_the_installed_driver_is_psycopg_v3_not_psycopg2():
    import importlib.util
    assert importlib.util.find_spec("psycopg") is not None
    assert importlib.util.find_spec("psycopg2") is None                                   # a bare postgresql:// URL would otherwise have selected a driver we do not ship


def test_production_without_database_url_is_a_configuration_error(env):
    env.setenv("RCS_ENV", "production")
    with pytest.raises(settings.DatabaseConfigError, match="DATABASE_URL is required"):
        settings.resolve_database_url(Path("/tmp/should-not-be-used.sqlite"))
    env.setattr(settings, "DATABASE_URL", "   ")
    with pytest.raises(settings.DatabaseConfigError):
        settings.resolve_database_url(Path("/tmp/x.sqlite"))


@pytest.mark.parametrize("bad", ["sqlite:///data/x.sqlite", "mysql://u@h/d", "mongodb://h/d", "not a url"])
def test_production_refuses_non_postgres_urls(env, bad):
    env.setenv("RCS_ENV", "production"); env.setattr(settings, "DATABASE_URL", bad)
    with pytest.raises(settings.DatabaseConfigError, match="PostgreSQL"):
        settings.resolve_database_url(Path("/tmp/x.sqlite"))


def test_production_accepts_and_normalises_a_postgres_url(env):
    env.setenv("RCS_ENV", "production"); env.setattr(settings, "DATABASE_URL", "postgresql://u:p@host.example/d")
    assert settings.resolve_database_url(Path("/tmp/x.sqlite")) == "postgresql+psycopg://u:p@host.example/d"


def test_local_development_keeps_the_sqlite_fallback(env, tmp_path):
    assert settings.resolve_database_url(tmp_path / "dev.sqlite") == f"sqlite:///{tmp_path / 'dev.sqlite'}"
    env.setenv("RCS_ENV", "development")
    assert settings.resolve_database_url(tmp_path / "dev.sqlite").startswith("sqlite:///")
    env.setattr(settings, "DATABASE_URL", "postgresql://u@h/d")                                # dev may still point at Postgres explicitly
    assert settings.resolve_database_url(tmp_path / "dev.sqlite") == "postgresql+psycopg://u@h/d"


def test_sqlalchemy_resolves_the_normalised_url_to_the_installed_driver():
    from sqlalchemy.engine import make_url
    url = make_url(settings.normalize_database_url("postgresql://u:p@h/d"))
    assert url.get_backend_name() == "postgresql" and url.get_driver_name() == "psycopg"


def _serve(*args, **env):
    e = {**os.environ, "DATABASE_URL": "", "GEMINI_API_KEY": "", "PERSISTENCE_BACKEND": "postgres", **env}
    return subprocess.run([sys.executable, "scripts/serve_reviewer_api.py", "--port", "8797", *args], cwd=ROOT, env=e, capture_output=True, text=True, timeout=90)


def test_the_server_entrypoint_refuses_to_start_in_production_without_a_database():
    p = _serve(RCS_ENV="production")
    assert p.returncode == 1 and "DATABASE_URL is required" in (p.stderr + p.stdout)


@pytest.mark.parametrize("flag", [["--seed"], ["--live-seed"], ["--db", "/tmp/rcs-x.sqlite"], ["--production-db"]])
def test_the_server_entrypoint_refuses_development_only_selectors_in_production(flag):
    p = _serve(*flag, RCS_ENV="production", DATABASE_URL="postgresql://u:p@127.0.0.1:1/d")
    assert p.returncode == 1 and "development-only" in (p.stderr + p.stdout)


# ---------------------------------------------------------------- HTTP: bodies, media types, origins
@pytest.fixture()
def server(tmp_path, monkeypatch):
    auth._rate_limit_attempts.clear()
    engine = get_engine(tmp_path / "h.sqlite"); init_db(engine)
    create_user(engine, email="a@x.org", password="right-password-1", display_name="A", role="ADMIN")
    srv, base = serve_in_thread(engine)
    host, port = base.replace("http://", "").split(":")
    yield types.SimpleNamespace(host=host, port=int(port), base=base)
    srv.shutdown()
    auth._rate_limit_attempts.clear()


import types  # noqa: E402


def raw(server, method, path, *, body=b"", headers=None, declare_length=None):
    c = http.client.HTTPConnection(server.host, server.port, timeout=10)
    h = dict(headers or {})
    c.putrequest(method, path)
    for k, v in h.items():
        c.putheader(k, v)
    c.putheader("Content-Length", str(len(body) if declare_length is None else declare_length))
    c.endheaders()
    if body:
        c.send(body)
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data or b"{}"), r
    except ValueError:
        return r.status, {}, r


JSON = {"Content-Type": "application/json"}
LOGIN = json.dumps({"email": "a@x.org", "password": "wrong"}).encode()


def test_body_limit_is_enforced_before_the_body_is_read(server):
    status, body, _ = raw(server, "POST", "/api/login", headers=JSON, declare_length=settings.MAX_JSON_BODY_BYTES + 1)
    assert (status, body) == (413, {"error": "payload_too_large"})


def test_a_body_at_the_limit_is_still_parsed(server):
    pad = json.dumps({"email": "a@x.org", "password": "x" * 1000}).encode()
    assert raw(server, "POST", "/api/login", body=pad, headers=JSON)[0] == 401              # parsed and judged, not rejected for size


@pytest.mark.parametrize("length", ["abc", "-5", "1.5", "0x10"])
def test_a_malformed_content_length_is_a_400(server, length):
    c = http.client.HTTPConnection(server.host, server.port, timeout=10)
    c.putrequest("POST", "/api/login"); c.putheader("Content-Type", "application/json"); c.putheader("Content-Length", length); c.endheaders()
    r = c.getresponse(); data = json.loads(r.read() or b"{}"); c.close()
    assert r.status == 400 and data.get("error") in ("invalid_content_length",)


def test_invalid_json_is_a_400_not_a_500(server):
    for bad in (b"{not json", b"\xff\xfe\x00", b"["):
        status, body, _ = raw(server, "POST", "/api/login", body=bad, headers=JSON)
        assert (status, body) == (400, {"error": "invalid_json"})


@pytest.mark.parametrize("ctype", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", "text/html", "application/octet-stream"])
def test_cross_site_form_content_types_are_refused_on_json_endpoints(server, ctype):
    for path in ("/api/login", "/api/admin/assign", "/api/logout", "/api/password-reset/request", "/api/items/abc/send-as-is"):
        status, body, _ = raw(server, "POST", path, body=LOGIN, headers={"Content-Type": ctype})
        assert (status, body) == (415, {"error": "unsupported_media_type"}), (path, ctype)


def test_json_content_type_with_charset_is_accepted(server):
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={"Content-Type": "application/json; charset=utf-8"})[0] == 401


def test_a_body_less_post_without_content_type_is_not_a_media_type_error(server):
    assert raw(server, "POST", "/api/logout")[0] == 200


def test_the_excel_upload_accepts_only_binary_content_types(server):
    for ctype in ("text/plain", "multipart/form-data; boundary=x", "application/json", "application/x-www-form-urlencoded"):
        assert raw(server, "POST", "/api/excel/upload", body=b"PK", headers={"Content-Type": ctype, "X-Filename": "a.xlsx"})[0] == 415, ctype
    for ctype in ("application/octet-stream", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
        assert raw(server, "POST", "/api/excel/upload", body=b"PK", headers={"Content-Type": ctype, "X-Filename": "a.xlsx"})[0] in (401, 403)   # reached auth


def test_cors_preflight_allows_the_upload_header_for_an_allowed_origin_only(server, monkeypatch):
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", ("https://app.test",))
    c = http.client.HTTPConnection(server.host, server.port, timeout=10)
    c.request("OPTIONS", "/api/excel/upload", headers={"Origin": "https://app.test", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type, x-filename"})
    r = c.getresponse(); r.read(); c.close()
    assert r.status == 204 and "X-Filename" in r.getheader("Access-Control-Allow-Headers") and "Content-Type" in r.getheader("Access-Control-Allow-Headers")
    assert r.getheader("Access-Control-Allow-Origin") == "https://app.test"
    c = http.client.HTTPConnection(server.host, server.port, timeout=10)
    c.request("OPTIONS", "/api/excel/upload", headers={"Origin": "https://evil.test"})
    r = c.getresponse(); r.read(); c.close()
    assert r.getheader("Access-Control-Allow-Origin") is None


@pytest.fixture()
def enforced(monkeypatch):
    monkeypatch.setattr(settings, "ENFORCE_ORIGIN_CHECK", True)
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", ("https://app.test",))


def test_a_foreign_origin_is_refused_for_state_changing_requests(server, enforced):
    for origin in ("https://evil.test", "http://app.test", "https://app.test.evil.test", "null"):
        status, body, _ = raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Origin": origin})
        assert (status, body) == (403, {"error": "origin_not_allowed"}), origin


def test_a_refused_origin_never_reaches_authentication(server, enforced):
    for _ in range(8):
        raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Origin": "https://evil.test"})
    assert not auth._rate_limit_attempts                                                     # no failed-login was recorded: the request died before the handler logic


def test_allowed_and_same_origin_requests_pass(server, enforced):
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Origin": "https://app.test"})[0] == 401
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Origin": f"http://{server.host}:{server.port}", "Host": f"{server.host}:{server.port}"})[0] == 401


def test_requests_without_origin_pass_but_a_cross_site_fetch_without_origin_does_not(server, enforced):
    assert raw(server, "POST", "/api/login", body=LOGIN, headers=JSON)[0] == 401              # curl / server-to-server: not a CSRF vector
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Sec-Fetch-Site": "same-origin"})[0] == 401
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Sec-Fetch-Site": "cross-site"})[0] == 403


def test_origin_enforcement_is_always_on_in_production(server, monkeypatch):
    monkeypatch.setenv("RCS_ENV", "production")
    monkeypatch.setattr(settings, "ENFORCE_ORIGIN_CHECK", False)
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", ("https://app.test",))
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Origin": "https://evil.test"})[0] == 403
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Origin": "https://app.test"})[0] == 401


def test_origin_is_not_enforced_in_plain_local_development(server, monkeypatch):
    monkeypatch.delenv("RCS_ENV", raising=False)
    monkeypatch.setattr(settings, "ENFORCE_ORIGIN_CHECK", False)
    assert raw(server, "POST", "/api/login", body=LOGIN, headers={**JSON, "Origin": "http://localhost:5199"})[0] == 401       # any dev-server port works


def test_a_cross_site_form_post_cannot_change_state_even_with_a_valid_session(server, enforced):
    status, _b, r = raw(server, "POST", "/api/login", body=json.dumps({"email": "a@x.org", "password": "right-password-1"}).encode(), headers=JSON)
    cookie = r.getheader("Set-Cookie").split(";")[0]
    payload = json.dumps({"comment_ids": ["x"], "reviewer_user_id": "y"}).encode()
    # what a hostile page can send without a preflight: text/plain (or form) bodies, with the victim's cookie and the attacker's Origin
    for ctype in ("text/plain", "application/x-www-form-urlencoded"):
        s, body, _ = raw(server, "POST", "/api/admin/assign", body=payload, headers={"Content-Type": ctype, "Cookie": cookie, "Origin": "https://evil.test"})
        assert s == 415
    s, body, _ = raw(server, "POST", "/api/admin/assign", body=payload, headers={**JSON, "Cookie": cookie, "Origin": "https://evil.test"})
    assert s == 403 and body == {"error": "origin_not_allowed"}
    s, body, _ = raw(server, "POST", "/api/admin/assign", body=payload, headers={**JSON, "Cookie": cookie, "Origin": "https://app.test"})
    assert s in (200, 400)                                                                    # the legitimate frontend origin reaches the endpoint


def test_the_handler_bounds_stalled_connections():
    assert ReviewerAPIHandler.timeout == settings.REQUEST_SOCKET_TIMEOUT_SECONDS and 5 <= ReviewerAPIHandler.timeout <= 120


# ---------------------------------------------------------------- gitignore + build configuration
def _ignored(path):
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


@pytest.mark.parametrize("path", [
    "data/processed/agent_reply_review/session_x.jsonl", "data/anything/visitor_comments.jsonl",
    "data/evaluation/label_audit_round1/coordinator_only/audit_provenance_mapping.csv", "data/evaluation/context_signals/dev_results.json",
    "scripts/offline_eval/reply_strategy/results/strategy_v3_ab_cases.jsonl", "scripts/offline_eval/routing_safety/results/routing_safety_labels.jsonl",
    "scripts/offline_eval/response_agent/results/m5f_report.md", "scripts/benchmarks/retrieval/results/m5b2_report.json",
    "scripts/offline_eval/response_agent/rg_v14_live_cases.json", "any/where/new_cases.jsonl", ".claude/settings.local.json",
    "PASTE_DB_URL_HERE", "tmp/local.db", "tmp/local.sqlite", "tmp/local.db-journal", ".env", ".env.production", ".env.local", "web/.env.production.local", "ca.pem", "private.key",
    "design-references/shared/brand-logo-original.png", "design-references/shared/anything.png",
])
def test_sensitive_paths_are_git_ignored(path):
    assert _ignored(path), path


@pytest.mark.parametrize("path", [".env.example", "web/.env.example", "design-references/README.md", "docs/uat_findings.md",
                                  "data/evaluation/label_audit_round1/reviewer/REVIEWER_INSTRUCTIONS.md", "config/frozen_models_manifest.json", "web/vercel.json"])
def test_templates_and_reviewable_files_are_not_ignored(path):
    assert not _ignored(path), path


KNOWN_TRACKED_BUT_IGNORED = {".claude/launch.json"}                  # local tooling config; to be removed from the index with `git rm --cached` (the file stays on disk)


def test_no_tracked_file_is_hidden_by_an_ignore_rule():
    out = subprocess.run(["git", "ls-files", "-ci", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert set(out) <= KNOWN_TRACKED_BUT_IGNORED, f"tracked files now hidden by an ignore rule: {sorted(set(out) - KNOWN_TRACKED_BUT_IGNORED)}"


def test_design_references_are_synthetic_and_stay_trackable():
    notes = " ".join(p.read_text(errors="ignore") for p in (ROOT / "design-references").rglob("*.md"))
    assert "HUMAN APPROVED" in notes                                                           # they are the approved visual-identity references, not data exports
    assert not _ignored("design-references/01-login/approved-reference.png")


def test_render_builds_from_server_only_requirements_with_cpu_torch_and_fetches_models():
    svc = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
    build = svc["buildCommand"]
    assert "requirements-server.txt" in build and "requirements.txt " not in build.replace("requirements-server.txt", "")
    assert build.index("whl/cpu") < build.index("-r requirements-server.txt") < build.index("scripts/fetch_frozen_models.py")
    assert svc["autoDeploy"] is False and svc["healthCheckPath"] == "/api/health"
    keys = {e["key"]: e for e in svc["envVars"]}
    assert keys["RCS_ENV"]["value"] == "production" and keys["DATABASE_URL"].get("sync") is False and keys["PERSISTENCE_BACKEND"]["value"] == "postgres"
    assert "RCS_TRUSTED_PROXIES" in keys and keys["HF_TOKEN"].get("sync") is False
    assert "value" not in keys["DATABASE_URL"] and "value" not in keys["GEMINI_API_KEY"]    # no secret is ever written into the blueprint


def test_dockerfile_uses_the_same_server_only_install_and_fetches_models_at_start():
    d = (ROOT / "Dockerfile").read_text(); e = (ROOT / "docker/entrypoint.sh").read_text()
    assert d.index("whl/cpu") < d.index("-r requirements-server.txt")
    assert "scripts/fetch_frozen_models.py" in e and "serve_reviewer_api.py" in e and "ENV PYTHONUNBUFFERED=1" in d.replace("\\\n    ", "").replace("  ", " ") or "RCS_ENV=production" in d


def test_server_requirements_exclude_training_and_dev_packages():
    reqs = {l.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip().lower() for l in (ROOT / "requirements-server.txt").read_text().splitlines() if l.strip() and not l.startswith("#")}
    assert not reqs & {"pytest", "gradio", "datasets", "accelerate", "jupyter", "notebook", "matplotlib", "seaborn"}
    assert {"pyjwt", "pymongo", "httpx", "psycopg", "torch", "transformers", "bcrypt", "sqlalchemy", "alembic", "huggingface_hub"} <= reqs
