"""Synthetic-data demo mode: the triple lock, the isolated stand-ins, and proof that activation changes only what it should. Activation patches process-wide state
(sockets, import system), so everything that activates runs in a SUBPROCESS and never in the pytest process."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from rcs.demo import classifier, guard, provider, state, workbook
from rcs.review.excel_review_flow import process_uploaded_excel_batch, write_fixture_workbook
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
GOOD_ENV = {guard.MODE_VAR: guard.MODE_VALUE, guard.ACK_VAR: guard.ACK_VALUE}
DB_OK = "postgresql+psycopg://u:p@host.example.invalid:5432/rcs_demo_online"


# ------------------------------------------------------------------------------------------------ the lock
def test_a_complete_demo_configuration_has_no_problems():
    assert guard.problems(GOOD_ENV, DB_OK) == []
    assert guard.problems(GOOD_ENV, "sqlite:////tmp/x/rcs_demo_local.sqlite") == []
    assert guard.problems(GOOD_ENV, Path("/tmp/x/rcs_demo_local.sqlite")) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: e.pop(guard.MODE_VAR), "RCS_DEMO_MODE"),
    (lambda e: e.update({guard.MODE_VAR: "1"}), "RCS_DEMO_MODE"),
    (lambda e: e.update({guard.MODE_VAR: "true"}), "RCS_DEMO_MODE"),
    (lambda e: e.pop(guard.ACK_VAR), "RCS_DEMO_ACK"),
    (lambda e: e.update({guard.ACK_VAR: "yes"}), "RCS_DEMO_ACK"),
    (lambda e: e.update({"GEMINI_API_KEY": "x"}), "GEMINI_API_KEY"),
    (lambda e: e.update({"OPENROUTER_API_KEY": "x"}), "OPENROUTER_API_KEY"),
    (lambda e: e.update({"MONGODB_URI": "mongodb://x"}), "MONGODB_URI"),
    (lambda e: e.update({"RCS_SMTP_HOST": "smtp.example.invalid"}), "RCS_SMTP_HOST"),
    (lambda e: e.update({"ENTRA_CLIENT_SECRET": "s"}), "ENTRA_CLIENT_SECRET"),
    (lambda e: e.update({"HF_TOKEN": "t"}), "HF_TOKEN"),
    (lambda e: e.update({"HF_MODEL_REPO_ID": "a/b"}), "HF_MODEL_REPO_ID"),
    (lambda e: e.update({"PERSISTENCE_BACKEND": "mongo"}), "mongo"),
])
def test_every_missing_or_unsafe_ingredient_blocks_demo_mode(mutate, needle):
    env = dict(GOOD_ENV)
    mutate(env)
    found = guard.problems(env, DB_OK)
    assert found and any(needle in p for p in found)


@pytest.mark.parametrize("url", [None, "", "postgresql://u:p@h/postgres", "postgresql://u:p@h/rcs", "postgresql://u:p@h/rcs_production", "postgresql://u:p@h/prod_rcs_demo",
                                 "sqlite:////tmp/review.sqlite", "mongodb://h/rcs_demo", "/tmp/data/db/review.sqlite"])
def test_only_a_dedicated_demo_database_is_accepted(url):
    assert guard.problems(GOOD_ENV, url)


def test_development_database_flags_are_refused_in_demo_mode():
    assert any("--seed" in p for p in guard.problems(GOOD_ENV, DB_OK, dev_db_flags_used=True))


def test_demo_is_only_requested_by_an_explicit_variable():
    assert guard.demo_requested({"RCS_ENV": "production", "DATABASE_URL": "x", "GEMINI_API_KEY": "k"}) is False
    assert guard.demo_requested({"RCS_DEMO_MODE": "synthetic-demo"}) is True
    assert guard.demo_requested({"RCS_DEMO_MODE": ""}) is False


def run_server_refusing(env_extra, args=()):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", **env_extra}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "serve_reviewer_api.py"), "--port", "0", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("env_extra,args,needle", [
    ({"RCS_DEMO_MODE": "synthetic-demo"}, (), "RCS_DEMO_ACK"),                                                                            # half configured
    ({"RCS_ENV": "production", "RCS_DEMO_MODE": "synthetic-demo", "DATABASE_URL": "postgresql://u:p@h/rcs_prod"}, (), "RCS_DEMO_ACK"),      # production + partial demo flags
    ({**GOOD_ENV, "DATABASE_URL": "postgresql://u:p@h/rcs_production"}, (), "rcs_demo"),                                                   # not a demo database
    ({**GOOD_ENV, "DATABASE_URL": "sqlite:////tmp/rcs_demo_x.sqlite", "GEMINI_API_KEY": "k"}, (), "GEMINI_API_KEY"),                      # a production secret is present
    ({**GOOD_ENV, "DATABASE_URL": "sqlite:////tmp/rcs_demo_x.sqlite"}, ("--seed",), "--seed"),                                              # development flags
])
def test_the_server_refuses_to_start_on_any_unsafe_demo_configuration(env_extra, args, needle):
    p = run_server_refusing(env_extra, args)
    assert p.returncode != 0 and needle in (p.stdout + p.stderr)
    assert "reviewer API on" not in p.stdout                                    # it never got as far as serving


def test_a_normal_start_never_loads_the_demo_machinery():
    code = "import sys; sys.path[:0]=['src','.']; import rcs.review.http_api as h; import rcs.demo.state as s; print('activation' , 'rcs.demo.activation' in sys.modules, 'active', s.is_active())"
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "activation False active False" in p.stdout, p.stdout + p.stderr


# ------------------------------------------------------------------------------------------------ the stand-ins
SAMPLES = {
    "THANKS_PRAISE": "شكرًا لكم على هذا المعجم المفيد وعلى جهودكم الواضحة",
    "COMPLAINT": "الصفحة لا تفتح من الهاتف وتظهر رسالة خطأ عند البحث",
    "SUGGESTION": "أقترح إضافة خاصية البحث بحسب الجذر لتسهيل الوصول إلى الكلمات",
    "INQUIRY": "ما معنى كلمة سوار وكيف تُنطق بالشكل الصحيح",
    "SPAM": "اشتروا منتجاتنا الآن بأسعار مخفضة وعروض حصرية",
}


def test_the_rule_classifier_returns_valid_taxonomy_labels_and_is_deterministic():
    from config import settings

    for want, text in SAMPLES.items():
        assert classifier.intent_of(text)[0] == want
        topic = classifier.topic_of(text, want)
        assert topic[0] in settings.TOPIC_SUBTYPES and classifier.sentiment_of(text, want)[0] in settings.SENTIMENTS and 0 < topic[1] <= 1
    assert classifier.predict_intent(list(SAMPLES.values())) == classifier.predict_intent(list(SAMPLES.values()))
    assert "FROZEN" not in classifier.provenance()["source"] and classifier.provenance()["source"] == "DEMO_RULES_SYNTHETIC_ONLY"


def test_the_demo_pipeline_runs_end_to_end_with_stand_ins_and_no_models_or_network(tmp_path, monkeypatch):
    import rcs.review.excel_review_flow as erf

    for name in ("predict_intent", "predict_topic", "predict_sentiment", "provenance", "apply_optional_topic_layer", "with_layer_provenance"):
        monkeypatch.setattr(erf, name, getattr(classifier, name))
    rows = [{"original_comment": f"{text} (عينة تجريبية رقم {i})", "original_title": None, "entry_id": None, "lemma": None, "visitor_name": f"زائر {i}", "visitor_email": f"d{i}@example.invalid",
             "original_type": "ملاحظة", "comment_created_at": dt.datetime(2026, 1, 1, 9) + dt.timedelta(hours=i), "platform": "web"} for i, text in enumerate(SAMPLES.values(), 1)]
    src = write_fixture_workbook(tmp_path / "w.xlsx", rows)
    engine = get_engine(tmp_path / "rcs_demo_t.sqlite"); init_db(engine)
    summary = process_uploaded_excel_batch(engine, src, batch_size=50, client=provider.DemoProvider())
    assert summary.batch_processed == 5
    with Session(engine) as s:
        got = {r.ai_primary_intent: r for r in s.query(ReviewItem).all()}
    assert set(got) == set(SAMPLES) and all(r.ai_classification_source == "DEMO_RULES_SYNTHETIC_ONLY" for r in got.values())
    for intent in ("THANKS_PRAISE", "COMPLAINT", "SUGGESTION"):
        assert got[intent].review_status == "DRAFT_READY" and got[intent].suggested_final_reply == provider.REPLIES[intent] and got[intent].generation_prompt_version == "rg-v14.4"
    for held_back in ("INQUIRY", "SPAM"):        # the normal eligibility policy still applies: factual questions (no grounding) and junk never get a draft
        assert got[held_back].suggested_final_reply in (None, "") and got[held_back].review_status != "DRAFT_READY"


def test_every_canned_reply_passes_the_real_validation_gates():
    from rcs.response_generation.validate import validate_draft

    for key, text in list(provider.REPLIES.items()) + [("DEFAULT", provider.DEFAULT)]:
        assert validate_draft(text) == (True, None), key


def test_every_canned_reply_is_safe_text():
    import re

    for text in list(provider.REPLIES.values()) + [provider.DEFAULT]:
        assert not re.search(r"https?://|www\.|@|\d{6,}", text) and 8 <= len(text) <= 200
        assert not any(w in text for w in ("سنقوم", "نعدكم", "سيتم", "خلال", "غدا", "أسبوع"))                           # no promises or timelines


# ------------------------------------------------------------------------------------------------ activation (always in a subprocess)
ACTIVATION_CHECK = textwrap.dedent("""
    import sys, json, socket, tempfile, datetime as dt
    sys.path[:0] = ["src", "."]
    from pathlib import Path
    import rcs.review.http_api as http_api, rcs.review.excel_review_flow as erf, rcs.review.account_requests as ar
    from rcs.demo import activation, state, workbook, provider, classifier
    from rcs.review.excel_review_flow import write_fixture_workbook
    out = {"before": state.is_active()}
    activation.activate()
    out["after"] = state.is_active()
    try:
        socket.create_connection(("93.184.216.34", 80), timeout=2); out["network"] = "ALLOWED"
    except ConnectionRefusedError:
        out["network"] = "refused"
    except Exception as exc:
        out["network"] = type(exc).__name__
    for mod in ("torch", "transformers", "sklearn", "huggingface_hub", "google.genai"):
        try:
            __import__(mod); out["import_" + mod] = "ALLOWED"
        except ImportError:
            out["import_" + mod] = "blocked"
    out["client"] = type(erf.build_live_gemini_client()).__name__
    out["classifier"] = erf.predict_intent is classifier.predict_intent
    rows = [{"original_comment": f"تعليق تجريبي رقم {i} حول إضافة الأمثلة", "original_title": None, "entry_id": None, "lemma": None, "visitor_name": "ز", "visitor_email": "a@example.invalid",
             "original_type": "ملاحظة", "comment_created_at": dt.datetime(2026, 1, 1, 9), "platform": "web"} for i in range(1, 6)]
    d = Path(tempfile.mkdtemp())
    plain = write_fixture_workbook(d / "plain.xlsx", rows)
    stamped = write_fixture_workbook(d / "stamped.xlsx", rows); workbook.stamp(stamped)
    big = write_fixture_workbook(d / "big.xlsx", rows * 50); workbook.stamp(big)
    out["plain_ok"] = http_api.validate_workbook(plain).ok
    out["stamped_ok"] = http_api.validate_workbook(stamped).ok
    out["big_ok"] = http_api.validate_workbook(big).ok
    try:
        ar.submit_request(None, email="a@example.invalid", display_name="x", password="long-enough-1", client_address="1")
    except ar.AccountRequestError as exc:
        out["account_request"] = exc.status
    print(json.dumps(out))
""")


def test_activation_applies_exactly_the_documented_changes():
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    p = subprocess.run([sys.executable, "-c", ACTIVATION_CHECK], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-800:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["before"] is False and out["after"] is True
    assert out["network"] == "refused"
    assert all(out[f"import_{m}"] == "blocked" for m in ("torch", "transformers", "sklearn", "huggingface_hub", "google.genai"))
    assert out["client"] == "DemoProvider" and out["classifier"] is True
    assert (out["plain_ok"], out["stamped_ok"], out["big_ok"]) == (False, True, False)          # marker required; 200-row cap
    assert out["account_request"] == 503                                                        # invite-only


def test_the_pytest_process_itself_is_never_in_demo_mode():
    assert state.is_active() is False


def test_the_dotenv_loader_is_inert_under_the_test_suite_so_real_credentials_cannot_be_reloaded(tmp_path, monkeypatch):
    from config import settings
    from rcs.review.demo_seed import load_dotenv_if_present

    assert os.environ.get("RCS_IGNORE_DOTENV") == "1" or os.environ.get("RCS_ALLOW_LIVE_TEST_SECRETS") == "1"
    (tmp_path / ".env").write_text("DEMO_PROBE_SECRET=should-not-load\n", encoding="utf-8")
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.delenv("DEMO_PROBE_SECRET", raising=False)
    monkeypatch.setenv("RCS_IGNORE_DOTENV", "1")
    load_dotenv_if_present()
    assert "DEMO_PROBE_SECRET" not in os.environ
    monkeypatch.delenv("RCS_IGNORE_DOTENV")
    load_dotenv_if_present()                                                  # without the flag the same file IS read (proves the probe is meaningful)
    assert os.environ.get("DEMO_PROBE_SECRET") == "should-not-load"
    monkeypatch.delenv("DEMO_PROBE_SECRET", raising=False)
