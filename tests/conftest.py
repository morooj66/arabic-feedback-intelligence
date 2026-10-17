import os
import sys
from pathlib import Path

import pytest

# The suite must never see, use or print real credentials. A failing assertion that compares a configured value (for example DATABASE_URL) would otherwise echo the
# developer's real connection string into the test report. Scrub them before any project module is imported, and stop the .env loader from re-reading them.
# Deliberate live runs (for example the opt-in MongoDB/Gemini tests) must set RCS_ALLOW_LIVE_TEST_SECRETS=1 explicitly and supply their own environment.
if os.environ.get("RCS_ALLOW_LIVE_TEST_SECRETS") != "1":
    for _secret in ("DATABASE_URL", "MONGODB_URI", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "ENTRA_CLIENT_SECRET", "RCS_SMTP_PASSWORD"):
        os.environ.pop(_secret, None)
    os.environ["RCS_IGNORE_DOTENV"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Opt-in: run the data-layer suites on PostgreSQL instead of SQLite. Set RCS_TEST_POSTGRES_URL to a throw-away PostgreSQL database (never an institutional one); every
# `get_engine(<path>)` call then receives its own freshly created schema in that database, so tests stay isolated. Unset (the default) = SQLite exactly as before.
_PG_TEST_URL = os.environ.get("RCS_TEST_POSTGRES_URL", "").strip()
if _PG_TEST_URL:
    import uuid

    import rcs.review.repository as _repository
    from sqlalchemy import create_engine, text

    _sqlite_get_engine = _repository.get_engine
    _schema_for_path: dict[str, str] = {}              # the same path always maps to the same schema (tests open several engines on one "file", e.g. one per thread)

    def _postgres_get_engine(db_path_or_url):
        if isinstance(db_path_or_url, str) and "://" in db_path_or_url:
            return _sqlite_get_engine(db_path_or_url)
        key = str(db_path_or_url)
        if key not in _schema_for_path:
            _schema_for_path[key] = "t_" + uuid.uuid4().hex[:12]
            admin = create_engine(_PG_TEST_URL, isolation_level="AUTOCOMMIT")
            with admin.connect() as conn:
                conn.execute(text(f'CREATE SCHEMA "{_schema_for_path[key]}"'))
            admin.dispose()
        schema = _schema_for_path[key]
        return create_engine(_PG_TEST_URL, future=True, pool_pre_ping=True, connect_args={"options": f"-csearch_path={schema}"})

    _repository.get_engine = _postgres_get_engine


@pytest.fixture(autouse=True)
def _reset_login_rate_limit_state():
    """rcs.review.auth's login rate limiter is a process-wide in-memory
    dict (Phase 7) — every test in this suite hits a serve_in_thread()
    test server from the same loopback address, so without this reset
    tests would share a rate-limit bucket across files and order-depend
    on each other (one test's intentional 429 exhausting the count for
    every later test's legitimate first login)."""
    from rcs.review.auth import _rate_limit_attempts

    _rate_limit_attempts.clear()
    yield
    _rate_limit_attempts.clear()


@pytest.fixture
def legacy_eligibility_policy(monkeypatch):
    """Restores the pre-safety eligibility behaviour for tests of DOWNSTREAM machinery (refinement, prompt labels, offline evaluation,
    Excel flow) whose fixtures are ungrounded questions or bare SPAM labels. Tests of the safety policy itself never use this fixture
    (see tests/test_reply_safety_policy.py, which runs on the production defaults)."""
    from config import settings

    monkeypatch.setattr(settings, "RESPONSE_REQUIRE_GROUNDING_FOR_FACTUAL_QUESTIONS", False)
    monkeypatch.setattr(settings, "RESPONSE_REQUIRE_CORROBORATION_FOR_EXCLUSION", False)


_SQLITE_DEFAULT_TESTS = ("test_default_sqlite_startup_unchanged", "test_get_engine_still_accepts_a_bare_path_like_every_existing_caller",
                         "test_a_stray_database_url_never_affects_a_plain_get_engine_path_call")


def _pg_mode_skips(config, items):
    """In PostgreSQL mode (RCS_TEST_POSTGRES_URL) skip the tests whose subject is the SQLite default itself."""
    if not _PG_TEST_URL:
        return
    skip = pytest.mark.skip(reason="asserts the SQLite default of get_engine(path), which RCS_TEST_POSTGRES_URL deliberately overrides")
    for item in items:
        if item.name in _SQLITE_DEFAULT_TESTS:
            item.add_marker(skip)


def pytest_collection_modifyitems(config, items):
    _pg_mode_skips(config, items)


# --- Public-release skips ----------------------------------------------------------------------------------------------------------------------------
# These tests exercise things that are deliberately NOT in the public repository (frozen classifier files, restricted evaluation text, private
# calibration workbooks, training data) or need a git work tree. Each is skipped with an explicit reason instead of failing. Run the full suite in the
# private repository to cover them.
import pytest as _pytest  # noqa: E402

_NEEDS_FROZEN_MODELS = (
    "tests/test_deployment_safety.py::test_resume_orphaned_jobs_marks_the_stale_job_failed_and_completes_the_work",
    "tests/test_dev_inspect_agent_inspector.py::test_classification_stage_matches_configured_runtime",
    "tests/test_excel_batch_resume.py", "tests/test_excel_content_dedup.py", "tests/test_excel_http_api.py::test_process_runs_full_pipeline_appears_in_queue_and_exports",
    "tests/test_excel_http_api.py::test_real_suggested_response_reaches_item_detail_and_analytics_same_engine", "tests/test_excel_prefilter_integration.py",
    "tests/test_import_identity.py::test_batch_c_regression_over_http_and_the_job_runner", "tests/test_review_sessions.py",
)
_RESTRICTED = (
    ("restricted evaluation text / calibration workbooks / training data are not distributed", (
        "tests/test_ambiguous_intent_handling.py::test_9_real_substantive_suggestion_from_evaluation_corpus_not_neutered",
        "tests/test_ambiguous_intent_handling.py::test_9b_real_substantive_suggestion_succeeds_end_to_end",
        "tests/test_ambiguous_intent_handling.py::test_historical_evaluation_evidence_is_not_rewritten",
        "tests/test_calibration_labeling.py::test_existing_calibration_workbook_loads", "tests/test_calibration_wizard_handlers.py",
        "tests/test_intent_candidate.py::test_synthetic_examples_are_training_only_and_theme_groups_keep_contrast_pairs_together")),
)
_NEEDS_DESIGN_REFERENCES = ("tests/test_deployment_hardening.py::test_design_references_are_synthetic_and_stay_trackable",)
_NEEDS_IGNORED_FILES = ("tests/test_response_quality_rgv14.py::test_rg_v14_live_manifest_has_exactly_18_verified_cases_and_full_taxonomy_coverage",)
_NEEDS_GIT = (
    "tests/test_deployment_hardening.py::test_sensitive_paths_are_git_ignored",
    "tests/test_playground.py::test_default_storage_dir_is_outside_gold_and_git_ignored",
)


def _public_release_skips(config, items):
    root = PROJECT_ROOT
    frozen_missing = not (root / "data" / "models" / "final_test" / "intent_tfidf_logistic.joblib").exists()
    git_missing = not (root / ".git").exists()
    rules = []
    if frozen_missing:
        rules.append((_NEEDS_FROZEN_MODELS, "needs the frozen classifier artifacts (not distributed; see README)"))
    rules.extend((ids, reason) for reason, ids in _RESTRICTED)
    if not (root / "scripts" / "offline_eval" / "response_agent" / "rg_v14_live_cases.json").exists():
        rules.append((_NEEDS_IGNORED_FILES, "needs a git-ignored local evaluation manifest (private repository)"))
    if not (root / "design-references" / "README.md").exists():
        rules.append((_NEEDS_DESIGN_REFERENCES, "the institution's design references are kept in the private repository"))
    if git_missing:
        rules.append((_NEEDS_GIT, "needs a git work tree (`git init` first)"))
    for item in items:
        for ids, reason in rules:
            if any(item.nodeid.startswith(i) for i in ids):
                item.add_marker(_pytest.mark.skip(reason=f"public release: {reason}"))
                break


def pytest_collection_modifyitems(config, items):
    _pg_mode_skips(config, items)
    _public_release_skips(config, items)
