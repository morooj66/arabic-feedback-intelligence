"""The server-only requirements must cover every third-party package the real server path imports (a missing one crashes the container at start-up).
Imports the actual entrypoint modules in a clean subprocess, maps each loaded module to its distribution, and checks it against requirements-server.txt + transitive deps."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_BY_DESIGN = {"pillow", "pyarrow"}            # imported opportunistically by transformers when present; never required for text models

PROBE = r"""
import importlib, json, sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
before = set(sys.modules)
for m in ("rcs.review.http_api", "rcs.review.job_runner", "rcs.review.persistence", "rcs.review.repository", "rcs.review.email_sender", "rcs.review.excel_review_flow",
          "rcs.classification.runtime", "rcs.classification.frozen_inference", "rcs.classification.context_signals", "rcs.training.hf_loading",
          "rcs.response_generation.service", "rcs.response_intelligence.orchestrator", "rcs.evaluation.genai_client", "scripts.serve_reviewer_api", "scripts.fetch_frozen_models"):
    importlib.import_module(m)
from importlib.metadata import packages_distributions
pd = packages_distributions()
stdlib = set(sys.stdlib_module_names)
dists = sorted({d.lower().replace("_", "-") for t in {n.split(".")[0] for n in sys.modules} if t not in stdlib and t not in ("rcs", "config", "scripts") for d in pd.get(t, [])})
print(json.dumps(dists))
"""


def _norm(name: str) -> str:
    return re.split(r"[<>=!~\[; ]", name.strip())[0].lower().replace("_", "-")


def _closure() -> set[str]:
    seen: set[str] = set()

    def walk(n: str):
        if n in seen:
            return
        seen.add(n)
        try:
            for r in requires(n) or []:
                if "extra ==" not in r:
                    walk(_norm(r))
        except Exception:                                                                  # noqa: BLE001 - a distribution that is not installed locally simply has no edges
            pass

    for line in (ROOT / "requirements-server.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            walk(_norm(line))
    return seen


def test_every_third_party_package_the_server_imports_is_in_the_server_requirements():
    env = {**os.environ, "RCS_ENV": "development", "DATABASE_URL": "", "GEMINI_API_KEY": ""}
    p = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr[-800:]
    imported = set(json.loads(p.stdout.strip().splitlines()[-1]))
    missing = sorted(d for d in imported - _closure() if d not in OPTIONAL_BY_DESIGN)
    assert missing == [], f"imported by the server but absent from requirements-server.txt: {missing}"


def test_the_three_previously_missing_packages_are_declared_directly():
    direct = {_norm(l) for l in (ROOT / "requirements-server.txt").read_text().splitlines() if l.strip() and not l.startswith("#")}
    assert {"pyjwt", "pymongo", "httpx"} <= direct
    assert "PyJWT[crypto]" in (ROOT / "requirements-server.txt").read_text()                 # RS256 verification needs the crypto extra
