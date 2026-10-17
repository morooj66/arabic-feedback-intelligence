#!/usr/bin/env python3
"""
Standalone environment doctor for Phase B.2 — NOT part of the pipeline,
never imported by run_ai_suggestions.py. Run this once after setting up
the venv to confirm the local environment can actually load the current
google-genai SDK and reach its Interactions API surface, before spending
any real API calls.

Makes NO network call and touches NO project data — it only checks Python
version, package import, package version, and that the expected client
methods exist (client.interactions.create). Constructing a Client with a
dummy key is safe; the SDK doesn't connect to anything until a method that
actually sends a request is called.

Usage (inside the activated venv):
    python3 scripts/check_genai_setup.py
"""

from __future__ import annotations

import sys

MIN_PYTHON = (3, 10)
RECOMMENDED_PYTHON = (3, 11)


def main() -> int:
    ok = True

    py_version = sys.version_info[:3]
    print(f"Python version: {'.'.join(map(str, py_version))}")
    if py_version[:2] < MIN_PYTHON:
        print(f"  FAIL: google-genai's Interactions API requires Python >= {'.'.join(map(str, MIN_PYTHON))}")
        ok = False
    elif py_version[:2] < RECOMMENDED_PYTHON:
        print(f"  OK (meets the {'.'.join(map(str, MIN_PYTHON))} floor), "
              f"but {'.'.join(map(str, RECOMMENDED_PYTHON))}+ is recommended")
    else:
        print("  OK")

    try:
        from google import genai
    except ImportError as exc:
        print(f"import `from google import genai`: FAIL ({exc})")
        print("  -> pip install -r requirements.txt inside the venv")
        return 1
    print("import `from google import genai`: OK")

    try:
        import importlib.metadata as importlib_metadata
        version = importlib_metadata.version("google-genai")
    except Exception as exc:  # noqa: BLE001
        version = f"<could not determine: {exc}>"
    print(f"google-genai package version: {version}")
    try:
        major, minor = (int(x) for x in version.split(".")[:2])
        if (major, minor) < (2, 3):
            print("  FAIL: Interactions API requires google-genai >= 2.3.0")
            ok = False
        else:
            print("  OK")
    except Exception:  # noqa: BLE001
        print("  WARNING: could not parse version to check the >=2.3.0 requirement")

    try:
        client = genai.Client(api_key="dummy-key-no-network-call-made-by-this-check")
    except Exception as exc:  # noqa: BLE001
        print(f"genai.Client(...) construction: FAIL ({exc})")
        return 1
    print("genai.Client(...) construction: OK")

    if not hasattr(client, "interactions"):
        print("client.interactions: FAIL (attribute missing)")
        ok = False
    elif not hasattr(client.interactions, "create"):
        print("client.interactions.create: FAIL (method missing)")
        ok = False
    else:
        print("client.interactions.create: OK (present, not called)")

    print()
    if ok:
        print("RESULT: environment looks correct. Safe to run the smoke test:")
        print("  python3 scripts/run_ai_suggestions.py --limit 10")
        return 0
    else:
        print("RESULT: one or more checks failed — fix the issue(s) above before running the smoke test.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
