#!/usr/bin/env python3
"""Build the closed local demo package (isolated DB + reviewed_responses_output.xlsx)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rcs.review.local_demo_package import run_closed_local_demo  # noqa: E402


def main() -> int:
    report = run_closed_local_demo()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
