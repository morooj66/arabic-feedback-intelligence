"""Isolated demo SQLite seed for the reviewer UI. Does not touch production gold datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from rcs.review.demo_seed import DEMO_DB, seed_demo_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEMO_DB)
    parser.add_argument("--live", action="store_true", help="Call current Gemini response-generation pipeline (dev/demo only)")
    args = parser.parse_args()
    path = seed_demo_db(args.db, live=True if args.live else None)
    print(f"seeded demo db: {path}")
    print(f"prompt_version={settings.RESPONSE_GENERATION_PROMPT_VERSION}")
    print(f"model={settings.RESPONSE_GENERATION_MODEL_NAME}")
    print(f"live={getattr(seed_demo_db, 'last_used_live', False)}")
    print(f"refreshed={getattr(seed_demo_db, 'last_refreshed_comment_ids', [])}")


if __name__ == "__main__":
    main()
