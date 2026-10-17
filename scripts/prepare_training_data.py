#!/usr/bin/env python3
"""Export frozen train/val splits and class distributions (no ML deps)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rcs.training.artifacts import SELECTION_DIR, utc_now_iso  # noqa: E402
from rcs.training.gold_splits import load_supervised_gold  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export train/val split manifest.")
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=None,
        help="Gold CSV path (default: canonical local path).",
    )
    args = parser.parse_args()
    data = load_supervised_gold(args.gold_csv, include_test=False)
    out_dir = SELECTION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": utc_now_iso(),
        "train_rows": data.train_rows,
        "val_rows": data.val_rows,
        "train_excluded_insufficient": data.train_excluded_insufficient,
        "val_excluded_insufficient": data.val_excluded_insufficient,
        "val_excluded_unlabeled": data.val_excluded_unlabeled,
        "val_excluded_details": data.val_excluded_details,
        "intent_train": data.intent.train_distribution(),
        "intent_val": data.intent.val_distribution(),
        "topic_train": data.topic.train_distribution(),
        "topic_val": data.topic.val_distribution(),
        "sentiment_train": data.sentiment.train_distribution(),
        "sentiment_val": data.sentiment.val_distribution(),
        "imbalance_flags": data.imbalance_flags,
        "rules": {
            "train_stages": ["human_train_seed", "ai_assisted_train"],
            "val_stage": "human_val",
            "supervised_status": "LABELED",
            "test_unopened": True,
        },
    }
    path = out_dir / "training_data_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
