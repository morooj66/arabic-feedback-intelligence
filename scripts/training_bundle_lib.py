"""Shared helpers for external GPU training bundles (Kaggle / Colab)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rcs.training.gold_splits import TEST_STAGE, load_supervised_gold

# Updated 2026-08-25: frozen canonical is gold_dataset_final_training.csv
# (sha256 ac05bb8d…; supervised train 455 / val 108 / sealed human_test 75).
CANONICAL_GOLD_SHA256 = (
    "ac05bb8dc47a9787e658049d8ac650cd0a319236d201698e3196c166d7cf1227"
)

BUNDLE_GOLD_REL = "data/gold/gold_train_val_only.csv"
BUNDLE_MANIFEST_REL = "data/gold/bundle_training_manifest.json"

COMMON_FORBIDDEN_FRAGMENTS = (
    ".env",
    "b3_human_review.json",
    "b2_calibration_human_review.json",
    "gold_dataset_round1.csv",
    "b3_train_gold_complete.json",
    "review.sqlite",
    "visitors-reports",
    "install_ml_deps.py",
    "/users/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_bundle_safe(rel_path: str, forbidden: tuple[str, ...]) -> None:
    lowered = rel_path.lower()
    for fragment in forbidden:
        if fragment.lower() in lowered:
            raise ValueError(f"forbidden path in bundle: {rel_path} (matches {fragment!r})")


def assert_portable_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "/Users/" in text or "\\Users\\" in text:
        raise ValueError(f"absolute user path found in bundled file: {path}")
    for fragment in ("b3_human_review.json", "b2_calibration_human_review.json"):
        if fragment in text:
            raise ValueError(f"reviewer JSON reference in bundled file: {path}")


def export_train_val_gold_csv(*, canonical_csv: Path, output_csv: Path) -> dict:
    if not canonical_csv.exists():
        raise FileNotFoundError(f"canonical gold CSV missing: {canonical_csv}")
    canonical_hash = sha256(canonical_csv)
    if canonical_hash != CANONICAL_GOLD_SHA256:
        raise AssertionError(
            f"canonical gold CSV hash mismatch: expected {CANONICAL_GOLD_SHA256}, got {canonical_hash}"
        )

    gold = pd.read_csv(canonical_csv, dtype=str, keep_default_na=False)
    test_rows = int((gold["annotation_stage"] == TEST_STAGE).sum())
    filtered = gold[gold["annotation_stage"] != TEST_STAGE].copy()
    if (filtered["annotation_stage"] == TEST_STAGE).any():
        raise AssertionError("human_test rows remain after export filter")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")

    data = load_supervised_gold(output_csv, include_test=False)
    return {
        "source_canonical_gold_sha256": CANONICAL_GOLD_SHA256,
        "train_val_csv_sha256": sha256(output_csv),
        "canonical_rows": len(gold),
        "train_val_rows": len(filtered),
        "human_test_rows_excluded": test_rows,
        "supervised_train_rows": data.train_rows,
        "supervised_val_rows": data.val_rows,
    }


def write_bundle_manifest(
    manifest_path: Path,
    export_stats: dict,
    *,
    bundle_target: str,
) -> Path:
    payload = {
        "bundle_version": "2",
        "bundle_target": bundle_target,
        "purpose": "external_gpu_training_train_val_only",
        "source_canonical_gold_sha256": export_stats["source_canonical_gold_sha256"],
        "train_val_csv": BUNDLE_GOLD_REL,
        "train_val_csv_sha256": export_stats["train_val_csv_sha256"],
        "human_test_rows_excluded": export_stats["human_test_rows_excluded"],
        "supervised_train_rows": export_stats["supervised_train_rows"],
        "supervised_val_rows": export_stats["supervised_val_rows"],
        "default_transformer_runs": [
            "marbertv2_intent",
            "marbertv2_topic",
            "sentiment_pretrained",
        ],
        "arabert_opt_in_flag": "--include-arabert",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path
