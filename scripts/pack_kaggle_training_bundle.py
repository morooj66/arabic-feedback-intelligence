#!/usr/bin/env python3
"""
Build a minimal zip bundle for Kaggle/Colab GPU model selection.

Exports a train+validation-only gold CSV (no human_test rows). The canonical
local gold_dataset_final_training.csv is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rcs.training.gold_splits import TEST_STAGE, load_supervised_gold  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "dist" / "rcs_kaggle_training_bundle.zip"
STAGING_DIR = PROJECT_ROOT / "dist" / "kaggle_staging"
CANONICAL_GOLD = PROJECT_ROOT / "data" / "gold" / "gold_dataset_final_training.csv"

# Frozen canonical gold CSV hash (gold_dataset_final_training.csv).
CANONICAL_GOLD_SHA256 = (
    "ac05bb8dc47a9787e658049d8ac650cd0a319236d201698e3196c166d7cf1227"
)

BUNDLE_GOLD_REL = "data/gold/gold_train_val_only.csv"
BUNDLE_MANIFEST_REL = "data/gold/bundle_training_manifest.json"

BUNDLE_REL_PATHS: tuple[str, ...] = (
    "config/__init__.py",
    "config/settings.py",
    "src/rcs/training/__init__.py",
    "src/rcs/training/artifacts.py",
    "src/rcs/training/classical.py",
    "src/rcs/training/gold_splits.py",
    "src/rcs/training/sentiment_pretrained.py",
    "src/rcs/training/transformer.py",
    "src/rcs/training/hf_loading.py",
    "src/rcs/evaluation/metrics.py",
    "src/rcs/evaluation/gold_dataset.py",
    "src/rcs/evaluation/ai_suggestion_schema.py",
    "scripts/run_model_selection.py",
    "scripts/prepare_training_data.py",
    "scripts/gpu_smoke_test.py",
    BUNDLE_GOLD_REL,
    BUNDLE_MANIFEST_REL,
    "kaggle/requirements.txt",
    "kaggle/setup_gpu_env.py",
    "kaggle/KAGGLE_SETUP.md",
    "kaggle/COLAB_SETUP.md",
    "kaggle/rcs_model_selection.ipynb",
)

FORBIDDEN_FRAGMENTS = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_bundle_safe(rel_path: str) -> None:
    lowered = rel_path.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment.lower() in lowered:
            raise ValueError(f"forbidden path in bundle: {rel_path} (matches {fragment!r})")


def _assert_portable_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "/Users/" in text or "\\Users\\" in text:
        raise ValueError(f"absolute user path found in bundled file: {path}")
    for fragment in ("b3_human_review.json", "b2_calibration_human_review.json"):
        if fragment in text and path.name not in ("pack_kaggle_training_bundle.py",):
            raise ValueError(f"reviewer JSON reference in bundled file: {path}")


def export_train_val_gold_csv(
    *,
    canonical_csv: Path = CANONICAL_GOLD,
    output_csv: Path,
) -> dict:
    """Filter human_test out of a copy for external GPU bundles."""
    if not canonical_csv.exists():
        raise FileNotFoundError(f"canonical gold CSV missing: {canonical_csv}")
    canonical_hash = _sha256(canonical_csv)
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
        "train_val_csv_sha256": _sha256(output_csv),
        "canonical_rows": len(gold),
        "train_val_rows": len(filtered),
        "human_test_rows_excluded": test_rows,
        "supervised_train_rows": data.train_rows,
        "supervised_val_rows": data.val_rows,
    }


def _write_bundle_manifest(staging: Path, export_stats: dict) -> Path:
    manifest_path = staging / BUNDLE_MANIFEST_REL
    payload = {
        "bundle_version": "2",
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


def _prepare_staging(staging: Path) -> dict:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for rel in BUNDLE_REL_PATHS:
        if rel in (BUNDLE_GOLD_REL, BUNDLE_MANIFEST_REL):
            continue
        src = PROJECT_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"bundle source file missing: {src}")
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    export_stats = export_train_val_gold_csv(
        output_csv=staging / BUNDLE_GOLD_REL,
    )
    _write_bundle_manifest(staging, export_stats)
    return export_stats


def build_bundle(*, output_zip: Path) -> dict:
    export_stats = _prepare_staging(STAGING_DIR)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict[str, str | int]] = []

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in BUNDLE_REL_PATHS:
            _assert_bundle_safe(rel)
            src = STAGING_DIR / rel
            if not src.exists():
                raise FileNotFoundError(f"staged bundle file missing: {src}")
            _assert_portable_text(src)
            zf.write(src, arcname=f"rcs_project/{rel}")
            manifest_files.append(
                {
                    "path": rel,
                    "sha256": _sha256(src),
                    "bytes": src.stat().st_size,
                }
            )

    manifest = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "bundle_root": "rcs_project/",
        "bundle_version": "2",
        "source_canonical_gold_sha256": CANONICAL_GOLD_SHA256,
        "train_val_csv_sha256": export_stats["train_val_csv_sha256"],
        "human_test_policy": "excluded from bundle CSV (0 human_test rows shipped)",
        "default_transformer_plan": "MARBERTv2 intent+topic + sentiment_pretrained",
        "files": manifest_files,
    }
    manifest_path = output_zip.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "zip_path": str(output_zip),
        "manifest_path": str(manifest_path),
        "file_count": len(manifest_files),
        "source_canonical_gold_sha256": CANONICAL_GOLD_SHA256,
        "train_val_csv_sha256": export_stats["train_val_csv_sha256"],
        "human_test_rows_excluded": export_stats["human_test_rows_excluded"],
        "supervised_train_rows": export_stats["supervised_train_rows"],
        "supervised_val_rows": export_stats["supervised_val_rows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack minimal Kaggle training bundle.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output zip path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    summary = build_bundle(output_zip=args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
