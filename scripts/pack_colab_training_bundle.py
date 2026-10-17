#!/usr/bin/env python3
"""
Build a self-contained Google Colab training bundle for RCS MARBERTv2 selection.

Exports train+validation-only gold CSV. Canonical local gold is never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training_bundle_lib import (  # noqa: E402
    BUNDLE_GOLD_REL,
    BUNDLE_MANIFEST_REL,
    CANONICAL_GOLD_SHA256,
    COMMON_FORBIDDEN_FRAGMENTS,
    assert_bundle_safe,
    assert_portable_text,
    export_train_val_gold_csv,
    sha256,
    write_bundle_manifest,
)

DEFAULT_OUT = PROJECT_ROOT / "dist" / "rcs_colab_training_bundle.zip"
STAGING_DIR = PROJECT_ROOT / "dist" / "colab_staging"
CANONICAL_GOLD = PROJECT_ROOT / "data" / "gold" / "gold_dataset_final_training.csv"

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
    "colab/requirements.txt",
    "colab/setup_gpu_env.py",
    "colab/preflight.py",
    "colab/package_artifacts.py",
    "colab/convert_hub_weights_to_safetensors.py",
    "colab/COLAB_SETUP.md",
    "colab/rcs_model_selection.ipynb",
    # 2026-08-24: sentiment (CAMeLBERT) must run in its own runtime -- see
    # colab/rcs_model_selection.ipynb Cell 6's --skip-sentiment. Bundling
    # this notebook too means one zip covers both required Colab runs.
    "colab/phase_e_sentiment_colab.ipynb",
)

FORBIDDEN_FRAGMENTS = COMMON_FORBIDDEN_FRAGMENTS + (
    "kaggle/",
    "/kaggle/",
)


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
        canonical_csv=CANONICAL_GOLD,
        output_csv=staging / BUNDLE_GOLD_REL,
    )
    write_bundle_manifest(
        staging / BUNDLE_MANIFEST_REL,
        export_stats,
        bundle_target="google_colab",
    )
    return export_stats


def build_bundle(*, output_zip: Path) -> dict:
    export_stats = _prepare_staging(STAGING_DIR)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict[str, str | int]] = []

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in BUNDLE_REL_PATHS:
            assert_bundle_safe(rel, FORBIDDEN_FRAGMENTS)
            src = STAGING_DIR / rel
            if not src.exists():
                raise FileNotFoundError(f"staged bundle file missing: {src}")
            assert_portable_text(src)
            zf.write(src, arcname=f"rcs_project/{rel}")
            manifest_files.append(
                {"path": rel, "sha256": sha256(src), "bytes": src.stat().st_size}
            )

    manifest = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "bundle_root": "rcs_project/",
        "bundle_target": "google_colab",
        "bundle_version": "2",
        "source_canonical_gold_sha256": CANONICAL_GOLD_SHA256,
        "train_val_csv_sha256": export_stats["train_val_csv_sha256"],
        "human_test_policy": "excluded from bundle CSV (0 human_test rows shipped)",
        "default_transformer_plan": "MARBERTv2 intent+topic (main runtime) + sentiment_pretrained (separate phase_e runtime, torch>=2.6)",
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
    parser = argparse.ArgumentParser(description="Pack Google Colab training bundle.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build_bundle(output_zip=args.output), indent=2))


if __name__ == "__main__":
    main()
