#!/usr/bin/env python3
"""Zip selection artifacts for download from Google Colab."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SELECTION = Path("/content/selection")
DEFAULT_OUT = Path("/content/rcs_selection_artifacts.zip")


def package_artifacts(selection_dir: Path, output_zip: Path) -> dict:
    if not selection_dir.exists():
        raise FileNotFoundError(f"selection directory not found: {selection_dir}")

    if output_zip.exists():
        output_zip.unlink()

    base = selection_dir.resolve()
    shutil.make_archive(str(output_zip.with_suffix("")), "zip", root_dir=base)

    expected = [
        "model_selection_summary.json",
        "model_selection_summary.txt",
        "marbertv2_intent",
        "marbertv2_topic",
        "sentiment_pretrained_val_report.json",
    ]
    present = [name for name in expected if (base / name).exists()]

    summary = {
        "packaged_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_dir": str(base),
        "output_zip": str(output_zip),
        "bytes": output_zip.stat().st_size if output_zip.exists() else 0,
        "expected_artifacts_present": present,
        "download_hint": "files.download('/content/rcs_selection_artifacts.zip')",
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Colab selection artifacts.")
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    package_artifacts(args.selection_dir, args.output)


if __name__ == "__main__":
    main()
