#!/usr/bin/env python3
"""
One-time: convert a Hugging Face model cache from pytorch_model.bin to safetensors.

Use when a hub repo lacks model.safetensors (e.g. CAMeLBERT sentiment) and torch<2.6
cannot load .bin. Requires torch>=2.6 for the conversion load only.

After conversion, revert to torch 2.5.1+cu121 and load with use_safetensors=True.

Example (temporary torch 2.6 on Colab):
  pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cu121
  python colab/convert_hub_weights_to_safetensors.py \\
    --model-id CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
  # reinstall torch 2.5.1 via colab/setup_gpu_env.py --install + restart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def convert(model_id: str, *, cache_dir: str | None = None) -> str:
    import torch
    from packaging import version
    from transformers import AutoModelForSequenceClassification

    if version.parse(torch.__version__) < version.parse("2.6.0"):
        raise SystemExit(
            f"conversion requires torch>=2.6 (have {torch.__version__}). "
            "Upgrade temporarily, convert, then reinstall torch 2.5.1+cu121."
        )

    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        use_safetensors=False,
        **kwargs,
    )
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(model_id, cache_dir=cache_dir))
    model.save_pretrained(snap, safe_serialization=True)
    return str(snap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF .bin weights to safetensors in cache.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    path = convert(args.model_id, cache_dir=args.cache_dir)
    print(f"Wrote safetensors into: {path}")


if __name__ == "__main__":
    main()
