#!/usr/bin/env python3
"""
Smoke test: load MARBERTv2 tokenizer/model and run one tiny forward pass.

Does not load gold data, does not call Trainer.train(), and does not write
checkpoints. Requires network on first run to download weights from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rcs.training.gold_splits import TASK_LABELS  # noqa: E402
from rcs.training.hf_loading import load_sequence_classifier, load_tokenizer  # noqa: E402
from rcs.training.transformer import DEFAULT_INTENT_MODELS  # noqa: E402

MARBERT_MODEL_ID = DEFAULT_INTENT_MODELS["marbertv2"]
DEFAULT_TEXT = "شكراً على هذا المعجم الرائع"


def run_smoke(*, model_id: str = MARBERT_MODEL_ID, text: str = DEFAULT_TEXT) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing ML stack. Install with:\n"
            "  pip install torch transformers\n"
            f"Original error: {exc}"
        ) from exc

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_labels = len(TASK_LABELS["intent"])
    tokenizer = load_tokenizer(model_id)
    model = load_sequence_classifier(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    return {
        "model_id": model_id,
        "device": str(device),
        "num_labels": num_labels,
        "input_text": text,
        "logits_shape": list(logits.shape),
        "pred_index": int(probs.argmax(dim=-1).item()),
        "use_safetensors": True,
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MARBERTv2 load + forward-pass smoke test.")
    parser.add_argument("--model-id", default=MARBERT_MODEL_ID)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()
    result = run_smoke(model_id=args.model_id, text=args.text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
