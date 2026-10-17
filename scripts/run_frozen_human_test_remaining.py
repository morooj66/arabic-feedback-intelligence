#!/usr/bin/env python3
"""Inference-only remaining human_test legs. Does not rewrite Intent artifacts or re-open the seal timestamp."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rcs.evaluation.metrics import evaluate
from rcs.training.artifacts import report_to_dict, utc_now_iso
from rcs.training.gold_splits import load_supervised_gold
from rcs.training.sentiment_pretrained import evaluate_pretrained_sentiment

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_frozen_human_test as rht

FROZEN_VAL_SENTIMENT = rht.FROZEN_VAL_SENTIMENT
FREEZE_JSON = rht.FREEZE_JSON
GOLD_CSV = rht.GOLD_CSV
OUT_DIR = rht.OUT_DIR
find_topic_checkpoint = rht.find_topic_checkpoint
evaluate_frozen_topic = rht.evaluate_frozen_topic
report_payload = rht.report_payload

INTENT_REPORT = OUT_DIR / "intent_tfidf_logistic_test_report.json"


def main() -> None:
    if not INTENT_REPORT.exists():
        raise SystemExit("Intent test report missing — aborting so Intent is not regenerated accidentally")
    intent_before = INTENT_REPORT.read_bytes()
    freeze = json.loads(FREEZE_JSON.read_text())
    data = load_supervised_gold(GOLD_CSV, include_test=True)

    sent = evaluate_pretrained_sentiment(
        data.sentiment.test,
        model_id=freeze["sentiment"]["model_id"],
        batch_size=int(freeze["sentiment"]["preprocessing"]["batch_size"]),
    )
    sent_payload = report_payload(
        sent.report,
        {
            "task": "sentiment",
            "model_id": sent.model_id,
            "split": "human_test",
            "val_macro_f1_frozen": FROZEN_VAL_SENTIMENT,
            "label_order": list(freeze["sentiment"]["label_order"]),
            "max_length": freeze["sentiment"]["preprocessing"]["max_length"],
            "inference_only": True,
            "fine_tune": False,
        },
    )
    (OUT_DIR / "sentiment_camelbert_test_report.json").write_text(
        json.dumps(sent_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "sentiment_camelbert_test_preds.json").write_text(
        json.dumps(
            {
                "comment_ids": sent.comment_ids,
                "y_true": sent.y_true,
                "y_pred": sent.y_pred,
                "labels": list(sent.report.labels),
                "probabilities": sent.probabilities,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint = find_topic_checkpoint()
    topic_status = {"scored": False}
    if checkpoint is None:
        topic_status["error"] = "checkpoint_missing"
        (OUT_DIR / "TOPIC_CHECKPOINT_REQUIRED.json").write_text(
            json.dumps(
                {
                    "status": "missing",
                    "do_not_retrain": True,
                    "colab_source_dir": "/content/selection/marbertv2_topic/best",
                    "optional_drive_copy": "/content/drive/MyDrive/rcs_selection/marbertv2_topic/best",
                    "local_destination": "data/models/final_test/checkpoints/marbertv2_topic/best",
                    "smallest_required_files": [
                        "config.json",
                        "model.safetensors",
                        "tokenizer_config.json",
                        "tokenizer.json",
                        "special_tokens_map.json",
                        "vocab.txt",
                    ],
                    "notes": [
                        "This is trainer.save_model(best) + tokenizer.save_pretrained from the frozen Colab run.",
                        "Do not download full rcs_selection_artifacts.zip or checkpoints/checkpoint-* optimizer states.",
                        "model.safetensors is the only large file (~0.5–0.7 GB). Metrics JSON is not a substitute.",
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        topic_run = evaluate_frozen_topic(data.topic.test, checkpoint)
        topic_payload = report_payload(
            topic_run["report"],
            {
                "task": "topic",
                "model_id": freeze["topic"]["model_id"],
                "huggingface_id": freeze["topic"]["huggingface_id"],
                "split": "human_test",
                "checkpoint": topic_run["checkpoint"],
                "device": topic_run["device"],
                "label_order": topic_run["labels"],
                "inference_only": True,
            },
        )
        (OUT_DIR / "topic_marbertv2_test_report.json").write_text(
            json.dumps(topic_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        topic_status = {"scored": True, "macro_f1": topic_payload["macro_f1"]}

    if INTENT_REPORT.read_bytes() != intent_before:
        raise SystemExit("Intent report changed — refusing to continue")

    print(json.dumps({"sentiment_macro_f1": sent_payload["macro_f1"], "n": sent_payload["n_scored"], "topic": topic_status, "finished_at": utc_now_iso()}, indent=2))


if __name__ == "__main__":
    main()
