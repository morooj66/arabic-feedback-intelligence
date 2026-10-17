#!/usr/bin/env python3
"""One-time human_test evaluation of models frozen in FROZEN_MODEL_SELECTION.json.

Does not retrain transformers, tune hyperparameters, or reselect models.
Classical intent reconstitutes the frozen sklearn pipeline on the frozen
train split only (weights were never serialized) and must reproduce the
frozen validation Macro-F1 before test scores are written.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rcs.evaluation.metrics import EvaluationReport, evaluate
from rcs.training.artifacts import report_to_dict, utc_now_iso
from rcs.training.gold_splits import LabeledExample, load_supervised_gold
from rcs.training.sentiment_pretrained import evaluate_pretrained_sentiment

GOLD_CSV = PROJECT_ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
TRAIN_VAL_CSV = PROJECT_ROOT / "dist" / "colab_staging" / "data" / "gold" / "gold_train_val_only.csv"
EXCLUSION_JSON = PROJECT_ROOT / "data" / "gold" / "human_test_exclusion_ids.json"
FREEZE_JSON = PROJECT_ROOT / "data" / "models" / "selection" / "FROZEN_MODEL_SELECTION.json"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "final_test"
CANONICAL_GOLD_SHA256 = "ac05bb8dc47a9787e658049d8ac650cd0a319236d201698e3196c166d7cf1227"
INTENT_EVAL_REMAP = {"DUA_BLESSING": "THANKS_PRAISE"}
FROZEN_VAL_INTENT = 0.31538918597742127
FROZEN_VAL_TOPIC = 0.40167700167700165
FROZEN_VAL_SENTIMENT = 0.5147274934508977


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_payload(report: EvaluationReport, extra: dict) -> dict:
    payload = report_to_dict(report)
    n = sum(report.per_class_support.values())
    correct = sum(report.confusion_matrix[i][i] for i in range(len(report.labels)))
    payload["accuracy"] = (correct / n) if n else 0.0
    payload["n_scored"] = n
    payload.update(extra)
    return payload


def remap_intent(examples: list[LabeledExample]) -> tuple[list[LabeledExample], int]:
    remapped = 0
    out: list[LabeledExample] = []
    for ex in examples:
        label = INTENT_EVAL_REMAP.get(ex.label, ex.label)
        if label != ex.label:
            remapped += 1
        out.append(
            LabeledExample(
                comment_id=ex.comment_id,
                text=ex.text,
                label=label,
                annotation_stage=ex.annotation_stage,
                split=ex.split,
            )
        )
    return out, remapped


def fit_frozen_intent_pipeline(train: list[LabeledExample]) -> Pipeline:
    freeze = json.loads(FREEZE_JSON.read_text())
    vec_cfg = freeze["intent"]["preprocessing"]["vectorizer"]
    clf_cfg = freeze["intent"]["preprocessing"]["classifier"]
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=vec_cfg["analyzer"],
                    ngram_range=tuple(vec_cfg["ngram_range"]),
                    min_df=vec_cfg["min_df"],
                    max_features=vec_cfg["max_features"],
                    sublinear_tf=vec_cfg["sublinear_tf"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=clf_cfg["max_iter"],
                    random_state=clf_cfg["random_state"],
                    class_weight=clf_cfg["class_weight"],
                ),
            ),
        ]
    )
    pipe.fit([ex.text for ex in train], [ex.label for ex in train])
    return pipe


def topic_checkpoint_dirs() -> list[Path]:
    env = os.environ.get("RCS_MARBERTV2_TOPIC_CHECKPOINT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            OUT_DIR / "checkpoints" / "marbertv2_topic" / "best",
            PROJECT_ROOT / "data" / "models" / "selection" / "expanded_val_108_2026-08-25" / "marbertv2_topic" / "best",
            Path("/content/selection/marbertv2_topic/best"),
        ]
    )
    return candidates


def find_topic_checkpoint() -> Path | None:
    for path in topic_checkpoint_dirs():
        if (path / "config.json").exists() and (
            (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
        ):
            return path
    return None


def evaluate_frozen_topic(examples: list[LabeledExample], checkpoint: Path) -> dict:
    import torch

    from rcs.training.hf_loading import load_sequence_classifier, load_tokenizer

    freeze = json.loads(FREEZE_JSON.read_text())
    labels = list(freeze["topic"]["label_order"])
    max_length = int(freeze["topic"]["train_config"]["max_length"])
    batch_size = int(freeze["topic"]["train_config"]["batch_size"])
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    tokenizer = load_tokenizer(str(checkpoint))
    model = load_sequence_classifier(str(checkpoint))
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    y_true = [ex.label for ex in examples]
    y_pred: list[str] = []
    comment_ids = [ex.comment_id for ex in examples]
    all_probs: list[list[float]] = []
    texts = [ex.text for ex in examples]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for i in range(len(batch)):
            pred_idx = int(probs[i].argmax())
            y_pred.append(id2label[pred_idx])
            row = []
            for lab in labels:
                idx = next((j for j, name in id2label.items() if name == lab), None)
                row.append(float(probs[i][idx]) if idx is not None else 0.0)
            all_probs.append(row)
    report = evaluate(y_true, y_pred, labels=labels)
    return {
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "comment_ids": comment_ids,
        "probabilities": all_probs,
        "labels": labels,
        "device": device,
        "checkpoint": str(checkpoint),
    }


def integrity_before_labels() -> dict:
    import pandas as pd

    freeze = json.loads(FREEZE_JSON.read_text())
    gold_sha = sha256_file(GOLD_CSV)
    freeze_sha = sha256_file(FREEZE_JSON)
    exclusion = json.loads(EXCLUSION_JSON.read_text())
    test_ids_manifest = set(exclusion["test_ids"])

    gold_ids = pd.read_csv(
        GOLD_CSV,
        dtype=str,
        keep_default_na=False,
        usecols=["comment_id", "split", "annotation_stage", "gold_annotation_status"],
    )
    human_test = gold_ids[gold_ids["annotation_stage"] == "human_test"]
    split_test = gold_ids[gold_ids["split"] == "test"]
    test_ids = set(human_test["comment_id"])
    tv = pd.read_csv(
        TRAIN_VAL_CSV,
        dtype=str,
        keep_default_na=False,
        usecols=["comment_id", "split", "annotation_stage"],
    )
    tv_ids = set(tv["comment_id"])

    val_prob_files = [
        PROJECT_ROOT / "data/models/selection/expanded_val_108_2026-08-25/marbertv2_intent/val_probs.json",
        PROJECT_ROOT / "data/models/selection/expanded_val_108_2026-08-25/marbertv2_topic/val_probs.json",
        PROJECT_ROOT / "data/models/selection/expanded_val_108_2026-08-25/sentiment_pretrained_val_probs.json",
    ]
    val_id_sets = {}
    leak_vs_selection = {}
    for path in val_prob_files:
        ids = set(json.loads(path.read_text())["comment_ids"])
        val_id_sets[path.name] = len(ids)
        leak_vs_selection[str(path.relative_to(PROJECT_ROOT))] = sorted(ids & test_ids)

    return {
        "freeze_status": freeze["status"],
        "freeze_all_tasks_frozen": freeze["all_tasks_frozen"],
        "frozen_intent": freeze["intent"]["model_id"],
        "frozen_topic": freeze["topic"]["model_id"],
        "frozen_sentiment": freeze["sentiment"]["model_id"],
        "freeze_json_sha256": freeze_sha,
        "canonical_gold_sha256": gold_sha,
        "canonical_gold_sha256_match": gold_sha == CANONICAL_GOLD_SHA256,
        "human_test_row_count_annotation_stage": int(len(human_test)),
        "human_test_row_count_split": int(len(split_test)),
        "human_test_unique_ids": len(test_ids),
        "exclusion_manifest_count": len(test_ids_manifest),
        "ids_match_exclusion_manifest": test_ids == test_ids_manifest,
        "annotation_stage_equals_split_test": test_ids == set(split_test["comment_id"]),
        "labeled_human_test_rows": int((human_test["gold_annotation_status"] == "LABELED").sum()),
        "train_val_only_csv_contains_human_test": len(tv_ids & test_ids),
        "selection_val_prob_sizes": val_id_sets,
        "human_test_ids_in_selection_val_probs": {k: v for k, v in leak_vs_selection.items() if v},
        "topic_checkpoint_found": find_topic_checkpoint() is not None,
        "topic_checkpoint_path": str(find_topic_checkpoint()) if find_topic_checkpoint() else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    integrity = integrity_before_labels()
    opened_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    preopen = {
        "phase": "human_test_preopen_integrity",
        "integrity_completed_at": utc_now_iso(),
        "human_test_labels_opened_at": opened_at,
        "note": "Timestamp recorded after ID/status integrity and before gold label columns are used for scoring.",
        "integrity": integrity,
    }
    (OUT_DIR / "HUMAN_TEST_OPENED.json").write_text(
        json.dumps(preopen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not integrity["canonical_gold_sha256_match"]:
        raise SystemExit("canonical gold sha256 mismatch — aborting before scoring")
    if not integrity["ids_match_exclusion_manifest"]:
        raise SystemExit("human_test IDs do not match sealed exclusion manifest — aborting")
    if integrity["train_val_only_csv_contains_human_test"] != 0:
        raise SystemExit("human_test IDs leaked into train/val-only CSV — aborting")
    if integrity["human_test_ids_in_selection_val_probs"]:
        raise SystemExit("human_test IDs found in selection val_probs — aborting")

    freeze = json.loads(FREEZE_JSON.read_text())
    data = load_supervised_gold(GOLD_CSV, include_test=True)
    intent_train, _ = remap_intent(data.intent.train)
    intent_val, _ = remap_intent(data.intent.val)
    intent_test, n_dua_remap = remap_intent(data.intent.test)
    labels_intent = list(freeze["intent"]["label_order"])

    pipe = fit_frozen_intent_pipeline(intent_train)
    val_pred = pipe.predict([ex.text for ex in intent_val]).tolist()
    val_report = evaluate([ex.label for ex in intent_val], val_pred, labels=labels_intent)
    val_repro_ok = abs(val_report.macro_f1 - FROZEN_VAL_INTENT) < 1e-12

    test_pred = pipe.predict([ex.text for ex in intent_test]).tolist()
    intent_test_report = evaluate(
        [ex.label for ex in intent_test], test_pred, labels=labels_intent
    )
    intent_payload = report_payload(
        intent_test_report,
        {
            "task": "intent",
            "model_id": freeze["intent"]["model_id"],
            "split": "human_test",
            "val_macro_f1_frozen": FROZEN_VAL_INTENT,
            "val_macro_f1_reproduced": val_report.macro_f1,
            "val_reproduction_exact": val_repro_ok,
            "dua_blessing_eval_remap_count": n_dua_remap,
            "dua_blessing_eval_remap": INTENT_EVAL_REMAP,
            "label_order": labels_intent,
        },
    )
    (OUT_DIR / "intent_tfidf_logistic_test_report.json").write_text(
        json.dumps(intent_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "intent_tfidf_logistic_test_preds.json").write_text(
        json.dumps(
            {
                "comment_ids": [ex.comment_id for ex in intent_test],
                "y_true": [ex.label for ex in intent_test],
                "y_pred": test_pred,
                "labels": labels_intent,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        import joblib

        joblib.dump(pipe, OUT_DIR / "intent_tfidf_logistic.joblib")
    except Exception as exc:  # pragma: no cover
        (OUT_DIR / "intent_pipeline_save_error.txt").write_text(str(exc))

    sent_payload: dict | None = None
    try:
        sent = evaluate_pretrained_sentiment(
            data.sentiment.test,
            model_id=freeze["sentiment"]["model_id"],
            batch_size=int(freeze["sentiment"]["preprocessing"]["batch_size"]),
        )
    except Exception as exc:
        sent = None
        sent_block = {
            "task": "sentiment",
            "model_id": freeze["sentiment"]["model_id"],
            "scored": False,
            "error": f"CAMeLBERT inference unavailable: {type(exc).__name__}: {exc}",
        }
    if sent is not None:
        sent_payload = report_payload(
            sent.report,
            {
                "task": "sentiment",
                "model_id": sent.model_id,
                "split": "human_test",
                "val_macro_f1_frozen": FROZEN_VAL_SENTIMENT,
                "label_order": list(freeze["sentiment"]["label_order"]),
                "max_length": freeze["sentiment"]["preprocessing"]["max_length"],
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
        sent_block = {
            "macro_f1": sent_payload["macro_f1"],
            "accuracy": sent_payload["accuracy"],
            "val_macro_f1": FROZEN_VAL_SENTIMENT,
            "generalization_gap_macro_f1": sent_payload["macro_f1"] - FROZEN_VAL_SENTIMENT,
            "per_class_f1": sent_payload["per_class_f1"],
            "per_class_support": sent_payload["per_class_support"],
            "scored": True,
        }

    topic_block: dict = {
        "task": "topic",
        "model_id": freeze["topic"]["model_id"],
        "huggingface_id": freeze["topic"]["huggingface_id"],
        "val_macro_f1_frozen": FROZEN_VAL_TOPIC,
        "scored": False,
    }
    checkpoint = find_topic_checkpoint()
    if checkpoint is None:
        topic_block["error"] = (
            "Frozen MARBERTv2 topic checkpoint weights were not present locally. "
            "Inference was not run. Place the Colab export "
            "marbertv2_topic/best (config.json + model.safetensors) under "
            "data/models/final_test/checkpoints/marbertv2_topic/best or set "
            "RCS_MARBERTV2_TOPIC_CHECKPOINT. Do not retrain."
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
                "val_macro_f1_frozen": FROZEN_VAL_TOPIC,
                "checkpoint": topic_run["checkpoint"],
                "device": topic_run["device"],
                "label_order": topic_run["labels"],
            },
        )
        (OUT_DIR / "topic_marbertv2_test_report.json").write_text(
            json.dumps(topic_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (OUT_DIR / "topic_marbertv2_test_preds.json").write_text(
            json.dumps(
                {
                    "comment_ids": topic_run["comment_ids"],
                    "y_true": topic_run["y_true"],
                    "y_pred": topic_run["y_pred"],
                    "labels": topic_run["labels"],
                    "probabilities": topic_run["probabilities"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        topic_block = {k: v for k, v in topic_payload.items() if k != "confusion_matrix"}
        topic_block["scored"] = True
        topic_block["confusion_matrix"] = topic_payload["confusion_matrix"]
        topic_block["per_class_f1"] = topic_payload["per_class_f1"]
        topic_block["per_class_support"] = topic_payload["per_class_support"]

    summary = {
        "evaluated_at": utc_now_iso(),
        "human_test_labels_opened_at": opened_at,
        "test_rows_sealed": integrity["human_test_row_count_annotation_stage"],
        "test_rows_scored_intent": intent_payload["n_scored"],
        "test_rows_scored_sentiment": None if sent_payload is None else sent_payload["n_scored"],
        "test_rows_scored_topic": topic_block.get("n_scored"),
        "intent": {
            "macro_f1": intent_payload["macro_f1"],
            "accuracy": intent_payload["accuracy"],
            "val_macro_f1": FROZEN_VAL_INTENT,
            "generalization_gap_macro_f1": intent_payload["macro_f1"] - FROZEN_VAL_INTENT,
            "val_reproduction_exact": val_repro_ok,
            "per_class_f1": intent_payload["per_class_f1"],
            "per_class_support": intent_payload["per_class_support"],
        },
        "sentiment": sent_block,
        "topic": topic_block,
        "frozen_candidates_unchanged": {
            "intent": freeze["intent"]["model_id"],
            "topic": freeze["topic"]["model_id"],
            "sentiment": freeze["sentiment"]["model_id"],
        },
        "reselection": False,
        "retrain": False,
    }
    (OUT_DIR / "HUMAN_TEST_EVALUATION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
