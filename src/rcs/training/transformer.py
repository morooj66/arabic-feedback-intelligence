"""Fine-tune Arabic Transformer models for single-task classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rcs.evaluation.metrics import EvaluationReport, evaluate
from rcs.training.gold_splits import LabeledExample, TaskName, TASK_LABELS
from rcs.training.hf_loading import load_sequence_classifier, load_tokenizer

DEFAULT_INTENT_MODELS = {
    "marbertv2": "UBC-NLP/MARBERTv2",
    "arabertv02": "aubmindlab/bert-base-arabertv02",
}

DEFAULT_TOPIC_MODELS = {
    "marbertv2": "UBC-NLP/MARBERTv2",
    "arabertv02": "aubmindlab/bert-base-arabertv02",
}

# External GPU default: MARBERTv2 only. AraBERT is opt-in via include_arabert=True.
DEFAULT_EXTERNAL_MODEL_KEYS = ("marbertv2",)


def transformer_models_for_task(
    task: TaskName,
    *,
    include_arabert: bool = False,
) -> dict[str, str]:
    """Return model_key -> HF id map for a task (MARBERTv2-only unless include_arabert)."""
    base = DEFAULT_INTENT_MODELS if task == "intent" else DEFAULT_TOPIC_MODELS
    if include_arabert:
        return dict(base)
    return {k: base[k] for k in DEFAULT_EXTERNAL_MODEL_KEYS if k in base}

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "max_length": 128,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "num_train_epochs": 5,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 2,
    "seed": 42,
}


@dataclass
class TransformerRunResult:
    model_key: str
    model_id: str
    task: TaskName
    train_size: int
    val_size: int
    report: EvaluationReport
    y_true: list[str]
    y_pred: list[str]
    comment_ids: list[str]
    probabilities: np.ndarray
    label_list: list[str]
    output_dir: Path
    train_config: dict[str, Any] = field(default_factory=dict)
    best_val_macro_f1: float = 0.0


def _require_training_stack():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "transformers, torch, and datasets are required for fine-tuning. "
            "Install with: pip install transformers torch datasets"
        ) from exc


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compute_metrics(eval_pred, label_list: list[str]):
    from rcs.evaluation.metrics import evaluate as rcs_evaluate

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    id2label = {i: lbl for i, lbl in enumerate(label_list)}
    y_true = [id2label[int(i)] for i in labels]
    y_pred = [id2label[int(i)] for i in preds]
    report = rcs_evaluate(y_true, y_pred, labels=label_list)
    return {"macro_f1": report.macro_f1}


def _training_args_supports_eval_strategy() -> bool:
    import inspect
    from transformers import TrainingArguments

    return "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters


def finetune_transformer(
    train: list[LabeledExample],
    val: list[LabeledExample],
    *,
    task: TaskName,
    model_key: str,
    model_id: str,
    output_dir: Path,
    train_config: dict[str, Any] | None = None,
) -> TransformerRunResult:
    _require_training_stack()
    import torch
    from datasets import Dataset
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    cfg = {**DEFAULT_TRAIN_CONFIG, **(train_config or {})}
    set_seed(int(cfg["seed"]))

    label_list = TASK_LABELS[task]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    train_ds = Dataset.from_dict(
        {
            "text": [ex.text for ex in train],
            "label": [label2id[ex.label] for ex in train],
            "comment_id": [ex.comment_id for ex in train],
        }
    )
    val_ds = Dataset.from_dict(
        {
            "text": [ex.text for ex in val],
            "label": [label2id[ex.label] for ex in val],
            "comment_id": [ex.comment_id for ex in val],
        }
    )

    tokenizer = load_tokenizer(model_id)
    device = _resolve_device()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(cfg["max_length"]),
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in train_ds.column_names:
        cols.append("token_type_ids")
    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in cols]
    )
    val_ds = val_ds.remove_columns(
        [c for c in val_ds.column_names if c not in cols]
    )

    model = load_sequence_classifier(
        model_id,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        seed=int(cfg["seed"]),
        report_to=[],
        dataloader_pin_memory=False,
        **({"eval_strategy": "epoch"} if _training_args_supports_eval_strategy() else {"evaluation_strategy": "epoch"}),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda ep: _compute_metrics(ep, label_list),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["early_stopping_patience"]))],
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    predictions = trainer.predict(val_ds)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = probs.argmax(axis=-1)

    y_true = [ex.label for ex in val]
    y_pred = [id2label[int(i)] for i in pred_ids]
    comment_ids = [ex.comment_id for ex in val]
    report = evaluate(y_true, y_pred, labels=label_list)

    best_f1 = max(
        (log.get("eval_macro_f1", 0.0) for log in trainer.state.log_history if "eval_macro_f1" in log),
        default=report.macro_f1,
    )

    full_config = {
        **cfg,
        "model_key": model_key,
        "model_id": model_id,
        "task": task,
        "device": device,
        "label_list": label_list,
    }

    return TransformerRunResult(
        model_key=model_key,
        model_id=model_id,
        task=task,
        train_size=len(train),
        val_size=len(val),
        report=report,
        y_true=y_true,
        y_pred=y_pred,
        comment_ids=comment_ids,
        probabilities=probs,
        label_list=label_list,
        output_dir=output_dir,
        train_config=full_config,
        best_val_macro_f1=float(best_f1),
    )
