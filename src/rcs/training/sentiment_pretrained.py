"""Pretrained Arabic sentiment model — zero-shot / inference-only baseline."""

from __future__ import annotations

from dataclasses import dataclass

from config import settings
from rcs.evaluation.metrics import EvaluationReport, evaluate
from rcs.training.gold_splits import LabeledExample
from rcs.training.hf_loading import load_sequence_classifier, load_tokenizer

DEFAULT_SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"

# Map Hugging Face id2label strings to project taxonomy.
_SENTIMENT_ALIASES = {
    "positive": "POSITIVE",
    "pos": "POSITIVE",
    "negative": "NEGATIVE",
    "neg": "NEGATIVE",
    "neutral": "NEUTRAL",
    "neu": "NEUTRAL",
    "POSITIVE": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "NEUTRAL": "NEUTRAL",
}


@dataclass
class SentimentBaselineResult:
    model_id: str
    val_size: int
    report: EvaluationReport
    y_true: list[str]
    y_pred: list[str]
    comment_ids: list[str]
    probabilities: list[list[float]] | None


def _normalize_sentiment(raw: str) -> str:
    key = raw.strip()
    mapped = _SENTIMENT_ALIASES.get(key) or _SENTIMENT_ALIASES.get(key.lower())
    if mapped is None:
        raise ValueError(f"unrecognized sentiment label from model: {raw!r}")
    return mapped


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for the sentiment baseline. "
            "Install with: pip install transformers torch"
        ) from exc


def evaluate_pretrained_sentiment(
    val: list[LabeledExample],
    *,
    model_id: str = DEFAULT_SENTIMENT_MODEL,
    batch_size: int = 16,
    device: str | None = None,
) -> SentimentBaselineResult:
    _require_transformers()
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = load_tokenizer(model_id)
    model = load_sequence_classifier(model_id)
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_order = list(settings.SENTIMENTS)

    y_true: list[str] = []
    y_pred: list[str] = []
    comment_ids: list[str] = []
    all_probs: list[list[float]] = []

    texts = [ex.text for ex in val]
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_examples = val[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i, ex in enumerate(batch_examples):
            pred_idx = int(probs[i].argmax())
            raw_label = id2label[pred_idx]
            y_true.append(ex.label)
            y_pred.append(_normalize_sentiment(raw_label))
            comment_ids.append(ex.comment_id)
            # Reorder probs to project label order where possible.
            row_probs = []
            for project_label in label_order:
                idx = next(
                    (j for j, lbl in id2label.items() if _normalize_sentiment(lbl) == project_label),
                    None,
                )
                row_probs.append(float(probs[i][idx]) if idx is not None else 0.0)
            total = sum(row_probs) or 1.0
            all_probs.append([p / total for p in row_probs])

    report = evaluate(y_true, y_pred, labels=label_order)
    return SentimentBaselineResult(
        model_id=model_id,
        val_size=len(val),
        report=report,
        y_true=y_true,
        y_pred=y_pred,
        comment_ids=comment_ids,
        probabilities=all_probs,
    )
