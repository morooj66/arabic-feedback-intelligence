"""Persist training configs, evaluation reports, and validation probabilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import settings
from rcs.evaluation.metrics import EvaluationReport

MODELS_DIR = settings.DATA_DIR / "models"
SELECTION_DIR = MODELS_DIR / "selection"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    return {
        "macro_f1": report.macro_f1,
        "labels": report.labels,
        "per_class_precision": report.per_class_precision,
        "per_class_recall": report.per_class_recall,
        "per_class_f1": report.per_class_f1,
        "per_class_support": report.per_class_support,
        "confusion_matrix": report.confusion_matrix,
    }


def save_evaluation_report(
    report: EvaluationReport,
    path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = report_to_dict(report)
    if extra:
        payload.update(extra)
    _atomic_write_json(path, payload)
    return path


def save_validation_probabilities(
    path: Path,
    *,
    comment_ids: list[str],
    labels: list[str],
    probabilities: np.ndarray,
    y_true: list[str],
    y_pred: list[str],
) -> Path:
    """Save softmax probabilities for later threshold calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comment_ids": comment_ids,
        "labels": labels,
        "probabilities": probabilities.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    _atomic_write_json(path, payload)
    return path


def save_training_config(path: Path, config: dict[str, Any]) -> Path:
    _atomic_write_json(path, config)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    return obj
