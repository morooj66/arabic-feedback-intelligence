#!/usr/bin/env python3
"""
Strict preflight before Colab training. Read-only on gold labels.

Checks: CUDA, GPU info, ML imports, MARBERTv2 forward pass, train/val splits,
human_test exclusion, taxonomy alignment, no train/val overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold" / "gold_train_val_only.csv"


def run_preflight(*, gold_csv: Path) -> dict:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            raise SystemExit(f"PREFLIGHT FAIL: {name} — {detail}")

    # CUDA + tensor
    try:
        import torch
    except ImportError as exc:
        check("torch_import", False, str(exc))
    check("cuda_available", torch.cuda.is_available(), "enable GPU runtime and restart after setup")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        x = torch.tensor([1.0], device="cuda") * 3
        check("cuda_tensor", float(x.item()) == 3.0, "cuda tensor op failed")
        gpu_detail = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "total_memory_gib": round(props.total_memory / (1024**3), 2),
        }
    else:
        gpu_detail = {}
    check("gpu_info", bool(gpu_detail), json.dumps(gpu_detail))

    # ML stack imports
    for mod in ("transformers", "datasets", "accelerate"):
        try:
            __import__(mod)
        except ImportError as exc:
            check(f"import_{mod}", False, str(exc))
        check(f"import_{mod}", True)

    # Gold splits
    from config import settings
    from rcs.training.gold_splits import TASK_LABELS, load_supervised_gold

    if not gold_csv.exists():
        check("gold_csv_exists", False, str(gold_csv))
    data = load_supervised_gold(gold_csv, include_test=False)
    check("train_rows_455", data.train_rows == 455, str(data.train_rows))
    check("val_rows_108", data.val_rows == 108, str(data.val_rows))
    check("test_rows_0", data.test_rows == 0, str(data.test_rows))
    check("intent_test_empty", len(data.intent.test) == 0)

    train_ids = {ex.comment_id for ex in data.intent.train}
    val_ids = {ex.comment_id for ex in data.intent.val}
    check("no_train_val_overlap", len(train_ids & val_ids) == 0, str(len(train_ids & val_ids)))

    check("taxonomy_intent", TASK_LABELS["intent"] == list(settings.PRIMARY_INTENTS))
    check("taxonomy_topic", TASK_LABELS["topic"] == list(settings.TOPIC_SUBTYPES))
    check("taxonomy_sentiment", TASK_LABELS["sentiment"] == list(settings.SENTIMENTS))

    for task, allowed in TASK_LABELS.items():
        bundle = getattr(data, task)
        for ex in bundle.train + bundle.val:
            if ex.label not in allowed:
                check(f"label_valid_{task}", False, f"{ex.comment_id}: {ex.label}")

    # human_test in CSV file
    import pandas as pd

    frame = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
    human_test_count = int((frame["annotation_stage"] == "human_test").sum())
    check("human_test_absent_in_csv", human_test_count == 0, str(human_test_count))

    # MARBERTv2-only default
    from rcs.training.transformer import transformer_models_for_task

    check(
        "marbert_only_default",
        list(transformer_models_for_task("intent").keys()) == ["marbertv2"],
    )

    # MARBERTv2 smoke (CUDA forward pass)
    import importlib.util

    smoke_path = PROJECT_ROOT / "scripts" / "gpu_smoke_test.py"
    spec = importlib.util.spec_from_file_location("gpu_smoke_test", smoke_path)
    smoke_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(smoke_mod)
    smoke = smoke_mod.run_smoke()
    check("marbert_smoke_ok", smoke.get("ok") is True, json.dumps(smoke))
    check("marbert_smoke_cuda", smoke.get("device") == "cuda", smoke.get("device", ""))

    return {
        "ok": True,
        "gpu": gpu_detail,
        "train_rows": data.train_rows,
        "val_rows": data.val_rows,
        "human_test_rows_in_csv": human_test_count,
        "marbert_smoke": smoke,
        "checks_passed": len(checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab training preflight.")
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=DEFAULT_GOLD,
        help="Train+val-only gold CSV (default: data/gold/gold_train_val_only.csv)",
    )
    args = parser.parse_args()
    result = run_preflight(gold_csv=args.gold_csv)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
