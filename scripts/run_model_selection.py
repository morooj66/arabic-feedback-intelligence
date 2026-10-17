#!/usr/bin/env python3
"""
Phase C–G orchestrator: classical baseline, sentiment baseline, transformer
fine-tuning, and validation-based model selection.

Uses frozen gold splits only. Never opens human_test for tuning/selection.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rcs.training.artifacts import (  # noqa: E402
    SELECTION_DIR,
    report_to_dict,
    save_evaluation_report,
    save_training_config,
    save_validation_probabilities,
    utc_now_iso,
)
from rcs.training.classical import train_evaluate_classical  # noqa: E402
from rcs.training.gold_splits import load_supervised_gold  # noqa: E402
from rcs.training.sentiment_pretrained import evaluate_pretrained_sentiment  # noqa: E402
from rcs.training.transformer import (  # noqa: E402
    DEFAULT_TRAIN_CONFIG,
    finetune_transformer,
    transformer_models_for_task,
)


def _environment_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        info["torch"] = None
    try:
        import sklearn

        info["sklearn"] = sklearn.__version__
    except ImportError:
        info["sklearn"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    return info


def _distribution_section(data) -> dict:
    return {
        "train_rows": data.train_rows,
        "val_rows": data.val_rows,
        "train_excluded_insufficient": data.train_excluded_insufficient,
        "val_excluded_insufficient": data.val_excluded_insufficient,
        "val_excluded_unlabeled": data.val_excluded_unlabeled,
        "val_excluded_details": data.val_excluded_details,
        "intent_train": data.intent.train_distribution(),
        "intent_val": data.intent.val_distribution(),
        "topic_train": data.topic.train_distribution(),
        "topic_val": data.topic.val_distribution(),
        "sentiment_train": data.sentiment.train_distribution(),
        "sentiment_val": data.sentiment.val_distribution(),
        "imbalance_flags": data.imbalance_flags,
    }


def run_model_selection(
    *,
    gold_csv_path: Path | None = None,
    output_dir: Path = SELECTION_DIR,
    skip_classical: bool = False,
    skip_transformers: bool = False,
    skip_sentiment: bool = False,
    include_arabert: bool = False,
    quick: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _environment_info()
    data = load_supervised_gold(gold_csv_path, include_test=False)
    distributions = _distribution_section(data)

    results: dict = {
        "started_at": utc_now_iso(),
        "gold_csv": str(gold_csv_path or "default"),
        "environment": env,
        "distributions": distributions,
        "models": {},
        "selection": {},
    }

    train_cfg = dict(DEFAULT_TRAIN_CONFIG)
    if quick:
        train_cfg["num_train_epochs"] = 2
        train_cfg["early_stopping_patience"] = 1

    # --- Classical baselines (intent + topic + sentiment via TF-IDF) ---
    if not skip_classical:
        for task in ("intent", "topic", "sentiment"):
            bundle = getattr(data, task)
            for clf in ("svm", "logistic"):
                key = f"tfidf_{clf}_{task}"
                run = train_evaluate_classical(
                    bundle.train,
                    bundle.val,
                    task=task,
                    classifier=clf,
                )
                report_path = output_dir / f"{key}_val_report.json"
                save_evaluation_report(
                    run.report,
                    report_path,
                    extra={
                        "model": key,
                        "classifier": clf,
                        "task": task,
                        "train_size": run.train_size,
                        "val_size": run.val_size,
                    },
                )
                results["models"][key] = {
                    "macro_f1": run.report.macro_f1,
                    "report_path": str(report_path),
                }

        primary_classical_key = "tfidf_svm_intent"
        results["classical_benchmark"] = results["models"][primary_classical_key]
    else:
        results["classical_benchmark"] = {"skipped": True}

    # --- Pretrained Arabic sentiment ---
    if not skip_sentiment:
        try:
            sent = evaluate_pretrained_sentiment(data.sentiment.val)
            sent_path = output_dir / "sentiment_pretrained_val_report.json"
            save_evaluation_report(
                sent.report,
                sent_path,
                extra={"model_id": sent.model_id, "val_size": sent.val_size},
            )
            if sent.probabilities:
                save_validation_probabilities(
                    output_dir / "sentiment_pretrained_val_probs.json",
                    comment_ids=sent.comment_ids,
                    labels=list(sent.report.labels),
                    probabilities=__import__("numpy").array(sent.probabilities),
                    y_true=sent.y_true,
                    y_pred=sent.y_pred,
                )
            results["models"]["sentiment_pretrained"] = {
                "model_id": sent.model_id,
                "macro_f1": sent.report.macro_f1,
                "report_path": str(sent_path),
            }
        except ImportError as exc:
            results["models"]["sentiment_pretrained"] = {"error": str(exc)}

    # --- Transformer fine-tuning ---
    transformer_runs: dict[str, dict] = {}
    if not skip_transformers:
        for task in ("intent", "topic"):
            model_map = transformer_models_for_task(task, include_arabert=include_arabert)
            bundle = getattr(data, task)
            for model_key, model_id in model_map.items():
                run_key = f"{model_key}_{task}"
                run_dir = output_dir / run_key
                try:
                    run = finetune_transformer(
                        bundle.train,
                        bundle.val,
                        task=task,
                        model_key=model_key,
                        model_id=model_id,
                        output_dir=run_dir,
                        train_config=train_cfg,
                    )
                    report_path = run_dir / "val_report.json"
                    save_evaluation_report(
                        run.report,
                        report_path,
                        extra={
                            "model_key": model_key,
                            "model_id": model_id,
                            "task": task,
                            "best_val_macro_f1": run.best_val_macro_f1,
                        },
                    )
                    save_training_config(run_dir / "train_config.json", run.train_config)
                    save_validation_probabilities(
                        run_dir / "val_probs.json",
                        comment_ids=run.comment_ids,
                        labels=run.label_list,
                        probabilities=run.probabilities,
                        y_true=run.y_true,
                        y_pred=run.y_pred,
                    )
                    transformer_runs[run_key] = {
                        "model_id": model_id,
                        "task": task,
                        "macro_f1": run.report.macro_f1,
                        "best_val_macro_f1": run.best_val_macro_f1,
                        "report_path": str(report_path),
                        "artifact_dir": str(run_dir),
                    }
                except ImportError as exc:
                    transformer_runs[run_key] = {"error": str(exc)}
                except Exception as exc:
                    transformer_runs[run_key] = {"error": f"{type(exc).__name__}: {exc}"}
        results["models"].update(transformer_runs)

    # --- Model selection on Validation Macro-F1 ---
    def _best(keys: list[str]) -> tuple[str, float]:
        best_key = ""
        best_score = float("-inf")
        for key in keys:
            entry = results["models"].get(key, {})
            score = entry.get("macro_f1")
            if score is not None and score > best_score:
                best_key = key
                best_score = float(score)
        return best_key, best_score

    intent_candidates = [
        k for k in results["models"] if k.endswith("_intent") and "macro_f1" in results["models"][k]
    ]
    topic_candidates = [
        k for k in results["models"] if k.endswith("_topic") and "macro_f1" in results["models"][k]
    ]

    selected_intent, intent_f1 = _best(intent_candidates)
    selected_topic, topic_f1 = _best(topic_candidates)

    results["selection"] = {
        "selected_intent_model": selected_intent,
        "selected_intent_val_macro_f1": intent_f1,
        "selected_topic_model": selected_topic,
        "selected_topic_val_macro_f1": topic_f1,
        "sentiment_baseline": results["models"].get("sentiment_pretrained", {}),
        "classical_benchmark_macro_f1": results["classical_benchmark"].get("macro_f1"),
    }

    # Comparison table
    comparison = []
    for key, entry in sorted(results["models"].items()):
        if "macro_f1" not in entry:
            continue
        comparison.append(
            {
                "model": key,
                "macro_f1": entry["macro_f1"],
                "model_id": entry.get("model_id"),
                "task": (
                    "sentiment"
                    if key == "sentiment_pretrained"
                    else key.rsplit("_", 1)[-1]
                ),
            }
        )
    comparison.sort(key=lambda row: row["macro_f1"], reverse=True)
    results["comparison_table"] = comparison
    results["finished_at"] = utc_now_iso()

    summary_path = output_dir / "model_selection_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    txt_lines = [
        "Model Selection Summary (Validation Macro-F1)",
        "=" * 60,
        f"Train rows: {data.train_rows} | Val rows: {data.val_rows}",
        "",
        "Comparison (Validation Macro-F1, descending):",
    ]
    for row in comparison:
        txt_lines.append(f"  {row['model']:30} {row['macro_f1']:.4f}")
    txt_lines.extend(
        [
            "",
            f"Selected Intent: {selected_intent} ({intent_f1:.4f})",
            f"Selected Topic:  {selected_topic} ({topic_f1:.4f})",
            "",
            f"Classical benchmark (TF-IDF+SVM intent): {results['classical_benchmark'].get('macro_f1', 'skipped')}",
        ]
    )
    (output_dir / "model_selection_summary.txt").write_text(
        "\n".join(txt_lines) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baselines and transformer model selection.")
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=None,
        help="Path to canonical gold CSV (default: data/gold/gold_dataset_final_training.csv).",
    )
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip TF-IDF baselines (transformer/sentiment only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SELECTION_DIR,
        help="Directory for metrics artifacts and fine-tuned checkpoints.",
    )
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Skip MARBERTv2/AraBERT fine-tuning (classical + sentiment only).",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip pretrained Arabic sentiment baseline.",
    )
    parser.add_argument(
        "--include-arabert",
        action="store_true",
        help="Also fine-tune AraBERTv02 (default: MARBERTv2 only).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer transformer epochs for smoke testing.",
    )
    args = parser.parse_args()
    results = run_model_selection(
        gold_csv_path=args.gold_csv,
        output_dir=args.output_dir,
        skip_classical=args.skip_classical,
        skip_transformers=args.skip_transformers,
        skip_sentiment=args.skip_sentiment,
        include_arabert=args.include_arabert,
        quick=args.quick,
    )
    print(json.dumps(results["selection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
