"""Score prediction files against local gold labels and write comparison_report.md (local)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_common import OUT_DIR  # noqa: E402
from jev_metrics import evaluate, to_markdown  # noqa: E402


def _load_preds(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        recs = [json.loads(l) for l in text.splitlines() if l.strip()]
        return {r["eval_id"]: r for r in recs}
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rcs", default=str(OUT_DIR / "rcs_predictions.json"))
    ap.add_argument("--jev", default=str(OUT_DIR / "jev_predictions.jsonl"))
    args = ap.parse_args()
    gold = json.loads((OUT_DIR / "gold_labels.local.json").read_text())
    parts = ["# Development-validation comparison (local; sanitized inputs; val split only)\n"]
    for name, p in (("Frozen RCS", Path(args.rcs)), ("Jev", Path(args.jev))):
        if p.exists():
            parts.append(to_markdown(name, evaluate(gold, _load_preds(p))))
        else:
            parts.append(f"## {name}\n\nNo predictions file ({p.name}); not run yet.\n")
    (OUT_DIR / "comparison_report.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
