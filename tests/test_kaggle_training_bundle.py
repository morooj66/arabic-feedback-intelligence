"""Verify Kaggle training bundle contents and gold-split invariants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CSV = PROJECT_ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
CANONICAL_GOLD_SHA256 = (
    "ac05bb8dc47a9787e658049d8ac650cd0a319236d201698e3196c166d7cf1227"
)

PACK_SCRIPT = PROJECT_ROOT / "scripts" / "pack_kaggle_training_bundle.py"


def _load_pack_module():
    spec = importlib.util.spec_from_file_location("pack_kaggle_training_bundle", PACK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="canonical gold CSV not present")
def test_canonical_gold_csv_hash_unchanged():
    assert _sha256(CANONICAL_CSV) == CANONICAL_GOLD_SHA256


def test_bundle_file_list_excludes_secrets():
    pack = _load_pack_module()
    joined = "\n".join(pack.BUNDLE_REL_PATHS).lower()
    for forbidden in pack.FORBIDDEN_FRAGMENTS:
        assert forbidden.lower() not in joined, f"forbidden fragment listed: {forbidden}"
    assert any("gold_train_val_only.csv" in p for p in pack.BUNDLE_REL_PATHS)
    assert not any("gold_dataset_round1.csv" in p for p in pack.BUNDLE_REL_PATHS)
    assert any("setup_gpu_env.py" in p for p in pack.BUNDLE_REL_PATHS)


def test_export_train_val_excludes_human_test(tmp_path):
    pack = _load_pack_module()
    if not CANONICAL_CSV.exists():
        pytest.skip("canonical gold CSV not present")
    out = tmp_path / "train_val.csv"
    stats = pack.export_train_val_gold_csv(output_csv=out)
    gold = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert stats["human_test_rows_excluded"] == 75
    assert (gold["annotation_stage"] == "human_test").sum() == 0
    assert stats["supervised_train_rows"] == 455
    assert stats["supervised_val_rows"] == 108


def test_pack_script_builds_zip(tmp_path):
    pack = _load_pack_module()
    if not CANONICAL_CSV.exists():
        pytest.skip("canonical gold CSV not present")
    out_zip = tmp_path / "bundle.zip"
    summary = pack.build_bundle(output_zip=out_zip)
    assert out_zip.exists()
    assert summary["source_canonical_gold_sha256"] == CANONICAL_GOLD_SHA256
    assert summary["human_test_rows_excluded"] == 75

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        assert any(n.endswith("gold_train_val_only.csv") for n in names)
        assert not any("gold_dataset_round1.csv" in n for n in names)
        assert not any("b3_human_review.json" in n for n in names)
        assert not any("b3_train_gold_complete.json" in n for n in names)
        assert not any(".env" in n for n in names)

        gold_member = next(n for n in names if n.endswith("gold_train_val_only.csv"))
        gold_bytes = zf.read(gold_member)
        import io

        frame = pd.read_csv(io.BytesIO(gold_bytes), dtype=str, keep_default_na=False)
        assert (frame["annotation_stage"] == "human_test").sum() == 0

        for name in names:
            if name.endswith((".py", ".md", ".json", ".ipynb", ".csv", ".txt")):
                content = zf.read(name).decode("utf-8", errors="replace")
                assert "/Users/" not in content
                assert "b3_human_review.json" not in content

    manifest_path = out_zip.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["human_test_policy"].startswith("excluded")


def test_marbert_only_by_default():
    from rcs.training.transformer import transformer_models_for_task

    assert list(transformer_models_for_task("intent").keys()) == ["marbertv2"]
    assert list(transformer_models_for_task("topic").keys()) == ["marbertv2"]
    assert "arabertv02" in transformer_models_for_task("intent", include_arabert=True)


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="canonical gold CSV not present")
def test_supervised_splits_unchanged_on_canonical():
    from rcs.training.gold_splits import load_supervised_gold

    data = load_supervised_gold(CANONICAL_CSV, include_test=False)
    assert data.train_rows == 455
    assert data.val_rows == 108
    assert data.test_rows == 0


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="canonical gold CSV not present")
def test_run_model_selection_module_importable():
    spec = importlib.util.spec_from_file_location(
        "run_model_selection",
        PROJECT_ROOT / "scripts" / "run_model_selection.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "run_model_selection")
