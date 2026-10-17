"""Verify Google Colab training bundle contents and invariants."""

from __future__ import annotations

import hashlib
import importlib.util
import io
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
PACK_SCRIPT = PROJECT_ROOT / "scripts" / "pack_colab_training_bundle.py"


def _load_pack():
    spec = importlib.util.spec_from_file_location("pack_colab_training_bundle", PACK_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_colab_bundle_excludes_kaggle_paths():
    pack = _load_pack()
    joined = "\n".join(pack.BUNDLE_REL_PATHS).lower()
    assert "kaggle/" not in joined
    assert "/kaggle/" not in joined
    assert any("colab/setup_gpu_env.py" in p for p in pack.BUNDLE_REL_PATHS)
    assert any("colab/preflight.py" in p for p in pack.BUNDLE_REL_PATHS)


def test_colab_bundle_file_list_excludes_secrets():
    pack = _load_pack()
    for forbidden in pack.FORBIDDEN_FRAGMENTS:
        assert forbidden.lower() not in "\n".join(pack.BUNDLE_REL_PATHS).lower()


def test_colab_setup_no_kaggle_paths():
    text = (PROJECT_ROOT / "colab" / "setup_gpu_env.py").read_text(encoding="utf-8")
    assert "/kaggle/" not in text
    assert "kaggle/" not in text
    assert "/content/" in text


def test_colab_pack_builds_zip(tmp_path):
    pack = _load_pack()
    if not CANONICAL_CSV.exists():
        pytest.skip("canonical gold CSV not present")
    out_zip = tmp_path / "colab.zip"
    summary = pack.build_bundle(output_zip=out_zip)
    assert summary["supervised_train_rows"] == 455
    assert summary["supervised_val_rows"] == 108
    assert summary["human_test_rows_excluded"] == 75

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        assert not any("kaggle/" in n for n in names)
        assert not any("gold_dataset_round1.csv" in n for n in names)
        assert any(n.endswith("gold_train_val_only.csv") for n in names)

        gold_member = next(n for n in names if n.endswith("gold_train_val_only.csv"))
        frame = pd.read_csv(io.BytesIO(zf.read(gold_member)), dtype=str, keep_default_na=False)
        assert (frame["annotation_stage"] == "human_test").sum() == 0

        for name in names:
            if name.endswith((".py", ".md", ".json", ".ipynb", ".csv", ".txt")):
                content = zf.read(name).decode("utf-8", errors="replace")
                assert "/Users/" not in content
                assert "b3_human_review.json" not in content


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="canonical gold CSV not present")
def test_canonical_gold_unchanged():
    digest = hashlib.sha256(CANONICAL_CSV.read_bytes()).hexdigest()
    assert digest == CANONICAL_GOLD_SHA256


def test_colab_torch_matrix():
    spec = importlib.util.spec_from_file_location(
        "setup_gpu_env", PROJECT_ROOT / "colab" / "setup_gpu_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.select_torch_build("6.0")["index_url"].endswith("cu118")
    assert mod.select_torch_build("7.5")["index_url"].endswith("cu121")
