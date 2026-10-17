# RCS Model Selection — Kaggle GPU Setup

MARBERTv2-only training on frozen train/val data (`gold_train_val_only.csv`).

## GPU compatibility (P100 / T4)

Kaggle preinstalls **PyTorch cu128**, which **does not support P100 (sm_60)**.
Use `kaggle/setup_gpu_env.py` to install a compatible wheel before training.

| GPU | Compute Cap | PyTorch wheel | Version |
|-----|-------------|---------------|---------|
| NVIDIA P100 | 6.0 (sm_60) | `cu118` index | 2.5.1 |
| NVIDIA T4 | 7.5 (sm_75) | `cu121` index | 2.5.1 |
| Other / unknown | — | `cu118` fallback | 2.5.1 |

`torchvision` / `torchaudio` are **not** installed (not required for training).

## Kaggle notebook cells

### Cell 0 — Copy bundle to working dir

```python
import os, shutil
from pathlib import Path
INPUT = Path("/kaggle/input/rcs-training-bundle/rcs_project")
WORK = Path("/kaggle/working/rcs_project")
if INPUT.exists():
    if WORK.exists(): shutil.rmtree(WORK)
    shutil.copytree(INPUT, WORK)
else:
    WORK = Path(".").resolve()
os.chdir(WORK)
os.environ["PYTHONPATH"] = "src:."
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
GOLD_CSV = "data/gold/gold_train_val_only.csv"
```

### Cell 1 — Install compatible stack

```bash
python kaggle/setup_gpu_env.py --install
```

**Restart kernel/session required** after this cell.

### Cell 2 — After restart: verify + smoke test

```python
import os
from pathlib import Path
WORK = Path("/kaggle/working/rcs_project")
if WORK.exists(): os.chdir(WORK)
os.environ["PYTHONPATH"] = "src:."
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
```

```bash
python kaggle/setup_gpu_env.py --verify
python scripts/gpu_smoke_test.py
```

Expect `"cuda_available": true` and smoke test `"device": "cuda"`.

## Full training (after smoke test)

```bash
python scripts/run_model_selection.py \
  --gold-csv data/gold/gold_train_val_only.csv \
  --output-dir /kaggle/working/selection \
  --skip-classical
```

## Build bundle locally

```bash
python scripts/pack_kaggle_training_bundle.py
```

Upload `dist/rcs_kaggle_training_bundle.zip` to Kaggle.
