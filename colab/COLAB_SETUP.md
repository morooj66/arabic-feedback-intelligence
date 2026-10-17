# RCS Model Selection — Google Colab

Self-contained MARBERTv2 training bundle. Uses existing `scripts/run_model_selection.py`.

## Build bundle (local Mac)

```bash
python scripts/pack_colab_training_bundle.py
```

Upload **`dist/rcs_colab_training_bundle.zip`** to Colab (Files panel or Google Drive).

## GPU compatibility

| GPU | Compute cap | PyTorch index | Version |
|-----|-------------|---------------|---------|
| P100 | 6.0 | cu118 | 2.5.1 |
| T4 | 7.5 | cu121 | 2.5.1 |
| Other | — | cu118 fallback | 2.5.1 |

No `torchvision` / `torchaudio`.

## Colab cells (in order)

See `colab/rcs_model_selection.ipynb` for the full notebook.

1. **Extract** → `/content/rcs_project`
2. **Install** → `python colab/setup_gpu_env.py --install`
3. **Restart runtime** (mandatory)
4. **Verify** → `python colab/setup_gpu_env.py --verify`
5. **Preflight** → `python colab/preflight.py`
6. **Train** → `python scripts/run_model_selection.py ...`
7. **Package** → `python colab/package_artifacts.py`

### Sentiment model (CAMeLBERT) and safetensors

Hub repos may ship only `pytorch_model.bin`. With torch 2.5.1, loaders require
safetensors (see `src/rcs/training/hf_loading.py`). If sentiment preflight fails,
run a one-time conversion (temporary torch 2.6), then reinstall torch 2.5.1:

```bash
pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cu121
python colab/convert_hub_weights_to_safetensors.py \
  --model-id CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
python colab/setup_gpu_env.py --install
# restart runtime, then re-run preflight
```

Or skip sentiment for the first run: add `--skip-sentiment` to training command.

## Full training command

```bash
python scripts/run_model_selection.py \
  --gold-csv data/gold/gold_train_val_only.csv \
  --output-dir /content/selection \
  --skip-classical
```

## Persistence & disconnection

| Event | Survives? |
|-------|-----------|
| Browser disconnect (runtime still alive) | Yes — `/content/selection/` keeps growing |
| **Runtime disconnect / crash / reset** | **No** — `/content/` wiped except Drive mounts |
| Completed files before reset | Lost unless copied to Drive or downloaded |

### Optional Google Drive backup

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/rcs_selection
!rsync -a /content/selection/ /content/drive/MyDrive/rcs_selection/
```

### Checkpoint / resume (current behavior)

- HF checkpoints saved under `{output_dir}/marbertv2_*/checkpoints/` during a run.
- **`run_model_selection.py` does not resume** interrupted runs; re-run restarts each model task.
- To preserve partial work: copy `/content/selection/` to Drive before reset, or use a fresh `--output-dir` per attempt.

## Download artifacts

```bash
python colab/package_artifacts.py
```

```python
from google.colab import files
files.download("/content/rcs_selection_artifacts.zip")
```

Merge locally into `data/models/selection/`.
