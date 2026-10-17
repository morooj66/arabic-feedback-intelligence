# Model artifacts and restricted data

None of the following is stored in Git.

| Item | Location | Access |
|---|---|---|
| Fine-tuned topic checkpoint (≈ 650 MB), fitted intent classifier, selection metadata | private model repository (Hugging Face) named by `HF_MODEL_REPO_ID`, pinned by `HF_MODEL_REVISION` and checksummed in `config/frozen_models_manifest.json` | read-only `HF_TOKEN` issued by the repository owner |
| Sentiment model `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` (≈ 440 MB) | public Hub, pinned revision in the manifest | none |
| Visitor datasets, gold labels, evaluation outputs, review databases, uploaded workbooks | institution-controlled storage | granted by the data owner; see the institution's retention policy |
| Credentials (database, Gemini, SMTP, Entra, tokens) | the host's secret manager / environment | operations owner |

**Provision a machine:** `HF_MODEL_REPO_ID=… HF_MODEL_REVISION=… HF_TOKEN=… python scripts/fetch_frozen_models.py` (verifies every file; any mismatch is fatal). Production never downloads a model at request time and refuses to start if a pinned file is missing.
**Regenerate nothing silently:** models are frozen. A retrain or re-freeze needs a new manifest revision and a documented evaluation.
**Tests without artifacts:** the suite skips what needs them; run the full suite on a machine that has them.
