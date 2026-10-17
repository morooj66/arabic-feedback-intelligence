# AI / ML pipeline

## 1. Ingest and prefilter
- `src/rcs/ingest`: workbook loading, header normalisation, text cleaning. A comment's identity is `sha256(source_file | row_index)`; a different file under the same name is imported as a separate batch (`import_identity.py`), an identical re-upload is idempotent, invalid rows are skipped and reported, in-file duplicates are kept but marked non-reviewable.
- `src/rcs/prefilter/spam_rules.py`: deterministic junk rules (known junk strings, unfilled templates, phone-number-only stubs, test markers, near-empty content, bursts/repetition). Rows are never deleted; they are flagged and routed to `EXCLUDE`.

## 2. Classification (three frozen models)
| Task | Labels | Model |
|---|---|---|
| Intent | `THANKS_PRAISE, COMPLAINT, SUGGESTION, INQUIRY, SPAM` | TF-IDF + balanced logistic regression |
| Topic | `WORD_SUBMISSION, ENTRY_CORRECTION, DICTIONARY_SUGGESTION, TECHNICAL_ISSUE, WORD_OF_YEAR_NOMINATION, OFF_TOPIC_PERSONAL, GENERAL` | fine-tuned MARBERTv2 |
| Sentiment | `POSITIVE, NEGATIVE, NEUTRAL` | CAMeLBERT dialectal sentiment model, pinned to an exact revision |
"Frozen" means: loaded from a verified local copy at a pinned revision with file checksums (`config/frozen_models_manifest.json`, `src/rcs/training/hf_loading.py`); in production a missing or mismatching artifact fails closed — nothing is downloaded at request time and no substitute model is used. Training/selection code and notebooks: `src/rcs/training`, `scripts/run_model_selection.py`, `colab/`, `kaggle/`.

## 3. Routing and context signals
`src/rcs/routing/engine.py` maps (intent, topic, …) to a routing decision (`GENERATE_DRAFT`, `REVIEW_ONLY`, `NO_RESPONSE_RECOMMENDED`, `EXCLUDE`). `src/rcs/classification/context_signals.py` adds deterministic corrections/flags (for example praise predicted as spam). Classification and routing are separate, independently tested steps.

## 4. Response generation (LLM)
1. **Eligibility** (`response_generation/eligibility.py`): an uncorroborated `EXCLUDE`, or a factual question without grounding, never reaches the model.
2. **Privacy boundary**: PII is redacted and the comment is wrapped as *untrusted visitor input*; prompts are guarded before leaving the process.
3. **Draft**: schema-constrained JSON from the provider (Google Gen AI SDK; the provider is injectable, so tests use a local double).
4. **Validation gates** (`validate.py`, `claim_gates.py`): content shape, internal-text leakage, PII, boilerplate, evaluator voice, pronouns, unsupported claims, and visitor-intent attribution.
5. **One bounded repair**, then **fail closed** to `REVIEW_ONLY` (no draft is invented).
Prompt version `rg-v14.4`. The same code path is exercised end to end in `tests/test_full_pipeline_integration.py` with real classifiers and a mocked provider.

## 5. What the pipeline deliberately does not do
It does not auto-send, does not let the model see personal data, does not let classification overrule a human, and does not claim a reply is good: that judgement belongs to the reviewer.
