# Models and evaluation

## Taxonomy
Intent (5): `THANKS_PRAISE, COMPLAINT, SUGGESTION, INQUIRY, SPAM` · Topic (7): `WORD_SUBMISSION, ENTRY_CORRECTION, DICTIONARY_SUGGESTION, TECHNICAL_ISSUE, WORD_OF_YEAR_NOMINATION, OFF_TOPIC_PERSONAL, GENERAL` · Sentiment (3): `POSITIVE, NEGATIVE, NEUTRAL`.

## Data and protocol
Human-labelled visitor comments (restricted; not distributed). Candidate models were trained on 455 comments and compared on a 108-comment validation set; the selected models were frozen and then scored **once** on a sealed test set of 75 comments (68 scored; 6 marked `INSUFFICIENT_CONTEXT` and 1 unlabeled were excluded). Test comments do not overlap training or validation. No hyperparameters or thresholds were tuned on the test set. Metric: macro-F1 (`src/rcs/evaluation/metrics.py`). Selection code: `src/rcs/training`, `scripts/run_model_selection.py`, `colab/`, `kaggle/`.

## Model selection (validation, n = 108, macro-F1)
| Task | Candidates | Selected |
|---|---|---|
| Intent | **TF-IDF + logistic regression 0.315** · TF-IDF + SVM 0.263 · MARBERTv2 0.249 | TF-IDF + logistic regression |
| Topic | **MARBERTv2 0.402** (0.469 excluding the unrepresented class) · TF-IDF + logistic 0.347 · TF-IDF + SVM 0.320 | fine-tuned MARBERTv2 |
| Sentiment | **CAMeLBERT `da-sentiment` 0.515** · TF-IDF + SVM 0.452 · TF-IDF + logistic 0.450 | CAMeLBERT (pretrained, not fine-tuned) |

## Final result (sealed test, n = 68)
| Task | Validation macro-F1 | Test macro-F1 | Test accuracy | Majority-class share |
|---|---:|---:|---:|---:|
| Intent | 0.315 | **0.245** | 0.412 | 0.485 (SPAM) |
| Topic | 0.402 | **0.161** | 0.368 | 0.412 (OFF_TOPIC_PERSONAL) |
| Sentiment | 0.515 | **0.364** | 0.485 | 0.574 (NEUTRAL) |

Per-class test F1 — Intent: SPAM 0.618, INQUIRY 0.364, SUGGESTION 0.244, COMPLAINT 0.000 (n = 4). Topic: WORD_SUBMISSION 0.500, OFF_TOPIC_PERSONAL 0.500, GENERAL 0.125, ENTRY_CORRECTION / DICTIONARY_SUGGESTION / TECHNICAL_ISSUE 0.000 (never predicted). Sentiment: NEUTRAL 0.651, NEGATIVE 0.250, POSITIVE 0.190.

## Reading the results
- All three classifiers score below the majority-class share on accuracy and drop sharply from validation to test (−0.07 / −0.24 / −0.15 macro-F1). The test set is small, so every figure is noisy; inconsistent human labels were the dominant problem observed.
- A later attempt to improve the intent tokenizer was evaluated and rejected; the frozen intent classifier was kept.
- For this reason classification only **routes** work and decides whether a draft may be generated; a human reviewer decides everything else.

## Reply drafts
Drafts come from Gemini (prompt `rg-v14.4`) behind an eligibility policy, an outbound privacy boundary, deterministic validation gates and one bounded repair. These safeguards are covered by automated tests with a mocked provider. No human rating study of draft quality has been completed.

## Reproducing the method on synthetic data
`python examples/make_synthetic_data.py && python examples/run_synthetic_evaluation.py` runs the repository's training and evaluation code on invented, template-generated comments with deliberately injected label noise. The scores show the pipeline runs; they say nothing about real accuracy.
