# Architecture

## Layers
| Layer | Location | Responsibility |
|---|---|---|
| Ingest & cleaning | `src/rcs/ingest`, `src/rcs/review/import_identity.py`, `excel_upload.py` | Excel validation, normalisation, stable `comment_id = sha256(source_file\|row_index)`, content-fingerprint batch identity (a different file under the same name becomes a separate batch; an identical re-upload is idempotent) |
| Prefilter | `src/rcs/prefilter` | Deterministic spam/junk rules; never deletes rows, routes them to `EXCLUDE` |
| Classification | `src/rcs/classification`, `src/rcs/training`, `src/rcs/routing` | Three frozen classifiers (intent TF-IDF + logistic regression, topic MARBERTv2, sentiment CAMeLBERT). `routing.engine.determine_routing` turns labels into a routing decision. Classification and routing are separate, independently tested steps |
| Response generation | `src/rcs/response_generation`, `src/rcs/response_intelligence` | Eligibility policy → outbound privacy boundary (PII redaction, prompt guard) → LLM draft → nine validation gates → one bounded repair → fail closed to review-only. Prompt version `rg-v14.4` |
| Review workflow | `src/rcs/review` | `ReviewWorkflowService`: claim, approve as-is, edit-and-approve, no-response, optimistic locking (`row_version`, HTTP 409), audit events |
| Persistence | `src/rcs/review/repositories` | One contract (`contracts.py`), two implementations: SQL (SQLite/PostgreSQL) and MongoDB. Selected by `PERSISTENCE_BACKEND` |
| HTTP API | `src/rcs/review/http_api.py` | stdlib `http.server`; server-verified sessions; role checks on every route |
| Messaging | `src/rcs/review/messaging.py` | One admin↔reviewer conversation per reviewer; text only; idempotent sends |
| Frontend | `web/` | React 18 + TypeScript SPA, RTL Arabic |

## Data flow
```
upload ─► validate ─► batch identity ─► rows (review_items)
rows ─► prefilter ─► classifiers ─► routing ─► eligibility ─► [PII redaction] ─► LLM ─► gates ─► draft
admin ─► assign (count / all / filter, atomic)
reviewer ─► approve | edit | no-response  ─► audit event
admin ─► export completed-only workbook (read-only)
```

## Roles
- **ADMIN**: import, assignment, completed-review export, analytics, messaging with any reviewer.
- **REVIEWER**: sees and decides only comments assigned to them; messaging with the admin only. Reviewers do **not** correct classifications.

## Design decisions worth knowing
- Everything that can send text to an external service goes through one privacy boundary; production fails closed if the pinned local model copy is missing.
- Assignment is a single atomic conditional `UPDATE … RETURNING` (PostgreSQL additionally uses `FOR UPDATE SKIP LOCKED`; that path is not verified here).
- The audit log stores identifiers, codes and timestamps — never comment or reply text.
