# Human-in-the-loop workflow

1. **Admin imports** a workbook → rows are classified and (where eligible) given a draft.
2. **Admin assigns** comments to a reviewer: by count (`count`), all matching (`all`), or by filter, in a single atomic compare-and-set `UPDATE … RETURNING` (PostgreSQL additionally uses `FOR UPDATE SKIP LOCKED`). Nothing is double-assigned; the selected-id path uses the same compare-and-set so concurrent assignments cannot overwrite each other.
3. **Reviewer works their list only** (isolation is enforced in the backend, not only hidden in the UI): approve a draft as-is, edit and approve, or mark *no reply needed*. Items without a draft cannot be approved as-is. A stale `row_version` is refused with HTTP 409.
4. **Everything is audited** (`review_item_events`: ids, codes and timestamps, never free text).
5. **Admin exports completed reviews only** (all / first N / filtered), read-only, formula-injection safe.
6. **Messaging**: one admin↔reviewer conversation per reviewer, text only, idempotent sends (`client_message_id`), unread counts, polling.
7. **Accounts**: self-service *requests* stay inactive until an admin approves them; approval always creates a reviewer.
Reviewers do **not** correct classifier labels in this version; that is a deliberate scope decision.
