import { describe, expect, it } from "vitest";
import type { QueueItem } from "../api/types";
import { orderStartingList } from "./orderStartingList";

function item(id: string, priority: number | null, ingested: string): QueueItem {
  return {
    comment_id: id,
    review_status: "DRAFT_READY",
    comment_preview: id,
    primary_intent: null,
    topic_subtype: null,
    review_priority: priority,
    priority_reason: null,
    claimed_by: null,
    comment_created_at: ingested,
    ingested_at: ingested,
    suggested_response: null,
    reviewer_action: null,
    reviewer_display_name: null,
    reviewed_at: null,
  };
}

describe("orderStartingList", () => {
  it("puts highest priority first, then oldest within the same priority", () => {
    const ordered = orderStartingList([
      item("late-routine", 3, "2026-08-25T18:00:00Z"),
      item("old-urgent", 1, "2026-08-20T10:00:00Z"),
      item("new-urgent", 1, "2026-08-25T12:00:00Z"),
      item("old-medium", 2, "2026-08-21T10:00:00Z"),
    ]);
    expect(ordered.map((row) => row.comment_id)).toEqual(["old-urgent", "new-urgent", "old-medium", "late-routine"]);
  });
});
