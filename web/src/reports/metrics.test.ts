import { describe, expect, it } from "vitest";
import { ageBucket, queueHealth, reviewedCount, reviewOutcomes } from "./metrics";
import type { QueueItem } from "../api/types";

function item(id: string, iso: string | null): QueueItem {
  return {
    comment_id: id,
    review_status: "DRAFT_READY",
    comment_preview: "x",
    primary_intent: null,
    topic_subtype: null,
    review_priority: 3,
    priority_reason: null,
    claimed_by: null,
    comment_created_at: iso,
    ingested_at: iso,
    suggested_response: null,
    reviewer_action: null,
    reviewer_display_name: null,
    reviewed_at: null,
  };
}

describe("report metrics from existing payloads", () => {
  const now = Date.parse("2026-08-25T12:00:00Z");

  it("buckets queue age from timestamps", () => {
    expect(ageBucket("2026-08-25T10:00:00Z", now)).toBe("under_day");
    expect(ageBucket("2026-08-23T12:00:00Z", now)).toBe("one_two");
    expect(ageBucket("2026-08-20T12:00:00Z", now)).toBe("three_seven");
    expect(ageBucket("2026-08-01T12:00:00Z", now)).toBe("over_seven");
    expect(ageBucket(null, now)).toBeNull();
  });

  it("ignores undated queue rows", () => {
    const { dated, counts } = queueHealth([item("1", "2026-08-25T10:00:00Z"), item("2", null)], now);
    expect(dated).toBe(1);
    expect(counts.under_day).toBe(1);
  });

  it("maps outcomes only from known completed statuses", () => {
    const o = reviewOutcomes({ APPROVED: 4, EDITED_AND_APPROVED: 2, NO_REPLY: 1, SENT: 9, DRAFT_READY: 3 });
    expect(o).toEqual({ sendAsIs: 4, edited: 2, noReply: 1, total: 7 });
  });

  it("treats reviewed as non-pending from counts", () => {
    expect(reviewedCount({ all: 10, pending: 3, approved: 5, no_reply: 2, send_failed: 0, by_status: {} })).toBe(7);
  });
});
