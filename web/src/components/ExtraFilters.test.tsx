import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { EMPTY_EXTRA, ExtraFilters, extraFilterCount, extraFilterLabels, extraFilterParams, type ExtraFilterValue } from "./ExtraFilters";

describe("ExtraFilters", () => {
  it("emits sentiment, priority and comment-date changes without touching other fields", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ExtraFilters value={EMPTY_EXTRA} onChange={onChange} />);
    await user.click(screen.getByRole("button", { name: "المشاعر" }));
    await user.click(screen.getByLabelText("سلبي"));
    expect(onChange).toHaveBeenLastCalledWith({ ...EMPTY_EXTRA, sentiments: ["NEGATIVE"] });
    await user.click(screen.getByRole("button", { name: "الأولوية" }));
    await user.click(screen.getByLabelText("عاجل"));
    expect(onChange).toHaveBeenLastCalledWith({ ...EMPTY_EXTRA, priorities: ["1"] });
    await user.type(screen.getByLabelText("من تاريخ التعليق"), "2026-01-05");
    expect(onChange).toHaveBeenLastCalledWith({ ...EMPTY_EXTRA, createdFrom: "2026-01-05" });
  });

  it("counts, labels and serialises the active filters", () => {
    const v: ExtraFilterValue = { sentiments: ["NEGATIVE"], priorities: ["1", "3"], createdFrom: "2026-01-01", createdTo: "" };
    expect(extraFilterCount(v)).toBe(4);
    expect(extraFilterLabels(v)).toEqual(["المشاعر: سلبي", "الأولوية: عاجل", "الأولوية: روتيني", "من تاريخ التعليق: 2026-01-01"]);
    expect(extraFilterParams(v)).toEqual({ sentiments: ["NEGATIVE"], priorities: ["1", "3"], createdFrom: "2026-01-01", createdTo: undefined });
    expect(extraFilterCount(EMPTY_EXTRA)).toBe(0);
  });
});

describe("api query strings", () => {
  afterEach(() => vi.restoreAllMocks());

  it("sends the new filters, comment-date sort and assignee to the queue and assignment endpoints", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      urls.push(String(input));
      return new Response(JSON.stringify({ items: [], total: 0, offset: 0, limit: 5 }), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    await api.queue({ filter: "pending", sort: "comment_oldest", search: "", offset: 0, limit: 5, sentiments: ["NEGATIVE", "POSITIVE"], priorities: ["1"], createdFrom: "2026-01-01", createdTo: "2026-03-01" });
    await api.adminAssignableItems({ offset: 0, limit: 5, sort: "comment_newest", sentiments: ["NEUTRAL"], priorities: ["2"], createdFrom: "2026-02-01", reviewerUserId: "u1", statuses: ["DRAFT_READY"] });
    const q = new URL(urls[0], "http://x").searchParams;
    expect([q.get("sort"), q.get("sentiment"), q.get("priority"), q.get("created_from"), q.get("created_to")]).toEqual(["comment_oldest", "NEGATIVE,POSITIVE", "1", "2026-01-01", "2026-03-01"]);
    const a = new URL(urls[1], "http://x").searchParams;
    expect([a.get("sort"), a.get("sentiment"), a.get("priority"), a.get("created_from"), a.get("reviewer_user_id"), a.get("review_status")]).toEqual(["comment_newest", "NEUTRAL", "2", "2026-02-01", "u1", "DRAFT_READY"]);
    expect(a.has("created_to")).toBe(false);
  });
});
