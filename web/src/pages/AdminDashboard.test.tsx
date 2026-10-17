import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { AnalyticsSnapshot } from "../api/types";
import { AdminDashboard } from "./AdminDashboard";

vi.mock("../api/client", () => ({ api: { counts: vi.fn(), adminSourceSummary: vi.fn(), adminReviewers: vi.fn(), activeExcelSession: vi.fn(), analytics: vi.fn(), queue: vi.fn() } }));
const session = { user_id: "test-admin", display_name: "اسم من الجلسة", role: "ADMIN" as const, auth_mode: "session" };
function snapshot(): AnalyticsSnapshot {
  return {
    range: "custom", range_start: "2026-09-05T00:00:00", range_end: "2026-09-18T23:59:59",
    kpis: { incoming: 10, reviewed: 5, pending: 8, urgent_pending: 1, overdue_pending: 0, no_reply: 2, avg_review_seconds: null, avg_review_sample: 0 },
    today: { reviewed_all: 0, reviewed_by_me: 0, actions: {} }, outcomes: { SEND_AS_IS: 3, EDIT_AND_SEND: 1, NO_RESPONSE_NEEDED: 1 },
    quality: { sample: 5, send_as_is_rate: .6, edited_rate: .2, no_reply_rate: .2 },
    queue_health: { under_day: 0, one_two: 0, three_seven: 0, over_seven: 0, dated: 0, oldest_pending_age_seconds: null },
    classification: { intent: [], topic: [], sentiment: [] }, trend: [{ bucket: "2026-09-18", incoming: 10, reviewed: 5 }], availability: { trend: "ok", outcomes: "ok" },
  };
}
function mount() {
  const actions = { onImport: vi.fn(), onGoAssignment: vi.fn(), onGoCompleted: vi.fn() };
  render(<AdminDashboard session={session} {...actions} />);
  return actions;
}
beforeEach(() => {
  vi.mocked(api.counts).mockResolvedValue({ all: 22, pending: 8, approved: 5, no_reply: 2, send_failed: 0, by_status: { IN_REVIEW: 2, EXCLUDED: 7 } });
  vi.mocked(api.adminSourceSummary).mockResolvedValue({ sources: [{ source_file: "test.xlsx", total_reviewable: 15, assigned: 12, unassigned: 3, completed: 7 }] });
  vi.mocked(api.adminReviewers).mockResolvedValue({ reviewers: [{ user_id: "r1", display_name: "مراجع الاختبار", email: "private@example.test", assigned_total: 12, completed: 7, pending: 5 }] });
  vi.mocked(api.activeExcelSession).mockResolvedValue({ active: null });
  vi.mocked(api.analytics).mockResolvedValue(snapshot());
  vi.mocked(api.queue).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 5 });
});
afterEach(() => vi.restoreAllMocks());

describe("Admin Home data integrity", () => {
  it("uses browser-local greeting and real session name; requests the existing 14-day analytics range", async () => {
    vi.spyOn(Date.prototype, "getHours").mockReturnValue(9);
    mount();
    expect(screen.getByRole("heading", { name: "صباح الخير، اسم من الجلسة" })).toBeInTheDocument();
    await waitFor(() => expect(api.analytics).toHaveBeenCalled());
    const request = vi.mocked(api.analytics).mock.calls.at(-1)![0];
    expect(request.range).toBe("custom");
    expect((new Date(request.end!).getTime() - new Date(request.start!).getTime()) / 86_400_000).toBe(13);
    expect(api.queue).toHaveBeenCalledWith({ filter: "pending", sort: "newest", search: "", offset: 0, limit: 5 });
  });
  it("distinguishes actual in-review work, completed source counts, and human outcomes", async () => {
    mount();
    const workload = screen.getByRole("region", { name: "حالة عبء المراجعة" });
    await waitFor(() => expect(within(workload).getByText("قيد المراجعة").closest("article")).toHaveTextContent("٢"));
    expect(within(workload).getByText("مكتملة").closest("article")).toHaveTextContent("٧");
    const outcomes = within(screen.getByRole("region", { name: "نتائج المراجعة البشرية" }));
    expect(outcomes.getByText("اعتماد دون تعديل").closest("article")).toHaveTextContent("٣");
    expect(outcomes.getByText(/النسب من/)).toHaveTextContent("٥");
    expect(screen.queryByText("private@example.test")).not.toBeInTheDocument();
    expect(screen.getByRole("table", { name: "تقدم المراجعين" })).toHaveTextContent("مراجع الاختبار");
  });
  it("keeps loading values unknown, without a fake zero or chart", async () => {
    vi.mocked(api.counts).mockImplementation(() => new Promise(() => {}));
    vi.mocked(api.analytics).mockImplementation(() => new Promise(() => {}));
    mount();
    await screen.findByRole("table", { name: "تقدم المراجعين" });
    expect(screen.getByText("إجمالي التعليقات").closest("article")).toHaveTextContent("—");
    expect(screen.getByText("إجمالي التعليقات").closest("article")).not.toHaveTextContent("٠");
    expect(screen.queryByText("عرض القيم اليومية")).not.toBeInTheDocument();
  });
  it("preserves successfully loaded sections when analytics fails, and retries", async () => {
    vi.mocked(api.analytics).mockRejectedValueOnce(new Error("sensitive driver detail"));
    mount();
    await screen.findByRole("alert");
    expect(screen.getByRole("table", { name: "تقدم المراجعين" })).toHaveTextContent("مراجع الاختبار");
    expect(screen.queryByText("sensitive driver detail")).not.toBeInTheDocument();
    expect(screen.queryByText("اعتماد دون تعديل")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "تحديث البيانات" }));
    await screen.findByText("اعتماد دون تعديل");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
  it("shows real zero states without invented rates or trend bars", async () => {
    const data = snapshot(); data.quality = { sample: 0, send_as_is_rate: null, edited_rate: null, no_reply_rate: null }; data.outcomes = { SEND_AS_IS: 0, EDIT_AND_SEND: 0, NO_RESPONSE_NEEDED: 0 }; data.trend = []; data.availability.trend = "empty_window";
    vi.mocked(api.analytics).mockResolvedValue(data);
    vi.mocked(api.adminSourceSummary).mockResolvedValue({ sources: [] });
    vi.mocked(api.adminReviewers).mockResolvedValue({ reviewers: [] });
    mount();
    await screen.findByText(/لا توجد قرارات مراجعة بشرية/);
    expect(screen.getByText(/لا تتوفر حركة مؤرخة/)).toBeInTheDocument();
    expect(screen.getByText("ابدأ باستيراد التعليقات")).toBeInTheDocument();
    expect(screen.queryByText("عرض القيم اليومية")).not.toBeInTheDocument();
    expect(document.querySelectorAll(".admin-rate")).toHaveLength(0);
  });
  it("keeps import, assignment and completed review actions connected to existing callbacks", async () => {
    const actions = mount();
    await userEvent.click(screen.getByRole("button", { name: "إدارة الإسناد" }));
    await userEvent.click(screen.getByRole("button", { name: "عرض المراجعات المكتملة" }));
    await userEvent.click(screen.getByRole("button", { name: "استيراد ملف جديد" }));
    expect(actions.onGoAssignment).toHaveBeenCalledWith(); expect(actions.onGoCompleted).toHaveBeenCalledOnce(); expect(actions.onImport).toHaveBeenCalledOnce();
  });
  it("does not apply a late request result after unmount", async () => {
    let resolve!: (value: AnalyticsSnapshot) => void;
    vi.mocked(api.analytics).mockImplementation(() => new Promise(done => { resolve = done; }));
    const view = render(<AdminDashboard session={session} onImport={vi.fn()} onGoAssignment={vi.fn()} onGoCompleted={vi.fn()} />);
    view.unmount();
    await act(async () => resolve(snapshot()));
    expect(screen.queryByText("نتائج المراجعة البشرية")).not.toBeInTheDocument();
  });
});
