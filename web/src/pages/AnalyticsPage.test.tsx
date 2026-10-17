import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { AnalyticsSnapshot } from "../api/types";
import { AnalyticsPage } from "./AnalyticsPage";

vi.mock("../api/client", () => ({ api: { analytics: vi.fn() } }));
const n = (value: number) => new Intl.NumberFormat("ar-SA").format(value);
const pct = (value: number) => new Intl.NumberFormat("ar-SA", { style: "percent", maximumFractionDigits: 1 }).format(value);
function snapshot(): AnalyticsSnapshot {
  return { range: "week", range_start: "2026-09-12T09:00:00+03:00", range_end: "2026-09-19T09:00:00+03:00",
    kpis: { incoming: 100, reviewed: 85, pending: 17, urgent_pending: 3, overdue_pending: 2, no_reply: 10, avg_review_seconds: 180, avg_review_sample: 70 },
    today: { reviewed_all: 5, reviewed_by_me: 0, actions: {} },
    outcomes: { SEND_AS_IS: 50, EDIT_AND_SEND: 25, NO_RESPONSE_NEEDED: 5 },
    quality: { sample: 80, send_as_is_rate: .625, edited_rate: .3125, no_reply_rate: .0625 },
    queue_health: { under_day: 10, one_two: 3, three_seven: 2, over_seven: 2, dated: 17, oldest_pending_age_seconds: 800000 },
    classification: { intent: [{ key: "SUGGESTION", count: 20 }, { key: "INQUIRY", count: 30 }], topic: [{ key: "GENERAL", count: 40 }, { key: "WORD_OF_YEAR", count: 999 }], sentiment: [{ key: "POSITIVE", count: 5 }] },
    trend: [{ bucket: "2026-09-18", incoming: 40, reviewed: 35 }, { bucket: "2026-09-19", incoming: 60, reviewed: 50 }],
    availability: { trend: "ok", classification: "ok", outcomes: "ok", avg_review_time: "ok" },
  };
}
beforeEach(() => { vi.resetAllMocks(); vi.mocked(api.analytics).mockResolvedValue(snapshot()); });

describe("Admin Analytics", () => {
  it("uses real human outcomes as the denominator and separates visitor distributions", async () => {
    render(<AnalyticsPage />);
    await screen.findByRole("heading", { name: "جودة وعملية المراجعة" });
    expect(screen.getByRole("heading", { name: "صوت الزائر والرؤى" })).toBeInTheDocument();
    const summary = screen.getByRole("region", { name: "ملخص الفترة" });
    expect(summary).toHaveTextContent(n(80)); // Not kpis.reviewed (85) or incoming (100).
    const quality = screen.getByRole("region", { name: "جودة الردود والمراجعة" });
    expect(quality).toHaveTextContent(pct(50 / 80));
    const suggestions = screen.getByRole("heading", { name: "التعليقات المصنفة كاقتراحات" }).closest("article")!;
    expect(suggestions).toHaveTextContent(n(20)); expect(suggestions).toHaveTextContent(pct(.4));
    const topics = screen.getByTestId("class-dist-topic");
    expect(topics.querySelectorAll("li")).toHaveLength(7);
    expect(topics).not.toHaveTextContent("WORD_OF_YEAR"); expect(topics).not.toHaveTextContent(n(999));
    expect(topics).toHaveTextContent(pct(1));
    expect(screen.getByText(/توجد قيم تصنيف خارج التصنيف المعتمد/)).toBeInTheDocument();
    expect(api.analytics).toHaveBeenCalledWith({ range: "week", start: undefined, end: undefined });
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByText("يستحق المتابعة")).not.toBeInTheDocument();
  });

  it("keeps the selected preset visible and updates the whole report on a range change", async () => {
    const user = userEvent.setup(); render(<AnalyticsPage />);
    await screen.findByRole("region", { name: "ملخص الفترة" });
    await user.click(screen.getByRole("button", { name: "آخر ٧ أيام" }));
    expect(screen.getByRole("region", { name: "ملخص الفترة" })).toBeInTheDocument();
    let resolve!: (s: AnalyticsSnapshot) => void;
    vi.mocked(api.analytics).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    await user.click(screen.getByRole("button", { name: "اليوم" }));
    expect(screen.getByText("جاري تحميل التحليلات…")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "ملخص الفترة" })).not.toBeInTheDocument();
    await act(async () => resolve({ ...snapshot(), range: "today" }));
    expect(screen.getByRole("button", { name: "اليوم" })).toHaveAttribute("aria-pressed", "true");
    expect(api.analytics).toHaveBeenLastCalledWith({ range: "today", start: undefined, end: undefined });
    await user.click(screen.getByRole("button", { name: "إعادة ضبط الفترة" }));
    await waitFor(() => expect(api.analytics).toHaveBeenLastCalledWith({ range: "week", start: undefined, end: undefined }));
  });

  it("passes valid custom dates and suppresses requests and stale results for invalid or incomplete ranges", async () => {
    const user = userEvent.setup(); render(<AnalyticsPage />);
    await screen.findByRole("region", { name: "ملخص الفترة" });
    await user.click(screen.getByRole("button", { name: "نطاق مخصص" }));
    fireEvent.change(screen.getByLabelText("من تاريخ"), { target: { value: "2026-08-20" } });
    fireEvent.change(screen.getByLabelText("إلى تاريخ"), { target: { value: "2026-08-01" } });
    await screen.findByRole("alert");
    expect(screen.getByLabelText("من تاريخ")).toHaveValue("2026-08-20");
    expect(screen.queryByRole("region", { name: "ملخص الفترة" })).not.toBeInTheDocument();
    const calls = vi.mocked(api.analytics).mock.calls.length;
    fireEvent.change(screen.getByLabelText("إلى تاريخ"), { target: { value: "" } });
    expect(api.analytics).toHaveBeenCalledTimes(calls);
    expect(screen.getByRole("alert")).toHaveTextContent("اختر تاريخ البداية والنهاية");
    fireEvent.change(screen.getByLabelText("إلى تاريخ"), { target: { value: "2026-08-25" } });
    await screen.findByRole("region", { name: "ملخص الفترة" });
    expect(api.analytics).toHaveBeenLastCalledWith({ range: "custom", start: "2026-08-20", end: "2026-08-25" });
  });

  it("ignores old responses after a new range has loaded", async () => {
    let resolve!: (s: AnalyticsSnapshot) => void;
    vi.mocked(api.analytics).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    const user = userEvent.setup(); render(<AnalyticsPage />);
    await user.click(screen.getByRole("button", { name: "اليوم" }));
    await screen.findByRole("region", { name: "ملخص الفترة" });
    await act(async () => resolve({ ...snapshot(), outcomes: { SEND_AS_IS: 999 } }));
    expect(screen.getByRole("region", { name: "ملخص الفترة" })).not.toHaveTextContent(n(999));
  });

  it("shows sanitized errors and retries the selected date range", async () => {
    vi.mocked(api.analytics).mockRejectedValueOnce(new Error("database secret"));
    const user = userEvent.setup(); render(<AnalyticsPage />);
    await screen.findByRole("alert");
    expect(screen.queryByText(/database secret/)).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "ملخص الفترة" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "إعادة المحاولة" }));
    await screen.findByRole("region", { name: "ملخص الفترة" });
    expect(api.analytics).toHaveBeenCalledTimes(2);
  });

  it("does not turn unavailable averages or zero-denominator rates into invented percentages", async () => {
    const empty = snapshot(); empty.outcomes = {}; empty.quality = { sample: 0, send_as_is_rate: null, edited_rate: null, no_reply_rate: null };
    empty.kpis = { ...empty.kpis, incoming: 0, reviewed: 0, avg_review_seconds: null, avg_review_sample: 0 };
    empty.classification = { intent: [], topic: [], sentiment: [] }; empty.trend = [];
    vi.mocked(api.analytics).mockResolvedValue(empty);
    render(<AnalyticsPage />);
    await screen.findByText("لا توجد حركة مؤرخة لعرضها في هذه الفترة.");
    expect(screen.getByRole("region", { name: "ملخص الفترة" })).toHaveTextContent("غير متاح");
    expect(screen.getByRole("region", { name: "جودة الردود والمراجعة" })).toHaveTextContent("النسب غير متاحة");
    expect(screen.getByTestId("class-dist-intent")).not.toHaveTextContent("NaN");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("keeps current queue health explicitly outside the selected reporting period", async () => {
    const user = userEvent.setup(); render(<AnalyticsPage />);
    await screen.findByText(/لقطة حالية · خارج مرشح الفترة/);
    await user.click(screen.getByText("حالة قائمة الانتظار", { exact: false, selector: "summary" }));
    expect(screen.getByText(/بصرف النظر عن فترة التقرير أعلاه/)).toBeVisible();
    expect(screen.getByRole("heading", { name: "أولوية عاجلة" }).closest("article")).toHaveTextContent(n(3));
  });

  it("provides expandable exact trend values and handles a single bucket", async () => {
    const s = snapshot(); s.trend = [{ bucket: "2026-09-19 09", incoming: 1, reviewed: 0 }];
    vi.mocked(api.analytics).mockResolvedValue(s);
    const user = userEvent.setup(); render(<AnalyticsPage />);
    await screen.findByRole("img", { name: /اتجاه الوارد والمراجعة/ });
    // Native summary keyboard activation is verified in Chrome, not simulated by jsdom.
    await user.click(screen.getByText("عرض القيم الزمنية"));
    const table = screen.getByRole("table", { name: "القيم المسجلة لكل فترة زمنية" });
    expect(within(table).getByRole("cell", { name: n(1) })).toBeVisible();
    expect(within(table).getByRole("cell", { name: n(0) })).toBeVisible();
  });
});
