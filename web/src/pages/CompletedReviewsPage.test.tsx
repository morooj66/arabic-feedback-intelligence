import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { QueueItem, QueuePage, ReviewDetail } from "../api/types";
import { CompletedReviewsPage } from "./CompletedReviewsPage";

vi.mock("../api/client", () => ({ api: { adminReviewers: vi.fn(), queue: vi.fn(), item: vi.fn(), downloadExcelExport: vi.fn(), downloadCompletedExport: vi.fn() } }));
const row = (n: number): QueueItem => ({ comment_id: `c${n}`, comment_preview: `تعليق آمن ${n} [PHONE]`, review_status: "APPROVED", primary_intent: "SUGGESTION", topic_subtype: "GENERAL", review_priority: 2, priority_reason: null, claimed_by: null, comment_created_at: null, ingested_at: null, suggested_response: "مسودة لا تظهر في الجدول", reviewer_action: "EDIT_AND_SEND", reviewer_display_name: "مراجع اختبار", reviewed_at: "2026-09-19T09:00:00Z" });
const detail: ReviewDetail = { ...row(1), source_reference: "fixture.xlsx", original_comment: "التعليق الكامل [PHONE]", original_title: "مدخل اختباري", entry_id: null, platform: null, original_type: null, sentiment: null, confidence: null, needs_review: null, generation_routing_decision: null, generation_status: null, generation_reason_code: null, reviewer_id: "r1", edited_response: "صيغة المراجع", final_response: "الرد النهائي المحفوظ", no_response_reason: null, ready_to_send: false, outbound_status: null, outbound_error_code: null, row_version: 1, allowed_actions: [] };
let records: QueueItem[];
beforeAll(() => {
  // jsdom has no modal dialog implementation; real focus containment is browser-verified.
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute("open", ""); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute("open"); };
});
beforeEach(() => {
  vi.resetAllMocks(); records = Array.from({ length: 23 }, (_, n) => row(n + 1));
  vi.mocked(api.adminReviewers).mockResolvedValue({ reviewers: [{ user_id: "r1", display_name: "مراجع اختبار", email: "hidden@example.test", pending: 0, completed: 23, assigned_total: 23 }] });
  vi.mocked(api.queue).mockImplementation(async p => {
    const filtered = records.filter(r => (!p.search || r.comment_preview?.includes(p.search)) && (!p.primaryIntents?.length || p.primaryIntents.includes(r.primary_intent!)) && (!p.topicSubtypes?.length || p.topicSubtypes.includes(r.topic_subtype!)) && (!p.reviewerActions?.length || p.reviewerActions.includes(r.reviewer_action!)));
    return { items: filtered.slice(p.offset, p.offset + p.limit), total: filtered.length, offset: p.offset, limit: p.limit };
  });
  vi.mocked(api.item).mockResolvedValue(detail);
  vi.mocked(api.downloadExcelExport).mockResolvedValue();
  vi.mocked(api.downloadCompletedExport).mockImplementation(async p => ({ rows: p.all ? 23 : Math.min(p.count ?? 0, 23), matched: 23, shortfall: 0 }));
});

describe("Completed Reviews workspace", () => {
  it("shows real totals, concise safe previews and bounded pagination", async () => {
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await screen.findByText("23 مراجعة مكتملة");
    expect(screen.getByText("تعليق آمن 1 [PHONE]")).toBeInTheDocument();
    expect(screen.queryByText(row(1).suggested_response!)).not.toBeInTheDocument();
    expect(screen.queryByText("يستحق المتابعة")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "التالي" }));
    await screen.findByRole("button", { name: "عرض مراجعة c21" });
    expect(api.queue).toHaveBeenLastCalledWith(expect.objectContaining({ filter: "completed", offset: 20, limit: 20 }));
    expect(screen.getByRole("button", { name: "التالي" })).toBeDisabled();
  });

  it("combines supported filters and resets pagination and active filters", async () => {
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await screen.findByText("23 مراجعة مكتملة");
    await user.click(screen.getByRole("button", { name: "التالي" }));
    await screen.findByRole("button", { name: "عرض مراجعة c21" });
    await user.selectOptions(screen.getByLabelText("اختر المراجع"), "r1");
    await user.type(screen.getByLabelText("بحث"), "آمن");
    await user.click(screen.getByRole("button", { name: "التصنيف" }));
    const panel = screen.getByRole("dialog", { name: "تصفية حسب التصنيف" });
    await user.click(within(panel).getByLabelText("اقتراح"));
    await user.click(within(panel).getByLabelText("عام"));
    await user.click(within(screen.getByRole("group", { name: "القرار النهائي" })).getByRole("button", { name: "تعديل واعتماد" }));
    await waitFor(() => expect(api.queue).toHaveBeenLastCalledWith(expect.objectContaining({ search: "آمن", primaryIntents: ["SUGGESTION"], topicSubtypes: ["GENERAL"], reviewerActions: ["EDIT_AND_SEND"], reviewerUserId: "r1", offset: 0 })));
    expect(screen.getByLabelText("الفلاتر النشطة")).toHaveTextContent("النية: اقتراح");
    await user.click(screen.getByRole("button", { name: "مسح الفلاتر" }));
    await screen.findByText("23 مراجعة مكتملة");
    expect(screen.getByLabelText("بحث")).toHaveValue("");
    expect(screen.queryByLabelText("الفلاتر النشطة")).not.toBeInTheDocument();
  });

  it("distinguishes empty data from no matches", async () => {
    records = []; const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await screen.findByText("لا توجد مراجعات مكتملة بعد");
    await user.type(screen.getByLabelText("بحث"), "بحث");
    await screen.findByText("لا توجد مراجعات تطابق البحث والتصفية");
    expect(screen.getByRole("button", { name: /تصدير المراجعات المكتملة/ })).toBeDisabled();
  });

  it("shows loading and sanitized failure, retaining filters on retry", async () => {
    vi.mocked(api.queue).mockRejectedValueOnce(new Error("database secret"));
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    expect(screen.getByText("جاري تحميل المراجعات المكتملة…")).toBeInTheDocument();
    await screen.findByRole("alert");
    expect(screen.queryByText(/database secret/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "إعادة تحميل المراجعات" }));
    await screen.findByText("23 مراجعة مكتملة");
  });

  it("ignores stale list responses", async () => {
    let resolve!: (p: QueuePage) => void;
    vi.mocked(api.queue).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await user.type(screen.getByLabelText("بحث"), "لا يوجد");
    await screen.findByText("لا توجد مراجعات تطابق البحث والتصفية");
    await act(async () => resolve({ items: [row(1)], total: 1, offset: 0, limit: 20 }));
    expect(screen.queryByText("تعليق آمن 1 [PHONE]")).not.toBeInTheDocument();
  });

  it("opens details with the existing read API and restores focus on close", async () => {
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    const trigger = await screen.findByRole("button", { name: "عرض مراجعة c1" });
    trigger.focus(); await user.keyboard("{Enter}");
    const drawer = await screen.findByRole("dialog", { name: "تفاصيل المراجعة" });
    expect(await within(drawer).findByText("التعليق الكامل [PHONE]")).toBeInTheDocument();
    expect(within(drawer).getByText("الرد النهائي المحفوظ")).toBeInTheDocument();
    expect(within(drawer).getByText(detail.suggested_response!)).toBeInTheDocument();
    expect(api.item).toHaveBeenCalledWith("c1");
    expect(within(drawer).queryByRole("textbox")).not.toBeInTheDocument();
    fireEvent(drawer, new Event("cancel", { bubbles: false, cancelable: true }));
    expect(screen.queryByRole("dialog", { name: "تفاصيل المراجعة" })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus(); expect(document.body.style.overflow).toBe("");
  });

  it("never substitutes a draft for an unavailable final response", async () => {
    vi.mocked(api.item).mockResolvedValue({ ...detail, final_response: null });
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await user.click(await screen.findByRole("button", { name: "عرض مراجعة c1" }));
    await screen.findByText("لا يوجد رد نهائي مسجل متاح.");
    expect(screen.getAllByText(detail.suggested_response!)).toHaveLength(1);
  });

  it("shows the no-response decision and reason without implying a reply", async () => {
    vi.mocked(api.item).mockResolvedValue({ ...detail, final_response: null, reviewer_action: "NO_RESPONSE_NEEDED", no_response_reason: "تكرار" });
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await user.click(await screen.findByRole("button", { name: "عرض مراجعة c1" }));
    await screen.findByText("اكتملت المراجعة بقرار عدم الحاجة إلى رد.");
    expect(screen.getByText("سبب عدم الرد: تكرار")).toBeInTheDocument();
  });

  it("supports detail retry and ignores a response after drawer closure", async () => {
    vi.mocked(api.item).mockRejectedValueOnce(new Error("secret"));
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await user.click(await screen.findByRole("button", { name: "عرض مراجعة c1" }));
    await screen.findByText("تعذر تحميل تفاصيل التعليق");
    let resolve!: (d: ReviewDetail) => void;
    vi.mocked(api.item).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    await user.click(screen.getByRole("button", { name: "إعادة تحميل التفاصيل" }));
    expect(screen.getByText("جاري تحميل تفاصيل المراجعة…")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "إغلاق تفاصيل المراجعة" }));
    await act(async () => resolve(detail));
    expect(screen.queryByRole("dialog", { name: "تفاصيل المراجعة" })).not.toBeInTheDocument();
  });

  it("exports ALL matching completed reviews on the server (not just the displayed page) and reports the actual number", async () => {
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await screen.findByText(/23 مراجعة مكتملة/);
    await user.click(screen.getByRole("button", { name: "تصدير المراجعات المكتملة (23)" }));              // 23 matches, only 20 on the first page
    expect(await screen.findByText("تم تصدير 23 من 23 مراجعة مكتملة مطابقة.")).toBeInTheDocument();
    expect(api.downloadCompletedExport).toHaveBeenCalledWith(expect.objectContaining({ all: true }));
    expect(api.downloadExcelExport).not.toHaveBeenCalled();                                                // not the id-list / full-batch endpoint
  });

  it("passes the active filters to the server and supports an exact count", async () => {
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await screen.findByText(/23 مراجعة مكتملة/);
    await user.type(screen.getByLabelText("بحث"), "1");
    await waitFor(() => expect(screen.getByRole("button", { name: /تصدير المراجعات المكتملة/ })).toBeEnabled());
    await user.type(screen.getByLabelText("عدد المراجعات المطلوب تصديرها"), "3");
    await user.click(await screen.findByRole("button", { name: "تصدير المراجعات المكتملة (3)" }));
    await screen.findByText(/تم تصدير 3 من/);
    expect(api.downloadCompletedExport).toHaveBeenCalledWith(expect.objectContaining({ count: 3, search: "1" }));
    expect(vi.mocked(api.downloadCompletedExport).mock.calls[0][0]).not.toHaveProperty("all");
  });

  it("rejects an invalid or too-large count before any request", async () => {
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await screen.findByText(/23 مراجعة مكتملة/);
    const input = screen.getByLabelText("عدد المراجعات المطلوب تصديرها");
    await user.type(input, "0");
    await user.click(screen.getByRole("button", { name: /تصدير المراجعات المكتملة/ }));
    expect(await screen.findByText("أدخل عددًا صحيحًا موجبًا.")).toBeInTheDocument();
    await user.clear(input); await user.type(input, "24");
    await user.click(screen.getByRole("button", { name: /تصدير المراجعات المكتملة/ }));
    expect(await screen.findByText("العدد المطلوب أكبر من المتاح (23).")).toBeInTheDocument();
    expect(api.downloadCompletedExport).not.toHaveBeenCalled();
  });

  it("reports failures without raw errors, and shows a shortfall when the server exported fewer", async () => {
    vi.mocked(api.downloadCompletedExport).mockRejectedValueOnce(new Error("secret"));
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await user.click(await screen.findByRole("button", { name: "تصدير المراجعات المكتملة (23)" }));
    await screen.findByRole("alert");
    expect(screen.queryByText(/secret/)).not.toBeInTheDocument();
    vi.mocked(api.downloadCompletedExport).mockResolvedValueOnce({ rows: 20, matched: 20, shortfall: 3 });
    await user.click(screen.getByRole("button", { name: "تصدير المراجعات المكتملة (23)" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(await screen.findByText(/النقص 3/)).toBeInTheDocument();
  });

  it("makes reviewer lookup failure explicit and retryable", async () => {
    vi.mocked(api.adminReviewers).mockRejectedValueOnce(new Error("secret"));
    const user = userEvent.setup(); render(<CompletedReviewsPage />);
    await screen.findByRole("alert");
    expect(screen.getByLabelText("اختر المراجع")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "إعادة المحاولة" }));
    await waitFor(() => expect(screen.getByLabelText("اختر المراجع")).toBeEnabled());
  });
});
