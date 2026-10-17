import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { AssignableItem, AssignableItemsPage } from "../api/types";
import { AssignmentPage } from "./AssignmentPage";

vi.mock("../api/client", () => ({ api: { adminSourceSummary: vi.fn(), adminReviewers: vi.fn(), adminAssignableItems: vi.fn(), adminAssign: vi.fn(), adminAssignByFilter: vi.fn() } }));
let records: AssignableItem[];
const reviewer = { user_id: "r1", display_name: "مراجع اختبار", email: "reviewer@example.test", assigned_total: 8, pending: 5, completed: 3 };
const row = (index: number): AssignableItem => ({ comment_id: `item-${index}`, comment_preview: `تعليق آمن ${index} [PHONE]`, source_file: "batch.xlsx", review_status: "DRAFT_READY", routing_decision: "GENERATE_DRAFT", reviewer_user_id: null, primary_intent: "INQUIRY", topic_subtype: "GENERAL", review_priority: 2, ingested_at: "2026-09-18T09:00:00Z" });
beforeEach(() => {
  vi.resetAllMocks();
  records = Array.from({ length: 23 }, (_, i) => row(i + 1));
  vi.mocked(api.adminSourceSummary).mockResolvedValue({ sources: [{ source_file: "batch.xlsx", total_reviewable: 23, unassigned: 23, assigned: 0, completed: 0 }] });
  vi.mocked(api.adminReviewers).mockResolvedValue({ reviewers: [reviewer] });
  vi.mocked(api.adminAssignableItems).mockImplementation(async params => {
    const filtered = records.filter(r => (!params.unassignedOnly || !r.reviewer_user_id) && (!params.search || r.comment_preview?.includes(params.search)) && (!params.sourceFile || r.source_file === params.sourceFile) && (!params.primaryIntents?.length || params.primaryIntents.includes(r.primary_intent!)) && (!params.topicSubtypes?.length || params.topicSubtypes.includes(r.topic_subtype!)));
    return { items: filtered.slice(params.offset, params.offset + params.limit), total: filtered.length, offset: params.offset, limit: params.limit };
  });
  vi.mocked(api.adminAssign).mockImplementation(async ids => {
    records = records.map(r => ids.includes(r.comment_id) ? { ...r, reviewer_user_id: reviewer.user_id } : r);
    return { assigned: ids, skipped_already_decided: [], skipped_not_reviewable: [], not_found: [] };
  });
  vi.mocked(api.adminAssignByFilter).mockImplementation(async params => {                       // behaves like the server: first N unassigned matches, never moves existing assignments
    const open = records.filter(r => !r.reviewer_user_id);
    const take = params.all ? open : open.slice(0, params.count);
    const ids = new Set(take.map(r => r.comment_id));
    records = records.map(r => ids.has(r.comment_id) ? { ...r, reviewer_user_id: reviewer.user_id } : r);
    return { requested: params.all ? null : params.count ?? null, assigned: take.length, shortfall: params.all ? 0 : Math.max(0, (params.count ?? 0) - take.length), matched_before: open.length, remaining_eligible: open.length - take.length };
  });
});

describe("Admin Assignment workspace", () => {
  it("keeps the unassigned default and source context, and clears selection when filters change", async () => {
    const user = userEvent.setup(); render(<AssignmentPage initialSourceFile="batch.xlsx" />);
    await screen.findByLabelText("تحديد item-1");
    expect(api.adminAssignableItems).toHaveBeenCalledWith(expect.objectContaining({ unassignedOnly: true, sourceFile: "batch.xlsx", limit: 20 }));
    expect(screen.queryByLabelText("اختر المراجع")).not.toBeInTheDocument();
    expect(screen.queryByText("استيراد ملف جديد")).not.toBeInTheDocument();
    await user.click(screen.getByLabelText("تحديد item-1"));
    expect(screen.getByRole("button", { name: "إسناد المحدد" })).toBeDisabled();
    await user.type(screen.getByLabelText("بحث"), "غير موجود");
    await screen.findByText("لا توجد تعليقات تطابق البحث والتصفية");
    expect(screen.queryByLabelText("اختر المراجع")).not.toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "مسح الفلاتر" })[0]);
    await screen.findByLabelText("تحديد item-1");
    expect(screen.getByLabelText("غير المسند فقط")).toBeChecked();
    expect(screen.getByLabelText("اختر الملف")).toHaveValue("");
  });

  it("keeps cross-page selection explicit and fetches all filtered results in bounded pages", async () => {
    const user = userEvent.setup(); render(<AssignmentPage />);
    await user.click(await screen.findByLabelText("تحديد item-1"));
    await user.click(screen.getByRole("button", { name: "التالي" }));
    await user.click(await screen.findByLabelText("تحديد item-21"));
    expect(screen.getByText(/يشمل 1 تعليقات محددة في صفحات أخرى/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "تحديد جميع نتائج الفلترة (23)" }));
    await screen.findByRole("heading", { name: "إسناد المحدد (23)" });
    expect(vi.mocked(api.adminAssignableItems).mock.calls.every(([params]) => params.limit === 20)).toBe(true);
    await user.selectOptions(screen.getByLabelText("اختر المراجع"), "r1");
    expect(screen.getByText(/عبء المراجع الحالي: 8 مسند · 5 قيد الإنجاز · 3 مكتمل/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "إسناد المحدد" }));
    await screen.findByText("لا توجد تعليقات غير مسندة حاليًا");
    expect(api.adminAssign).toHaveBeenCalledWith(records.map(r => r.comment_id), "r1");
    expect(screen.queryByLabelText("اختر المراجع")).not.toBeInTheDocument();
    expect(api.adminSourceSummary).toHaveBeenCalledTimes(2);
  });

  it("toggles current-page selection without silently selecting other pages", async () => {
    const user = userEvent.setup(); render(<AssignmentPage />);
    await user.click(await screen.findByLabelText("تحديد item-1"));
    expect(screen.getByLabelText("تحديد جميع عناصر الصفحة")).toBePartiallyChecked();
    await user.click(screen.getByLabelText("تحديد جميع عناصر الصفحة"));
    expect(screen.getByRole("heading", { name: "إسناد المحدد (20)" })).toBeInTheDocument();
    await user.click(screen.getByLabelText("تحديد جميع عناصر الصفحة"));
    expect(screen.queryByLabelText("اختر المراجع")).not.toBeInTheDocument();
  });

  it("preserves selection and sanitizes errors when assignment fails, then supports retry", async () => {
    vi.mocked(api.adminAssign).mockRejectedValueOnce(new Error("MongoDB secret raw backend exception"));
    const user = userEvent.setup(); render(<AssignmentPage />);
    await user.click(await screen.findByLabelText("تحديد item-1"));
    await user.selectOptions(screen.getByLabelText("اختر المراجع"), "r1");
    await user.click(screen.getByRole("button", { name: "إسناد المحدد" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("التحديد محفوظ");
    expect(screen.queryByText(/MongoDB/)).not.toBeInTheDocument();
    expect(screen.queryByText(/عنصر بنجاح/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("تحديد item-1")).toBeChecked();
    await user.click(screen.getByRole("button", { name: "إسناد المحدد" }));
    await waitFor(() => expect(screen.queryByLabelText("تحديد item-1")).not.toBeInTheDocument());
    expect(screen.getByText(/عنصر بنجاح/)).toBeInTheDocument();
  });

  it("shows truthful empty and no-reviewer states with no fake assignment action", async () => {
    records = [];
    vi.mocked(api.adminReviewers).mockResolvedValue({ reviewers: [] });
    render(<AssignmentPage />);
    await screen.findByText("لا توجد تعليقات غير مسندة حاليًا");
    expect(screen.getByText("لا يوجد مراجعون نشطون لإسناد العناصر إليهم.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "إسناد المحدد" })).not.toBeInTheDocument();
  });

  it("preserves classification filters and renders only the server's safe preview", async () => {
    const user = userEvent.setup(); render(<AssignmentPage />);
    await screen.findByText("تعليق آمن 1 [PHONE]");
    await user.click(screen.getByRole("button", { name: "التصنيف" }));
    const panel = screen.getByRole("dialog", { name: "تصفية حسب التصنيف" });
    await user.click(within(panel).getByLabelText("اقتراح"));
    await screen.findByText("لا توجد تعليقات تطابق البحث والتصفية");
    expect(api.adminAssignableItems).toHaveBeenLastCalledWith(expect.objectContaining({ primaryIntents: ["SUGGESTION"], offset: 0 }));
    await user.click(within(panel).getByLabelText("اقتراح"));
    await screen.findByLabelText("تحديد item-1");
    await user.click(within(panel).getByLabelText("استفسار"));
    await user.click(within(panel).getByLabelText("عام"));
    await screen.findByLabelText("تحديد item-1");
    expect(api.adminAssignableItems).toHaveBeenLastCalledWith(expect.objectContaining({ primaryIntents: ["INQUIRY"], topicSubtypes: ["GENERAL"], offset: 0 }));
    await user.click(within(panel).getByLabelText("عام"));
    await user.click(within(panel).getByLabelText("مشكلة تقنية"));
    await screen.findByText("لا توجد تعليقات تطابق البحث والتصفية");
    expect(api.adminAssignableItems).toHaveBeenLastCalledWith(expect.objectContaining({ primaryIntents: ["INQUIRY"], topicSubtypes: ["TECHNICAL_ISSUE"], offset: 0 }));
  });

  it("ignores a stale list response after a newer filter request completes", async () => {
    let resolve!: (page: AssignableItemsPage) => void;
    vi.mocked(api.adminAssignableItems).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    const user = userEvent.setup(); render(<AssignmentPage />);
    await user.type(screen.getByLabelText("بحث"), "غير موجود");
    await screen.findByText("لا توجد تعليقات تطابق البحث والتصفية");
    await act(async () => resolve({ items: [row(1)], total: 1, offset: 0, limit: 20 }));
    expect(screen.queryByText("تعليق آمن 1 [PHONE]")).not.toBeInTheDocument();
  });

  it("assigns a typed count on the server across all pages and reports the actual number", async () => {
    const user = userEvent.setup(); render(<AssignmentPage initialSourceFile="batch.xlsx" />);
    await screen.findByLabelText("تحديد item-1");
    await user.selectOptions(screen.getByLabelText("المراجع للإسناد بالعدد"), "r1");
    await user.type(screen.getByLabelText("عدد التعليقات المطلوب إسنادها"), "22");              // more than one page (20) of the 23 matches
    await user.click(screen.getByRole("button", { name: "إسناد العدد" }));
    await screen.findByText(/تم إسناد/);
    expect(api.adminAssignByFilter).toHaveBeenCalledWith(expect.objectContaining({ reviewerUserId: "r1", count: 22, sourceFile: "batch.xlsx", sort: "oldest" }));
    expect(api.adminAssignByFilter).not.toHaveBeenCalledWith(expect.objectContaining({ all: true }));
    expect(api.adminAssign).not.toHaveBeenCalled();                                              // not the id-list endpoint
    expect(screen.getByText(/تم إسناد/).textContent).toMatch(/22\s*من\s*22/);
    expect(screen.getByText(/المتبقي غير مسند ضمن التصفية: 1/)).toBeInTheDocument();
    expect(records.filter(r => r.reviewer_user_id).length).toBe(22);
  });

  it("assigns every matching comment in one request", async () => {
    const user = userEvent.setup(); render(<AssignmentPage />);
    await screen.findByLabelText("تحديد item-1");
    await user.selectOptions(screen.getByLabelText("المراجع للإسناد بالعدد"), "r1");
    await user.click(screen.getByRole("button", { name: "إسناد كل المطابقة (23)" }));
    await screen.findByText(/من جميع التعليقات المطابقة/);
    expect(api.adminAssignByFilter).toHaveBeenCalledWith(expect.objectContaining({ reviewerUserId: "r1", all: true }));
    expect(records.every(r => r.reviewer_user_id === "r1")).toBe(true);
  });

  it("rejects an invalid or too-large count before any request", async () => {
    const user = userEvent.setup(); render(<AssignmentPage />);
    await screen.findByLabelText("تحديد item-1");
    await user.selectOptions(screen.getByLabelText("المراجع للإسناد بالعدد"), "r1");
    const input = screen.getByLabelText("عدد التعليقات المطلوب إسنادها");
    const button = screen.getByRole("button", { name: "إسناد العدد" });
    expect(button).toBeDisabled();                                                                // no count typed
    await user.type(input, "0");
    await user.click(button);
    expect(await screen.findByText("أدخل عددًا صحيحًا موجبًا.")).toBeInTheDocument();
    await user.clear(input);
    await user.type(input, "24");
    await user.click(button);
    expect(await screen.findByText("العدد المطلوب أكبر من المتاح (23).")).toBeInTheDocument();
    expect(api.adminAssignByFilter).not.toHaveBeenCalled();
  });

  it("shows the shortfall reported by the server", async () => {
    const user = userEvent.setup(); render(<AssignmentPage />);
    await screen.findByLabelText("تحديد item-1");
    vi.mocked(api.adminAssignByFilter).mockResolvedValueOnce({ requested: 10, assigned: 7, shortfall: 3, matched_before: 23, remaining_eligible: 16 });
    await user.selectOptions(screen.getByLabelText("المراجع للإسناد بالعدد"), "r1");
    await user.type(screen.getByLabelText("عدد التعليقات المطلوب إسنادها"), "10");
    await user.click(screen.getByRole("button", { name: "إسناد العدد" }));
    expect(await screen.findByText(/النقص 3 لعدم توفر تعليقات مؤهلة كافية/)).toBeInTheDocument();
    expect(screen.getByText(/تم إسناد/).textContent).toContain("7");
  });

  it("disables count assignment when already-assigned comments are included, because it never reassigns", async () => {
    const user = userEvent.setup(); render(<AssignmentPage />);
    await screen.findByLabelText("تحديد item-1");
    await user.click(screen.getByLabelText("غير المسند فقط"));
    await screen.findByText(/الإسناد بالعدد يشمل التعليقات غير المسندة فقط/);
    expect(screen.getByLabelText("عدد التعليقات المطلوب إسنادها")).toBeDisabled();
    expect(screen.getByRole("button", { name: /إسناد كل المطابقة/ })).toBeDisabled();
  });
});
