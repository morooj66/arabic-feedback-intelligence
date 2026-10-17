import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { QueueItem, QueuePage, ReviewDetail } from "../api/types";
import { ReviewerWorkspace } from "./ReviewerWorkspace";

vi.mock("../api/client", () => ({ api: { session: vi.fn(), counts: vi.fn(), queue: vi.fn(), item: vi.fn(), claim: vi.fn(), activeExcelSession: vi.fn(), analytics: vi.fn() } }));
let records: QueueItem[];
const row = (i: number): QueueItem => ({ comment_id: `assigned-${i}`, comment_preview: `تعليق آمن ${i} [PHONE]`, review_status: "DRAFT_READY", primary_intent: i % 2 ? "SUGGESTION" : "INQUIRY", topic_subtype: i % 2 ? "DICTIONARY_SUGGESTION" : "GENERAL", review_priority: i % 3 + 1, priority_reason: null, claimed_by: null, comment_created_at: null, ingested_at: "2026-09-19T10:00:00+03:00", suggested_response: "مسودة للمراجعة", reviewer_action: null, reviewer_display_name: null, reviewed_at: null });
function detail(id: string): ReviewDetail {
  return { ...records.find(r => r.comment_id === id)!, comment_id: id, source_reference: null, original_comment: "التعليق الكامل [PHONE]", original_title: null, entry_id: null, platform: null, original_type: null, sentiment: null, confidence: null, needs_review: true, generation_routing_decision: null, generation_status: null, generation_reason_code: null, reviewer_id: "reviewer-self", edited_response: null, final_response: null, no_response_reason: null, ready_to_send: false, outbound_status: null, outbound_error_code: null, row_version: 1, allowed_actions: ["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"] };
}
function renderAssigned() { return render(<MemoryRouter initialEntries={["/review"]}><ReviewerWorkspace /></MemoryRouter>); }
beforeEach(() => {
  vi.resetAllMocks(); records = Array.from({ length: 10 }, (_, i) => row(i + 1));
  vi.mocked(api.session).mockResolvedValue({ user_id: "reviewer-self", display_name: "مراجع اختبار", role: "REVIEWER", auth_mode: "session" });
  vi.mocked(api.counts).mockResolvedValue({ all: 12, pending: 10, approved: 2, no_reply: 0, send_failed: 0, by_status: {} });
  vi.mocked(api.queue).mockImplementation(async p => {
    const filtered = records.filter(r => (p.filter !== "pending" || r.review_status === "DRAFT_READY" || r.review_status === "IN_REVIEW") && (!p.search || `${r.comment_id} ${r.comment_preview}`.includes(p.search)) && (!p.primaryIntents?.length || p.primaryIntents.includes(r.primary_intent!)) && (!p.topicSubtypes?.length || p.topicSubtypes.includes(r.topic_subtype!)));
    return { items: filtered.slice(p.offset, p.offset + p.limit), total: filtered.length, offset: p.offset, limit: p.limit };
  });
  vi.mocked(api.item).mockImplementation(async id => detail(id));
  vi.mocked(api.claim).mockImplementation(async id => ({ ...detail(id), review_status: "IN_REVIEW", row_version: 2 }));
});

describe("Reviewer Assigned work", () => {
  it("shows scoped work and real counts without auto-opening or admin data requests", async () => {
    renderAssigned(); await screen.findByText("قائمة المراجعة (10)");
    const nav = within(screen.getByRole("navigation", { name: "التنقل الرئيسي" }));
    expect(nav.getAllByRole("button")).toHaveLength(2);
    expect(nav.getByRole("button", { name: "المسند لي" })).toHaveAttribute("aria-current", "page");
    expect(nav.getByRole("button", { name: "الرسائل" })).toBeInTheDocument();
    expect(screen.getByText("تعليق آمن 1 [PHONE]")).toBeInTheDocument();
    expect(screen.queryByText("مسودة للمراجعة")).not.toBeInTheDocument();
    expect(api.item).not.toHaveBeenCalled(); expect(api.claim).not.toHaveBeenCalled();
    expect(api.analytics).not.toHaveBeenCalled(); expect(api.activeExcelSession).not.toHaveBeenCalled();
    expect(api.queue).toHaveBeenLastCalledWith(expect.objectContaining({ filter: "pending", sourceFile: undefined, limit: 8 }));
    expect(vi.mocked(api.queue).mock.calls.every(([p]) => !p.reviewerUserId)).toBe(true);
    expect(screen.getAllByRole("button", { name: "بدء المراجعة" })).toHaveLength(1);
    expect(screen.getByText("مكتمل").nextSibling).toHaveTextContent("2");
  });

  it("combines real Intent/Topic/search and retains context on entry and return", async () => {
    const user = userEvent.setup(); renderAssigned(); await screen.findByText("قائمة المراجعة (10)");
    await user.click(screen.getByRole("button", { name: "التصنيف" }));
    const panel = screen.getByRole("dialog", { name: "تصفية حسب التصنيف" });
    await user.click(within(panel).getByLabelText("اقتراح", { exact: true }));
    await user.click(within(panel).getByLabelText("اقتراح معجمي"));
    await user.type(screen.getByLabelText("بحث"), "assigned-3");
    await screen.findByText("قائمة المراجعة (1)");
    expect(api.queue).toHaveBeenLastCalledWith(expect.objectContaining({ primaryIntents: ["SUGGESTION"], topicSubtypes: ["DICTIONARY_SUGGESTION"], search: "assigned-3" }));
    await user.click(screen.getByRole("button", { name: "بدء المراجعة" }));
    await screen.findByLabelText("الرد المقترح");
    expect(api.item).toHaveBeenCalledWith("assigned-3");
    expect(api.claim).toHaveBeenCalledWith("assigned-3", 1);
    await user.click(screen.getByRole("button", { name: "العودة للقائمة" }));
    expect(screen.getByLabelText("بحث")).toHaveValue("assigned-3");
    expect(screen.getByText("النية: اقتراح")).toBeInTheDocument();
    expect(screen.getByText("الموضوع: اقتراح معجمي")).toBeInTheDocument();
  });

  it("starts from the current page rather than claiming the entire filtered set", async () => {
    const user = userEvent.setup(); renderAssigned(); await screen.findByText("قائمة المراجعة (10)");
    await user.click(screen.getByRole("button", { name: "التالي" }));
    await screen.findByText("#assigned-9");
    expect(api.queue).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 8, limit: 8 }));
    await user.click(screen.getByRole("button", { name: "بدء المراجعة" }));
    await screen.findByLabelText("الرد المقترح");
    expect(api.item).toHaveBeenCalledWith("assigned-9");
    expect(api.claim).toHaveBeenCalledTimes(1);
  });

  it("disables start and hides stale rows while reloading or after a queue failure", async () => {
    const user = userEvent.setup(); renderAssigned(); await screen.findByText("قائمة المراجعة (10)");
    let reject!: (e: Error) => void;
    vi.mocked(api.queue).mockImplementationOnce(() => new Promise<QueuePage>((_, r) => { reject = r; }));
    fireEvent.change(screen.getByLabelText("بحث"), { target: { value: "عطل" } });
    expect(screen.getByRole("button", { name: "بدء المراجعة" })).toBeDisabled();
    expect(screen.queryByText("تعليق آمن 1 [PHONE]")).not.toBeInTheDocument();
    await act(async () => reject(new Error("private database error")));
    expect(screen.getByRole("alert")).toHaveTextContent("تعذر تحميل القائمة");
    expect(screen.queryByText(/private database/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "إعادة تحميل القائمة" }));
    await screen.findByText("لا توجد نتائج");
    expect(screen.getByLabelText("بحث")).toHaveValue("عطل");
  });

  it("distinguishes unassigned, caught-up and filtered-empty states", async () => {
    records = []; vi.mocked(api.counts).mockResolvedValue({ all: 0, pending: 0, approved: 0, no_reply: 0, send_failed: 0, by_status: {} });
    const { unmount } = renderAssigned(); await screen.findByText("لا توجد تعليقات مسندة لك حاليًا");
    expect(screen.queryByRole("button", { name: "بدء المراجعة" })).not.toBeInTheDocument(); unmount();
    vi.mocked(api.counts).mockResolvedValue({ all: 2, pending: 0, approved: 2, no_reply: 0, send_failed: 0, by_status: {} });
    const user = userEvent.setup(); renderAssigned(); await screen.findByText("اكتملت جميع المراجعات");
    await user.type(screen.getByLabelText("بحث"), "لا يوجد"); await screen.findByText("لا توجد نتائج");
    expect(screen.getByRole("button", { name: "بدء المراجعة" })).toBeDisabled();
  });

  it("keeps Priority as sorting only and resets the supported filters", async () => {
    const user = userEvent.setup(); renderAssigned(); await screen.findByText("قائمة المراجعة (10)");
    expect(screen.getAllByRole("combobox")).toHaveLength(1);
    await user.selectOptions(screen.getByLabelText("ترتيب"), "newest");
    await waitFor(() => expect(api.queue).toHaveBeenLastCalledWith(expect.objectContaining({ sort: "newest" })));
    await user.click(screen.getByRole("button", { name: "مسح الفلاتر" }));
    await waitFor(() => expect(api.queue).toHaveBeenLastCalledWith(expect.objectContaining({ sort: "priority", filter: "pending", primaryIntents: [], topicSubtypes: [], search: "" })));
  });

  it("reports a missing counts summary without blocking the available queue", async () => {
    vi.mocked(api.counts).mockRejectedValueOnce(new Error("private database"));
    const user = userEvent.setup(); renderAssigned(); await screen.findByText("قائمة المراجعة (10)");
    expect(screen.getByRole("alert")).toHaveTextContent("تعذر تحميل ملخص العمل");
    expect(screen.getByRole("button", { name: "بدء المراجعة" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "إعادة تحميل الملخص" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("labels terminal history as viewing, without reclaiming completed comments", async () => {
    records = [{ ...row(1), review_status: "APPROVED" }];
    const user = userEvent.setup(); renderAssigned(); await screen.findByText("لا توجد نتائج");
    await user.click(within(screen.getByRole("group", { name: "تصفية القائمة" })).getByRole("button", { name: "تمت الموافقة" }));
    await screen.findByText("قائمة المراجعة (1)");
    expect(screen.queryByRole("button", { name: "بدء المراجعة" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "عرض المراجعات" }));
    await screen.findByLabelText("الرد المقترح");
    expect(api.claim).not.toHaveBeenCalled();
  });
});
