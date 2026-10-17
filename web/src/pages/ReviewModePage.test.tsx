import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { QueueItem, ReviewDetail } from "../api/types";
import { ReviewerWorkspace } from "./ReviewerWorkspace";

vi.mock("../api/client", () => ({ api: { session: vi.fn(), counts: vi.fn(), queue: vi.fn(), item: vi.fn(), claim: vi.fn(), sendAsIs: vi.fn(), editAndSend: vi.fn(), noResponse: vi.fn(), logout: vi.fn() } }));
let rows: ReviewDetail[];
const make = (id: string): ReviewDetail => ({ comment_id: id, source_reference: null, original_comment: "أقترح إضافة أمثلة. [PHONE]", comment_created_at: "2026-09-19T10:00:00Z", original_title: null, entry_id: null, platform: null, original_type: null, primary_intent: "SUGGESTION", topic_subtype: "DICTIONARY_SUGGESTION", sentiment: null, confidence: null, needs_review: true, generation_routing_decision: null, generation_status: "GENERATED", generation_reason_code: null, suggested_response: "شكرًا لمقترحكم، وملاحظاتكم محل تقدير.", review_status: "DRAFT_READY", reviewer_id: "self", reviewer_display_name: null, reviewer_action: null, reviewed_at: null, edited_response: null, final_response: null, no_response_reason: null, ready_to_send: false, outbound_status: null, outbound_error_code: null, review_priority: 2, priority_reason: null, row_version: 4, allowed_actions: ["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"] });
function complete(id: string, action: string, text?: string): ReviewDetail {
  const d = rows.find(r => r.comment_id === id)!;
  Object.assign(d, { review_status: action === "NO_RESPONSE_NEEDED" ? "NO_REPLY" : action === "EDIT_AND_SEND" ? "EDITED_AND_APPROVED" : "APPROVED", reviewer_action: action, row_version: d.row_version + 1, allowed_actions: [], final_response: text || d.suggested_response, outbound_status: action === "NO_RESPONSE_NEEDED" ? null : "PENDING" });
  return { ...d };
}
async function start() {
  const user = userEvent.setup(); render(<MemoryRouter initialEntries={["/review"]}><ReviewerWorkspace /></MemoryRouter>);
  await screen.findByText(`قائمة المراجعة (${rows.length})`);
  await user.click(screen.getByRole("button", { name: "بدء المراجعة" }));
  return user;
}
const button = (name: string) => screen.getByRole("button", { name });
beforeEach(() => {
  vi.resetAllMocks(); rows = [make("one"), make("two")];
  vi.mocked(api.session).mockResolvedValue({ user_id: "self", display_name: "مراجع اختبار", role: "REVIEWER", auth_mode: "session" });
  vi.mocked(api.counts).mockResolvedValue({ all: 2, pending: 2, approved: 0, no_reply: 0, send_failed: 0, by_status: {} });
  vi.mocked(api.queue).mockImplementation(async p => {
    const pending = rows.filter(r => r.allowed_actions.length);
    return { items: pending.slice(p.offset, p.offset + p.limit).map(d => ({ ...d, comment_preview: d.original_comment, claimed_by: null, ingested_at: null } as QueueItem)), total: pending.length, offset: p.offset, limit: p.limit };
  });
  vi.mocked(api.item).mockImplementation(async id => ({ ...rows.find(r => r.comment_id === id)! }));
  vi.mocked(api.claim).mockImplementation(async (id, version) => { const d = rows.find(r => r.comment_id === id)!; expect(version).toBe(d.row_version); d.review_status = "IN_REVIEW"; d.row_version++; return { ...d }; });
  vi.mocked(api.sendAsIs).mockImplementation(async id => complete(id, "SEND_AS_IS"));
  vi.mocked(api.editAndSend).mockImplementation(async (id, _v, text) => complete(id, "EDIT_AND_SEND", text));
  vi.mocked(api.noResponse).mockImplementation(async (id, _v, reason) => ({ ...complete(id, "NO_RESPONSE_NEEDED"), no_response_reason: reason, final_response: null }));
});
afterEach(() => vi.restoreAllMocks());

describe("Focused Review Mode", () => {
  it("shows safe real context, scoped progress and waits for confirmed approval before advancing", async () => {
    let resolve!: (d: ReviewDetail) => void;
    vi.mocked(api.sendAsIs).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    const user = await start(); await screen.findByLabelText("الرد المقترح");
    expect(screen.getByText("1 من 2")).toBeInTheDocument();
    expect(screen.getByText("أقترح إضافة أمثلة. [PHONE]")).toBeInTheDocument();
    expect(screen.queryByText("الثقة")).not.toBeInTheDocument();
    expect(vi.mocked(api.queue).mock.calls.every(([p]) => !p.reviewerUserId && !p.sourceFile)).toBe(true);
    await user.click(button("اعتماد كما هو"));
    expect(api.sendAsIs).toHaveBeenCalledWith("one", 5);
    expect(button("التعليق التالي")).toBeDisabled(); expect(button("العودة للقائمة")).toBeDisabled();
    expect(api.item).not.toHaveBeenCalledWith("two"); expect(screen.queryByText(/تم اعتماد الرد/)).not.toBeInTheDocument();
    await act(async () => resolve(complete("one", "SEND_AS_IS")));
    await screen.findByRole("heading", { name: "تعليق #two" });
    expect(screen.getByRole("status")).toHaveTextContent("تم اعتماد الرد — بانتظار تكامل الإرسال");
  });

  it("edits explicitly, preserves the original and sends text with expected version", async () => {
    const user = await start(); const input = await screen.findByLabelText("الرد المقترح");
    await user.click(button("تعديل الرد")); expect(input).toHaveFocus();
    fireEvent.change(input, { target: { value: "شكرًا لكم. نقدر مقترحكم بتوفير أمثلة توضيحية." } });
    expect(screen.getByText("تعديلات غير محفوظة")).toBeInTheDocument();
    expect(button("اعتماد كما هو")).toBeDisabled();
    expect(rows[0].suggested_response).toBe(make("one").suggested_response);
    await user.click(button("تعديل واعتماد"));
    expect(api.editAndSend).toHaveBeenCalledWith("one", 5, "شكرًا لكم. نقدر مقترحكم بتوفير أمثلة توضيحية.");
    await screen.findByRole("heading", { name: "تعليق #two" });
  });

  it("confirms no-response with real reason, focus trapping and no reply delivery claim", async () => {
    const user = await start(); await screen.findByLabelText("الرد المقترح");
    await user.click(button("لا يحتاج رد"));
    const modal = screen.getByRole("dialog"); const reason = within(modal).getByLabelText("سبب عدم الرد");
    expect(reason).toHaveFocus(); await user.tab({ shift: true }); expect(within(modal).getByText("إلغاء")).toHaveFocus();
    await user.selectOptions(reason, "تكرار");
    expect(within(modal).getByText("اختر السبب. لن يُرسل رد إلى الزائر.")).toBeInTheDocument();
    await user.click(within(modal).getByText("تأكيد"));
    expect(api.noResponse).toHaveBeenCalledWith("one", 5, "تكرار");
    await screen.findByRole("heading", { name: "تعليق #two" });
    expect(screen.getByRole("status")).toHaveTextContent("أُغلق بدون إرسال رد");
    expect(api.sendAsIs).not.toHaveBeenCalled();
  });

  it("preserves draft and item after mutation failure, without fake completion", async () => {
    vi.mocked(api.editAndSend).mockRejectedValueOnce(new Error("private backend error"));
    const user = await start(); fireEvent.change(await screen.findByLabelText("الرد المقترح"), { target: { value: "تعديل محفوظ في الواجهة" } });
    await user.click(button("تعديل واعتماد"));
    expect(await screen.findByRole("alert")).toHaveTextContent("تعذر تنفيذ الإجراء");
    expect(screen.getByLabelText("الرد المقترح")).toHaveValue("تعديل محفوظ في الواجهة");
    expect(api.item).not.toHaveBeenCalledWith("two"); expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(button("تعديل واعتماد")).toBeEnabled();
  });

  it("blocks stale resubmission after version conflict and reloads without losing the local draft", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(api.editAndSend).mockRejectedValueOnce(Object.assign(new Error(), { code: "version_conflict" }));
    const user = await start(); fireEvent.change(await screen.findByLabelText("الرد المقترح"), { target: { value: "رد معدّل" } });
    await user.click(button("تعديل واعتماد")); await screen.findByText(/تعارض في الحفظ/);
    expect(button("تعديل واعتماد")).toBeDisabled(); rows[0].row_version = 9;
    await user.click(button("تحديث التعليق")); await waitFor(() => expect(button("تعديل واعتماد")).toBeEnabled());
    expect(screen.getByLabelText("الرد المقترح")).toHaveValue("رد معدّل");
    await user.click(button("تعديل واعتماد")); expect(api.editAndSend).toHaveBeenLastCalledWith("one", 9, "رد معدّل");
  });

  it("respects FAILED_REVIEW_ONLY with no generated draft and only allowed decisions", async () => {
    Object.assign(rows[0], { review_status: "REVIEW_ONLY", generation_status: "FAILED_REVIEW_ONLY", suggested_response: null, allowed_actions: ["EDIT_AND_SEND", "NO_RESPONSE_NEEDED"] });
    const user = await start(); const input = await screen.findByLabelText("الرد المقترح");
    expect(input).toHaveValue(""); expect(button("اعتماد كما هو")).toBeDisabled(); expect(button("تعديل واعتماد")).toBeDisabled();
    expect(screen.getByText("لا توجد مسودة مولَّدة. اكتب الرد المؤسسي أو أغلق بدون رد.")).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "رد مؤسسي من المراجع" } }); await user.click(button("تعديل واعتماد"));
    expect(api.editAndSend).toHaveBeenCalledWith("one", 5, "رد مؤسسي من المراجع"); expect(api.sendAsIs).not.toHaveBeenCalled();
  });

  it("guards next, list, sidebar, logout and browser unload when edits are unsaved", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = await start(); fireEvent.change(await screen.findByLabelText("الرد المقترح"), { target: { value: "مسودة غير محفوظة" } });
    await user.click(button("التعليق التالي")); await user.click(button("العودة للقائمة")); await user.click(button("الرسائل"));
    await user.click(button("الحساب")); await user.click(button("تسجيل الخروج"));
    expect(confirm).toHaveBeenCalledTimes(4); expect(api.logout).not.toHaveBeenCalled(); expect(api.item).not.toHaveBeenCalledWith("two");
    const event = new Event("beforeunload", { cancelable: true }); window.dispatchEvent(event); expect(event.defaultPrevented).toBe(true);
    confirm.mockReturnValue(true); await user.click(button("التعليق التالي")); await screen.findByRole("heading", { name: "تعليق #two" });
    await user.click(button("التعليق السابق")); await screen.findByRole("heading", { name: "تعليق #one" });
    expect(api.sendAsIs).not.toHaveBeenCalled(); expect(api.editAndSend).not.toHaveBeenCalled();
  });

  it("distinguishes item loading and item failure without showing a stale comment", async () => {
    let reject!: (e: Error) => void;
    const user = await start(); await screen.findByLabelText("الرد المقترح");
    vi.mocked(api.item).mockImplementationOnce(() => new Promise((_, r) => { reject = r; }));
    await user.click(button("التعليق التالي")); expect(screen.getByRole("status")).toHaveTextContent("جاري تحميل التعليق");
    expect(screen.queryByLabelText("الرد المقترح")).not.toBeInTheDocument();
    await act(async () => reject(Object.assign(new Error(), { code: "not_found" })));
    expect(screen.getByRole("alert")).toHaveTextContent("تعذر العثور على التعليق");
    expect(screen.queryByText(/لا توجد عناصر أخرى/)).not.toBeInTheDocument();
    await user.click(button("تحديث التعليق")); await screen.findByRole("heading", { name: "تعليق #two" });
  });

  it("disallows decisions when a claim is held by another reviewer", async () => {
    vi.mocked(api.claim).mockRejectedValueOnce(Object.assign(new Error(), { code: "claimed" }));
    await start(); await screen.findByText("هذا التعليق قيد المراجعة لدى مراجع آخر.");
    expect(button("اعتماد كما هو")).toBeDisabled(); expect(button("لا يحتاج رد")).toBeDisabled();
    expect(screen.getByLabelText("الرد المقترح")).toHaveAttribute("readonly");
  });

  it("keeps a confirmed decision read-only when the following queue refresh fails", async () => {
    const user = await start(); await screen.findByLabelText("الرد المقترح");
    vi.mocked(api.queue).mockRejectedValueOnce(new Error("unavailable"));
    await user.click(button("اعتماد كما هو")); await screen.findByText(/تعذر تحديث قائمة العمل/);
    expect(button("اعتماد كما هو")).toBeDisabled(); expect(screen.getByLabelText("الرد المقترح")).toHaveAttribute("readonly");
    expect(api.item).not.toHaveBeenCalledWith("two");
  });

  it("keeps no-response failure in its dialog without progression", async () => {
    vi.mocked(api.noResponse).mockRejectedValueOnce(new Error("failure"));
    const user = await start(); await screen.findByLabelText("الرد المقترح"); await user.click(button("لا يحتاج رد"));
    await user.click(within(screen.getByRole("dialog")).getByText("تأكيد"));
    expect(within(screen.getByRole("dialog")).getByRole("alert")).toHaveTextContent("تعذر إغلاق التعليق");
    expect(api.item).not.toHaveBeenCalledWith("two"); await user.keyboard("{Escape}"); expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows honest current-page exhaustion after the last confirmed decision", async () => {
    rows = [make("one")]; const user = await start(); await screen.findByLabelText("الرد المقترح"); await user.click(button("اعتماد كما هو"));
    await waitFor(() => expect(button("التعليق التالي")).toBeDisabled());
    expect(screen.queryByText("اكتملت جميع المراجعات")).not.toBeInTheDocument();
    await user.click(button("العودة للقائمة")); await screen.findByText("قائمة المراجعة (0)");
  });
});
