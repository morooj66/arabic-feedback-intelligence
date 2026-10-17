import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { ApiError, QueueItem, ReviewDetail } from "../api/types";
import { ReviewerWorkspace } from "./ReviewerWorkspace";

vi.mock("../api/client", () => ({ api: { session: vi.fn(), counts: vi.fn(), queue: vi.fn(), item: vi.fn(), claim: vi.fn(), sendAsIs: vi.fn(), editAndSend: vi.fn(), validateReply: vi.fn(), noResponse: vi.fn(), logout: vi.fn() } }));

const DRAFT = "شكرًا لمقترحكم، وملاحظاتكم محل تقدير.";
const FACT = "  معنى كلمة «سوار» هو حلية تلبس في المعصم.  ";   // leading / trailing spaces must survive untouched in what the reviewer sees
let rows: ReviewDetail[];
const make = (id: string): ReviewDetail => ({ comment_id: id, source_reference: null, original_comment: "ما معنى سوار", comment_created_at: "2026-09-19T10:00:00Z", original_title: null, entry_id: null, platform: null, original_type: null, primary_intent: "INQUIRY", topic_subtype: "GENERAL", sentiment: null, confidence: null, needs_review: true, generation_routing_decision: null, generation_status: "GENERATED", generation_reason_code: null, suggested_response: DRAFT, review_status: "DRAFT_READY", reviewer_id: "self", reviewer_display_name: null, reviewer_action: null, reviewed_at: null, edited_response: null, final_response: null, no_response_reason: null, ready_to_send: false, outbound_status: null, outbound_error_code: null, review_priority: 2, priority_reason: null, row_version: 4, allowed_actions: ["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"] });
function complete(id: string, action: string, text?: string): ReviewDetail {
  const d = rows.find(r => r.comment_id === id)!;
  Object.assign(d, { review_status: action === "EDIT_AND_SEND" ? "EDITED_AND_APPROVED" : "APPROVED", reviewer_action: action, row_version: d.row_version + 1, allowed_actions: [], final_response: text || d.suggested_response, outbound_status: "PENDING" });
  return { ...d };
}
function apiError(status: number, code: string, payload: Record<string, unknown>): ApiError {
  const e: ApiError = new Error(code); e.status = status; e.code = code; e.payload = { error: code, ...payload }; return e;
}
const ATTEST = (codes = ["unsupported_definition"]) => apiError(422, "attestation_required", { attestable_findings: codes, draft_preserved: true, sent: false });
const validation = (over: Partial<Awaited<ReturnType<typeof api.validateReply>>> = {}) => ({ valid: false, blocking_findings: [], attestable_findings: ["unsupported_definition"], attestation_required: true, attestation_permitted: true, ...over });

async function start() {
  const user = userEvent.setup(); render(<MemoryRouter initialEntries={["/review"]}><ReviewerWorkspace /></MemoryRouter>);
  await screen.findByText(`قائمة المراجعة (${rows.length})`);
  await user.click(screen.getByRole("button", { name: "بدء المراجعة" }));
  await screen.findByLabelText("الرد المقترح");
  return user;
}
async function writeFact(user: ReturnType<typeof userEvent.setup>) {
  const box = screen.getByLabelText("الرد المقترح") as HTMLTextAreaElement;
  await user.clear(box); await user.click(box); await user.paste(FACT);
  return box;
}
const button = (name: string) => screen.getByRole("button", { name });
const RAW_CODE = /unsupported_|attestation_|draft_validation|unapproved_url|pii_in_draft|internal_leak/;

beforeEach(() => {
  vi.resetAllMocks(); rows = [make("one"), make("two")];
  vi.mocked(api.session).mockResolvedValue({ user_id: "self", display_name: "مراجع اختبار", role: "REVIEWER", auth_mode: "session" });
  vi.mocked(api.counts).mockResolvedValue({ all: 2, pending: 2, approved: 0, no_reply: 0, send_failed: 0, by_status: {} });
  vi.mocked(api.queue).mockImplementation(async p => {
    const pending = rows.filter(r => r.allowed_actions.length);
    return { items: pending.slice(p.offset, p.offset + p.limit).map(d => ({ ...d, comment_preview: d.original_comment, claimed_by: null, ingested_at: null } as QueueItem)), total: pending.length, offset: p.offset, limit: p.limit };
  });
  vi.mocked(api.item).mockImplementation(async id => ({ ...rows.find(r => r.comment_id === id)! }));
  vi.mocked(api.claim).mockImplementation(async (id) => { const d = rows.find(r => r.comment_id === id)!; d.review_status = "IN_REVIEW"; d.row_version++; return { ...d }; });
});
afterEach(() => vi.restoreAllMocks());

describe("Reviewer attestation of unverifiable facts", () => {
  it("explains in Arabic, never shows backend codes, preserves the reply, and sends exactly the required codes after revalidation", async () => {
    vi.mocked(api.editAndSend).mockRejectedValueOnce(ATTEST()).mockImplementationOnce(async (id, _v, text) => complete(id, "EDIT_AND_SEND", text));
    vi.mocked(api.validateReply).mockResolvedValue(validation());
    const user = await start(); const box = await writeFact(user);
    await user.click(button("تعديل واعتماد"));
    expect(await screen.findByText("يتضمن الرد معلومات لا يمكن التحقق منها من مصدر معتمد")).toBeInTheDocument();
    expect(screen.getByText("تعريف أو معلومة معجمية عن كلمة لا يمكن التحقق منها من مصدر معتمد.")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(RAW_CODE);
    const before = box.value;
    expect(before).toBe(FACT);                                                          // exactly what the reviewer wrote, spaces included
    const confirm = button("تأكيد وإرسال"); expect(confirm).toBeDisabled();            // missing confirmation cannot send
    expect(api.editAndSend).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("checkbox")); expect(confirm).toBeEnabled();
    await user.click(confirm);
    await waitFor(() => expect(api.editAndSend).toHaveBeenCalledTimes(2));
    expect(api.validateReply).toHaveBeenCalledWith("one", before);                       // revalidated against the current text first
    expect(vi.mocked(api.editAndSend).mock.calls[1]).toEqual(["one", 5, before, { confirmed: true, finding_codes: ["unsupported_definition"] }]);
    expect(vi.mocked(api.editAndSend).mock.calls[0][2]).toBe(before);                    // same exact text both times
    await waitFor(() => expect(screen.queryByText("تأكيد وإرسال")).not.toBeInTheDocument());
  });

  it("reply text is untouched by every safety outcome", async () => {
    vi.mocked(api.editAndSend).mockRejectedValue(ATTEST());
    const user = await start(); const box = await writeFact(user); const typed = box.value;
    await user.click(button("تعديل واعتماد")); await screen.findByText("تأكيد وإرسال");
    expect((screen.getByLabelText("الرد المقترح") as HTMLTextAreaElement).value).toBe(typed);
  });

  it("an unauthorised reviewer gets an Arabic explanation and no way to confirm", async () => {
    vi.mocked(api.editAndSend).mockRejectedValue(apiError(403, "attestation_not_permitted", { attestable_findings: ["unsupported_definition"] }));
    const user = await start(); await writeFact(user); await user.click(button("تعديل واعتماد"));
    expect(await screen.findByText("لا تملك صلاحية تأكيد هذه المعلومات")).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "تأكيد وإرسال" })).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(RAW_CODE);
    expect(api.editAndSend).toHaveBeenCalledTimes(1);
  });

  it("stale findings: a changed result re-opens the confirmation instead of sending", async () => {
    vi.mocked(api.editAndSend).mockRejectedValue(ATTEST());
    vi.mocked(api.validateReply).mockResolvedValue(validation({ attestable_findings: ["unsupported_definition", "unsupported_number"] }));
    const user = await start(); await writeFact(user); await user.click(button("تعديل واعتماد"));
    await user.click(await screen.findByRole("checkbox")); await user.click(button("تأكيد وإرسال"));
    expect(await screen.findByText("تغيّرت نتائج الفحص منذ آخر مرة؛ راجع القائمة المحدّثة وأكّدها من جديد.")).toBeInTheDocument();
    expect(screen.getByText("رقم أو إحصاء غير وارد في تعليق الزائر ولا في مصدر معتمد.")).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(api.editAndSend).toHaveBeenCalledTimes(1);                                    // nothing was sent with the stale confirmation
  });

  it("stale findings detected by the server on the send itself also re-open the confirmation", async () => {
    vi.mocked(api.editAndSend).mockRejectedValueOnce(ATTEST()).mockRejectedValueOnce(ATTEST(["unsupported_correctness_claim"]));
    vi.mocked(api.validateReply).mockResolvedValue(validation());
    const user = await start(); await writeFact(user); await user.click(button("تعديل واعتماد"));
    await user.click(await screen.findByRole("checkbox")); await user.click(button("تأكيد وإرسال"));
    expect(await screen.findByText("تغيّرت نتائج الفحص منذ آخر مرة؛ راجع القائمة المحدّثة وأكّدها من جديد.")).toBeInTheDocument();
    expect(screen.getByText("حكم بصحة مدخل أو ادعاء أو بخطئه لا يمكن التحقق منه من مصدر معتمد.")).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).not.toBeChecked();
  });

  it.each([
    ["unsupported_commitment", "الرد يتضمن وعدًا أو التزامًا أو ادعاء إجراء (مثل: سنقوم، تم، قريبًا، الفريق المختص). أعد صياغته دون وعود."],
    ["unapproved_url", "الرد يحتوي على رابط غير معتمد. احذف الرابط، أو استخدم الرابط الرسمي المعتمد فقط."],
    ["unsupported_policy_claim", "الرد يذكر سياسة أو إجراءً أو مدة أو قناة تواصل غير معتمدة."],
    ["pii_in_draft", "الرد يتضمن بيانات شخصية (مثل رقم هاتف)."],
  ])("%s is blocked with Arabic text and cannot be confirmed away", async (code, arabic) => {
    vi.mocked(api.editAndSend).mockRejectedValue(apiError(422, "draft_validation_failed", { blocking_findings: [code], attestable_findings: ["unsupported_definition"] }));
    const user = await start(); await writeFact(user); await user.click(button("تعديل واعتماد"));
    expect(await screen.findByText("لا يمكن إرسال هذا الرد")).toBeInTheDocument();
    expect(screen.getByText(arabic)).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(RAW_CODE);
    expect(api.editAndSend).toHaveBeenCalledTimes(1);
  });

  it("a blocking finding discovered at revalidation stops the confirmation", async () => {
    vi.mocked(api.editAndSend).mockRejectedValue(ATTEST());
    vi.mocked(api.validateReply).mockResolvedValue(validation({ blocking_findings: ["unapproved_url"] }));
    const user = await start(); await writeFact(user); await user.click(button("تعديل واعتماد"));
    await user.click(await screen.findByRole("checkbox")); await user.click(button("تأكيد وإرسال"));
    expect(await screen.findByText("لا يمكن إرسال هذا الرد")).toBeInTheDocument();
    expect(api.editAndSend).toHaveBeenCalledTimes(1);
  });

  it("permission removed at revalidation is shown as denied", async () => {
    vi.mocked(api.editAndSend).mockRejectedValue(ATTEST());
    vi.mocked(api.validateReply).mockResolvedValue(validation({ attestation_permitted: false }));
    const user = await start(); await writeFact(user); await user.click(button("تعديل واعتماد"));
    await user.click(await screen.findByRole("checkbox")); await user.click(button("تأكيد وإرسال"));
    expect(await screen.findByText("لا تملك صلاحية تأكيد هذه المعلومات")).toBeInTheDocument();
    expect(api.editAndSend).toHaveBeenCalledTimes(1);
  });

  it("the as-is action uses the same attestation flow", async () => {
    vi.mocked(api.sendAsIs).mockRejectedValueOnce(ATTEST()).mockImplementationOnce(async id => complete(id, "SEND_AS_IS"));
    vi.mocked(api.validateReply).mockResolvedValue(validation());
    const user = await start();
    await user.click(button("اعتماد كما هو"));
    await user.click(await screen.findByRole("checkbox")); await user.click(button("تأكيد وإرسال"));
    await waitFor(() => expect(api.sendAsIs).toHaveBeenCalledTimes(2));
    expect(api.validateReply).toHaveBeenCalledWith("one", DRAFT);
    expect(vi.mocked(api.sendAsIs).mock.calls[1]).toEqual(["one", 5, { confirmed: true, finding_codes: ["unsupported_definition"] }]);
  });

  it("editing the reply dismisses the panel so a confirmation can never apply to different text", async () => {
    vi.mocked(api.editAndSend).mockRejectedValue(ATTEST());
    const user = await start(); const box = await writeFact(user); await user.click(button("تعديل واعتماد"));
    await screen.findByRole("checkbox");
    await user.type(box, "x");
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("ordinary sends are unchanged: no extra arguments and no validation call", async () => {
    vi.mocked(api.sendAsIs).mockImplementation(async id => complete(id, "SEND_AS_IS"));
    const user = await start(); await user.click(button("اعتماد كما هو"));
    await waitFor(() => expect(api.sendAsIs).toHaveBeenCalledWith("one", 5));
    expect(api.validateReply).not.toHaveBeenCalled();
  });
});
