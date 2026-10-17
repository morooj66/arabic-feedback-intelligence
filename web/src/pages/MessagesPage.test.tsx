import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { DirectMessage } from "../api/types";
import { MessagesPage } from "./MessagesPage";

vi.mock("../api/client", () => ({
  api: { messagesConversations: vi.fn(), messagesConversation: vi.fn(), messagesSend: vi.fn(), messagesMarkRead: vi.fn() },
}));
const mocked = api as unknown as Record<"messagesConversations" | "messagesConversation" | "messagesSend" | "messagesMarkRead", ReturnType<typeof vi.fn>>;
const msg = (id: string, role: "ADMIN" | "REVIEWER", body: string, read: string | null = null): DirectMessage => ({
  message_id: id, reviewer_user_id: "r1", sender_user_id: role === "ADMIN" ? "a1" : "r1", sender_role: role, sender_name: role === "ADMIN" ? "المسؤول" : "مراجع اختبار", body,
  created_at: "2026-08-01T09:00:00Z", read_at: read,
});
beforeEach(() => { Object.values(mocked).forEach(f => f.mockReset()); mocked.messagesMarkRead.mockResolvedValue({ reviewer_user_id: "r1", marked: 1 }); });

describe("Work Messages", () => {
  it("keeps production truthful when the messaging service is not configured on the server", async () => {
    mocked.messagesConversation.mockRejectedValue(Object.assign(new Error("x"), { status: 503 }));
    render(<MessagesPage role="REVIEWER" />);
    expect(await screen.findByText("التراسل الداخلي غير متاح بعد")).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("reviewer: shows history, flags and marks unread admin messages, and sends with the reviewer's own conversation", async () => {
    const user = userEvent.setup();
    mocked.messagesConversation.mockResolvedValue({ reviewer_user_id: "r1", reviewer_name: "مراجع اختبار", unread: 1, messages: [msg("m1", "REVIEWER", "سؤال سابق", "t"), msg("m2", "ADMIN", "رد الإدارة الجديد")] });
    mocked.messagesSend.mockResolvedValue(msg("m3", "REVIEWER", "شكرا"));
    render(<MessagesPage role="REVIEWER" />);
    expect(await screen.findByText("رد الإدارة الجديد")).toBeInTheDocument();
    expect(screen.getByText("سؤال سابق")).toBeInTheDocument();
    expect(screen.getAllByText("جديدة")).toHaveLength(1);
    await waitFor(() => expect(mocked.messagesMarkRead).toHaveBeenCalledWith(undefined));
    expect(mocked.messagesConversations).not.toHaveBeenCalled();                       // a reviewer never asks for the list of conversations
    expect(screen.getByRole("button", { name: "إرسال" })).toBeDisabled();              // empty / whitespace is not sendable
    await user.type(screen.getByRole("textbox"), "   ");
    expect(screen.getByRole("button", { name: "إرسال" })).toBeDisabled();
    await user.clear(screen.getByRole("textbox")); await user.type(screen.getByRole("textbox"), "شكرا");
    await user.click(screen.getByRole("button", { name: "إرسال" }));
    await waitFor(() => expect(mocked.messagesSend).toHaveBeenCalledTimes(1));
    expect(mocked.messagesSend.mock.calls[0][0]).toBe("شكرا");
    expect(mocked.messagesSend.mock.calls[0][1]).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
    expect(mocked.messagesSend.mock.calls[0][2]).toBeUndefined();
    await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue(""));
  });

  it("a failed send keeps the text and a retry re-uses the SAME client id (so a retry cannot duplicate)", async () => {
    const user = userEvent.setup();
    mocked.messagesConversation.mockResolvedValue({ reviewer_user_id: "r1", reviewer_name: "م", unread: 0, messages: [] });
    mocked.messagesSend.mockRejectedValueOnce(Object.assign(new Error("net"), { status: 500 })).mockResolvedValueOnce(msg("m9", "REVIEWER", "نص"));
    render(<MessagesPage role="REVIEWER" />);
    await user.type(await screen.findByRole("textbox"), "نص");
    await user.click(screen.getByRole("button", { name: "إرسال" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("يمكنك إعادة المحاولة دون تكرارها");
    expect(screen.getByRole("textbox")).toHaveValue("نص");
    await user.click(screen.getByRole("button", { name: "إرسال" }));
    await waitFor(() => expect(mocked.messagesSend).toHaveBeenCalledTimes(2));
    expect(mocked.messagesSend.mock.calls[1][1]).toBe(mocked.messagesSend.mock.calls[0][1]);
  });

  it("admin: lists reviewer conversations with unread counts, opens one, and replies to THAT reviewer", async () => {
    const user = userEvent.setup();
    mocked.messagesConversations.mockResolvedValue({ conversations: [
      { reviewer_user_id: "r1", reviewer_name: "المراجع الأول", unread: 2, last_message: { body: "آخر رسالة", sender_role: "REVIEWER", created_at: "2026-08-01T09:00:00Z" } },
      { reviewer_user_id: "r2", reviewer_name: "المراجع الثاني", unread: 0, last_message: null },
    ] });
    mocked.messagesConversation.mockResolvedValue({ reviewer_user_id: "r1", reviewer_name: "المراجع الأول", unread: 2, messages: [msg("m1", "REVIEWER", "نص المراجع")] });
    mocked.messagesSend.mockResolvedValue(msg("m2", "ADMIN", "رد"));
    render(<MessagesPage role="ADMIN" />);
    expect(await screen.findByText("2 غير مقروءة", { selector: "bdi" })).toBeInTheDocument();
    expect(screen.getByText("لا توجد رسائل بعد", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByText("اختر محادثة عمل")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /المراجع الأول/ }));
    expect(await screen.findByText("نص المراجع")).toBeInTheDocument();
    expect(mocked.messagesConversation).toHaveBeenLastCalledWith("r1");
    await user.type(screen.getByRole("textbox"), "رد");
    await user.click(screen.getByRole("button", { name: "إرسال" }));
    await waitFor(() => expect(mocked.messagesSend).toHaveBeenCalled());
    expect(mocked.messagesSend.mock.calls[0][2]).toBe("r1");
    await user.click(screen.getByRole("button", { name: "غير المقروءة" }));
    expect(screen.queryByRole("button", { name: /المراجع الثاني/ })).not.toBeInTheDocument();
  });

  it("renders the isolated design preview with work context and restrained thread", async () => {
    const user = userEvent.setup();
    render(<MessagesPage preview />);
    expect(screen.getByRole("heading", { name: "الرسائل" })).toBeInTheDocument();
    expect(screen.getByText("تعليق #review-123")).toBeInTheDocument();
    expect(screen.getByText("مسؤول النظام")).toBeInTheDocument();
    expect(screen.getByText("مراجع")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "عرض المراجعة" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "إرسال" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /review-108/ }));
    expect(screen.getByText("تعليق #review-108")).toBeInTheDocument();
  });

  it("supports preview search and a truthful no-match state", async () => {
    const user = userEvent.setup();
    render(<MessagesPage preview />);
    await user.type(screen.getByRole("searchbox", { name: "بحث في المحادثات" }), "غير موجود");
    expect(screen.getByText("لا توجد محادثات مطابقة")).toBeInTheDocument();
    expect(screen.queryByText("تعليق #review-123")).not.toBeInTheDocument();
  });
});
