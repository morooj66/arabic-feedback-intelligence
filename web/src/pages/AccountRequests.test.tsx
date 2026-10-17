import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { LoginPage } from "./LoginPage";
import { RequestAccountPage } from "./RequestAccountPage";
import { SettingsPage } from "./SettingsPage";

vi.mock("../api/client", () => ({
  microsoftSignInUrl: "/api/auth/microsoft/start",
  api: {
    health: vi.fn(), login: vi.fn(), requestAccount: vi.fn(), accountRequests: vi.fn(), approveAccountRequest: vi.fn(), rejectAccountRequest: vi.fn(), adminUsers: vi.fn(),
  },
}));
const m = api as unknown as Record<string, ReturnType<typeof vi.fn>>;
beforeEach(() => { Object.values(m).forEach(f => f.mockReset()); m.adminUsers.mockResolvedValue({ users: [] }); });

describe("new account request", () => {
  it("login page offers the request link only when the server supports it, and keeps Microsoft visible but disabled", async () => {
    m.health.mockResolvedValue({ ok: true, demo_auth: false, entra: false, account_requests: true });
    render(<MemoryRouter><LoginPage /></MemoryRouter>);
    expect(await screen.findByRole("button", { name: /طلب حساب جديد/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "الدخول عبر Microsoft" })).toBeDisabled();
    expect(screen.getByText("الدخول عبر Microsoft غير متاح حاليًا — لم يتم إعداده بعد.")).toBeInTheDocument();
  });

  it("login page hides the request link when the server does not offer it", async () => {
    m.health.mockResolvedValue({ ok: true, demo_auth: false, entra: false, account_requests: false });
    render(<MemoryRouter><LoginPage /></MemoryRouter>);
    await waitFor(() => expect(m.health).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /طلب حساب جديد/ })).not.toBeInTheDocument();
  });

  it("submits name, e-mail and password only (no role field exists) and says the account stays inactive until approved", async () => {
    const user = userEvent.setup();
    m.health.mockResolvedValue({ ok: true, demo_auth: false, entra: false, account_requests: true });
    m.requestAccount.mockResolvedValue({ ok: true });
    render(<MemoryRouter><RequestAccountPage /></MemoryRouter>);
    await user.type(await screen.findByLabelText("الاسم"), "مستخدم جديد");
    await user.type(screen.getByLabelText("البريد الإلكتروني"), "new@example.invalid");
    await user.type(screen.getByLabelText("كلمة المرور"), "a-good-password");
    expect(screen.queryByLabelText(/الصلاحية|الدور/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "إرسال الطلب" }));
    await waitFor(() => expect(m.requestAccount).toHaveBeenCalledWith("new@example.invalid", "مستخدم جديد", "a-good-password"));
    expect(await screen.findByText(/سيبقى الحساب غير مفعّل/)).toBeInTheDocument();
    expect(screen.queryByLabelText("كلمة المرور")).not.toBeInTheDocument();     // the form is gone after submission
  });

  it("shows a specific message for a rejected password and a rate-limit message for 429", async () => {
    const user = userEvent.setup();
    m.health.mockResolvedValue({ ok: true, demo_auth: false, entra: false, account_requests: true });
    m.requestAccount.mockRejectedValueOnce(Object.assign(new Error("x"), { status: 400, code: "invalid_password" })).mockRejectedValueOnce(Object.assign(new Error("x"), { status: 429, code: "too_many_requests" }));
    render(<MemoryRouter><RequestAccountPage /></MemoryRouter>);
    await user.type(await screen.findByLabelText("الاسم"), "ن");
    await user.type(screen.getByLabelText("البريد الإلكتروني"), "n@example.invalid");
    await user.type(screen.getByLabelText("كلمة المرور"), "12345678");
    await user.click(screen.getByRole("button", { name: "إرسال الطلب" }));
    expect(await screen.findByText("كلمة المرور يجب ألا تقل عن 8 أحرف.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "إرسال الطلب" }));
    expect(await screen.findByText(/تم تجاوز عدد المحاولات/)).toBeInTheDocument();
  });

  it("says plainly that requests are unavailable when the server does not support them", async () => {
    m.health.mockResolvedValue({ ok: true, demo_auth: false, entra: false });
    render(<MemoryRouter><RequestAccountPage /></MemoryRouter>);
    expect(await screen.findByText(/طلب حساب جديد غير متاح حاليًا/)).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });
});

describe("admin approval panel (Settings → Users)", () => {
  it("lists pending requests, approves one as a reviewer and rejects another, refreshing after each decision", async () => {
    const user = userEvent.setup();
    const first = { request_id: "r1", email: "a@example.invalid", display_name: "أول", created_at: "2026-10-16T09:00:00Z" };
    const second = { request_id: "r2", email: "b@example.invalid", display_name: "ثاني", created_at: "2026-10-16T09:05:00Z" };
    m.accountRequests.mockResolvedValueOnce({ requests: [first, second] }).mockResolvedValueOnce({ requests: [second] }).mockResolvedValueOnce({ requests: [] });
    m.approveAccountRequest.mockResolvedValue({ request_id: "r1", user_id: "u", email: "a@example.invalid", role: "REVIEWER" });
    m.rejectAccountRequest.mockResolvedValue({ request_id: "r2", status: "REJECTED" });
    render(<SettingsPage sort="oldest" onSort={() => {}} isAdmin />);
    const panel = async () => within(await screen.findByLabelText("طلبات الحسابات المعلّقة"));
    expect(await (await panel()).findByText(/أول —/)).toBeInTheDocument();
    expect(screen.getByText(/حساب «مراجع» فقط/)).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "موافقة" })[0]);
    await waitFor(() => expect(m.approveAccountRequest).toHaveBeenCalledWith("r1"));
    await waitFor(async () => expect((await panel()).queryByText(/أول —/)).not.toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "رفض" }));
    await waitFor(() => expect(m.rejectAccountRequest).toHaveBeenCalledWith("r2"));
    expect(await screen.findByText("لا توجد طلبات معلّقة.")).toBeInTheDocument();
  });

  it("shows a clear message when a request was already decided elsewhere", async () => {
    const user = userEvent.setup();
    m.accountRequests.mockResolvedValue({ requests: [{ request_id: "r1", email: "a@example.invalid", display_name: "أول", created_at: "2026-10-16T09:00:00Z" }] });
    m.approveAccountRequest.mockRejectedValue(Object.assign(new Error("x"), { status: 409, code: "already_decided" }));
    render(<SettingsPage sort="oldest" onSort={() => {}} isAdmin />);
    await user.click(await screen.findByRole("button", { name: "موافقة" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("تم البت في هذا الطلب مسبقًا.");
  });
});
