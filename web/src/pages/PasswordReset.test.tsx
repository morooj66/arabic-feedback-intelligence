import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { ForgotPasswordPage } from "./ForgotPasswordPage";
import { LoginPage } from "./LoginPage";
import { ResetPasswordPage } from "./ResetPasswordPage";

vi.mock("../api/client", () => ({
  microsoftSignInUrl: "/api/auth/microsoft/start",
  api: {
    health: vi.fn(),
    login: vi.fn(),
    requestPasswordReset: vi.fn(),
    confirmPasswordReset: vi.fn(),
  },
}));

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname + loc.hash}</output>;
}

function GoButton({ to, label }: { to: string; label: string }) {
  const navigate = useNavigate();
  return <button type="button" onClick={() => navigate(to)}>{label}</button>;
}

function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/" element={<><LoginPage /><Probe /></>} />
        <Route path="/forgot-password" element={<><ForgotPasswordPage /><Probe /></>} />
        <Route path="/reset-password" element={<><ResetPasswordPage /><Probe /></>} />
      </Routes>
    </MemoryRouter>,
  );
}

function apiError(code: string, status: number) {
  const e = new Error(code) as Error & { code?: string; status?: number };
  e.code = code;
  e.status = status;
  return e;
}

const TOKEN = "tok_ABCdef-123_456789012345678901234567890123";

beforeEach(() => {
  vi.mocked(api.health).mockReset().mockResolvedValue({ ok: true, demo_auth: false, entra: false, password_reset: true });
  vi.mocked(api.login).mockReset();
  vi.mocked(api.requestPasswordReset).mockReset().mockResolvedValue({ ok: true });
  vi.mocked(api.confirmPasswordReset).mockReset().mockResolvedValue({ ok: true });
});

describe("login page entry point", () => {
  it("opens the reset flow when the server has email reset configured", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await waitFor(() => expect(api.health).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });                       // let the health answer reach component state
    await user.click(await screen.findByRole("button", { name: "نسيت كلمة المرور؟" }));
    expect(await screen.findByRole("heading", { name: "إعادة تعيين كلمة المرور" })).toBeInTheDocument();
    expect(screen.getByTestId("loc")).toHaveTextContent("/forgot-password");
    expect(api.requestPasswordReset).not.toHaveBeenCalled();
  });

  it("keeps the truthful contact-admin message when email reset is not configured (fail closed)", async () => {
    vi.mocked(api.health).mockResolvedValue({ ok: true, demo_auth: false, entra: false });
    const user = userEvent.setup();
    renderApp("/");
    await waitFor(() => expect(api.health).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "نسيت كلمة المرور؟" }));
    expect(screen.getByText("تواصل مع مسؤول النظام لإعادة تعيين كلمة المرور.")).toBeInTheDocument();
    expect(screen.getByTestId("loc")).toHaveTextContent("/");
    expect(screen.queryByText(/تم إرسال/)).not.toBeInTheDocument();
  });
});

describe("request flow", () => {
  it("sends the trimmed email and shows the same neutral notice (no claim that anything was sent)", async () => {
    const user = userEvent.setup();
    renderApp("/forgot-password");
    await user.type(await screen.findByLabelText("البريد الإلكتروني"), "  alice@example.com ");
    await user.click(screen.getByRole("button", { name: "إرسال رابط إعادة التعيين" }));
    await waitFor(() => expect(api.requestPasswordReset).toHaveBeenCalledWith("alice@example.com"));
    expect(await screen.findByText(/إذا كان هذا البريد الإلكتروني مسجَّلًا لدينا/)).toBeInTheDocument();
    expect(screen.queryByText(/تم إرسال/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "إرسال رابط إعادة التعيين" })).not.toBeInTheDocument();
  });

  it("an unknown address gets the identical notice (the server answers identically)", async () => {
    const user = userEvent.setup();
    renderApp("/forgot-password");
    await user.type(await screen.findByLabelText("البريد الإلكتروني"), "nobody@example.com");
    await user.click(screen.getByRole("button", { name: "إرسال رابط إعادة التعيين" }));
    expect(await screen.findByText(/إذا كان هذا البريد الإلكتروني مسجَّلًا لدينا/)).toBeInTheDocument();
  });

  it("shows a rate-limit message on 429 and the unavailable message on 503", async () => {
    const user = userEvent.setup();
    vi.mocked(api.requestPasswordReset).mockRejectedValueOnce(apiError("too_many_attempts", 429));
    renderApp("/forgot-password");
    await user.type(await screen.findByLabelText("البريد الإلكتروني"), "a@b.com");
    await user.click(screen.getByRole("button", { name: "إرسال رابط إعادة التعيين" }));
    expect(await screen.findByText(/تم تجاوز عدد المحاولات/)).toBeInTheDocument();
    vi.mocked(api.requestPasswordReset).mockRejectedValueOnce(apiError("password_reset_unavailable", 503));
    await user.click(screen.getByRole("button", { name: "إرسال رابط إعادة التعيين" }));
    expect(await screen.findByText(/غير متاحة حاليًا/)).toBeInTheDocument();
  });

  it("when the server reports reset unavailable, shows the message and no form", async () => {
    vi.mocked(api.health).mockResolvedValue({ ok: true, demo_auth: false, entra: false, password_reset: false });
    renderApp("/forgot-password");
    expect(await screen.findByText(/غير متاحة حاليًا/)).toBeInTheDocument();
    expect(screen.queryByLabelText("البريد الإلكتروني")).not.toBeInTheDocument();
    expect(api.requestPasswordReset).not.toHaveBeenCalled();
  });
});

describe("confirmation flow", () => {
  it("reads the token from the URL fragment, removes it from the address, and never shows it", async () => {
    renderApp(`/reset-password#token=${TOKEN}`);
    await screen.findByLabelText("كلمة المرور الجديدة");
    await waitFor(() => expect(screen.getByTestId("loc")).toHaveTextContent(/^\/reset-password$/));
    expect(document.body.textContent).not.toContain(TOKEN);
  });

  it("validates locally (length, match) without calling the API", async () => {
    const user = userEvent.setup();
    renderApp(`/reset-password#token=${TOKEN}`);
    await user.type(await screen.findByLabelText("كلمة المرور الجديدة"), "short");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "short");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    expect(await screen.findByText("كلمة المرور يجب ألا تقل عن 8 أحرف.")).toBeInTheDocument();
    await user.clear(screen.getByLabelText("كلمة المرور الجديدة"));
    await user.clear(screen.getByLabelText("تأكيد كلمة المرور"));
    await user.type(screen.getByLabelText("كلمة المرور الجديدة"), "long-enough-1");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "long-enough-2");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    expect(await screen.findByText("كلمتا المرور غير متطابقتين.")).toBeInTheDocument();
    expect(api.confirmPasswordReset).not.toHaveBeenCalled();
  });

  it("submits the token and password, then offers the login screen", async () => {
    const user = userEvent.setup();
    renderApp(`/reset-password#token=${TOKEN}`);
    await user.type(await screen.findByLabelText("كلمة المرور الجديدة"), "brand-new-pass-1");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "brand-new-pass-1");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    await waitFor(() => expect(api.confirmPasswordReset).toHaveBeenCalledWith({ token: TOKEN, password: "brand-new-pass-1" }));
    expect(await screen.findByText(/تم تعيين كلمة المرور الجديدة/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("brand-new-pass-1");
    await user.click(screen.getByRole("button", { name: "الذهاب إلى تسجيل الدخول" }));
    expect(await screen.findByRole("heading", { name: "تسجيل الدخول" })).toBeInTheDocument();
  });

  it("an invalid or expired token shows one neutral message and offers a new request", async () => {
    const user = userEvent.setup();
    vi.mocked(api.confirmPasswordReset).mockRejectedValueOnce(apiError("invalid_or_expired_token", 400));
    renderApp(`/reset-password#token=${TOKEN}`);
    await user.type(await screen.findByLabelText("كلمة المرور الجديدة"), "brand-new-pass-1");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "brand-new-pass-1");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    expect(await screen.findByText(/غير صالح أو انتهت صلاحيته/)).toBeInTheDocument();
    expect(screen.queryByLabelText("كلمة المرور الجديدة")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "طلب رابط جديد" }));
    expect(await screen.findByRole("heading", { name: "إعادة تعيين كلمة المرور" })).toBeInTheDocument();
  });

  it("a link with no token shows the invalid-link message and no form", async () => {
    renderApp("/reset-password");
    expect(await screen.findByText(/غير صالح أو انتهت صلاحيته/)).toBeInTheDocument();
    expect(screen.queryByLabelText("كلمة المرور الجديدة")).not.toBeInTheDocument();
    expect(api.confirmPasswordReset).not.toHaveBeenCalled();
  });

  it("maps a server-side weak-password and rate-limit answers to Arabic messages", async () => {
    const user = userEvent.setup();
    vi.mocked(api.confirmPasswordReset).mockRejectedValueOnce(apiError("weak_password", 400)).mockRejectedValueOnce(apiError("too_many_attempts", 429));
    renderApp(`/reset-password#token=${TOKEN}`);
    await user.type(await screen.findByLabelText("كلمة المرور الجديدة"), "password-ok-1");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "password-ok-1");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    expect(await screen.findByText(/لا تستوفي الشروط/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    expect(await screen.findByText(/تم تجاوز عدد المحاولات/)).toBeInTheDocument();
  });
});


describe("opening another reset link in the same tab (only the URL fragment changes)", () => {
  function renderWithNavigator(path: string) {
    return render(
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/reset-password" element={<><ResetPasswordPage /><Probe /><GoButton to={`/reset-password#token=${TOKEN}`} label="open-link-A" /><GoButton to="/reset-password#token=SECOND_TOKEN_value_0123456789abcdefghij" label="open-link-B" /></>} />
        </Routes>
      </MemoryRouter>,
    );
  }

  it("recovers from the stale invalid-link state when a valid link arrives", async () => {
    const user = userEvent.setup();
    renderWithNavigator("/reset-password");
    expect(await screen.findByText(/غير صالح أو انتهت صلاحيته/)).toBeInTheDocument();
    expect(screen.queryByLabelText("كلمة المرور الجديدة")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "open-link-A" }));
    await user.type(await screen.findByLabelText("كلمة المرور الجديدة"), "brand-new-pass-1");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "brand-new-pass-1");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    await waitFor(() => expect(api.confirmPasswordReset).toHaveBeenCalledWith({ token: TOKEN, password: "brand-new-pass-1" }));
    await waitFor(() => expect(screen.getByTestId("loc")).toHaveTextContent(/^\/reset-password$/));      // the fragment is removed again
  });

  it("a newer link replaces the older token, clears typed passwords and the previous error", async () => {
    const user = userEvent.setup();
    vi.mocked(api.confirmPasswordReset).mockRejectedValueOnce(apiError("invalid_or_expired_token", 400));
    renderWithNavigator(`/reset-password#token=${TOKEN}`);
    await user.type(await screen.findByLabelText("كلمة المرور الجديدة"), "brand-new-pass-1");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "brand-new-pass-1");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    expect(await screen.findByText(/غير صالح أو انتهت صلاحيته/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "open-link-B" }));
    const field = await screen.findByLabelText("كلمة المرور الجديدة");
    expect(field).toHaveValue("");
    expect(screen.queryByText(/غير صالح أو انتهت صلاحيته/)).not.toBeInTheDocument();
    await user.type(field, "another-pass-22");
    await user.type(screen.getByLabelText("تأكيد كلمة المرور"), "another-pass-22");
    await user.click(screen.getByRole("button", { name: "حفظ كلمة المرور" }));
    await waitFor(() => expect(api.confirmPasswordReset).toHaveBeenLastCalledWith({ token: "SECOND_TOKEN_value_0123456789abcdefghij", password: "another-pass-22" }));
    expect(document.body.textContent).not.toContain("SECOND_TOKEN");
  });

  it("a fragment without a token is cleared and does not disturb the page", async () => {
    renderApp("/reset-password#something=else");
    expect(await screen.findByText(/غير صالح أو انتهت صلاحيته/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("loc")).toHaveTextContent(/^\/reset-password$/));
  });
});
