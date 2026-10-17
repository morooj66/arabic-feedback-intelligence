import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { api } from "../api/client";
import { EmptyState } from "./EmptyState";
import { SystemState, type SystemStateKind } from "./SystemState";
import { STATE_COPY } from "./stateCopy";

vi.mock("../api/client", () => ({ microsoftSignInUrl: "/api/auth/microsoft/start", api: { session: vi.fn(), health: vi.fn(), counts: vi.fn(), queue: vi.fn() } }));
beforeEach(() => { vi.resetAllMocks(); vi.mocked(api.health).mockResolvedValue({ ok: true, demo_auth: false, entra: false }); });

describe("System state semantics", () => {
  it.each<[SystemStateKind, string]>([["loading", "status"], ["empty", "status"], ["noResults", "status"], ["unavailable", "status"], ["success", "status"], ["error", "alert"], ["permission", "alert"]])("announces %s with the right urgency and RTL", (kind, role) => {
    render(<SystemState kind={kind} className="existing-class">رسالة الحالة</SystemState>);
    const state = screen.getByRole(role);
    expect(state).toHaveAttribute("dir", "rtl"); expect(state).toHaveAttribute("aria-atomic", "true");
    expect(state).toHaveAttribute("aria-live", role === "alert" ? "assertive" : "polite");
    expect(state).toHaveClass("existing-class"); expect(state).not.toHaveAttribute("style");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("preserves the existing empty-state DOM and uses only the supplied real action", async () => {
    const reset = vi.fn(); const user = userEvent.setup();
    render(<EmptyState kind="noResults" {...STATE_COPY.noResults} action={<button onClick={reset}>مسح الفلاتر</button>} />);
    expect(screen.getByRole("status")).toHaveClass("empty-state");
    expect(screen.getByText(STATE_COPY.noResults.title)).toHaveClass("empty-state-title");
    await user.tab(); expect(screen.getByRole("button")).toHaveFocus(); await user.keyboard("{Enter}"); expect(reset).toHaveBeenCalledOnce();
  });

  it("avoids a duplicate background announcement while the dialog announces the same error", () => {
    render(<><SystemState kind="error" announce={false}>{STATE_COPY.mutationFailed}</SystemState><SystemState as="p" kind="error">{STATE_COPY.mutationFailed}</SystemState></>);
    expect(screen.getAllByRole("alert")).toHaveLength(1); expect(screen.getByRole("alert").tagName).toBe("P");
  });

  it("renders supplied copy as text, not HTML", () => {
    const { container } = render(<SystemState kind="error">{"<script>unsafe()</script>"}</SystemState>);
    expect(container.querySelector("script")).toBeNull();
  });
});

describe("Existing authentication and empty semantics", () => {
  function boot() { render(<MemoryRouter initialEntries={["/review"]}><App /></MemoryRouter>); }
  it("sanitizes service-unavailable session restore without calling it bad credentials", async () => {
    vi.mocked(api.session).mockRejectedValue(Object.assign(new Error("secret database traceback"), { status: 503, code: "service_unavailable" }));
    boot(); expect(await screen.findByRole("alert")).toHaveTextContent(STATE_COPY.serviceUnavailable);
    expect(screen.queryByText(/secret|traceback|كلمة المرور غير صحيحة/)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "تسجيل الدخول" })).toBeInTheDocument();
    expect(api.queue).not.toHaveBeenCalled();
  });

  it("preserves the session-expired sign-in flow without inventing an expiry distinction", async () => {
    vi.mocked(api.session).mockRejectedValue(Object.assign(new Error("expired session secret"), { status: 401 }));
    boot(); await screen.findByRole("heading", { name: "تسجيل الدخول" });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText(/expired session/)).not.toBeInTheDocument(); expect(api.queue).not.toHaveBeenCalled();
  });

  it("distinguishes an empty assigned queue from service failure without a fake start action", async () => {
    vi.mocked(api.session).mockResolvedValue({ user_id: "self", display_name: "مراجع اختبار", role: "REVIEWER", auth_mode: "session" });
    vi.mocked(api.counts).mockResolvedValue({ all: 0, pending: 0, approved: 0, no_reply: 0, send_failed: 0, by_status: {} });
    vi.mocked(api.queue).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 8 });
    boot(); await screen.findByText(STATE_COPY.assignedEmpty.title);
    expect(screen.getByRole("status")).toHaveTextContent(STATE_COPY.assignedEmpty.description);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument(); expect(screen.queryByRole("button", { name: "بدء المراجعة" })).not.toBeInTheDocument();
  });
});
