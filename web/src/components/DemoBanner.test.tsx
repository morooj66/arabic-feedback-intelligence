import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { DemoBanner } from "./DemoBanner";

vi.mock("../api/client", () => ({ api: { health: vi.fn() } }));
const health = api.health as unknown as ReturnType<typeof vi.fn>;
beforeEach(() => health.mockReset());

describe("demo notice", () => {
  it("shows the synthetic-data notice only when the server reports demo mode", async () => {
    health.mockResolvedValue({ ok: true, demo_auth: false, entra: false, demo: true });
    render(<DemoBanner />);
    expect(await screen.findByRole("note")).toHaveTextContent("بيانات اصطناعية فقط");
  });

  it("renders nothing on a normal deployment (demo false or absent)", async () => {
    health.mockResolvedValue({ ok: true, demo_auth: false, entra: false });
    const { container } = render(<DemoBanner />);
    await waitFor(() => expect(health).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
    health.mockResolvedValue({ ok: true, demo_auth: false, entra: false, demo: false });
    const second = render(<DemoBanner />);
    await waitFor(() => expect(health).toHaveBeenCalledTimes(2));
    expect(second.container).toBeEmptyDOMElement();
  });

  it("renders nothing when the health call fails (never shows a false notice)", async () => {
    health.mockRejectedValueOnce(new Error("down"));
    const { container } = render(<DemoBanner />);
    await waitFor(() => expect(health).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });
});
