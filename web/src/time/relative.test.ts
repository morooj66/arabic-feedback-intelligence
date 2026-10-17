import { describe, expect, it } from "vitest";
import { formatRelativeAr, isOverdue } from "./relative";

const NOW = new Date("2026-08-25T18:00:00Z").getTime();

describe("formatRelativeAr", () => {
  it("formats minutes, hours, days, and weeks", () => {
    expect(formatRelativeAr("2026-08-25T17:45:00Z", NOW)).toBe("منذ 15 دقيقة");
    expect(formatRelativeAr("2026-08-25T16:00:00Z", NOW)).toBe("منذ ساعتين");
    expect(formatRelativeAr("2026-08-24T18:00:00Z", NOW)).toBe("منذ يوم");
    expect(formatRelativeAr("2026-08-18T18:00:00Z", NOW)).toBe("منذ أسبوع");
  });

  it("marks week-old items as overdue", () => {
    expect(isOverdue("2026-08-10T18:00:00Z", NOW)).toBe(true);
    expect(isOverdue("2026-08-25T17:00:00Z", NOW)).toBe(false);
  });
});
