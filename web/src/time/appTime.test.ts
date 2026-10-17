import { describe, expect, it } from "vitest";
import { firstName, formatTrendBucket, greetingPrefix, hourInAppTz, parseAppInstant } from "./appTime";
import { formatAbsoluteAr } from "./relative";

describe("app time", () => {
  it("uses morning and evening greetings from Asia/Riyadh", () => {
    expect(greetingPrefix(new Date("2026-08-26T08:00:00+03:00"))).toBe("صباح الخير");
    expect(greetingPrefix(new Date("2026-08-26T21:00:00+03:00"))).toBe("مساء الخير");
    expect(hourInAppTz(new Date("2026-08-26T08:00:00+03:00"))).toBe(8);
    expect(firstName("مروج الراجحي")).toBe("مروج");
  });

  it("formats timestamps in Asia/Riyadh without a manual offset", () => {
    const text = formatAbsoluteAr("2026-08-26T10:00:00+03:00");
    expect(text).toMatch(/26|٢٦/);
    expect(text).toMatch(/10|١٠/);
    expect(parseAppInstant("2026-08-26T10:00:00").toISOString()).toBe(parseAppInstant("2026-08-26T10:00:00+03:00").toISOString());
  });

  it("formats trend buckets for tooltips", () => {
    expect(formatTrendBucket("2026-08-26 10")).toMatch(/26|٢٦/);
    expect(formatTrendBucket("2026-08-26")).toMatch(/26|٢٦/);
  });
});
