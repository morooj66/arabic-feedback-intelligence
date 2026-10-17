import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

// jsdom has no layout engine, so the real 390px / 320px overflow check lives in the browser UAT script (scripts/e2e). This guard pins the CSS that fixes it:
// the 64-character comment id must be allowed to wrap, never to push the page wider than the viewport.
const css = (file: string) => readFileSync(new URL(file, import.meta.url), "utf8");

describe("long comment ids wrap instead of overflowing small screens", () => {
  it("reviewer list id", () => {
    const rule = css("./ReviewerAssignedPage.css").match(/\.assigned-id\s*\{[^}]*\}/)?.[0] || "";
    expect(rule).toMatch(/overflow-wrap:\s*anywhere/);
    expect(rule).toMatch(/word-break:\s*break-all/);
    expect(rule).toMatch(/max-width:\s*100%/);
    expect(css("./ReviewerAssignedPage.css")).toMatch(/\.assigned-row > \*[^{]*\{\s*min-width:\s*0/);
  });
  it("review detail heading id", () => {
    const rule = css("./ReviewModePage.css").match(/\.focus-heading bdi\s*\{[^}]*\}/)?.[0] || "";
    expect(rule).toMatch(/overflow-wrap:\s*anywhere/);
    expect(rule).toMatch(/word-break:\s*break-all/);
  });
  it("the id stays in the markup (nothing is hidden)", () => {
    expect(css("./ReviewerAssignedPage.tsx")).toContain("#{item.comment_id}");
    expect(css("./ReviewModePage.tsx")).toContain("#{d.comment_id}");
  });
});
