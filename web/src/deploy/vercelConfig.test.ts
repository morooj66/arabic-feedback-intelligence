import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { API_BASE } from "../api/client";
import { buildProxyConfig, validateBackendOrigin } from "../../scripts/configure-same-origin-proxy.mjs";

const WEB = process.cwd();                                    // vitest runs from the web/ project root
const read = (file: string) => readFileSync(join(WEB, file), "utf8");
type Rewrite = { source: string; destination: string };

// Vercel's `source` uses path-to-regexp; the two forms used here convert directly to anchored JS regular expressions.
const toRegExp = (source: string) => new RegExp("^" + source.replace(":path*", "(.*)") + "$");
const match = (rewrites: Rewrite[], path: string) => rewrites.find((r) => toRegExp(r.source).test(path));

const SPA_PAGES = ["/", "/review", "/forgot-password", "/reset-password", "/reset-password/"];
const API_PATHS = ["/api/health", "/api/login", "/api/password-reset/request", "/api/excel/upload", "/api/excel/export", "/api/items/abc/send-as-is"];

describe("default vercel.json (frontend only)", () => {
  const cfg = JSON.parse(read("vercel.json"));
  it("serves the SPA shell for every client route, including emailed reset links", () => {
    for (const p of SPA_PAGES) expect(match(cfg.rewrites, p)?.destination).toBe("/index.html");
  });
  it("never answers an /api request with index.html", () => {
    for (const p of API_PATHS) expect(match(cfg.rewrites, p)).toBeUndefined();
  });
  it("contains no invented domain or backend address", () => {
    expect(JSON.stringify(cfg.rewrites)).not.toMatch(/https?:/);
    expect(read("vercel.json")).not.toMatch(/onrender|vercel\.app|localhost|127\.0\.0\.1/);
  });
  it("adds basic security headers without touching the visual design", () => {
    const headers = Object.fromEntries(cfg.headers[0].headers.map((h: { key: string; value: string }) => [h.key, h.value]));
    expect(headers).toMatchObject({ "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer" });
  });
  it("is a Vite static build with the same output directory the build writes", () => {
    expect(cfg).toMatchObject({ framework: "vite", buildCommand: "npm run build", outputDirectory: "dist" });
  });
});

describe("same-origin proxy configuration", () => {
  const template = read("vercel.same-origin.template.json");
  const generated = JSON.parse(buildProxyConfig(template, "https://rcs-api.gov.example-host.sa"));

  it("the template carries a placeholder, never a real host", () => {
    expect(template).toContain("__BACKEND_ORIGIN__");
    expect(template).not.toMatch(/onrender|vercel\.app|localhost/);
  });
  it("forwards every API route (including the Excel upload and export) to the backend with the path intact", () => {
    for (const p of API_PATHS) {
      const rule = match(generated.rewrites, p);
      expect(rule?.destination).toBe("https://rcs-api.gov.example-host.sa/api/:path*");
    }
  });
  it("evaluates the API rule before the SPA fallback, and the fallback still serves client routes", () => {
    expect(generated.rewrites[0].source).toBe("/api/:path*");
    for (const p of SPA_PAGES) expect(match(generated.rewrites, p)?.destination).toBe("/index.html");
  });
  it("keeps API calls relative, so the browser only ever talks to the frontend origin (cookie stays first-party)", () => {
    expect(API_BASE).toBe("");
  });
  it("preserves the default's headers and build settings", () => {
    expect(generated.headers).toEqual(JSON.parse(read("vercel.json")).headers);
    expect(generated).toMatchObject({ framework: "vite", outputDirectory: "dist" });
  });
});

describe("backend origin validation (no invented or unsafe domains)", () => {
  it("accepts a bare https origin and normalises it", () => {
    expect(validateBackendOrigin("https://api.rcs.rcs-demo.net/")).toBe("https://api.rcs.rcs-demo.net");
  });
  it.each([
    ["", "not a valid URL"], [undefined, "not a valid URL"], ["rcs-api.rcs-demo.net", "not a valid URL"],
    ["http://api.rcs.rcs-demo.net", "https"], ["https://user:pw@api.rcs.rcs-demo.net", "credentials"],
    ["https://api.rcs.rcs-demo.net/api", "bare origin"], ["https://api.rcs.rcs-demo.net?x=1", "bare origin"], ["https://api.rcs.rcs-demo.net/#f", "bare origin"],
    ["https://localhost", "loopback"], ["https://127.0.0.1", "loopback"], ["https://my_service.rcs-demo.net", "placeholder"],
    ["https://example.org", "placeholder"], ["https://your-backend.rcs-demo.net", "placeholder"], ["https://intranet", "placeholder"],
  ])("rejects %s", (value, why) => {
    expect(() => validateBackendOrigin(value)).toThrow(new RegExp(why));
  });
  it("refuses a template without the placeholder", () => {
    expect(() => buildProxyConfig('{"rewrites": []}', "https://api.rcs.rcs-demo.net")).toThrow(/placeholder/);
  });
});

describe("command line", () => {
  const script = join(WEB, "scripts/configure-same-origin-proxy.mjs");
  it("fails without writing anything when the origin is unsafe", () => {
    const before = read("vercel.json");
    const r = spawnSync("node", [script, "http://insecure.rcs-demo.net"], { encoding: "utf8" });
    expect(r.status).toBe(1);
    expect(r.stderr).toMatch(/https/);
    expect(read("vercel.json")).toBe(before);                                   // the committed default is untouched
  });
  it("is a local-only script (no network APIs)", () => {
    const src = read("scripts/configure-same-origin-proxy.mjs");
    expect(src).not.toMatch(/fetch\(|node:http|node:https|node:net|XMLHttpRequest/);
  });
});
