#!/usr/bin/env node
// Writes web/vercel.json for the SAME-ORIGIN deployment: the browser talks only to the Vercel domain, and Vercel forwards /api/* to the backend. The session cookie is
// then first-party (no third-party-cookie blocking, no CORS needed) and VITE_API_BASE_URL stays unset.
//
//   node scripts/configure-same-origin-proxy.mjs https://<backend-host>
//
// It never invents a domain: the backend origin must be supplied, must be https, and must be a bare origin (no path, query, credentials, loopback or placeholder host).
// Local file operation only - nothing is contacted.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PLACEHOLDER = "__BACKEND_ORIGIN__";

export function validateBackendOrigin(raw) {
  let url;
  try {
    url = new URL(String(raw ?? "").trim());
  } catch {
    throw new Error("backend origin is not a valid URL");
  }
  if (url.protocol !== "https:") throw new Error("backend origin must use https://");
  if (url.username || url.password) throw new Error("backend origin must not contain credentials");
  if (url.pathname !== "/" || url.search || url.hash) throw new Error("backend origin must be a bare origin (no path, query or fragment)");
  const host = url.hostname.toLowerCase();
  if (host === "localhost" || host.endsWith(".localhost") || /^127\./.test(host) || host === "[::1]" || host === "0.0.0.0") throw new Error("backend origin must be a public host, not loopback");
  if (!host.includes(".") || host.includes("_") || /example\.(com|org|net)$/.test(host) || host.includes("your-") || host.includes("<")) throw new Error("backend origin looks like a placeholder; supply the real backend host");
  return url.origin;
}

export function buildProxyConfig(templateText, backendOrigin) {
  const origin = validateBackendOrigin(backendOrigin);
  if (!templateText.includes(PLACEHOLDER)) throw new Error("template has no backend placeholder");
  return templateText.replaceAll(PLACEHOLDER, origin);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const template = readFileSync(join(HERE, "..", "vercel.same-origin.template.json"), "utf8");
    const out = buildProxyConfig(template, process.argv[2]);
    JSON.parse(out);
    writeFileSync(join(HERE, "..", "vercel.json"), out);
    console.log(`web/vercel.json now proxies /api/* to ${validateBackendOrigin(process.argv[2])}`);
  } catch (err) {
    console.error(`configure-same-origin-proxy: ${err.message}`);
    process.exit(1);
  }
}
