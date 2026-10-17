// Minimal typings for the few Node built-ins used by test files that read project files
// (vercelConfig.test.ts, mobileOverflowCss.test.ts). Replace with `@types/node` once it can be installed.
declare module "node:fs" {
  export function readFileSync(path: string | URL, encoding: "utf8"): string;
}
declare module "node:path" {
  export function join(...parts: string[]): string;
}
declare module "node:child_process" {
  export function spawnSync(
    command: string,
    args: string[],
    options: { encoding: "utf8" },
  ): { status: number | null; stdout: string; stderr: string };
}
declare const process: { cwd(): string };
