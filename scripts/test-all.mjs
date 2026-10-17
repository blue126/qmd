#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

const cleanEnv = { ...process.env };
for (const name of [
  "OPENAI_BASE_URL", "OPENAI_API_KEY", "VOYAGE_API_KEY", "TYPESAFE_API_KEY",
  "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "QMD_EMBED_MODEL",
  "QMD_GENERATE_MODEL", "QMD_RERANK_MODEL",
]) delete cleanEnv[name];

function run(label, command, args, options = {}) {
  console.log(`==> ${label}`);
  const { env: extraEnv, ...spawnOptions } = options;
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    shell: process.platform === "win32",
    env: { ...cleanEnv, ...(extraEnv ?? {}) },
    ...spawnOptions,
  });
  if (result.status !== 0) {
    console.error(`Test task failed: ${label}`);
    process.exit(result.status ?? 1);
  }
}

run("TypeScript build typecheck", process.execPath, [join(root, "node_modules", "typescript", "bin", "tsc"), "-p", "tsconfig.build.json", "--noEmit"]);
run("Vitest suite under Node", process.execPath, [join(root, "node_modules", "vitest", "vitest.mjs"), "run", "--reporter=verbose", "--testTimeout", "60000", "test/"], { env: { CI: "true" } });
run("Package smoke", process.execPath, ["scripts/package-smoke.mjs"]);
