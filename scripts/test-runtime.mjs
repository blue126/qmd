#!/usr/bin/env node
import { spawnSync } from "node:child_process";

// All model regressions use in-process mocks or loopback HTTP servers.
// This command never downloads models or opts into external API tests.
const tests = [
  "test/remote-llm.test.ts",
  "test/remote-jev.test.ts",
  "test/remote-embedding.test.ts",
  "test/remote-embedding-chunking.test.ts",
  "test/openai-embedding-provider.test.ts",
  "test/voyage-embedding-provider.test.ts",
  "test/cli-openai-embed.test.ts",
  "test/remote-cli-mcp.test.ts",
];
const args = process.argv.slice(2);
if (args.some(arg => arg !== "--dry-run")) {
  console.error("Usage: node scripts/test-runtime.mjs [--dry-run]");
  process.exit(2);
}
const command = ["./node_modules/vitest/vitest.mjs", "run", "--reporter=verbose", "--testTimeout", "60000", ...tests];
if (args.includes("--dry-run")) {
  console.log([process.execPath, ...command].join(" "));
} else {
  const env = { ...process.env, CI: "true", QMD_MODEL_INTEGRATION: "0" };
  for (const name of ["OPENAI_API_KEY", "OPENAI_BASE_URL", "VOYAGE_API_KEY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "QMD_EMBED_MODEL", "QMD_GENERATE_MODEL", "QMD_RERANK_MODEL"]) delete env[name];
  const result = spawnSync(process.execPath, command, {
    stdio: "inherit",
    env,
  });
  if (result.error) console.error(result.error.message);
  process.exitCode = result.status ?? 1;
}
