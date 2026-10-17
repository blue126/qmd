import { describe, expect, test } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";

describe("LLM module loading", () => {
  test("the inference module has no native model dependency", () => {
    const source = readFileSync(join(process.cwd(), "src", "llm.ts"), "utf-8");

    expect(source).not.toMatch(/import\s+(?!type\b)[\s\S]*?from\s+["']node-llama-cpp["']/);
    expect(source).not.toMatch(/import\s*\(["']node-llama-cpp["']\)/);
  });

  test("importing the CLI for lightweight commands succeeds", async () => {
    const mod = await import("../src/cli/qmd.ts");
    expect(mod).toMatchObject({
      buildEditorUri: expect.any(Function),
      termLink: expect.any(Function),
    });
  });
});
