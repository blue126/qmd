import { describe, expect, test } from "vitest";
import { createCliResourceCloser, finishSuccessfulCliCommand } from "../src/cli/qmd.ts";

describe("CLI successful-exit lifecycle", () => {
  test("shares one provider-to-database cleanup across concurrent callers", async () => {
    const calls: string[] = [];
    let releaseOwner!: () => void;
    const ownerSettled = new Promise<void>(resolve => {
      releaseOwner = resolve;
    });
    const owner = {
      provider: {} as never,
      close: async () => {
        calls.push("owner:start");
        await ownerSettled;
        calls.push("owner:end");
      },
    };
    const cleanup = createCliResourceCloser({
      getOwner: () => owner,
      clearOwner: () => calls.push("owner:clear"),
      disposeFallback: async () => { calls.push("fallback"); },
      closeStore: () => calls.push("store"),
    });

    const first = cleanup();
    const second = cleanup();
    expect(second).toBe(first);
    await Promise.resolve();
    expect(calls).toEqual(["owner:start"]);

    releaseOwner();
    await Promise.all([first, second]);
    expect(calls).toEqual(["owner:start", "owner:end", "owner:clear", "store"]);
  });

  test("exits 0 after successful output when post-output LLM cleanup fails", async () => {
    const exitCodes: number[] = [];
    const stderr: string[] = [];
    const flushed: string[] = [];

    await finishSuccessfulCliCommand({
      command: "query",
      format: "json",
      cleanup: async () => {
        throw new Error("remote client cleanup failure");
      },
      exit: (code) => {
        exitCodes.push(code);
      },
      stdout: { write: (chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { flushed.push(String(chunk)); cb?.(); return true; } },
      stderr: { write: (chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { stderr.push(String(chunk)); cb?.(); return true; } },
    });

    expect(exitCodes).toEqual([0]);
    expect(stderr.join("")).toContain("QMD Warning: cleanup after successful output failed");
    expect(flushed).toEqual([""]);
  });

  test("flushes stdout, runs cleanup, flushes stderr, then exits (when exit is provided)", async () => {
    // The legacy lifecycle order is preserved for callers that pass an
    // explicit `exit` function — primarily this test, which needs an
    // observable terminating step.
    const calls: string[] = [];

    await finishSuccessfulCliCommand({
      command: "query",
      format: "json",
      cleanup: async () => { calls.push("cleanup"); },
      exit: (code) => { calls.push(`exit:${code}`); },
      stdout: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stdout-flush"); cb?.(); return true; } },
      stderr: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stderr-flush"); cb?.(); return true; } },
    });

    expect(calls).toEqual(["stdout-flush", "cleanup", "stderr-flush", "exit:0"]);
  });

  test("production path: sets process.exitCode=0 and returns instead of calling process.exit", async () => {
    // Let pending output and remote-client cleanup settle before natural exit.
    const prevCode = process.exitCode;
    process.exitCode = 1; // poison the state to verify we set it
    try {
      const calls: string[] = [];
      await finishSuccessfulCliCommand({
        command: "query",
        format: "json",
        cleanup: async () => { calls.push("cleanup"); },
        stdout: { write: (_c: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stdout-flush"); cb?.(); return true; } },
        stderr: { write: (_c: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stderr-flush"); cb?.(); return true; } },
      });

      expect(calls).toEqual(["stdout-flush", "cleanup", "stderr-flush"]);
      expect(process.exitCode).toBe(0);
    } finally {
      process.exitCode = prevCode;
    }
  });

});
