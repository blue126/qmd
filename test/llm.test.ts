import { describe, test, expect, vi } from "vitest";
import {
  withLLMSessionForLlm,
  waitForLLMSessionsToDrain,
  SessionReleasedError,
  formatDocForEmbedding,
  formatQueryForEmbedding,
  type LLM,
  type ILLMSession,
} from "../src/llm.js";
import { RemoteLLM } from "../src/remote-llm.js";
import { RemoteJev } from "../src/remote-jev.js";
import { createConfiguredLLM } from "../src/hybrid.js";

function runtime(overrides: Partial<LLM> = {}): LLM {
  return {
    embed: async () => null,
    generate: async () => null,
    modelExists: async (name) => ({ name, exists: false }),
    expandQuery: async () => [],
    rerank: async () => ({ results: [], model: "test" }),
    dispose: async () => {},
    ...overrides,
  };
}

describe("remote model sessions", () => {
  test("missing remote configuration permits construction and rejects model operations", async () => {
    const llm = createConfiguredLLM();
    expect(llm.supportsExpand).toBe(false);
    expect(llm.supportsRerank).toBe(false);
    await expect(llm.embed("text")).rejects.toThrow("configured remote");
    await expect(llm.generate("text")).rejects.toThrow("configured remote");
    await expect(llm.expandQuery("query")).rejects.toThrow("configured remote");
    await expect(llm.rerank("query", [])).rejects.toThrow("configured remote");
    await expect(
      llm.modelExists("hf:legacy/model.gguf"),
    ).resolves.toMatchObject({ exists: false });
    await llm.dispose();
  });
  test("release rejects subsequent operations and removes the external abort listener", async () => {
    const controller = new AbortController();
    const remove = vi.spyOn(controller.signal, "removeEventListener");
    let captured!: ILLMSession;
    await withLLMSessionForLlm(
      runtime(),
      async (session) => {
        captured = session;
      },
      { signal: controller.signal },
    );
    expect(captured.signal.aborted).toBe(true);
    expect(remove).toHaveBeenCalledWith("abort", expect.any(Function));
    await expect(captured.expandQuery("late")).rejects.toBeInstanceOf(
      SessionReleasedError,
    );
  });
  test("abort reaches the actual remote operation and drain waits for its settlement", async () => {
    const controller = new AbortController();
    let settle!: () => void;
    let operation!: Promise<unknown>;
    let observedSignal!: AbortSignal;
    const llm = runtime({
      expandQuery: async (_query, options) => {
        observedSignal = options!.signal!;
        await new Promise<void>((resolve) => {
          settle = resolve;
        });
        return [];
      },
    });
    await withLLMSessionForLlm(
      llm,
      async (session) => {
        operation = session.expandQuery("pending").catch((error) => error);
        controller.abort();
        expect(observedSignal.aborted).toBe(true);
      },
      { signal: controller.signal },
    );
    let drained = false;
    const drain = waitForLLMSessionsToDrain(llm).then(() => {
      drained = true;
    });
    await Promise.resolve();
    expect(drained).toBe(false);
    settle();
    expect(await operation).toBeInstanceOf(SessionReleasedError);
    await drain;
    expect(drained).toBe(true);
  });
  test.each(["embed", "embedBatch", "expandQuery", "rerank"] as const)(
    "%s honors operation cancellation as well as session cancellation",
    async (method) => {
      for (const source of ["operation", "session"] as const) {
        const operationController = new AbortController();
        const sessionController = new AbortController();
        let observedSignal!: AbortSignal;
        const wait = (signal: AbortSignal) => {
          observedSignal = signal;
          return new Promise<never>((_resolve, reject) => {
            signal.addEventListener("abort", () => reject(signal.reason), { once: true });
          });
        };
        const llm = runtime({
          embed: (_text, options) => wait(options!.signal!),
          expandQuery: (_query, options) => wait(options!.signal!),
          rerank: (_query, _documents, options) => wait(options!.signal!),
        });
        await withLLMSessionForLlm(llm, async (session) => {
          const options = { signal: operationController.signal };
          const invoke = () => {
            switch (method) {
              case "embed": return session.embed("text", options);
              case "embedBatch": return session.embedBatch(["text"], options);
              case "expandQuery": return session.expandQuery("query", options);
              case "rerank": return session.rerank("query", [], options);
            }
          };
          const result = invoke().catch(error => error);
          const reason = new Error(`${source} cancelled`);
          (source === "operation" ? operationController : sessionController).abort(reason);
          expect(observedSignal.aborted).toBe(true);
          expect(await result).toBe(reason);
          expect(session.isValid).toBe(source === "operation");
        }, { signal: sessionController.signal });
      }
    },
  );
  test("pre-aborted operation signals never admit model work", async () => {
    const signal = AbortSignal.abort(new Error("already cancelled"));
    const embed = vi.fn(async () => null);
    const expandQuery = vi.fn(async () => []);
    const rerank = vi.fn(async () => ({ results: [], model: "test" }));
    await withLLMSessionForLlm(runtime({ embed, expandQuery, rerank }), async (session) => {
      for (const operation of [
        session.embed("text", { signal }),
        session.embedBatch(["text"], { signal }),
        session.expandQuery("query", { signal }),
        session.rerank("query", [], { signal }),
      ]) await expect(operation).rejects.toThrow("already cancelled");
      expect(session.isValid).toBe(true);
    });
    expect(embed).not.toHaveBeenCalled();
    expect(expandQuery).not.toHaveBeenCalled();
    expect(rerank).not.toHaveBeenCalled();
  });
  test("disposing a remote runtime aborts and drains a pending response body", async () => {
    let bodyStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      bodyStarted = resolve;
    });
    let observedSignal!: AbortSignal;
    const fetch: typeof globalThis.fetch = vi.fn(async (_input, init) => {
      observedSignal = init!.signal!;
      return {
        ok: true,
        json: () =>
          new Promise((_resolve, reject) => {
            bodyStarted();
            observedSignal.addEventListener(
              "abort",
              () => reject(observedSignal.reason),
              { once: true },
            );
          }),
      } as Response;
    });
    const remote = new RemoteLLM({
      generateApiUrl: "https://models.test/v1",
      generateApiModel: "remote",
      fetch,
    });
    const operation = remote.expandQuery("query").catch((error) => error);
    await started;
    await remote.dispose();
    expect(observedSignal.aborted).toBe(true);
    expect(await operation).toBeInstanceOf(Error);
    expect(remote.supportsExpand).toBe(false);
    await expect(remote.expandQuery("late")).rejects.toThrow("disposed");
    expect(fetch).toHaveBeenCalledOnce();
  });
  test("caller cancellation does not poison the configured remote provider", async () => {
    const controller = new AbortController();
    let requested!: () => void;
    const started = new Promise<void>((resolve) => {
      requested = resolve;
    });
    const fetch: typeof globalThis.fetch = vi.fn(async (_input, init) => {
      requested();
      await new Promise<void>((_resolve, reject) =>
        init!.signal!.addEventListener(
          "abort",
          () => reject(init!.signal!.reason),
          { once: true },
        ),
      );
      return new Response();
    });
    const remote = new RemoteLLM({
      generateApiUrl: "https://models.test/v1",
      generateApiModel: "remote",
      fetch,
    });
    const operation = remote
      .expandQuery("query", { signal: controller.signal })
      .catch((error) => error);
    await started;
    controller.abort(new Error("cancel request"));
    expect(await operation).toBeInstanceOf(Error);
    expect(remote.supportsExpand).toBe(true);
    await remote.dispose();
  });
  test("Jev receives SDK request cancellation and disposal waits for its complete work", async () => {
    let finish!: () => void;
    let observedSignal!: AbortSignal;
    const client = {
      systemOne: vi.fn(async (_request, options) => {
        observedSignal = options.signal;
        await new Promise<void>((resolve) => {
          finish = resolve;
        });
        return {
          answers: {
            strategy: { choice: "code_search", confidence: 1 },
            needs_hyde: { noul: 0 },
          },
        };
      }),
    };
    const jev = new RemoteJev({ client });
    const operation = jev.classifyIntent("query").catch((error) => error);
    let disposed = false;
    const dispose = jev.dispose().then(() => {
      disposed = true;
    });
    await Promise.resolve();
    expect(observedSignal.aborted).toBe(true);
    expect(disposed).toBe(false);
    finish();
    expect(await operation).toBeInstanceOf(Error);
    await dispose;
    expect(disposed).toBe(true);
    await expect(jev.classifyIntent("late")).rejects.toThrow("disposed");
  });
  test("sessions from separate stores drain independently", async () => {
    const first = runtime(),
      second = runtime();
    let release!: () => void;
    const session = withLLMSessionForLlm(
      first,
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    await waitForLLMSessionsToDrain(second);
    release();
    await session;
    await waitForLLMSessionsToDrain(first);
  });
  test("retains pure legacy formatter behavior for identity diagnostics", () => {
    expect(formatQueryForEmbedding("q", "qwen-embedding")).toContain(
      "Query: q",
    );
    expect(formatDocForEmbedding("body", "title", "qwen-embedding")).toBe(
      "title\nbody",
    );
    expect(formatDocForEmbedding("body", "title", "legacy")).toBe(
      "title: title | text: body",
    );
  });
});
