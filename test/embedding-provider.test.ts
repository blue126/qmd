import { describe, expect, test, vi } from "vitest";
import { withLLMSessionForLlm, type EmbeddingResult, type ILLMSession } from "../src/llm.js";
import {
  EmbeddingProviderError,
  type EmbeddingProvider,
  type EmbeddingVector,
} from "../src/embedding/provider.js";
import {
  UnavailableLocalEmbeddingProvider,
} from "../src/embedding/local.js";
import { CompositeEmbeddingProviderOwner } from "../src/embedding/owner.js";
import { createCliEmbeddingProviderOwner } from "../src/cli/embedding-owner.js";
import {
  EmbeddingConfigError,
  OPENAI_EMBEDDING_DIMENSION,
  OPENAI_EMBEDDING_MODEL,
  readCanonicalEmbeddingConfig,
  resolveEmbeddingConfig,
  resolveEmbeddingModelOverride,
  writeCanonicalEmbeddingConfig,
} from "../src/embedding/config.js";

const DOCUMENT_OPTIONS = { purpose: "index-build", kind: "document", identityFingerprint: "test-fingerprint" } as const;
const QUERY_OPTIONS = { purpose: "query-embedding", kind: "query", identityFingerprint: "test-fingerprint" } as const;
import {
  createStore,
  generateEmbeddings,
  hybridQuery,
  resetInflightState,
  structuredSearch,
} from "../src/store.js";

function createSession(overrides: Partial<ILLMSession> = {}): ILLMSession {
  const controller = new AbortController();
  const embeddingModel = overrides.embeddingModel ?? "local-model";
  return {
    embeddingModel,
    embed: vi.fn(async (_text: string, options?: { model?: string }) => ({
      embedding: [0.1, 0.2, 0.3],
      model: embeddingModel,
    })),
    embedBatch: vi.fn(async (texts: string[], options?: { model?: string }) => texts.map((_, index) => ({
      embedding: [index, index + 0.5, index + 1],
      model: embeddingModel,
    }))),
    expandQuery: vi.fn(async () => []),
    rerank: vi.fn(async () => ({ results: [], model: "unused" })),
    isValid: true,
    signal: controller.signal,
    ...overrides,
  };
}

function expectProviderError(error: unknown, code: EmbeddingProviderError["code"]): void {
  expect(error).toBeInstanceOf(EmbeddingProviderError);
  expect(error).toMatchObject({ code });
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(resolvePromise => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("retired local embedding configuration", () => {
  test("preserves diagnostic identity and rejects all model execution", async () => {
    const provider = new UnavailableLocalEmbeddingProvider({model:"legacy-model",dimension:3});
    expect(JSON.parse(provider.canonicalIdentityMaterial())).toMatchObject({model:"legacy-model",dimension:3,remote:false});
    await expect(provider.embed("query", QUERY_OPTIONS)).rejects.toThrow("Local inference is no longer supported");
    await expect(provider.embedBatch(["doc"], DOCUMENT_OPTIONS)).rejects.toThrow("Local inference is no longer supported");
    expect(() => new UnavailableLocalEmbeddingProvider({model:"legacy-model"}).canonicalIdentityMaterial()).toThrow("persisted vector dimension");
    await provider.close();
  });
});

describe("embedding config resolver", () => {
  const defaultLocalModel = "hf:default/embed.gguf";

  test("uses new embedding config before legacy, DB, and defaults", () => {
    const resolved = resolveEmbeddingConfig({
      config: {
        collections: {},
        embedding: { provider: "local", model: "hf:explicit/embed.gguf", dimension: 768 },
        models: { embed: "hf:legacy/embed.gguf" },
      },
      dbConfig: { provider: "local", model: "hf:db/embed.gguf", dimension: 512 },
      defaultLocalModel,
      env: {},
    });

    expect(resolved).toEqual({
      canonical: { provider: "local", model: "hf:explicit/embed.gguf", dimension: 768 },
      source: "embedding-block",
      credentialAvailable: true,
      remoteRequestsEnabled: false,
    });
    expect(Object.isFrozen(resolved)).toBe(true);
    expect(Object.isFrozen(resolved.canonical)).toBe(true);
  });

  test("maps legacy models.embed to local, then falls back to DB and local defaults", () => {
    expect(resolveEmbeddingConfig({
      config: { collections: {}, models: { embed: "hf:legacy/embed.gguf" } },
      dbConfig: { provider: "openai", model: "text-embedding-3-small", dimension: 1536 },
      defaultLocalModel,
      env: {},
    })).toMatchObject({
      canonical: { provider: "local", model: "hf:legacy/embed.gguf", dimension: null },
      source: "legacy-models",
    });

    expect(resolveEmbeddingConfig({
      config: { collections: {} },
      dbConfig: { provider: "openai", model: "text-embedding-3-small", dimension: 1536 },
      defaultLocalModel,
      env: {},
    })).toMatchObject({
      canonical: { provider: "openai", model: "text-embedding-3-small", dimension: 1536 },
      source: "database",
      credentialAvailable: false,
      remoteRequestsEnabled: false,
    });

    expect(resolveEmbeddingConfig({
      config: { collections: {} },
      defaultLocalModel,
      env: {},
    })).toMatchObject({
      canonical: { provider: "local", model: defaultLocalModel, dimension: null },
      source: "local-default",
    });
  });

  test("fails closed on an invalid higher-precedence source instead of falling back", () => {
    expect(() => resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed: "hf:legacy/embed.gguf",
          embed_api_url: "https://api.openai.com/v1",
          embed_api_model: "text-embedding-ada-002",
          embed_dimension: 1536,
        },
      },
      dbConfig: { provider: "local", model: "hf:db/embed.gguf", dimension: 768 },
      defaultLocalModel,
      env: {},
    })).toThrowError(EmbeddingConfigError);

    expect(() => resolveEmbeddingConfig({
      config: { collections: {}, models: { embed: "" } },
      dbConfig: { provider: "local", model: "hf:db/embed.gguf", dimension: 768 },
      defaultLocalModel,
      env: {},
    })).toThrowError(/models\.embed/);
  });

  test("accepts supported OpenAI models with their native dimensions without requiring the key at resolve time", () => {
    const resolved = resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.openai.com/v1",
          embed_api_model: "text-embedding-3-small",
        },
      },
      defaultLocalModel,
      env: {},
    });
    expect(resolved).toMatchObject({
      canonical: { provider: "openai", model: "text-embedding-3-small", dimension: 1536 },
      credentialAvailable: false,
      remoteRequestsEnabled: false,
    });
    expect(resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.openai.com/v1",
          embed_api_model: "text-embedding-3-small",
        },
      },
      defaultLocalModel,
      env: { OPENAI_API_KEY: "[REDACTED]" },
    }).remoteRequestsEnabled).toBe(true);

    expect(resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.openai.com/v1",
          embed_api_model: "text-embedding-3-large",
        },
      },
      defaultLocalModel,
      env: { OPENAI_API_KEY: "[REDACTED]" },
    }).canonical).toEqual({
      provider: "openai",
      model: "text-embedding-3-large",
      dimension: 3072,
      baseUrl: "https://api.openai.com/v1",
    });

    for (const config of [
      { collections: {}, models: { embed_api_url: "https://api.openai.com/v1", embed_api_model: "unknown-model" } },
      { collections: {}, models: { embed_api_url: "https://api.openai.com/v1", embed_api_model: "text-embedding-3-small", embed_dimension: 768 } },
      { collections: {}, embedding: { provider: "remote" } },
    ]) {
      expect(() => resolveEmbeddingConfig({
        config,
        defaultLocalModel,
        env: { OPENAI_API_KEY: "[REDACTED]" },
      })).toThrowError(EmbeddingConfigError);
    }

    expect(resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.voyageai.com/v1",
          embed_api_model: "voyage-4",
        },
      },
      defaultLocalModel,
      env: { OPENAI_API_KEY: "[REDACTED]" },
    }).canonical).toEqual({
      provider: "voyageai",
      model: "voyage-4",
      dimension: 1024,
      baseUrl: "https://api.voyageai.com/v1",
    });

    expect(resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.voyageai.com/v1",
          embed_api_model: "voyage-4",
          embed_dimension: 512,
        },
      },
      defaultLocalModel,
      env: { VOYAGE_API_KEY: "vy-secret" },
    }).canonical).toEqual({
      provider: "voyageai",
      model: "voyage-4",
      dimension: 512,
      baseUrl: "https://api.voyageai.com/v1",
    });

    expect(resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.voyageai.com/v1",
          embed_api_model: "voyage-4",
        },
      },
      defaultLocalModel,
      env: { VOYAGE_API_KEY: "vy-secret" },
    }).credentialAvailable).toBe(true);

    expect(resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "http://127.0.0.1:8080/v1",
          embed_api_model: "custom-bge-large",
          embed_dimension: 768,
        },
      },
      defaultLocalModel,
      env: {},
    }).canonical).toEqual({
      provider: "openai",
      model: "custom-bge-large",
      dimension: 768,
      baseUrl: "http://127.0.0.1:8080/v1",
    });
  });

  test("creates isolated immutable results for concurrent SDK configs", () => {
    const first = resolveEmbeddingConfig({
      config: { collections: {}, embedding: { provider: "local", model: "hf:first/embed.gguf" } },
      defaultLocalModel,
    });
    const second = resolveEmbeddingConfig({
      config: { collections: {}, embedding: { provider: "local", model: "hf:second/embed.gguf" } },
      defaultLocalModel,
    });

    expect(first.canonical.model).toBe("hf:first/embed.gguf");
    expect(second.canonical.model).toBe("hf:second/embed.gguf");
    expect(first.canonical).not.toBe(second.canonical);
  });

  test("preserves local model overrides and requires the configured OpenAI model", () => {
    const local = resolveEmbeddingConfig({
      config: { collections: {}, embedding: { provider: "local", model: "hf:configured/embed.gguf" } },
      defaultLocalModel,
    });
    const openai = resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.openai.com/v1",
          embed_api_model: "text-embedding-3-small",
        },
      },
      defaultLocalModel,
    });

    expect(resolveEmbeddingModelOverride(local, "hf:override/embed.gguf")).toBe("hf:override/embed.gguf");
    expect(resolveEmbeddingModelOverride(openai, "text-embedding-3-small")).toBe("text-embedding-3-small");
    expect(() => resolveEmbeddingModelOverride(openai, "text-embedding-3-large")).toThrowError(EmbeddingConfigError);

    const largeOpenAI = resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.openai.com/v1",
          embed_api_model: "text-embedding-3-large",
        },
      },
      defaultLocalModel,
    });
    expect(resolveEmbeddingModelOverride(largeOpenAI, "text-embedding-3-large"))
      .toBe("text-embedding-3-large");
    expect(() => resolveEmbeddingModelOverride(largeOpenAI, "text-embedding-3-small"))
      .toThrowError(EmbeddingConfigError);
  });

  test("persists only canonical non-secret configuration in SQLite", () => {
    const store = createStore(":memory:");
    const resolved = resolveEmbeddingConfig({
      config: {
        collections: {},
        models: {
          embed_api_url: "https://api.openai.com/v1",
          embed_api_model: "text-embedding-3-small",
        },
      },
      defaultLocalModel,
      env: { OPENAI_API_KEY: "[REDACTED]" },
    });

    try {
      writeCanonicalEmbeddingConfig(store.db, resolved.canonical);
      expect(readCanonicalEmbeddingConfig(store.db)).toEqual(resolved.canonical);
      const row = store.db.prepare(
        "SELECT value FROM store_config WHERE key = 'embedding_config'",
      ).get() as { value: string };
      expect(row.value).toBe(JSON.stringify(resolved.canonical));
      expect(row.value).not.toContain("OPENAI_API_KEY");
      expect(row.value).not.toContain("credentialAvailable");
      expect(row.value).not.toContain("source");
    } finally {
      store.close();
    }
  });
});

describe("EmbeddingProviderOwner lifecycle", () => {
  test("drains active remote sessions before disposing the OpenAI CLI runtime", async () => {
    const sessionStarted = deferred<void>();
    const sessionSettled = deferred<void>();
    const events: string[] = [];
    const runtime = {
      dispose: vi.fn(async () => { events.push("runtime:dispose"); }),
    } as any;
    const remoteProvider: EmbeddingProvider = {
      providerId: "openai",
      model: OPENAI_EMBEDDING_MODEL,
      dimension: OPENAI_EMBEDDING_DIMENSION,
      remote: true,
      canonicalIdentityMaterial: () => "openai-test",
      formatQuery: text => text,
      formatDocument: text => text,
      embed: vi.fn(async (): Promise<EmbeddingVector> => ({
        vector: [1],
        model: OPENAI_EMBEDDING_MODEL,
        dimension: 1,
      })),
      embedBatch: vi.fn(async (): Promise<EmbeddingVector[]> => []),
      close: vi.fn(async () => { events.push("provider:close"); }),
    };
    const owner = createCliEmbeddingProviderOwner({
      provider: "openai",
      model: OPENAI_EMBEDDING_MODEL,
      dimension: OPENAI_EMBEDDING_DIMENSION,
    }, runtime, remoteProvider);
    const activeSession = withLLMSessionForLlm(runtime, async () => {
      sessionStarted.resolve();
      await sessionSettled.promise;
    });
    await sessionStarted.promise;

    const close = owner.close();
    expect(remoteProvider.close).toHaveBeenCalledTimes(1);
    expect(runtime.dispose).not.toHaveBeenCalled();

    sessionSettled.resolve();
    await activeSession;
    await close;
    expect(events).toEqual(["provider:close", "runtime:dispose"]);
  });

  test("CLI rejects creation of a local execution owner", () => {
    expect(() => createCliEmbeddingProviderOwner({provider:"local", model:"legacy", dimension:3}, {} as any)).toThrow("Local model inference is retired");
  });

  test("closes provider before runtime exactly once across concurrent callers", async () => {
    const events: string[] = [];
    const provider = {
      close: vi.fn(async () => { events.push("provider"); }),
    } as any;
    const runtime = {
      dispose: vi.fn(async () => { events.push("runtime"); }),
    };
    const owner = new CompositeEmbeddingProviderOwner(provider, runtime);

    await Promise.all([owner.close(), owner.close()]);

    expect(events).toEqual(["provider", "runtime"]);
    expect(provider.close).toHaveBeenCalledTimes(1);
    expect(runtime.dispose).toHaveBeenCalledTimes(1);
  });

  test("still disposes the runtime when provider close fails", async () => {
    const runtime = { dispose: vi.fn(async () => undefined) };
    const owner = new CompositeEmbeddingProviderOwner({
      close: vi.fn(async () => { throw new Error("provider close failed"); }),
    } as any, runtime);

    await expect(owner.close()).rejects.toThrow("provider close failed");
    expect(runtime.dispose).toHaveBeenCalledTimes(1);
  });
});
