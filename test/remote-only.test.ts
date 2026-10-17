import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore as createSdkStore } from "../src/index.js";
import {
  createStore,
  generateEmbeddings,
  canonicalEmbeddingBuildMaterial,
  hybridQuery,
  type Store,
} from "../src/store.js";
import { UnavailableLocalEmbeddingProvider } from "../src/embedding/local.js";
import { UnavailableOpenAIEmbeddingProvider } from "../src/embedding/openai.js";
import {
  beginEmbeddingBuild,
  completeEmbeddingBuild,
  createEmbeddingIdentity,
} from "../src/embedding/identity.js";
import { writeCanonicalEmbeddingConfig } from "../src/embedding/config.js";
import type { EmbeddingProvider } from "../src/embedding/provider.js";

const cleanups: Array<() => Promise<void> | void> = [];
beforeEach(() => {
  vi.stubEnv("TYPESAFE_API_KEY", "");
});
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
  vi.unstubAllEnvs();
});
function addDocument(
  store: Store,
  hash = "doc",
  body = "# Authentication\n\nAuthentication details for the service.",
) {
  const now = new Date().toISOString();
  store.insertContent(hash, body, now);
  store.insertDocument("docs", `${hash}.md`, "Authentication", hash, now, now);
}
function snapshot(store: Store) {
  return {
    config: store.db.prepare("SELECT * FROM store_config ORDER BY key").all(),
    state: store.db.prepare("SELECT * FROM embedding_index_state").all(),
    vectors: store.db
      .prepare("SELECT * FROM content_vectors ORDER BY hash,seq")
      .all(),
    payload: store.db
      .prepare("SELECT hash_seq,embedding FROM vectors_vec ORDER BY hash_seq")
      .all(),
  };
}
function remoteProvider(
  overrides: Partial<EmbeddingProvider> = {},
): EmbeddingProvider {
  return {
    providerId: "remote-test",
    model: "remote-model",
    remote: true,
    dimension: 3,
    canonicalIdentityMaterial: () =>
      JSON.stringify({
        provider: "remote-test",
        model: "remote-model",
        dimension: 3,
      }),
    formatQuery: (text) => text,
    formatDocument: (text) => text,
    embed: vi.fn(async () => ({
      vector: [1, 0, 0],
      model: "remote-model",
      dimension: 3,
    })),
    embedBatch: vi.fn(async (texts) =>
      texts.map(() => ({
        vector: [1, 0, 0],
        model: "remote-model",
        dimension: 3,
      })),
    ),
    close: async () => {},
    ...overrides,
  };
}

describe("remote-only core boundaries", () => {
  test("a legacy local database remains lexically readable and rejects vector writes without changing data", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-legacy-read-"));
    cleanups.push(() => rm(dir, { recursive: true, force: true }));
    const dbPath = join(dir, "legacy.sqlite");
    const legacy = createStore(dbPath);
    addDocument(legacy);
    const provider = new UnavailableLocalEmbeddingProvider({
      model: "legacy-local",
      dimension: 3,
    });
    writeCanonicalEmbeddingConfig(legacy.db, {
      provider: "local",
      model: provider.model,
      dimension: 3,
    });
    legacy.ensureVecTable(3);
    const identity = createEmbeddingIdentity({
      providerId: provider.providerId,
      model: provider.model,
      dimension: 3,
      remote: false,
      canonicalMaterial: canonicalEmbeddingBuildMaterial(
        provider.canonicalIdentityMaterial(),
        "regex",
      ),
    });
    const lease = beginEmbeddingBuild(legacy.db, identity, {
      ownerId: "old-generation",
      now: Date.now(),
      leaseMs: 60_000,
      allowDestructiveRebuild: true,
    });
    legacy.insertEmbedding(
      "doc",
      0,
      0,
      new Float32Array([1, 0, 0]),
      provider.model,
      new Date().toISOString(),
      1,
      identity.fingerprint,
      lease,
    );
    completeEmbeddingBuild(legacy.db, lease);
    const before = snapshot(legacy);
    legacy.close();
    const sdk = await createSdkStore({ dbPath });
    cleanups.push(() => sdk.close());
    expect((await sdk.searchLex("authentication")).length).toBe(1);
    expect((await sdk.search({ query: "authentication details", expansion: "skip", rerank: false })).length).toBe(
      1,
    );
    expect(await sdk.getStatus()).toBeDefined();
    await expect(sdk.searchVector("authentication")).rejects.toThrow(
      "configured remote",
    );
    await expect(sdk.embed({ force: true })).rejects.toThrow(
      "configured remote",
    );
    expect(snapshot(sdk.internal)).toEqual(before);
  });
  test("missing model configuration permits explicit BM25 but never silently skips model operations", async () => {
    const store = createStore(":memory:");
    cleanups.push(() => store.close());
    addDocument(store);
    expect((await hybridQuery(store, "authentication details", { expansion: "skip", skipRerank: true })).length).toBe(1);
    await expect(hybridQuery(store, "missing weak evidence")).rejects.toThrow("configured remote");
    await expect(hybridQuery(store, "authentication details", { expansion: "skip" })).rejects.toThrow("configured remote");
    await expect(store.expandQuery("authentication")).rejects.toThrow(
      "configured remote",
    );
    await expect(
      store.rerank("authentication", [{ file: "a", text: "a" }]),
    ).rejects.toThrow("configured remote");
    await expect(generateEmbeddings(store, { force: true })).rejects.toThrow(
      "configured remote",
    );
    expect(
      store.db.prepare("SELECT * FROM embedding_index_state").get(),
    ).toBeUndefined();
  });
  test("missing remote credentials fail before starting or resetting an embedding build", async () => {
    const store = createStore(":memory:");
    cleanups.push(() => store.close());
    addDocument(store);
    store.embeddingProvider = new UnavailableOpenAIEmbeddingProvider({
      model: "text-embedding-3-small",
      dimension: 1536,
      baseUrl: "https://api.openai.com/v1",
    });
    await expect(generateEmbeddings(store, { force: true })).rejects.toThrow(
      "configured remote",
    );
    expect(
      store.db.prepare("SELECT * FROM embedding_index_state").get(),
    ).toBeUndefined();
    expect(store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
  });
  test("remote document batching retries failed batches under the same active lease", async () => {
    const batch = vi.fn(async () => {
      throw new Error("transient batch");
    });
    const provider = remoteProvider({ embedBatch: batch });
    const store = createStore(":memory:", { embeddingProvider: provider });
    cleanups.push(() => store.close());
    addDocument(store, "first");
    addDocument(store, "second");
    store.authorizeRemoteRequest = vi.fn();
    const result = await generateEmbeddings(store, { maxDocsPerBatch: 1 });
    expect(result).toMatchObject({
      docsProcessed: 2,
      chunksEmbedded: 2,
      errors: 0,
    });
    expect(batch).toHaveBeenCalledTimes(2);
    expect(provider.embed).toHaveBeenCalledTimes(2);
    expect(
      store.db.prepare("SELECT status FROM embedding_index_state").get(),
    ).toMatchObject({ status: "ready" });
    expect(store.authorizeRemoteRequest).toHaveBeenCalledWith(
      "index-build",
      expect.objectContaining({
        lease: expect.objectContaining({ ownerId: expect.any(String) }),
      }),
    );
  });
});
