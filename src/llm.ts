/** Neutral model contracts and scoped remote-operation ownership. */

export function isQwen3EmbeddingModel(modelUri: string): boolean {
  return /qwen.*embed/i.test(modelUri) || /embed.*qwen/i.test(modelUri);
}

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma (default).
 * Uses Qwen3-Embedding instruct format when a Qwen embedding model is active.
 */
export function formatQueryForEmbedding(
  query: string,
  modelUri?: string,
): string {
  const uri = modelUri ?? resolveEmbedModel();
  if (isQwen3EmbeddingModel(uri)) {
    return `Instruct: Retrieve relevant documents for the given query\nQuery: ${query}`;
  }
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields (default).
 * Qwen3-Embedding encodes documents as raw text without special prefixes.
 */
export function formatDocForEmbedding(
  text: string,
  title?: string,
  modelUri?: string,
): string {
  const uri = modelUri ?? resolveEmbedModel();
  if (isQwen3EmbeddingModel(uri)) {
    // Qwen3-Embedding: documents are raw text, no task prefix
    return title ? `${title}\n${text}` : text;
  }
  return `title: ${title || "none"} | text: ${text}`;
}

export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  signal?: AbortSignal;
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  signal?: AbortSignal;
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  signal?: AbortSignal;
  model?: string;
  timeZone?: string;
  context?: string;
};

/**
 * Options for LLM sessions
 */
export type LLMSessionOptions = {
  /** Max session duration in ms (default: 10 minutes) */
  maxDuration?: number;
  /** External abort signal */
  signal?: AbortSignal;
  /** Debug name for logging */
  name?: string;
};

/**
 * Session interface for scoped LLM access with lifecycle guarantees
 */
export interface ILLMSession {
  /** Embedding model actually loaded by the session owner. */
  readonly embeddingModel: string;
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  embedBatch(
    texts: string[],
    options?: EmbedOptions,
  ): Promise<(EmbeddingResult | null)[]>;
  expandQuery(
    query: string,
    options?: {
      context?: string;
      includeLexical?: boolean;
      includeHyde?: boolean;
      searchIntent?: SearchIntentGuidance;
      signal?: AbortSignal;
    },
  ): Promise<Queryable[]>;
  rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions,
  ): Promise<RerankResult>;
  /** Whether this session is still valid (not released or aborted) */
  readonly isValid: boolean;
  /** Abort signal for this session (aborts on release or maxDuration) */
  readonly signal: AbortSignal;
}

/**
 * Supported query types for different search backends
 */
export type QueryType = "lex" | "vec" | "hyde";

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

/**
 * Structured search intent strategy and guidance for query expansion
 */
export interface SearchIntentGuidance {
  label: string;
  objective: string;
  lexGuidance: string;
  vecGuidance: string;
}

// Legacy identifiers are retained only for existing configuration and index diagnostics.
const DEFAULT_EMBED_MODEL =
  "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL =
  "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const DEFAULT_GENERATE_MODEL =
  "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

export const DEFAULT_EMBED_MODEL_URI = DEFAULT_EMBED_MODEL;
export const DEFAULT_RERANK_MODEL_URI = DEFAULT_RERANK_MODEL;
export const DEFAULT_GENERATE_MODEL_URI = DEFAULT_GENERATE_MODEL;

export type ModelResolutionConfig = {
  embed?: string;
  generate?: string;
  rerank?: string;
  embed_api_model?: string;
  generate_api_model?: string;
  rerank_api_model?: string;
  generate_api_key?: string;
  rerank_api_key?: string;
  jev_api_key?: string;
  jev_base_url?: string;
  jev_api_model?: string;
  embed_api_url?: string;
  generate_api_url?: string;
  generate_url?: string;
  generate_base_url?: string;
  rerank_api_url?: string;
  rerank_url?: string;
  rerank_base_url?: string;
};

export function resolveEmbedModel(config?: ModelResolutionConfig): string {
  if (config?.embed_api_model) return config.embed_api_model;
  return config?.embed || process.env.QMD_EMBED_MODEL || DEFAULT_EMBED_MODEL;
}

export function resolveGenerateModel(config?: ModelResolutionConfig): string {
  if (config?.generate_api_model) return config.generate_api_model;
  return (
    config?.generate || process.env.QMD_GENERATE_MODEL || DEFAULT_GENERATE_MODEL
  );
}

export function resolveRerankModel(config?: ModelResolutionConfig): string {
  if (config?.rerank_api_model) return config.rerank_api_model;
  return config?.rerank || process.env.QMD_RERANK_MODEL || DEFAULT_RERANK_MODEL;
}

export function resolveModels(
  config?: ModelResolutionConfig,
): Required<Pick<ModelResolutionConfig, "embed" | "generate" | "rerank">> {
  return {
    embed: resolveEmbedModel(config),
    generate: resolveGenerateModel(config),
    rerank: resolveRerankModel(config),
  };
}

export interface LLM {
  readonly embedModelName?: string;
  readonly generateModelName?: string;
  readonly rerankModelName?: string;
  readonly supportsExpand?: boolean;
  readonly supportsRerank?: boolean;
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(
    prompt: string,
    options?: GenerateOptions,
  ): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(
    query: string,
    options?: {
      context?: string;
      includeLexical?: boolean;
      includeHyde?: boolean;
      searchIntent?: SearchIntentGuidance;
      signal?: AbortSignal;
    },
  ): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions,
  ): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

export class ModelConfigurationError extends Error {
  constructor(operation: string) {
    super(
      `${operation} requires a configured remote model. Local inference is no longer supported.`,
    );
    this.name = "ModelConfigurationError";
  }
}
export class SessionReleasedError extends Error {
  constructor(message = "LLM session has been released or aborted") {
    super(message);
    this.name = "SessionReleasedError";
  }
}
class LLMSessionManager {
  sessions = 0;
  operations = 0;
  private readonly waiters = new Set<() => void>();
  changed(): void {
    if (this.sessions || this.operations) return;
    for (const resolve of this.waiters) resolve();
    this.waiters.clear();
  }
  drain(): Promise<void> {
    return this.sessions || this.operations
      ? new Promise((resolve) => this.waiters.add(resolve))
      : Promise.resolve();
  }
}
const managers = new WeakMap<LLM, LLMSessionManager>();
class LLMSession implements ILLMSession {
  private released = false;
  private readonly controller = new AbortController();
  private timer?: ReturnType<typeof setTimeout>;
  private readonly abortExternal: () => void;
  constructor(
    private readonly llm: LLM,
    private readonly manager: LLMSessionManager,
    private readonly options: LLMSessionOptions = {},
  ) {
    this.abortExternal = () => this.controller.abort(options.signal?.reason);
    if (options.signal?.aborted) this.abortExternal();
    else
      options.signal?.addEventListener("abort", this.abortExternal, {
        once: true,
      });
    const duration = options.maxDuration ?? 10 * 60 * 1000;
    if (duration > 0) {
      this.timer = setTimeout(
        () =>
          this.controller.abort(
            new Error(
              `Session "${options.name ?? "unnamed"}" exceeded max duration of ${duration}ms`,
            ),
          ),
        duration,
      );
      this.timer.unref();
    }
    manager.sessions++;
  }
  get embeddingModel(): string {
    return this.llm.embedModelName ?? "";
  }
  get signal(): AbortSignal {
    return this.controller.signal;
  }
  get isValid(): boolean {
    return !this.released && !this.signal.aborted;
  }
  release(): void {
    if (this.released) return;
    this.released = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    this.options.signal?.removeEventListener("abort", this.abortExternal);
    this.controller.abort(new SessionReleasedError());
    this.manager.sessions--;
    this.manager.changed();
  }
  private async operation<T>(
    fn: (signal: AbortSignal) => Promise<T>,
    operationSignal?: AbortSignal,
  ): Promise<T> {
    if (!this.isValid) throw new SessionReleasedError();
    const signal = operationSignal
      ? AbortSignal.any([this.signal, operationSignal])
      : this.signal;
    signal.throwIfAborted();
    this.manager.operations++;
    try {
      const result = await fn(signal);
      if (!this.isValid) throw new SessionReleasedError();
      signal.throwIfAborted();
      return result;
    } finally {
      this.manager.operations--;
      this.manager.changed();
    }
  }
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.operation(
      (signal) => this.llm.embed(text, { ...options, signal }),
      options?.signal,
    );
  }
  embedBatch(
    texts: string[],
    options?: EmbedOptions,
  ): Promise<(EmbeddingResult | null)[]> {
    return this.operation(
      (signal) =>
        Promise.all(
          texts.map((text) => this.llm.embed(text, { ...options, signal })),
        ),
      options?.signal,
    );
  }
  expandQuery(
    query: string,
    options?: {
      context?: string;
      includeLexical?: boolean;
      includeHyde?: boolean;
      searchIntent?: SearchIntentGuidance;
      signal?: AbortSignal;
    },
  ): Promise<Queryable[]> {
    return this.operation(
      (signal) => this.llm.expandQuery(query, { ...options, signal }),
      options?.signal,
    );
  }
  rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions,
  ): Promise<RerankResult> {
    return this.operation(
      (signal) => this.llm.rerank(query, documents, { ...options, signal }),
      options?.signal,
    );
  }
}
export async function withLLMSessionForLlm<T>(
  llm: LLM,
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions,
): Promise<T> {
  let manager = managers.get(llm);
  if (!manager) {
    manager = new LLMSessionManager();
    managers.set(llm, manager);
  }
  const session = new LLMSession(llm, manager, options);
  try {
    return await fn(session);
  } finally {
    session.release();
  }
}
export async function waitForLLMSessionsToDrain(llm: LLM): Promise<void> {
  await managers.get(llm)?.drain();
}
