import {
  ModelConfigurationError,
  formatDocForEmbedding,
  formatQueryForEmbedding,
} from "../llm.js";
import type {
  EmbeddingProvider,
  EmbeddingOperationOptions,
  EmbeddingVector,
} from "./provider.js";
import { EmbeddingProviderError } from "./provider.js";
import { canonicalLocalEmbeddingIdentityMaterial } from "./local-identity.js";

/** Diagnostic-only representation of a retired local configuration. Never loads a model. */
export class UnavailableLocalEmbeddingProvider implements EmbeddingProvider {
  readonly providerId = "local-llama-cpp";
  readonly remote = false;
  readonly available = false;
  readonly model: string;
  readonly dimension: number | null;
  constructor(options: { model: string; dimension?: number | null }) {
    this.model = options.model;
    this.dimension = options.dimension ?? null;
  }
  canonicalIdentityMaterialForDimension(dimension: number): string {
    return canonicalLocalEmbeddingIdentityMaterial(this.model, dimension);
  }
  canonicalIdentityMaterial(): string {
    if (this.dimension === null) {
      throw new EmbeddingProviderError(
        "DIMENSION_UNKNOWN",
        "Legacy embedding identity requires a persisted vector dimension.",
      );
    }
    return this.canonicalIdentityMaterialForDimension(this.dimension);
  }
  formatQuery(query: string): string {
    return formatQueryForEmbedding(query, this.model);
  }
  formatDocument(text: string, title?: string): string {
    return formatDocForEmbedding(text, title, this.model);
  }
  async embed(
    _text: string,
    _options: EmbeddingOperationOptions,
  ): Promise<EmbeddingVector> {
    throw new ModelConfigurationError("Vector operations");
  }
  async embedBatch(
    _texts: string[],
    _options: EmbeddingOperationOptions,
  ): Promise<EmbeddingVector[]> {
    throw new ModelConfigurationError("Vector operations");
  }
  async close(): Promise<void> {}
}
