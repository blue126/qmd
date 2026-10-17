import { CompositeEmbeddingProviderOwner } from "../embedding/owner.js";
import type { EmbeddingProvider, EmbeddingProviderOwner } from "../embedding/provider.js";
import { waitForLLMSessionsToDrain, type LLM } from "../llm.js";

export interface CliEmbeddingProviderConfig {
  provider: "local" | "openai" | "voyageai";
  model: string;
  dimension: number | null;
}

export function createCliEmbeddingProviderOwner(
  config: CliEmbeddingProviderConfig,
  runtime: LLM,
  remoteProvider?: EmbeddingProvider,
): EmbeddingProviderOwner {
  if (!remoteProvider?.remote) {
    throw new Error("Local model inference is retired. Configure a remote embedding provider and explicitly rebuild with qmd embed --force.");
  }
  return new CompositeEmbeddingProviderOwner(remoteProvider, {
    dispose: async () => {
      await waitForLLMSessionsToDrain(runtime);
      await runtime.dispose();
    },
  });
}
