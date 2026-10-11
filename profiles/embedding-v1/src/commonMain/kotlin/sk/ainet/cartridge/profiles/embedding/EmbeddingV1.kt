package sk.ainet.cartridge.profiles.embedding

/**
 * The proposed `embedding/v1` task profile (fills OQ12 of the cartridge spec): the swap point
 * every text-embedding cartridge implements, regardless of whether the vectors come from a
 * hash projection, a BERT encoder on SKaiNET tensors, or an IREE-compiled module.
 *
 * Contract:
 *  - [dimensions] is fixed for the engine's lifetime and equals the length of every returned
 *    vector.
 *  - [embed] is deterministic for a given cartridge version: same text, same bytes (up to the
 *    numeric tolerance the cartridge's golden verification declares for its backends).
 *  - Returned vectors are L2-normalized when the cartridge descriptor says `normalized: true`;
 *    cosine similarity is then a plain dot product.
 *  - Vectors are only comparable within one `space_fingerprint`. Mixing spaces is the silent
 *    retrieval killer the parity contract exists to catch; hosts must compare the descriptor's
 *    fingerprint, dimensions and prefixes against the index they query.
 *
 * This interface deliberately matches the shape of SKaiNET-transformers' `llm-api`
 * `EmbeddingModel` SPI (`embed(text): FloatArray`, `dimensions`, `AutoCloseable`), so one
 * engine class can implement both: the cartridge contract for hosts that speak descriptors and
 * parity, and the provider-neutral SPI for hosts that already speak Spring-AI-shaped embedding.
 */
public interface EmbeddingEngine : AutoCloseable {
    public val dimensions: Int

    public fun embed(text: String): FloatArray
}

/** What the engine provably does; hosts compare this against the index they want to query. */
public data class EmbeddingDescriptor(
    public val spaceFingerprint: String,
    public val dimensions: Int,
    public val normalized: Boolean,
    public val queryPrefix: String,
    public val documentPrefix: String,
)
