package sk.ainet.cartridge.asr.moonshine

/**
 * The contract of a streaming speech-to-text cartridge: an utterance is one [StreamingAsrSession] — 16 kHz mono
 * float PCM in as it arrives, cumulative partial transcripts out, one exact final on [StreamingAsrSession.finish].
 * The session is reusable: after `finish()` (or [StreamingAsrSession.reset]) it starts the next utterance.
 *
 * Partials are cumulative and prefix-stable *within the runtime's approximation*: the streaming loop reuses its
 * self-attention cache across hops and never retracts a token; `finish()` re-decodes the whole utterance exactly,
 * so the final may differ from the last partial. Hosts that act on text act on the final.
 */
public interface StreamingAsrCartridge : AutoCloseable {
    /** Cartridge id from the pack's descriptor. */
    public val id: String

    /** BCP-47 language subtag of the loaded flavor (`en`, `de`). */
    public val language: String

    /** Sample rate the session expects. */
    public val sampleRateHz: Int get() = 16_000

    /** Opens the (single) streaming session. The cartridge owns the engine; the session owns one utterance at a time. */
    public fun open(): StreamingAsrSession
}

public interface StreamingAsrSession : AutoCloseable {
    /** Feed mono 16 kHz PCM in [-1, 1]. Returns the updated cumulative partial, or `null` when nothing changed. */
    public fun feed(pcm: FloatArray): String?

    /** End of utterance: the exact final transcript (empty when nothing was recognized). Resets for the next utterance. */
    public fun finish(): String

    /** Abort the current utterance; the session stays usable. */
    public fun reset()
}
