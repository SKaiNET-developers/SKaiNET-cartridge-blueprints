package sk.ainet.cartridge.asr.whisper

/**
 * The contract of a short-utterance speech-to-text cartridge: one utterance of 16 kHz mono float PCM in, its
 * transcript out. The model sees a fixed window ([maxUtteranceSeconds]); a host that hands it more audio gets the
 * window's transcript, so utterance segmentation (push-to-talk, voice activity) stays in front of the cartridge.
 */
public interface AsrCartridge : AutoCloseable {
    /** Cartridge id from the pack's descriptor. */
    public val id: String

    /** BCP-47 language subtag of the loaded flavor (`en`, `de`). */
    public val language: String

    /** Sample rate [transcribe] expects. */
    public val sampleRateHz: Int get() = 16_000

    /** Length of the encoder window; shorter audio is zero-padded, longer audio is cut. */
    public val maxUtteranceSeconds: Float

    /** Transcribe one utterance of mono 16 kHz PCM in [-1, 1]; empty when nothing was recognized. */
    public fun transcribe(pcm: FloatArray): String
}
