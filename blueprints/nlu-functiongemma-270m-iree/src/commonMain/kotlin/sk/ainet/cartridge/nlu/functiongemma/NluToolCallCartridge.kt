package sk.ainet.cartridge.nlu.functiongemma

/**
 * What an NLU tool-calling cartridge returns for one transcript. The cartridge maps text to a
 * *function name + string arguments* from its catalog — nothing more. Turning that into an application
 * command (ids, entity canonicalisation, fallbacks) is the host's job.
 */
public sealed interface NluResolution {
    /** The model called one of the catalog's functions. [args] are the raw string arguments as emitted. */
    public data class Call(val name: String, val args: Map<String, String>, val raw: String, val timing: Timing) : NluResolution

    /** The model answered without calling a function (prose, refusal, or an unparsable call). */
    public data class NoCall(val text: String, val timing: Timing) : NluResolution

    /** The cartridge could not produce an answer inside the budget or the runtime failed. */
    public data class Failed(val reason: String, val cause: Throwable? = null, val timing: Timing? = null) : NluResolution

    /** Wall-clock breakdown of one [NluToolCallCartridge.resolve] call, in milliseconds. */
    public data class Timing(val tokenizeMs: Long, val restoreMs: Long, val chunkMs: Long, val decodeMs: Long, val decodeTokens: Int) {
        val totalMs: Long get() = tokenizeMs + restoreMs + chunkMs + decodeMs
    }
}

/**
 * Synchronous NLU cartridge contract: one transcript in, one [NluResolution] out, within
 * [resolve]'s budget. Synchronous because hosts call their NLU synchronously, and because a
 * cartridge must not own threads the host cannot see.
 */
public interface NluToolCallCartridge : AutoCloseable {
    /** The id from the cartridge's descriptor. */
    public val id: String

    /** The function names the model may call — exactly the catalog's names. */
    public val toolNames: Set<String>

    /** Loads the runtime and prefills the catalog prefix; idempotent. Expensive (tens of seconds on a small device). */
    public fun warmUp()

    /** Resolve [transcript]. Returns [NluResolution.Failed] rather than exceeding [budgetMs] by more than one decode step. */
    public fun resolve(transcript: String, budgetMs: Long): NluResolution
}
