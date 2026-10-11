package sk.ainet.cartridge.profiles.llm

import kotlinx.coroutines.flow.Flow

/**
 * The `llm/v1` core contract (Flow-based layer, ADR-011). Implemented by cartridge adapters
 * and consumed by applications' closed adapters and by runners — never the other way around:
 * this artifact depends on no application and no cartridge.
 *
 * Lifecycle: create (implementation-specific factory) → [session]* → [close].
 */
public interface LlmEngine : AutoCloseable {

    public val info: LlmEngineInfo

    /**
     * Open a new generation session. Sessions are independent; whether they may run
     * concurrently is an engine/cartridge property (descriptor `requirements`), not a
     * profile guarantee.
     */
    public fun session(config: LlmSessionConfig = LlmSessionConfig()): LlmSession
}

/**
 * One generation turn.
 *
 * [prompt] supplies the conversation as data; the cartridge renders the checkpoint's own chat
 * template around it (ADR-011's boundary: the template stays with the weights it was trained
 * for). Multi-turn chat is the host replaying its history as messages into a fresh session.
 *
 * Threading: [prompt] and the [events] collection from one consumer at a time; [cancel] from
 * anywhere (the one cross-thread call, mirroring asr/v1). [events] is cold until collected
 * and completes after the terminal event.
 */
public interface LlmSession : AutoCloseable {

    /** Supply the turn's input. One prompt per session; call before collecting [events]. */
    public suspend fun prompt(
        messages: List<LlmMessage>,
        tools: List<LlmToolDef> = emptyList(),
    )

    /** Cooperative cancel; [events] terminates with [LlmEvent.Final] (CANCELLED, partial text). */
    public fun cancel()

    public val events: Flow<LlmEvent>
}
