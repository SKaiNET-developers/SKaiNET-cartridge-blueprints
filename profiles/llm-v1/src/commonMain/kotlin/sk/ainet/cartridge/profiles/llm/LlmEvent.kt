package sk.ainet.cartridge.profiles.llm

/**
 * The event vocabulary of one generation turn (ADR-011).
 *
 * Ordering contract: [Ready] → ([Delta] | [ToolCall])* → ([Final] | [Failed]).
 * The events flow completes after the terminal event.
 *
 * ABI mapping (ADR-007/ADR-011): [Delta] = `PARTIAL` payload; [ToolCall] = `PARTIAL` payload
 * whose port schema marks it structured; [Final] = `FINAL`. An LLM cartridge never emits
 * the ASR-only `CTG_RESULT_ENDPOINT`.
 */
public sealed interface LlmEvent {

    /** The session is initialized and generation is starting. */
    public data object Ready : LlmEvent

    /** Incremental generated text, in emission order. */
    public data class Delta(val text: String) : LlmEvent

    /**
     * A structured tool call; [json] is validated against the port's ADR-007 `output_schema`
     * when the descriptor declares one. Only engines with `supports_tool_calling` emit it.
     */
    public data class ToolCall(val json: String) : LlmEvent

    /**
     * The completed turn. Terminal. [text] is the full generated text (the concatenation of
     * the deltas); on [LlmFinishReason.CANCELLED] it is the partial output up to the cancel.
     */
    public data class Final(
        val text: String,
        val finishReason: LlmFinishReason,
        /** Raw generated token ids, when the engine exposes them (tokenizer-specific). */
        val tokens: List<Int>? = null,
    ) : LlmEvent

    /** Terminal failure. */
    public data class Failed(val error: LlmError) : LlmEvent {
        override fun toString(): String = "Failed(${error.code}: ${error.message})"
    }
}
