package sk.ainet.cartridge.profiles.llm

/**
 * Task profile `llm/v1` — the public, versioned consumer contract for generation cartridges
 * (ADR-011, over ADR-007's `output_schema` + attributes groundwork).
 *
 * The profile owns ALL of its types, like asr/v1: nothing here references any application's
 * contract and nothing here references any one cartridge's io shape. Capability differences
 * between engines stay visible in [LlmEngineInfo] and the cartridge descriptor — the profile
 * hides the io shape, never the capability.
 *
 * The one place this profile sharpens ADR-011's sketch: the prompt API takes **messages and
 * tools as data**, not a pre-rendered string. That is the ADR's own stated decision ("prompt
 * rendering happens inside the cartridge — the profile passes messages/tools as data; a host
 * that concatenates `<start_of_turn>` strings is in the wrong layer"); the `prompt(rendered)`
 * signature in the same ADR's sketch predates that sentence, and implementing the sketch would
 * put the chat template on the wrong side of the boundary it draws.
 */

/** Profile-owned engine description — mirrors descriptor facts (ADR-007), never app concepts. */
public class LlmEngineInfo(
    /** The cartridge id this engine wraps (descriptor `id`), if cartridge-backed. */
    public val cartridgeId: String?,
    public val name: String,
    public val version: String,
    /** Descriptor `attributes.context_window_tokens` — token count, never seconds (ADR-007). */
    public val contextWindowTokens: Int,
    /** Descriptor `attributes.max_output_tokens`; null when only the context window bounds it. */
    public val maxOutputTokens: Int?,
    /** Descriptor `attributes.supports_tool_calling`. */
    public val supportsToolCalling: Boolean,
    /** Descriptor `attributes.supports_json_mode`. */
    public val supportsJsonMode: Boolean,
    /** The model card's recommended sampling defaults (ADR-011 RECOMMENDED attributes). */
    public val defaultTemperature: Float?,
    public val defaultTopK: Int?,
    public val defaultTopP: Float?,
)

/** One turn of conversation content. The cartridge renders the chat template around these. */
public class LlmMessage(
    public val role: LlmRole,
    public val content: String,
)

public enum class LlmRole { SYSTEM, USER, ASSISTANT, TOOL }

/**
 * A tool the model may call, passed as data (name + JSON-Schema arguments); how it is declared
 * to the model (template section, grammar, special tokens) is the cartridge's concern.
 */
public class LlmToolDef(
    public val name: String,
    public val description: String,
    /** JSON-Schema fragment for the arguments, matching the port's ADR-007 `output_schema`. */
    public val parametersJsonSchema: String? = null,
)

/**
 * Per-session overrides of the cartridge's model-card sampling defaults (ADR-011: a cartridge
 * ships its recommendation, a host may override per session). Null means the engine default.
 */
public class LlmSessionConfig(
    public val temperature: Float? = null,
    public val topK: Int? = null,
    public val topP: Float? = null,
    public val maxOutputTokens: Int? = null,
    /** Sampling seed for reproducible runs; irrelevant under greedy decoding. */
    public val seed: Long? = null,
)

public class LlmError(
    public val code: LlmErrorCode,
    public val message: String,
    public val recoverable: Boolean,
)

/** Reuses asr/v1's error-code conventions (ADR-011: reused, not re-invented). */
public enum class LlmErrorCode {
    ENGINE_UNAVAILABLE,
    MODEL_LOAD_FAILED,
    PROMPT_TOO_LONG,
    GENERATION_FAILED,
    INFERENCE_TIMEOUT,
    UNKNOWN,
}

public enum class LlmFinishReason {
    /** The model emitted a stop token. */
    STOP,
    /** The output-token budget ran out first. */
    LENGTH,
    /** [LlmSession.cancel] interrupted generation; the Final text is the partial output. */
    CANCELLED,
}
