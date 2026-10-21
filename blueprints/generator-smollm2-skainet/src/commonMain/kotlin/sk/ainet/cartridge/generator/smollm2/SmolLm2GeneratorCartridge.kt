package sk.ainet.cartridge.generator.smollm2

import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.cartridge.profiles.llm.LlmEngine
import sk.ainet.cartridge.profiles.llm.LlmEngineInfo
import sk.ainet.cartridge.profiles.llm.LlmMessage
import sk.ainet.cartridge.profiles.llm.LlmRole
import sk.ainet.cartridge.profiles.llm.LlmSession
import sk.ainet.cartridge.profiles.llm.LlmSessionConfig
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.openRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaConfigParser
import sk.ainet.models.llama.LlamaNetworkLoader

/**
 * The SmolLM2 generator cartridge: Hugging Face's `SmolLM2-135M-Instruct` running entirely on
 * SKaiNET-transformers' commonMain llama decoder path, eager mode — byte-level BPE tokenize →
 * 30-layer llama (576 hidden, 9 heads / 3 KV heads, GQA, tied embeddings, RoPE base 100 000) →
 * stop-token-aware decode. The first `task: "generation"` blueprint and the first `llm/v1`
 * implementation.
 *
 * Opening needs the pinned model snapshot on disk (the blueprint's fetch step; ~260 MB). The
 * engine facts are intrinsic: context window, vocab, RoPE base and the stop ids are properties
 * of the pinned checkpoint, asserted against the loaded config at open time, never read from a
 * file a host could drift from.
 *
 * The checkpoint's ChatML template (including its default system prompt) is rendered INSIDE the
 * cartridge, per ADR-011's boundary: hosts pass messages as data and never concatenate
 * `<|im_start|>` strings themselves.
 */
public object SmolLm2GeneratorCartridge {

    /** Hugging Face checkpoint this cartridge is pinned to. */
    public const val MODEL_REPO: String = "HuggingFaceTB/SmolLM2-135M-Instruct"
    public const val MODEL_REVISION: String = "12fd25f77366fa6b3b4b768ec3050bf629380bac"

    public const val CONTEXT_WINDOW_TOKENS: Int = 8192
    public const val VOCAB_SIZE: Int = 49152

    /** The model card's recommended sampling defaults (ADR-011 RECOMMENDED attributes). */
    public const val DEFAULT_TEMPERATURE: Float = 0.2f
    public const val DEFAULT_TOP_P: Float = 0.9f
    public const val DEFAULT_MAX_OUTPUT_TOKENS: Int = 256

    /** `<|im_start|>` — also the BOS the chat template opens with. */
    internal const val IM_START_TOKEN_ID: Int = 1

    /** `<|im_end|>` — the turn-closing stop token. */
    internal const val IM_END_TOKEN_ID: Int = 2

    /**
     * SmolLM2's RoPE base (`rope_theta` 100 000). [LlamaConfigParser] does not read the field
     * (its `GgufDecoderMetadata` default of 10 000 would silently skew every attention
     * position), so the cartridge pins it and asserts the config agrees at open time.
     */
    internal const val ROPE_FREQ_BASE: Float = 100_000f

    /**
     * The default snapshot location: `SMOLLM2_MODEL_DIR` when set, else the shared SKaiNET
     * model cache (`~/.cache/skainet/models/HuggingFaceTB_SmolLM2-135M-Instruct`).
     */
    public fun defaultModelDir(): String? {
        val explicit = envVar("SMOLLM2_MODEL_DIR")
        if (!explicit.isNullOrBlank()) return explicit
        val home = homeDir() ?: return null
        return "$home/.cache/skainet/models/${MODEL_REPO.replace('/', '_')}"
    }

    /**
     * Load the snapshot and build the engine. Suspend because weight loading is real I/O
     * (~260 MB of bf16 tensors, widened to f32 on load).
     *
     * @param modelDir snapshot directory holding `config.json`, `model.safetensors`,
     *   `tokenizer.json` and `tokenizer_config.json`; defaults to [defaultModelDir]
     * @param ctx execution context; the decoder runtime manages its own per-step scratch
     */
    public suspend fun open(
        modelDir: String? = null,
        ctx: ExecutionContext = DirectCpuExecutionContext(),
    ): SmolLm2Engine {
        val dir = modelDir ?: defaultModelDir()
            ?: error("No model directory: set SMOLLM2_MODEL_DIR or pass modelDir explicitly")

        val configJson = readTextFile("$dir/config.json")
            ?: error("config.json not found in $dir — run the blueprint's fetch step first")
        val tokenizerJson = readTextFile("$dir/tokenizer.json")
            ?: error("tokenizer.json not found in $dir")
        val tokenizerConfigJson = readTextFile("$dir/tokenizer_config.json")

        val parsed = LlamaConfigParser.parse(configJson)
        check(parsed.vocabSize == VOCAB_SIZE && parsed.contextLength == CONTEXT_WINDOW_TOKENS) {
            "Loaded config (vocab ${parsed.vocabSize}, context ${parsed.contextLength}) is not " +
                "the pinned $MODEL_REPO checkpoint — wrong snapshot in $dir?"
        }
        check(Regex("\"rope_theta\"\\s*:\\s*100000(\\.0)?\\b").containsMatchIn(configJson)) {
            "config.json rope_theta does not match the pinned ROPE_FREQ_BASE — wrong snapshot in $dir?"
        }
        val metadata = parsed.copy(ropeFreqBase = ROPE_FREQ_BASE)

        val model = LlamaNetworkLoader.fromSafeTensors(
            metadata = metadata,
            randomAccessProvider = {
                openRandomAccessSource("$dir/model.safetensors")
                    ?: error("cannot open $dir/model.safetensors")
            },
            tiedEmbeddings = LlamaConfigParser.isTiedEmbeddings(configJson),
        ).load<FP32, Float>(ctx)

        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            bos = metadata.bosTokenId,
        )
        check(runtime.vocabSize == VOCAB_SIZE) {
            "Loaded network has vocab ${runtime.vocabSize}, expected $VOCAB_SIZE"
        }

        val tokenizer = TokenizerFactory.fromTokenizerJsonString(tokenizerJson, tokenizerConfigJson)
        return SmolLm2Engine(runtime, tokenizer, ctx)
    }
}

/**
 * Render SmolLM2's own chat template (ChatML, byte-exact with the checkpoint's
 * `tokenizer_config.json` `chat_template`, including its default system prompt and the
 * generation prompt).
 */
internal fun renderChatMl(messages: List<LlmMessage>): String = buildString {
    if (messages.firstOrNull()?.role != LlmRole.SYSTEM) {
        append("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n")
    }
    for (message in messages) {
        append("<|im_start|>")
        append(
            when (message.role) {
                LlmRole.SYSTEM -> "system"
                LlmRole.USER -> "user"
                LlmRole.ASSISTANT -> "assistant"
                LlmRole.TOOL -> "tool"
            },
        )
        append('\n')
        append(message.content)
        append("<|im_end|>\n")
    }
    append("<|im_start|>assistant\n")
}

/**
 * The llm/v1 engine over the loaded decoder.
 *
 * **Threading:** not thread-safe — one generation at a time (the decoder's KV cache is
 * engine state); open one engine per worker or serialize sessions.
 */
public class SmolLm2Engine internal constructor(
    internal val runtime: OptimizedLLMRuntime<FP32>,
    internal val tokenizer: Tokenizer,
    private val ctx: ExecutionContext,
) : LlmEngine {

    override val info: LlmEngineInfo = LlmEngineInfo(
        cartridgeId = "generator-smollm2-skainet",
        name = "SmolLM2-135M-Instruct",
        version = "0.1.0",
        contextWindowTokens = SmolLm2GeneratorCartridge.CONTEXT_WINDOW_TOKENS,
        maxOutputTokens = null,
        supportsToolCalling = false,
        supportsJsonMode = false,
        defaultTemperature = SmolLm2GeneratorCartridge.DEFAULT_TEMPERATURE,
        defaultTopK = null,
        defaultTopP = SmolLm2GeneratorCartridge.DEFAULT_TOP_P,
    )

    override fun session(config: LlmSessionConfig): LlmSession = SmolLm2Session(this, config)

    override fun close() {
        (ctx as? AutoCloseable)?.close()
    }
}
