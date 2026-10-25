package sk.ainet.cartridge.generator.llama32

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
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaNetworkLoader

/**
 * The Llama 3.2 1B generator cartridge: Meta's `Llama-3.2-1B-Instruct` (Unsloth's Q4_K_M GGUF)
 * running entirely on SKaiNET-transformers' commonMain llama decoder path, eager mode — the
 * second `task: "generation"` / llm/v1 blueprint after SmolLM2, and the first on a **quantized
 * GGUF** checkpoint: packed Q4_K blocks served zero-copy (MAPPED) through the engine's
 * streaming GGUF loader into the quantized matmul kernels.
 *
 * Opening needs the pinned GGUF on disk (the blueprint's fetch step; ~0.8 GB, Llama 3.2
 * Community License — acceptance is yours, never the blueprint's). The engine facts are
 * intrinsic: vocab, context window, stop ids and the Llama 3 turn structure are properties of
 * the pinned checkpoint, asserted against the loaded GGUF metadata at open time.
 *
 * The checkpoint's Llama 3 chat template is rendered INSIDE the cartridge, per ADR-011's
 * boundary: hosts pass messages as data and never concatenate `<|start_header_id|>` strings.
 */
public object Llama32GeneratorCartridge {

    /** Hugging Face checkpoint this cartridge is pinned to. */
    public const val MODEL_REPO: String = "unsloth/Llama-3.2-1B-Instruct-GGUF"
    public const val MODEL_REVISION: String = "b69aef112e9f895e6f98d7ae0949f72ff09aa401"
    public const val MODEL_FILE: String = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    public const val MODEL_BYTES: Long = 807_694_368L

    public const val CONTEXT_WINDOW_TOKENS: Int = 131_072
    public const val VOCAB_SIZE: Int = 128_256

    /** The model card's recommended sampling defaults (ADR-011 RECOMMENDED attributes). */
    public const val DEFAULT_TEMPERATURE: Float = 0.6f
    public const val DEFAULT_TOP_P: Float = 0.9f
    public const val DEFAULT_MAX_OUTPUT_TOKENS: Int = 256

    /** `<|begin_of_text|>`. */
    internal const val BOS_TOKEN_ID: Int = 128_000

    /**
     * The checkpoint's `generation_config.json` stop set: `<|end_of_text|>` (128001),
     * `<|eom_id|>` (128008) and `<|eot_id|>` (128009, the turn-closing stop the instruct
     * tune actually emits).
     */
    internal val STOP_TOKEN_IDS: Set<Int> = setOf(128_001, 128_008, 128_009)

    /**
     * The default checkpoint location: `LLAMA32_MODEL_DIR` when set, else the shared SKaiNET
     * model cache (`~/.cache/skainet/models/unsloth_Llama-3.2-1B-Instruct-GGUF`).
     */
    public fun defaultModelDir(): String? {
        val explicit = envVar("LLAMA32_MODEL_DIR")
        if (!explicit.isNullOrBlank()) return explicit
        val home = homeDir() ?: return null
        return "$home/.cache/skainet/models/${MODEL_REPO.replace('/', '_')}"
    }

    /**
     * Open the pinned GGUF and build the engine. Suspend because the streaming load is real
     * I/O (the header pass plus mapping ~0.8 GB of packed blocks).
     *
     * @param modelDir directory holding [MODEL_FILE]; defaults to [defaultModelDir]
     * @param ctx execution context; the decoder runtime manages its own per-step scratch
     */
    public suspend fun open(
        modelDir: String? = null,
        ctx: ExecutionContext = DirectCpuExecutionContext(),
    ): Llama32Engine {
        installKernelPacks()
        val dir = modelDir ?: defaultModelDir()
            ?: error("No model directory: set LLAMA32_MODEL_DIR or pass modelDir explicitly")
        val path = "$dir/$MODEL_FILE"
        val size = fileSize(path)
            ?: error("$MODEL_FILE not found in $dir — run the blueprint's fetch step first")
        check(size == MODEL_BYTES) {
            "$path is $size bytes, the pinned checkpoint is $MODEL_BYTES — wrong or truncated file"
        }

        val loader = DecoderGgufWeightLoader(
            randomAccessProvider = {
                openRandomAccessSource(path) ?: error("cannot open $path")
            },
            // The pinned checkpoint is a llama GGUF; anything else is the wrong file.
            acceptedArchitectures = setOf("llama"),
        )
        val weights = loader.loadToMapStreaming<FP32, Float>(ctx)
        val metadata = weights.metadata
        check(metadata.vocabSize == VOCAB_SIZE && metadata.contextLength == CONTEXT_WINDOW_TOKENS) {
            "Loaded GGUF (vocab ${metadata.vocabSize}, context ${metadata.contextLength}) is " +
                "not the pinned $MODEL_REPO checkpoint — wrong file in $dir?"
        }
        check(metadata.bosTokenId == BOS_TOKEN_ID && metadata.eosTokenId in STOP_TOKEN_IDS) {
            "Loaded GGUF stop/bos ids (bos ${metadata.bosTokenId}, eos ${metadata.eosTokenId}) " +
                "do not match the pinned checkpoint"
        }

        val model = LlamaNetworkLoader.fromWeights(weights)
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            bos = metadata.bosTokenId,
        )

        val tokenizer = openRandomAccessSource(path)?.use { source ->
            TokenizerFactory.fromGgufSource(source)
        } ?: error("cannot open $path for the tokenizer pass")

        return Llama32Engine(runtime, tokenizer, ctx)
    }
}

/**
 * Render the checkpoint's own Llama 3 turn structure (byte-exact with the no-tools path of the
 * family template: `<|begin_of_text|>` then `<|start_header_id|>{role}<|end_header_id|>\n\n
 * {content}<|eot_id|>` per message, closed with the assistant generation prompt).
 */
internal fun renderLlama3(messages: List<LlmMessage>): String = buildString {
    append("<|begin_of_text|>")
    for (message in messages) {
        append("<|start_header_id|>")
        append(
            when (message.role) {
                LlmRole.SYSTEM -> "system"
                LlmRole.USER -> "user"
                LlmRole.ASSISTANT -> "assistant"
                LlmRole.TOOL -> "tool"
            },
        )
        append("<|end_header_id|>\n\n")
        append(message.content)
        append("<|eot_id|>")
    }
    append("<|start_header_id|>assistant<|end_header_id|>\n\n")
}

/**
 * The llm/v1 engine over the loaded decoder.
 *
 * **Threading:** not thread-safe — one generation at a time (the decoder's KV cache is
 * engine state); open one engine per worker or serialize sessions.
 */
public class Llama32Engine internal constructor(
    internal val runtime: OptimizedLLMRuntime<FP32>,
    internal val tokenizer: Tokenizer,
    private val ctx: ExecutionContext,
) : LlmEngine {

    override val info: LlmEngineInfo = LlmEngineInfo(
        cartridgeId = "generator-llama32-1b-skainet",
        name = "Llama-3.2-1B-Instruct",
        version = "0.1.0",
        contextWindowTokens = Llama32GeneratorCartridge.CONTEXT_WINDOW_TOKENS,
        maxOutputTokens = null,
        supportsToolCalling = false,
        supportsJsonMode = false,
        defaultTemperature = Llama32GeneratorCartridge.DEFAULT_TEMPERATURE,
        defaultTopK = null,
        defaultTopP = Llama32GeneratorCartridge.DEFAULT_TOP_P,
    )

    override fun session(config: LlmSessionConfig): LlmSession = Llama32Session(this, config)

    override fun close() {
        (ctx as? AutoCloseable)?.close()
    }
}
