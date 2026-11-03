package sk.ainet.cartridge.generator.lfm2

import sk.ainet.models.lfm2.HybridLfm2Model
import sk.ainet.apps.klfm2.Lfm2ChatMessage
import sk.ainet.apps.klfm2.Lfm2DecodingPolicy
import sk.ainet.apps.klfm2.Lfm2GeneratedToken
import sk.ainet.apps.klfm2.Lfm2GenerationOptions
import sk.ainet.apps.klfm2.Lfm2TextFrontend
import sk.ainet.apps.klfm2.Lfm2TextGenerator
import sk.ainet.apps.klfm2.Lfm2TextResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext
import sk.ainet.cartridge.profiles.llm.*
import java.util.concurrent.atomic.AtomicBoolean

/** Test seam at the already qualified synchronous text-generation boundary. */
internal interface Lfm2GenerationBackend {
    val contextLimit: Int
    fun promptTokenCount(messages: List<Lfm2ChatMessage>): Int
    fun generate(
        messages: List<Lfm2ChatMessage>,
        maxOutputTokens: Int,
        isCancelled: () -> Boolean,
        onToken: (Lfm2GeneratedToken) -> Unit,
    ): Lfm2TextResult
}

/**
 * JVM implementation of the pinned upstream llm/v1 contract over the local LFM2 model.
 *
 * Sessions are independent and one-shot. Generation is serialized per engine because the
 * caller's model may share a CPU execution context. Work runs on Dispatchers.IO because
 * synchronous token callbacks can block on stream backpressure; they must not occupy
 * the Default worker needed by the collector. The caller must also serialize any access
 * to this model or shared context outside this engine; this engine's mutex is local.
 * cancel() is cooperative at the underlying generator's forward boundaries. The engine
 * does not close the caller-owned model/execution context. Closing prevents new sessions
 * and cancels existing ones; closing is not a synchronous join of a kernel already running.
 *
 * This API alone is not a sealed cartridge. Its engine info therefore has no cartridge ID.
 * Decoding defaults to source checkpoint semantics; HF compatibility is explicit.
 */
class Lfm2LlmEngine internal constructor(
    backend: Lfm2GenerationBackend,
    private val defaultMaxOutputTokens: Int = 32,
    private val backendName: String = "LFM2.5-350M (local eager CPU)",
) : LlmEngine {
    constructor(
        model: HybridLfm2Model,
        frontend: Lfm2TextFrontend,
        decodingPolicy: Lfm2DecodingPolicy = Lfm2DecodingPolicy.CHECKPOINT,
        defaultMaxOutputTokens: Int = 32,
    ) : this(Lfm2TextGenerator(model, frontend, decodingPolicy), frontend,
        model.dimensions.contextLimit, "LFM2.5-350M (local eager CPU)", defaultMaxOutputTokens)

    /** Wrap a caller-owned CPU or accelerator generator using its actual compiled capacity. */
    constructor(
        generator: Lfm2TextGenerator,
        frontend: Lfm2TextFrontend,
        contextLimit: Int,
        backendName: String,
        defaultMaxOutputTokens: Int = 32,
    ) : this(object : Lfm2GenerationBackend {
        override val contextLimit: Int = contextLimit
        override fun promptTokenCount(messages: List<Lfm2ChatMessage>): Int = frontend.encodeMessages(messages).size
        override fun generate(messages: List<Lfm2ChatMessage>, maxOutputTokens: Int,
            isCancelled: () -> Boolean, onToken: (Lfm2GeneratedToken) -> Unit,
        ): Lfm2TextResult = generator.generate(messages, Lfm2GenerationOptions(maxOutputTokens), isCancelled, onToken)
    }, defaultMaxOutputTokens, backendName)

    private val contextLimit = backend.contextLimit
    private val lifecycleLock = Any()
    private val generationLock = Mutex()
    private var backend: Lfm2GenerationBackend? = backend
    private var closed = false
    private val sessions = mutableSetOf<ProfileSession>()

    init {
        require(contextLimit > 0)
        require(defaultMaxOutputTokens in 1..contextLimit)
    }

    override val info: LlmEngineInfo = LlmEngineInfo(
        cartridgeId = null,
        name = backendName,
        version = "0.1.0",
        contextWindowTokens = contextLimit,
        maxOutputTokens = contextLimit,
        supportsToolCalling = false,
        supportsJsonMode = false,
        // These describe the supported greedy policy, not the model card's sampling advice.
        defaultTemperature = 0f,
        defaultTopK = 1,
        defaultTopP = 1f,
    )

    override fun session(config: LlmSessionConfig): LlmSession {
        require(config.temperature == null || config.temperature == 0f) {
            "Only greedy temperature=0 is supported"
        }
        require(config.topK == null || config.topK == 1) { "Only greedy topK=1 is supported" }
        require(config.topP == null || config.topP == 1f) { "Only topP=1 is supported" }
        // A seed has no effect under greedy decoding, as specified by LlmSessionConfig.
        val maximum = config.maxOutputTokens ?: defaultMaxOutputTokens
        require(maximum in 1..contextLimit) { "Output budget must be within the model context limit" }
        return synchronized(lifecycleLock) {
            check(!closed) { "Engine is closed" }
            ProfileSession(this, maximum).also(sessions::add)
        }
    }

    override fun close() {
        val active = synchronized(lifecycleLock) {
            if (closed) return
            closed = true
            backend = null
            sessions.toList().also { sessions.clear() }
        }
        active.forEach { it.close() }
    }

    private fun release(session: ProfileSession) {
        synchronized(lifecycleLock) { sessions.remove(session) }
    }

    private suspend fun <T> withBackend(block: suspend (Lfm2GenerationBackend?) -> T): T {
        generationLock.lock()
        try {
            val current = synchronized(lifecycleLock) { backend }
            return block(current)
        } finally {
            generationLock.unlock()
        }
    }

    private class PromptTooLong : IllegalArgumentException("Prompt and output budget exceed the model context limit")

    private class ProfileSession(
        private val engine: Lfm2LlmEngine,
        private val maxOutputTokens: Int,
    ) : LlmSession {
        private val lifecycleLock = Any()
        private val cancelled = AtomicBoolean(false)
        private var closed = false
        private var collected = false
        private var messages: List<Lfm2ChatMessage>? = null

        override suspend fun prompt(messages: List<LlmMessage>, tools: List<LlmToolDef>) {
            require(tools.isEmpty()) { "Tool definitions are unsupported" }
            require(messages.isNotEmpty()) { "Conversation must not be empty" }
            val converted = messages.mapIndexed { index, message ->
                val role = when (message.role) {
                    LlmRole.SYSTEM -> {
                        require(index == 0) { "System content is supported only at the start" }
                        "system"
                    }
                    LlmRole.USER -> "user"
                    LlmRole.ASSISTANT -> "assistant"
                    LlmRole.TOOL -> throw IllegalArgumentException("Tool messages are unsupported")
                }
                Lfm2ChatMessage(role, message.content)
            }
            synchronized(lifecycleLock) {
                check(!closed) { "Session is closed" }
                check(this.messages == null && !collected) { "Each session accepts exactly one prompt" }
                this.messages = converted
            }
        }

        override fun cancel() { cancelled.set(true) }

        override fun close() {
            cancelled.set(true)
            synchronized(lifecycleLock) { closed = true; messages = null }
            engine.release(this)
        }

        override val events: Flow<LlmEvent> = channelFlow {
            val input = synchronized(lifecycleLock) {
                check(!closed) { "Session is closed" }
                check(!collected) { "Session events may be collected only once" }
                val accepted = checkNotNull(messages) { "Call prompt before collecting events" }
                collected = true
                accepted
            }
            try {
                send(LlmEvent.Ready)
                withContext(Dispatchers.IO) {
                    engine.withBackend { backend ->
                        val context = currentCoroutineContext()
                        context.ensureActive()
                        if (cancelled.get() || backend == null) {
                            send(LlmEvent.Final("", LlmFinishReason.CANCELLED, emptyList()))
                            return@withBackend
                        }
                        val stream = StableByteLevelText()
                        try {
                            val promptSize = backend.promptTokenCount(input)
                            if (promptSize.toLong() + maxOutputTokens - 1 > backend.contextLimit) throw PromptTooLong()
                            val result = backend.generate(input, maxOutputTokens,
                                isCancelled = { cancelled.get() || !context.isActive },
                                onToken = { token ->
                                    context.ensureActive()
                                    stream.append(token.text, terminal = false)?.let { delta ->
                                        trySendBlocking(LlmEvent.Delta(delta)).getOrThrow()
                                    }
                                },
                            )
                            context.ensureActive()
                            stream.append(result.text, terminal = true)?.let { send(LlmEvent.Delta(it)) }
                            val reason = when (result.stopReason) {
                                "eos" -> LlmFinishReason.STOP
                                "length" -> LlmFinishReason.LENGTH
                                "cancelled" -> LlmFinishReason.CANCELLED
                                else -> error("Unknown generation stop reason")
                            }
                            send(LlmEvent.Final(result.text, reason, result.generatedTokenIds.toList()))
                        } catch (cancel: CancellationException) {
                            throw cancel
                        } catch (failure: Exception) {
                            // Collection failure detaches the observer; do not invent a terminal event.
                            context.ensureActive()
                            val code = if (failure is PromptTooLong) LlmErrorCode.PROMPT_TOO_LONG
                                else LlmErrorCode.GENERATION_FAILED
                            send(LlmEvent.Failed(LlmError(code, failure.message ?: "Generation failed", recoverable = true)))
                        }
                    }
                }
            } finally {
                this@ProfileSession.close()
            }
        }.buffer(Channel.RENDEZVOUS)
    }
}

/**
 * The pinned tokenizer has a ByteLevel decoder. Appending byte tokens can replace an
 * incomplete UTF-8 suffix; earlier complete code points remain stable. Hold trailing
 * replacement characters (including genuine ones) until a later stable prefix or final.
 * An unexpected non-prefix revision fails instead of violating llm/v1's append-only API.
 */
internal class StableByteLevelText {
    private var emitted = ""

    fun append(cumulative: String, terminal: Boolean): String? {
        check(cumulative.startsWith(emitted)) { "Decoder revised an already emitted text prefix" }
        var end = cumulative.length
        if (!terminal) {
            while (end > emitted.length &&
                (cumulative[end - 1] == '\uFFFD' || Character.isHighSurrogate(cumulative[end - 1]))) end--
        }
        val next = cumulative.substring(0, end)
        val delta = next.substring(emitted.length)
        emitted = next
        return delta.takeIf { it.isNotEmpty() }
    }
}
