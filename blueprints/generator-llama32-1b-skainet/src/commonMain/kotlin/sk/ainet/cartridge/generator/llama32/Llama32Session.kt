package sk.ainet.cartridge.generator.llama32

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import sk.ainet.apps.llm.PrefillStrategy
import sk.ainet.apps.llm.generateUntilStop
import sk.ainet.cartridge.profiles.llm.LlmError
import sk.ainet.cartridge.profiles.llm.LlmErrorCode
import sk.ainet.cartridge.profiles.llm.LlmEvent
import sk.ainet.cartridge.profiles.llm.LlmFinishReason
import sk.ainet.cartridge.profiles.llm.LlmMessage
import sk.ainet.cartridge.profiles.llm.LlmSession
import sk.ainet.cartridge.profiles.llm.LlmSessionConfig
import sk.ainet.cartridge.profiles.llm.LlmToolDef
import kotlin.math.min
import kotlin.random.Random

/** Thrown from the token callback to stop decode on [LlmSession.cancel]; never escapes. */
private class CancelSignal : RuntimeException()

/**
 * One generation turn, mirroring the SmolLM2 blueprint's session exactly (same llm/v1 event
 * mapping, same one-shot shape): [events] is cold, collecting it runs the decode on
 * [Dispatchers.Default], and the flow completes after the terminal event. One prompt and one
 * collection per session — multi-turn chat replays history into a fresh session, the honest
 * shape for a KV-cache-per-engine decoder.
 */
internal class Llama32Session(
    private val engine: Llama32Engine,
    private val config: LlmSessionConfig,
) : LlmSession {

    private var rendered: String? = null
    private var collected = false
    private var cancelRequested = false

    override suspend fun prompt(messages: List<LlmMessage>, tools: List<LlmToolDef>) {
        require(tools.isEmpty()) {
            "This blueprint declares supports_tool_calling=false; Llama 3.2's tool grammar is a " +
                "separate capability the cartridge does not expose yet"
        }
        require(messages.isNotEmpty()) { "at least one message" }
        check(rendered == null) { "one prompt per session; open a new session for the next turn" }
        rendered = renderLlama3(messages)
    }

    override fun cancel() {
        cancelRequested = true
    }

    override fun close() {}

    override val events: Flow<LlmEvent> = channelFlow {
        val text = checkNotNull(rendered) { "call prompt() before collecting events" }
        check(!collected) { "events already collected; one generation per session" }
        collected = true

        val promptTokens = engine.tokenizer.encode(text)
        val contextBudget =
            Llama32GeneratorCartridge.CONTEXT_WINDOW_TOKENS - promptTokens.size
        if (contextBudget <= 0) {
            send(
                LlmEvent.Failed(
                    LlmError(
                        LlmErrorCode.PROMPT_TOO_LONG,
                        "prompt is ${promptTokens.size} tokens, context window is " +
                            "${Llama32GeneratorCartridge.CONTEXT_WINDOW_TOKENS}",
                        recoverable = true,
                    ),
                ),
            )
            return@channelFlow
        }
        send(LlmEvent.Ready)

        val maxTokens = min(
            config.maxOutputTokens ?: Llama32GeneratorCartridge.DEFAULT_MAX_OUTPUT_TOKENS,
            contextBudget,
        )
        val generated = StringBuilder()
        val generatedTokens = mutableListOf<Int>()
        try {
            // Each turn starts from a clean KV cache and position 0 — generateUntilStop does
            // not reset, and a previous session's state would silently corrupt this decode.
            engine.runtime.reset()
            val result = engine.runtime.generateUntilStop(
                prompt = promptTokens,
                maxTokens = maxTokens,
                eosTokenIds = Llama32GeneratorCartridge.STOP_TOKEN_IDS,
                temperature = config.temperature ?: Llama32GeneratorCartridge.DEFAULT_TEMPERATURE,
                topK = config.topK ?: 0,
                topP = config.topP ?: Llama32GeneratorCartridge.DEFAULT_TOP_P,
                random = config.seed?.let { Random(it) } ?: Random.Default,
                // Batched prefill: 3–10× faster prompt ingestion; upstream pins its
                // equivalence with the autoregressive path.
                prefillStrategy = PrefillStrategy.Batched(),
                onToken = { id ->
                    if (cancelRequested) throw CancelSignal()
                    if (id !in Llama32GeneratorCartridge.STOP_TOKEN_IDS) {
                        generatedTokens += id
                        val piece = engine.tokenizer.decodeToken(id)
                        generated.append(piece)
                        trySend(LlmEvent.Delta(piece))
                    }
                },
            )
            send(
                LlmEvent.Final(
                    text = generated.toString(),
                    finishReason = if (result.stoppedByEos) LlmFinishReason.STOP else LlmFinishReason.LENGTH,
                    tokens = generatedTokens,
                ),
            )
        } catch (signal: CancelSignal) {
            send(
                LlmEvent.Final(
                    text = generated.toString(),
                    finishReason = LlmFinishReason.CANCELLED,
                    tokens = generatedTokens,
                ),
            )
        } catch (failure: Exception) {
            send(
                LlmEvent.Failed(
                    LlmError(
                        LlmErrorCode.GENERATION_FAILED,
                        failure.message ?: "generation failed",
                        recoverable = false,
                    ),
                ),
            )
        }
    }
        .buffer(Channel.UNLIMITED)
        .flowOn(Dispatchers.Default)
}
