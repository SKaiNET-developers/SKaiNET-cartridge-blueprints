package sk.ainet.cartridge.profiles.llm

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/**
 * The callback layer of `llm/v1` (ADR-005: two layers, one artifact; ADR-011 reuses asr/v1's
 * bridge conventions). Provided so callback-structured consumers don't each reinvent the
 * Flow→callback bridging and its cancellation edge cases privately.
 *
 * The Flow contract ([LlmSession.events]) stays the normative core; this is a bridge, not a
 * second source of truth.
 */
public interface LlmSessionCallback {
    public fun onReady() {}
    public fun onDelta(text: String) {}
    public fun onToolCall(json: String) {}
    public fun onFinal(text: String, finishReason: LlmFinishReason, tokens: List<Int>?) {}
    public fun onError(error: LlmError) {}
    /** Always invoked last, after the terminal event or collection failure. */
    public fun onSessionEnded() {}
}

/**
 * Collect this session's events into [callback] on [scope]. Returns the collection [Job];
 * cancelling it detaches the callback without cancelling the session itself — use
 * [LlmSession.cancel] for that.
 */
public fun LlmSession.collectInto(scope: CoroutineScope, callback: LlmSessionCallback): Job =
    scope.launch {
        try {
            events.collect { event ->
                when (event) {
                    is LlmEvent.Ready -> callback.onReady()
                    is LlmEvent.Delta -> callback.onDelta(event.text)
                    is LlmEvent.ToolCall -> callback.onToolCall(event.json)
                    is LlmEvent.Final -> callback.onFinal(event.text, event.finishReason, event.tokens)
                    is LlmEvent.Failed -> callback.onError(event.error)
                }
            }
        } finally {
            callback.onSessionEnded()
        }
    }
