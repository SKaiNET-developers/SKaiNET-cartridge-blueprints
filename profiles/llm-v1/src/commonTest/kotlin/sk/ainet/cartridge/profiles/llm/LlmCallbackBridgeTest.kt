package sk.ainet.cartridge.profiles.llm

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals

class LlmCallbackBridgeTest {

    private class ScriptedSession(private val script: List<LlmEvent>) : LlmSession {
        override suspend fun prompt(messages: List<LlmMessage>, tools: List<LlmToolDef>) = Unit
        override fun cancel() = Unit
        override fun close() = Unit
        override val events: Flow<LlmEvent> get() = flowOf(*script.toTypedArray())
    }

    @Test
    fun bridgePreservesOrderAndAlwaysEnds() = runTest {
        val session = ScriptedSession(
            listOf(
                LlmEvent.Ready,
                LlmEvent.Delta("Par"),
                LlmEvent.Delta("is"),
                LlmEvent.Final("Paris", LlmFinishReason.STOP, tokens = listOf(7, 11)),
            ),
        )
        val seen = mutableListOf<String>()
        val job = session.collectInto(
            this,
            object : LlmSessionCallback {
                override fun onReady() { seen += "ready" }
                override fun onDelta(text: String) { seen += "delta:$text" }
                override fun onFinal(text: String, finishReason: LlmFinishReason, tokens: List<Int>?) {
                    seen += "final:$text:$finishReason:${tokens?.size}"
                }
                override fun onError(error: LlmError) { seen += "error" }
                override fun onSessionEnded() { seen += "ended" }
            },
        )
        job.join()
        assertEquals(listOf("ready", "delta:Par", "delta:is", "final:Paris:STOP:2", "ended"), seen)
    }

    @Test
    fun failureStillEndsTheSession() = runTest {
        val session = ScriptedSession(
            listOf(
                LlmEvent.Ready,
                LlmEvent.Failed(LlmError(LlmErrorCode.GENERATION_FAILED, "boom", recoverable = false)),
            ),
        )
        val seen = mutableListOf<String>()
        session.collectInto(
            this,
            object : LlmSessionCallback {
                override fun onReady() { seen += "ready" }
                override fun onError(error: LlmError) { seen += "error:${error.code}" }
                override fun onSessionEnded() { seen += "ended" }
            },
        ).join()
        assertEquals(listOf("ready", "error:GENERATION_FAILED", "ended"), seen)
    }
}
