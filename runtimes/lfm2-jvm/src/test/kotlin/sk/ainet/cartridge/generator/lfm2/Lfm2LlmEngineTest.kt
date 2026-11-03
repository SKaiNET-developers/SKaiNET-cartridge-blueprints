package sk.ainet.cartridge.generator.lfm2

import sk.ainet.apps.klfm2.Lfm2ChatMessage
import sk.ainet.apps.klfm2.Lfm2GeneratedToken
import sk.ainet.apps.klfm2.Lfm2TextResult
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.toList
import sk.ainet.cartridge.profiles.llm.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.*

class Lfm2LlmEngineTest {
    private val prompt = listOf(LlmMessage(LlmRole.USER, "Hello"))

    private class Backend(
        override val contextLimit: Int = 64,
        var promptSize: Int = 2,
        val run: (Int, () -> Boolean, (Lfm2GeneratedToken) -> Unit) -> Lfm2TextResult = { _, _, emit ->
            emit(Lfm2GeneratedToken(4, 0, "A", false))
            emit(Lfm2GeneratedToken(5, 1, "AB", false))
            emit(Lfm2GeneratedToken(7, 2, "AB", true))
            result("AB", "eos", listOf(4, 5, 7))
        },
    ) : Lfm2GenerationBackend {
        val calls = AtomicInteger()
        val active = AtomicInteger()
        val peak = AtomicInteger()
        val cleaned = AtomicInteger()
        var encodes = 0
        var received: List<Lfm2ChatMessage>? = null
        override fun promptTokenCount(messages: List<Lfm2ChatMessage>): Int {
            encodes++
            return promptSize
        }
        override fun generate(messages: List<Lfm2ChatMessage>, maxOutputTokens: Int,
            isCancelled: () -> Boolean, onToken: (Lfm2GeneratedToken) -> Unit): Lfm2TextResult {
            received = messages
            calls.incrementAndGet()
            peak.accumulateAndGet(active.incrementAndGet(), ::maxOf)
            try { return run(maxOutputTokens, isCancelled, onToken) }
            finally { active.decrementAndGet(); cleaned.incrementAndGet() }
        }
    }

    companion object {
        private fun result(text: String, reason: String, ids: List<Int>) =
            Lfm2TextResult(intArrayOf(1, 2), ids.toIntArray(), text, reason, 2 + maxOf(0, ids.size - 1))
        private fun await(latch: CountDownLatch) = check(latch.await(5, TimeUnit.SECONDS)) { "Timed out waiting for test worker" }
        private fun waitForCancel(cancelled: () -> Boolean) {
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
            while (!cancelled()) {
                check(System.nanoTime() < deadline) { "Timed out waiting for cancellation" }
                Thread.sleep(1)
            }
        }
    }

    private suspend fun Lfm2LlmEngine.turn(maximum: Int = 4): List<LlmEvent> =
        session(LlmSessionConfig(maxOutputTokens = maximum)).use {
            it.prompt(prompt)
            it.events.toList()
        }

    private fun assertContract(events: List<LlmEvent>): LlmEvent.Final {
        assertEquals(LlmEvent.Ready, events.first())
        assertEquals(1, events.count { it is LlmEvent.Final || it is LlmEvent.Failed })
        val final = assertIs<LlmEvent.Final>(events.last())
        assertEquals(events.filterIsInstance<LlmEvent.Delta>().joinToString("") { it.text }, final.text)
        return final
    }

    @Test fun backendInfoAdvertisesActualCapacityAndRejectsOversizedGeneration() {
        Lfm2LlmEngine(Backend(contextLimit = 64), 8, "LFM2.5-350M (IREE Vulkan)").use { engine ->
            assertEquals("LFM2.5-350M (IREE Vulkan)", engine.info.name)
            assertEquals(64, engine.info.contextWindowTokens)
            assertFailsWith<IllegalArgumentException> { engine.session(LlmSessionConfig(maxOutputTokens = 65)) }
        }
    }

    @Test fun coldFlowCopiesMessagesAndReturnsExactProfileEvents() = runBlocking {
        val backend = Backend()
        Lfm2LlmEngine(backend).use { engine ->
            val session = engine.session(LlmSessionConfig(maxOutputTokens = 4))
            val mutable = prompt.toMutableList()
            session.prompt(mutable)
            mutable.clear()
            assertEquals(0, backend.calls.get())
            assertEquals(0, backend.encodes)
            val events = session.events.toList()
            assertEquals(listOf(LlmEvent.Ready, LlmEvent.Delta("A"), LlmEvent.Delta("B"),
                LlmEvent.Final("AB", LlmFinishReason.STOP, listOf(4, 5, 7))), events)
            assertContract(events)
            assertEquals("Hello", backend.received!!.single().content)
            assertEquals(1, backend.cleaned.get())
            assertNull(engine.info.cartridgeId)
            assertEquals(0f, engine.info.defaultTemperature)
        }
    }

    @Test fun onePromptOneCollectionAndCloseAreEnforced() = runBlocking {
        val backend = Backend()
        val engine = Lfm2LlmEngine(backend)
        val session = engine.session()
        assertFailsWith<IllegalStateException> { session.events.toList() }
        session.prompt(prompt)
        assertFailsWith<IllegalStateException> { session.prompt(prompt) }
        session.events.toList()
        assertFailsWith<IllegalStateException> { session.events.toList() }
        session.close(); session.close()
        assertFailsWith<IllegalStateException> { session.prompt(prompt) }
        val unused = engine.session()
        engine.close(); engine.close()
        assertFailsWith<IllegalStateException> { engine.session() }
        assertFailsWith<IllegalStateException> { unused.prompt(prompt) }
        assertEquals(1, backend.calls.get())
    }

    @Test fun utf8RevisionAndTerminalReplacementProduceAppendOnlyDeltas() = runBlocking {
        val backend = Backend(run = { _, _, emit ->
            emit(Lfm2GeneratedToken(1, 0, "A\uFFFD", false))
            emit(Lfm2GeneratedToken(2, 1, "Aé", false))
            emit(Lfm2GeneratedToken(3, 2, "Aé\uFFFD", false))
            result("Aé\uFFFD", "length", listOf(1, 2, 3))
        })
        Lfm2LlmEngine(backend).use { engine ->
            val events = engine.turn()
            assertEquals(listOf("A", "é", "\uFFFD"), events.filterIsInstance<LlmEvent.Delta>().map { it.text })
            assertEquals(LlmFinishReason.LENGTH, assertContract(events).finishReason)
        }
        val stream = StableByteLevelText()
        assertNull(stream.append("\uD83D", terminal = false))
        assertEquals("😀", stream.append("😀", terminal = false))
        assertNull(stream.append("😀", terminal = true))
    }

    @Test fun unexpectedDecoderRevisionFailsInsteadOfRewritingEmittedText() = runBlocking {
        val backend = Backend(run = { _, _, emit ->
            emit(Lfm2GeneratedToken(1, 0, "A", false))
            emit(Lfm2GeneratedToken(2, 1, "B", false))
            error("Must reject the non-prefix revision")
        })
        Lfm2LlmEngine(backend).use { engine ->
            val events = engine.turn()
            assertEquals(LlmEvent.Ready, events.first())
            assertEquals(listOf("A"), events.filterIsInstance<LlmEvent.Delta>().map { it.text })
            assertEquals(LlmErrorCode.GENERATION_FAILED, assertIs<LlmEvent.Failed>(events.last()).error.code)
            assertTrue(events.none { it is LlmEvent.Final })
            assertEquals(1, backend.cleaned.get())
        }
    }

    @Test fun unsupportedSamplingToolsAndRolesAreExplicitlyRejected() = runBlocking {
        val backend = Backend()
        Lfm2LlmEngine(backend).use { engine ->
            for (config in listOf(LlmSessionConfig(temperature = 0.2f), LlmSessionConfig(temperature = Float.NaN),
                LlmSessionConfig(topK = 5), LlmSessionConfig(topP = 0.9f), LlmSessionConfig(topP = Float.NaN),
                LlmSessionConfig(maxOutputTokens = 0), LlmSessionConfig(maxOutputTokens = 65))) {
                assertFailsWith<IllegalArgumentException> { engine.session(config) }
            }
            engine.session().use { session ->
                assertFailsWith<IllegalArgumentException> { session.prompt(emptyList()) }
                assertFailsWith<IllegalArgumentException> { session.prompt(prompt, listOf(LlmToolDef("x", "x"))) }
                assertFailsWith<IllegalArgumentException> { session.prompt(listOf(LlmMessage(LlmRole.TOOL, "x"))) }
                assertFailsWith<IllegalArgumentException> { session.prompt(prompt + LlmMessage(LlmRole.SYSTEM, "x")) }
            }
            engine.session(LlmSessionConfig(temperature = 0f, topK = 1, topP = 1f, seed = 42)).close()
            assertEquals(0, backend.calls.get())
        }
    }

    @Test fun contextOverflowIsProfileFailureBeforeAnyModelCall() = runBlocking {
        val backend = Backend(contextLimit = 8, promptSize = 7)
        Lfm2LlmEngine(backend, defaultMaxOutputTokens = 2).use { engine ->
            val events = engine.turn(maximum = 3)
            assertEquals(2, events.size)
            assertEquals(LlmEvent.Ready, events.first())
            assertEquals(LlmErrorCode.PROMPT_TOO_LONG, assertIs<LlmEvent.Failed>(events.last()).error.code)
            assertEquals(0, backend.calls.get())
            // The final selected token is not fed back: prompt + maximum - 1 fits exactly.
            assertContract(engine.turn(maximum = 2))
            assertEquals(1, backend.calls.get())
        }
    }

    @Test fun preCancelledSessionEndsWithoutEncodingOrGeneration() = runBlocking {
        val backend = Backend()
        Lfm2LlmEngine(backend).use { engine ->
            val session = engine.session()
            session.cancel()
            session.prompt(prompt)
            val final = assertContract(session.events.toList())
            assertEquals(LlmFinishReason.CANCELLED, final.finishReason)
            assertEquals("", final.text)
            assertEquals(emptyList(), final.tokens)
            assertEquals(0, backend.calls.get())
            assertEquals(0, backend.encodes)
        }
    }

    @Test fun crossThreadCancelPreservesPartialTextAndCleansRequest() = runBlocking {
        val started = CountDownLatch(1)
        val backend = Backend(run = { _, cancelled, emit ->
            emit(Lfm2GeneratedToken(4, 0, "A\uFFFD", false))
            started.countDown()
            waitForCancel(cancelled)
            result("A\uFFFD", "cancelled", listOf(4))
        })
        Lfm2LlmEngine(backend).use { engine ->
            val session = engine.session()
            session.prompt(prompt)
            val collecting = async(Dispatchers.Default) { session.events.toList() }
            withContext(Dispatchers.IO) { await(started) }
            val canceller = Thread { session.cancel() }
            canceller.start(); canceller.join(5000)
            assertFalse(canceller.isAlive)
            val final = assertContract(withTimeout(5000) { collecting.await() })
            assertEquals("A\uFFFD", final.text)
            assertEquals(LlmFinishReason.CANCELLED, final.finishReason)
            assertEquals(listOf(4), final.tokens)
            assertEquals(0, backend.active.get())
            assertEquals(1, backend.cleaned.get())
        }
    }

    @Test fun closeCancelsAnActiveSessionAndRejectsFutureSessions() = runBlocking {
        val started = CountDownLatch(1)
        val backend = Backend(run = { _, cancelled, _ ->
            started.countDown()
            waitForCancel(cancelled)
            result("", "cancelled", emptyList())
        })
        val engine = Lfm2LlmEngine(backend)
        val session = engine.session()
        session.prompt(prompt)
        val collecting = async(Dispatchers.Default) { session.events.toList() }
        withContext(Dispatchers.IO) { await(started) }
        engine.close()
        assertEquals(LlmFinishReason.CANCELLED, assertContract(withTimeout(5000) { collecting.await() }).finishReason)
        assertFailsWith<IllegalStateException> { engine.session() }
        assertEquals(1, backend.cleaned.get())
    }

    @Test fun modelFailureHasOneTerminalAndFreshSessionCanRetry() = runBlocking {
        var attempt = 0
        val backend = Backend(run = { _, _, emit ->
            if (attempt++ == 0) error("model failed")
            emit(Lfm2GeneratedToken(7, 0, "", true))
            result("", "eos", listOf(7))
        })
        Lfm2LlmEngine(backend).use { engine ->
            val failed = engine.turn()
            assertEquals(2, failed.size)
            assertEquals(LlmErrorCode.GENERATION_FAILED, assertIs<LlmEvent.Failed>(failed.last()).error.code)
            assertEquals(LlmFinishReason.STOP, assertContract(engine.turn()).finishReason)
            assertEquals(2, backend.cleaned.get())
        }
    }

    @Test fun consumerFailureAndCollectionCancellationReleaseTheEngine() = runBlocking {
        val backend = Backend()
        Lfm2LlmEngine(backend).use { engine ->
            val session = engine.session()
            session.prompt(prompt)
            val failure = IllegalStateException("consumer failed")
            val delivered = mutableListOf<LlmEvent>()
            val propagated = assertFailsWith<IllegalStateException> {
                session.events.collect {
                    delivered.add(it)
                    if (it is LlmEvent.Delta) throw failure
                }
            }
            // Coroutine stack-trace recovery may copy the exception and retain the original
            // as its cause. The consumer failure must propagate, never become a profile event.
            assertEquals(failure.message, propagated.message)
            assertTrue(generateSequence<Throwable>(propagated) { it.cause }.take(16).any { it === failure })
            assertTrue(delivered.none { it is LlmEvent.Final || it is LlmEvent.Failed })
            assertEquals(0, backend.active.get())
            assertEquals(1, backend.cleaned.get())
            assertContract(engine.turn())
        }

        val started = CountDownLatch(1)
        val waiting = Backend(run = { _, cancelled, _ ->
            started.countDown()
            waitForCancel(cancelled)
            result("", "cancelled", emptyList())
        })
        Lfm2LlmEngine(waiting).use { engine ->
            val session = engine.session()
            session.prompt(prompt)
            val collecting = launch(Dispatchers.Default) { session.events.collect() }
            withContext(Dispatchers.IO) { await(started) }
            collecting.cancelAndJoin()
            assertEquals(0, waiting.active.get())
            assertEquals(1, waiting.cleaned.get())
        }
    }

    @Test fun simultaneousSessionsSerializeSharedModelExecution() = runBlocking {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val entered = AtomicInteger()
        val backend = Backend(run = { _, _, _ ->
            if (entered.incrementAndGet() == 1) { started.countDown(); await(release) }
            result("", "eos", listOf(7))
        })
        Lfm2LlmEngine(backend).use { engine ->
            val first = async(Dispatchers.Default) { engine.turn() }
            withContext(Dispatchers.IO) { await(started) }
            val secondReady = CompletableDeferred<Unit>()
            val second = async(Dispatchers.Default) {
                val events = mutableListOf<LlmEvent>()
                engine.session().use { session ->
                    session.prompt(prompt)
                    session.events.collect { events.add(it); if (it == LlmEvent.Ready) secondReady.complete(Unit) }
                }
                events
            }
            withTimeout(5000) { secondReady.await() }
            assertEquals(1, backend.calls.get())
            release.countDown()
            assertContract(withTimeout(5000) { first.await() })
            assertContract(withTimeout(5000) { second.await() })
            assertEquals(1, backend.peak.get())
            assertEquals(2, backend.cleaned.get())
        }
    }
}
