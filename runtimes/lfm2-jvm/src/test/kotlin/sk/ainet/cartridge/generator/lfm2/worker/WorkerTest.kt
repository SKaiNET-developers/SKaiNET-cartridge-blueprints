package sk.ainet.cartridge.generator.lfm2.worker

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.*
import org.junit.jupiter.api.Timeout
import sk.ainet.cartridge.profiles.llm.*
import java.io.*
import java.nio.ByteBuffer
import java.util.concurrent.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.*

@Timeout(15)
class WorkerTest {
    private val request = """{"messages":[{"role":"user","content":"Hello"}],"max_output_tokens":2}""".toByteArray()

    private class Engine(
        val script: suspend FlowCollector<LlmEvent>.(Int, AtomicBoolean) -> Unit = { _, cancelled ->
            emit(LlmEvent.Ready)
            if (cancelled.get()) emit(LlmEvent.Final("", LlmFinishReason.CANCELLED, emptyList()))
            else { emit(LlmEvent.Delta("A")); emit(LlmEvent.Final("A", LlmFinishReason.STOP, listOf(4, 7))) }
        },
    ) : LlmEngine {
        val calls = AtomicInteger()
        val closedSessions = AtomicInteger()
        val closes = AtomicInteger()
        override val info = LlmEngineInfo(null, "test", "1", 64, 4, false, false, 0f, 1, 1f)
        override fun session(config: LlmSessionConfig): LlmSession {
            val index = calls.incrementAndGet()
            return object : LlmSession {
                val cancelled = AtomicBoolean(false)
                val closed = AtomicBoolean(false)
                override suspend fun prompt(messages: List<LlmMessage>, tools: List<LlmToolDef>) { check(messages.isNotEmpty()) }
                override val events: Flow<LlmEvent> = flow { script(index, cancelled) }
                override fun cancel() { cancelled.set(true) }
                override fun close() { if (closed.compareAndSet(false, true)) closedSessions.incrementAndGet() }
            }
        }
        override fun close() { closes.incrementAndGet() }
    }

    private class Harness(engine: Engine, encode: (List<LlmMessage>) -> IntArray = { intArrayOf(1, 2) }) : AutoCloseable {
        private val commands = PipedOutputStream()
        private val serviceInput = PipedInputStream(commands, 65536)
        private val responseOutput = PipedOutputStream()
        private val responses = PipedInputStream(responseOutput, 65536)
        val writer = FrameWriter(commands)
        private val reader = FrameReader(responses)
        private val reads = Executors.newSingleThreadExecutor { Thread(it, "test-frame-reader").apply { isDaemon = true } }
        private val service = Executors.newSingleThreadExecutor { Thread(it, "test-worker-server").apply { isDaemon = true } }
        private val result = service.submit<Int> {
            WorkerService(WorkerRuntime(engine, encode, buildJsonObject { put("test", true) }, 2, 4),
                serviceInput, responseOutput, shutdownSeconds = 3).run()
        }
        fun read(): Frame = requireNotNull(reads.submit<Frame?> { reader.read() }.get(5, TimeUnit.SECONDS))
        fun untilDone(id: Long): List<Frame> {
            val result = mutableListOf<Frame>()
            repeat(20) {
                val frame = read()
                assertEquals(id, frame.requestId)
                result += frame
                if (frame.type == Type.DONE) return result
            }
            error("Missing DONE")
        }
        override fun close() {
            try {
                if (!result.isDone) writer.write(Frame(Type.STOP, 0))
                assertEquals(0, result.get(5, TimeUnit.SECONDS))
            } finally {
                commands.close(); serviceInput.close(); responses.close(); responseOutput.close()
                reads.shutdownNow(); service.shutdownNow()
            }
        }
    }

    private fun assertError(frames: List<Frame>, expected: Int) {
        assertEquals(listOf(Type.ERROR, Type.DONE), frames.map { it.type })
        assertEquals(expected, ByteBuffer.wrap(frames.first().payload).int)
        assertTrue(utf8(frames.first().payload.copyOfRange(4, frames.first().payload.size)).isNotEmpty())
        assertTrue(frames.last().payload.isEmpty())
    }

    @Test fun loadedEngineServesRepeatedRequestsWithIndependentSessionsAndHighBitIds() {
        val engine = Engine()
        Harness(engine).use { wire ->
            assertEquals(Type.LOADED, wire.read().type)
            for (id in listOf(1L, Long.MIN_VALUE)) {
                wire.writer.write(Frame(Type.RUN, id, request))
                val frames = wire.untilDone(id)
                assertEquals(listOf(Type.PARTIAL, Type.PARTIAL, Type.FINAL, Type.DONE), frames.map { it.type })
                assertEquals(setOf("type", "prompt_ids"), strictObject(frames.first().payload).keys)
                val final = strictObject(frames[2].payload)
                assertEquals("STOP", final.getValue("finish_reason").jsonPrimitive.content)
                assertEquals(listOf(4, 7), final.getValue("tokens").jsonArray.map { it.jsonPrimitive.int })
            }
            assertEquals(2, engine.calls.get())
            assertEquals(2, engine.closedSessions.get())
        }
        assertEquals(1, engine.closes.get())
    }

    @Test fun cancellationAfterReadyIsErrorThenDoneAndLateCancelDoesNotTouchNextRun() {
        val allowSecond = AtomicBoolean(false)
        val engine = Engine { index, cancelled ->
            emit(LlmEvent.Ready)
            if (index == 1) {
                while (!cancelled.get()) delay(1)
                emit(LlmEvent.Final("", LlmFinishReason.CANCELLED, emptyList()))
            } else {
                while (!allowSecond.get() && !cancelled.get()) delay(1)
                check(!cancelled.get()) { "Late cancellation reached a different request" }
                emit(LlmEvent.Delta("A")); emit(LlmEvent.Final("A", LlmFinishReason.STOP, listOf(4, 7)))
            }
        }
        Harness(engine).use { wire ->
            wire.read()
            wire.writer.write(Frame(Type.RUN, 1, request))
            assertEquals(Type.PARTIAL, wire.read().type)
            wire.writer.write(Frame(Type.CANCEL, 1))
            assertError(wire.untilDone(1), Status.CANCELLED)
            wire.writer.write(Frame(Type.RUN, 2, request))
            assertEquals(Type.PARTIAL, wire.read().type)
            wire.writer.write(Frame(Type.CANCEL, 1))
            wire.writer.write(Frame(255, 99)) // Ordered control-reader acknowledgment.
            assertError(wire.untilDone(99), Status.INVALID)
            allowSecond.set(true)
            assertEquals(listOf(Type.PARTIAL, Type.FINAL, Type.DONE), wire.untilDone(2).map { it.type })
        }
        assertEquals(2, engine.closedSessions.get())
    }

    @Test fun cancellationBeforeSessionPublicationIsRetained() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val engine = Engine()
        Harness(engine) {
            entered.countDown()
            check(release.await(5, TimeUnit.SECONDS))
            intArrayOf(1, 2)
        }.use { wire ->
            wire.read()
            wire.writer.write(Frame(Type.RUN, 1, request))
            assertTrue(entered.await(5, TimeUnit.SECONDS))
            wire.writer.write(Frame(Type.CANCEL, 1))
            wire.writer.write(Frame(255, 99))
            assertError(wire.untilDone(99), Status.INVALID)
            release.countDown()
            assertError(wire.untilDone(1), Status.CANCELLED)
        }
    }

    @Test fun busyRequestAndDuplicateActiveIdHaveOneUnambiguousTerminal() {
        val engine = Engine { _, cancelled ->
            emit(LlmEvent.Ready)
            while (!cancelled.get()) delay(1)
            emit(LlmEvent.Final("", LlmFinishReason.CANCELLED, emptyList()))
        }
        Harness(engine).use { wire ->
            wire.read(); wire.writer.write(Frame(Type.RUN, 1, request)); wire.read()
            wire.writer.write(Frame(Type.RUN, 2, request))
            assertError(wire.untilDone(2), Status.BAD_STATE)
            wire.writer.write(Frame(Type.RUN, 1, request))
            assertError(wire.untilDone(1), Status.BAD_STATE)
        }
        assertEquals(1, engine.calls.get())
    }

    @Test fun invalidRequestsDoNotPoisonTheLoadedEngine() {
        val engine = Engine()
        Harness(engine).use { wire ->
            wire.read()
            val invalid = listOf(
                """{"messages":[],"messages":[]}""".toByteArray(),
                """{"messages":[],"\u006dessages":[]}""".toByteArray(),
                """{"messages":[{"role":"user","content":"x","content":"y"}]}""".toByteArray(),
                """{"messages":[],"decoder_policy":"CHECKPOINT"}""".toByteArray(),
                """{"messages":[{"role":"user","content":"x"}],"temperature":1}""".toByteArray(),
                """{"messages":[{"role":"user","content":"x"}],"max_output_tokens":"2"}""".toByteArray(),
                """{"messages":[{"role":"user","content":"\uD800"}]}""".toByteArray(),
                byteArrayOf(0xC0.toByte(), 0xAF.toByte()),
            )
            invalid.forEachIndexed { index, bytes ->
                wire.writer.write(Frame(Type.RUN, index + 1L, bytes))
                assertError(wire.untilDone(index + 1L), Status.INVALID)
            }
            assertEquals(0, engine.calls.get())
            wire.writer.write(Frame(Type.RUN, 20, request))
            assertTrue(wire.untilDone(20).any { it.type == Type.FINAL })
        }
    }

    @Test fun backendFailureAndPromptOverflowUseAbiErrorsWithoutSuccessFinal() {
        Harness(Engine { _, _ ->
            emit(LlmEvent.Ready)
            emit(LlmEvent.Failed(LlmError(LlmErrorCode.GENERATION_FAILED, "test failure", true)))
        }).use { wire ->
            wire.read(); wire.writer.write(Frame(Type.RUN, 1, request)); wire.read()
            assertError(wire.untilDone(1), Status.BACKEND)
        }
        val engine = Engine()
        Harness(engine) { IntArray(64) { 1 } }.use { wire ->
            wire.read(); wire.writer.write(Frame(Type.RUN, 1, request))
            assertError(wire.untilDone(1), Status.INVALID)
            assertEquals(0, engine.calls.get())
        }
    }

    @Test fun frameReaderBoundsAllocationsAndRejectsTruncation() {
        val tooLarge = ByteBuffer.allocate(13).put(Type.RUN.toByte()).putLong(19).putInt(MAX_PAYLOAD + 1).array()
        val error = assertFailsWith<FrameException> { FrameReader(ByteArrayInputStream(tooLarge)).read() }
        assertEquals(19L, error.requestId)
        val truncated = ByteBuffer.allocate(14).put(Type.RUN.toByte()).putLong(2).putInt(3).put(1).array()
        assertFailsWith<FrameException> { FrameReader(ByteArrayInputStream(truncated)).read() }
        assertFailsWith<FrameException> { FrameReader(ByteArrayInputStream(byteArrayOf(1))).read() }
        assertNull(FrameReader(ByteArrayInputStream(byteArrayOf())).read())
    }
}
