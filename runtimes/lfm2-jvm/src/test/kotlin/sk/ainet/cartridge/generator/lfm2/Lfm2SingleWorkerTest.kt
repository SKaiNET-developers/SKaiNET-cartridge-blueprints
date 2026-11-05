package sk.ainet.cartridge.generator.lfm2

import sk.ainet.apps.klfm2.Lfm2ChatMessage
import sk.ainet.apps.klfm2.Lfm2GeneratedToken
import sk.ainet.apps.klfm2.Lfm2TextResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.toList
import sk.ainet.cartridge.profiles.llm.*
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.*

/** Run only by singleDefaultWorkerTest, with an explicitly supplied test runtime classpath. */
class Lfm2SingleWorkerTest {
    @Test fun defaultCollectorCompletesWithSynchronousBackpressureAndOneCpuWorker() {
        val classpath = checkNotNull(System.getProperty("lfm2.singleWorker.classpath"))
        val output = Files.createTempFile("lfm2-single-worker-", ".log")
        val child = ProcessBuilder(
            Path.of(System.getProperty("java.home"), "bin", "java").toString(),
            "--enable-preview", "--add-modules", "jdk.incubator.vector",
            "-Dkotlinx.coroutines.scheduler.core.pool.size=1",
            "-cp", classpath, "sk.ainet.cartridge.generator.lfm2.Lfm2SingleWorkerProbe",
        ).redirectErrorStream(true).redirectOutput(output.toFile()).start()
        try {
            val completed = child.waitFor(8, TimeUnit.SECONDS)
            if (!completed) {
                child.destroyForcibly()
                child.waitFor(3, TimeUnit.SECONDS)
            }
            val log = Files.readString(output)
            assertTrue(completed, "Single-worker probe exceeded its OS-level timeout; output retained at $output: $log")
            assertEquals(0, child.exitValue(), "Single-worker probe failed; output retained at $output: $log")
            assertTrue(log.contains("SINGLE_DEFAULT_WORKER_PASSED"), log)
            println(log.trim())
        } finally {
            // Coroutine timeout cannot bound a starved producer's join; the test owns and
            // terminates this exact child process even if test-thread cancellation occurs.
            if (child.isAlive) child.destroyForcibly()
        }
    }
}

/** Separate process so a scheduler regression cannot hang the Gradle test worker. */
object Lfm2SingleWorkerProbe {
    @JvmStatic fun main(args: Array<String>) {
        check(System.getProperty("kotlinx.coroutines.scheduler.core.pool.size") == "1")
        runBlocking {
            var cleaned = false
            val backend = object : Lfm2GenerationBackend {
                override val contextLimit = 64
                override fun promptTokenCount(messages: List<Lfm2ChatMessage>) = 2
                override fun generate(messages: List<Lfm2ChatMessage>, maxOutputTokens: Int,
                    isCancelled: () -> Boolean, onToken: (Lfm2GeneratedToken) -> Unit): Lfm2TextResult {
                    try {
                        onToken(Lfm2GeneratedToken(4, 0, "A", false))
                        onToken(Lfm2GeneratedToken(5, 1, "AB", false))
                        return Lfm2TextResult(intArrayOf(1, 2), intArrayOf(4, 5), "AB", "length", 3)
                    } finally { cleaned = true }
                }
            }
            Lfm2LlmEngine(backend).use { engine ->
                val events = withContext(Dispatchers.Default) {
                    engine.session(LlmSessionConfig(maxOutputTokens = 2)).use { session ->
                        session.prompt(listOf(LlmMessage(LlmRole.USER, "Hello")))
                        session.events.toList()
                    }
                }
                check(events == listOf(LlmEvent.Ready, LlmEvent.Delta("A"), LlmEvent.Delta("B"),
                    LlmEvent.Final("AB", LlmFinishReason.LENGTH, listOf(4, 5))))
                check(cleaned)
            }
        }
        println("SINGLE_DEFAULT_WORKER_PASSED: append-only deltas, final text, and cleanup")
    }
}
