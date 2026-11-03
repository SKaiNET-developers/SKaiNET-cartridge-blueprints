package sk.ainet.cartridge.generator.lfm2.worker

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import sk.ainet.cartridge.profiles.llm.*
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

private data class Failure(val status: Int, val message: String)
private class ActiveRequest(val id: Long) {
    val cancelled = AtomicBoolean(false)
    val session = AtomicReference<LlmSession?>()
    val forcedFailure = AtomicReference<Failure?>()
    fun cancel() { cancelled.set(true); session.get()?.cancel() }
    fun reject(failure: Failure) { forcedFailure.compareAndSet(null, failure); cancel() }
}

/** One loaded model, one active request, independent command reader and generation executor. */
internal class WorkerService(
    private val runtime: WorkerRuntime,
    input: InputStream,
    output: OutputStream,
    private val shutdownSeconds: Long = 20,
) {
    private val reader = FrameReader(input)
    private val writer = FrameWriter(output)
    private val lock = Any()
    private var active: ActiveRequest? = null
    private val worker = Executors.newSingleThreadExecutor { task ->
        Thread(task, "lfm2-native-generation").apply { isDaemon = true }
    }
    private val transportFailure = AtomicReference<Throwable?>()

    fun run(): Int {
        var failed = false
        try {
            writer.write(Frame(Type.LOADED, 0, runtime.metadata.toString().toByteArray(Charsets.UTF_8)))
            commandLoop@ while (true) {
                val frame = try { reader.read() } catch (error: FrameException) {
                    reject(error.requestId, Failure(Status.INVALID, error.message ?: "Invalid frame"))
                    failed = true
                    break
                } ?: break
                when (frame.type) {
                    Type.RUN -> {
                        if (frame.requestId == 0L) {
                            reject(0, Failure(Status.INVALID, "RUN requires a nonzero request ID"))
                            continue
                        }
                        val current = synchronized(lock) { active }
                        if (current != null) {
                            reject(frame.requestId, Failure(Status.BAD_STATE, "Another RUN is active"))
                            continue
                        }
                        val request = try {
                            RunRequest.parse(frame.payload, runtime.defaultMaximum, runtime.maximum)
                        } catch (error: Exception) {
                            reject(frame.requestId, Failure(Status.INVALID, error.message ?: "Invalid request"))
                            continue
                        }
                        val owned = ActiveRequest(frame.requestId)
                        synchronized(lock) { check(active == null); active = owned }
                        worker.execute { generate(owned, request) }
                    }
                    Type.CANCEL -> {
                        if (frame.payload.isNotEmpty() || frame.requestId == 0L) {
                            reject(frame.requestId, Failure(Status.INVALID, "CANCEL requires an ID and empty payload"))
                        } else {
                            synchronized(lock) { active?.takeIf { it.id == frame.requestId } }?.cancel()
                        }
                    }
                    Type.STOP -> {
                        if (frame.requestId != 0L || frame.payload.isNotEmpty()) {
                            reject(frame.requestId, Failure(Status.INVALID, "STOP requires ID zero and empty payload"))
                        } else break@commandLoop
                    }
                    else -> reject(frame.requestId, Failure(Status.INVALID, "Unknown command type"))
                }
                if (transportFailure.get() != null) { failed = true; break }
            }
        } catch (error: Exception) {
            failed = true
            System.err.println("Worker transport failed: ${error.message}")
        } finally {
            synchronized(lock) { active }?.cancel()
            worker.shutdown()
            if (!worker.awaitTermination(shutdownSeconds, TimeUnit.SECONDS)) {
                worker.shutdownNow()
                failed = true
                System.err.println("Worker generation did not finish before shutdown deadline")
            }
            runtime.close()
        }
        return if (failed || transportFailure.get() != null) 1 else 0
    }

    private fun reject(id: Long, failure: Failure) {
        val current = synchronized(lock) { active?.takeIf { it.id == id } }
        // A duplicate active ID must not produce an early DONE followed by that same
        // request's remaining output. Fail its single terminal response instead.
        if (current != null) current.reject(failure)
        else writer.error(id, failure.status, failure.message)
    }

    private fun generate(owned: ActiveRequest, request: RunRequest) {
        var failure: Failure? = null
        var terminal: LlmEvent.Final? = null
        var session: LlmSession? = null
        var phase = Status.INVALID
        try {
            if (!owned.cancelled.get()) {
                val prompt = runtime.promptIds(request.messages)
                require(prompt.isNotEmpty()) { "Empty encoded prompt" }
                require(prompt.size.toLong() + request.maximum - 1 <= runtime.engine.info.contextWindowTokens) {
                    "Prompt and output budget exceed model context"
                }
                val generationSession = runtime.engine.session(LlmSessionConfig(maxOutputTokens = request.maximum,
                    temperature = 0f, topK = 1, topP = 1f))
                session = generationSession
                owned.session.set(generationSession)
                if (owned.cancelled.get()) generationSession.cancel()
                phase = Status.BACKEND
                var ready = false
                val text = StringBuilder()
                runBlocking {
                    generationSession.prompt(request.messages)
                    generationSession.events.collect { event ->
                        check(terminal == null && failure == null) { "Profile emitted after its terminal event" }
                        val item = when (event) {
                            LlmEvent.Ready -> {
                                check(!ready) { "Repeated Ready event" }; ready = true
                                buildJsonObject {
                                    put("type", "ready"); put("prompt_ids", JsonArray(prompt.map(::JsonPrimitive)))
                                }
                            }
                            is LlmEvent.Delta -> {
                                check(ready) { "Delta before Ready" }
                                text.append(event.text)
                                buildJsonObject { put("type", "delta"); put("text", event.text) }
                            }
                            is LlmEvent.Final -> {
                                check(ready && event.text == text.toString() && event.tokens != null) {
                                    "Profile final violates text/token contract"
                                }
                                terminal = event
                                null
                            }
                            is LlmEvent.Failed -> {
                                failure = Failure(if (event.error.code == LlmErrorCode.PROMPT_TOO_LONG) Status.INVALID
                                    else Status.BACKEND, "${event.error.code}: ${event.error.message}")
                                null
                            }
                            is LlmEvent.ToolCall -> error("Unexpected tool event")
                        }
                        if (item != null && !owned.cancelled.get()) {
                            writer.write(Frame(Type.PARTIAL, owned.id, item.toString().toByteArray(Charsets.UTF_8)))
                        }
                    }
                }
                check(terminal != null || failure != null) { "Profile completed without a terminal event" }
            }
        } catch (error: Exception) {
            failure = Failure(phase, error.message ?: "Request execution failed")
        } finally {
            try { session?.close() } catch (error: Exception) {
                failure = Failure(Status.BACKEND, error.message ?: "Session cleanup failed")
            }
            owned.session.set(null)
            synchronized(lock) {
                try {
                    val final = terminal
                    val problem = owned.forcedFailure.get()
                        ?: if (owned.cancelled.get() || final?.finishReason == LlmFinishReason.CANCELLED)
                            Failure(Status.CANCELLED, "Generation cancelled") else failure
                    if (problem != null) writer.error(owned.id, problem.status, problem.message)
                    else if (final != null) {
                        val json = buildJsonObject {
                            put("type", "final"); put("text", final.text)
                            put("finish_reason", final.finishReason.name)
                            put("tokens", JsonArray(requireNotNull(final.tokens).map(::JsonPrimitive)))
                        }
                        writer.finish(Frame(Type.FINAL, owned.id, json.toString().toByteArray(Charsets.UTF_8)))
                    } else writer.error(owned.id, Status.BACKEND, "Request ended without a result")
                } catch (error: Exception) {
                    transportFailure.compareAndSet(null, error)
                    System.err.println("Worker output failed: ${error.message}")
                } finally {
                    check(active === owned)
                    active = null
                }
            }
        }
    }
}
