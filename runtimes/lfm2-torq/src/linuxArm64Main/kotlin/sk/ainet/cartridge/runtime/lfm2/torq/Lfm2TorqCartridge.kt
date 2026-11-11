package sk.ainet.cartridge.runtime.lfm2.torq

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.free
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.ptr
import kotlinx.cinterop.usePinned
import platform.posix.R_OK
import platform.posix.SEEK_END
import platform.posix.SEEK_SET
import platform.posix.access
import platform.posix.fclose
import platform.posix.fopen
import platform.posix.fread
import platform.posix.fseek
import platform.posix.ftell
import platform.posix.pthread_mutex_destroy
import platform.posix.pthread_mutex_init
import platform.posix.pthread_mutex_lock
import platform.posix.pthread_mutex_t
import platform.posix.pthread_mutex_unlock
import sk.ainet.cartridge.generator.lfm25.torq.DecodeStepPorts
import sk.ainet.cartridge.generator.lfm25.torq.DecodeStepRunner
import sk.ainet.cartridge.generator.lfm25.torq.EmbeddingRows
import sk.ainet.cartridge.generator.lfm25.torq.GreedyDecoder
import sk.ainet.cartridge.generator.lfm25.torq.StepControls
import sk.ainet.cartridge.runtime.lfm2.torq.iree.HostTensor
import sk.ainet.cartridge.runtime.lfm2.torq.iree.IreeRuntime
import sk.ainet.cartridge.runtime.lfm2.torq.iree.IreeSession

/** A failure with the ABI status it maps to. */
internal class CartridgeException(val status: Int, message: String) : Exception(message)

/** The pack_dir files this runtime reads (paths from the blueprint's outputs). */
internal class PackFiles(val dir: String) {
    val module = "$dir/artifacts/model/decode_step.vmfb"
    val ioTable = "$dir/artifacts/model/io.txt"
    val exportManifest = "$dir/artifacts/model/export.json"
    val embedding = "$dir/artifacts/weights/embed.bf16.bin"
    val generationConfig = "$dir/artifacts/tokenizer/generation_config.json"
    val descriptor = "$dir/descriptor.json"

    val required: List<String> get() = listOf(descriptor, module, ioTable, exportManifest, embedding, generationConfig)
}

@OptIn(ExperimentalForeignApi::class)
internal fun readable(path: String): Boolean = access(path, R_OK) == 0

@OptIn(ExperimentalForeignApi::class)
internal fun readBytes(path: String): ByteArray {
    val f = fopen(path, "rb") ?: throw CartridgeException(Status.LOAD, "cannot open $path")
    try {
        fseek(f, 0, SEEK_END)
        val size = ftell(f)
        if (size < 0 || size > Int.MAX_VALUE) throw CartridgeException(Status.LOAD, "cannot size $path")
        fseek(f, 0, SEEK_SET)
        val bytes = ByteArray(size.toInt())
        if (bytes.isNotEmpty()) {
            val n = bytes.usePinned { fread(it.addressOf(0), 1u, bytes.size.convert(), f) }
            if (n.toLong() != size) throw CartridgeException(Status.LOAD, "short read of $path")
        }
        return bytes
    } finally {
        fclose(f)
    }
}

/** RoPE base from export.json's `dimensions.rope_theta`. */
internal fun ropeTheta(exportManifest: String): Double =
    Regex(""""rope_theta"\s*:\s*([0-9.eE+-]+)""").find(exportManifest)?.groupValues?.get(1)?.toDouble()
        ?: throw CartridgeException(Status.LOAD, "export.json: no dimensions.rope_theta")

/** The checkpoint's end-of-sequence ids from generation_config.json (`eos_token_id`, a number or a list). */
internal fun stopTokens(generationConfig: String): Set<Int> {
    val m = Regex(""""eos_token_id"\s*:\s*(\[[^\]]*]|-?\d+)""").find(generationConfig)
        ?: throw CartridgeException(Status.LOAD, "generation_config.json: no eos_token_id")
    val ids = Regex("""-?\d+""").findAll(m.groupValues[1]).map { it.value.toInt() }.toSet()
    if (ids.isEmpty()) throw CartridgeException(Status.LOAD, "generation_config.json: empty eos_token_id")
    return ids
}

/**
 * One opened cartridge: the decode step loaded once on the torq device, the host-side tables, and a lock that
 * serializes the step across sessions (one IREE session, one NPU).
 */
@OptIn(ExperimentalForeignApi::class)
internal class Lfm2TorqCartridge(val files: PackFiles) {
    val ports: DecodeStepPorts = DecodeStepPorts.parse(readBytes(files.ioTable).decodeToString())
    private val controls = StepControls(ports.headDim, ropeTheta(readBytes(files.exportManifest).decodeToString()), ports.capacity)
    private val embedding = EmbeddingRows(ports.vocabulary, ports.width, readBytes(files.embedding))
    val stopTokens: Set<Int> = stopTokens(readBytes(files.generationConfig).decodeToString())
    private val runtime = IreeRuntime()
    private val module: IreeSession = try {
        runtime.session("torq", files.module)
    } catch (e: Throwable) {
        runtime.close()
        throw CartridgeException(Status.LOAD, e.message ?: "cannot load ${files.module}")
    }
    private val mutex = nativeHeap.alloc<pthread_mutex_t>().also { pthread_mutex_init(it.ptr, null) }
    val errors = ErrorSlot()
    var openSessions: Int = 0

    private val runner = DecodeStepRunner { inputs ->
        val tensors = ports.inputs.map { p ->
            val values = inputs[p.role] ?: error("no value for ${p.role}")
            HostTensor.bf16(LongArray(p.shape.size) { p.shape[it].toLong() }, values)
        }
        pthread_mutex_lock(mutex.ptr)
        val outs = try {
            module.call(FUNCTION, tensors)
        } finally {
            pthread_mutex_unlock(mutex.ptr)
        }
        check(outs.size == ports.outputs.size) { "decode_step returned ${outs.size} results, io.txt names ${ports.outputs.size}" }
        ports.outputs.withIndex().associate { (i, p) -> p.role to outs[i].toFloats() }
    }

    /** A fresh decoder (zero state, position 0) for one session; sessions share the module through [runner]. */
    fun decoder(): GreedyDecoder = GreedyDecoder(ports, controls, embedding, runner)

    fun close() {
        module.close()
        runtime.close()
        pthread_mutex_destroy(mutex.ptr)
        nativeHeap.free(mutex)
        errors.clear()
    }

    companion object {
        const val FUNCTION: String = "module.decode_step"
    }
}

/**
 * One generation (the ABI's streaming session): token ids pushed until flush, then each pull runs the step(s) for
 * one new token (the first pull also runs the prompt's steps). PARTIAL carries that token; once a stop token or the
 * compiled capacity ends the generation, FINAL carries every generated token (the stop token included), then the
 * stream is drained. A host that wants fewer tokens stops pulling and ends (or cancels) the session.
 */
internal class GenerationSession(val cartridge: Lfm2TorqCartridge) {
    enum class State { INPUT, FLUSHED, GENERATING, DRAINED, CANCELLED }

    var state: State = State.INPUT
    val errors = ErrorSlot()
    val cancelRequested = kotlin.concurrent.AtomicInt(0)
    private val prompt = ArrayList<Int>()
    private val generated = ArrayList<Int>()
    private val decoder = cartridge.decoder()
    private var last = -1
    private var finalSent = false

    fun push(tokens: IntArray) {
        if (state != State.INPUT) throw CartridgeException(Status.STATE, "push after flush")
        for (t in tokens) if (t !in 0 until cartridge.ports.vocabulary) throw CartridgeException(Status.INVALID_ARG, "token $t outside the vocabulary")
        if (prompt.size + tokens.size > cartridge.ports.capacity) {
            throw CartridgeException(Status.INVALID_ARG, "prompt of ${prompt.size + tokens.size} tokens exceeds the compiled capacity ${cartridge.ports.capacity}")
        }
        prompt += tokens.toList()
    }

    fun flush() {
        if (state != State.INPUT) throw CartridgeException(Status.STATE, "flush twice")
        if (prompt.isEmpty()) throw CartridgeException(Status.INVALID_ARG, "flush with an empty prompt")
        state = State.FLUSHED
    }

    /** The next result: (kind, tokens), or null when nothing is pending. */
    fun pull(): Pair<Int, IntArray>? {
        when (state) {
            State.INPUT, State.DRAINED -> return null
            State.CANCELLED -> throw CartridgeException(Status.CANCELLED, "session cancelled")
            State.FLUSHED -> {
                state = State.GENERATING
                var logits = FloatArray(0)
                for (t in prompt) {
                    checkCancel()
                    logits = decoder.step(t)
                }
                return emit(argmax(logits))
            }
            State.GENERATING -> {
                if (finished()) {
                    state = State.DRAINED
                    finalSent = true
                    return KIND_FINAL to generated.toIntArray()
                }
                checkCancel()
                return emit(argmax(decoder.step(last)))
            }
        }
    }

    private fun emit(token: Int): Pair<Int, IntArray> {
        generated += token
        last = token
        return KIND_PARTIAL to intArrayOf(token)
    }

    private fun finished(): Boolean = last in cartridge.stopTokens || decoder.position >= cartridge.ports.capacity

    private fun checkCancel() {
        if (cancelRequested.value != 0) {
            state = State.CANCELLED
            throw CartridgeException(Status.CANCELLED, "session cancelled")
        }
    }

    private fun argmax(v: FloatArray): Int {
        var best = 0
        for (i in 1 until v.size) if (v[i] > v[best]) best = i
        return best
    }

    companion object {
        const val KIND_PARTIAL: Int = 1
        const val KIND_FINAL: Int = 2
    }
}
