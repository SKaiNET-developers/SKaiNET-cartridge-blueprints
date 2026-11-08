package sk.ainet.cartridge.generator.lfm25.torq

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/** One argument or result of the compiled decode step: its role (`embedded`, `mask`, `key_2`, `logits`, …) and shape. */
public data class StepPort(val role: String, val shape: List<Int>) {
    val size: Int get() = shape.fold(1) { a, b -> a * b }
}

/**
 * The decode step's argument and result order, as the exporter writes it to `io.txt`
 * (`in <i> <role> [d0, d1, …]`, then `out <i> <role> [d0, …]`). Every state port (anything but the step
 * controls and `logits`) is both an argument and a result of the same shape.
 */
public class DecodeStepPorts(public val inputs: List<StepPort>, public val outputs: List<StepPort>) {
    /** The state carried from one step to the next, in argument order. */
    public val state: List<StepPort> = inputs.filter { it.role !in CONTROLS }

    init {
        require(inputs.map { it.role }.containsAll(CONTROLS)) { "decode step: missing control inputs ${CONTROLS - inputs.map { it.role }.toSet()}" }
        require(outputs.any { it.role == "logits" }) { "decode step: no logits result" }
        val results = outputs.associateBy { it.role }
        for (s in state) require(results[s.role]?.shape == s.shape) { "decode step: state ${s.role} ${s.shape} is not a result of the same shape" }
    }

    public val width: Int get() = inputs.single { it.role == "embedded" }.shape.last()
    public val headDim: Int get() = inputs.single { it.role == "cos" }.shape.last()
    public val capacity: Int get() = inputs.single { it.role == "write_mask" }.shape[1]
    public val vocabulary: Int get() = outputs.single { it.role == "logits" }.shape.last()

    public companion object {
        public val CONTROLS: Set<String> = setOf("embedded", "cos", "sin", "mask", "write_mask")

        public fun parse(ioTable: String): DecodeStepPorts {
            val line = Regex("""^(in|out) (\d+) (\S+) \[([0-9, ]*)]$""")
            val ins = ArrayList<StepPort>()
            val outs = ArrayList<StepPort>()
            for (raw in ioTable.lines().map(String::trim).filter(String::isNotEmpty)) {
                val m = requireNotNull(line.matchEntire(raw)) { "io table: unreadable line '$raw'" }
                val (dir, index, role, dims) = m.destructured
                val list = if (dir == "in") ins else outs
                require(index.toInt() == list.size) { "io table: $dir $index out of order" }
                list += StepPort(role, dims.split(',').map(String::trim).filter(String::isNotEmpty).map(String::toInt))
            }
            return DecodeStepPorts(ins, outs)
        }
    }
}

/**
 * The host-side controls of one step at [position][forPosition]: split-half RoPE (`cos`/`sin`, the same angle in both
 * halves, base [ropeTheta]), the additive attention mask over the cache's [capacity] slots plus the current token
 * (0 for the filled prefix and the current slot, [MASKED] elsewhere) and the one-hot write mask that stores this
 * token's K/V at slot [position]. The cache is a bounded prefix: positions run 0 until [capacity], no wrap-around.
 */
public class StepControls(public val headDim: Int, public val ropeTheta: Double, public val capacity: Int) {
    init {
        require(headDim > 0 && headDim % 2 == 0 && capacity > 0 && ropeTheta > 0)
    }

    public class Controls(public val cos: FloatArray, public val sin: FloatArray, public val mask: FloatArray, public val writeMask: FloatArray)

    public fun forPosition(position: Int): Controls {
        require(position in 0 until capacity) { "position $position outside the compiled capacity $capacity" }
        val c = FloatArray(headDim)
        val s = FloatArray(headDim)
        for (i in 0 until headDim / 2) {
            val angle = position * ropeTheta.pow(-2.0 * i / headDim)
            c[i] = cos(angle).toFloat(); c[i + headDim / 2] = c[i]
            s[i] = sin(angle).toFloat(); s[i + headDim / 2] = s[i]
        }
        val mask = FloatArray(capacity + 1) { if (it < position || it == capacity) 0f else MASKED }
        val write = FloatArray(capacity) { if (it == position) 1f else 0f }
        return Controls(c, s, mask, write)
    }

    public companion object {
        /** Large, finite and exact in bf16, so a masked logit underflows to 0 after the softmax without an infinity. */
        public const val MASKED: Float = -1e30f
    }
}

/** The host's embedding table: [vocabulary] rows of [width] bf16 values, little-endian (`embed.bf16.bin`). */
public class EmbeddingRows(public val vocabulary: Int, public val width: Int, private val bf16: ByteArray) {
    init {
        require(bf16.size.toLong() == vocabulary.toLong() * width * 2) { "embedding rows: ${bf16.size} bytes for $vocabulary x $width bf16" }
    }

    public fun row(token: Int): FloatArray {
        require(token in 0 until vocabulary) { "token $token outside the vocabulary" }
        val base = token * width * 2
        return FloatArray(width) { i ->
            val lo = bf16[base + 2 * i].toInt() and 0xFF
            val hi = bf16[base + 2 * i + 1].toInt() and 0xFF
            Float.fromBits(((hi shl 8) or lo) shl 16)
        }
    }
}

/**
 * One call of the compiled decode step on its device: every argument by role, values as floats (the runner
 * converts to the module's bf16), and every result by role.
 */
public fun interface DecodeStepRunner {
    public fun run(inputs: Map<String, FloatArray>): Map<String, FloatArray>
}

/** Why a generation ended. */
public enum class FinishReason { STOP, LENGTH, CAPACITY }

public class Generation(public val tokens: IntArray, public val finishReason: FinishReason, public val steps: Int)

/**
 * Greedy generation over the decode step: the prompt is fed one token per step, then each step's arg-max is fed
 * back until a [stopTokens] token, [maxNewTokens] or the compiled capacity. One instance holds one session's state;
 * [reset] starts the next.
 */
/** A [GreedyDecoder]'s conv histories, K/V and position, taken by [GreedyDecoder.snapshot]. */
public class DecoderSnapshot internal constructor(internal val state: Map<String, FloatArray>, public val position: Int)

public class GreedyDecoder(
    private val ports: DecodeStepPorts,
    private val controls: StepControls,
    private val embedding: EmbeddingRows,
    private val runner: DecodeStepRunner,
) {
    init {
        require(controls.capacity == ports.capacity && controls.headDim == ports.headDim) { "controls do not match the step's ports" }
        require(embedding.width == ports.width && embedding.vocabulary == ports.vocabulary) { "embedding rows do not match the step's ports" }
    }

    private var state: Map<String, FloatArray> = initialState()
    public var position: Int = 0
        private set

    private fun initialState(): Map<String, FloatArray> = ports.state.associate { it.role to FloatArray(it.size) }

    public fun reset() {
        state = initialState(); position = 0
    }

    /** The state after the steps so far, e.g. a shared prompt prefix, for [restore] before each prompt that continues it. */
    public fun snapshot(): DecoderSnapshot = DecoderSnapshot(state.mapValues { it.value.copyOf() }, position)

    public fun restore(snapshot: DecoderSnapshot) {
        state = snapshot.state.mapValues { it.value.copyOf() }; position = snapshot.position
    }

    /** One step for [token] at the current position; returns the logits and advances. */
    public fun step(token: Int): FloatArray {
        val c = controls.forPosition(position)
        val inputs = HashMap<String, FloatArray>(state)
        inputs["embedded"] = embedding.row(token)
        inputs["cos"] = c.cos; inputs["sin"] = c.sin; inputs["mask"] = c.mask; inputs["write_mask"] = c.writeMask
        val out = runner.run(inputs)
        state = ports.state.associate { p -> p.role to requireNotNull(out[p.role]) { "runner returned no ${p.role}" } }
        position++
        return requireNotNull(out["logits"]) { "runner returned no logits" }
    }

    public fun generate(prompt: IntArray, maxNewTokens: Int, stopTokens: Set<Int>, onToken: (Int) -> Unit = {}): Generation {
        require(prompt.isNotEmpty()) { "empty prompt" }
        val start = position
        var logits = FloatArray(0)
        for (t in prompt) {
            if (position >= controls.capacity) return Generation(IntArray(0), FinishReason.CAPACITY, position - start)
            logits = step(t)
        }
        val generated = ArrayList<Int>()
        while (generated.size < maxNewTokens) {
            val next = logits.indices.maxBy { logits[it] }
            generated += next
            onToken(next)
            if (next in stopTokens) return Generation(generated.toIntArray(), FinishReason.STOP, position - start)
            if (generated.size == maxNewTokens) break
            if (position >= controls.capacity) return Generation(generated.toIntArray(), FinishReason.CAPACITY, position - start)
            logits = step(next)
        }
        return Generation(generated.toIntArray(), FinishReason.LENGTH, position - start)
    }
}
