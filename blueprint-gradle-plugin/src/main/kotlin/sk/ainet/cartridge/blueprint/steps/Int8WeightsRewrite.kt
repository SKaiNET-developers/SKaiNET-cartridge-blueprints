package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException
import java.io.BufferedReader
import java.io.StringReader
import java.io.StringWriter
import java.io.Writer

/**
 * A `convert` step on exported StableHLO: weight-only int8 for the large matrices.
 *
 * Every rank-2 `f32` constant with at least [DEFAULT_MIN_ELEMENTS] elements becomes an `i8` constant with one
 * symmetric scale per row (`scale = max|row| / 127`, round half to even, clamp to ±127), dequantized in the graph:
 * `convert` to f32, `broadcast_in_dim` of the scales over the rows, `multiply`. The result keeps the original SSA
 * name, so downstream uses are unchanged. Only matrices are touched: higher-rank constants (masks, position bands)
 * and vectors stay f32.
 *
 * Why: the CPU decoder of a streaming ASR model is bound by weight bandwidth (all matrices streamed once per token);
 * four times fewer weight bytes is the cheapest lever. Compile the rewritten module with const-expr hoisting and
 * const-eval off, otherwise the compiler folds the dequantization back into f32 constants.
 *
 * Streams line by line (exported decoders are several hundred MB of text, single constants >100 MB), and is
 * text-level on purpose, like [HostGatherRewrite]: reviewable next to the exported module.
 */
object Int8WeightsRewrite {
    const val DEFAULT_MIN_ELEMENTS = 16384

    /** Flags `iree-compile` needs so the dequantization stays in the graph (and the weights stay int8). */
    val COMPILE_FLAGS = listOf("--iree-opt-const-expr-hoisting=false", "--iree-opt-const-eval=false")

    private val HEAD = Regex("""^( *)(%[\w]+) = stablehlo\.constant dense<\[\[""")
    private val TAIL = Regex("""\]\]> : tensor<(\d+)x(\d+)xf32>$""")

    data class Result(val rewritten: Int, val bytesSaved: Long)

    fun rewrite(mlir: String, minElements: Int = DEFAULT_MIN_ELEMENTS): Pair<String, Result> {
        val out = StringWriter()
        val r = rewrite(BufferedReader(StringReader(mlir)), out, minElements)
        return out.toString() to r
    }

    fun rewrite(input: BufferedReader, output: Writer, minElements: Int = DEFAULT_MIN_ELEMENTS): Result {
        var count = 0
        var saved = 0L
        while (true) {
            val line = input.readLine() ?: break
            val q = if (line.length > 4096) matrix(line, minElements) else null
            if (q == null) {
                output.write(line); output.write("\n"); continue
            }
            output.write(q.mlir)
            count++
            saved += q.saved
        }
        if (count == 0) throw MaterializationException(
            "int8-weights: no rank-2 f32 constant with at least $minElements elements found — the module has none, " +
                "or the exporter's output format changed.",
        )
        return Result(count, saved)
    }

    private class Quantized(val mlir: String, val saved: Long)

    private fun matrix(line: String, minElements: Int): Quantized? {
        val head = HEAD.find(line.substring(0, minOf(line.length, 256))) ?: return null
        val tailFrom = maxOf(0, line.length - 128)
        val tail = TAIL.find(line.substring(tailFrom)) ?: return null
        val rows = tail.groupValues[1].toInt()
        val cols = tail.groupValues[2].toInt()
        val n = rows.toLong() * cols
        if (n < minElements) return null
        val (ind, name) = head.destructured
        val w = parseFloats(line, head.range.last + 1, tailFrom + tail.range.first, n.toInt(), name)

        val scales = FloatArray(rows)
        val q = ByteArray(n.toInt())
        for (r in 0 until rows) {
            var max = 0f
            for (c in 0 until cols) max = maxOf(max, Math.abs(w[r * cols + c]))
            val s = if (max == 0f) 1f else max / 127f
            scales[r] = s
            for (c in 0 until cols) {
                val v = Math.rint((w[r * cols + c] / s).toDouble()).coerceIn(-127.0, 127.0)
                q[r * cols + c] = v.toInt().toByte()
            }
        }
        val m = "${rows}x$cols"
        val text = buildString(q.size * 2 + rows * 8 + 512) {
            append("$ind${name}_q = stablehlo.constant dense<\"0x"); hex(q); append("\"> : tensor<${m}xi8>\n")
            append("$ind${name}_s = stablehlo.constant dense<\"0x"); hex(scales); append("\"> : tensor<${rows}xf32>\n")
            append("$ind${name}_f = stablehlo.convert ${name}_q : (tensor<${m}xi8>) -> tensor<${m}xf32>\n")
            append("$ind${name}_b = stablehlo.broadcast_in_dim ${name}_s, dims = [0] : (tensor<${rows}xf32>) -> tensor<${m}xf32>\n")
            append("$ind$name = stablehlo.multiply ${name}_f, ${name}_b : tensor<${m}xf32>\n")
        }
        return Quantized(text, n * 4 - n - rows * 4L)
    }

    /** The decimal values of `[[a, b, …], […]]` in `line[from, to)`, exactly [expected] of them. */
    private fun parseFloats(line: String, from: Int, to: Int, expected: Int, name: String): FloatArray {
        val out = FloatArray(expected)
        var k = 0
        var i = from
        while (i < to) {
            val ch = line[i]
            if (ch == '[' || ch == ']' || ch == ',' || ch == ' ') { i++; continue }
            var j = i
            while (j < to && line[j] != ',' && line[j] != ']') j++
            if (k == expected) throw MaterializationException("int8-weights: $name has more than $expected values")
            out[k++] = line.substring(i, j).trim().toFloat()
            i = j
        }
        if (k != expected) throw MaterializationException("int8-weights: $name has $k values, its type says $expected")
        return out
    }

    private val HEX = "0123456789ABCDEF".toCharArray()

    private fun StringBuilder.hex(bytes: ByteArray) {
        for (b in bytes) { val v = b.toInt() and 0xFF; append(HEX[v ushr 4]); append(HEX[v and 0xF]) }
    }

    /** Little-endian IEEE-754, the layout of a hex `dense` literal. */
    private fun StringBuilder.hex(values: FloatArray) {
        for (f in values) {
            val bits = java.lang.Float.floatToRawIntBits(f)
            for (shift in 0 until 32 step 8) { val v = (bits ushr shift) and 0xFF; append(HEX[v ushr 4]); append(HEX[v and 0xF]) }
        }
    }
}
