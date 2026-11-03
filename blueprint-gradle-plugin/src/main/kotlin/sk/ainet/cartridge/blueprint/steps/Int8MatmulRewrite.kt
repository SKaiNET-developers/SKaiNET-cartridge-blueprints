package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException
import java.io.BufferedReader
import java.io.StringReader
import java.io.StringWriter
import java.io.Writer

/**
 * A `convert` step after [Int8WeightsRewrite]: its weight matmuls become int8 × int8 matmuls (W8A8).
 *
 * A weight matmul there reads `transpose(multiply(convert(W_q), broadcast(W_s)), [1, 0])`: the `[out, in]` int8 matrix
 * dequantized per output row. Here its activation is quantized per row instead (`s_x = max|row| / 127`, round half
 * to even), the matmul runs on the two int8 operands with an i32 result, and both scales are applied to that result:
 * `y = (x_q · W_qᵀ) · s_x · W_s`. The dequantized weights are then unused.
 *
 * Why: the IREE CPU backend dequantizes int8 weights inside the matmul on every call, which on a Cortex-A55 runs at
 * ~1 GB/s of weights; with int8 operands, data tiling and the `mmt4d` ukernel (dot-product instructions) the same
 * matmul is ~3× faster, and the weights are packed once at compile time. Moonshine v2 tiny decoder step on an SL2610:
 * 29 → 17 ms. Activation rounding changes the numerics slightly, so check the transcripts per graph. Compile the
 * output with [COMPILE_FLAGS], not [Int8WeightsRewrite.COMPILE_FLAGS].
 *
 * Every int8 weight matmul must have this form (contracting the activation's last dimension), otherwise the step
 * fails: a leftover dequantizing matmul under data tiling is several times slower than before. Streams line by line.
 */
object Int8MatmulRewrite {
    /**
     * Data tiling on the GlobalOptimization path (the DispatchCreation path crashes the Torq-fork 3.10 compiler on any
     * f32 batch matmul) and the ukernels; const-eval stays on, so the int8 weights are packed at compile time.
     */
    val COMPILE_FLAGS = listOf("--iree-global-opt-data-tiling", "--iree-llvmcpu-enable-ukernels=all")

    private const val LONG = 4096
    private val DEQUANT = Regex("""^ *(%[\w]+)_f = stablehlo\.convert \1_q : \(tensor<(\d+)x(\d+)xi8>\) -> tensor<\d+x\d+xf32>$""")
    private val TRANSPOSE = Regex("""^ *(%[\w]+) = stablehlo\.transpose (%[\w]+), dims = \[1, 0\] : \(tensor<\d+x\d+xf32>\) -> tensor<\d+x\d+xf32>$""")
    private val DOT = Regex("""^( *)(%[\w]+) = stablehlo\.dot_general (%[\w]+), (%[\w]+), contracting_dims = \[(\d)\] x \[(\d)\] : \(tensor<([0-9x]+)xf32>, tensor<\d+x\d+xf32>\) -> tensor<([0-9x]+)xf32>$""")

    data class Result(val matmuls: Int)

    fun rewrite(mlir: String): Pair<String, Result> {
        val out = StringWriter()
        val r = rewrite(BufferedReader(StringReader(mlir)), out)
        return out.toString() to r
    }

    fun rewrite(input: BufferedReader, output: Writer): Result {
        val matrices = HashMap<String, Pair<Int, Int>>()  // dequantized weight -> rows (out), cols (in)
        val transposed = HashMap<String, String>()        // transpose result -> dequantized weight
        var n = 0
        while (true) {
            val line = input.readLine() ?: break
            if (line.length > LONG) { output.line(line); continue }
            DEQUANT.matchEntire(line)?.let { matrices[it.groupValues[1]] = it.groupValues[2].toInt() to it.groupValues[3].toInt() }
            TRANSPOSE.matchEntire(line)?.let { if (it.groupValues[2] in matrices) transposed[it.groupValues[1]] = it.groupValues[2] }
            val dot = DOT.matchEntire(line)
            if (dot == null) { output.line(line); continue }
            val (ind, r, a, b, ca, cb, aShape, outShape) = dot.destructured
            if (b in matrices || a in matrices || a in transposed) throw MaterializationException(
                "int8-matmul: `${line.trim().take(160)}` reads an int8 matrix in a form this step does not rewrite",
            )
            val w = transposed[b]
            if (w == null) { output.line(line); continue }
            val rank = aShape.split('x').size
            if (ca.toInt() != rank - 1 || cb != "0") throw MaterializationException(
                "int8-matmul: `${line.trim().take(160)}` does not contract the activation's last dimension",
            )
            val (rows, cols) = matrices.getValue(w)
            val lead = aShape.split('x').dropLast(1)
            val rowT = if (lead.isEmpty()) "tensor<f32>" else "tensor<${lead.joinToString("x")}xf32>"
            val leadDims = lead.indices.joinToString(", ")
            val aT = "tensor<${aShape}xf32>"
            val oT = "tensor<${outShape}xf32>"
            val p = "${r}_"
            output.line("$ind${p}z = stablehlo.constant dense<0.0> : tensor<f32>")
            output.line("$ind${p}abs = stablehlo.abs $a : $aT")
            output.line("$ind${p}max = stablehlo.reduce(${p}abs init: ${p}z) applies stablehlo.maximum across dimensions = [${rank - 1}] : ($aT, tensor<f32>) -> $rowT")
            output.line("$ind${p}tiny = stablehlo.constant dense<1.0e-30> : $rowT")
            output.line("$ind${p}c127 = stablehlo.constant dense<127.0> : $rowT")
            output.line("$ind${p}m = stablehlo.maximum ${p}max, ${p}tiny : $rowT")
            output.line("$ind${p}sx = stablehlo.divide ${p}m, ${p}c127 : $rowT")
            output.line("$ind${p}sxa = stablehlo.broadcast_in_dim ${p}sx, dims = [$leadDims] : ($rowT) -> $aT")
            output.line("$ind${p}xs = stablehlo.divide $a, ${p}sxa : $aT")
            output.line("$ind${p}xr = stablehlo.round_nearest_even ${p}xs : $aT")
            output.line("$ind${p}xq = stablehlo.convert ${p}xr : ($aT) -> tensor<${aShape}xi8>")
            output.line("$ind${p}d = stablehlo.dot_general ${p}xq, ${w}_q, contracting_dims = [$ca] x [1] : (tensor<${aShape}xi8>, tensor<${rows}x${cols}xi8>) -> tensor<${outShape}xi32>")
            output.line("$ind${p}df = stablehlo.convert ${p}d : (tensor<${outShape}xi32>) -> $oT")
            output.line("$ind${p}ws = stablehlo.broadcast_in_dim ${w}_s, dims = [${rank - 1}] : (tensor<${rows}xf32>) -> $oT")
            output.line("$ind${p}xso = stablehlo.broadcast_in_dim ${p}sx, dims = [$leadDims] : ($rowT) -> $oT")
            output.line("$ind${p}y = stablehlo.multiply ${p}df, ${p}ws : $oT")
            output.line("$ind$r = stablehlo.multiply ${p}y, ${p}xso : $oT")
            n++
        }
        if (n == 0) throw MaterializationException("int8-matmul: no int8 weight matmul found — run int8-weights on the graph first")
        return Result(n)
    }

    private fun Writer.line(s: String) { write(s); write("\n") }
}
