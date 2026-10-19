package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException

/**
 * A `convert` step on exported StableHLO: express each 1-D convolution as im2col + one matrix multiplication.
 *
 * For `stablehlo.convolution` with layout `[b, f, 0]x[o, i, 0]->[b, f, 0]`, batch 1, stride s, no padding, no dilation
 * and one feature group, `y[o, t] = Σ_k Σ_i w[o, i, k] · x[i, s·t + k]`. The rewrite builds the K strided slices
 * `x[:, k : k + s·(T-1) + 1 : s]`, stacks them over channels into a `[K·I, T]` column matrix, and multiplies it by
 * the filter laid out as `[O, K·I]`. Same sum, same order of terms within each k, so numerics are those of a matmul
 * over the same products.
 *
 * Why: some IREE CPU code generators (seen with the Torq fork's llvm-cpu backend) generalize a strided 1-D
 * convolution into a loop nest they do not tile or vectorize; the same arithmetic as a matmul takes the tuned path.
 * On a Cortex-A55 the Moonshine frontend (two such convolutions) goes from 3.9 s to 0.17 s per window.
 *
 * Text-level on purpose, like [HostGatherRewrite]: reviewable next to the exported module.
 */
object Conv1dIm2colRewrite {
    private val CONV = Regex(
        """(?m)^( *)(%[\w]+) = stablehlo\.convolution\((%[\w]+), (%[\w]+)\) dim_numbers = \[b, f, 0\]x\[o, i, 0\]->\[b, f, 0\], """ +
            """window = \{stride = \[(\d+)\], pad = \[\[0, 0\]\], rhs_dilate = \[1\]\} """ +
            """\{batch_group_count = 1 : i64, feature_group_count = 1 : i64\} : """ +
            """\(tensor<1x(\d+)x(\d+)xf32>, tensor<(\d+)x(\d+)x(\d+)xf32>\) -> tensor<1x(\d+)x(\d+)xf32>$""",
    )

    data class Result(val mlir: String, val rewritten: Int)

    fun rewrite(mlir: String): Result {
        var count = 0
        val out = CONV.replace(mlir) { m ->
            val g = m.groupValues
            val (ind, res, x, w) = g.subList(1, 5)
            val (s, ci, wi, co) = g.subList(5, 9).map(String::toInt)
            val (ci2, k, co2, wo) = g.subList(9, 13).map(String::toInt)
            if (ci != ci2 || co != co2 || (wi - k) / s + 1 != wo) throw MaterializationException(
                "conv1d-im2col: inconsistent shapes in `${m.value.trim().take(160)}…`",
            )
            count++
            val p = "${res}_i2c"
            buildString {
                append("$ind${p}_x = stablehlo.reshape $x : (tensor<1x${ci}x${wi}xf32>) -> tensor<${ci}x${wi}xf32>\n")
                for (j in 0 until k) {
                    append("$ind${p}_s$j = \"stablehlo.slice\"(${p}_x) {start_indices = array<i64: 0, $j>, ")
                    append("limit_indices = array<i64: $ci, ${j + s * (wo - 1) + 1}>, strides = array<i64: 1, $s>} : ")
                    append("(tensor<${ci}x${wi}xf32>) -> tensor<${ci}x${wo}xf32>\n")
                }
                append("$ind${p}_col = stablehlo.concatenate ${(0 until k).joinToString { "${p}_s$it" }}, dim = 0 : ")
                append("(${List(k) { "tensor<${ci}x${wo}xf32>" }.joinToString()}) -> tensor<${k * ci}x${wo}xf32>\n")
                append("$ind${p}_wt = stablehlo.transpose $w, dims = [0, 2, 1] : (tensor<${co}x${ci}x${k}xf32>) -> tensor<${co}x${k}x${ci}xf32>\n")
                append("$ind${p}_w = stablehlo.reshape ${p}_wt : (tensor<${co}x${k}x${ci}xf32>) -> tensor<${co}x${k * ci}xf32>\n")
                append("$ind${p}_y = stablehlo.dot_general ${p}_w, ${p}_col, contracting_dims = [1] x [0] : ")
                append("(tensor<${co}x${k * ci}xf32>, tensor<${k * ci}x${wo}xf32>) -> tensor<${co}x${wo}xf32>\n")
                append("$ind$res = stablehlo.reshape ${p}_y : (tensor<${co}x${wo}xf32>) -> tensor<1x${co}x${wo}xf32>")
            }
        }
        if (count == 0) throw MaterializationException(
            "conv1d-im2col: no 1-D `stablehlo.convolution` ([b, f, 0]x[o, i, 0]->[b, f, 0], unpadded, batch 1) found — " +
                "the module has none, or the exporter's output format changed.",
        )
        return Result(out, count)
    }
}
