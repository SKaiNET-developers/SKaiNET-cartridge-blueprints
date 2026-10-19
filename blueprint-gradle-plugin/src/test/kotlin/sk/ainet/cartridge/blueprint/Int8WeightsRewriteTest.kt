package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.Int8WeightsRewrite
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class Int8WeightsRewriteTest {
    private fun matrix(rows: Int, cols: Int, value: (Int, Int) -> Float) =
        (0 until rows).joinToString(", ", "[", "]") { r -> (0 until cols).joinToString(", ", "[", "]") { c -> value(r, c).toString() } }

    // 2 x 4096: row 0 ramps to 1.27 (scale 0.01), row 1 is all zero (scale falls back to 1)
    private val module = """
        module {
          func.func @step(%arg0: tensor<1x4096xf32>) -> (tensor<1x2xf32>) {
            %v0 = stablehlo.constant dense<[0.5, 0.25]> : tensor<2xf32>
            %v1 = stablehlo.constant dense<${matrix(2, 4096) { r, c -> if (r == 0) (c % 255 - 127) * 0.01f else 0f }}> : tensor<2x4096xf32>
            %v2 = stablehlo.constant dense<${matrix(1, 8) { _, c -> c.toFloat() }}> : tensor<1x8xf32>
            %v3 = stablehlo.dot_general %arg0, %v1, contracting_dims = [1] x [1] : (tensor<1x4096xf32>, tensor<2x4096xf32>) -> tensor<1x2xf32>
            return %v3 : tensor<1x2xf32>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun `a large matrix becomes int8 with per-row scales, dequantized under its own name`() {
        val (mlir, r) = Int8WeightsRewrite.rewrite(module, minElements = 4096)
        assertEquals(1, r.rewritten)
        assertEquals(2L * 4096 * 4 - 2 * 4096 - 2 * 4, r.bytesSaved)
        assertFalse(mlir.contains("tensor<2x4096xf32>\n            %v2"), "the f32 literal is gone")
        // row 0: -127 … 127 steps of 1 → bytes 0x81 (−127), 0x82, …; row 1: zeros
        assertContains(mlir, "    %v1_q = stablehlo.constant dense<\"0x818283")
        assertContains(mlir, "> : tensor<2x4096xi8>\n")
        // scales 0.01f and 1.0f, little-endian
        val s0 = java.lang.Float.floatToRawIntBits(1.27f / 127f).let { b -> (0 until 4).joinToString("") { "%02X".format((b ushr (8 * it)) and 0xFF) } }
        assertContains(mlir, "%v1_s = stablehlo.constant dense<\"0x${s0}0000803F\"> : tensor<2xf32>")
        assertContains(mlir, "%v1_f = stablehlo.convert %v1_q : (tensor<2x4096xi8>) -> tensor<2x4096xf32>")
        assertContains(mlir, "%v1_b = stablehlo.broadcast_in_dim %v1_s, dims = [0] : (tensor<2xf32>) -> tensor<2x4096xf32>")
        assertContains(mlir, "    %v1 = stablehlo.multiply %v1_f, %v1_b : tensor<2x4096xf32>\n")
        assertContains(mlir, "%v0 = stablehlo.constant dense<[0.5, 0.25]> : tensor<2xf32>", message = "vectors stay f32")
        assertContains(mlir, "%v2 = stablehlo.constant dense<[[0.0, 1.0", message = "small matrices stay f32")
        assertContains(mlir, "%v3 = stablehlo.dot_general %arg0, %v1,", message = "uses keep the original name")
    }

    @Test
    fun `a value count that disagrees with the type is refused`() {
        val broken = module.replace(": tensor<2x4096xf32>\n", ": tensor<2x4097xf32>\n")
        val e = assertFailsWith<MaterializationException> { Int8WeightsRewrite.rewrite(broken, minElements = 4096) }
        assertContains(e.message!!, "%v1 has 8192 values, its type says 8194")
    }

    @Test
    fun `a module without a large matrix is refused`() {
        val e = assertFailsWith<MaterializationException> { Int8WeightsRewrite.rewrite(module) }
        assertContains(e.message!!, "no rank-2 f32 constant with at least 16384 elements")
    }
}
