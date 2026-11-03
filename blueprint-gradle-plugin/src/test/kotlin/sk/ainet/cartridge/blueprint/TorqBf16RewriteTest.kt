package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.TorqBf16Rewrite
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class TorqBf16RewriteTest {
    // w[r][c] = (64 r + c) mod 7 - 3: small integers, exact in bf16
    private val weight = (0 until 64).joinToString(", ", "[", "]") { r -> (0 until 64).joinToString(", ", "[", "]") { c -> "${(64 * r + c) % 7 - 3}.0" } }

    // The encoder's patterns in small: layer-norm mean, weight matmul, head transpose, attention Q.K^T, softmax max.
    private val module = """
        module {
          func.func @enc(%arg0: tensor<1x4x64xf32>) -> (tensor<1x4x64xf32>, tensor<1x4x4x4xf32>) {
            %v0 = stablehlo.constant dense<$weight> : tensor<64x64xf32>
            %v1 = stablehlo.constant dense<0.0> : tensor<f32>
            %v2 = stablehlo.constant dense<64.0> : tensor<1x4xf32>
            %v3 = stablehlo.reduce(%arg0 init: %v1) applies stablehlo.add across dimensions = [2] : (tensor<1x4x64xf32>, tensor<f32>) -> tensor<1x4xf32>
            %v4 = stablehlo.divide %v3, %v2 : tensor<1x4xf32>
            // Unsqueeze at dimension 2
            %v5 = stablehlo.reshape %v4 : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
            %v6 = stablehlo.broadcast_in_dim %v5, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x64xf32>
            %v7 = stablehlo.subtract %arg0, %v6 : tensor<1x4x64xf32>
            %v8 = stablehlo.transpose %v0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
            %v9 = stablehlo.dot_general %v7, %v8, contracting_dims = [2] x [0] : (tensor<1x4x64xf32>, tensor<64x64xf32>) -> tensor<1x4x64xf32>
            %v10 = stablehlo.reshape %v9 : (tensor<1x4x64xf32>) -> tensor<1x4x4x16xf32>
            %v11 = stablehlo.transpose %v10, dims = [0, 2, 1, 3] : (tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
            %v12 = stablehlo.dot_general %v11, %v11, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x4x16xf32>, tensor<1x4x4x16xf32>) -> tensor<1x4x4x4xf32>
            %v13 = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %v14 = stablehlo.reduce(%v12 init: %v13) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x4x4xf32>, tensor<f32>) -> tensor<1x4x4xf32>
            %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1, 2] : (tensor<1x4x4xf32>) -> tensor<1x4x4x4xf32>
            %v16 = stablehlo.subtract %v12, %v15 : tensor<1x4x4x4xf32>
            return %v9, %v16 : tensor<1x4x64xf32>, tensor<1x4x4x4xf32>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun `weight matmuls read a pre-transposed bf16 constant, the f32 one is dropped`() {
        val r = TorqBf16Rewrite.rewrite(module)
        assertEquals(1, r.weightMatmuls)
        // [in, out] = w^T, column by column of w: w[0][0] = -3 (0xC040), w[1][0] = -2 (0xC000), little-endian
        assertContains(r.mlir, "%v8_w16 = stablehlo.constant dense<\"0x40C000C0")
        assertContains(r.mlir, "> : tensor<64x64xbf16>\n")
        assertContains(r.mlir, "%v9_o = stablehlo.dot_general %v7, %v8_w16, contracting_dims = [2] x [0] : (tensor<1x4x64xbf16>, tensor<64x64xbf16>) -> tensor<1x4x64xbf16>")
        assertContains(r.mlir, "%v9 = stablehlo.optimization_barrier %v9_o : tensor<1x4x64xbf16>")
        assertFalse(r.mlir.contains("%v0 = "), "the f32 weight is gone")
        assertFalse(r.mlir.contains("%v8 = stablehlo.transpose"), "and its transpose")
    }

    @Test
    fun `a sum over the last dimension is a ones-matmul, the ops up to its broadcast move to the full shape`() {
        val r = TorqBf16Rewrite.rewrite(module)
        assertEquals(1, r.sumsAsMatmuls)
        assertContains(r.mlir, "%ones64 = stablehlo.constant dense<1.0> : tensor<64x64xbf16>")
        assertContains(r.mlir, "%v3_o = stablehlo.dot_general %arg0_16, %ones64, contracting_dims = [2] x [0] : (tensor<1x4x64xbf16>, tensor<64x64xbf16>) -> tensor<1x4x64xbf16>")
        assertContains(r.mlir, "%v4_v2c = stablehlo.constant dense<64.0> : tensor<1x4x64xbf16>")
        assertContains(r.mlir, "%v4_e = stablehlo.divide %v3, %v4_v2c : tensor<1x4x64xbf16>")
        assertContains(r.mlir, "%v7 = stablehlo.subtract %arg0_16, %v4 : tensor<1x4x64xbf16>")
        assertFalse(r.mlir.contains("%v5 = ") || r.mlir.contains("%v6 = "), "reshape and broadcast are gone")
        assertContains(r.mlir, "%v14 = stablehlo.reduce(%v12 init: %v13) applies stablehlo.maximum", message = "max stays a reduce")
    }

    @Test
    fun `bf16 inside, f32 at the boundary, barriers on transposes and attention without transposed-B`() {
        val r = TorqBf16Rewrite.rewrite(module)
        assertContains(r.mlir, "%arg0_16 = stablehlo.convert %arg0 : (tensor<1x4x64xf32>) -> tensor<1x4x64xbf16>")
        assertContains(r.mlir, "%v9_out = stablehlo.convert %v9 : (tensor<1x4x64xbf16>) -> tensor<1x4x64xf32>")
        assertContains(r.mlir, "return %v9_out, %v16_out : tensor<1x4x64xf32>, tensor<1x4x4x4xf32>")
        assertContains(r.mlir, "%v11_t = stablehlo.transpose %v10, dims = [0, 2, 1, 3]")
        assertContains(r.mlir, "%v11 = stablehlo.optimization_barrier %v11_t : tensor<1x4x4x16xbf16>")
        assertContains(r.mlir, "%v12_kt = stablehlo.transpose %v11, dims = [0, 1, 3, 2] : (tensor<1x4x4x16xbf16>) -> tensor<1x4x16x4xbf16>")
        assertContains(r.mlir, "%v12_o = stablehlo.dot_general %v11, %v12_ktb, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x4x16xbf16>, tensor<1x4x16x4xbf16>) -> tensor<1x4x4x4xbf16>")
        assertContains(r.mlir, "%v13 = stablehlo.constant dense<0xFF80> : tensor<bf16>")
        assertContains(r.mlir, "func.func @enc(%arg0: tensor<1x4x64xf32>) -> (tensor<1x4x64xf32>, tensor<1x4x4x4xf32>)")
        val f32 = r.mlir.lines().filter { "xf32>" in it && "convert" !in it && "func.func" !in it && "return" !in it }
        assertEquals(emptyList(), f32, "no f32 left inside")
        assertEquals(6, r.barriers)  // 3 matmul results, the head transpose, the K transpose, the mean (below)
    }

    @Test
    fun `an elementwise op feeding another one gets a barrier, so each runs alone`() {
        val r = TorqBf16Rewrite.rewrite(module)
        assertContains(r.mlir, "%v4 = stablehlo.optimization_barrier %v4_e : tensor<1x4x64xbf16>")
        assertContains(r.mlir, "%v7 = stablehlo.subtract %arg0_16, %v4 : tensor<1x4x64xbf16>")
        assertFalse(r.mlir.contains("%v7_e = "), "the subtract feeds only a matmul")
        assertFalse(r.mlir.contains("%v16_e = "), "the last subtract feeds only the output convert")
    }

    // w[o][i][k] = (64 o + 4 i + k) mod 7 - 3
    private val convWeight = (0 until 16).joinToString(", ", "[", "]") { o ->
        (0 until 16).joinToString(", ", "[", "]") { i -> (0 until 4).joinToString(", ", "[", "]") { k -> "${(64 * o + 4 * i + k) % 7 - 3}.0" } }
    }

    // The frontend's patterns in small: a strided conv as im2col (slices, concatenate, weight transpose + reshape),
    // its activation, and the transpose back to time-major.
    private val conv = """
        module {
          func.func @fe(%arg0: tensor<16x9xf32>) -> (tensor<3x16xf32>) {
            %v0 = stablehlo.constant dense<$convWeight> : tensor<16x16x4xf32>
${(0 until 4).joinToString("\n") { k -> "            %s$k = \"stablehlo.slice\"(%arg0) {start_indices = array<i64: 0, $k>, limit_indices = array<i64: 16, ${k + 5}>, strides = array<i64: 1, 2>} : (tensor<16x9xf32>) -> tensor<16x3xf32>" }}
            %col = stablehlo.concatenate %s0, %s1, %s2, %s3, dim = 0 : (tensor<16x3xf32>, tensor<16x3xf32>, tensor<16x3xf32>, tensor<16x3xf32>) -> tensor<64x3xf32>
            %wt = stablehlo.transpose %v0, dims = [0, 2, 1] : (tensor<16x16x4xf32>) -> tensor<16x4x16xf32>
            %w = stablehlo.reshape %wt : (tensor<16x4x16xf32>) -> tensor<16x64xf32>
            %y = stablehlo.dot_general %w, %col, contracting_dims = [1] x [0] : (tensor<16x64xf32>, tensor<64x3xf32>) -> tensor<16x3xf32>
            %m = stablehlo.multiply %y, %y : tensor<16x3xf32>
            %t = stablehlo.transpose %m, dims = [1, 0] : (tensor<16x3xf32>) -> tensor<3x16xf32>
            return %t : tensor<3x16xf32>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun `an im2col conv weight becomes its folded bf16 matrix, strided slices and transposes get barriers`() {
        val r = TorqBf16Rewrite.rewrite(conv)
        assertEquals(1, r.weightMatmuls)
        // [o, k*16 + i] = w[o][i][k]: row 0 starts w[0][0][0] = -3, w[0][1][0] = 1, w[0][2][0] = -2
        assertContains(r.mlir, "%w = stablehlo.constant dense<\"0x40C0803F00C0")
        assertContains(r.mlir, "> : tensor<16x64xbf16>")
        assertFalse(r.mlir.contains("%wt = ") || r.mlir.contains("16x16x4x"), "the 3-D weight and its transpose are gone")
        assertContains(r.mlir, "%s0 = stablehlo.optimization_barrier %s0_s : tensor<16x3xbf16>")
        assertContains(r.mlir, "%m = stablehlo.optimization_barrier %m_e : tensor<16x3xbf16>")
        assertContains(r.mlir, "%t = stablehlo.optimization_barrier %t_t : tensor<3x16xbf16>")
    }

    @Test
    fun `a row sum used by an op that cannot move to the full shape is refused`() {
        val broken = module.replace("%v4 = stablehlo.divide %v3, %v2 : tensor<1x4xf32>", "%v4 = stablehlo.divide %v3, %v2 : tensor<1x4xf32>\n    %v4b = stablehlo.maximum %v4, %v2 : tensor<1x4xf32>")
        val e = assertFailsWith<MaterializationException> { TorqBf16Rewrite.rewrite(broken) }
        assertContains(e.message!!, "%v4 is a row sum used by")
    }

    @Test
    fun `a second function gets its own argument conversions`() {
        val two = module.replace("\n}\n", "\n  func.func @kv(%arg0: tensor<1x4x64xf32>) -> (tensor<1x4x64xf32>) {\n    %x = stablehlo.dot_general %arg0, %v0x, contracting_dims = [2] x [0] : (tensor<1x4x64xf32>, tensor<64x64xf32>) -> tensor<1x4x64xf32>\n    return %x : tensor<1x4x64xf32>\n  }\n}\n")
            .replace("%v0x,", "%w,").replace("  func.func @kv(%arg0: tensor<1x4x64xf32>) -> (tensor<1x4x64xf32>) {\n", "  func.func @kv(%arg0: tensor<1x4x64xf32>) -> (tensor<1x4x64xf32>) {\n    %w = stablehlo.constant dense<$weight> : tensor<64x64xf32>\n")
        val r = TorqBf16Rewrite.rewrite(two)
        val kv = r.mlir.substringAfter("func.func @kv")
        assertContains(kv, "%arg0_16 = stablehlo.convert %arg0 : (tensor<1x4x64xf32>) -> tensor<1x4x64xbf16>")
        assertContains(kv, "%x_o = stablehlo.dot_general %arg0_16, %w, contracting_dims = [2] x [0] : (tensor<1x4x64xbf16>, tensor<64x64xbf16>) -> tensor<1x4x64xbf16>")
        assertContains(kv, "%x_out = stablehlo.convert %x : (tensor<1x4x64xbf16>) -> tensor<1x4x64xf32>")
        assertEquals(2, r.weightMatmuls)
    }

    @Test
    fun `a module without weight matmuls is refused`() {
        val e = assertFailsWith<MaterializationException> { TorqBf16Rewrite.rewrite(module.replace("%v9 = stablehlo.dot_general %v7, %v8,", "%v9 = stablehlo.dot_general %v7, %v7,")) }
        assertContains(e.message!!, "no `dot_general` with a constant operand found")
    }
}
