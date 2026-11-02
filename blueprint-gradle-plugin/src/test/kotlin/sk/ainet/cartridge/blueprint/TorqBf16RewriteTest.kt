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
        assertContains(r.mlir, "%v4 = stablehlo.divide %v3, %v4_v2c : tensor<1x4x64xbf16>")
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
        assertEquals(5, r.barriers)  // 3 matmul results, the head transpose, the K transpose
    }

    @Test
    fun `a row sum used by an op that cannot move to the full shape is refused`() {
        val broken = module.replace("%v4 = stablehlo.divide %v3, %v2 : tensor<1x4xf32>", "%v4 = stablehlo.divide %v3, %v2 : tensor<1x4xf32>\n    %v4b = stablehlo.maximum %v4, %v2 : tensor<1x4xf32>")
        val e = assertFailsWith<MaterializationException> { TorqBf16Rewrite.rewrite(broken) }
        assertContains(e.message!!, "%v4 is a row sum used by")
    }

    @Test
    fun `a module without weight matmuls is refused`() {
        val e = assertFailsWith<MaterializationException> { TorqBf16Rewrite.rewrite(module.replace("%v9 = stablehlo.dot_general %v7, %v8,", "%v9 = stablehlo.dot_general %v7, %v7,")) }
        assertContains(e.message!!, "no `dot_general %a, transpose(constant)` found")
    }
}
