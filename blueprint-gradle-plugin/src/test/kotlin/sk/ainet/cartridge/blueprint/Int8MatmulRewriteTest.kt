package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.Int8MatmulRewrite
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class Int8MatmulRewriteTest {
    // What int8-weights leaves behind for a [6, 4] linear layer (out, in) read transposed, as the exporter emits it.
    private val module = """
        module {
          func.func @step(%arg0: tensor<1x1x4xf32>) -> (tensor<1x1x6xf32>) {
            %v0_q = stablehlo.constant dense<1> : tensor<6x4xi8>
            %v0_s = stablehlo.constant dense<0.5> : tensor<6xf32>
            %v0_f = stablehlo.convert %v0_q : (tensor<6x4xi8>) -> tensor<6x4xf32>
            %v0_b = stablehlo.broadcast_in_dim %v0_s, dims = [0] : (tensor<6xf32>) -> tensor<6x4xf32>
            %v0 = stablehlo.multiply %v0_f, %v0_b : tensor<6x4xf32>
            %v1 = stablehlo.transpose %v0, dims = [1, 0] : (tensor<6x4xf32>) -> tensor<4x6xf32>
            %v2 = stablehlo.dot_general %arg0, %v1, contracting_dims = [2] x [0] : (tensor<1x1x4xf32>, tensor<4x6xf32>) -> tensor<1x1x6xf32>
            return %v2 : tensor<1x1x6xf32>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun `a weight matmul runs on int8 operands, both scales applied to its i32 result`() {
        val (mlir, r) = Int8MatmulRewrite.rewrite(module)
        assertEquals(1, r.matmuls)
        assertContains(mlir, "%v2_max = stablehlo.reduce(%v2_abs init: %v2_z) applies stablehlo.maximum across dimensions = [2] : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1x1xf32>")
        assertContains(mlir, "%v2_xq = stablehlo.convert %v2_xr : (tensor<1x1x4xf32>) -> tensor<1x1x4xi8>")
        assertContains(mlir, "%v2_d = stablehlo.dot_general %v2_xq, %v0_q, contracting_dims = [2] x [1] : (tensor<1x1x4xi8>, tensor<6x4xi8>) -> tensor<1x1x6xi32>")
        assertContains(mlir, "%v2_ws = stablehlo.broadcast_in_dim %v0_s, dims = [2] : (tensor<6xf32>) -> tensor<1x1x6xf32>")
        assertContains(mlir, "%v2_xso = stablehlo.broadcast_in_dim %v2_sx, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x6xf32>")
        assertContains(mlir, "%v2 = stablehlo.multiply %v2_y, %v2_xso : tensor<1x1x6xf32>")
        assertContains(mlir, "return %v2 : tensor<1x1x6xf32>")
    }

    @Test
    fun `an int8 matrix read in another form is refused`() {
        val e = assertFailsWith<MaterializationException> {
            Int8MatmulRewrite.rewrite(module.replace(
                "%v2 = stablehlo.dot_general %arg0, %v1, contracting_dims = [2] x [0] : (tensor<1x1x4xf32>, tensor<4x6xf32>)",
                "%v2 = stablehlo.dot_general %arg0, %v0, contracting_dims = [2] x [1] : (tensor<1x1x4xf32>, tensor<6x4xf32>)",
            ))
        }
        assertContains(e.message!!, "in a form this step does not rewrite")
    }

    @Test
    fun `a module without int8 weight matmuls is refused`() {
        val e = assertFailsWith<MaterializationException> { Int8MatmulRewrite.rewrite("module {\n}\n") }
        assertContains(e.message!!, "run int8-weights on the graph first")
    }
}
