package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.Conv1dIm2colRewrite
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class Conv1dIm2colRewriteTest {
    private val module = """
        module {
          func.func @frontend(%arg0: tensor<1x320x276xf32>) -> (tensor<1x640x136xf32>) {
            %v3 = stablehlo.constant dense<0.5> : tensor<640x320x5xf32>
            %v46_conv = stablehlo.convolution(%arg0, %v3) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [2], pad = [[0, 0]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x320x276xf32>, tensor<640x320x5xf32>) -> tensor<1x640x136xf32>
            %v46 = stablehlo.add %v46_conv, %v46_conv : tensor<1x640x136xf32>
            return %v46 : tensor<1x640x136xf32>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun `a strided 1-D convolution becomes K strided slices and one matmul`() {
        val r = Conv1dIm2colRewrite.rewrite(module)
        assertEquals(1, r.rewritten)
        assertFalse(r.mlir.contains("stablehlo.convolution"))
        // slice k covers x[:, k : k + 2*(136-1) + 1 : 2]
        assertContains(r.mlir, "start_indices = array<i64: 0, 0>, limit_indices = array<i64: 320, 271>, strides = array<i64: 1, 2>")
        assertContains(r.mlir, "start_indices = array<i64: 0, 4>, limit_indices = array<i64: 320, 275>, strides = array<i64: 1, 2>")
        assertContains(r.mlir, "-> tensor<1600x136xf32>")
        assertContains(r.mlir, "%v46_conv_i2c_wt = stablehlo.transpose %v3, dims = [0, 2, 1] : (tensor<640x320x5xf32>) -> tensor<640x5x320xf32>")
        assertContains(r.mlir, "(tensor<640x1600xf32>, tensor<1600x136xf32>) -> tensor<640x136xf32>")
        assertContains(r.mlir, "    %v46_conv = stablehlo.reshape %v46_conv_i2c_y : (tensor<640x136xf32>) -> tensor<1x640x136xf32>\n")
        assertContains(r.mlir, "%v46 = stablehlo.add %v46_conv, %v46_conv", message = "downstream uses keep the original name")
    }

    @Test
    fun `a module without a 1-D convolution is refused`() {
        val e = assertFailsWith<MaterializationException> { Conv1dIm2colRewrite.rewrite(module.replace("pad = [[0, 0]]", "pad = [[2, 2]]")) }
        assertContains(e.message!!, "no 1-D `stablehlo.convolution`")
    }
}
